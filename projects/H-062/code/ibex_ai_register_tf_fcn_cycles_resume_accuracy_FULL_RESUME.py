#!/usr/bin/env python3
"""
IBEX registration with TILE-SAFE TensorFlow FCN training + multi-cycle refinement + resume.

Key design choices:
- Segmentation is 2D max-projection nuclei labels (Y,X). No 3D masks required.
- TensorFlow FCN is trained on RANDOM TILES (no full-image forward pass -> avoids OOM).
- Per-nucleus embeddings are computed by TILE INFERENCE + MASK POOLING (no per-cell patch extraction).
- Matching is done with bounded radius around predicted positions; cycles tighten radius and refine local field.
- Local XY field is fit on a regular grid by binning anchor residuals + Gaussian smoothing (O(N), no RBF NxN memory).
- Optional dz correction by sparse tilewise z-profile correlation.

Outputs in out_dir:
  fixed_padded.npy / moving_padded.npy
  fixed_norm.npy / moving_norm.npy
  fixed_mip_int.npy / moving_mip_int.npy
  fixed_mip_str.npy / moving_mip_str.npy
  fixed_labels_2d.tif / moving_labels_2d.tif
  fft_shift.json
  affine_2d.tfm
  tf_fcn_embedder.keras
  cache_fixed_embeddings.npz
  cache_moving_embeddings_global.npz
  cycle_00/ ... cycle_K/
  transform_fields_total.npz
  moving_aligned_to_fixed.tif
  confidence_map_fixed_centroids.tif

Run example:
  python ibex_ai_register_tf_fcn_cycles_resume_accuracy_FULL_RESUME.py --moving first_round.tif --fixed second_round.tif --out out_dir --gpu --train --cycles 8

"""

import os, json, math, argparse, gc, time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import tifffile as tiff
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator

import SimpleITK as sitk

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from scipy.ndimage import gaussian_filter

import torch
from tqdm import tqdm

class MatchHW(keras.layers.Layer):
    """Resize ctx to loc's dynamic H,W using bilinear (serializable; no Lambda)."""
    def __init__(self, method="bilinear", **kwargs):
        super().__init__(**kwargs)
        self.method = method

    def call(self, inputs):
        loc, ctx = inputs
        loc_hw = tf.shape(loc)[1:3]
        ctx_rs = tf.image.resize(ctx, size=loc_hw, method=self.method)
        return loc, ctx_rs

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"method": self.method})
        return cfg

# ------------------------------------------------------------
# Basic utilities
# ------------------------------------------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def exists(p: str) -> bool:
    return p is not None and os.path.exists(p)

def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str):
    print(f"{msg}", flush=True)

def read_stack(path: str) -> np.ndarray:
    arr = tiff.imread(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D stack (Z,Y,X). Got {arr.shape} for {path}")
    return arr.astype(np.float32)

def write_stack(path: str, arr: np.ndarray):
    tiff.imwrite(path, arr.astype(np.float32), compression="zlib")

def write_labels_2d(path: str, lab: np.ndarray):
    tiff.imwrite(path, lab.astype(np.int32), compression="zlib")

def write_u8(path: str, arr: np.ndarray):
    tiff.imwrite(path, arr.astype(np.uint8), compression="zlib")

def normalize01(img: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    lo, hi = np.percentile(img, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    out = (img - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def max_proj(vol: np.ndarray) -> np.ndarray:
    return np.max(vol, axis=0).astype(np.float32)

def pad_to_match_shape(F: np.ndarray, M: np.ndarray):
    Zf, Hf, Wf = F.shape
    Zm, Hm, Wm = M.shape
    Zt = max(Zf, Zm)
    Ht = max(Hf, Hm)
    Wt = max(Wf, Wm)

    def pad(V):
        Z, H, W = V.shape
        return np.pad(V, ((0, Zt-Z), (0, Ht-H), (0, Wt-W)), mode="constant", constant_values=0)

    return pad(F).astype(np.float32), pad(M).astype(np.float32), (Zf,Hf,Wf), (Zm,Hm,Wm), (Zt,Ht,Wt)

def crop_to_fixed(Vp: np.ndarray, fixed_shape: Tuple[int,int,int]) -> np.ndarray:
    Zf, Hf, Wf = fixed_shape
    return Vp[:Zf, :Hf, :Wf]

def setup_tf_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    log(f"[GPU] TensorFlow sees {len(gpus)} GPU(s): {gpus}")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

def setup_torch_gpu():
    ok = torch.cuda.is_available()
    log(f"[GPU] PyTorch CUDA available: {ok}")
    if ok:
        log(f"[GPU] PyTorch device: {torch.cuda.get_device_name(0)}")


# ------------------------------------------------------------
# Manual tiling helpers
# ------------------------------------------------------------

def iter_tiles_nonoverlap(H: int, W: int, tile: int):
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            y1 = min(H, y0 + tile)
            x1 = min(W, x0 + tile)
            yield y0, y1, x0, x1

def iter_tiles_overlap(H: int, W: int, tile: int, overlap: int):
    stride = tile - overlap
    y_starts = list(range(0, max(1, H - tile + 1), stride))
    x_starts = list(range(0, max(1, W - tile + 1), stride))
    if not y_starts or y_starts[-1] != max(0, H - tile):
        y_starts.append(max(0, H - tile))
    if not x_starts or x_starts[-1] != max(0, W - tile):
        x_starts.append(max(0, W - tile))
    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(H, y0 + tile)
            x1 = min(W, x0 + tile)
            yield y0, y1, x0, x1

def relabel_sequential(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int32, copy=False)
    u = np.unique(labels)
    u = u[u != 0]
    if len(u) == 0:
        return labels
    mapping = np.zeros((u.max()+1,), dtype=np.int32)
    mapping[u] = np.arange(1, len(u)+1, dtype=np.int32)
    return mapping[labels]

def place_tile_labels(global_lab: np.ndarray, tile_lab: np.ndarray, y0: int, x0: int, next_id: int, overlap: int):
    if tile_lab is None or tile_lab.max() == 0:
        return next_id

    tile_lab = relabel_sequential(tile_lab)
    tile_lab[tile_lab > 0] += next_id

    Ht, Wt = tile_lab.shape
    y_in0 = 0 if y0 == 0 else overlap // 2
    x_in0 = 0 if x0 == 0 else overlap // 2
    y_in1 = Ht if (y0 + Ht) >= global_lab.shape[0] else (Ht - overlap // 2)
    x_in1 = Wt if (x0 + Wt) >= global_lab.shape[1] else (Wt - overlap // 2)

    gy0, gy1 = y0 + y_in0, y0 + y_in1
    gx0, gx1 = x0 + x_in0, x0 + x_in1

    sub = global_lab[gy0:gy1, gx0:gx1]
    sub_new = tile_lab[y_in0:y_in1, x_in0:x_in1]
    mask = (sub == 0) & (sub_new > 0)
    sub[mask] = sub_new[mask]
    global_lab[gy0:gy1, gx0:gx1] = sub

    return int(tile_lab.max())


# ------------------------------------------------------------
# Segmentation: Cellpose-SAM (cpsam) with manual tiling
# ------------------------------------------------------------

def segment_mip_cellpose_manual_tiling(
    mip01: np.ndarray,
    model_spec: str,
    diameter_px: float,
    flow_threshold: float,
    cellprob_threshold: float,
    tile_size: int,
    overlap_frac: float,
    use_gpu: bool
) -> np.ndarray:
    from cellpose import models

    H, W = mip01.shape
    overlap = int(round(tile_size * float(overlap_frac)))
    overlap = max(64, min(overlap, tile_size // 2))
    log(f"  [AUTOSEG] tile_size={tile_size}, overlap={overlap}px, gpu={use_gpu}")

    model = models.CellposeModel(gpu=use_gpu, pretrained_model=model_spec)

    global_lab = np.zeros((H, W), dtype=np.int32)
    next_id = 0

    tiles = list(iter_tiles_overlap(H, W, tile_size, overlap))
    for (y0, y1, x0, x1) in tqdm(tiles, desc="[AUTOSEG] tiles", ncols=110):
        tile_img = mip01[y0:y1, x0:x1]
        masks, flows, styles = model.eval(
            tile_img,
            channels=[0, 0],
            diameter=diameter_px,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            augment=False,
            resample=True
        )
        next_id = place_tile_labels(global_lab, masks, y0, x0, next_id, overlap)

        del tile_img, masks, flows, styles
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    global_lab = relabel_sequential(global_lab)
    return global_lab.astype(np.int32)

def get_or_make_labels_2d(
    vol01: np.ndarray,
    labels_path: Optional[str],
    out_path: str,
    cpsam_spec: str,
    diameter_px: float,
    flow_threshold: float,
    cellprob_threshold: float,
    tile_size: int,
    tile_overlap: float,
    use_gpu: bool
) -> np.ndarray:
    if labels_path and exists(labels_path):
        lab = tiff.imread(labels_path)
        if lab.ndim != 2:
            raise ValueError(f"Expected 2D labels (Y,X). Got {lab.shape}")
        return lab.astype(np.int32)

    if exists(out_path):
        return tiff.imread(out_path).astype(np.int32)

    mip = normalize01(max_proj(vol01), 1.0, 99.0)
    lab = segment_mip_cellpose_manual_tiling(
        mip01=mip,
        model_spec=cpsam_spec,
        diameter_px=diameter_px,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        tile_size=tile_size,
        overlap_frac=tile_overlap,
        use_gpu=use_gpu
    )
    write_labels_2d(out_path, lab)
    return lab


# ------------------------------------------------------------
# Centroids (fast)
# ------------------------------------------------------------

def centroids_from_labels_fast(labels2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lab = labels2d.astype(np.int32, copy=False)
    max_id = int(lab.max())
    if max_id == 0:
        return np.zeros((0,2), np.float32), np.zeros((0,), np.int32)

    ys, xs = np.nonzero(lab)
    ids = lab[ys, xs]
    count = np.bincount(ids, minlength=max_id+1).astype(np.float64)
    sumx = np.bincount(ids, weights=xs.astype(np.float64), minlength=max_id+1)
    sumy = np.bincount(ids, weights=ys.astype(np.float64), minlength=max_id+1)

    valid = np.where(count > 0)[0]
    valid = valid[valid != 0]
    cx = (sumx[valid] / count[valid]).astype(np.float32)
    cy = (sumy[valid] / count[valid]).astype(np.float32)
    Cxy = np.stack([cx, cy], axis=1)
    return Cxy, valid.astype(np.int32)


# ------------------------------------------------------------
# Structure + FFT shift
# ------------------------------------------------------------

def structure_log_2d_from_vol(vol01: np.ndarray, sigma_xy: float) -> np.ndarray:
    mip = max_proj(vol01)
    s = -ndi.gaussian_laplace(mip, sigma=sigma_xy)
    return normalize01(s, 1.0, 99.0)

def phase_corr_shift_2d(A: np.ndarray, B: np.ndarray) -> Tuple[float, float]:
    A = A.astype(np.float32); B = B.astype(np.float32)
    FA = np.fft.fft2(A)
    FB = np.fft.fft2(B)
    R = FA * np.conj(FB)
    R /= np.maximum(np.abs(R), 1e-9)
    r = np.abs(np.fft.ifft2(R))
    maxpos = np.unravel_index(np.argmax(r), r.shape)
    dy, dx = maxpos[0], maxpos[1]
    H, W = A.shape
    if dy > H//2: dy -= H
    if dx > W//2: dx -= W
    return float(dy), float(dx), r


# ------------------------------------------------------------
# SimpleITK 2D affine multiscale
# ------------------------------------------------------------

def sitk_from_np2d(img: np.ndarray) -> sitk.Image:
    return sitk.GetImageFromArray(img.astype(np.float32))

def np_from_sitk2d(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img).astype(np.float32)

def multiscale_affine_2d(fixed: np.ndarray, moving: np.ndarray,
                         sampling: float, lr: float, min_step: float, iters: int,
                         shrinks: Tuple[int,int,int], smooths: Tuple[float,float,float]) -> sitk.Transform:
    F = sitk_from_np2d(fixed)
    M = sitk_from_np2d(moving)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(float(sampling))
    R.SetInterpolator(sitk.sitkLinear)

    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=float(lr),
        minStep=float(min_step),
        numberOfIterations=int(iters),
        gradientMagnitudeTolerance=1e-6
    )
    R.SetOptimizerScalesFromPhysicalShift()

    tx0 = sitk.AffineTransform(2)
    tx0 = sitk.CenteredTransformInitializer(F, M, tx0, sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R.SetInitialTransform(tx0, inPlace=False)

    R.SetShrinkFactorsPerLevel(shrinkFactors=list(shrinks))
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=list(smooths))
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    return R.Execute(F, M)

def resample_2d_affine(moving2d: np.ndarray, fixed2d: np.ndarray, T: sitk.Transform) -> np.ndarray:
    M = sitk_from_np2d(moving2d)
    F = sitk_from_np2d(fixed2d)
    res = sitk.Resample(M, F, T, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    return np_from_sitk2d(res)


def resample_2d_affine_labels(moving_lab2d: np.ndarray, fixed2d_ref: np.ndarray, T: sitk.Transform) -> np.ndarray:
    """Nearest-neighbor resample of integer label image using a 2D affine transform."""
    M = sitk.GetImageFromArray(moving_lab2d.astype(np.int32))
    F = sitk_from_np2d(fixed2d_ref.astype(np.float32))
    res = sitk.Resample(M, F, T, sitk.sitkNearestNeighbor, 0, sitk.sitkInt32)
    out = sitk.GetArrayFromImage(res).astype(np.int32)
    return out


def affine_resample_volume(M: np.ndarray, F_ref: np.ndarray, T: sitk.Transform, desc: str) -> np.ndarray:
    Z = F_ref.shape[0]
    out = np.zeros_like(F_ref, dtype=np.float32)
    for z in tqdm(range(Z), desc=desc, ncols=110):
        out[z] = resample_2d_affine(M[z], F_ref[z], T)
    return out.astype(np.float32)

def apply_affine_to_points_xy(T: sitk.Transform, pts_xy: np.ndarray) -> np.ndarray:
    out = np.zeros_like(pts_xy, dtype=np.float32)
    for i in range(len(pts_xy)):
        x, y = float(pts_xy[i,0]), float(pts_xy[i,1])
        xp, yp = T.TransformPoint((x, y))
        out[i,0] = xp
        out[i,1] = yp
    return out


# ------------------------------------------------------------
# TensorFlow FCN embedder (tile-safe)
# ------------------------------------------------------------

def build_fcn_embedder(emb_dim: int, context_pool: int = 8) -> keras.Model:
    """
    FCN embedder with an explicit context branch (fast, no patch extraction).

    - Local branch: standard convs at full resolution.
    - Context branch: average-pool -> convs -> upsample back to full res.
    - Fusion: concat + 1x1 conv -> L2 normalize.

    This gives each pixel embedding both nucleus-local cues and neighborhood layout cues.
    """
    inp = keras.Input(shape=(None, None, 2), dtype=tf.float32)

    # ---- local path ----
    x = layers.Conv2D(32, 5, padding="same", activation="relu")(inp)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)

    # ---- context path ----
    c = layers.AveragePooling2D(pool_size=(context_pool, context_pool), padding="same")(inp)
    c = layers.Conv2D(32, 3, padding="same", activation="relu")(c)
    c = layers.Conv2D(64, 3, padding="same", activation="relu")(c)
    c = layers.UpSampling2D(size=(context_pool, context_pool), interpolation="bilinear")(c)

    # Crop/pad context to match local spatial dims (handles edges when H/W not divisible by pool)
    def _match_hw(tensors):
        loc, ctx = tensors
        lh = tf.shape(loc)[1]; lw = tf.shape(loc)[2]
        ctx = ctx[:, :lh, :lw, :]
        return ctx
    c = layers.Lambda(_match_hw)([x, c])

    # ---- fuse ----
    f = layers.Concatenate(axis=-1)([x, c])
    f = layers.Conv2D(emb_dim, 1, padding="same")(f)
    f = tf.math.l2_normalize(f, axis=-1)

    return keras.Model(inp, f, name="fcn_embedder_ctx")

@tf.function(reduce_retracing=True)
def info_nce_loss(Ef: tf.Tensor, Em: tf.Tensor, temperature: float) -> tf.Tensor:
    logits = tf.matmul(Ef, Em, transpose_b=True) / temperature  # [B,B]
    labels = tf.range(tf.shape(logits)[0])
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    return loss

def random_tile_coords(rng, H, W, tile):
    y0 = int(rng.integers(0, max(1, H - tile + 1)))
    x0 = int(rng.integers(0, max(1, W - tile + 1)))
    return y0, x0

def _augment_pair(rng: np.random.Generator, F: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Lightweight paired augmentations that preserve correspondence."""
    # Random brightness/contrast (same params for both)
    if rng.random() < 0.8:
        a = float(rng.uniform(0.85, 1.15))  # contrast
        b = float(rng.uniform(-0.05, 0.05))  # brightness
        F = np.clip(a * F + b, 0.0, 1.0)
        M = np.clip(a * M + b, 0.0, 1.0)
    # Mild Gaussian noise (independent)
    if rng.random() < 0.5:
        s = float(rng.uniform(0.0, 0.02))
        F = np.clip(F + rng.normal(0.0, s, size=F.shape).astype(np.float32), 0.0, 1.0)
        M = np.clip(M + rng.normal(0.0, s, size=M.shape).astype(np.float32), 0.0, 1.0)
    # Occasional blur (same sigma)
    if rng.random() < 0.2:
        sig = float(rng.uniform(0.6, 1.2))
        for ch in range(F.shape[-1]):
            F[..., ch] = ndi.gaussian_filter(F[..., ch], sigma=sig)
            M[..., ch] = ndi.gaussian_filter(M[..., ch], sigma=sig)
    return F.astype(np.float32), M.astype(np.float32)

def _pool_embeddings_for_labels(E_map: tf.Tensor, L_tile: np.ndarray, label_ids: np.ndarray) -> tf.Tensor:
    """
    Pool pixel embeddings into per-label embeddings for a given set of labels present in the tile.

    E_map: [h,w,D] float32 tensor
    L_tile: [h,w] int32 numpy
    label_ids: [B] int32 numpy (labels to pool)
    Returns: [B,D] float32 tensor (L2-normalized)
    """
    h = int(L_tile.shape[0]); w = int(L_tile.shape[1])
    D = int(E_map.shape[-1])

    # Flatten
    Lf = L_tile.reshape(-1).astype(np.int32)
    Ef = tf.reshape(E_map, (-1, D))  # [h*w,D]

    # Build a remap table: original label -> 0..B-1 (or -1)
    max_id = int(L_tile.max())
    remap = -np.ones((max_id + 1,), dtype=np.int32)
    for bi, lid in enumerate(label_ids.tolist()):
        if 0 <= lid <= max_id:
            remap[lid] = bi
    bi = remap[Lf]  # [-1..B-1]
    m = bi >= 0
    bi = bi[m].astype(np.int32)
    Ef_m = tf.gather(Ef, tf.where(m)[:, 0])

    # Sum/count per pooled id
    cnt = tf.math.unsorted_segment_sum(tf.ones((tf.shape(Ef_m)[0],), tf.float32), bi, num_segments=len(label_ids))
    sumv = tf.math.unsorted_segment_sum(Ef_m, bi, num_segments=len(label_ids))
    emb = sumv / (cnt[:, None] + 1e-6)
    emb = tf.math.l2_normalize(emb, axis=-1)
    return emb

def train_embedder_on_random_tiles(
    model: keras.Model,
    fixed_int: np.ndarray,
    fixed_str: np.ndarray,
    moving_int_aligned: np.ndarray,
    moving_str_aligned: np.ndarray,
    fixed_labels2d: np.ndarray,
    moving_labels2d_aligned: np.ndarray,
    out_ckpt: str,
    steps: int,
    tile: int,
    batch_pairs: int,
    temperature: float,
    lr: float,
    seed: int,
    pos_radius_px: float = 12.0,
    jitter_px: float = 6.0
):
    """
    High-accuracy, tile-safe training using OBJECT-LEVEL (nucleus) positives.

    Changes vs the old pixel-coordinate InfoNCE:
    - Build positives by MUTUAL nearest-neighbor nuclei after coarse global alignment (tight radius).
    - Pool embeddings per nucleus label (no patch extraction; uses label mask pooling).
    - Apply random paired augmentations.
    - Add random geometric jitter to the moving sampling window (robust to residual misalignment).

    This is much more stable than 'same (y,x) is positive' early in registration.
    """
    rng = np.random.default_rng(seed)
    H, W = fixed_int.shape
    opt = keras.optimizers.Adam(learning_rate=lr)
    ensure_dir(os.path.dirname(out_ckpt))

    # Precompute centroids for fast positive mining (in aligned frame!)
    CF_xy, _ = centroids_from_labels_fast(fixed_labels2d)
    CM_xy, _ = centroids_from_labels_fast(moving_labels2d_aligned)
    treeM = cKDTree(CM_xy.astype(np.float32))
    treeF = cKDTree(CF_xy.astype(np.float32))

    log(f"[TRAIN] tile={tile} steps={steps} batch_pairs={batch_pairs} pos_r={pos_radius_px}px jitter={jitter_px}px temp={temperature} lr={lr}")

    for s in range(1, steps + 1):
        # Random tile in fixed
        y0, x0 = random_tile_coords(rng, H, W, tile)
        y1, x1 = y0 + tile, x0 + tile

        # Add random jitter for moving crop (robust to residuals)
        jy = int(rng.integers(-int(jitter_px), int(jitter_px) + 1))
        jx = int(rng.integers(-int(jitter_px), int(jitter_px) + 1))
        my0 = int(np.clip(y0 + jy, 0, H - tile))
        mx0 = int(np.clip(x0 + jx, 0, W - tile))
        my1, mx1 = my0 + tile, mx0 + tile

        F_tile = np.stack([fixed_int[y0:y1, x0:x1], fixed_str[y0:y1, x0:x1]], axis=-1).astype(np.float32)
        M_tile = np.stack([moving_int_aligned[my0:my1, mx0:mx1], moving_str_aligned[my0:my1, mx0:mx1]], axis=-1).astype(np.float32)
        LF_tile = fixed_labels2d[y0:y1, x0:x1].astype(np.int32)
        LM_tile = moving_labels2d_aligned[my0:my1, mx0:mx1].astype(np.int32)

        F_tile, M_tile = _augment_pair(rng, F_tile, M_tile)

        # Mine mutual-NN positives among nuclei whose centroids fall within the tile windows
        # Fixed nuclei in tile:
        inF = np.where((CF_xy[:, 0] >= x0) & (CF_xy[:, 0] < x1) & (CF_xy[:, 1] >= y0) & (CF_xy[:, 1] < y1))[0]
        if len(inF) == 0:
            continue
        CF_sub = CF_xy[inF]

        # Nearest moving for each fixed, require tight radius
        dF, j_idx = treeM.query(CF_sub, k=1)
        ok = dF <= float(pos_radius_px)
        if not np.any(ok):
            continue
        inF_ok = inF[ok]
        j_idx_ok = j_idx[ok]

        # Mutual check
        CM_back = CM_xy[j_idx_ok]
        _, i_back = treeF.query(CM_back, k=1)
        mutual = (i_back == inF_ok)
        if not np.any(mutual):
            continue
        inF_pos = inF_ok[mutual]
        j_pos = j_idx_ok[mutual]

        # Sample batch_pairs positives
        if len(inF_pos) > batch_pairs:
            pick = rng.integers(0, len(inF_pos), size=(batch_pairs,), endpoint=False)
            inF_pos = inF_pos[pick]
            j_pos = j_pos[pick]

        # Convert global centroid positions to tile-local pixel coordinates for pooling
        # Get label IDs at those centroid coords (tile-local indices)
        fx = np.clip(np.round(CF_xy[inF_pos, 0]).astype(int) - x0, 0, tile - 1)
        fy = np.clip(np.round(CF_xy[inF_pos, 1]).astype(int) - y0, 0, tile - 1)
        mx = np.clip(np.round(CM_xy[j_pos, 0]).astype(int) - mx0, 0, tile - 1)
        my = np.clip(np.round(CM_xy[j_pos, 1]).astype(int) - my0, 0, tile - 1)

        labF = LF_tile[fy, fx].astype(np.int32)
        labM = LM_tile[my, mx].astype(np.int32)

        # Filter out any that hit background due to rounding/jitter
        keep = (labF > 0) & (labM > 0)
        if not np.any(keep):
            continue
        labF = labF[keep]
        labM = labM[keep]

        # Deduplicate label ids (avoid repeated labels in same batch causing weird positives)
        # Keep only first occurrence per fixed label
        seen = set()
        keep2 = []
        for idx, lid in enumerate(labF.tolist()):
            if lid not in seen:
                seen.add(lid)
                keep2.append(idx)
        if len(keep2) < 4:
            continue
        keep2 = np.array(keep2, dtype=np.int32)
        labF = labF[keep2]
        labM = labM[keep2]

        with tf.GradientTape() as tape:
            Ef_map = model(F_tile[None, ...], training=True)[0]  # [tile,tile,D]
            Em_map = model(M_tile[None, ...], training=True)[0]

            Ef = _pool_embeddings_for_labels(Ef_map, LF_tile, labF)
            Em = _pool_embeddings_for_labels(Em_map, LM_tile, labM)

            # InfoNCE across nucleus pairs (B x B)
            loss = info_nce_loss(Ef, Em, temperature)

        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        if s == 1 or s == steps or (s % max(10, steps // 20) == 0):
            log(f"  [TRAIN] step {s}/{steps} ({100.0*s/steps:5.1f}%) loss={float(loss):.4f} tileF@({y0},{x0}) tileM@({my0},{mx0}) pairs={len(labF)}")

        del F_tile, M_tile, LF_tile, LM_tile
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    model.save(out_ckpt)
    log(f"[TRAIN] saved: {out_ckpt}")



# ------------------------------------------------------------
# Tile inference + per-label pooling (no patches)
# ------------------------------------------------------------

def compute_embeddings_by_label_tiled_pooling(
    mip_int: np.ndarray,
    mip_str: np.ndarray,
    labels2d: np.ndarray,
    model: keras.Model,
    out_cache_npz: str,
    tile: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-label embeddings by:
      - iterate non-overlapping tiles
      - run model on each tile to get embedding map
      - accumulate sum of embeddings per label using bincount per dimension
    Saves/resumes from out_cache_npz.

    Returns:
      emb [N_labels, D], centroids_xy [N_labels,2], label_ids [N_labels]
    """
    if exists(out_cache_npz):
        data = np.load(out_cache_npz, allow_pickle=False)
        return data["emb"].astype(np.float32), data["centroids_xy"].astype(np.float32), data["label_ids"].astype(np.int32)

    H, W = labels2d.shape
    max_id = int(labels2d.max())
    D = int(model.output_shape[-1])

    if max_id == 0:
        emb = np.zeros((0, D), np.float32)
        centroids_xy = np.zeros((0,2), np.float32)
        label_ids = np.zeros((0,), np.int32)
        np.savez_compressed(out_cache_npz, emb=emb, centroids_xy=centroids_xy, label_ids=label_ids)
        return emb, centroids_xy, label_ids

    centroids_xy, label_ids = centroids_from_labels_fast(labels2d)

    # accumulators (float32 is enough; normalize at end)
    sum_by = np.zeros((max_id + 1, D), dtype=np.float32)
    cnt_by = np.zeros((max_id + 1,), dtype=np.float32)

    tiles = list(iter_tiles_nonoverlap(H, W, tile))
    log(f"[EMB] tile inference (non-overlap) tile={tile}  tiles={len(tiles)}  HxW={H}x{W}  labels={max_id}")

    for (y0, y1, x0, x1) in tqdm(tiles, desc="[EMB] tiles", ncols=110):
        A = np.stack([mip_int[y0:y1, x0:x1], mip_str[y0:y1, x0:x1]], axis=-1).astype(np.float32)
        L = labels2d[y0:y1, x0:x1].astype(np.int32)

        E = model(A[None, ...], training=False).numpy()[0]  # [hE,wE,D]
        # Safety: some models can produce lower-res feature maps if they include pooling/strides.
        # Resize embedding map back to label resolution so boolean masks and pooling are consistent.
        if (E.shape[0] != L.shape[0]) or (E.shape[1] != L.shape[1]):
            E = tf.image.resize(E, size=(L.shape[0], L.shape[1]), method="bilinear").numpy()
        Lf = L.reshape(-1)
        Ef = E.reshape((-1, D))

        m = Lf > 0
        if np.any(m):
            ids = Lf[m].astype(np.int32)
            cnt_by += np.bincount(ids, minlength=max_id+1).astype(np.float32)
            # bincount per dim (D=64; tile count is manageable with tile=1024)
            for d in range(D):
                sum_by[:, d] += np.bincount(ids, weights=Ef[m, d].astype(np.float32), minlength=max_id+1).astype(np.float32)

        del A, L, E, Lf, Ef
        gc.collect()

    # mean + normalize
    emb_by = np.zeros((max_id+1, D), dtype=np.float32)
    valid = cnt_by > 0
    emb_by[valid] = sum_by[valid] / cnt_by[valid, None]
    emb = emb_by[label_ids].astype(np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

    np.savez_compressed(out_cache_npz, emb=emb, centroids_xy=centroids_xy, label_ids=label_ids)
    log(f"[EMB] saved: {out_cache_npz}")
    return emb, centroids_xy, label_ids


# ------------------------------------------------------------
# Matching + confidence
# ------------------------------------------------------------

def build_kdtree(CM_xy: np.ndarray) -> cKDTree:
    return cKDTree(CM_xy.astype(np.float32))

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def match_with_predicted_radius(
    eF: np.ndarray,
    eM: np.ndarray,
    CF_xy: np.ndarray,
    CM_xy: np.ndarray,
    pred_xy: np.ndarray,
    radius_px: float
):
    tree = build_kdtree(CM_xy)
    Nf = len(CF_xy)

    best_j = np.full((Nf,), -1, dtype=np.int32)
    best_s = np.full((Nf,), np.nan, dtype=np.float32)
    margin = np.full((Nf,), np.nan, dtype=np.float32)

    for i in tqdm(range(Nf), desc=f"[MATCH] radius={radius_px:.1f}", ncols=110):
        cand = tree.query_ball_point(pred_xy[i], float(radius_px))
        if not cand:
            continue
        sims = np.array([cosine_sim(eF[i], eM[j]) for j in cand], dtype=np.float32)
        order = np.argsort(-sims)
        j1 = cand[int(order[0])]
        s1 = float(sims[int(order[0])])
        s2 = float(sims[int(order[1])]) if len(order) > 1 else -1.0
        best_j[i] = int(j1)
        best_s[i] = float(s1)
        margin[i] = float(s1 - s2)

    # enforce one-to-one by best similarity
    match_j = best_j.copy()
    inv: Dict[int, List[int]] = {}
    for i, j in enumerate(match_j):
        if j >= 0:
            inv.setdefault(int(j), []).append(int(i))
    for j, ilist in inv.items():
        if len(ilist) <= 1:
            continue
        ilist_sorted = sorted(ilist, key=lambda ii: -float(best_s[ii]))
        for ii in ilist_sorted[1:]:
            match_j[ii] = -1

    resid = np.full((Nf,), np.inf, dtype=np.float32)
    ok = np.where(match_j >= 0)[0]
    if len(ok) > 0:
        resid[ok] = np.linalg.norm(CM_xy[match_j[ok]] - pred_xy[ok], axis=1).astype(np.float32)

    return match_j, best_s, margin, resid

def classify_confidence(best_s: np.ndarray, margin: np.ndarray, resid: np.ndarray,
                        green_min_sim: float, green_min_margin: float, green_max_resid: float,
                        yellow_min_sim: float, yellow_min_margin: float, yellow_max_resid: float) -> np.ndarray:
    conf = np.zeros((len(best_s),), dtype=np.uint8)  # 0 red, 1 yellow, 2 green
    green = (best_s >= green_min_sim) & (margin >= green_min_margin) & (resid <= green_max_resid)
    yellow = (best_s >= yellow_min_sim) & (margin >= yellow_min_margin) & (resid <= yellow_max_resid)
    conf[green] = 2
    conf[(~green) & yellow] = 1
    return conf


# ------------------------------------------------------------
# Local field fitting on grid (O(N))
# ------------------------------------------------------------

def fit_incremental_field_grid(
    CF_xy: np.ndarray,
    CM_xy: np.ndarray,
    match_j: np.ndarray,
    anchor_idx: np.ndarray,
    current_u: np.ndarray,
    current_v: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    sigma_smooth: float
) -> Tuple[np.ndarray, np.ndarray]:
    Gy = len(grid_y)
    Gx = len(grid_x)

    u_interp = RegularGridInterpolator((grid_y, grid_x), current_u, bounds_error=False, fill_value=0.0)
    v_interp = RegularGridInterpolator((grid_y, grid_x), current_v, bounds_error=False, fill_value=0.0)

    du_sum = np.zeros((Gy, Gx), dtype=np.float64)
    dv_sum = np.zeros((Gy, Gx), dtype=np.float64)
    cnt = np.zeros((Gy, Gx), dtype=np.float64)

    def nearest_idx(val, axis):
        i = int(np.clip(np.searchsorted(axis, val) - 1, 0, len(axis)-1))
        if i+1 < len(axis) and abs(axis[i+1] - val) < abs(axis[i] - val):
            return i+1
        return i

    for i in anchor_idx.tolist():
        j = int(match_j[i])
        if j < 0:
            continue
        x, y = float(CF_xy[i,0]), float(CF_xy[i,1])

        u0 = float(u_interp([[y, x]])[0])
        v0 = float(v_interp([[y, x]])[0])

        du = float((CM_xy[j,0] - CF_xy[i,0]) - u0)
        dv = float((CM_xy[j,1] - CF_xy[i,1]) - v0)

        iy = nearest_idx(y, grid_y)
        ix = nearest_idx(x, grid_x)

        du_sum[iy, ix] += du
        dv_sum[iy, ix] += dv
        cnt[iy, ix] += 1.0

    du = np.zeros((Gy, Gx), dtype=np.float32)
    dv = np.zeros((Gy, Gx), dtype=np.float32)
    m = cnt > 0
    du[m] = (du_sum[m] / cnt[m]).astype(np.float32)
    dv[m] = (dv_sum[m] / cnt[m]).astype(np.float32)

    empty = ~m
    if np.any(empty):
        dist, inds = ndi.distance_transform_edt(empty, return_indices=True)
        du = du[inds[0], inds[1]]
        dv = dv[inds[0], inds[1]]

    if sigma_smooth and sigma_smooth > 0:
        du = ndi.gaussian_filter(du, sigma=float(sigma_smooth))
        dv = ndi.gaussian_filter(dv, sigma=float(sigma_smooth))

    return du.astype(np.float32), dv.astype(np.float32)


# ------------------------------------------------------------
# dz (optional)
# ------------------------------------------------------------

def tile_stack(vol: np.ndarray, x: int, y: int, half: int) -> np.ndarray:
    Z, H, W = vol.shape
    x0, x1 = x-half, x+half
    y0, y1 = y-half, y+half
    out = np.zeros((Z, 2*half, 2*half), dtype=np.float32)
    xs0 = max(0, x0); ys0 = max(0, y0)
    xs1 = min(W, x1); ys1 = min(H, y1)
    px0 = xs0 - x0; py0 = ys0 - y0
    out[:, py0:py0+(ys1-ys0), px0:px0+(xs1-xs0)] = vol[:, ys0:ys1, xs0:xs1]
    return out

def z_shift_profile_mean(Ft: np.ndarray, Mt: np.ndarray) -> float:
    Z = Ft.shape[0]
    fz = np.array([float(Ft[z].mean()) for z in range(Z)], dtype=np.float32)
    mz = np.array([float(Mt[z].mean()) for z in range(Z)], dtype=np.float32)
    fz = (fz - fz.mean()) / (fz.std() + 1e-6)
    mz = (mz - mz.mean()) / (mz.std() + 1e-6)
    cc = np.correlate(fz, mz, mode="full")
    k = int(np.argmax(cc))
    dz = k - (len(fz) - 1)
    return float(dz)


# ------------------------------------------------------------
# Final warp
# ------------------------------------------------------------

def warp_moving_to_fixed(M_aff: np.ndarray, u_func, v_func, dz_func) -> np.ndarray:
    Z, H, W = M_aff.shape
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32),
                         indexing="ij")
    u = u_func(xx, yy)
    v = v_func(xx, yy)
    dz = dz_func(xx, yy)

    out = np.zeros_like(M_aff, dtype=np.float32)
    for z in tqdm(range(Z), desc="[WARP] trilinear", ncols=110):
        zz = (z + dz).astype(np.float32)
        coords = np.array([zz, (yy + v).astype(np.float32), (xx + u).astype(np.float32)], dtype=np.float32)
        out[z] = ndi.map_coordinates(M_aff, coords, order=1, mode="nearest")
    return out.astype(np.float32)


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

@dataclass
class Config:
    p_low: float = 1.0
    p_high: float = 99.0
    sigma_xy: float = 1.2

    cpsam_spec: str = "cpsam"
    cpsam_diameter_px: float = 50.0
    cpsam_flow_threshold: float = 0.4
    cpsam_cellprob_threshold: float = 0.0
    cpsam_tile_size: int = 2048
    cpsam_tile_overlap: float = 0.20

    affine_sampling: float = 0.15
    affine_lr: float = 2.0
    affine_min_step: float = 1e-3
    affine_iters: int = 250
    affine_shrinks: Tuple[int,int,int] = (8, 4, 2)
    affine_smooths: Tuple[float,float,float] = (2.0, 1.0, 0.0)

    emb_dim: int = 64

    # TRAINING (tile-safe, nucleus-level)
    train_steps: int = 800
    train_tile: int = 512
    train_batch_pairs: int = 256
    train_pos_radius_px: float = 12.0
    train_jitter_px: float = 6.0
    train_temp: float = 0.08
    train_lr: float = 2e-4
    train_seed: int = 123

    # EMB inference tile (bigger reduces tile count)
    emb_tile: int = 1024

    # Cycles
    cycles: int = 3
    radius_schedule: Tuple[float,float,float] = (240.0, 170.0, 120.0)

    # Confidence thresholds (start a bit forgiving; you can tighten later)
    green_min_sim: float = 0.60
    green_min_margin: float = 0.05
    green_max_resid: float = 30.0

    yellow_min_sim: float = 0.50
    yellow_min_margin: float = 0.02
    yellow_max_resid: float = 80.0

    # Field grid
    field_stride: int = 48
    field_sigma: float = 1.25

    # dz
    do_dz: bool = True
    dz_tile_half: int = 64
    dz_stride: int = 256


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------

def run_pipeline(cfg: Config,
                 fixed_path: str,
                 moving_path: str,
                 out_dir: str,
                 fixed_labels_in: Optional[str],
                 moving_labels_in: Optional[str],
                 use_gpu: bool,
                 do_train: bool):

    ensure_dir(out_dir)

    # ------------------ STEP 1: pad + save ------------------
    log(f"[STEP 1/10] Read stacks + pad to same shape ({now()})")
    meta_path = os.path.join(out_dir, "shapes.json")
    if not exists(meta_path):
        F0 = read_stack(fixed_path)
        M0 = read_stack(moving_path)
        Fp, Mp, fixed_shape, moving_shape, padded_shape = pad_to_match_shape(F0, M0)
        np.save(os.path.join(out_dir, "fixed_padded.npy"), Fp)
        np.save(os.path.join(out_dir, "moving_padded.npy"), Mp)
        json.dump({
            "fixed_shape": fixed_shape,
            "moving_shape": moving_shape,
            "padded_shape": list(padded_shape)
        }, open(meta_path, "w"), indent=2)
        del F0, M0, Fp, Mp
        gc.collect()
    meta = json.load(open(meta_path, "r"))
    fixed_shape = tuple(meta["fixed_shape"])
    padded_shape = tuple(meta["padded_shape"])
    log(f"  fixed_shape={fixed_shape} padded_shape={padded_shape}")

    Fp = np.load(os.path.join(out_dir, "fixed_padded.npy"), mmap_mode="r")
    Mp = np.load(os.path.join(out_dir, "moving_padded.npy"), mmap_mode="r")

    fixed_padded_tif = os.path.join(out_dir, "fixed_padded.tif")
    if not os.path.exists(fixed_padded_tif):
        import tifffile as tiff
        tiff.imwrite(
            fixed_padded_tif,
            Fp.astype(np.float32),
            bigtiff=True,
            metadata={"axes": "ZYX"}
        )

    # ------------------ STEP 2: normalize + MIPs ------------------
    log(f"[STEP 2/10] Normalize + make MIPs ({now()})")
    fixed_norm_path = os.path.join(out_dir, "fixed_norm.npy")
    moving_norm_path = os.path.join(out_dir, "moving_norm.npy")
    fixed_mip_int_path = os.path.join(out_dir, "fixed_mip_int.npy")
    moving_mip_int_path = os.path.join(out_dir, "moving_mip_int.npy")
    fixed_mip_str_path = os.path.join(out_dir, "fixed_mip_str.npy")
    moving_mip_str_path = os.path.join(out_dir, "moving_mip_str.npy")

    if not (exists(fixed_norm_path) and exists(moving_norm_path)
            and exists(fixed_mip_int_path) and exists(moving_mip_int_path)
            and exists(fixed_mip_str_path) and exists(moving_mip_str_path)):
        F01 = normalize01(np.asarray(Fp), cfg.p_low, cfg.p_high)
        M01 = normalize01(np.asarray(Mp), cfg.p_low, cfg.p_high)
        np.save(fixed_norm_path, F01)
        np.save(moving_norm_path, M01)
        np.save(fixed_mip_int_path, max_proj(F01))
        np.save(moving_mip_int_path, max_proj(M01))
        np.save(fixed_mip_str_path, structure_log_2d_from_vol(F01, cfg.sigma_xy))
        np.save(moving_mip_str_path, structure_log_2d_from_vol(M01, cfg.sigma_xy))
        del F01, M01
        gc.collect()

    F01 = np.load(fixed_norm_path)
    M01 = np.load(moving_norm_path)
    fixed_mip_int = np.load(fixed_mip_int_path)
    moving_mip_int = np.load(moving_mip_int_path)
    fixed_mip_str = np.load(fixed_mip_str_path)
    moving_mip_str = np.load(moving_mip_str_path)

    # ------------------ STEP 3: labels ------------------
    log(f"[STEP 3/10] Labels 2D (load/autoseg, resumable) ({now()})")
    fixed_lab_out = os.path.join(out_dir, "fixed_labels_2d.tif")
    moving_lab_out = os.path.join(out_dir, "moving_labels_2d.tif")

    LF = get_or_make_labels_2d(
        vol01=np.asarray(F01),
        labels_path=fixed_labels_in,
        out_path=fixed_lab_out,
        cpsam_spec=cfg.cpsam_spec,
        diameter_px=cfg.cpsam_diameter_px,
        flow_threshold=cfg.cpsam_flow_threshold,
        cellprob_threshold=cfg.cpsam_cellprob_threshold,
        tile_size=cfg.cpsam_tile_size,
        tile_overlap=cfg.cpsam_tile_overlap,
        use_gpu=use_gpu
    )
    LM = get_or_make_labels_2d(
        vol01=np.asarray(M01),
        labels_path=moving_labels_in,
        out_path=moving_lab_out,
        cpsam_spec=cfg.cpsam_spec,
        diameter_px=cfg.cpsam_diameter_px,
        flow_threshold=cfg.cpsam_flow_threshold,
        cellprob_threshold=cfg.cpsam_cellprob_threshold,
        tile_size=cfg.cpsam_tile_size,
        tile_overlap=cfg.cpsam_tile_overlap,
        use_gpu=use_gpu
    )

    # ------------------ STEP 4: FFT shift ------------------
    log(f"[STEP 4/10] FFT shift (coarse) ({now()})")
    fft_path = os.path.join(out_dir, "fft_shift.json")
    if not exists(fft_path):
        dy, dx , R_copy = phase_corr_shift_2d(np.asarray(fixed_mip_str), np.asarray(moving_mip_str))
        json.dump({"dy": dy, "dx": dx}, open(fft_path, "w"), indent=2)
    sh = json.load(open(fft_path, "r"))
    
    dy_fft, dx_fft = float(sh["dy"]), float(sh["dx"])
    log(f"  FFT dy={dy_fft:.2f} dx={dx_fft:.2f}")
    np.save("R.npy", R_copy)

    # ------------------ STEP 5: 2D affine ------------------
    log(f"[STEP 5/10] Multiscale 2D affine ({now()})")
    affine_path = os.path.join(out_dir, "affine_2d.tfm")
    if not exists(affine_path):
        mov_str_fft = ndi.shift(np.asarray(moving_mip_str), shift=(dy_fft, dx_fft), order=1, mode="nearest")
        T_aff = multiscale_affine_2d(
            fixed=np.asarray(fixed_mip_str),
            moving=mov_str_fft,
            sampling=cfg.affine_sampling,
            lr=cfg.affine_lr,
            min_step=cfg.affine_min_step,
            iters=cfg.affine_iters,
            shrinks=cfg.affine_shrinks,
            smooths=cfg.affine_smooths
        )
        sitk.WriteTransform(T_aff, affine_path)
        del mov_str_fft
        gc.collect()
    T_aff = sitk.ReadTransform(affine_path)

    # Prepare globally aligned moving MIPs (for training + embeddings)
    log(f"[STEP 6/10] Build global-aligned moving MIPs ({now()})")
    fixed_int = np.asarray(fixed_mip_int).astype(np.float32)
    fixed_str = np.asarray(fixed_mip_str).astype(np.float32)

    mov_int = np.asarray(moving_mip_int).astype(np.float32)
    mov_str = np.asarray(moving_mip_str).astype(np.float32)
    mov_int_fft = ndi.shift(mov_int, shift=(dy_fft, dx_fft), order=1, mode="nearest")
    mov_str_fft = ndi.shift(mov_str, shift=(dy_fft, dx_fft), order=1, mode="nearest")
    mov_int_aff = resample_2d_affine(mov_int_fft, fixed_int, T_aff)
    mov_str_aff = resample_2d_affine(mov_str_fft, fixed_str, T_aff)
    del mov_int, mov_str, mov_int_fft, mov_str_fft
    gc.collect()


    # IMPORTANT: align moving labels into the SAME global frame as mov_*_aff before pooling embeddings.
    # Otherwise per-label pooling mixes wrong pixels and embeddings become meaningless.
    moving_lab_global_path = os.path.join(out_dir, "moving_labels_2d_global.tif")
    if exists(moving_lab_global_path):
        LM_global = tiff.imread(moving_lab_global_path).astype(np.int32)
    else:
        LM_fft = ndi.shift(LM.astype(np.int32), shift=(dy_fft, dx_fft), order=0, mode="constant", cval=0)
        LM_global = resample_2d_affine_labels(LM_fft, fixed_int, T_aff)
        LM_global = relabel_sequential(LM_global)
        write_labels_2d(moving_lab_global_path, LM_global)
        del LM_fft
        gc.collect()


    # ------------------ STEP 7: model + training (tile-safe) ------------------
    log(f"[STEP 7/10] TensorFlow FCN embedder (tile-safe training) ({now()})")
    fcn_model_path = os.path.join(out_dir, "tf_fcn_embedder.keras")
    train_flag = os.path.join(out_dir, "train_done.flag")

    if exists(fcn_model_path):
        model = keras.models.load_model(fcn_model_path, compile=False, custom_objects={'MatchHW': MatchHW})
    else:
        model = build_fcn_embedder(cfg.emb_dim, context_pool=cfg.emb_context_pool)
        model.save(fcn_model_path)

    if do_train and not exists(train_flag):
        
        train_embedder_on_random_tiles(
            model=model,
            fixed_int=fixed_int,
            fixed_str=fixed_str,
            moving_int_aligned=mov_int_aff,
            moving_str_aligned=mov_str_aff,
            fixed_labels2d=LF,
            moving_labels2d_aligned=LM_global,
            out_ckpt=fcn_model_path,
            steps=cfg.train_steps,
            tile=cfg.train_tile,
            batch_pairs=cfg.train_batch_pairs,
            temperature=cfg.train_temp,
            lr=cfg.train_lr,
            seed=cfg.train_seed,
            pos_radius_px=cfg.train_pos_radius_px,
            jitter_px=cfg.train_jitter_px
        )
        with open(train_flag, "w") as f:
            f.write("ok\n")
        model = keras.models.load_model(fcn_model_path, compile=False, custom_objects={'MatchHW': MatchHW})
    elif do_train and exists(train_flag):
        log("[TRAIN] train_done.flag exists -> skipping training")
    else:
        log("[TRAIN] --train not set -> using existing model (may be weak if never trained)")

    # ------------------ STEP 8: compute embeddings (tiled pooling) ------------------
    log(f"[STEP 8/10] Compute per-nucleus embeddings (tiled inference + pooling) ({now()})")
    cache_fixed = os.path.join(out_dir, "cache_fixed_embeddings.npz")
    cache_moving = os.path.join(out_dir, "cache_moving_embeddings_global.npz")

    eF, CF_xy, F_ids = compute_embeddings_by_label_tiled_pooling(
        mip_int=fixed_int,
        mip_str=fixed_str,
        labels2d=LF,
        model=model,
        out_cache_npz=cache_fixed,
        tile=cfg.emb_tile
    )
    eM, CM_xy_local, M_ids = compute_embeddings_by_label_tiled_pooling(
        mip_int=mov_int_aff,
        mip_str=mov_str_aff,
        labels2d=LM_global,
        model=model,
        out_cache_npz=cache_moving,
        tile=cfg.emb_tile
    )

    # Moving centroids already in global frame (labels were resampled to fixed grid)
    CM_xy, _ = centroids_from_labels_fast(LM_global)

    log(f"  fixed_nuclei={len(CF_xy)} moving_nuclei={len(CM_xy)}")

    # ------------------ STEP 9: multi-cycle refinement ------------------
    log(f"[STEP 9/10] Multi-cycle match → field refine ({now()})")
    Zt, Ht, Wt = padded_shape
    s = int(cfg.field_stride)
    grid_y = np.arange(0, Ht, s, dtype=np.float32)
    grid_x = np.arange(0, Wt, s, dtype=np.float32)
    Gy, Gx = len(grid_y), len(grid_x)

    total_u = np.zeros((Gy, Gx), dtype=np.float32)
    total_v = np.zeros((Gy, Gx), dtype=np.float32)

    for k in range(cfg.cycles):
        cyc_dir = os.path.join(out_dir, f"cycle_{k:02d}")
        ensure_dir(cyc_dir)
        report_path = os.path.join(cyc_dir, "report_cycle.json")
        match_npz = os.path.join(cyc_dir, "match_arrays.npz")
        field_npz = os.path.join(cyc_dir, "field_total_uv.npz")

        radius_k = cfg.radius_schedule[k] if k < len(cfg.radius_schedule) else cfg.radius_schedule[-1]

        # resume
        if exists(report_path) and exists(match_npz) and exists(field_npz):
            log(f"[CYCLE {k}] resume: loading saved field")
            dat = np.load(field_npz, allow_pickle=False)
            total_u = dat["total_u"].astype(np.float32)
            total_v = dat["total_v"].astype(np.float32)
            continue

        uI = RegularGridInterpolator((grid_y, grid_x), total_u, bounds_error=False, fill_value=0.0)
        vI = RegularGridInterpolator((grid_y, grid_x), total_v, bounds_error=False, fill_value=0.0)
        pts = np.stack([CF_xy[:,1], CF_xy[:,0]], axis=1).astype(np.float32)  # (y,x)
        u_at = uI(pts).astype(np.float32)
        v_at = vI(pts).astype(np.float32)
        pred_xy = CF_xy + np.stack([u_at, v_at], axis=1).astype(np.float32)

        log(f"[CYCLE {k}] matching radius={radius_k}")
        match_j, best_s, margin, resid = match_with_predicted_radius(
            eF=eF, eM=eM,
            CF_xy=CF_xy,
            CM_xy=CM_xy,
            pred_xy=pred_xy,
            radius_px=radius_k
        )

        conf = classify_confidence(
            best_s=best_s, margin=margin, resid=resid,
            green_min_sim=cfg.green_min_sim,
            green_min_margin=cfg.green_min_margin,
            green_max_resid=cfg.green_max_resid,
            yellow_min_sim=cfg.yellow_min_sim,
            yellow_min_margin=cfg.yellow_min_margin,
            yellow_max_resid=cfg.yellow_max_resid
        )
        green_idx = np.where((match_j >= 0) & (conf == 2))[0].astype(np.int32)
        yellow_idx = np.where((match_j >= 0) & (conf == 1))[0].astype(np.int32)

        log(f"  [CYCLE {k}] matched={(match_j>=0).sum()} / {len(CF_xy)}   GREEN={len(green_idx)}  YELLOW={len(yellow_idx)}")

        np.savez_compressed(match_npz, match_j=match_j, best_s=best_s, margin=margin, resid=resid, conf=conf)

        # Adaptive anchor choice:
        # If greens are scarce, include yellows to stabilize the field (you can tighten thresholds later).
        if len(green_idx) < 200:
            log(f"  [CYCLE {k}] greens low -> using GREEN+YELLOW as anchors for this cycle")
            anchors = np.unique(np.concatenate([green_idx, yellow_idx])).astype(np.int32)
        else:
            anchors = green_idx

        if len(anchors) < 20:
            log(f"  [CYCLE {k}] STOP: not enough anchors ({len(anchors)}) to fit a stable field")
            json.dump({
                "cycle": k,
                "radius": float(radius_k),
                "matched": int((match_j>=0).sum()),
                "green": int(len(green_idx)),
                "yellow": int(len(yellow_idx)),
                "note": "Stopped: not enough anchors"
            }, open(report_path, "w"), indent=2)
            np.savez_compressed(field_npz, total_u=total_u, total_v=total_v, grid_x=grid_x, grid_y=grid_y)
            break

        du, dv = fit_incremental_field_grid(
            CF_xy=CF_xy,
            CM_xy=CM_xy,
            match_j=match_j,
            anchor_idx=anchors,
            current_u=total_u,
            current_v=total_v,
            grid_x=grid_x,
            grid_y=grid_y,
            sigma_smooth=float(cfg.field_sigma)
        )
        total_u = (total_u + du).astype(np.float32)
        total_v = (total_v + dv).astype(np.float32)

        json.dump({
            "cycle": k,
            "radius": float(radius_k),
            "matched": int((match_j>=0).sum()),
            "green": int(len(green_idx)),
            "yellow": int(len(yellow_idx)),
            "mean_resid_green": float(np.mean(resid[green_idx])) if len(green_idx) else float("nan"),
            "median_resid_green": float(np.median(resid[green_idx])) if len(green_idx) else float("nan"),
            "mean_cos_green": float(np.mean(best_s[green_idx])) if len(green_idx) else float("nan"),
            "notes": "Field updated using anchors (greens, and yellows if greens sparse)."
        }, open(report_path, "w"), indent=2)

        np.savez_compressed(field_npz, total_u=total_u, total_v=total_v, grid_x=grid_x, grid_y=grid_y)
        gc.collect()

    # Save total field
    total_field_npz = os.path.join(out_dir, "transform_fields_total.npz")
    np.savez_compressed(
        total_field_npz,
        total_u=total_u, total_v=total_v, grid_x=grid_x, grid_y=grid_y,
        fft_dx=np.float32(dx_fft), fft_dy=np.float32(dy_fft),
        affine_params=np.array(T_aff.GetParameters(), dtype=np.float64),
        padded_shape=np.array(list(padded_shape), dtype=np.int32),
        fixed_shape=np.array(list(fixed_shape), dtype=np.int32),
    )
    log(f"[FIELD] saved: {total_field_npz}")

    # Confidence map from last available cycle
    conf_map_path = os.path.join(out_dir, "confidence_map_fixed_centroids.tif")
    if not exists(conf_map_path):
        conf_final = None
        for k in reversed(range(cfg.cycles)):
            p = os.path.join(out_dir, f"cycle_{k:02d}", "match_arrays.npz")
            if exists(p):
                conf_final = np.load(p, allow_pickle=False)["conf"].astype(np.uint8)
                break
        if conf_final is None:
            conf_final = np.zeros((len(CF_xy),), dtype=np.uint8)

        H, W = LF.shape
        conf_map = np.zeros((H, W), dtype=np.uint8)
        for i in range(len(CF_xy)):
            x = int(round(CF_xy[i,0])); y = int(round(CF_xy[i,1]))
            if 0 <= x < W and 0 <= y < H:
                conf_map[y, x] = conf_final[i]
        write_u8(conf_map_path, conf_map)

    # ------------------ STEP 10: optional dz + final warp ------------------
    log(f"[STEP 10/10] Final 3D warp (resume) ({now()})")
    aligned_path = os.path.join(out_dir, "moving_aligned_to_fixed.tif")
    if exists(aligned_path):
        log(f"[FINAL] Found aligned output, skipping: {aligned_path}")
        log(f"[DONE] out_dir={out_dir}")
        log(f"  fixed labels:  {os.path.join(out_dir,'fixed_labels_2d.tif')}")
        log(f"  moving labels: {os.path.join(out_dir,'moving_labels_2d.tif')}")
        return

    # dz grid (optional)
    dz_grid = None
    dz_meta_path = os.path.join(out_dir, "dz_grid.npz")
    if cfg.do_dz:
        if exists(dz_meta_path):
            dz_meta = np.load(dz_meta_path, allow_pickle=False)
            dz_grid = dz_meta["dz"].astype(np.float32)
            ys = dz_meta["ys"].astype(np.float32)
            xs = dz_meta["xs"].astype(np.float32)
            log("[DZ] loaded dz_grid.npz")
        else:
            log("[DZ] computing sparse dz grid (tilewise z-profile correlation)")
            # build globally aligned moving volume (FFT + affine) for dz sampling
            M_fft = ndi.shift(np.asarray(M01), shift=(0, dy_fft, dx_fft), order=1, mode="nearest").astype(np.float32)
            M_aff = affine_resample_volume(M_fft, np.asarray(F01), T_aff, desc="[AFF] for dz")
            del M_fft
            gc.collect()

            Z, H, W = np.asarray(F01).shape
            xs = np.arange(cfg.dz_tile_half+2, W-(cfg.dz_tile_half+2), cfg.dz_stride, dtype=np.int32)
            ys = np.arange(cfg.dz_tile_half+2, H-(cfg.dz_tile_half+2), cfg.dz_stride, dtype=np.int32)
            dz_grid = np.zeros((len(ys), len(xs)), dtype=np.float32)

            uI = RegularGridInterpolator((grid_y, grid_x), total_u, bounds_error=False, fill_value=0.0)
            vI = RegularGridInterpolator((grid_y, grid_x), total_v, bounds_error=False, fill_value=0.0)

            for yi, y0 in enumerate(tqdm(ys.tolist(), desc="[DZ] rows", ncols=110)):
                for xi, x0 in enumerate(xs.tolist()):
                    Ft = tile_stack(np.asarray(F01), x0, y0, cfg.dz_tile_half)
                    Mt = tile_stack(M_aff, x0, y0, cfg.dz_tile_half)

                    u0 = float(uI([[y0, x0]])[0])
                    v0 = float(vI([[y0, x0]])[0])
                    Mt_xy = ndi.shift(Mt, shift=(0, v0, u0), order=1, mode="nearest")
                    dz_grid[yi, xi] = float(z_shift_profile_mean(Ft, Mt_xy))

                    del Ft, Mt, Mt_xy
                gc.collect()

            np.savez_compressed(dz_meta_path, dz=dz_grid.astype(np.float32),
                                ys=ys.astype(np.float32), xs=xs.astype(np.float32))
            log(f"[DZ] saved: {dz_meta_path}")
            del M_aff
            gc.collect()

    # build interpolators u,v,dz
    u_interp = RegularGridInterpolator((grid_y, grid_x), total_u, bounds_error=False, fill_value=0.0)
    v_interp = RegularGridInterpolator((grid_y, grid_x), total_v, bounds_error=False, fill_value=0.0)

    def u_func(x, y):
        pts = np.stack([y, x], axis=-1)
        return u_interp(pts).astype(np.float32)

    def v_func(x, y):
        pts = np.stack([y, x], axis=-1)
        return v_interp(pts).astype(np.float32)

    if cfg.do_dz and dz_grid is not None:
        dz_interp = RegularGridInterpolator((ys.astype(np.float32), xs.astype(np.float32)),
                                            dz_grid.astype(np.float32),
                                            bounds_error=False, fill_value=0.0)
        def dz_func(x, y):
            x = np.asarray(x, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            pts = np.stack([y, x], axis=-1)
            return dz_interp(pts).astype(np.float32)
    else:
        def dz_func(x, y):
            return np.zeros_like(x, dtype=np.float32)

    # build globally aligned moving volume (FFT + affine), then apply local warp
    log("[FINAL] applying FFT + affine to full moving volume")
    M_fft = ndi.shift(np.asarray(Mp), shift=(0, dy_fft, dx_fft), order=1, mode="nearest").astype(np.float32)
    M_aff = affine_resample_volume(M_fft, np.asarray(Fp), T_aff, desc="[AFF] full")
    del M_fft
    gc.collect()

    log("[FINAL] applying local XY (+ optional dz) warp")
    M_warp = warp_moving_to_fixed(M_aff, u_func=u_func, v_func=v_func, dz_func=dz_func)
    M_out = crop_to_fixed(M_warp, fixed_shape)
    write_stack(aligned_path, M_out)

    del M_aff, M_warp, M_out
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    log(f"[DONE] out_dir={out_dir}")
    log(f"  fixed labels:  {os.path.join(out_dir,'fixed_labels_2d.tif')}")
    log(f"  moving labels: {os.path.join(out_dir,'moving_labels_2d.tif')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", required=True, help="Fixed 3D stack TIFF (Z,Y,X)")
    ap.add_argument("--moving", required=True, help="Moving 3D stack TIFF (Z,Y,X)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--fixed_labels", default=None, help="Optional fixed 2D labels TIFF")
    ap.add_argument("--moving_labels", default=None, help="Optional moving 2D labels TIFF")
    ap.add_argument("--gpu", action="store_true", help="Use GPU where possible")
    ap.add_argument("--train", action="store_true", help="Train TF embedder (tile-safe, recommended)")

    ap.add_argument("--diameter", type=float, default=None, help="Cellpose diameter px")
    ap.add_argument("--cycles", type=int, default=None, help="Number of cycles")
    ap.add_argument("--train_steps", type=int, default=None, help="Training steps")
    ap.add_argument("--train_tile", type=int, default=None, help="Training tile size")
    ap.add_argument("--emb_tile", type=int, default=None, help="Embedding inference tile size")
    ap.add_argument("--no_dz", action="store_true", help="Disable dz estimation")

    args = ap.parse_args()

    if args.gpu:
        setup_torch_gpu()
        setup_tf_gpu()
    else:
        log("[GPU] --gpu not set (still runs, slower).")

    cfg = Config()
    if args.diameter is not None:
        cfg.cpsam_diameter_px = float(args.diameter)
    if args.cycles is not None:
        cfg.cycles = int(args.cycles)
    if args.train_steps is not None:
        cfg.train_steps = int(args.train_steps)
    if args.train_tile is not None:
        cfg.train_tile = int(args.train_tile)
    if args.emb_tile is not None:
        cfg.emb_tile = int(args.emb_tile)
    if args.no_dz:
        cfg.do_dz = False

    run_pipeline(
        cfg=cfg,
        fixed_path=args.fixed,
        moving_path=args.moving,
        out_dir=args.out,
        fixed_labels_in=args.fixed_labels,
        moving_labels_in=args.moving_labels,
        use_gpu=bool(args.gpu),
        do_train=bool(args.train)
    )

if __name__ == "__main__":
    main()