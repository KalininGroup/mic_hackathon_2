"""Patch selection and window utilities (matching latest notebook)."""
from __future__ import annotations

import numpy as np
from scipy import fftpack
from scipy.ndimage import sobel, laplace

__all__ = [
    "hann2d",
    "split_into_patches",
    "compute_patch_feature",
    "select_diverse_patches",
    "choose_hr_patches",
]


def hann2d(ps: int) -> np.ndarray:
    """2D Hann window for overlap blending."""
    w = np.hanning(ps).astype(np.float32)
    win = np.outer(w, w)
    return win / (win.max() + 1e-8)


def split_into_patches(img: np.ndarray, grid: int = 8):
    """Split a square image into grid×grid patches."""
    img = np.squeeze(img)
    if img.ndim != 2:
        raise ValueError("2D image is required.")
    H, W = img.shape
    if H != W:
        raise ValueError("Only square images are supported.")
    if H % grid != 0:
        raise ValueError(f"Image size {H} is not divisible by grid {grid}.")

    patch_size = H // grid
    patches, coords = [], []
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patches.append(img[i:i + patch_size, j:j + patch_size])
            coords.append((i, j))
    return patches, coords, patch_size


def compute_patch_feature(patch: np.ndarray):
    """Compute patch features (mean/std/grad/lap/high-freq)."""
    patch = patch.astype(np.float32)
    mean = patch.mean()
    std = patch.std()
    gx = sobel(patch, axis=0)
    gy = sobel(patch, axis=1)
    grad_energy = np.mean(np.sqrt(gx ** 2 + gy ** 2))
    lap_energy = np.mean(np.abs(laplace(patch)))
    fft_mag = np.abs(fftpack.fftshift(fftpack.fft2(patch)))
    h, w = fft_mag.shape
    cy, cx = h // 2, w // 2
    high_freq = fft_mag[cy - 5:cy + 5, cx - 5:cx + 5].mean()
    return np.array([mean, std, grad_energy, lap_energy, high_freq])


def select_diverse_patches(features: np.ndarray, hf_scores: np.ndarray, n_select: int, lambda_hf: float = 0.3):
    """Select patches with farthest-first + high-frequency weighting."""
    selected = []
    first = int(np.argmax(hf_scores))
    selected.append(first)
    while len(selected) < n_select:
        best_idx, best_score = None, -np.inf
        for i in range(len(features)):
            if i in selected:
                continue
            min_dist = min(np.linalg.norm(features[i] - features[j]) for j in selected)
            score = min_dist + lambda_hf * hf_scores[i]
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(int(best_idx))
    return selected


def choose_hr_patches(lr_npy_path, n_select: int = 10, lambda_hf: float = 0.3, grid: int = 8):
    """Patch-selection pipeline from an LR npy path."""
    img = np.load(lr_npy_path)
    if img.ndim == 3:
        img = img.squeeze()
    patches_list, coords, patch_size = split_into_patches(img, grid=grid)
    feats, hf_scores = [], []
    for p in patches_list:
        feat = compute_patch_feature(p)
        feats.append(feat)
        hf_scores.append(feat[2] + feat[3])  # grad + lap
    feats = np.stack(feats)
    hf_scores = np.array(hf_scores)
    selected_ids = select_diverse_patches(feats, hf_scores, n_select=n_select, lambda_hf=lambda_hf)
    selected_coords = [coords[i] for i in selected_ids]
    return selected_ids, selected_coords, img, patch_size
