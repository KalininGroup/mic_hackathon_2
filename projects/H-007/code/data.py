"""Data loading and patch preparation (latest notebook logic)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.ndimage import zoom

from .util import split_into_patches, compute_patch_feature, select_diverse_patches

__all__ = [
    "load_pair",
    "normalize_pair",
    "prepare_patch_pairs",
]


def load_pair(hr_path: str | Path, lr_path: str | Path, interp_order: int = 0):
    """Load HR/LR npy and resize LR to HR shape if needed."""
    hr = np.load(hr_path)
    lr = np.load(lr_path)
    hr = np.squeeze(hr)
    lr = np.squeeze(lr)
    if hr.shape != lr.shape:
        factors = [h / l for h, l in zip(hr.shape, lr.shape)]
        lr = zoom(lr, factors, order=interp_order)
    return hr, lr


def normalize_pair(hr: np.ndarray, lr: np.ndarray):
    """Normalize using HR mean/std and return normalized HR/LR plus stats."""
    mean = float(hr.mean())
    std = float(hr.std() + 1e-8)
    hr_n = (hr - mean) / std
    lr_n = (lr - mean) / std
    return hr_n, lr_n, mean, std


def _select_coords(lr: np.ndarray, n_select: int, grid: int, lambda_hf: float):
    patches, coords, patch_size = split_into_patches(lr, grid=grid)
    feats, hf_scores = [], []
    for p in patches:
        feat = compute_patch_feature(p)
        feats.append(feat)
        hf_scores.append(feat[2] + feat[3])
    feats = np.stack(feats)
    hf_scores = np.array(hf_scores)
    selected_ids = select_diverse_patches(feats, hf_scores, n_select=n_select, lambda_hf=lambda_hf)
    selected_coords = [coords[i] for i in selected_ids]
    return selected_ids, selected_coords, coords, patch_size


def _extract_patches(img: np.ndarray, coords: List[Tuple[int, int]], patch_size: int):
    return np.stack([img[i:i + patch_size, j:j + patch_size] for (i, j) in coords])


def prepare_patch_pairs(
    lr: np.ndarray,
    hr: np.ndarray,
    n_select: int,
    grid: int = 8,
    lambda_hf: float = 0.3,
    return_val: bool = True,
):
    """Prepare LR/HR patch pairs at identical coords; optionally include val patches."""
    selected_ids, selected_coords, all_coords, patch_size = _select_coords(
        lr, n_select=n_select, grid=grid, lambda_hf=lambda_hf
    )

    lr_patches = _extract_patches(lr, selected_coords, patch_size)
    hr_patches = _extract_patches(hr, selected_coords, patch_size)

    val_lr = val_hr = val_coords = None
    if return_val:
        val_coords = [c for idx, c in enumerate(all_coords) if idx not in selected_ids]
        if val_coords:
            val_lr = _extract_patches(lr, val_coords, patch_size)
            val_hr = _extract_patches(hr, val_coords, patch_size)

    return {
        "lr_patches": lr_patches,
        "hr_patches": hr_patches,
        "coords": selected_coords,
        "patch_size": patch_size,
        "val_lr": val_lr,
        "val_hr": val_hr,
        "val_coords": val_coords,
    }
