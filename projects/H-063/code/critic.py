import math
from dataclasses import dataclass
from typing import Any, Dict, List

import cv2
import numpy as np
from skimage import measure

from .preprocess import cleanup_mask_u8


def mask_boundary(mask_u8: np.ndarray) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(m, k, 1)
    ero = cv2.erode(m, k, 1)
    return (dil > ero).astype(np.uint8)


def edge_map(gray_u8: np.ndarray) -> np.ndarray:
    return (cv2.Canny(gray_u8, 50, 150) > 0).astype(np.uint8)


def edge_alignment(gray_u8: np.ndarray, mask_u8: np.ndarray) -> float:
    b = mask_boundary(mask_u8)
    e = edge_map(gray_u8)
    e2 = cv2.dilate(e, np.ones((3, 3), np.uint8), 1)
    return float((b & e2).sum() / (b.sum() + 1e-6))


def intensity_separation(gray_u8: np.ndarray, mask_u8: np.ndarray) -> float:
    g = gray_u8.astype(np.float32)
    m = mask_u8 > 0
    if m.sum() < 30 or (~m).sum() < 30:
        return 0.0
    inside, outside = g[m], g[~m]
    mu_in, mu_out = float(np.median(inside)), float(np.median(outside))
    mad_in = float(np.median(np.abs(inside - mu_in)) + 1e-6)
    mad_out = float(np.median(np.abs(outside - mu_out)) + 1e-6)
    return abs(mu_in - mu_out) / (0.5 * (mad_in + mad_out) + 1e-6)


def shape_stats(mask_u8: np.ndarray) -> Dict[str, float]:
    m = (mask_u8 > 0).astype(np.uint8)
    H, W = m.shape
    area_frac = float(m.sum() / (H * W + 1e-6))
    lbl = measure.label(m, connectivity=2)
    regs = measure.regionprops(lbl)
    n_comp = float(len(regs))
    largest = float(max([r.area for r in regs], default=0))
    return {"area_frac": area_frac, "n_comp": n_comp, "largest_area": largest}


def critic_score(gray_u8: np.ndarray, mask_u8: np.ndarray, sam_score: float, module_weight: float) -> Dict[str, Any]:
    if (mask_u8 > 0).sum() < 30:
        return {"quality": 0.0, "reason": "empty"}
    e = edge_alignment(gray_u8, mask_u8)
    sep = intensity_separation(gray_u8, mask_u8)
    sep_n = 1.0 - math.exp(-0.35 * sep)
    st = shape_stats(mask_u8)
    penalty = 0.0
    if st["area_frac"] > 0.80:
        penalty += 0.30
    if st["n_comp"] > 4000:
        penalty += 0.25
    s_n = 1.0 / (1.0 + math.exp(-6.0 * (sam_score - 0.5)))
    raw = (0.48 * e + 0.30 * sep_n + 0.22 * s_n) * module_weight - penalty
    quality = float(np.clip(raw, 0.0, 1.0))
    return {"quality": quality, "edge": float(e), "sep": float(sep), "sam": float(sam_score), "shape": st, "penalty": float(penalty)}


def critic_score_classical(gray_u8: np.ndarray, mask_u8: np.ndarray, module_weight: float) -> Dict[str, Any]:
    if (mask_u8 > 0).sum() < 30:
        return {"quality": 0.0, "reason": "empty"}
    e = edge_alignment(gray_u8, mask_u8)
    sep = intensity_separation(gray_u8, mask_u8)
    sep_n = 1.0 - math.exp(-0.35 * sep)
    st = shape_stats(mask_u8)
    penalty = 0.0
    if st["area_frac"] > 0.80:
        penalty += 0.30
    if st["n_comp"] > 4000:
        penalty += 0.25
    raw = (0.58 * e + 0.42 * sep_n) * module_weight - penalty
    quality = float(np.clip(raw, 0.0, 1.0))
    return {"quality": quality, "edge": float(e), "sep": float(sep), "shape": st, "penalty": float(penalty)}


def masks_iou_u8(a_u8: np.ndarray, b_u8: np.ndarray, eps=1e-6) -> float:
    a = a_u8 > 0
    b = b_u8 > 0
    inter = (a & b).sum()
    uni = (a | b).sum()
    return float(inter / (uni + eps))


@dataclass
class Cand:
    mask_u8: np.ndarray
    proposal: Any
    sam_score: float
    critic: Dict[str, Any]
    total: float


def select_diverse_covering(
    cands: List[Cand],
    max_instances=200,
    min_quality=0.30,
    min_area_frac=0.00006,
    max_area_frac=0.30,
    iou_thresh=0.22,
    coverage_boost=0.55,
) -> List[Cand]:
    if not cands:
        return []
    H, W = cands[0].mask_u8.shape
    covered = np.zeros((H, W), dtype=np.uint8)
    selected: List[Cand] = []
    cands = sorted(cands, key=lambda c: c.total, reverse=True)
    for c in cands:
        if c.total < min_quality:
            continue
        area = (c.mask_u8 > 0).sum()
        area_frac = float(area / (H * W))
        if area_frac < min_area_frac or area_frac > max_area_frac:
            continue
        ok = True
        for s in selected:
            if masks_iou_u8(c.mask_u8, s.mask_u8) > iou_thresh:
                ok = False
                break
        if not ok:
            continue
        new_pix = ((c.mask_u8 > 0) & (covered == 0)).sum()
        gain = float(new_pix / (area + 1e-6))
        boosted = c.total + coverage_boost * gain
        selected.append(Cand(mask_u8=c.mask_u8, proposal=c.proposal, sam_score=c.sam_score, critic=c.critic, total=float(boosted)))
        covered = np.maximum(covered, c.mask_u8)
        if len(selected) >= max_instances:
            break
    selected.sort(key=lambda c: c.total, reverse=True)
    return selected


def cleanup_and_score(gray_u8, mask, sam_score, module_weight):
    cleaned = cleanup_mask_u8(mask)
    crit = critic_score(gray_u8, cleaned, sam_score=sam_score, module_weight=module_weight)
    total = float(crit.get("quality", 0.0))
    return cleaned, crit, total


def cleanup_and_score_classical(gray_u8, mask, module_weight):
    cleaned = cleanup_mask_u8(mask)
    crit = critic_score_classical(gray_u8, cleaned, module_weight=module_weight)
    total = float(crit.get("quality", 0.0))
    return cleaned, crit, total
