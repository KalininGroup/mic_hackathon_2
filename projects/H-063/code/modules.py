import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from skimage import feature as skfeature
from skimage import filters, measure, morphology, segmentation
from scipy import ndimage as ndi


@dataclass
class Proposal:
    box: Optional[List[int]]
    pos_points: List[List[int]] = field(default_factory=list)
    neg_points: List[List[int]] = field(default_factory=list)
    module: str = "unknown"
    note: str = ""
    weight: float = 1.0
    target: str = "generic"


def invert_if_needed(gray_u8: np.ndarray, polarity: str) -> np.ndarray:
    if polarity.lower() == "dark":
        return (255 - gray_u8)
    return gray_u8


def bbox_from_region(bbox_rc, W, H, pad=6):
    minr, minc, maxr, maxc = bbox_rc
    x1 = max(0, minc - pad)
    y1 = max(0, minr - pad)
    x2 = min(W - 1, maxc + pad)
    y2 = min(H - 1, maxr + pad)
    return [int(x1), int(y1), int(x2), int(y2)]


def sample_points_in_mask(mask: np.ndarray, k=2, seed=0):
    rng = np.random.default_rng(seed)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    idx = rng.choice(len(xs), size=min(k, len(xs)), replace=False)
    return [[int(xs[i]), int(ys[i])] for i in idx]


def module_threshold_regions(
    gray_u8: np.ndarray,
    polarity: str,
    target: str,
    methods=("otsu", "li", "yen"),
    top_k=120,
    pad=8,
    min_size=30,
) -> List[Proposal]:
    g = invert_if_needed(gray_u8, polarity).astype(np.float32)
    H, W = gray_u8.shape
    props = []
    thr_fns = {"otsu": filters.threshold_otsu, "li": filters.threshold_li, "yen": filters.threshold_yen}
    methods = list(methods) if isinstance(methods, (list, tuple)) else [methods]
    for m in methods:
        if m not in thr_fns:
            continue
        thr = thr_fns[m](g)
        msk = (g > thr)
        msk = morphology.remove_small_objects(msk, min_size=min_size)
        lbl = measure.label(msk, connectivity=2)
        regs = measure.regionprops(lbl)
        regs.sort(key=lambda r: r.area, reverse=True)
        for r in regs[:top_k]:
            box = bbox_from_region(r.bbox, W, H, pad=pad)
            cy, cx = r.centroid
            props.append(
                Proposal(
                    box=box,
                    pos_points=[[int(cx), int(cy)]],
                    module=f"threshold_regions:{m}",
                    note=f"area={r.area}",
                    weight=1.0,
                    target=target,
                )
            )
    return props


def module_adaptive_threshold(
    gray_u8: np.ndarray,
    polarity: str,
    target: str,
    block_size=15,
    offset=-5,
    top_k=160,
    pad=8,
    min_size=30,
) -> List[Proposal]:
    H, W = gray_u8.shape
    g = invert_if_needed(gray_u8, polarity)
    bs = int(block_size)
    if bs < 3:
        bs = 3
    if bs % 2 == 0:
        bs += 1
    th = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=bs, C=float(offset)
    )
    msk = th > 0
    msk = morphology.remove_small_objects(msk, min_size=min_size)
    lbl = measure.label(msk, connectivity=2)
    regs = measure.regionprops(lbl)
    regs.sort(key=lambda r: r.area, reverse=True)
    props = []
    for r in regs[:top_k]:
        box = bbox_from_region(r.bbox, W, H, pad=pad)
        cy, cx = r.centroid
        props.append(
            Proposal(
                box=box,
                pos_points=[[int(cx), int(cy)]],
                module="adaptive_threshold",
                note=f"area={r.area}, bs={bs}, C={offset}",
                weight=1.05,
                target=target,
            )
        )
    return props


def module_log_blobs(
    gray_u8: np.ndarray,
    polarity: str,
    target: str,
    min_sigma=2,
    max_sigma=18,
    num_sigma=10,
    threshold=0.03,
    max_blobs=250,
) -> List[Proposal]:
    g = invert_if_needed(gray_u8, polarity).astype(np.float32) / 255.0
    H, W = gray_u8.shape
    blobs = skfeature.blob_log(g, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
    props = []
    if blobs is None or len(blobs) == 0:
        return props
    blobs = blobs[:max_blobs]
    for (y, x, s) in blobs:
        r = int(max(6, 3 * s))
        x1 = int(np.clip(x - r, 0, W - 1))
        y1 = int(np.clip(y - r, 0, H - 1))
        x2 = int(np.clip(x + r, 0, W - 1))
        y2 = int(np.clip(y + r, 0, H - 1))
        props.append(
            Proposal(
                box=[x1, y1, x2, y2],
                pos_points=[[int(x), int(y)]],
                module="blob_log",
                note=f"sigma={s:.1f}",
                weight=1.0,
                target=target,
            )
        )
    return props


def module_watershed_instances(
    gray_u8: np.ndarray,
    polarity: str,
    target: str,
    min_size=60,
    peak_footprint=9,
    compactness=0.0,
    top_k=250,
    pad=8,
) -> List[Proposal]:
    H, W = gray_u8.shape
    g = invert_if_needed(gray_u8, polarity).astype(np.float32) / 255.0

    thr = filters.threshold_otsu(g)
    msk = g > thr
    msk = morphology.remove_small_objects(msk, min_size=min_size)
    if msk.sum() < 50:
        return []

    dist = ndi.distance_transform_edt(msk)
    coords = skfeature.peak_local_max(dist, footprint=np.ones((int(peak_footprint), int(peak_footprint))), labels=msk)
    markers = np.zeros_like(gray_u8, dtype=np.int32)
    for i, (y, x) in enumerate(coords, start=1):
        markers[y, x] = i
    if markers.max() == 0:
        ys, xs = np.where(msk)
        if len(ys) == 0:
            return []
        markers[int(np.median(ys)), int(np.median(xs))] = 1

    labels = segmentation.watershed(-dist, markers, mask=msk, compactness=float(compactness))
    regs = measure.regionprops(labels)
    regs.sort(key=lambda r: r.area, reverse=True)

    props = []
    for r in regs[:top_k]:
        if r.area < min_size:
            continue
        region_mask = labels == r.label
        box = bbox_from_region(r.bbox, W, H, pad=pad)
        cy, cx = r.centroid
        pts = [[int(cx), int(cy)]] + sample_points_in_mask(region_mask, k=2, seed=r.label)
        props.append(
            Proposal(
                box=box,
                pos_points=pts[:3],
                neg_points=[],
                module="watershed",
                note=f"area={r.area}",
                weight=1.20,
                target=target,
            )
        )
    return props


def module_edge_contours(
    gray_u8: np.ndarray,
    target: str,
    sigma=1.0,
    low_threshold=0.08,
    high_threshold=0.20,
    min_area=150,
    top_k=250,
    pad=6,
) -> List[Proposal]:
    H, W = gray_u8.shape
    g = gray_u8.astype(np.float32) / 255.0
    edges = skfeature.canny(g, sigma=float(sigma), low_threshold=float(low_threshold), high_threshold=float(high_threshold))
    edges = morphology.binary_dilation(edges, morphology.disk(1))
    edges_u8 = edges.astype(np.uint8) * 255

    cnts, _ = cv2.findContours(edges_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    props = []
    for c in cnts[:top_k]:
        area = cv2.contourArea(c)
        if area < float(min_area):
            continue
        x, y, w, h = cv2.boundingRect(c)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W - 1, x + w + pad)
        y2 = min(H - 1, y + h + pad)
        cx, cy = x + w // 2, y + h // 2
        props.append(
            Proposal(
                box=[x1, y1, x2, y2],
                pos_points=[[int(cx), int(cy)]],
                module="edge_contours",
                note=f"area={area:.0f}",
                weight=0.90,
                target=target,
            )
        )
    return props


MODULE_SPECS = {
    "threshold_regions": {"allowed": {"methods", "top_k", "pad", "min_size"}, "aliases": {"method": "methods"}},
    "adaptive_threshold": {"allowed": {"block_size", "offset", "top_k", "pad", "min_size"}, "aliases": {}},
    "watershed": {"allowed": {"min_size", "peak_footprint", "compactness", "top_k", "pad"}, "aliases": {"markers": None}},
    "blob_log": {"allowed": {"min_sigma", "max_sigma", "num_sigma", "threshold", "max_blobs"}, "aliases": {}},
    "edge_contours": {"allowed": {"sigma", "low_threshold", "high_threshold", "min_area", "top_k", "pad"}, "aliases": {}},
}


def sanitize_module_params(module: str, params: dict) -> dict:
    params = params or {}
    if module not in MODULE_SPECS:
        return {}
    spec = MODULE_SPECS[module]
    allowed = spec["allowed"]
    aliases = spec.get("aliases", {})
    out = {}
    for k, v in params.items():
        if k in aliases:
            mapped = aliases[k]
            if mapped is None:
                continue
            k = mapped
        if module == "threshold_regions" and k == "methods" and isinstance(v, str):
            v = [v]
        if k in allowed:
            out[k] = v
    if module == "adaptive_threshold" and "block_size" in out:
        bs = int(out["block_size"])
        if bs < 3:
            bs = 3
        if bs % 2 == 0:
            bs += 1
        out["block_size"] = bs
    return out


def default_targets():
    return [
        {
            "name": "Voids",
            "polarity": "dark",
            "goal": "instances",
            "suggest_modules": [
                {"module": "watershed", "params": {"min_size": 60, "peak_footprint": 9, "compactness": 0.0, "top_k": 250, "pad": 8}},
                {"module": "blob_log", "params": {"min_sigma": 2, "max_sigma": 18, "num_sigma": 10, "threshold": 0.03, "max_blobs": 250}},
                {"module": "threshold_regions", "params": {"methods": ["otsu", "li"], "top_k": 200, "pad": 8, "min_size": 30}},
                {"module": "adaptive_threshold", "params": {"block_size": 15, "offset": -5, "top_k": 200, "pad": 8, "min_size": 30}},
                {"module": "edge_contours", "params": {"sigma": 1.0, "low_threshold": 0.08, "high_threshold": 0.20, "min_area": 150, "top_k": 200, "pad": 6}},
            ],
            "selection": {"min_quality": 0.30, "iou_thresh": 0.22, "max_instances": 250},
        }
    ]


def sanitize_plan(plan: dict) -> dict:
    if not isinstance(plan, dict):
        plan = {}
    targets = plan.get("targets", [])
    if not isinstance(targets, list) or len(targets) == 0:
        targets = default_targets()
    clean_targets = []
    for t in targets:
        if not isinstance(t, dict):
            continue
        name = t.get("name", "Target")
        polarity = t.get("polarity", "dark")
        if polarity not in ("dark", "bright"):
            polarity = "dark"
        goal = t.get("goal", "instances")

        mods = []
        for m in t.get("suggest_modules") or []:
            if not isinstance(m, dict):
                continue
            mname = m.get("module")
            if mname not in MODULE_SPECS:
                continue
            params = sanitize_module_params(mname, m.get("params", {}) or {})
            mods.append({"module": mname, "params": params})
        if not mods:
            mods = [
                {"module": "watershed", "params": {"min_size": 60, "peak_footprint": 9, "compactness": 0.0, "top_k": 250, "pad": 8}},
                {"module": "threshold_regions", "params": {"methods": ["otsu", "li"], "top_k": 200, "pad": 8, "min_size": 30}},
            ]

        sel = t.get("selection", {}) or {}
        min_q = float(np.clip(float(sel.get("min_quality", 0.35)), 0.15, 0.75))
        iou_t = float(np.clip(float(sel.get("iou_thresh", 0.25)), 0.10, 0.60))
        max_i = int(np.clip(int(sel.get("max_instances", 200)), 10, 600))

        clean_targets.append(
            {
                "name": name,
                "polarity": polarity,
                "goal": goal,
                "suggest_modules": mods,
                "selection": {"min_quality": min_q, "iou_thresh": iou_t, "max_instances": max_i},
            }
        )

    gsel = plan.get("selection", {}) or {}
    out = {
        "targets": clean_targets,
        "selection": {
            "min_quality": float(np.clip(float(gsel.get("min_quality", 0.35)), 0.15, 0.75)),
            "iou_thresh": float(np.clip(float(gsel.get("iou_thresh", 0.25)), 0.10, 0.60)),
            "max_instances": int(np.clip(int(gsel.get("max_instances", 300)), 10, 800)),
        },
    }
    return out


def apply_target_priors(plan: dict, target_priors: dict) -> dict:
    """
    Apply per-target area priors by updating selection thresholds.
    target_priors: {target_name: {min_quality, iou_thresh, max_instances}}
    """
    if not target_priors:
        return plan
    for t in plan.get("targets", []):
        name = t.get("name")
        pri = target_priors.get(name) or {}
        sel = t.setdefault("selection", {})
        if "min_quality" in pri:
            sel["min_quality"] = float(pri["min_quality"])
        if "iou_thresh" in pri:
            sel["iou_thresh"] = float(pri["iou_thresh"])
        if "max_instances" in pri:
            sel["max_instances"] = int(pri["max_instances"])
    return plan


def apply_prompt_priors(plan: dict, user_prompt: str) -> dict:
    """
    Heuristic prompt parsing to adjust module params/polarity.
    """
    if not user_prompt:
        return plan
    text = user_prompt.lower()
    wants_small = "small" in text or "tiny" in text
    wants_large = "large" in text or "big" in text
    wants_void = "void" in text or "pore" in text
    wants_grain = "grain" in text or "cell" in text or "particle" in text
    wants_bright = "bright" in text
    wants_dark = "dark" in text

    targets = plan.get("targets", [])
    apply_all = len(targets) == 1
    for t in targets:
        name = str(t.get("name", "")).lower()
        match_void = ("void" in name or "pore" in name) and wants_void
        match_grain = ("grain" in name or "cell" in name or "particle" in name) and wants_grain
        if not apply_all and not (match_void or match_grain):
            continue

        if wants_dark:
            t["polarity"] = "dark"
        elif wants_bright:
            t["polarity"] = "bright"
        elif match_void:
            t["polarity"] = "dark"
        elif match_grain:
            t["polarity"] = "bright"

        for m in t.get("suggest_modules", []):
            params = m.setdefault("params", {})
            if wants_small:
                if "min_size" in params:
                    params["min_size"] = max(10, int(params["min_size"]) // 2)
                if "min_sigma" in params:
                    params["min_sigma"] = max(1, float(params["min_sigma"]) * 0.5)
                if "max_blobs" in params:
                    params["max_blobs"] = int(params["max_blobs"]) + 100
            if wants_large:
                if "min_size" in params:
                    params["min_size"] = int(params["min_size"]) * 2
                if "min_sigma" in params:
                    params["min_sigma"] = float(params["min_sigma"]) * 1.5
                if "max_blobs" in params:
                    params["max_blobs"] = max(50, int(params["max_blobs"]) - 50)
    return plan


# -------- Classical masks (no SAM) --------
def module_threshold_masks(
    gray_u8: np.ndarray,
    polarity: str,
    target: str,
    methods=("otsu", "li", "yen"),
    top_k=120,
    pad=8,
    min_size=30,
):
    g = invert_if_needed(gray_u8, polarity).astype(np.float32)
    H, W = gray_u8.shape
    masks_props = []
    thr_fns = {"otsu": filters.threshold_otsu, "li": filters.threshold_li, "yen": filters.threshold_yen}
    methods = list(methods) if isinstance(methods, (list, tuple)) else [methods]
    for m in methods:
        if m not in thr_fns:
            continue
        thr = thr_fns[m](g)
        msk = g > thr
        msk = morphology.remove_small_objects(msk, min_size=min_size)
        lbl = measure.label(msk, connectivity=2)
        regs = measure.regionprops(lbl)
        regs.sort(key=lambda r: r.area, reverse=True)
        for r in regs[:top_k]:
            region_mask = (lbl == r.label).astype(np.uint8) * 255
            box = bbox_from_region(r.bbox, W, H, pad=pad)
            cy, cx = r.centroid
            prop = Proposal(
                box=box,
                pos_points=[[int(cx), int(cy)]],
                module=f"classical_threshold:{m}",
                note=f"area={r.area}",
                weight=1.0,
                target=target,
            )
            masks_props.append((region_mask, prop))
    return masks_props


def module_adaptive_masks(
    gray_u8: np.ndarray,
    polarity: str,
    target: str,
    block_size=15,
    offset=-5,
    top_k=160,
    pad=8,
    min_size=30,
):
    H, W = gray_u8.shape
    g = invert_if_needed(gray_u8, polarity)
    bs = int(block_size)
    if bs < 3:
        bs = 3
    if bs % 2 == 0:
        bs += 1
    th = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=bs, C=float(offset)
    )
    msk = th > 0
    msk = morphology.remove_small_objects(msk, min_size=min_size)
    lbl = measure.label(msk, connectivity=2)
    regs = measure.regionprops(lbl)
    regs.sort(key=lambda r: r.area, reverse=True)
    masks_props = []
    for r in regs[:top_k]:
        region_mask = (lbl == r.label).astype(np.uint8) * 255
        box = bbox_from_region(r.bbox, W, H, pad=pad)
        cy, cx = r.centroid
        prop = Proposal(
            box=box,
            pos_points=[[int(cx), int(cy)]],
            module="classical_adaptive",
            note=f"area={r.area}, bs={bs}, C={offset}",
            weight=1.05,
            target=target,
        )
        masks_props.append((region_mask, prop))
    return masks_props


def module_blob_masks(
    gray_u8: np.ndarray,
    polarity: str,
    target: str,
    min_sigma=2,
    max_sigma=18,
    num_sigma=10,
    threshold=0.03,
    max_blobs=250,
):
    g = invert_if_needed(gray_u8, polarity).astype(np.float32) / 255.0
    H, W = gray_u8.shape
    blobs = skfeature.blob_log(g, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
    masks_props = []
    if blobs is None or len(blobs) == 0:
        return masks_props
    blobs = blobs[:max_blobs]
    for (y, x, s) in blobs:
        r = int(max(6, 3 * s))
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(mask, (int(x), int(y)), r, 255, -1)
        x1 = int(np.clip(x - r, 0, W - 1))
        y1 = int(np.clip(y - r, 0, H - 1))
        x2 = int(np.clip(x + r, 0, W - 1))
        y2 = int(np.clip(y + r, 0, H - 1))
        prop = Proposal(
            box=[x1, y1, x2, y2],
            pos_points=[[int(x), int(y)]],
            module="classical_blob_log",
            note=f"sigma={s:.1f}",
            weight=1.0,
            target=target,
        )
        masks_props.append((mask, prop))
    return masks_props


def module_watershed_masks(
    gray_u8: np.ndarray,
    polarity: str,
    target: str,
    min_size=60,
    peak_footprint=9,
    compactness=0.0,
    top_k=250,
    pad=8,
):
    H, W = gray_u8.shape
    g = invert_if_needed(gray_u8, polarity).astype(np.float32) / 255.0

    thr = filters.threshold_otsu(g)
    msk = g > thr
    msk = morphology.remove_small_objects(msk, min_size=min_size)
    if msk.sum() < 50:
        return []

    dist = ndi.distance_transform_edt(msk)
    coords = skfeature.peak_local_max(dist, footprint=np.ones((int(peak_footprint), int(peak_footprint))), labels=msk)
    markers = np.zeros_like(gray_u8, dtype=np.int32)
    for i, (y, x) in enumerate(coords, start=1):
        markers[y, x] = i
    if markers.max() == 0:
        ys, xs = np.where(msk)
        if len(ys) == 0:
            return []
        markers[int(np.median(ys)), int(np.median(xs))] = 1

    labels = segmentation.watershed(-dist, markers, mask=msk, compactness=float(compactness))
    regs = measure.regionprops(labels)
    regs.sort(key=lambda r: r.area, reverse=True)

    masks_props = []
    for r in regs[:top_k]:
        if r.area < min_size:
            continue
        region_mask = (labels == r.label).astype(np.uint8) * 255
        box = bbox_from_region(r.bbox, W, H, pad=pad)
        cy, cx = r.centroid
        pts = [[int(cx), int(cy)]] + sample_points_in_mask(region_mask > 0, k=2, seed=r.label)
        prop = Proposal(
            box=box,
            pos_points=pts[:3],
            neg_points=[],
            module="classical_watershed",
            note=f"area={r.area}",
            weight=1.15,
            target=target,
        )
        masks_props.append((region_mask, prop))
    return masks_props


def module_edge_masks(
    gray_u8: np.ndarray,
    target: str,
    sigma=1.0,
    low_threshold=0.08,
    high_threshold=0.20,
    min_area=150,
    top_k=250,
    pad=6,
):
    H, W = gray_u8.shape
    g = gray_u8.astype(np.float32) / 255.0
    edges = skfeature.canny(g, sigma=float(sigma), low_threshold=float(low_threshold), high_threshold=float(high_threshold))
    edges = morphology.binary_dilation(edges, morphology.disk(1))
    edges_u8 = edges.astype(np.uint8) * 255

    cnts, _ = cv2.findContours(edges_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    masks_props = []
    for c in cnts[:top_k]:
        area = cv2.contourArea(c)
        if area < float(min_area):
            continue
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, thickness=-1)
        x, y, w, h = cv2.boundingRect(c)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W - 1, x + w + pad)
        y2 = min(H - 1, y + h + pad)
        cx, cy = x + w // 2, y + h // 2
        prop = Proposal(
            box=[x1, y1, x2, y2],
            pos_points=[[int(cx), int(cy)]],
            module="classical_edge_contours",
            note=f"area={area:.0f}",
            weight=0.90,
            target=target,
        )
        masks_props.append((mask, prop))
    return masks_props


def classical_masks_from_plan(gray_u8: np.ndarray, plan: dict):
    """
    Generate masks directly from classical modules (no SAM).
    Returns list of (mask_u8, Proposal).
    """
    plan = sanitize_plan(plan)
    masks_props = []
    for target in plan["targets"]:
        polarity = target.get("polarity", "dark")
        for mod in target["suggest_modules"]:
            mname = mod["module"]
            params = sanitize_module_params(mname, mod.get("params", {}) or {})
            if mname == "threshold_regions":
                masks_props.extend(module_threshold_masks(gray_u8, polarity=polarity, target=target["name"], **params))
            elif mname == "adaptive_threshold":
                masks_props.extend(module_adaptive_masks(gray_u8, polarity=polarity, target=target["name"], **params))
            elif mname == "watershed":
                masks_props.extend(module_watershed_masks(gray_u8, polarity=polarity, target=target["name"], **params))
            elif mname == "blob_log":
                masks_props.extend(module_blob_masks(gray_u8, polarity=polarity, target=target["name"], **params))
            elif mname == "edge_contours":
                masks_props.extend(module_edge_masks(gray_u8, target=target["name"], **params))
            else:
                continue
    return masks_props


def run_one_module(gray_u8: np.ndarray, target: dict, mod: dict) -> List[Proposal]:
    tname = target["name"]
    polarity = target.get("polarity", "dark")
    mname = mod["module"]
    params = sanitize_module_params(mname, mod.get("params", {}) or {})

    if mname == "threshold_regions":
        return module_threshold_regions(gray_u8, polarity=polarity, target=tname, **params)
    if mname == "adaptive_threshold":
        return module_adaptive_threshold(gray_u8, polarity=polarity, target=tname, **params)
    if mname == "watershed":
        return module_watershed_instances(gray_u8, polarity=polarity, target=tname, **params)
    if mname == "blob_log":
        return module_log_blobs(gray_u8, polarity=polarity, target=tname, **params)
    if mname == "edge_contours":
        return module_edge_contours(gray_u8, target=tname, **params)
    return []
