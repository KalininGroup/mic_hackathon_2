import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .critic import Cand, cleanup_and_score, cleanup_and_score_classical, select_diverse_covering
from .human_loop import hints_to_proposals
from .llm_agent import (
    adjust_via_langgraph,
    apply_adjustments,
    apply_reviewer_updates,
    fallback_plan,
    plan_via_langgraph,
    review_via_langgraph,
    fallback_reviewer_decision,
)
from .modules import (
    Proposal,
    apply_prompt_priors,
    apply_target_priors,
    classical_masks_from_plan,
    run_one_module,
    sanitize_plan,
)
from .preprocess import (
    cleanup_mask_u8,
    draw_boundary,
    ensure_rgb_uint8,
    mask_to_uint8,
    overlay_instances,
    overlay_mask_rgb,
    preprocess_micrograph,
    to_gray,
    to_uint8,
    union_mask,
)
from .results_logger import save_run
from .sam_wrapper import SAM2Wrapper
from .state import HistoryItem, HumanHints, RunSettings


def run_modules_parallel(gray_u8: np.ndarray, plan: Dict[str, Any]) -> List[Proposal]:
    plan = sanitize_plan(plan)
    proposals: List[Proposal] = []
    futures = []
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        for target in plan["targets"]:
            for mod in target["suggest_modules"]:
                futures.append(ex.submit(run_one_module, gray_u8, target, mod))
        for f in as_completed(futures):
            try:
                proposals.extend(f.result())
            except Exception:
                continue

    seen = set()
    out = []
    for p in proposals:
        key = (p.target, p.module, tuple(p.box) if p.box else None, tuple(p.pos_points[0]) if p.pos_points else None)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def add_fallback_grid(proposals: List[Proposal], gray_u8: np.ndarray, grid_n: int = 11, min_needed: int = 120):
    if len(proposals) >= min_needed:
        return proposals
    H, W = gray_u8.shape
    for y in np.linspace(H * 0.08, H * 0.92, grid_n).astype(int):
        for x in np.linspace(W * 0.08, W * 0.92, grid_n).astype(int):
            proposals.append(
                Proposal(
                    box=[0, 0, W - 1, H - 1],
                    pos_points=[[int(x), int(y)]],
                    module="fallback_grid",
                    note="auto",
                    weight=0.85,
                    target="generic",
                )
            )
    return proposals


def group_cands_by_target(cands: List[Cand]) -> Dict[str, List[Cand]]:
    out: Dict[str, List[Cand]] = {}
    for c in cands:
        t = getattr(c.proposal, "target", None) or "generic"
        out.setdefault(t, []).append(c)
    return out


def selection_for_target(plan: Dict[str, Any], target_name: str) -> Dict[str, Any]:
    for t in plan.get("targets", []):
        if t.get("name") == target_name:
            return t.get("selection", {}) or {}
    return plan.get("selection", {}) or {}


def select_per_target(cands_by_target: Dict[str, List[Cand]], plan: Dict[str, Any]) -> Dict[str, List[Cand]]:
    selected_by_target: Dict[str, List[Cand]] = {}
    for tname, cands in cands_by_target.items():
        sel = selection_for_target(plan, tname)
        min_q = float(sel.get("min_quality", 0.30))
        iou_t = float(sel.get("iou_thresh", 0.22))
        max_i = int(sel.get("max_instances", 300))
        selected_by_target[tname] = select_diverse_covering(
            cands,
            max_instances=max_i,
            min_quality=min_q,
            iou_thresh=iou_t,
            min_area_frac=0.00006,
            max_area_frac=0.30,
            coverage_boost=0.55,
        )
    return selected_by_target


def select_per_target_adaptive(
    cands_by_target: Dict[str, List[Cand]], plan: Dict[str, Any], relax_steps: int
) -> Dict[str, List[Cand]]:
    selected_by_target: Dict[str, List[Cand]] = {}
    for tname, cands in cands_by_target.items():
        sel = selection_for_target(plan, tname)
        min_q = float(sel.get("min_quality", 0.30))
        iou_t = float(sel.get("iou_thresh", 0.22))
        max_i = int(sel.get("max_instances", 300))
        scores = [c.total for c in cands]
        if scores:
            p60 = float(np.percentile(scores, 60))
            min_q = max(min_q, min(p60, 0.8))
        mq, it = min_q, iou_t
        selected: List[Cand] = []
        for _ in range(relax_steps + 1):
            selected = select_diverse_covering(
                cands,
                max_instances=max_i,
                min_quality=mq,
                iou_thresh=it,
                min_area_frac=0.00006,
                max_area_frac=0.30,
                coverage_boost=0.55,
            )
            if selected:
                break
            mq = max(0.15, mq - 0.08)
            it = max(0.10, it - 0.08)
        selected_by_target[tname] = selected
    return selected_by_target


def union_by_target(selected_by_target: Dict[str, List[Cand]], gray_u8: np.ndarray) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for tname, selected in selected_by_target.items():
        masks = [c.mask_u8 for c in selected]
        out[tname] = union_mask(masks) if masks else np.zeros_like(gray_u8, dtype=np.uint8)
    return out


def summarize_target(selected: List[Cand], union_mask: np.ndarray, sel_cfg: Dict[str, Any]) -> Dict[str, Any]:
    top_scores = [float(c.total) for c in selected[:10]]
    edge_vals = [float(c.critic.get("edge", 0.0)) for c in selected]
    sep_vals = [float(c.critic.get("sep", 0.0)) for c in selected]
    quality_vals = [float(c.total) for c in selected]
    return {
        "num_instances_selected": len(selected),
        "selection_used": {
            "min_quality": float(sel_cfg.get("min_quality", 0.30)),
            "iou_thresh": float(sel_cfg.get("iou_thresh", 0.22)),
            "max_instances": int(sel_cfg.get("max_instances", 300)),
        },
        "top_scores": top_scores,
        "quality_mean": float(np.mean(quality_vals)) if quality_vals else 0.0,
        "quality_median": float(np.median(quality_vals)) if quality_vals else 0.0,
        "edge_mean": float(np.mean(edge_vals)) if edge_vals else 0.0,
        "sep_mean": float(np.mean(sep_vals)) if sep_vals else 0.0,
        "union_area_frac": float((union_mask > 0).sum() / (union_mask.size + 1e-6)),
        "largest_component_frac": largest_component_fraction(union_mask),
    }


def summarize_all_targets(
    selected_by_target: Dict[str, List[Cand]],
    unions_by_target: Dict[str, np.ndarray],
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for tname, selected in selected_by_target.items():
        sel_cfg = selection_for_target(plan, tname)
        out[tname] = summarize_target(selected, unions_by_target[tname], sel_cfg)
    return out


def flatten_selected(selected_by_target: Dict[str, List[Cand]]) -> List[Cand]:
    return [c for lst in selected_by_target.values() for c in lst]


def default_action_for_target(classical_summary: Dict[str, Any], sam_summary: Dict[str, Any], target: str) -> str:
    c = (classical_summary.get("target_summaries") or {}).get(target, {})
    s = (sam_summary.get("target_summaries") or {}).get(target, {})
    c_q = float(c.get("quality_median", 0.0))
    s_q = float(s.get("quality_median", 0.0))
    c_cov = float(c.get("union_area_frac", 0.0))
    s_cov = float(s.get("union_area_frac", 0.0))
    if s and (s_q > c_q + 0.05 or s_cov > c_cov * 1.2):
        return "accept_sam"
    return "accept_classical"


def masks_iou(a_u8: np.ndarray, b_u8: np.ndarray, eps: float = 1e-6) -> float:
    a = a_u8 > 0
    b = b_u8 > 0
    inter = (a & b).sum()
    uni = (a | b).sum()
    return float(inter / (uni + eps))


def proposals_from_masks(masks_by_target: Dict[str, List[np.ndarray]]) -> List[Proposal]:
    proposals: List[Proposal] = []
    for tname, masks in masks_by_target.items():
        for m in masks:
            ys, xs = np.where(m > 0)
            if len(xs) == 0:
                continue
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            cx = int(xs.mean())
            cy = int(ys.mean())
            proposals.append(
                Proposal(
                    box=[x1, y1, x2, y2],
                    pos_points=[[cx, cy]],
                    module="classical_seed",
                    note="seed from classical mask",
                    weight=1.10,
                    target=tname,
                )
            )
    return proposals


def apply_reference_priors(
    plan: Dict[str, Any],
    reference_image_u8: Optional[np.ndarray],
    reference_mask_u8: Optional[np.ndarray],
    reference_target: Optional[str],
) -> Dict[str, Any]:
    if reference_image_u8 is None or reference_mask_u8 is None:
        return plan
    g = to_uint8(to_gray(reference_image_u8))
    m = reference_mask_u8 > 0
    if m.sum() < 10 or (~m).sum() < 10:
        return plan
    inside = g[m].astype(np.float32)
    outside = g[~m].astype(np.float32)
    if np.median(inside) < np.median(outside):
        polarity = "dark"
    else:
        polarity = "bright"

    # Estimate typical area from connected components
    import skimage.measure as measure

    lbl = measure.label(m, connectivity=2)
    areas = [r.area for r in measure.regionprops(lbl)]
    if areas:
        med_area = float(np.median(areas))
    else:
        med_area = 60.0
    min_size = max(10, int(med_area * 0.5))
    radius = max(2.0, (med_area / np.pi) ** 0.5)
    min_sigma = max(1.0, radius / 4.0)
    max_sigma = max(min_sigma * 2.0, radius * 2.0)

    targets = plan.get("targets", [])
    if not targets:
        return plan
    apply_all = reference_target is None and len(targets) == 1

    for t in targets:
        if not apply_all and reference_target and t.get("name") != reference_target:
            continue
        t["polarity"] = polarity
        for mdef in t.get("suggest_modules", []):
            params = mdef.setdefault("params", {})
            if "min_size" in params:
                params["min_size"] = min_size
            if mdef.get("module") == "blob_log":
                params["min_sigma"] = float(min_sigma)
                params["max_sigma"] = float(max_sigma)
    return plan


def run_classical_stage(gray_u8: np.ndarray, plan: Dict[str, Any]) -> List[Cand]:
    masks_props = classical_masks_from_plan(gray_u8, plan)
    cands: List[Cand] = []
    for mask_u8, prop in masks_props:
        cleaned, crit, total = cleanup_and_score_classical(gray_u8, mask_u8, module_weight=float(prop.weight))
        cands.append(Cand(mask_u8=cleaned, proposal=prop, sam_score=0.0, critic=crit, total=total))
    return cands


def classical_good_enough(selected: List[Cand], union_mask: np.ndarray, settings: RunSettings) -> bool:
    if not selected:
        return False
    area_frac = float((union_mask > 0).sum() / (union_mask.size + 1e-6))
    if area_frac < settings.classical_union_min or area_frac > settings.classical_union_max:
        return False
    if len(selected) < settings.classical_min_instances:
        return False
    best = selected[0].total if selected else 0.0
    if best < settings.classical_min_quality_stop:
        return False
    return True


def largest_component_fraction(mask: np.ndarray) -> float:
    import skimage.measure as measure

    m = mask > 0
    lbl = measure.label(m, connectivity=2)
    if lbl.max() == 0:
        return 0.0
    H, W = m.shape
    areas = np.bincount(lbl.ravel())[1:]
    return float(areas.max() / (H * W + 1e-6))


def edge_alignment_mean(selected: List[Cand]) -> float:
    vals = []
    for c in selected:
        if "edge" in c.critic:
            vals.append(float(c.critic["edge"]))
    if not vals:
        return 0.0
    return float(np.mean(vals))


def human_needed(summary: Dict[str, Any], union_mask: np.ndarray, selected: List[Cand], settings: RunSettings) -> bool:
    cov = summary.get("union_area_frac", 0.0)
    top_scores = summary.get("top_scores", [])
    q_top = top_scores[0] if top_scores else 0.0
    q_med = float(np.median(top_scores)) if top_scores else 0.0
    inst = summary.get("num_instances_selected", 0)
    largest = largest_component_fraction(union_mask)
    edge_mean = edge_alignment_mean(selected)
    gain = summary.get("coverage_gain", 0.0)
    if cov < settings.human_min_coverage or cov > settings.human_max_coverage:
        return True
    if inst < settings.human_min_instances:
        return True
    if q_top < settings.human_min_quality and q_med < settings.human_min_quality:
        return True
    if largest > settings.human_max_component_frac:
        return True
    if edge_mean < settings.human_edge_thresh:
        return True
    if gain < settings.human_gain_thresh and q_top < (settings.human_min_quality + 0.1):
        return True
    return False


def box_iou(a, b, eps=1e-6):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1 + 1)
    ih = max(0, inter_y2 - inter_y1 + 1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
    uni = area_a + area_b - inter + eps
    return inter / uni


def dedup_and_cap_proposals(proposals: List[Proposal], cap: int, iou_thresh: float = 0.8) -> List[Proposal]:
    out: List[Proposal] = []
    for p in proposals:
        if p.box is None:
            out.append(p)
            continue
        keep = True
        for q in out:
            if q.box is None:
                continue
            if box_iou(p.box, q.box) > iou_thresh:
                keep = False
                break
        if keep:
            out.append(p)
        if len(out) >= cap:
            break
    return out


def run_sam_on_proposals(
    sam, image_rgb_u8, gray_u8, proposals: List[Proposal], multimask_output=True, cap_masks_total=3500
) -> List[Cand]:
    sam.set_image(image_rgb_u8)
    cands: List[Cand] = []
    for p in proposals:
        masks, scores = sam.predict(box=p.box, pos_points=p.pos_points, neg_points=p.neg_points, multimask_output=multimask_output)
        for m, s in zip(masks, scores):
            m_u8 = mask_to_uint8(m)
            cleaned, crit, total = cleanup_and_score(gray_u8, m_u8, sam_score=float(s), module_weight=float(p.weight))
            cands.append(Cand(mask_u8=cleaned, proposal=p, sam_score=float(s), critic=crit, total=total))
            if len(cands) >= cap_masks_total:
                return cands
    return cands


def run_sam_round(
    sam,
    gray_u8: np.ndarray,
    rgb_u8: np.ndarray,
    plan: Dict[str, Any],
    hints: HumanHints,
    settings: RunSettings,
    round_idx: int,
    fast: bool,
    seed_masks_by_target: Optional[Dict[str, List[np.ndarray]]] = None,
):
    proposals = run_modules_parallel(gray_u8, plan)
    if seed_masks_by_target:
        proposals.extend(proposals_from_masks(seed_masks_by_target))
    proposals.extend(hints_to_proposals(hints, rgb_u8.shape))
    prop_cap = settings.proposal_cap_fast if fast else settings.proposal_cap
    proposals = dedup_and_cap_proposals(proposals, cap=prop_cap, iou_thresh=0.8)
    grid_n = 6 if fast else 11
    min_needed = 60 if fast else 120
    proposals = add_fallback_grid(proposals, gray_u8, grid_n=grid_n, min_needed=min_needed)
    proposals = dedup_and_cap_proposals(proposals, cap=prop_cap, iou_thresh=0.8)

    cap_masks = settings.cap_masks_total
    if fast:
        cap_masks = min(cap_masks, settings.cap_masks_fast)
    if settings.runtime_budget:
        budget_cap = int(settings.runtime_budget * 40)
        if budget_cap > 0:
            cap_masks = min(cap_masks, budget_cap)

    cands = run_sam_on_proposals(
        sam=sam,
        image_rgb_u8=rgb_u8,
        gray_u8=gray_u8,
        proposals=proposals,
        multimask_output=settings.multimask_output if not fast else False,
        cap_masks_total=cap_masks,
    )
    cands.sort(key=lambda c: c.total, reverse=True)

    cands_by_target = group_cands_by_target(cands)
    selected_by_target = select_per_target_adaptive(cands_by_target, plan, settings.relax_steps)
    selected = [c for lst in selected_by_target.values() for c in lst]

    unions_by_target = union_by_target(selected_by_target, gray_u8)
    union = union_mask(list(unions_by_target.values())) if unions_by_target else np.zeros_like(gray_u8, dtype=np.uint8)
    instance_masks = [c.mask_u8 for c in selected[:80]]

    overlay_union = draw_boundary(overlay_mask_rgb(rgb_u8, union, alpha=0.35), union)
    overlay_inst = overlay_instances(rgb_u8, instance_masks[:35], alpha=0.30, seed=round_idx + 1)

    summary = {
        "stage": "sam_light" if fast else "sam",
        "round": round_idx,
        "num_proposals": len(proposals),
        "num_candidate_masks": len(cands),
        "num_instances_selected": len(selected),
        "selection_used": plan.get("selection", {}),
        "top_scores": [float(c.total) for c in selected[:10]],
        "union_area_frac": float((union > 0).sum() / (union.size + 1e-6)),
        "target_summaries": summarize_all_targets(selected_by_target, unions_by_target, plan),
    }

    return {
        "proposals": proposals,
        "cands": cands,
        "selected_by_target": selected_by_target,
        "selected": selected,
        "unions_by_target": unions_by_target,
        "union": union,
        "overlay_union": overlay_union,
        "overlay_instances": overlay_inst,
        "summary": summary,
    }


def run_segmentation(
    image_rgb,
    sam_paths,
    settings: RunSettings,
    hints: Optional[HumanHints] = None,
    save_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    reference_image_u8: Optional[np.ndarray] = None,
    reference_mask_u8: Optional[np.ndarray] = None,
):
    gray_u8 = preprocess_micrograph(image_rgb, clahe=True, denoise=True)
    rgb_u8 = ensure_rgb_uint8(gray_u8)

    sam = SAM2Wrapper(sam2_root=sam_paths.sam2_root, ckpt_path=sam_paths.checkpoint, config_path=sam_paths.config)

    llm_plan_raw = None
    if settings.use_llm:
        plan0 = plan_via_langgraph(
            rgb_u8,
            model_name=settings.llm_model,
            temperature=settings.temperature,
            user_prompt=settings.user_prompt,
            reference_image_u8=reference_image_u8,
            reference_mask_u8=reference_mask_u8,
        )
        llm_plan_raw = plan0
    else:
        plan0 = fallback_plan()
    plan = sanitize_plan(plan0)
    if plan.get("selection") is None:
        plan["selection"] = {}
    # Apply user overrides for selection thresholds if provided.
    if settings.min_quality_override is not None:
        plan["selection"]["min_quality"] = float(settings.min_quality_override)
    if settings.iou_thresh_override is not None:
        plan["selection"]["iou_thresh"] = float(settings.iou_thresh_override)
    if settings.max_instances_override is not None:
        plan["selection"]["max_instances"] = int(settings.max_instances_override)
    plan = apply_target_priors(plan, settings.target_priors)
    plan = apply_prompt_priors(plan, settings.user_prompt or "")
    plan = apply_reference_priors(plan, reference_image_u8, reference_mask_u8, settings.reference_target)

    history: List[Dict[str, Any]] = []
    final_selected: List[Cand] = []

    hints = hints or HumanHints()

    # Stage 1: Classical-only masks
    classical_cands = run_classical_stage(gray_u8, plan)
    classical_cands.sort(key=lambda c: c.total, reverse=True)
    classical_by_target = group_cands_by_target(classical_cands)
    classical_selected_by_target = select_per_target(classical_by_target, plan)
    classical_selected = [c for lst in classical_selected_by_target.values() for c in lst]

    classical_unions_by_target = union_by_target(classical_selected_by_target, gray_u8)
    classical_union = union_mask(list(classical_unions_by_target.values())) if classical_unions_by_target else np.zeros_like(gray_u8, dtype=np.uint8)

    classical_masks = [c.mask_u8 for c in classical_selected[:80]]
    classical_overlay_union = draw_boundary(overlay_mask_rgb(rgb_u8, classical_union, alpha=0.35), classical_union)
    classical_overlay_inst = overlay_instances(rgb_u8, classical_masks[:35], alpha=0.30, seed=0)

    classical_summary = {
        "stage": "classical",
        "round": -1,
        "num_proposals": len(classical_cands),
        "num_candidate_masks": len(classical_cands),
        "num_instances_selected": len(classical_selected),
        "target_summaries": summarize_all_targets(classical_selected_by_target, classical_unions_by_target, plan),
        "union_area_frac": float((classical_union > 0).sum() / (classical_union.size + 1e-6)),
    }
    classical_summary["top_scores"] = [float(c.total) for c in classical_selected[:10]]
    classical_summary["selection_used"] = plan.get("selection", {})

    classical_summary["human_needed"] = False
    classical_summary["coverage_gain"] = classical_summary["union_area_frac"]

    if settings.use_human_loop and human_needed(classical_summary, classical_union, classical_selected, settings):
        classical_summary["human_needed"] = True

    history.append(
        {
            "plan": plan,
            "summary": classical_summary,
            "overlay_union": classical_overlay_union,
            "overlay_instances": classical_overlay_inst,
            "selected": classical_selected,
            "llm_plan_raw": llm_plan_raw,
        }
    )

    if not settings.use_reviewer:
        if classical_good_enough(classical_selected, classical_union, settings):
            final_selected = classical_selected
            return finalize_and_save(
                gray_u8,
                rgb_u8,
                plan,
                history,
        classical_selected,
        classical_union,
        sam_paths,
        save_dir,
        run_id,
        hints,
        settings,
        reference_image_u8=reference_image_u8,
        reference_mask_u8=reference_mask_u8,
    )
        if classical_summary["human_needed"] and settings.use_human_loop and settings.auto_pause_for_human:
            return finalize_and_save(
                gray_u8,
                rgb_u8,
                plan,
                history,
        classical_selected,
        classical_union,
        sam_paths,
        save_dir,
        run_id,
        hints,
        settings,
        reference_image_u8=reference_image_u8,
        reference_mask_u8=reference_mask_u8,
    )
        # Stage 2+: SAM-based rounds
        rounds_eff = 1 if settings.fast_mode else settings.rounds
        prev_union_frac = classical_summary["union_area_frac"]
        for r in range(rounds_eff):
            sam_round = run_sam_round(sam, gray_u8, rgb_u8, plan, hints, settings, r, fast=settings.fast_mode)
            summary = sam_round["summary"]
            summary["coverage_gain"] = summary["union_area_frac"] - prev_union_frac
            summary["human_needed"] = False
            if settings.use_human_loop and human_needed(summary, sam_round["union"], sam_round["selected"], settings):
                summary["human_needed"] = True

            history.append(
                {
                    "plan": plan,
                    "summary": summary,
                    "overlay_union": sam_round["overlay_union"],
                    "overlay_instances": sam_round["overlay_instances"],
                    "selected": sam_round["selected"],
                }
            )
            final_selected = sam_round["selected"]

            if r < rounds_eff - 1 and settings.use_llm:
                adj = adjust_via_langgraph(
                    rgb_u8,
                    sam_round["overlay_union"],
                    sam_round["overlay_instances"],
                    summary,
                    model_name=settings.llm_model,
                    temperature=settings.temperature,
                )
                history[-1]["llm_adjustments_raw"] = adj
                plan = apply_adjustments(plan, adj)

            prev_union_frac = summary["union_area_frac"]
            if summary["coverage_gain"] < 0.01:
                break
            if summary["human_needed"] and settings.use_human_loop and settings.auto_pause_for_human:
                break
    else:
        # Reviewer flow: compare classical vs lightweight SAM
        classical_seed_masks = {k: [c.mask_u8 for c in v[:25]] for k, v in classical_selected_by_target.items()}
        sam_light = run_sam_round(
            sam,
            gray_u8,
            rgb_u8,
            plan,
            hints,
            settings,
            0,
            fast=True,
            seed_masks_by_target=classical_seed_masks,
        )
        sam_light_summary = sam_light["summary"]
        sam_light_summary["coverage_gain"] = sam_light_summary["union_area_frac"] - classical_summary["union_area_frac"]
        sam_light_summary["human_needed"] = False
        if settings.use_human_loop and human_needed(sam_light_summary, sam_light["union"], sam_light["selected"], settings):
            sam_light_summary["human_needed"] = True

        review_payload = {"classical": classical_summary, "sam": sam_light_summary, "user_prompt": settings.user_prompt}
        review_payload["agreement_iou"] = masks_iou(classical_union, sam_light["union"])
        reviewer_output = review_via_langgraph(
            rgb_u8,
            classical_overlay_union,
            sam_light["overlay_union"],
            review_payload,
            model_name=settings.reviewer_model,
            temperature=settings.reviewer_temperature,
            reference_image_u8=reference_image_u8,
            reference_mask_u8=reference_mask_u8,
        )
        decisions = reviewer_output.get("decisions", [])
        if not decisions:
            reviewer_output = fallback_reviewer_decision(classical_summary, sam_light_summary)
            decisions = reviewer_output.get("decisions", [])
        sam_light_summary["reviewer_decisions"] = reviewer_output
        if any(d.get("action") == "ask_human" for d in decisions if isinstance(d, dict)):
            sam_light_summary["human_needed"] = True

        history.append(
            {
                "plan": plan,
                "summary": sam_light_summary,
                "overlay_union": sam_light["overlay_union"],
                "overlay_instances": sam_light["overlay_instances"],
                "selected": sam_light["selected"],
            }
        )

        decisions_map = {d.get("target"): d for d in decisions if isinstance(d, dict) and d.get("target")}
        all_targets = set(classical_selected_by_target.keys()) | set(sam_light["selected_by_target"].keys())

        # Apply reviewer updates to plan
        for d in decisions:
            updates = d.get("param_updates") or {}
            tname = d.get("target")
            plan = apply_reviewer_updates(plan, updates, tname)

        # Optional rerun of classical if requested
        if any(d.get("action") == "rerun_classical" for d in decisions):
            rerun_cands = run_classical_stage(gray_u8, plan)
            rerun_cands.sort(key=lambda c: c.total, reverse=True)
            classical_by_target = group_cands_by_target(rerun_cands)
            classical_selected_by_target = select_per_target(classical_by_target, plan)
            classical_selected = flatten_selected(classical_selected_by_target)
            classical_unions_by_target = union_by_target(classical_selected_by_target, gray_u8)
            classical_union = union_mask(list(classical_unions_by_target.values())) if classical_unions_by_target else np.zeros_like(gray_u8, dtype=np.uint8)
            classical_overlay_union = draw_boundary(overlay_mask_rgb(rgb_u8, classical_union, alpha=0.35), classical_union)
            classical_overlay_inst = overlay_instances(rgb_u8, [c.mask_u8 for c in classical_selected[:35]], alpha=0.30, seed=1)
            classical_summary = {
                "stage": "classical_rerun",
                "round": -1,
                "num_proposals": len(rerun_cands),
                "num_candidate_masks": len(rerun_cands),
                "num_instances_selected": len(classical_selected),
                "target_summaries": summarize_all_targets(classical_selected_by_target, classical_unions_by_target, plan),
                "union_area_frac": float((classical_union > 0).sum() / (classical_union.size + 1e-6)),
                "top_scores": [float(c.total) for c in classical_selected[:10]],
                "selection_used": plan.get("selection", {}),
            }
            history.append(
                {
                    "plan": plan,
                    "summary": classical_summary,
                    "overlay_union": classical_overlay_union,
                    "overlay_instances": classical_overlay_inst,
                    "selected": classical_selected,
                }
            )

        final_selected_by_target: Dict[str, List[Cand]] = {}
        refine_targets = set()
        ask_human_targets = set()
        for tname in all_targets:
            decision = decisions_map.get(tname, {})
            action = decision.get("action") or default_action_for_target(classical_summary, sam_light_summary, tname)
            if action == "accept_classical":
                final_selected_by_target[tname] = classical_selected_by_target.get(tname, [])
            elif action == "accept_sam":
                final_selected_by_target[tname] = sam_light["selected_by_target"].get(tname, [])
            elif action == "rerun_classical":
                final_selected_by_target[tname] = classical_selected_by_target.get(tname, [])
            elif action == "refine_sam":
                refine_targets.add(tname)
            elif action == "ask_human":
                ask_human_targets.add(tname)

        if ask_human_targets and settings.use_human_loop and settings.auto_pause_for_human:
            final_selected = flatten_selected(final_selected_by_target)
        else:
            # Refine SAM for requested targets
            if refine_targets:
                rounds_eff = settings.rounds
                prev_union_frac = sam_light_summary["union_area_frac"]
                for r in range(rounds_eff):
                    sam_round = run_sam_round(sam, gray_u8, rgb_u8, plan, hints, settings, r, fast=False)
                    summary = sam_round["summary"]
                    summary["coverage_gain"] = summary["union_area_frac"] - prev_union_frac
                    summary["human_needed"] = False
                    if settings.use_human_loop and human_needed(summary, sam_round["union"], sam_round["selected"], settings):
                        summary["human_needed"] = True

                    history.append(
                        {
                            "plan": plan,
                            "summary": summary,
                            "overlay_union": sam_round["overlay_union"],
                            "overlay_instances": sam_round["overlay_instances"],
                            "selected": sam_round["selected"],
                        }
                    )

                    for tname in refine_targets:
                        final_selected_by_target[tname] = sam_round["selected_by_target"].get(tname, [])

                    if r < rounds_eff - 1 and settings.use_llm:
                        adj = adjust_via_langgraph(
                            rgb_u8,
                            sam_round["overlay_union"],
                            sam_round["overlay_instances"],
                            summary,
                            model_name=settings.llm_model,
                            temperature=settings.temperature,
                        )
                        history[-1]["llm_adjustments_raw"] = adj
                        plan = apply_adjustments(plan, adj)

                    prev_union_frac = summary["union_area_frac"]
                    if summary["coverage_gain"] < 0.01:
                        break
                    if summary["human_needed"] and settings.use_human_loop and settings.auto_pause_for_human:
                        break

            # Fill any remaining targets from sam_light or classical
            for tname in all_targets:
                if tname in final_selected_by_target:
                    continue
                final_selected_by_target[tname] = sam_light["selected_by_target"].get(tname, [])

            final_selected = flatten_selected(final_selected_by_target)

    best_union = union_mask([c.mask_u8 for c in final_selected]) if final_selected else np.zeros_like(gray_u8, dtype=np.uint8)
    return finalize_and_save(
        gray_u8,
        rgb_u8,
        plan,
        history,
        final_selected,
        best_union,
        sam_paths,
        save_dir,
        run_id,
        hints,
        settings,
        reference_image_u8=reference_image_u8,
        reference_mask_u8=reference_mask_u8,
    )


def finalize_and_save(
    gray_u8,
    rgb_u8,
    plan,
    history,
    final_selected,
    union_mask,
    sam_paths,
    save_dir,
    run_id,
    hints,
    settings: RunSettings,
    reference_image_u8: Optional[np.ndarray] = None,
    reference_mask_u8: Optional[np.ndarray] = None,
):
    best_union = union_mask
    best_overlay = draw_boundary(overlay_mask_rgb(rgb_u8, best_union, alpha=0.35), best_union)

    result = {
        "gray_u8": gray_u8,
        "rgb_u8": rgb_u8,
        "plan": plan,
        "history": history,
        "selected": final_selected,
        "union_mask_u8": best_union,
        "overlay_union_rgb_u8": best_overlay,
        "user_prompt": settings.user_prompt,
        "reference_image_u8": reference_image_u8,
        "reference_mask_u8": reference_mask_u8,
    }

    if save_dir is not None:
        rid = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")

        def _save():
            try:
                save_run(rid, result, save_dir)
            except Exception:
                pass

        threading.Thread(target=_save, daemon=True).start()

    return result
