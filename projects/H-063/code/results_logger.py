import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image


def _save_png(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def save_run(run_id: str, result: Dict, base_dir: Path):
    """
    Persist run artifacts: plan, history, overlays, masks.
    """
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # JSON artifacts
    (run_dir / "plan.json").write_text(json.dumps(result.get("plan", {}), indent=2))
    if result.get("user_prompt"):
        (run_dir / "user_prompt.txt").write_text(str(result["user_prompt"]))
    histories = []
    for h in result.get("history", []):
        hist_copy = {
            "summary": h.get("summary", {}),
            "plan": h.get("plan", {}),
            "llm_adjustments_raw": h.get("llm_adjustments_raw"),
            "llm_plan_raw": h.get("llm_plan_raw"),
        }
        histories.append(hist_copy)
    (run_dir / "history.json").write_text(json.dumps(histories, indent=2))

    # Masks and overlays
    if result.get("reference_image_u8") is not None:
        _save_png(run_dir / "reference_image.png", result["reference_image_u8"])
    if result.get("reference_mask_u8") is not None:
        _save_png(run_dir / "reference_mask.png", result["reference_mask_u8"])
    if result.get("union_mask_u8") is not None:
        _save_png(run_dir / "union_mask.png", result["union_mask_u8"])
    if result.get("overlay_union_rgb_u8") is not None:
        _save_png(run_dir / "overlay_union.png", result["overlay_union_rgb_u8"])

    # Per-round overlays
    for h in result.get("history", []):
        r = h["summary"]["round"]
        if h.get("overlay_union") is not None:
            _save_png(run_dir / f"round_{r}_overlay_union.png", h["overlay_union"])
        if h.get("overlay_instances") is not None:
            _save_png(run_dir / f"round_{r}_overlay_instances.png", h["overlay_instances"])

    # Instance masks as individual PNGs
    for idx, c in enumerate(result.get("selected", [])):
        if hasattr(c, "mask_u8"):
            _save_png(run_dir / f"instance_{idx:04d}.png", c.mask_u8)

    return run_dir
