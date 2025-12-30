from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class HumanHints:
    pos_points: List[List[int]] = field(default_factory=list)
    neg_points: List[List[int]] = field(default_factory=list)
    boxes: List[List[int]] = field(default_factory=list)  # [x1,y1,x2,y2]


@dataclass
class RunSettings:
    rounds: int = 2
    relax_steps: int = 3
    cap_masks_total: int = 3500
    multimask_output: bool = True
    use_llm: bool = True
    llm_model: str = "gpt-4o"
    temperature: float = 0.2
    downsample_max: int = 1500  # None to disable
    min_quality_override: Optional[float] = None
    iou_thresh_override: Optional[float] = None
    max_instances_override: Optional[int] = None
    fast_mode: bool = False
    runtime_budget: Optional[float] = None
    proposal_cap: int = 800
    proposal_cap_fast: int = 400
    cap_masks_fast: int = 1200
    user_prompt: Optional[str] = None
    reference_target: Optional[str] = None
    # Reviewer agent
    use_reviewer: bool = False
    reviewer_model: str = "gpt-4o"
    reviewer_temperature: float = 0.2
    target_priors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Human-in-the-loop controls
    use_human_loop: bool = False
    auto_pause_for_human: bool = True
    human_min_coverage: float = 0.005
    human_max_coverage: float = 0.60
    human_min_quality: float = 0.35
    human_min_instances: int = 3
    human_max_component_frac: float = 0.35
    human_edge_thresh: float = 0.25
    human_gain_thresh: float = 0.01
    # Classical early-stop thresholds
    classical_min_quality_stop: float = 0.42
    classical_union_min: float = 0.002
    classical_union_max: float = 0.45
    classical_min_instances: int = 5


@dataclass
class RoundSummary:
    round: int
    num_proposals: int
    num_candidate_masks: int
    num_instances_selected: int
    selection_used: Dict[str, Any]
    top_scores: List[float]
    union_area_frac: float


@dataclass
class HistoryItem:
    plan: Dict[str, Any]
    summary: Dict[str, Any]
    overlay_union: np.ndarray
    overlay_instances: np.ndarray
    selected: List[Any]
    llm_adjustments_raw: Optional[Dict[str, Any]] = None
