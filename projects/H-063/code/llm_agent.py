import json
import os
from typing import Any, Dict, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .modules import MODULE_SPECS, default_targets, sanitize_plan
from .preprocess import encode_png_uint8


def fallback_plan() -> Dict[str, Any]:
    return {"targets": default_targets(), "selection": {"min_quality": 0.30, "iou_thresh": 0.22, "max_instances": 300}}


def img_part_from_b64(b64: str) -> Dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def _make_json_llm(model_name: str, temperature: float):
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def _content_to_str(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                out.append(part.get("text", ""))
            else:
                out.append(str(part))
        return "".join(out)
    return str(content)


def _safe_json_loads(text: str, default: Any) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return default


class PlanState(TypedDict, total=False):
    image_b64: str
    schema_hint: Dict[str, Any]
    plan: Dict[str, Any]
    user_prompt: str
    reference_b64: str
    reference_mask_b64: str


class AdjustState(TypedDict, total=False):
    image_b64: str
    overlay_union_b64: str
    overlay_inst_b64: str
    summary: Dict[str, Any]
    adjustments: Dict[str, Any]


class ReviewState(TypedDict, total=False):
    image_b64: str
    classical_b64: str
    sam_b64: str
    summary: Dict[str, Any]
    decisions: Dict[str, Any]
    reference_b64: str
    reference_mask_b64: str


def _build_plan_graph(model_name: str, temperature: float):
    graph = StateGraph(PlanState)

    def plan_node(state: PlanState):
        llm = _make_json_llm(model_name, temperature)
        schema_hint = state.get("schema_hint") or fallback_plan()
        b64 = state["image_b64"]
        user_prompt = (
            state.get("user_prompt")
            or "Plan robust microscopy segmentation. Prioritize user intent and identify multiple targets if present. Use the schema example."
        )
        content_parts = [
            {"type": "text", "text": user_prompt},
            {"type": "text", "text": json.dumps(schema_hint)},
            img_part_from_b64(b64),
        ]
        if state.get("reference_b64"):
            content_parts.append({"type": "text", "text": "Reference example image:"})
            content_parts.append(img_part_from_b64(state["reference_b64"]))
        if state.get("reference_mask_b64"):
            content_parts.append({"type": "text", "text": "Reference mask (if provided):"})
            content_parts.append(img_part_from_b64(state["reference_mask_b64"]))
        messages = [
            SystemMessage(
                content=(
                    "Return ONLY JSON matching the example schema. "
                    "If the image appears to have multiple structure types, create multiple targets. "
                    "Choose modules ONLY from: threshold_regions, adaptive_threshold, watershed, blob_log, edge_contours. "
                    "Use ONLY allowed params in the schema example. Do NOT invent module or param names."
                )
            ),
            HumanMessage(
                content=content_parts
            ),
        ]
        resp = llm.invoke(messages)
        data = _safe_json_loads(_content_to_str(resp.content), fallback_plan())
        return {"plan": data}

    graph.add_node("plan", plan_node)
    graph.set_entry_point("plan")
    graph.add_edge("plan", END)
    return graph.compile()


def _build_adjust_graph(model_name: str, temperature: float):
    graph = StateGraph(AdjustState)

    def adjust_node(state: AdjustState):
        llm = _make_json_llm(model_name, temperature)
        messages = [
            SystemMessage(
                content=(
                    "You are a segmentation reviewer. Return ONLY JSON with keys: "
                    "action ('stop'|'continue'), adjustments. "
                    "adjustments: {selection_updates, module_param_updates, notes}. "
                    "selection_updates may include min_quality, iou_thresh, max_instances. "
                    "module_param_updates: list of {target_name, module, params_patch}. "
                    "Allowed modules: threshold_regions, adaptive_threshold, watershed, blob_log, edge_contours. "
                    "Allowed params are the same as in the schema example."
                )
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": json.dumps(state.get("summary", {}))},
                    img_part_from_b64(state["image_b64"]),
                    img_part_from_b64(state["overlay_union_b64"]),
                    img_part_from_b64(state["overlay_inst_b64"]),
                ]
            ),
        ]
        resp = llm.invoke(messages)
        data = _safe_json_loads(_content_to_str(resp.content), {"action": "stop", "adjustments": {}})
        return {"adjustments": data}

    graph.add_node("adjust", adjust_node)
    graph.set_entry_point("adjust")
    graph.add_edge("adjust", END)
    return graph.compile()


def _build_review_graph(model_name: str, temperature: float):
    graph = StateGraph(ReviewState)

    def review_node(state: ReviewState):
        llm = _make_json_llm(model_name, temperature)
        content_parts = [
            {"type": "text", "text": json.dumps(state.get("summary", {}))},
            img_part_from_b64(state["image_b64"]),
            img_part_from_b64(state["classical_b64"]),
            img_part_from_b64(state["sam_b64"]),
        ]
        if state.get("reference_b64"):
            content_parts.append({"type": "text", "text": "Reference example image:"})
            content_parts.append(img_part_from_b64(state["reference_b64"]))
        if state.get("reference_mask_b64"):
            content_parts.append({"type": "text", "text": "Reference mask (if provided):"})
            content_parts.append(img_part_from_b64(state["reference_mask_b64"]))
        messages = [
            SystemMessage(
                content=(
                    "You are a segmentation reviewer comparing classical vs SAM outputs. "
                    "Return ONLY JSON with key: decisions. "
                    "decisions is a list of {target, action, param_updates}. "
                    "action must be one of: accept_classical, accept_sam, refine_sam, rerun_classical, ask_human. "
                    "param_updates is optional and may include module_param_updates and selection_updates."
                )
            ),
            HumanMessage(
                content=content_parts
            ),
        ]
        resp = llm.invoke(messages)
        data = _safe_json_loads(_content_to_str(resp.content), {"decisions": []})
        return {"decisions": data}

    graph.add_node("review", review_node)
    graph.set_entry_point("review")
    graph.add_edge("review", END)
    return graph.compile()


def plan_via_langgraph(
    image_rgb_u8,
    model_name: str = "gpt-4o",
    temperature: float = 0.2,
    user_prompt: Optional[str] = None,
    reference_image_u8: Optional[Any] = None,
    reference_mask_u8: Optional[Any] = None,
) -> Dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        return fallback_plan()
    plan_graph = _build_plan_graph(model_name, temperature)
    b64 = encode_png_uint8(image_rgb_u8)
    prompt_text = "Plan robust microscopy segmentation. Prioritize user intent and identify multiple targets if present. Use the schema example."
    if user_prompt:
        prompt_text += f" User intent: {user_prompt}"
    payload = {"image_b64": b64, "schema_hint": fallback_plan(), "user_prompt": prompt_text}
    if reference_image_u8 is not None:
        payload["reference_b64"] = encode_png_uint8(reference_image_u8)
    if reference_mask_u8 is not None:
        payload["reference_mask_b64"] = encode_png_uint8(reference_mask_u8)
    result = plan_graph.invoke(payload)
    return result.get("plan", fallback_plan())


def adjust_via_langgraph(
    image_rgb_u8,
    overlay_union_u8,
    overlay_instances_u8,
    summary: Dict[str, Any],
    model_name: str = "gpt-4o",
    temperature: float = 0.2,
) -> Dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        return {"action": "stop", "adjustments": {}}
    adjust_graph = _build_adjust_graph(model_name, temperature)
    result = adjust_graph.invoke(
        {
            "image_b64": encode_png_uint8(image_rgb_u8),
            "overlay_union_b64": encode_png_uint8(overlay_union_u8),
            "overlay_inst_b64": encode_png_uint8(overlay_instances_u8),
            "summary": summary,
        }
    )
    return result.get("adjustments", {"action": "stop", "adjustments": {}})


def fallback_reviewer_decision(classical_summary: Dict[str, Any], sam_summary: Dict[str, Any]) -> Dict[str, Any]:
    decisions = []
    c_targets = (classical_summary or {}).get("target_summaries", {}) or {}
    s_targets = (sam_summary or {}).get("target_summaries", {}) or {}
    all_targets = set(c_targets.keys()) | set(s_targets.keys())
    for t in all_targets:
        c = c_targets.get(t, {})
        s = s_targets.get(t, {})
        c_q = float(c.get("quality_median", 0.0))
        s_q = float(s.get("quality_median", 0.0))
        c_cov = float(c.get("union_area_frac", 0.0))
        s_cov = float(s.get("union_area_frac", 0.0))
        action = "accept_classical"
        if s and (s_q > c_q + 0.05 or s_cov > c_cov * 1.2):
            action = "accept_sam"
        if max(c_q, s_q) < 0.3 and (s_cov < 0.005 or s_cov > 0.6):
            action = "ask_human"
        decisions.append({"target": t, "action": action})
    return {"decisions": decisions}


def review_via_langgraph(
    image_rgb_u8,
    classical_overlay_u8,
    sam_overlay_u8,
    summary: Dict[str, Any],
    model_name: str = "gpt-4o",
    temperature: float = 0.2,
    reference_image_u8: Optional[Any] = None,
    reference_mask_u8: Optional[Any] = None,
) -> Dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        return fallback_reviewer_decision(summary.get("classical", {}), summary.get("sam", {}))
    review_graph = _build_review_graph(model_name, temperature)
    payload = {
        "image_b64": encode_png_uint8(image_rgb_u8),
        "classical_b64": encode_png_uint8(classical_overlay_u8),
        "sam_b64": encode_png_uint8(sam_overlay_u8),
        "summary": summary,
    }
    if reference_image_u8 is not None:
        payload["reference_b64"] = encode_png_uint8(reference_image_u8)
    if reference_mask_u8 is not None:
        payload["reference_mask_b64"] = encode_png_uint8(reference_mask_u8)
    result = review_graph.invoke(payload)
    decisions = result.get("decisions", {"decisions": []})
    if isinstance(decisions, dict) and "decisions" in decisions:
        return decisions
    return {"decisions": []}


def apply_adjustments(plan: Dict[str, Any], adj: Dict[str, Any]) -> Dict[str, Any]:
    if not adj:
        return plan
    if adj.get("action") == "stop":
        return plan
    updates = (adj.get("adjustments") or {})
    sel_updates = updates.get("selection_updates") or {}
    if sel_updates:
        plan.setdefault("selection", {})
        if "min_quality" in sel_updates:
            plan["selection"]["min_quality"] = float(sel_updates["min_quality"])
        if "iou_thresh" in sel_updates:
            plan["selection"]["iou_thresh"] = float(sel_updates["iou_thresh"])
        if "max_instances" in sel_updates:
            plan["selection"]["max_instances"] = int(sel_updates["max_instances"])

    for upd in updates.get("module_param_updates") or []:
        tname = upd.get("target_name")
        mname = upd.get("module")
        if mname not in MODULE_SPECS:
            continue
        patch = upd.get("params_patch") or {}
        for t in plan.get("targets", []):
            if t.get("name") != tname:
                continue
            for m in t.get("suggest_modules", []):
                if m.get("module") == mname:
                    m.setdefault("params", {})
                    m["params"].update(patch)
    return sanitize_plan(plan)


def apply_reviewer_updates(plan: Dict[str, Any], updates: Dict[str, Any], target_name: Optional[str] = None) -> Dict[str, Any]:
    if not updates:
        return plan
    sel_updates = updates.get("selection_updates") or {}
    if sel_updates:
        if target_name:
            for t in plan.get("targets", []):
                if t.get("name") != target_name:
                    continue
                t.setdefault("selection", {})
                if "min_quality" in sel_updates:
                    t["selection"]["min_quality"] = float(sel_updates["min_quality"])
                if "iou_thresh" in sel_updates:
                    t["selection"]["iou_thresh"] = float(sel_updates["iou_thresh"])
                if "max_instances" in sel_updates:
                    t["selection"]["max_instances"] = int(sel_updates["max_instances"])
        else:
            plan.setdefault("selection", {})
            if "min_quality" in sel_updates:
                plan["selection"]["min_quality"] = float(sel_updates["min_quality"])
            if "iou_thresh" in sel_updates:
                plan["selection"]["iou_thresh"] = float(sel_updates["iou_thresh"])
            if "max_instances" in sel_updates:
                plan["selection"]["max_instances"] = int(sel_updates["max_instances"])

    for upd in updates.get("module_param_updates") or []:
        tname = upd.get("target_name") or target_name
        mname = upd.get("module")
        if mname not in MODULE_SPECS:
            continue
        patch = upd.get("params_patch") or {}
        for t in plan.get("targets", []):
            if tname and t.get("name") != tname:
                continue
            for m in t.get("suggest_modules", []):
                if m.get("module") == mname:
                    m.setdefault("params", {})
                    m["params"].update(patch)
    return sanitize_plan(plan)
