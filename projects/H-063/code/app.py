import json
from io import BytesIO
from typing import List

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.config import AppConfig
from src.export import build_zip_export
from src.image_io import load_image_any, maybe_downsample
from src.pipeline import run_segmentation
from src.preprocess import mask_to_uint8
from src.state import HumanHints, RunSettings
from pathlib import Path

# Load .env early to pick up OPENAI_API_KEY and SAM2 paths
load_dotenv()


st.set_page_config(page_title="MicroSeg Lab", layout="wide")

cfg = AppConfig()


def parse_points(text: str) -> List[List[int]]:
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [[int(p[0]), int(p[1])] for p in data if isinstance(p, (list, tuple)) and len(p) == 2]
    except Exception:
        pass
    return []


def parse_boxes(text: str) -> List[List[int]]:
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [[int(v) for v in b] for b in data if isinstance(b, (list, tuple)) and len(b) == 4]
    except Exception:
        pass
    return []


def canvas_points(json_data, scale_x: float, scale_y: float) -> List[List[int]]:
    points: List[List[int]] = []
    if not json_data:
        return points
    for obj in json_data.get("objects", []):
        if obj.get("type") != "circle":
            continue
        radius = obj.get("radius", 0)
        x = obj.get("left", 0) + radius
        y = obj.get("top", 0) + radius
        points.append([int(x * scale_x), int(y * scale_y)])
    return points


def canvas_boxes(json_data, scale_x: float, scale_y: float) -> List[List[int]]:
    boxes: List[List[int]] = []
    if not json_data:
        return boxes
    for obj in json_data.get("objects", []):
        if obj.get("type") != "rect":
            continue
        x1 = obj.get("left", 0)
        y1 = obj.get("top", 0)
        x2 = x1 + obj.get("width", 0)
        y2 = y1 + obj.get("height", 0)
        boxes.append([int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)])
    return boxes


logo_path = Path(__file__).parent / "src/175-1757107_tamu-texas-a-m-university-logo.png.jpeg"
header_cols = st.columns([1, 5])
with header_cols[0]:
    if logo_path.exists():
        st.image(str(logo_path), width=80)
with header_cols[1]:
    st.title("MicroSeg Lab")
    st.write("Microscopy‑agnostic segmentation with classical, SAM2, reviewer, and human‑in‑loop options.")

with st.sidebar:
    st.header("Run Settings")
    with st.expander("Planning & Review", expanded=True):
        use_llm = st.checkbox("Use automatic planning", value=True)
        llm_model = st.text_input("Planner model", value="gpt-4o")
        temperature = st.slider("Planner temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        user_prompt = st.text_area("Segmentation intent", value="", height=80, help="Describe what you want segmented.")
        use_reviewer = st.checkbox("Enable reviewer", value=False)
        reviewer_model = st.text_input("Reviewer model", value="gpt-4o")
        reviewer_temperature = st.slider("Reviewer temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    with st.expander("Performance"):
        rounds = st.slider("Rounds", min_value=1, max_value=4, value=cfg.default_rounds)
        relax_steps = st.slider("Relax steps if empty", min_value=0, max_value=5, value=cfg.default_relax_steps)
        cap_masks_total = st.number_input("Cap total SAM masks", min_value=500, max_value=6000, value=cfg.cap_masks_total, step=100)
        fast_mode = st.checkbox("Fast mode", value=False, help="Single SAM round, smaller caps, no multimask.")
        runtime_budget = st.number_input("Runtime budget (s, optional)", min_value=0.0, max_value=300.0, value=0.0, step=5.0)
        multimask_output = st.checkbox("SAM multimask output", value=True)
        downsample_max = st.number_input("Max image side (px) for downsample", min_value=500, max_value=4000, value=1500, step=100)
    with st.expander("Selection Thresholds"):
        min_quality = st.slider("min_quality", min_value=0.15, max_value=0.75, value=cfg.min_quality, step=0.01)
        iou_thresh = st.slider("iou_thresh", min_value=0.10, max_value=0.60, value=cfg.iou_thresh, step=0.01)
        max_instances = st.number_input("max_instances", min_value=10, max_value=800, value=cfg.max_instances, step=10)
    with st.expander("Human Trigger Controls"):
        use_human = st.checkbox("Enable human-in-loop trigger", value=False)
        auto_pause_human = st.checkbox("Pause when human needed", value=True)
        human_min_coverage = st.slider("min_coverage", min_value=0.0, max_value=0.2, value=0.005, step=0.001)
        human_max_coverage = st.slider("max_coverage", min_value=0.2, max_value=0.95, value=0.60, step=0.01)
        human_min_quality = st.slider("min_quality", min_value=0.05, max_value=0.8, value=0.35, step=0.01)
        human_min_instances = st.number_input("min_instances", min_value=1, max_value=50, value=3, step=1)
        human_max_component_frac = st.slider("max_component_frac", min_value=0.1, max_value=0.9, value=0.35, step=0.01)
        human_edge_thresh = st.slider("edge_threshold", min_value=0.05, max_value=0.6, value=0.25, step=0.01)
        human_gain_thresh = st.slider("min_coverage_gain", min_value=0.0, max_value=0.1, value=0.01, step=0.005)
    with st.expander("Manual Prompts"):
        pos_text = st.text_area("Positive points [[x,y],...]", value="[]", height=80)
        neg_text = st.text_area("Negative points [[x,y],...]", value="[]", height=80)
        box_text = st.text_area("Boxes [[x1,y1,x2,y2],...]", value="[]", height=80)
    with st.expander("Reference Example"):
        ref_upload = st.file_uploader("Reference image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="ref_img")
        ref_mask_upload = st.file_uploader("Reference mask", type=["tif", "tiff", "png", "jpg", "jpeg"], key="ref_mask")
        reference_target = st.text_input("Reference target name (optional)", value="")

uploaded = st.file_uploader("Upload image (tiff/jpg/png)", type=["tif", "tiff", "png", "jpg", "jpeg"])

if uploaded:
    image_np = load_image_any(uploaded.getvalue())
    image_np, scale = maybe_downsample(image_np, target_max=downsample_max)
    if "pos_points" not in st.session_state:
        st.session_state.pos_points = []
    if "neg_points" not in st.session_state:
        st.session_state.neg_points = []
    if "boxes" not in st.session_state:
        st.session_state.boxes = []
    if "needs_human" not in st.session_state:
        st.session_state.needs_human = False
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "reference_image_u8" not in st.session_state:
        st.session_state.reference_image_u8 = None
    if "reference_mask_u8" not in st.session_state:
        st.session_state.reference_mask_u8 = None
    st.write(f"Loaded image: shape {image_np.shape}, scale {scale:.3f}")
    st.image(image_np, caption="Input", width=720)

    if ref_upload is not None:
        ref_img = load_image_any(ref_upload.getvalue())
        st.session_state.reference_image_u8 = ref_img
        st.image(ref_img, caption="Reference image", width=360)
    if ref_mask_upload is not None:
        ref_mask_img = load_image_any(ref_mask_upload.getvalue())
        st.session_state.reference_mask_u8 = mask_to_uint8(ref_mask_img)
        st.image(st.session_state.reference_mask_u8, caption="Reference mask", width=360)

    if st.session_state.needs_human:
        st.warning("Human prompting required. Add hints below, then rerun.")
        disp_w = 720
        disp_h = int(image_np.shape[0] * (disp_w / image_np.shape[1]))
        scale_x = image_np.shape[1] / disp_w
        scale_y = image_np.shape[0] / disp_h
        bg_img = Image.fromarray(image_np)

        st.subheader("Click-based prompts")
        pos_canvas = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=4,
            stroke_color="#ff0000",
            background_image=bg_img,
            update_streamlit=True,
            height=disp_h,
            width=disp_w,
            drawing_mode="point",
            key="pos_canvas",
        )
        neg_canvas = st_canvas(
            fill_color="rgba(0, 0, 255, 0.3)",
            stroke_width=4,
            stroke_color="#0000ff",
            background_image=bg_img,
            update_streamlit=True,
            height=disp_h,
            width=disp_w,
            drawing_mode="point",
            key="neg_canvas",
        )
        box_canvas = st_canvas(
            fill_color="rgba(0, 255, 0, 0.2)",
            stroke_width=2,
            stroke_color="#00aa00",
            background_image=bg_img,
            update_streamlit=True,
            height=disp_h,
            width=disp_w,
            drawing_mode="rect",
            key="box_canvas",
        )
        cols = st.columns(2)
        if cols[0].button("Use these hints"):
            st.session_state.pos_points = canvas_points(pos_canvas.json_data, scale_x, scale_y)
            st.session_state.neg_points = canvas_points(neg_canvas.json_data, scale_x, scale_y)
            st.session_state.boxes = canvas_boxes(box_canvas.json_data, scale_x, scale_y)
            st.success("Hints updated. Click 'Run segmentation' to rerun.")
        if cols[1].button("Clear hints"):
            st.session_state.pos_points = []
            st.session_state.neg_points = []
            st.session_state.boxes = []
            st.success("Hints cleared.")

    hints = HumanHints(
        pos_points=st.session_state.pos_points or parse_points(pos_text),
        neg_points=st.session_state.neg_points or parse_points(neg_text),
        boxes=st.session_state.boxes or parse_boxes(box_text),
    )

    if st.button("Run segmentation"):
        with st.spinner("Running segmentation..."):
            settings = RunSettings(
                rounds=rounds,
                relax_steps=relax_steps,
                cap_masks_total=cap_masks_total,
                multimask_output=multimask_output if not fast_mode else False,
                use_llm=use_llm,
                llm_model=llm_model,
                temperature=temperature,
                downsample_max=downsample_max,
                min_quality_override=min_quality,
                iou_thresh_override=iou_thresh,
                max_instances_override=max_instances,
                user_prompt=user_prompt if user_prompt.strip() else None,
                fast_mode=fast_mode,
                runtime_budget=runtime_budget if runtime_budget > 0 else None,
                use_human_loop=use_human,
                auto_pause_for_human=auto_pause_human,
                human_min_coverage=human_min_coverage,
                human_max_coverage=human_max_coverage,
                human_min_quality=human_min_quality,
                human_min_instances=human_min_instances,
                human_max_component_frac=human_max_component_frac,
                human_edge_thresh=human_edge_thresh,
                human_gain_thresh=human_gain_thresh,
                use_reviewer=use_reviewer,
                reviewer_model=reviewer_model,
                reviewer_temperature=reviewer_temperature,
                reference_target=reference_target.strip() or None,
            )
            result = run_segmentation(
                image_rgb=image_np,
                sam_paths=cfg.sam2_paths,
                settings=settings,
                hints=hints,
                save_dir=Path("results"),
                reference_image_u8=st.session_state.reference_image_u8,
                reference_mask_u8=st.session_state.reference_mask_u8,
            )
        st.session_state.last_result = result
        st.session_state.needs_human = bool(result["history"][-1]["summary"].get("human_needed"))
        st.success(f"Done. Instances: {len(result['selected'])}")
        tabs = st.tabs(["Overview", "History"])
        with tabs[0]:
            st.image(result["overlay_union_rgb_u8"], caption="Final union overlay", width=720)

            reviewer_entries = [h for h in result["history"] if h["summary"].get("reviewer_decisions")]
            if reviewer_entries:
                reviewer = reviewer_entries[-1]["summary"]["reviewer_decisions"]
                decisions = reviewer.get("decisions", [])
                if isinstance(decisions, list) and decisions:
                    st.subheader("Reviewer decisions")
                    st.dataframe(decisions, use_container_width=True)

        with tabs[1]:
            st.subheader("Stage summaries")
            for h in result["history"]:
                s = h["summary"]
                tag = " (human needed)" if s.get("human_needed") else ""
                with st.expander(
                    f"Stage {s.get('stage','sam')} Round {s.get('round','?')} — instances {s['num_instances_selected']}{tag}"
                ):
                    st.json(s)
                    st.image(h["overlay_instances"], caption="Instances overlay", width=720)
                    st.image(h["overlay_union"], caption="Union overlay", width=720)
                    if s.get("human_needed"):
                        st.warning("Human input recommended: segmentation quality/coverage looks low. Add hints and rerun.")

        if st.button("Download results as zip"):
            instance_masks = [c.mask_u8 for c in result["selected"]]
            zip_bytes = build_zip_export(result["union_mask_u8"], instance_masks, result["plan"], result["history"])
            st.download_button("Download ZIP", data=zip_bytes, file_name="segmentation_results.zip")
else:
    st.info("Upload an image to begin.")
