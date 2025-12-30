"""
Architecture page for VantaScope Pro
Graph + live parameter controls stored in session_state.
"""

import streamlit as st


def render():
    st.markdown("## 🏗️ System Architecture")
    st.caption("High-level pipeline + tunable inference knobs (demo).")

    _controls()
    st.markdown("---")
    _diagram()
    st.markdown("---")
    _notes()


def _controls():
    st.markdown("### 🎛️ Live Parameters (Demo)")
    with st.expander("Open Controls", expanded=True):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.session_state.conf_threshold = st.slider(
                "Confidence threshold", 0.0, 1.0,
                float(st.session_state.get("conf_threshold", 0.65)), 0.01
            )
            st.session_state.fuzzy_sensitivity = st.slider(
                "Fuzzy sensitivity", 0.0, 2.0,
                float(st.session_state.get("fuzzy_sensitivity", 1.0)), 0.05
            )

        with c2:
            st.session_state.k_neighbors = st.slider(
                "Graph K-neighbors", 2, 24,
                int(st.session_state.get("k_neighbors", 8)), 1
            )
            st.session_state.uncertainty_cal = st.slider(
                "Uncertainty calibration", 0.5, 2.0,
                float(st.session_state.get("uncertainty_cal", 1.0)), 0.05
            )

        with c3:
            st.session_state.image_size = st.selectbox(
                "Inference image size",
                [128, 256, 384, 512],
                index=[128, 256, 384, 512].index(int(st.session_state.get("image_size", 256)))
            )
            st.session_state.overlay_density = st.selectbox(
                "Overlay density",
                ["Low", "Medium", "High"],
                index=["Low", "Medium", "High"].index(st.session_state.get("overlay_density", "Medium"))
            )

    st.markdown("#### Active Config")
    st.code(
        {
            "conf_threshold": st.session_state.get("conf_threshold"),
            "fuzzy_sensitivity": st.session_state.get("fuzzy_sensitivity"),
            "k_neighbors": st.session_state.get("k_neighbors"),
            "uncertainty_cal": st.session_state.get("uncertainty_cal"),
            "image_size": st.session_state.get("image_size"),
            "overlay_density": st.session_state.get("overlay_density"),
        },
        language="json"
    )


def _diagram():
    st.markdown("### 🔁 Pipeline Overview")

    # Graphviz renders well on Streamlit
    st.graphviz_chart(
        """
digraph VantaScope {
  rankdir=LR;
  node [shape=box, style="rounded,filled", color="#1565c0", fillcolor="#e3f2fd"];

  Upload [label="Input\\n(Upload / Examples)"];
  Pre [label="Preprocess\\n(crop/normalize)"];
  ViT [label="DINOv2 ViT\\n(feature extraction)"];
  AE [label="Disentangled AE\\n(physics-informed)"];
  GNN [label="Fuzzy-GAT\\n(spatial reasoning)"];
  Heads [label="Property Heads\\n+ Uncertainty"];
  Viz [label="Visualizer\\n(overlay/charts)"];
  Report [label="Report\\n(narrative/export)"];

  Upload -> Pre -> ViT -> AE -> GNN -> Heads -> Viz -> Report;
}
        """
    )


def _notes():
    st.markdown("### 🧾 What’s Implemented vs. Wired-In")
    st.markdown(
        """
- **UI + workflow**: upload/examples → analysis → visuals → narrative ✅  
- **Metrology proxies**: entropy/sharpness/edge density/periodicity ✅  
- **Full model** (DINOv2 + AE + Fuzzy-GAT + property heads): *hook points prepared* (wire weights + inference next)  
- **Explainable views**: proxy attention/gradients/uncertainty now; replace with true model outputs later  
        """
    )
