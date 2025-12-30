"""
Explainable AI page for VantaScope Pro
Demo-safe visualizations using current session image/results.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def render():
    st.markdown("## 🧠 Explainable AI")
    st.caption("Interactive interpretability views (demo proxies until full model wiring).")

    img = st.session_state.get("uploaded_image", None)
    results = st.session_state.get("analysis_results", None)

    if img is None and results is None:
        st.info("Run an analysis or upload/select a sample on the Main page to enable XAI views.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "Attention Heatmaps",
        "Uncertainty Maps",
        "Fuzzy Membership",
        "Gradient Explanations",
    ])

    with tab1:
        _render_attention(img)

    with tab2:
        _render_uncertainty(img, results)

    with tab3:
        _render_fuzzy_membership(results)

    with tab4:
        _render_gradients(img)


def _safe_img(img):
    if img is None:
        return None
    a = np.array(img, dtype=np.float32)
    if a.ndim != 2:
        a = a.squeeze()
    a = a - a.min()
    a = a / (a.max() + 1e-9)
    return a


def _heatmap(z, height=420, title=None, opacity=1.0):
    fig = go.Figure(go.Heatmap(z=z, colorscale="Greys", showscale=False, opacity=opacity))
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40 if title else 20, b=20),
        title=title,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
    )
    return fig


def _render_attention(img):
    st.markdown("### 🔥 Attention Heatmaps (proxy)")

    a = _safe_img(img)
    if a is None:
        st.info("No image available in session. Use Main page upload/examples.")
        return

    # Proxy “attention”: emphasize mid-frequency structures
    gx = np.gradient(a, axis=1)
    gy = np.gradient(a, axis=0)
    att = np.sqrt(gx**2 + gy**2)
    att = (att - att.min()) / (att.max() + 1e-9)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_heatmap(a, title="Base Image"), use_container_width=True, config={"displayModeBar": False})
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=a, colorscale="Greys", showscale=False, opacity=0.85, hoverinfo="skip"))
        fig.add_trace(go.Heatmap(z=att, colorscale="Hot", showscale=False, opacity=0.55, hoverinfo="skip"))
        fig.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=40, b=20),
            title="Attention Overlay (proxy)",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.caption("Note: This is a demo proxy. Replace with DINOv2 attention extraction when model is wired.")


def _render_uncertainty(img, results):
    st.markdown("### 🎯 Uncertainty Maps (proxy)")

    a = _safe_img(img)
    if a is None:
        st.info("No image available in session. Use Main page upload/examples.")
        return

    # Proxy aleatoric: local noise magnitude (high-frequency energy)
    gx = np.gradient(a, axis=1)
    gy = np.gradient(a, axis=0)
    hf = np.sqrt(gx**2 + gy**2)
    aleatoric = (hf - hf.min()) / (hf.max() + 1e-9)

    # Proxy epistemic: confidence drop where structure is ambiguous (entropy-ish)
    # Use simple nonlinearity: more mid-gray variance => more epistemic
    local = np.abs(a - a.mean())
    epistemic = (local - local.min()) / (local.max() + 1e-9)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=a, colorscale="Greys", showscale=False, opacity=0.85, hoverinfo="skip"))
        fig.add_trace(go.Heatmap(z=aleatoric, colorscale="Viridis", showscale=False, opacity=0.55, hoverinfo="skip"))
        fig.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=40, b=20),
            title="Aleatoric Uncertainty (proxy)",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=a, colorscale="Greys", showscale=False, opacity=0.85, hoverinfo="skip"))
        fig.add_trace(go.Heatmap(z=epistemic, colorscale="Plasma", showscale=False, opacity=0.55, hoverinfo="skip"))
        fig.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=40, b=20),
            title="Epistemic Uncertainty (proxy)",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if results:
        st.markdown("#### Session Summary")
        st.write(
            {
                "quality_grade": results.get("quality_grade"),
                "confidence": results.get("confidence"),
                "crystallinity": results.get("crystallinity"),
                "defect_density": results.get("defect_density"),
            }
        )


def _render_fuzzy_membership(results):
    st.markdown("### 🧩 Fuzzy Membership Functions (demo)")

    x = np.linspace(0, 1, 200)

    # Simple membership functions (demo)
    mu_perfect = np.clip(1 - 2 * np.abs(x - 0.75), 0, 1)
    mu_vacancy = np.clip(1 - 3 * np.abs(x - 0.35), 0, 1)
    mu_interstitial = np.clip(1 - 3 * np.abs(x - 0.90), 0, 1)
    mu_boundary = np.clip(1 - 2.5 * np.abs(x - 0.55), 0, 1)

    df = {
        "x": x,
        "Perfect lattice": mu_perfect,
        "Vacancy": mu_vacancy,
        "Interstitial": mu_interstitial,
        "Grain boundary": mu_boundary,
    }
    fig = px.line(
        x=df["x"],
        y=[df["Perfect lattice"], df["Vacancy"], df["Interstitial"], df["Grain boundary"]],
        labels={"x": "Normalized feature intensity", "value": "Membership μ"},
    )
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if results:
        st.caption(
            f"Current session: defect_density={results.get('defect_density', 0):.2f}, "
            f"crystallinity={results.get('crystallinity', 0):.2f}"
        )


def _render_gradients(img):
    st.markdown("### 🧭 Gradient-Based Explanations (proxy)")

    a = _safe_img(img)
    if a is None:
        st.info("No image available in session. Use Main page upload/examples.")
        return

    # Proxy Grad-CAM: edge emphasis with smoothing
    gx = np.gradient(a, axis=1)
    gy = np.gradient(a, axis=0)
    cam = np.sqrt(gx**2 + gy**2)
    cam = (cam - cam.min()) / (cam.max() + 1e-9)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_heatmap(a, title="Base Image"), use_container_width=True, config={"displayModeBar": False})
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=a, colorscale="Greys", showscale=False, opacity=0.85, hoverinfo="skip"))
        fig.add_trace(go.Heatmap(z=cam, colorscale="Magma", showscale=False, opacity=0.55, hoverinfo="skip"))
        fig.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=40, b=20),
            title="Grad-CAM Overlay (proxy)",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
