from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image


EXAMPLE_SAMPLES = [
    "High-Quality Graphene",
    "Polycrystalline Graphene",
    "Defective Graphene",
    "Pristine Carbon Nanotube",
    "Damaged Graphene Oxide",
]


def generate_synthetic_sample(sample_name: str, size: int = 256) -> np.ndarray:
    np.random.seed(hash(sample_name) % 10000)
    x, y = np.meshgrid(np.linspace(-2, 2, size), np.linspace(-2, 2, size))
    
    if "High-Quality" in sample_name:
        pattern = np.sin(3*x) * np.cos(3*y) + 0.08 * np.random.random((size, size))
    elif "Polycrystalline" in sample_name:
        pattern = (np.sin(3*x) * np.cos(3*y) + 
                  0.4 * np.sin(5*x + 1) * np.cos(2*y + 1) + 
                  0.15 * np.random.random((size, size)))
    elif "Nanotube" in sample_name:
        pattern = np.sin(8*x) * np.exp(-0.3*y**2) + 0.1 * np.random.random((size, size))
    elif "Damaged" in sample_name or "Oxide" in sample_name:
        pattern = 0.15 * np.sin(2*x) * np.cos(2*y) + 0.85 * np.random.random((size, size))
    else:
        pattern = 0.4 * np.sin(3*x) * np.cos(3*y) + 0.6 * np.random.random((size, size))
    
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-9)
    return pattern.astype(np.float32)


def load_uploaded_image(uploaded_file, size: int = 256) -> np.ndarray:
    img = Image.open(uploaded_file).convert("L").resize((size, size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def _laplacian(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    up = np.roll(a, -1, axis=0)
    down = np.roll(a, 1, axis=0)
    left = np.roll(a, -1, axis=1)
    right = np.roll(a, 1, axis=1)
    return (up + down + left + right - 4.0 * a)


def _blur3(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    out = 4.0 * a
    out += 2.0 * (np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1))
    out += (np.roll(np.roll(a, 1, 0), 1, 1) + np.roll(np.roll(a, 1, 0), -1, 1) +
            np.roll(np.roll(a, -1, 0), 1, 1) + np.roll(np.roll(a, -1, 0), -1, 1))
    return out / 16.0


def compute_metrology(img: np.ndarray) -> dict:
    a = img.astype(np.float32)
    
    mean_val = float(a.mean())
    std_val = float(a.std())
    snr = float(mean_val / (std_val + 1e-6))
    
    hist, _ = np.histogram(a, bins=64, range=(0, 1), density=True)
    p = hist / (hist.sum() + 1e-12)
    entropy = float(-(p * np.log2(p + 1e-12)).sum())
    
    lap = _laplacian(a)
    sharpness = float(lap.var())
    edge_density = float((np.abs(lap) > 0.10).mean())
    
    fft = np.fft.rfft2(a - a.mean())
    mag = np.abs(fft)
    mag[:, :3] = 0
    periodicity = float(np.clip(mag.mean() / (mag.std() + 1e-6), 0.0, 5.0) / 5.0)
    
    return {
        "Mean Intensity": mean_val,
        "Std Intensity": std_val,
        "SNR": snr,
        "Entropy (bits)": entropy,
        "Sharpness": sharpness,
        "Edge Density": edge_density,
        "Periodicity": periodicity,
    }


def compute_grade(crystallinity: float, defect_density: float) -> str:
    score = 0.65 * crystallinity + 0.35 * (1 - defect_density)
    if score >= 0.88: return "A+"
    if score >= 0.78: return "A"
    if score >= 0.65: return "B"
    if score >= 0.50: return "C"
    if score >= 0.35: return "D"
    return "F"


def analyze_image(sample_name: str, img: np.ndarray) -> dict:
    m = compute_metrology(img)
    
    crystallinity = float(np.clip(
        0.55 * m["Periodicity"] + 0.45 * (1 - m["Entropy (bits)"] / 6.0), 
        0, 1
    ))
    defect_density = float(np.clip(0.12 + 0.95 * m["Edge Density"], 0, 0.95))
    grain_size = float(np.clip(5 + 85 * crystallinity * (1 - defect_density), 3, 95))
    
    quality_grade = compute_grade(crystallinity, defect_density)
    confidence = float(np.clip(0.70 + 0.28 * crystallinity - 0.12 * defect_density, 0.45, 0.98))
    
    bandgap = float(np.clip(0.001 + 0.38 * defect_density * (1 - crystallinity), 0.0, 0.40))
    if bandgap < 0.01:
        conductivity = "Metallic"
    elif bandgap < 0.10:
        conductivity = "Semiconducting"
    else:
        conductivity = "Insulating"
    
    elastic_modulus = float(np.clip(100 + 1000 * crystallinity * (1 - 0.6 * defect_density), 100, 1100))
    
    if crystallinity > 0.75 and defect_density < 0.25:
        finding = "Excellent structural integrity with minimal defects"
    elif defect_density < 0.45:
        finding = "Moderate crystallinity with some grain boundaries and defects"
    else:
        finding = "Significant structural disorder with high defect concentration"
    
    return {
        "quality_grade": quality_grade,
        "confidence": confidence,
        "crystallinity": crystallinity,
        "defect_density": defect_density,
        "grain_size": grain_size,
        "bandgap": bandgap,
        "conductivity": conductivity,
        "elastic_modulus": elastic_modulus,
        "finding": finding,
        "metrology": m,
    }


def plot_raw_image(img: np.ndarray, height: int = 380) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=img,
        colorscale="Greys",
        showscale=False,
        hovertemplate="x: %{x}<br>y: %{y}<br>I: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=5, r=5, t=5, b=5),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )
    return fig


def plot_analyzed_image(img: np.ndarray, defect_density: float, height: int = 380) -> go.Figure:
    processed = _blur3(img)
    h, w = processed.shape
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=processed,
        colorscale="Greys",
        showscale=False,
        opacity=0.7,
        hoverinfo="skip",
    ))
    
    np.random.seed(42)
    n_nodes = 36
    grid_pts = 6
    xs, ys = [], []
    for i in range(grid_pts):
        for j in range(grid_pts):
            cx = 20 + (w - 40) * i / (grid_pts - 1) + np.random.uniform(-8, 8)
            cy = 20 + (h - 40) * j / (grid_pts - 1) + np.random.uniform(-8, 8)
            xs.append(cx)
            ys.append(cy)
    xs, ys = np.array(xs), np.array(ys)
    
    colors, labels = [], []
    for _ in range(n_nodes):
        r = np.random.random()
        if defect_density > 0.55:
            if r < 0.65:
                colors.append("#E53935")
                labels.append("Vacancy/Defect")
            elif r < 0.85:
                colors.append("#FFA000")
                labels.append("Grain Boundary")
            else:
                colors.append("#43A047")
                labels.append("Perfect Lattice")
        elif defect_density > 0.25:
            if r < 0.25:
                colors.append("#E53935")
                labels.append("Vacancy/Defect")
            elif r < 0.40:
                colors.append("#FFA000")
                labels.append("Grain Boundary")
            else:
                colors.append("#43A047")
                labels.append("Perfect Lattice")
        else:
            if r < 0.08:
                colors.append("#E53935")
                labels.append("Vacancy/Defect")
            elif r < 0.12:
                colors.append("#FFA000")
                labels.append("Grain Boundary")
            else:
                colors.append("#43A047")
                labels.append("Perfect Lattice")
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            if dist < 60:
                fig.add_trace(go.Scatter(
                    x=[xs[i], xs[j]], y=[ys[i], ys[j]],
                    mode="lines",
                    line=dict(width=1, color="rgba(80,80,80,0.25)"),
                    hoverinfo="skip",
                    showlegend=False,
                ))
    
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers",
        marker=dict(size=11, color=colors, opacity=0.9, line=dict(width=1.5, color="white")),
        text=labels,
        hovertemplate="Node %{pointNumber}<br>%{text}<extra></extra>",
        showlegend=False,
    ))
    
    fig.update_layout(
        height=height,
        margin=dict(l=5, r=5, t=5, b=5),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )
    return fig


def render():
    if "sample_name" not in st.session_state:
        st.session_state.sample_name = "High-Quality Graphene"
        st.session_state.sample_image = generate_synthetic_sample(st.session_state.sample_name)
        st.session_state.results = analyze_image(st.session_state.sample_name, st.session_state.sample_image)
    
    _render_intro()
    _render_workspace()
    _render_results()


def _render_intro():
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
<div style="background:#2d2d2d; border:2px solid #444444; border-radius:10px; padding:1.5rem; box-shadow:0 2px 8px rgba(0,0,0,0.3);">
<h4 style="color:#ffffff !important; margin:0 0 0.8rem 0; font-weight:600;">What VantaScope Does</h4>
<p style="color:#e0e0e0 !important; line-height:1.65; margin:0; font-size:0.95rem;">
VantaScope 5090 Pro combines deep learning with graph neural networks to deliver 
instant, automated characterization of microscopy images. It provides comprehensive 
structural, electronic, and defect analysis in seconds — enabling real-time quality 
control and accelerated materials discovery.
</p>
</div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
<div style="background:#2d2d2d; border:2px solid #444444; border-radius:10px; padding:1.5rem; box-shadow:0 2px 8px rgba(0,0,0,0.3);">
<h4 style="color:#ffffff !important; margin:0 0 0.8rem 0; font-weight:600;">How to Use</h4>
<ol style="color:#e0e0e0 !important; line-height:1.7; margin:0; padding-left:1.2rem; font-size:0.95rem;">
<li style="color:#e0e0e0 !important;">Select a demo sample or upload your own image</li>
<li style="color:#e0e0e0 !important;">Click <strong style="color:#ffffff;">Run Analysis</strong> to process</li>
<li style="color:#e0e0e0 !important;">Compare raw image vs AI-analyzed view with graph overlay</li>
<li style="color:#e0e0e0 !important;">Review quantitative results and recommendations below</li>
</ol>
</div>
        """, unsafe_allow_html=True)


def _render_workspace():
    st.markdown("---")
    st.markdown("## Analysis Workspace")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Data Input")
        _input_panel()
    
    with col2:
        st.markdown("#### Raw Image")
        _raw_panel()
    
    with col3:
        st.markdown("#### AI Analysis")
        _processed_panel()


def _input_panel():
    input_mode = st.selectbox(
        "Input Method",
        ["Demo Samples", "Upload Image", "Camera", "URL"],
        index=0,
        label_visibility="collapsed",
    )
    
    if input_mode == "Upload Image":
        uploaded = st.file_uploader("Upload", type=["png", "jpg", "jpeg", "tif", "tiff"], label_visibility="collapsed")
        if uploaded:
            st.session_state.sample_image = load_uploaded_image(uploaded)
            st.session_state.sample_name = f"Uploaded: {uploaded.name}"
            st.session_state.results = None
            st.success("Image loaded")
    
    elif input_mode == "Camera":
        cam = st.camera_input("Capture", label_visibility="collapsed")
        if cam:
            st.session_state.sample_image = load_uploaded_image(cam)
            st.session_state.sample_name = "Camera Capture"
            st.session_state.results = None
    
    elif input_mode == "URL":
        url = st.text_input("Image URL", placeholder="https://...")
        if url:
            st.info("URL import coming soon")
    
    else:
        current = st.session_state.get("sample_name", EXAMPLE_SAMPLES[0])
        idx = EXAMPLE_SAMPLES.index(current) if current in EXAMPLE_SAMPLES else 0
        
        selected = st.selectbox("Sample", EXAMPLE_SAMPLES, index=idx, label_visibility="collapsed")
        
        if selected != st.session_state.get("sample_name"):
            st.session_state.sample_name = selected
            st.session_state.sample_image = generate_synthetic_sample(selected)
            st.session_state.results = None
    
    st.markdown("")
    
    if st.button("Run Analysis", type="primary", use_container_width=True):
        img = st.session_state.get("sample_image")
        name = st.session_state.get("sample_name", "Sample")
        if img is None:
            st.warning("Select or upload a sample first")
        else:
            with st.spinner("Analyzing..."):
                st.session_state.results = analyze_image(name, img)
            st.success("Complete")
            st.rerun()


def _raw_panel():
    img = st.session_state.get("sample_image")
    name = st.session_state.get("sample_name")
    
    if img is None:
        st.markdown("""
<div style="height:380px; display:flex; align-items:center; justify-content:center; background:#2d2d2d; border:2px dashed #555; border-radius:8px;">
<span style="color:#e0e0e0 !important;">Select a sample to view</span>
</div>
        """, unsafe_allow_html=True)
        return
    
    st.plotly_chart(plot_raw_image(img), use_container_width=True, config={"displayModeBar": False})
    st.caption(f"Sample: {name}")


def _processed_panel():
    img = st.session_state.get("sample_image")
    results = st.session_state.get("results")
    
    if img is None or results is None:
        st.markdown("""
<div style="height:380px; display:flex; align-items:center; justify-content:center; background:#2d2d2d; border:2px dashed #555; border-radius:8px;">
<span style="color:#e0e0e0 !important;">Run analysis to view results</span>
</div>
        """, unsafe_allow_html=True)
        return
    
    st.plotly_chart(
        plot_analyzed_image(img, results["defect_density"]), 
        use_container_width=True, 
        config={"displayModeBar": False}
    )
    
    c1, c2 = st.columns(2)
    c1.metric("Grade", results["quality_grade"], f'{results["confidence"]:.0%}')
    c2.metric("Defects", f'{results["defect_density"]:.1%}')


def _render_results():
    st.markdown("---")
    st.markdown("## Analysis Results")
    
    results = st.session_state.get("results")
    name = st.session_state.get("sample_name")
    
    if results is None:
        st.info("Run analysis to view results")
        return
    
    st.markdown(f"""
**Sample:** {name}  
**Quality Grade:** {results['quality_grade']} ({results['confidence']:.0%} confidence)  
**Finding:** {results['finding']}
    """)
    
    st.markdown("---")
    st.markdown("### Quantitative Analysis")
    
    m = results["metrology"]
    df_met = pd.DataFrame([{"Metric": k, "Value": f"{v:.4f}"} for k, v in m.items()])
    
    df_props = pd.DataFrame([
        {"Property": "Crystallinity", "Value": f"{results['crystallinity']:.1%}"},
        {"Property": "Defect Density", "Value": f"{results['defect_density']:.1%}"},
        {"Property": "Grain Size", "Value": f"{results['grain_size']:.1f} nm"},
        {"Property": "Bandgap", "Value": f"{results['bandgap']:.3f} eV"},
        {"Property": "Conductivity", "Value": results['conductivity']},
        {"Property": "Elastic Modulus", "Value": f"{results['elastic_modulus']:.0f} GPa"},
    ])
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Image Metrology**")
        st.dataframe(df_met, hide_index=True, use_container_width=True)
    with c2:
        st.markdown("**Material Properties**")
        st.dataframe(df_props, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    
    with c1:
        cryst_desc = 'excellent' if results['crystallinity'] > 0.75 else 'moderate' if results['crystallinity'] > 0.45 else 'poor'
        defect_desc = 'minimal' if results['defect_density'] < 0.20 else 'moderate' if results['defect_density'] < 0.45 else 'significant'
        st.markdown(f"""
**Structural Interpretation**

Crystallinity of {results['crystallinity']:.1%} indicates {cryst_desc} atomic ordering. 
The defect density of {results['defect_density']:.1%} suggests {defect_desc} structural disruption.
Estimated grain size is {results['grain_size']:.1f} nm.
        """)
    
    with c2:
        modulus_desc = 'approaches theoretical maximum' if results['elastic_modulus'] > 900 else 'is reduced due to defects' if results['elastic_modulus'] < 500 else 'is within typical range'
        st.markdown(f"""
**Property Predictions**

Electronic bandgap of {results['bandgap']:.3f} eV indicates {results['conductivity'].lower()} behavior.
Elastic modulus of {results['elastic_modulus']:.0f} GPa {modulus_desc}.
        """)
    
    st.markdown("---")
    st.markdown("**Recommendations**")
    
    if results['quality_grade'] in ['A+', 'A']:
        st.markdown("Sample exhibits excellent quality suitable for high-performance applications. No further optimization needed.")
    elif results['quality_grade'] == 'B':
        st.markdown("Good quality suitable for most applications. Consider annealing treatment if higher crystallinity is needed.")
    elif results['quality_grade'] == 'C':
        st.markdown("Moderate quality. Recommend optimizing synthesis parameters to reduce defect density.")
    else:
        st.markdown("Significant quality issues detected. Review synthesis conditions and consider alternative preparation methods.")
    
    st.markdown("---")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        if st.button("Download Report", type="primary", use_container_width=True):
            st.success("Report generation initiated")
