"""
Model comparison page for VantaScope Pro
Demo benchmarking tables + plots.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render():
    st.markdown("## 📊 Model Comparison")
    st.caption("Benchmarking view (demo metrics until full evaluation pipeline is connected).")

    df = _load_demo_benchmarks()

    st.markdown("### 📋 Benchmark Table")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### ⚡ Speed vs Accuracy")
        fig = px.scatter(
            df,
            x="Inference (ms)",
            y="Accuracy",
            size="Interpretability",
            hover_name="Method",
        )
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        st.markdown("### 🧠 Interpretability Radar")
        method = st.selectbox("Select method", df["Method"].tolist(), index=0)
        row = df[df["Method"] == method].iloc[0]

        categories = ["Accuracy", "Speed (inv)", "Interpretability", "User Score"]
        speed_inv = 1.0 / max(float(row["Inference (ms)"]), 1.0)
        values = [row["Accuracy"], speed_inv, row["Interpretability"], row["User Score"]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=method
        ))
        fig.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=40, b=20),
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")
    st.markdown("### 🧪 Notes")
    st.markdown(
        """
- Replace these demo metrics with your **evaluation script outputs** when ready (CSV/JSON).
- Recommended: log **Accuracy/F1**, **calibration error**, **coverage**, and **inference latency** on CPU vs GPU.
        """
    )


def _load_demo_benchmarks():
    data = [
        {"Method": "VantaScope 5090 Pro", "Accuracy": 0.92, "Inference (ms)": 220, "Interpretability": 0.90, "User Score": 0.94},
        {"Method": "Manual Analysis",      "Accuracy": 0.85, "Inference (ms)": 3600000, "Interpretability": 0.95, "User Score": 0.70},
        {"Method": "ImageJ Automation",    "Accuracy": 0.78, "Inference (ms)": 1500, "Interpretability": 0.55, "User Score": 0.74},
        {"Method": "Basic CNN",            "Accuracy": 0.84, "Inference (ms)": 140, "Interpretability": 0.30, "User Score": 0.78},
        {"Method": "Traditional GNN",      "Accuracy": 0.86, "Inference (ms)": 420, "Interpretability": 0.55, "User Score": 0.80},
    ]
    return pd.DataFrame(data)
