"""
About author page for VantaScope Pro
"""

import streamlit as st
from pathlib import Path


def render():
    st.markdown("## 👤 About the Author")
    st.caption("Project context + credits.")

    _header()
    st.markdown("---")
    _bio()
    st.markdown("---")
    _acknowledgements()


def _header():
    col1, col2 = st.columns([1, 2])

    with col1:
        # Optional headshot if you add it later
        assets = Path(__file__).resolve().parents[1] / "assets"
        photo = assets / "author_photo.jpg"
        if photo.exists():
            st.image(str(photo), use_container_width=True)
        else:
            st.info("Add `src/vantascope/interface/assets/author_photo.jpg` to show a headshot.")

    with col2:
        st.markdown(
            """
<div class="intro-box">
  <h3>Haidar Bin Hamid</h3>
  <p>
    AI Engineer & Scientist — building practical, explainable AI systems for scientific workflows.
    VantaScope 5090 Pro was developed for the <strong>Hack the Microscope</strong> event by team
    <strong>Microscope Megaminds</strong> (University of Cincinnati).
  </p>
</div>
            """,
            unsafe_allow_html=True
        )


def _bio():
    st.markdown("### 🧬 Focus Areas")
    st.markdown(
        """
- Microscopy interpretation and materials characterization  
- Physics-aware representation learning and disentanglement  
- Graph reasoning for defect interactions  
- Uncertainty-aware scientific inference  
        """
    )

    st.markdown("### 🎯 Project Goal")
    st.markdown(
        """
Deliver a professional, lab-grade interface that reduces expert microscopy interpretation time
from hours/days to seconds—while keeping outputs interpretable, reviewable, and publishable.
        """
    )


def _acknowledgements():
    st.markdown("### 🙏 Credits & Acknowledgements")
    st.markdown(
        """
- Team: **Microscope Megaminds**  
- Event: **Hack the Microscope**  
- Institution: **University of Cincinnati**  
- Tools: Streamlit, Plotly, PyTorch (model wiring), Hugging Face Spaces (deployment)  
        """
    )

    st.markdown("### 📎 Contact / Links")
    st.markdown(
        """
Add your GitHub / HF Space / LinkedIn links here when ready.
        """
    )
