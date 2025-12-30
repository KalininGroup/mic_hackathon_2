import streamlit as st
from components.styling import apply_professional_theme
from pages import main_page

def main():
    st.set_page_config(
        page_title="VantaScope Pro | Computational Metrology",
        page_icon="🔬",
        layout="wide"
    )

    apply_professional_theme()

    with st.sidebar:
        st.markdown("### 🛰️ Metrology Suite")
        page = st.radio(
            "Navigation",
            ["Specimen Analysis", "Explainable AI (XAI)", "Physics-Informed Architecture", "Lead Investigator"],
            label_visibility="collapsed"
        )
        st.markdown("---")
        st.caption("University of Cincinnati | Microscope Megaminds")

    if page == "Specimen Analysis":
        st.markdown("""
            <div class="main-header">
                <h1 class="main-title">🔬 VantaScope 5090 Pro ⚛️</h1>
                <div class="attribution-section">
                    <p class="event-attribution" style="color: white !important;">Automated Materials Characterization Platform</p>
                    <p class="author-attribution" style="color: #aaaaaa !important;">Haidar Bin Hamid | AI Engineer & Scientist</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        main_page.render()
    else:
        st.title(page)
        st.info(f"The {page} module is currently calibrating weights for publication-grade inference.")

if __name__ == "__main__":
    main()
