"""
Professional styling for VantaScope Pro - readable in light/dark, hides default nav.
"""
import streamlit as st

def apply_professional_theme():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ---------- Theme tokens ---------- */
:root{
  --vs-font: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;

  --vs-bg: #ffffff;
  --vs-panel: #ffffff;
  --vs-text: #0f172a;
  --vs-text2: #475569;
  --vs-border: rgba(15, 23, 42, 0.14);
  --vs-shadow: 0 6px 20px rgba(0,0,0,0.08);

  --vs-blue: #1565c0;
  --vs-blue2: #0d47a1;
  --vs-blueSoft: rgba(21, 101, 192, 0.10);
}

@media (prefers-color-scheme: dark){
  :root{
    --vs-bg: #0b1220;
    --vs-panel: #0f172a;
    --vs-text: rgba(255,255,255,0.92);
    --vs-text2: rgba(255,255,255,0.70);
    --vs-border: rgba(255,255,255,0.14);
    --vs-shadow: 0 10px 30px rgba(0,0,0,0.45);
    --vs-blueSoft: rgba(21, 101, 192, 0.22);
  }
}

/* Honor Streamlit data-theme if present */
html[data-theme="dark"], body[data-theme="dark"], [data-theme="dark"]{
  --vs-bg: #0b1220;
  --vs-panel: #0f172a;
  --vs-text: rgba(255,255,255,0.92);
  --vs-text2: rgba(255,255,255,0.70);
  --vs-border: rgba(255,255,255,0.14);
  --vs-shadow: 0 10px 30px rgba(0,0,0,0.45);
  --vs-blueSoft: rgba(21, 101, 192, 0.22);
}

/* ---------- App base ---------- */
.stApp, [data-testid="stAppViewContainer"]{
}

/* ---------- Hide Streamlit chrome ---------- */
#MainMenu{ visibility: hidden !important; }

/* ---------- Hide default nav links (sidebar) ---------- */

/* ---------- Sidebar styling ---------- */
section[data-testid="stSidebar"]{
}
section[data-testid="stSidebar"] *{
}

/* ---------- Typography (scope to Markdown only) ---------- */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span{
  line-height: 1.65;
}
.stMarkdown small, .stMarkdown .caption{
}
.stMarkdown a{
  text-decoration: none;
}
.stMarkdown a:hover{
  text-decoration: underline;
}

/* ---------- Custom panels ---------- */
.main-header{
  background: var(--vs-panel);
  border-bottom: 3px solid var(--vs-blue);
  padding: 2rem 3rem;
  margin: -1rem -1rem 2rem -1rem;
  box-shadow: var(--vs-shadow);
}
.main-title{
  font-size: 2.5rem;
  font-weight: 700;
  margin: 0;
}
.event-attribution{
  font-size: 1.05rem;
  margin: 0.5rem 0 0 0;
  font-weight: 500;
}
.author-attribution{
  font-size: 0.98rem;
  margin: 0.35rem 0 0 0;
  font-style: italic;
}

.intro-box, .analysis-panel{
  border-radius: 12px;
  border: 1px solid var(--vs-border);
  padding: 1.75rem;
  box-shadow: var(--vs-shadow);
  margin: 1rem 0;
}
.intro-box h3, .analysis-panel h3{
  font-weight: 650;
  margin: 0 0 0.75rem 0;
}

/* ---------- Widgets ---------- */
div[data-testid="stWidgetLabel"] label{
}

.stButton > button{
}
.stButton > button:hover{
  transform: translateY(-1px);
  box-shadow: 0 8px 22px rgba(21,101,192,0.20);
}
.stButton > button[kind="primary"]{
  color: #ffffff !important;
}
.stButton > button[kind="primary"]:hover{
}

/* ---------- Alerts ---------- */
div[data-testid="stAlert"]{
}
div[data-testid="stAlert"] p{
}
</style>
        """,
        unsafe_allow_html=True,
    )
