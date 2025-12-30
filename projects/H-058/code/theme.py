"""
Professional dark theme for VantaScope - National Lab aesthetic
"""

import streamlit as st

def apply_dark_theme():
    """Apply the professional 'Midnight Precision' theme."""
    
    st.markdown("""
    <style>
    /* Global dark theme */
    .main {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1A1C24;
        border-right: 1px solid #30333D;
    }
    
    /* Cards and containers */
    .stContainer, .element-container {
        background-color: #1A1C24;
        border-radius: 8px;
        border: 1px solid #30333D;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Buttons - Primary action (Electric Blue) */
    .stButton > button[kind="primary"] {
        background-color: #00A3FF;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* Buttons - Secondary */
    .stButton > button {
        background-color: transparent;
        color: #FFFFFF;
        border: 1px solid #30333D;
        border-radius: 6px;
    }
    
    /* Metrics styling */
    .metric-container {
        background-color: #1A1C24;
        border: 1px solid #30333D;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    
    /* Headers with professional spacing */
    h1, h2, h3 {
        color: #FFFFFF;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* Code/data text - monospace for alignment */
    .stCode, .metric-value {
        font-family: 'JetBrains Mono', 'Roboto Mono', monospace;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1A1C24;
        border-bottom: 1px solid #30333D;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #888888;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #00A3FF;
        border-bottom: 2px solid #00A3FF;
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: #1A1C24;
        border: 2px dashed #30333D;
        border-radius: 8px;
    }
    
    /* Progress bars */
    .stProgress .st-bo {
        background-color: #00A3FF;
    }
    
    /* Success/error messages */
    .stSuccess {
        background-color: #00D4AA;
        color: #0E1117;
        border-radius: 6px;
    }
    
    .stError {
        background-color: #FF4B4B;
        color: #FFFFFF;
        border-radius: 6px;
    }
    
    /* Remove Streamlit branding for professional look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Professional scientific color classes */
    .lattice-perfect { color: #00D4AA; }
    .defect-vacancy { color: #FF4B4B; }
    .defect-grain-boundary { color: #FFD700; }
    .attention-high { color: #00A3FF; }
    
    </style>
    """, unsafe_allow_html=True)
