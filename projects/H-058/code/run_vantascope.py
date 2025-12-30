"""
Launch VantaScope Streamlit Interface
"""

import streamlit as st
import subprocess
import sys
from pathlib import Path

def main():
    """Launch the VantaScope Streamlit app."""
    app_path = Path(__file__).parent / "src" / "vantascope" / "interface" / "streamlit_app.py"
    
    # Launch Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        str(app_path),
        "--theme.base", "light",
        "--theme.primaryColor", "#1e3c72",
        "--theme.backgroundColor", "#ffffff"
    ])

if __name__ == "__main__":
    main()
