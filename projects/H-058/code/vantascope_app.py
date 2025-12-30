"""
VantaScope 5090 Pro - Professional Scientific Instrument Interface
National Laboratory Grade Materials Analysis Platform
"""

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from PIL import Image
import cv2
import io
import base64
from pathlib import Path
import sys
import os

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent
sys.path.append(str(src_dir))

from vantascope.models.integrated_model import create_vantascope_model_with_uncertainty
from vantascope.interface.components.theme import apply_dark_theme
from vantascope.interface.components.model_manager import VantaScopeModelManager
from vantascope.interface.components.image_processor import IntelligentImageProcessor
from vantascope.interface.components.visualizer import ScientificVisualizer
from vantascope.interface.components.linguistic_engine import LinguisticReportGenerator

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="VantaScope 5090 Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply professional dark theme
apply_dark_theme()

# Initialize session state
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = None
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = "Ready"

class VantaScopeInterface:
    """Professional VantaScope interface controller."""
    
    def __init__(self):
        self.image_processor = IntelligentImageProcessor()
        self.visualizer = ScientificVisualizer()
        self.linguistic_engine = LinguisticReportGenerator()
    
    def render_global_toolbar(self):
        """Render the top toolbar with system status."""
        col1, col2, col3 = st.columns([2, 6, 2])
        
        with col1:
            st.markdown("### 🔬 VantaScope 5090 Pro")
        
        with col2:
            # Workflow tabs
            tab1, tab2, tab3 = st.tabs(["🔍 ANALYZE", "📊 COMPARE", "📋 REPORT"])
            
            with tab1:
                st.markdown("**Active Workspace** - Materials Analysis")
            with tab2:
                st.markdown("**Batch Comparison** - Coming Soon")
            with tab3:
                st.markdown("**Scientific Reporting** - Coming Soon")
        
        with col3:
            # System health indicators
            if st.session_state.model_manager and st.session_state.model_manager.is_loaded():
                gpu_status = "🟢 RTX 5090: Active"
                memory_usage = torch.cuda.memory_allocated() / 1024**3
                st.markdown(f"**{gpu_status}**")
                st.markdown(f"**VRAM: {memory_usage:.1f}GB**")
            else:
                st.markdown("**🔴 Model: Loading...**")
    
    def render_control_sidebar(self):
        """Render the collapsible control sidebar."""
        with st.sidebar:
            st.markdown("## 🎛️ Control Panel")
            
            # Model loading section
            st.markdown("### Model Status")
            if st.session_state.model_manager is None:
                if st.button("🚀 Initialize VantaScope AI", type="primary"):
                    with st.spinner("Loading neural networks..."):
                        st.session_state.model_manager = VantaScopeModelManager()
                        st.success("✅ Model loaded!")
                        st.rerun()
            else:
                if st.session_state.model_manager.is_loaded():
                    st.success("✅ Model Active")
                    st.markdown(f"**Parameters:** {st.session_state.model_manager.get_param_count():,}")
                    
                    # Model info
                    with st.expander("🔍 Model Details"):
                        st.markdown("""
                        **Architecture:** DINOv2-CAE + Fuzzy-GAT  
                        **Training:** 5k samples, 3 epochs  
                        **Capabilities:** Energy prediction, defect detection  
                        **Uncertainty:** Bayesian ensemble (5 members)
                        """)
            
            st.divider()
            
            # File upload section
            st.markdown("### 📂 Sample Input")
            uploaded_file = st.file_uploader(
                "Upload microscopy image",
                type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
                help="Supported: STEM, AFM, PFM images"
            )
            
            if uploaded_file is not None:
                if st.button("🔬 Analyze Sample", type="primary"):
                    if st.session_state.model_manager and st.session_state.model_manager.is_loaded():
                        self.process_uploaded_image(uploaded_file)
                    else:
                        st.error("Please initialize the model first!")
            
            st.divider()
            
            # Analysis controls
            if st.session_state.analysis_results is not None:
                st.markdown("### ⚙️ Analysis Controls")
                
                overlay_type = st.selectbox(
                    "Overlay Type",
                    ["None", "Defect Classes", "Attention Map", "Digital Twin"],
                    index=1
                )
                
                confidence_threshold = st.slider(
                    "Confidence Threshold",
                    0.0, 1.0, 0.5, 0.05,
                    help="Filter low-confidence predictions"
                )
                
                st.session_state.overlay_config = {
                    'type': overlay_type,
                    'confidence_threshold': confidence_threshold
                }
    
    def process_uploaded_image(self, uploaded_file):
        """Process uploaded image through the AI pipeline."""
        try:
            # Update status
            st.session_state.processing_status = "Processing..."
            
            # Load and preprocess image
            image = Image.open(uploaded_file)
            st.session_state.current_image = image
            
            # Intelligent cropping and preprocessing
            processed_tensor = self.image_processor.process_image(image)
            
            # AI inference
            st.session_state.processing_status = "Running AI analysis..."
            results = st.session_state.model_manager.analyze_sample(processed_tensor)
            
            # Store results
            st.session_state.analysis_results = results
            st.session_state.processing_status = "Complete"
            
            st.success("✅ Analysis complete!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.session_state.processing_status = "Error"
    
    def render_main_viewport(self):
        """Render the main analysis viewport."""
        st.markdown("## 🖼️ Analysis Viewport")
        
        if st.session_state.current_image is None:
            # Welcome screen
            st.markdown("""
            <div style='text-align: center; padding: 100px 20px; background: #1A1C24; border-radius: 10px; border: 1px solid #30333D;'>
                <h2 style='color: #00A3FF; margin-bottom: 20px;'>🔬 VantaScope 5090 Pro</h2>
                <p style='color: #FFFFFF; font-size: 18px; margin-bottom: 30px;'>
                    Professional Materials Analysis Platform
                </p>
                <p style='color: #888888; font-size: 14px;'>
                    Upload a microscopy image to begin analysis<br>
                    Supports STEM, AFM, PFM formats
                </p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        if st.session_state.analysis_results is None:
            # Processing state
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"**Status:** {st.session_state.processing_status}")
                if st.session_state.processing_status == "Processing...":
                    st.progress(0.5)
            return
        
        # Main analysis view - two columns
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Image viewport with overlays
            self.render_image_viewport()
        
        with col2:
            # Analysis inspector panel
            self.render_inspector_panel()
    
    def render_image_viewport(self):
        """Render the main image viewport with overlays."""
        results = st.session_state.analysis_results
        overlay_config = getattr(st.session_state, 'overlay_config', {'type': 'Defect Classes'})
        
        # Create image visualization
        if overlay_config['type'] == 'Digital Twin':
            # Show reconstruction
            fig = self.visualizer.create_reconstruction_comparison(
                st.session_state.current_image, 
                results['reconstruction']
            )
        elif overlay_config['type'] == 'Defect Classes':
            # Show defect overlay
            fig = self.visualizer.create_defect_overlay(
                st.session_state.current_image,
                results['patch_defect_probs'],
                confidence_threshold=overlay_config.get('confidence_threshold', 0.5)
            )
        elif overlay_config['type'] == 'Attention Map':
            # Show attention heatmap
            fig = self.visualizer.create_attention_overlay(
                st.session_state.current_image,
                results['attention_maps']
            )
        else:
            # Show original image
            fig = self.visualizer.create_simple_image_view(st.session_state.current_image)
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    def render_inspector_panel(self):
        """Render the analysis inspector panel."""
        st.markdown("### 🔍 Analysis Inspector")
        
        results = st.session_state.analysis_results
        
        # Material DNA card
        self.render_material_dna_card(results)
        
        st.markdown("---")
        
        # Disentanglement radar
        self.render_disentanglement_radar(results)
        
        st.markdown("---")
        
        # Predictions & physics
        self.render_predictions_panel(results)
    
    def render_material_dna_card(self, results):
        """Render the Material DNA identification card."""
        st.markdown("#### 🧬 Material DNA")
        
        # Main classification
        crystallinity = results['properties']['crystallinity'].item()
        defect_density = results['properties']['defect_density'].item()
        
        if crystallinity > 0.7:
            material_type = "GRAPHENE (Single Crystal)"
            quality_grade = "Grade A"
        elif crystallinity > 0.4:
            material_type = "GRAPHENE (Polycrystalline)"
            quality_grade = "Grade B"
        else:
            material_type = "GRAPHENE (Amorphous)"
            quality_grade = "Grade C"
        
        st.markdown(f"**Classification:** {material_type}")
        st.markdown(f"**Quality:** {quality_grade}")
        
        # Key metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Energy",
                f"{results['energy_mean'].item():.1f} eV",
                delta=f"±{results['energy_std'].item():.1f}"
            )
        
        with col2:
            st.metric(
                "Crystallinity",
                f"{crystallinity:.1%}",
                delta=None
            )
        
        with col3:
            st.metric(
                "Defect Density",
                f"{defect_density:.2%}",
                delta=None
            )
    
    def render_disentanglement_radar(self, results):
        """Render the disentanglement radar chart."""
        st.markdown("#### 🎯 Structural Analysis")
        
        # Extract latent dimensions
        geometric = results['geometric'][0].cpu().numpy()
        topological = results['topological'][0].cpu().numpy()
        disorder = results['disorder'][0].cpu().numpy()
        
        # Create radar chart
        fig = self.visualizer.create_disentanglement_radar(geometric, topological, disorder)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    def render_predictions_panel(self, results):
        """Render physics predictions panel."""
        st.markdown("#### ⚡ Material Properties")
        
        # Energy prediction with uncertainty
        energy_mean = results['energy_mean'].item()
        energy_std = results['energy_std'].item()
        
        st.markdown(f"**Total Energy:** {energy_mean:.1f} ± {energy_std:.1f} eV")
        
        # Material properties
        bandgap = results['bandgap_mean'].item()
        modulus = results['modulus_mean'].item()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Bandgap", f"{bandgap:.3f} eV", help="Electronic bandgap prediction")
        
        with col2:
            st.metric("Elastic Modulus", f"{modulus:.3f} TPa", help="Young's modulus prediction")
        
        # Defect analysis
        defect_probs = results['properties']['defect_probs'][0].cpu().numpy()
        defect_names = ['Perfect', 'Vacancy', 'Interstitial', 'Grain Boundary']
        
        st.markdown("**Defect Analysis:**")
        for name, prob in zip(defect_names, defect_probs):
            if prob > 0.1:  # Only show significant probabilities
                st.markdown(f"- {name}: {prob:.1%}")

def main():
    """Main application entry point."""
    interface = VantaScopeInterface()
    
    # Render UI components
    interface.render_global_toolbar()
    interface.render_control_sidebar()
    interface.render_main_viewport()

if __name__ == "__main__":
    main()
