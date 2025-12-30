"""
VantaScope Streamlit Interface: Materials Science Focused Digital Twin.
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import tempfile
from pathlib import Path
import sys
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.vantascope.models.autoencoder import create_autoencoder
from src.vantascope.models.fuzzy_gat import create_fuzzy_gat
from src.vantascope.visualization.graph_overlay import create_graph_overlay
from src.vantascope.data.preprocessing import MicroscopyPreprocessor
from src.vantascope.utils.helpers import get_device, set_seed
from src.vantascope.utils.logging import logger

# Page configuration
st.set_page_config(
    page_title="VantaScope 5090 Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 2rem;
}
.main-header h1 {
    color: white;
    text-align: center;
    margin: 0;
    font-family: 'Arial Black', sans-serif;
}
.metric-card {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #1e3c72;
}
.material-insight {
    background: #e8f4f8;
    padding: 1rem;
    border-radius: 8px;
    border-left: 3px solid #2e8b57;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Material property database
MATERIAL_PROPERTIES = {
    'perfect_lattice': {
        'description': 'Highly ordered crystalline structure',
        'properties': ['High electrical conductivity', 'Mechanical strength', 'Predictable electronic properties'],
        'applications': ['Electronic devices', 'Structural materials', 'Optical components'],
        'quality_indicator': 'Excellent'
    },
    'defect': {
        'description': 'Point defects or structural irregularities',
        'properties': ['Modified electronic properties', 'Potential nucleation sites', 'Altered mechanical response'],
        'applications': ['Catalysis enhancement', 'Dopant sites', 'Property tuning'],
        'quality_indicator': 'Needs attention'
    },
    'grain_boundary': {
        'description': 'Interface between crystalline regions',
        'properties': ['Reduced conductivity', 'Mechanical interfaces', 'Ion transport pathways'],
        'applications': ['Solid electrolytes', 'Mechanical reinforcement', 'Segregation sites'],
        'quality_indicator': 'Typical'
    },
    'amorphous': {
        'description': 'Non-crystalline, disordered regions',
        'properties': ['Isotropic properties', 'Higher free energy', 'Variable local structure'],
        'applications': ['Glass formation', 'Flexible electronics', 'Coatings'],
        'quality_indicator': 'Variable'
    },
    'noise': {
        'description': 'Measurement artifacts or contamination',
        'properties': ['No structural information', 'Experimental artifacts'],
        'applications': ['Quality control indicator'],
        'quality_indicator': 'Poor data quality'
    }
}

@st.cache_resource
def load_models():
    """Load models once and cache them."""
    try:
        device = get_device()
        set_seed(42)
        
        # Load models
        autoencoder = create_autoencoder().to(device)
        fuzzy_gat = create_fuzzy_gat().to(device)
        visualizer = create_graph_overlay()
        preprocessor = MicroscopyPreprocessor(enable_intelligent_crop=True)
        
        # Set to eval mode
        autoencoder.eval()
        fuzzy_gat.eval()
        
        return autoencoder, fuzzy_gat, visualizer, preprocessor, device
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        return None, None, None, None, None

def analyze_material_composition(fuzzy_memberships):
    """Analyze material composition and provide scientific insights."""
    category_names = ['perfect_lattice', 'defect', 'grain_boundary', 'amorphous', 'noise']
    dominant_categories = np.argmax(fuzzy_memberships, axis=1)
    
    # Calculate percentages
    total_nodes = len(dominant_categories)
    composition = {}
    for i, name in enumerate(category_names):
        count = np.sum(dominant_categories == i)
        percentage = (count / total_nodes) * 100
        composition[name] = {'count': count, 'percentage': percentage}
    
    # Determine primary material characteristics
    primary_phase = max(composition.keys(), key=lambda x: composition[x]['percentage'])
    primary_percentage = composition[primary_phase]['percentage']
    
    # Material quality assessment
    quality_score = (
        composition['perfect_lattice']['percentage'] * 0.4 +
        composition['grain_boundary']['percentage'] * 0.2 +
        composition['defect']['percentage'] * 0.1 +
        composition['amorphous']['percentage'] * 0.05 -
        composition['noise']['percentage'] * 0.1
    )
    
    return composition, primary_phase, quality_score

def process_image(image_array, models):
    """Process uploaded image through complete pipeline."""
    autoencoder, fuzzy_gat, visualizer, preprocessor, device = models
    
    if any(model is None for model in models):
        return None, None, None, None
    
    try:
        # Process with crop information
        processing_result = preprocessor.process_with_crop_info(image_array)
        
        # Use cropped version for analysis
        processed_tensor = processing_result['cropped_tensor'].unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            cae_outputs = autoencoder(processed_tensor)
            gat_outputs = fuzzy_gat(cae_outputs['patch_embeddings'])
        
        return cae_outputs, gat_outputs, processed_tensor, processing_result
    
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        return None, None, None, None

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔬 VantaScope 5090 Pro</h1>
        <p style="color: white; text-align: center; margin: 0;">
            AI-Powered Digital Twin for Nanoscale Materials Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Model loading status
        with st.spinner("Loading AI models..."):
            models = load_models()
        
        if all(model is not None for model in models):
            st.success("✅ Models loaded successfully")
            st.info("📊 Ready: 88M+ parameters")
        else:
            st.error("❌ Model loading failed")
            return
        
        st.divider()
        
        # File upload
        uploaded_file = st.file_uploader(
            "📁 Upload Microscopy Image",
            type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
            help="Upload STEM, AFM, or other microscopy images"
        )
        
        if uploaded_file:
            st.success(f"📎 Loaded: {uploaded_file.name}")
            st.info("💡 **Tip**: The AI will automatically crop out legends, axes, and colorbars to focus on the actual microscopy data.")
    
    # Main content area
    if uploaded_file is not None:
        # Load and display uploaded image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Process image
        with st.spinner("🧠 Analyzing material structure with VantaScope AI..."):
            cae_outputs, gat_outputs, processed_tensor, crop_info = process_image(image_array, models)
        
        if cae_outputs is not None:
            # Image comparison section
            st.subheader("📷 Intelligent Image Processing")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Image**")
                st.image(image, width=None)  # Fixed deprecated parameter
                
                # Original image properties
                st.markdown(f"""
                <div class="metric-card">
                    <strong>Original Properties:</strong><br>
                    Size: {image_array.shape[1]} × {image_array.shape[0]} pixels<br>
                    Channels: {len(image_array.shape) if len(image_array.shape) < 3 else image_array.shape[2]}<br>
                    Data Type: {image_array.dtype}<br>
                    File Size: {uploaded_file.size:,} bytes
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("**Detected Microscopy Region**")
                cropped_display = Image.fromarray((crop_info['cropped_image'] * 255).astype(np.uint8))
                st.image(cropped_display, width=None)  # Fixed deprecated parameter
                
                # Detected region properties
                cropped_shape = crop_info['cropped_image'].shape
                st.markdown(f"""
                <div class="metric-card">
                    <strong>Detected Region Properties:</strong><br>
                    Size: {cropped_shape[1]} × {cropped_shape[0]} pixels<br>
                    Crop Method: {crop_info['crop_method'].title()}<br>
                    Data Retained: {crop_info['crop_ratio']:.1%}<br>
                    Removed Artifacts: {crop_info['removed_pixels']:,} pixels
                </div>
                """, unsafe_allow_html=True)
            
            # Material Analysis Results
            st.divider()
            st.subheader("🔬 Material Structure Analysis")
            
            # Analyze composition
            fuzzy_memberships = gat_outputs['fuzzy_memberships'].cpu().numpy()
            composition, primary_phase, quality_score = analyze_material_composition(fuzzy_memberships)
            
            # Primary findings
            st.markdown(f"""
            <div class="material-insight">
                <h4>🎯 Primary Material Classification</h4>
                <p><strong>Dominant Structure:</strong> {primary_phase.replace('_', ' ').title()}</p>
                <p><strong>Confidence:</strong> {composition[primary_phase]['percentage']:.1f}%</p>
                <p><strong>Overall Quality Score:</strong> {quality_score:.1f}/100</p>
                <p><strong>Assessment:</strong> {MATERIAL_PROPERTIES[primary_phase]['quality_indicator']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed composition
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📊 Structural Composition")
                
                # Create composition dataframe
                comp_data = []
                for structure, data in composition.items():
                    comp_data.append({
                        'Structure Type': structure.replace('_', ' ').title(),
                        'Area %': f"{data['percentage']:.1f}%",
                        'Node Count': data['count'],
                        'Properties': ', '.join(MATERIAL_PROPERTIES[structure]['properties'][:2])
                    })
                
                df = pd.DataFrame(comp_data)
                st.dataframe(df, use_container_width=True)
            
            with col2:
                st.markdown("### 🔍 Material Properties")
                
                # Show properties of dominant phase
                props = MATERIAL_PROPERTIES[primary_phase]
                st.markdown(f"""
                **{primary_phase.replace('_', ' ').title()}**
                
                *{props['description']}*
                
                **Key Properties:**
                """)
                
                for prop in props['properties']:
                    st.write(f"• {prop}")
                
                st.markdown(f"""
                **Typical Applications:**
                """)
                for app in props['applications']:
                    st.write(f"• {app}")
            
            # Tabs for detailed analysis
            tab1, tab2, tab3, tab4 = st.tabs([
                "🎭 Digital Twin", 
                "🕸️ Graph Network", 
                "🔍 Attention Analysis", 
                "📈 Technical Details"
            ])
            
            with tab1:
                st.subheader("�� Digital Twin Reconstruction")
                st.info("**AI's understanding**: How well the model reconstructs the material structure")
                
                # Get reconstruction
                reconstruction = cae_outputs['reconstruction'][0, 0].cpu().numpy()
                original = processed_tensor[0, 0].cpu().numpy()
                
                # Side-by-side comparison
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                
                ax1.imshow(original, cmap='gray')
                ax1.set_title('Processed Input')
                ax1.axis('off')
                
                ax2.imshow(reconstruction, cmap='gray')
                ax2.set_title('AI Reconstruction')
                ax2.axis('off')
                
                # Difference map
                diff = np.abs(original - reconstruction)
                im3 = ax3.imshow(diff, cmap='hot')
                ax3.set_title('Reconstruction Fidelity')
                ax3.axis('off')
                plt.colorbar(im3, ax=ax3, fraction=0.046)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Quality metrics for scientists
                mse = np.mean((original - reconstruction)**2)
                psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Reconstruction Fidelity", f"{(1-mse)*100:.1f}%")
                col2.metric("Signal Quality (PSNR)", f"{psnr:.1f} dB")
                col3.metric("Model Confidence", f"{100-mse*100:.1f}%")
            
            with tab2:
                st.subheader("🕸️ Graph Network Analysis")
                st.info("**Expert View**: How AI identifies atomic bonds and material interfaces")
                
                # Create graph overlay
                autoencoder, fuzzy_gat, visualizer, preprocessor, device = models
                original_image = processed_tensor[0, 0].cpu().numpy()
                
                fig = visualizer.create_overlay(
                    image=original_image,
                    gat_outputs=gat_outputs,
                    title="Material Structure Graph Analysis"
                )
                
                st.pyplot(fig)
                
                # Graph insights for materials scientists
                edge_weights = gat_outputs['edge_weights'].cpu().numpy()
                num_nodes = len(fuzzy_memberships)
                num_edges = len(edge_weights)
                
                st.markdown("### 📋 Network Connectivity Analysis")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Atomic Sites", f"{num_nodes}")
                col2.metric("Structural Bonds", f"{num_edges}")
                col3.metric("Connectivity", f"{(num_edges*2)/num_nodes:.1f}")
                col4.metric("Bond Strength", f"{np.mean(np.abs(edge_weights)):.2f}")
                
                # Download button
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                
                st.download_button(
                    label="📥 Download Network Analysis (High-Res)",
                    data=buf.getvalue(),
                    file_name=f"vantascope_network_{uploaded_file.name.split('.')[0]}.png",
                    mime="image/png"
                )
            
            with tab3:
                st.subheader("🔍 AI Attention Analysis")
                st.info("**Model Focus**: What structural features the AI considers most important")
                
                attention_maps = cae_outputs['attention_maps'][0, 0].cpu().numpy()
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                ax1.imshow(processed_tensor[0, 0].cpu().numpy(), cmap='gray')
                ax1.set_title('Material Structure')
                ax1.axis('off')
                
                im = ax2.imshow(attention_maps, cmap='hot', alpha=0.8)
                ax2.imshow(processed_tensor[0, 0].cpu().numpy(), cmap='gray', alpha=0.3)
                ax2.set_title('AI Attention Heatmap')
                ax2.axis('off')
                plt.colorbar(im, ax=ax2, fraction=0.046, label='Attention Strength')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Attention statistics
                attention_stats = {
                    'Peak Attention': f"{np.max(attention_maps):.3f}",
                    'Average Focus': f"{np.mean(attention_maps):.3f}",
                    'Focus Variance': f"{np.std(attention_maps):.3f}",
                    'Active Regions': f"{np.sum(attention_maps > np.percentile(attention_maps, 90))}"
                }
                
                st.markdown("### 🎯 Attention Statistics")
                for key, value in attention_stats.items():
                    st.metric(key, value)
            
            with tab4:
                st.subheader("📈 Technical Analysis")
                
                # Latent space analysis for scientists
                latent = cae_outputs['latent'][0].cpu().numpy()
                geometric = cae_outputs['geometric'][0].cpu().numpy()
                topological = cae_outputs['topological'][0].cpu().numpy()
                noise = cae_outputs['noise'][0].cpu().numpy()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🧬 Structural Descriptors**")
                    
                    # Component analysis
                    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                    
                    axes[0, 0].hist(geometric, bins=20, alpha=0.7, color='green')
                    axes[0, 0].set_title('Geometric Features\n(Lattice Parameters)')
                    
                    axes[0, 1].hist(topological, bins=20, alpha=0.7, color='red')
                    axes[0, 1].set_title('Topological Features\n(Defects & Bonds)')
                    
                    axes[1, 0].hist(noise, bins=20, alpha=0.7, color='blue')
                    axes[1, 0].set_title('Noise Estimation\n(Data Quality)')
                    
                    axes[1, 1].hist(latent, bins=30, alpha=0.7, color='purple')
                    axes[1, 1].set_title('Combined Descriptor\n(Full Latent Space)')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    st.markdown("**📊 Quantitative Analysis**")
                    
                    # Technical metrics for researchers
                    analysis_data = {
                        'Descriptor': ['Geometric', 'Topological', 'Noise', 'Combined'],
                        'Mean': [f"{np.mean(geometric):.4f}", f"{np.mean(topological):.4f}", 
                                f"{np.mean(noise):.4f}", f"{np.mean(latent):.4f}"],
                        'Std Dev': [f"{np.std(geometric):.4f}", f"{np.std(topological):.4f}", 
                                   f"{np.std(noise):.4f}", f"{np.std(latent):.4f}"],
                        'Range': [f"{np.ptp(geometric):.4f}", f"{np.ptp(topological):.4f}", 
                                 f"{np.ptp(noise):.4f}", f"{np.ptp(latent):.4f}"]
                    }
                    
                    df_analysis = pd.DataFrame(analysis_data)
                    st.dataframe(df_analysis, use_container_width=True)
                    
                    # Export option for researchers
                    csv = df_analysis.to_csv(index=False)
                    st.download_button(
                        label="📥 Export Analysis Data",
                        data=csv,
                        file_name=f"vantascope_analysis_{uploaded_file.name.split('.')[0]}.csv",
                        mime="text/csv"
                    )
                    
                    # Model interpretation
                    st.markdown("### 🎓 Scientific Interpretation")
                    st.write(f"""
                    **Geometric Score**: {np.mean(np.abs(geometric)):.3f} - Indicates lattice regularity
                    
                    **Topological Score**: {np.mean(np.abs(topological)):.3f} - Reflects defect density
                    
                    **Noise Level**: {np.mean(np.abs(noise)):.3f} - Data quality assessment
                    
                    **Overall Confidence**: {100 - np.mean(np.abs(latent))*10:.1f}% - Model certainty
                    """)
    
    else:
        # Welcome screen
        st.markdown("""
        ## 👋 Welcome to VantaScope 5090 Pro
        
        **AI-Powered Digital Twin for Materials Science**
        
        ### 🚀 Advanced Capabilities:
        - **🧠 DINOv2 Foundation Model**: 768-dimensional feature extraction
        - **🕸️ Fuzzy Graph Networks**: Material phase identification
        - **🎯 Intelligent Cropping**: Automatic removal of legends and axes
        - **🔬 Materials Classification**: Lattice, defects, grain boundaries
        
        ### 📊 What You'll Discover:
        - **Material composition percentages**
        - **Structural quality assessment** 
        - **Bond connectivity analysis**
        - **Property predictions based on structure**
        
        ### 📁 Instructions:
        1. Upload your microscopy image (STEM, AFM, etc.)
        2. VantaScope automatically crops out non-data regions
        3. View comprehensive materials analysis
        4. Download high-resolution network visualizations
        
        ### 🎯 Perfect for:
        - **Materials characterization**
        - **Quality control analysis** 
        - **Research documentation**
        - **Student education**
        """)

if __name__ == "__main__":
    main()
