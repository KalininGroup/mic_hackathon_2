"""
Scientific Visualization Components for VantaScope
Professional-grade plotting for national laboratory standards
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from PIL import Image
import cv2

class ScientificVisualizer:
    """Professional scientific visualization for materials analysis."""
    
    def __init__(self):
        # VantaScope color palette
        self.colors = {
            'background': '#0E1117',
            'surface': '#1A1C24', 
            'primary': '#00A3FF',
            'lattice': '#00D4AA',
            'defect_vacancy': '#FF4B4B',
            'defect_grain_boundary': '#FFD700',
            'defect_interstitial': '#FF6B35',
            'text': '#FFFFFF',
            'text_secondary': '#888888'
        }
        
        # Professional plot template
        self.template = {
            'layout': {
                'paper_bgcolor': self.colors['background'],
                'plot_bgcolor': self.colors['surface'],
                'font': {'color': self.colors['text'], 'family': 'Inter, sans-serif'},
                'colorway': [self.colors['primary'], self.colors['lattice'], 
                           self.colors['defect_vacancy'], self.colors['defect_grain_boundary']],
                'margin': dict(l=40, r=40, t=40, b=40)
            }
        }
    
    def create_reconstruction_comparison(self, original_image, reconstruction_tensor):
        """Create side-by-side original vs reconstruction comparison."""
        
        # Convert reconstruction tensor to numpy
        if hasattr(reconstruction_tensor, 'cpu'):
            recon_array = reconstruction_tensor[0, 0].cpu().numpy()
        else:
            recon_array = reconstruction_tensor[0, 0]
        
        # Convert original to numpy if needed
        if isinstance(original_image, Image.Image):
            orig_array = np.array(original_image.convert('L'))
        else:
            orig_array = original_image
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Original Sample', 'AI Digital Twin'),
            horizontal_spacing=0.05
        )
        
        # Original image
        fig.add_trace(
            go.Heatmap(
                z=orig_array,
                colorscale='Greys',
                showscale=False,
                hovertemplate='x: %{x}<br>y: %{y}<br>Intensity: %{z:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Reconstruction
        fig.add_trace(
            go.Heatmap(
                z=recon_array,
                colorscale='Greys', 
                showscale=False,
                hovertemplate='x: %{x}<br>y: %{y}<br>Intensity: %{z:.3f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            **self.template['layout'],
            title='Digital Twin Reconstruction',
            height=400
        )
        
        # Remove axis ticks for clean look
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        
        return fig
    
    def create_defect_overlay(self, original_image, patch_defect_probs, confidence_threshold=0.5):
        """Create defect classification overlay on original image."""
        
        # Convert image to numpy
        if isinstance(original_image, Image.Image):
            img_array = np.array(original_image.convert('L'))
        else:
            img_array = original_image
        
        # Get defect predictions
        if hasattr(patch_defect_probs, 'cpu'):
            defect_probs = patch_defect_probs[0].cpu().numpy()
        else:
            defect_probs = patch_defect_probs[0]
        
        # Create figure
        fig = go.Figure()
        
        # Add base image
        fig.add_trace(
            go.Heatmap(
                z=img_array,
                colorscale='Greys',
                opacity=0.7,
                showscale=False,
                hoverinfo='skip'
            )
        )
        
        # Create overlay grid
        patch_grid_size = int(np.sqrt(defect_probs.shape[0]))  # Should be 36x36 = 1296
        h, w = img_array.shape
        
        defect_classes = np.argmax(defect_probs, axis=1)
        max_probs = np.max(defect_probs, axis=1)
        
        # Reshape to grid
        defect_grid = defect_classes.reshape(patch_grid_size, patch_grid_size)
        prob_grid = max_probs.reshape(patch_grid_size, patch_grid_size)
        
        # Create overlay annotations
        patch_h = h // patch_grid_size
        patch_w = w // patch_grid_size
        
        defect_colors = ['rgba(0,0,0,0)', 'rgba(255,75,75,0.6)', 
                        'rgba(255,107,53,0.6)', 'rgba(255,215,0,0.6)']
        defect_names = ['Perfect', 'Vacancy', 'Interstitial', 'Grain Boundary']
        
        # Add defect markers
        for i in range(patch_grid_size):
            for j in range(patch_grid_size):
                if prob_grid[i, j] > confidence_threshold and defect_grid[i, j] > 0:
                    y_center = i * patch_h + patch_h // 2
                    x_center = j * patch_w + patch_w // 2
                    
                    fig.add_shape(
                        type="circle",
                        x0=x_center - 8, y0=y_center - 8,
                        x1=x_center + 8, y1=y_center + 8,
                        fillcolor=defect_colors[defect_grid[i, j]],
                        line=dict(color=defect_colors[defect_grid[i, j]], width=2)
                    )
        
        # Update layout
        fig.update_layout(
            **self.template['layout'],
            title='Defect Classification Overlay',
            height=500,
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False)
        )
        
        return fig
    
    def create_attention_overlay(self, original_image, attention_maps):
        """Create attention heatmap overlay."""
        
        # Convert image to numpy
        if isinstance(original_image, Image.Image):
            img_array = np.array(original_image.convert('L'))
        else:
            img_array = original_image
        
        # Get attention map
        if hasattr(attention_maps, 'cpu'):
            attention = attention_maps[0, 0].cpu().numpy()
        else:
            attention = attention_maps[0, 0]
        
        # Resize attention to match image if needed
        if attention.shape != img_array.shape:
            attention = cv2.resize(attention, (img_array.shape[1], img_array.shape[0]))
        
        # Create figure with overlaid heatmaps
        fig = go.Figure()
        
        # Base image
        fig.add_trace(
            go.Heatmap(
                z=img_array,
                colorscale='Greys',
                opacity=0.6,
                showscale=False,
                hoverinfo='skip'
            )
        )
        
        # Attention overlay
        fig.add_trace(
            go.Heatmap(
                z=attention,
                colorscale='Blues',
                opacity=0.5,
                showscale=True,
                colorbar=dict(title="Attention", titleside="right"),
                hovertemplate='Attention: %{z:.4f}<extra></extra>'
            )
        )
        
        # Update layout
        fig.update_layout(
            **self.template['layout'],
            title='AI Attention Heatmap',
            height=500,
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False)
        )
        
        return fig
    
    def create_simple_image_view(self, image):
        """Create simple image view."""
        
        if isinstance(image, Image.Image):
            img_array = np.array(image.convert('L'))
        else:
            img_array = image
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Heatmap(
                z=img_array,
                colorscale='Greys',
                showscale=False,
                hovertemplate='x: %{x}<br>y: %{y}<br>Intensity: %{z:.3f}<extra></extra>'
            )
        )
        
        fig.update_layout(
            **self.template['layout'],
            title='Original Sample',
            height=500,
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False)
        )
        
        return fig
    
    def create_disentanglement_radar(self, geometric, topological, disorder):
        """Create radar chart for disentangled latent analysis."""
        
        # Calculate summary statistics for each latent dimension
        geo_stats = {
            'Magnitude': np.linalg.norm(geometric),
            'Variance': np.var(geometric),
            'Coherence': 1.0 - np.std(geometric) / (np.mean(np.abs(geometric)) + 1e-8)
        }
        
        topo_stats = {
            'Complexity': np.linalg.norm(topological),
            'Variance': np.var(topological), 
            'Coherence': 1.0 - np.std(topological) / (np.mean(np.abs(topological)) + 1e-8)
        }
        
        disorder_stats = {
            'Entropy': np.linalg.norm(disorder),
            'Variance': np.var(disorder),
            'Coherence': 1.0 - np.std(disorder) / (np.mean(np.abs(disorder)) + 1e-8)
        }
        
        # Normalize values for radar chart (0-1 scale)
        all_values = [geo_stats['Magnitude'], topo_stats['Complexity'], disorder_stats['Entropy']]
        max_val = max(all_values) + 1e-8
        
        categories = ['Geometric<br>Structure', 'Topological<br>Defects', 'Disorder<br>Entropy']
        values = [
            geo_stats['Magnitude'] / max_val,
            topo_stats['Complexity'] / max_val, 
            disorder_stats['Entropy'] / max_val
        ]
        
        # Close the radar chart
        values += values[:1]
        categories += categories[:1]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor=f'rgba(0, 163, 255, 0.2)',
            line=dict(color=self.colors['primary'], width=3),
            marker=dict(size=8, color=self.colors['primary']),
            hovertemplate='%{theta}<br>Value: %{r:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            **self.template['layout'],
            polar=dict(
                bgcolor=self.colors['surface'],
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(color=self.colors['text_secondary'], size=10),
                    gridcolor=self.colors['text_secondary']
                ),
                angularaxis=dict(
                    tickfont=dict(color=self.colors['text'], size=12),
                    gridcolor=self.colors['text_secondary']
                )
            ),
            title='Structural Fingerprint',
            height=300,
            showlegend=False
        )
        
        return fig
