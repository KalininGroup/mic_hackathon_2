"""
Graph Network Visualization: Overlay GAT reasoning on microscopy images.
The killer feature for expert evaluation of AI reasoning.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import networkx as nx
from typing import Dict, Tuple, Optional, List

from ..utils.logging import logger


class GraphOverlayVisualizer:
    """Visualize Fuzzy-GAT graph reasoning overlaid on microscopy images."""
    
    def __init__(self):
        # Color scheme for fuzzy categories
        self.category_colors = {
            'perfect_lattice': '#00ff00',  # Green
            'defect': '#ff0000',          # Red  
            'grain_boundary': '#ffff00',   # Yellow
            'amorphous': '#0000ff',       # Blue
            'noise': '#ff00ff'            # Magenta
        }
        
        self.category_names = ['perfect_lattice', 'defect', 'grain_boundary', 'amorphous', 'noise']
        
        logger.info("🎨 Graph overlay visualizer initialized")
    
    def create_overlay(self, 
                      image: np.ndarray,
                      gat_outputs: Dict[str, torch.Tensor],
                      patch_positions: Optional[np.ndarray] = None,
                      title: str = "GAT Reasoning Overlay") -> plt.Figure:
        """
        Create complete graph overlay visualization.
        
        Args:
            image: [H, W] microscopy image
            gat_outputs: Dictionary from Fuzzy-GAT forward pass
            patch_positions: Optional [num_patches, 2] positions
            title: Plot title
            
        Returns:
            Matplotlib figure with interactive overlay
        """
        # Extract data from GAT outputs
        fuzzy_memberships = gat_outputs['fuzzy_memberships'].cpu().numpy()
        edge_index = gat_outputs['edge_index'].cpu().numpy()
        edge_weights = gat_outputs['edge_weights'].cpu().numpy()
        attention_weights = gat_outputs['attention_weights']
        num_patches = gat_outputs['num_patches']
        
        # Handle batch dimension (take first sample)
        if len(fuzzy_memberships) > num_patches:
            fuzzy_memberships = fuzzy_memberships[:num_patches]
            # Filter edges for first sample
            edge_mask = edge_index[0] < num_patches
            edge_index = edge_index[:, edge_mask]
            edge_weights = edge_weights[edge_mask]
        
        # Generate patch positions if not provided
        if patch_positions is None:
            patch_positions = self._generate_patch_positions(num_patches, image.shape)
        
        # Create the visualization
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], width_ratios=[1, 1, 1])
        
        # Main overlay plot
        ax_main = fig.add_subplot(gs[0, :])
        self._plot_main_overlay(ax_main, image, fuzzy_memberships, edge_index, 
                              edge_weights, patch_positions, title)
        
        # Fuzzy membership distribution
        ax_fuzzy = fig.add_subplot(gs[1, 0])
        self._plot_fuzzy_distribution(ax_fuzzy, fuzzy_memberships)
        
        # Edge weight distribution
        ax_edges = fig.add_subplot(gs[1, 1])
        self._plot_edge_distribution(ax_edges, edge_weights)
        
        # Network statistics
        ax_stats = fig.add_subplot(gs[1, 2])
        self._plot_network_stats(ax_stats, edge_index, fuzzy_memberships)
        
        plt.tight_layout()
        return fig
    
    def _plot_main_overlay(self, ax, image: np.ndarray, fuzzy_memberships: np.ndarray,
                          edge_index: np.ndarray, edge_weights: np.ndarray,
                          patch_positions: np.ndarray, title: str):
        """Plot the main overlay with image + graph."""
        
        # Display microscopy image
        ax.imshow(image, cmap='gray', alpha=0.7, aspect='equal')
        
        # Get dominant fuzzy categories for each node
        dominant_categories = np.argmax(fuzzy_memberships, axis=1)
        max_memberships = np.max(fuzzy_memberships, axis=1)
        
        # Plot edges first (so they appear behind nodes)
        self._plot_edges(ax, edge_index, edge_weights, patch_positions)
        
        # Plot nodes colored by fuzzy category
        self._plot_nodes(ax, patch_positions, dominant_categories, max_memberships)
        
        # Add legend
        self._add_category_legend(ax)
        
        ax.set_title(f"{title}\nNodes: Fuzzy Categories | Edges: Attention Weights", 
                    fontsize=14, pad=20)
        ax.set_xlabel("X Position (pixels)")
        ax.set_ylabel("Y Position (pixels)")
    
    def _plot_edges(self, ax, edge_index: np.ndarray, edge_weights: np.ndarray, 
                   patch_positions: np.ndarray):
        """Plot graph edges with thickness based on weights."""
        
        # Normalize edge weights for visualization
        edge_weights_norm = np.abs(edge_weights)
        edge_weights_norm = (edge_weights_norm - edge_weights_norm.min()) / (edge_weights_norm.max() - edge_weights_norm.min() + 1e-8)
        
        # Create line segments for edges
        lines = []
        linewidths = []
        colors = []
        
        for i in range(edge_index.shape[1]):
            source_idx = edge_index[0, i]
            target_idx = edge_index[1, i]
            
            if source_idx < len(patch_positions) and target_idx < len(patch_positions):
                source_pos = patch_positions[source_idx]
                target_pos = patch_positions[target_idx]
                
                lines.append([source_pos, target_pos])
                linewidths.append(0.2 + edge_weights_norm[i] * 2.0)  # 0.2 to 2.2 width
                
                # Color by edge strength
                color_intensity = edge_weights_norm[i]
                colors.append((0.5, 0.5, 0.5, 0.3 + color_intensity * 0.7))  # Gray with varying alpha
        
        # Plot all edges
        if lines:
            line_collection = LineCollection(lines, linewidths=linewidths, colors=colors)
            ax.add_collection(line_collection)
    
    def _plot_nodes(self, ax, patch_positions: np.ndarray, dominant_categories: np.ndarray,
                   max_memberships: np.ndarray):
        """Plot graph nodes colored by fuzzy category."""
        
        for category_idx in range(5):
            # Get nodes belonging to this category
            category_mask = dominant_categories == category_idx
            
            if np.any(category_mask):
                positions = patch_positions[category_mask]
                memberships = max_memberships[category_mask]
                
                # Node size based on membership confidence
                sizes = 10 + memberships * 30  # Size range: 10-40
                
                # Plot nodes for this category
                ax.scatter(positions[:, 0], positions[:, 1],
                          c=self.category_colors[self.category_names[category_idx]],
                          s=sizes, alpha=0.8, edgecolors='black', linewidth=0.5,
                          label=f"{self.category_names[category_idx]} ({np.sum(category_mask)})")
    
    def _add_category_legend(self, ax):
        """Add legend for fuzzy categories."""
        legend = ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0),
                         framealpha=0.9, fontsize=10)
        legend.set_title("Fuzzy Categories\n(Node Count)", prop={'size': 10, 'weight': 'bold'})
    
    def _plot_fuzzy_distribution(self, ax, fuzzy_memberships: np.ndarray):
        """Plot fuzzy membership distribution."""
        
        # Calculate average membership for each category
        avg_memberships = np.mean(fuzzy_memberships, axis=0)
        
        bars = ax.bar(range(5), avg_memberships, 
                     color=[self.category_colors[name] for name in self.category_names],
                     alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel("Fuzzy Categories")
        ax.set_ylabel("Average Membership")
        ax.set_title("Fuzzy Membership Distribution", fontweight='bold')
        ax.set_xticks(range(5))
        ax.set_xticklabels([name.replace('_', '\n') for name in self.category_names], 
                          rotation=45, fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_memberships):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_edge_distribution(self, ax, edge_weights: np.ndarray):
        """Plot edge weight distribution."""
        
        ax.hist(edge_weights, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel("Edge Weight")
        ax.set_ylabel("Frequency")
        ax.set_title("Edge Weight Distribution", fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_weight = np.mean(edge_weights)
        std_weight = np.std(edge_weights)
        ax.axvline(mean_weight, color='red', linestyle='--', label=f'Mean: {mean_weight:.3f}')
        ax.axvline(mean_weight + std_weight, color='orange', linestyle=':', label=f'+1σ: {mean_weight + std_weight:.3f}')
        ax.axvline(mean_weight - std_weight, color='orange', linestyle=':', label=f'-1σ: {mean_weight - std_weight:.3f}')
        ax.legend(fontsize=8)
    
    def _plot_network_stats(self, ax, edge_index: np.ndarray, fuzzy_memberships: np.ndarray):
        """Plot network connectivity statistics."""
        
        # Calculate degree distribution
        num_nodes = len(fuzzy_memberships)
        degrees = np.bincount(edge_index.flatten(), minlength=num_nodes)
        
        # Basic statistics
        stats = {
            'Nodes': num_nodes,
            'Edges': edge_index.shape[1],
            'Avg Degree': np.mean(degrees),
            'Max Degree': np.max(degrees),
            'Density': edge_index.shape[1] / (num_nodes * (num_nodes - 1) / 2)
        }
        
        # Display as text
        ax.axis('off')
        stats_text = "Network Statistics:\n\n"
        for key, value in stats.items():
            if isinstance(value, float):
                stats_text += f"{key}: {value:.2f}\n"
            else:
                stats_text += f"{key}: {value}\n"
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax.set_title("Graph Properties", fontweight='bold')
    
    def _generate_patch_positions(self, num_patches: int, image_shape: Tuple[int, int]) -> np.ndarray:
        """Generate 2D grid positions for patches."""
        patches_per_side = int(np.sqrt(num_patches))
        
        # Scale to image dimensions
        height, width = image_shape
        y_coords = np.linspace(0, height-1, patches_per_side)
        x_coords = np.linspace(0, width-1, patches_per_side)
        
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        positions = np.stack([xx.flatten(), yy.flatten()], axis=1)
        
        return positions[:num_patches]  # Handle non-square cases
    
    def save_overlay(self, fig: plt.Figure, save_path: str, dpi: int = 300):
        """Save overlay visualization to file."""
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        logger.info(f"💾 Saved graph overlay to {save_path}")


# Factory function
def create_graph_overlay() -> GraphOverlayVisualizer:
    """Create graph overlay visualizer."""
    return GraphOverlayVisualizer()
