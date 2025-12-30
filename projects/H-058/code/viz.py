"""
Visualization utilities for VantaScope.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, List, Dict, Any
from pathlib import Path


def plot_attention_heatmap(image: np.ndarray, attention: np.ndarray, 
                          title: str = "Attention Map", save_path: Optional[Path] = None) -> plt.Figure:
    """Plot image with attention overlay."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original')
    ax1.axis('off')
    
    # Attention map
    im2 = ax2.imshow(attention, cmap='hot')
    ax2.set_title('Attention')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # Overlay
    ax3.imshow(image, cmap='gray', alpha=0.7)
    ax3.imshow(attention, cmap='hot', alpha=0.5)
    ax3.set_title('Overlay')
    ax3.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_curves(metrics: Dict[str, List[float]], save_path: Optional[Path] = None) -> plt.Figure:
    """Plot training and validation curves."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, (metric_name, values) in zip(axes, metrics.items()):
        ax.plot(values)
        ax.set_title(f'{metric_name.title()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_dataset_samples(images: List[np.ndarray], titles: Optional[List[str]] = None,
                        save_path: Optional[Path] = None) -> plt.Figure:
    """Plot a grid of dataset samples."""
    n_images = len(images)
    cols = min(5, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i, img in enumerate(images):
        ax = axes[i]
        ax.imshow(img, cmap='gray')
        if titles and i < len(titles):
            ax.set_title(titles[i])
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


class LivePlotter:
    """Real-time plotting for training monitoring."""
    
    def __init__(self):
        self.metrics = {}
        self.fig = None
        self.axes = {}
    
    def update(self, metric_name: str, value: float) -> None:
        """Update a metric value."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
        
        # Update plot if in interactive mode
        if plt.isinteractive():
            self._redraw()
    
    def _redraw(self) -> None:
        """Redraw the plots."""
        if not self.metrics:
            return
        
        if self.fig is None:
            self.fig, self.axes = plt.subplots(1, len(self.metrics), figsize=(15, 4))
            if len(self.metrics) == 1:
                self.axes = {'list': [self.axes]}
            else:
                self.axes = {'list': self.axes}
        
        for i, (metric_name, values) in enumerate(self.metrics.items()):
            ax = self.axes['list'][i]
            ax.clear()
            ax.plot(values)
            ax.set_title(metric_name)
            ax.grid(True, alpha=0.3)
        
        self.fig.canvas.draw()
        plt.pause(0.01)
