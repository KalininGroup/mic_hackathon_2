"""
Ellipse drawer for Plotly figures.

Draws ellipses as scatter traces using parametric equations with rotation.
"""

import numpy as np
from typing import Any, Dict, Optional

import plotly.graph_objects as go

from nanorange.image_analyzer.plotter.shape_plotter.base_drawer import BaseShapeDrawer, DrawStyle


class EllipseDrawer(BaseShapeDrawer):
    """
    Draws ellipses on Plotly figures.
    
    Uses parametric representation with rotation support.
    """
    
    def __init__(
        self, 
        style: Optional[DrawStyle] = None,
        num_points: int = 100
    ):
        """
        Initialize ellipse drawer.
        
        Args:
            style: Drawing style configuration
            num_points: Number of points for ellipse approximation
        """
        super().__init__(style)
        self.num_points = num_points
    
    @property
    def shape_type(self) -> str:
        """Return 'Ellipse' as the shape type."""
        return "Ellipse"
    
    def draw(
        self, 
        fig: go.Figure, 
        x: float, 
        y: float,
        semi_major: float,
        semi_minor: float,
        theta: float = 0.0,
        **params
    ) -> go.Figure:
        """
        Draw an ellipse on the figure.
        
        Args:
            fig: Plotly figure to draw on
            x: X coordinate of center
            y: Y coordinate of center
            semi_major: Semi-major axis length (a)
            semi_minor: Semi-minor axis length (b)
            theta: Rotation angle in radians
            **params: Additional parameters (ignored)
            
        Returns:
            Updated figure with ellipse drawn
        """
        t = np.linspace(0, 2 * np.pi, self.num_points)
        
        x_ellipse = semi_major * np.cos(t)
        y_ellipse = semi_minor * np.sin(t)
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        x_rotated = x_ellipse * cos_theta - y_ellipse * sin_theta
        y_rotated = x_ellipse * sin_theta + y_ellipse * cos_theta
        
        x_points = x + x_rotated
        y_points = y + y_rotated
        
        fig.add_trace(go.Scatter(
            x=x_points,
            y=y_points,
            mode="lines",
            line=self.style.to_line_dict(),
            opacity=self.style.opacity,
            fill="toself" if self.style.fill_color else None,
            fillcolor=self.style.fill_color,
            hoverinfo="text",
            hovertext=(
                f"Ellipse: center=({x:.1f}, {y:.1f}), "
                f"a={semi_major:.1f}, b={semi_minor:.1f}, θ={np.degrees(theta):.1f}°"
            ),
            showlegend=False
        ))
        
        return fig
    
    def from_csv_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a CSV row into ellipse parameters.
        
        Expected columns: X, Y, R_or_A (semi-major), B (semi-minor), Theta
        
        Args:
            row: Dictionary from CSV row
            
        Returns:
            Dictionary with x, y, semi_major, semi_minor, theta keys
        """
        return {
            "x": float(row["X"]),
            "y": float(row["Y"]),
            "semi_major": float(row["R_or_A"]),
            "semi_minor": float(row["B"]),
            "theta": float(row["Theta"])
        }
