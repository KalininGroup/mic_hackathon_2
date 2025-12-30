"""
Circle drawer for Plotly figures.

Draws circles as scatter traces using parametric equations.
"""

import numpy as np
from typing import Any, Dict, Optional

import plotly.graph_objects as go

from nanorange.image_analyzer.plotter.shape_plotter.base_drawer import BaseShapeDrawer, DrawStyle


class CircleDrawer(BaseShapeDrawer):
    """
    Draws circles on Plotly figures.
    
    Uses parametric representation for smooth circular shapes.
    """
    
    def __init__(
        self, 
        style: Optional[DrawStyle] = None,
        num_points: int = 100
    ):
        """
        Initialize circle drawer.
        
        Args:
            style: Drawing style configuration
            num_points: Number of points for circle approximation
        """
        super().__init__(style)
        self.num_points = num_points
    
    @property
    def shape_type(self) -> str:
        """Return 'Circle' as the shape type."""
        return "Circle"
    
    def draw(
        self, 
        fig: go.Figure, 
        x: float, 
        y: float,
        radius: float,
        **params
    ) -> go.Figure:
        """
        Draw a circle on the figure.
        
        Args:
            fig: Plotly figure to draw on
            x: X coordinate of center
            y: Y coordinate of center
            radius: Circle radius
            **params: Additional parameters (ignored)
            
        Returns:
            Updated figure with circle drawn
        """
        theta = np.linspace(0, 2 * np.pi, self.num_points)
        x_points = x + radius * np.cos(theta)
        y_points = y + radius * np.sin(theta)
        
        fig.add_trace(go.Scatter(
            x=x_points,
            y=y_points,
            mode="lines",
            line=self.style.to_line_dict(),
            opacity=self.style.opacity,
            fill="toself" if self.style.fill_color else None,
            fillcolor=self.style.fill_color,
            hoverinfo="text",
            hovertext=f"Circle: center=({x:.1f}, {y:.1f}), r={radius:.1f}",
            showlegend=False
        ))
        
        return fig
    
    def from_csv_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a CSV row into circle parameters.
        
        Expected columns: X, Y, R_or_A (radius)
        
        Args:
            row: Dictionary from CSV row
            
        Returns:
            Dictionary with x, y, radius keys
        """
        return {
            "x": float(row["X"]),
            "y": float(row["Y"]),
            "radius": float(row["R_or_A"])
        }
