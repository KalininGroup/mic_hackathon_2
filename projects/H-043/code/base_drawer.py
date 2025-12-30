"""
Base class for shape drawers.

Provides an abstract interface for drawing shapes on Plotly figures.
Each shape drawer must implement the draw method for rendering shapes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import plotly.graph_objects as go


@dataclass
class DrawStyle:
    """Style configuration for shape drawing."""
    line_color: str = "lime"
    line_width: float = 2.0
    fill_color: Optional[str] = None
    opacity: float = 1.0
    
    def to_line_dict(self) -> Dict[str, Any]:
        """Convert to Plotly line style dictionary."""
        return {
            "color": self.line_color,
            "width": self.line_width,
        }


class BaseShapeDrawer(ABC):
    """
    Abstract base class for shape drawers.
    
    Each implementation draws a specific shape type onto a Plotly figure.
    """
    
    def __init__(self, style: Optional[DrawStyle] = None):
        """
        Initialize the drawer with optional style.
        
        Args:
            style: Drawing style configuration
        """
        self.style = style or DrawStyle()
    
    @property
    @abstractmethod
    def shape_type(self) -> str:
        """Return the type of shape this drawer handles (e.g., 'Circle')."""
        pass
    
    @abstractmethod
    def draw(
        self, 
        fig: go.Figure, 
        x: float, 
        y: float,
        **params
    ) -> go.Figure:
        """
        Draw the shape on the figure.
        
        Args:
            fig: Plotly figure to draw on
            x: X coordinate of shape center
            y: Y coordinate of shape center
            **params: Shape-specific parameters
            
        Returns:
            Updated figure with shape drawn
        """
        pass
    
    @abstractmethod
    def from_csv_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a CSV row into shape parameters.
        
        Args:
            row: Dictionary from CSV row
            
        Returns:
            Dictionary of parameters for draw method
        """
        pass
