"""
Base class for shape fitting algorithms.

Each shape fitter provides three core operations:
- fit: Fit a shape to a set of contour points
- error: Calculate the fit quality/error
- merge: Merge similar/overlapping shapes
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, TypeVar, Generic
import numpy as np
from numpy.typing import NDArray

from nanorange.settings import (
    SHAPE_FITTING_MIN_RADIUS,
    SHAPE_FITTING_MAX_RADIUS,
    SHAPE_FITTING_CIRCLE_REL_ERROR_THRESH,
    SHAPE_FITTING_MIN_ARC_FRACTION,
    SHAPE_FITTING_MIN_CONTOUR_POINTS,
    SHAPE_FITTING_MERGE_DIST_THRESH,
    SHAPE_FITTING_MERGE_SIZE_THRESH,
)


ShapeParams = TypeVar('ShapeParams')


@dataclass
class FitResult(Generic[ShapeParams]):
    """Result of a shape fitting operation."""
    params: ShapeParams
    error: float
    arc_fraction: float
    
    @property
    def is_valid(self) -> bool:
        """Check if the fit result is valid based on error and coverage."""
        return self.error < float('inf') and self.arc_fraction > 0


class BaseShapeFitter(ABC, Generic[ShapeParams]):
    """
    Abstract base class for shape fitting algorithms.
    
    Each implementation must define:
    - fit(): Fit the shape to contour points
    - compute_error(): Calculate fitting error
    - merge(): Merge similar shapes
    """
    
    def __init__(
        self,
        min_radius: float = SHAPE_FITTING_MIN_RADIUS,
        max_radius: float = SHAPE_FITTING_MAX_RADIUS,
        max_rel_error: float = SHAPE_FITTING_CIRCLE_REL_ERROR_THRESH,
        min_arc_fraction: float = SHAPE_FITTING_MIN_ARC_FRACTION,
        min_contour_points: int = SHAPE_FITTING_MIN_CONTOUR_POINTS
    ):
        """
        Initialize the shape fitter with validation parameters.
        
        Args:
            min_radius: Minimum acceptable radius/size
            max_radius: Maximum acceptable radius/size
            max_rel_error: Maximum relative error for valid fit
            min_arc_fraction: Minimum arc coverage fraction
            min_contour_points: Minimum points needed for fitting
        """
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.max_rel_error = max_rel_error
        self.min_arc_fraction = min_arc_fraction
        self.min_contour_points = min_contour_points
    
    @abstractmethod
    def fit(self, x: NDArray, y: NDArray) -> ShapeParams:
        """
        Fit the shape to the given contour points.
        
        Args:
            x: X coordinates of contour points
            y: Y coordinates of contour points
            
        Returns:
            Shape parameters (implementation specific)
        """
        pass
    
    @abstractmethod
    def compute_error(
        self, 
        x: NDArray, 
        y: NDArray, 
        params: ShapeParams
    ) -> Tuple[float, float]:
        """
        Compute the fitting error and arc coverage.
        
        Args:
            x: X coordinates of contour points
            y: Y coordinates of contour points
            params: Shape parameters from fit()
            
        Returns:
            Tuple of (relative_error, arc_fraction)
        """
        pass
    
    @abstractmethod
    def merge(
        self, 
        shapes: List[ShapeParams],
        dist_thresh: float = SHAPE_FITTING_MERGE_DIST_THRESH,
        size_thresh: float = SHAPE_FITTING_MERGE_SIZE_THRESH
    ) -> List[ShapeParams]:
        """
        Merge similar or overlapping shapes.
        
        Args:
            shapes: List of shape parameters
            dist_thresh: Distance threshold for merging centers
            size_thresh: Size threshold for merging
            
        Returns:
            List of merged shape parameters
        """
        pass
    
    @abstractmethod
    def is_size_valid(self, params: ShapeParams) -> bool:
        """
        Check if the shape size is within valid bounds.
        
        Args:
            params: Shape parameters
            
        Returns:
            True if size is valid
        """
        pass
    
    def fit_and_validate(
        self, 
        x: NDArray, 
        y: NDArray
    ) -> FitResult[ShapeParams] | None:
        """
        Fit shape and validate the result.
        
        Args:
            x: X coordinates of contour points
            y: Y coordinates of contour points
            
        Returns:
            FitResult if valid, None otherwise
        """
        if len(x) < self.min_contour_points:
            return None
        
        try:
            params = self.fit(x, y)
        except (np.linalg.LinAlgError, ValueError):
            return None
        
        if not self.is_size_valid(params):
            return None
        
        rel_error, arc_fraction = self.compute_error(x, y, params)
        
        if rel_error > self.max_rel_error:
            return None
        
        if arc_fraction < self.min_arc_fraction:
            return None
        
        return FitResult(
            params=params,
            error=rel_error,
            arc_fraction=arc_fraction
        )

