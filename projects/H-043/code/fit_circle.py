"""
Circle fitting using the Kasa method.

Provides circle fitting, error calculation, and merging of concentric circles.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray

from nanorange.image_analyzer.geometry_fitter.fit_shapes.base_shape import BaseShapeFitter, FitResult
from nanorange.settings import (
    SHAPE_FITTING_MERGE_DIST_THRESH,
    SHAPE_FITTING_MERGE_SIZE_THRESH,
)


@dataclass(frozen=True)
class CircleParams:
    """Parameters defining a circle."""
    xc: float
    yc: float
    r: float
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple (xc, yc, r)."""
        return (self.xc, self.yc, self.r)
    
    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float]) -> 'CircleParams':
        """Create from tuple (xc, yc, r)."""
        return cls(xc=t[0], yc=t[1], r=t[2])


class CircleFitter(BaseShapeFitter[CircleParams]):
    """
    Circle fitter using the Kasa method (algebraic least squares).
    
    The Kasa method minimizes the algebraic distance to fit a circle
    to a set of 2D points. It's fast and works well for near-complete
    circles with low noise.
    """
    
    def fit(self, x: NDArray, y: NDArray) -> CircleParams:
        """
        Fit a circle to the given points using the Kasa method.
        
        The method solves:
            (x - xc)² + (y - yc)² = r²
        
        Rearranged to linear form:
            2*xc*x + 2*yc*y + (r² - xc² - yc²) = x² + y²
        
        Args:
            x: X coordinates of points
            y: Y coordinates of points
            
        Returns:
            CircleParams with center (xc, yc) and radius r
        """
        A = np.c_[2 * x, 2 * y, np.ones(len(x))]
        b = x**2 + y**2
        
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        xc, yc, d = c
        
        r = np.sqrt(d + xc**2 + yc**2)
        
        return CircleParams(xc=float(xc), yc=float(yc), r=float(r))
    
    def compute_error(
        self, 
        x: NDArray, 
        y: NDArray, 
        params: CircleParams
    ) -> Tuple[float, float]:
        """
        Compute the relative fitting error and arc coverage.
        
        Relative error is computed as the standard deviation of 
        distances from points to the center, divided by radius.
        
        Arc fraction is the angular coverage of the contour points
        around the fitted circle.
        
        Args:
            x: X coordinates of points
            y: Y coordinates of points
            params: Fitted circle parameters
            
        Returns:
            Tuple of (relative_error, arc_fraction)
        """
        xc, yc, r = params.xc, params.yc, params.r
        
        distances = np.sqrt((x - xc)**2 + (y - yc)**2)
        
        rel_error = np.std(distances) / r if r > 0 else float('inf')
        
        angles = np.arctan2(y - yc, x - xc)
        angles = np.unwrap(angles)
        arc_fraction = (angles.max() - angles.min()) / (2 * np.pi)
        
        return float(rel_error), float(arc_fraction)
    
    def merge(
        self, 
        shapes: List[CircleParams],
        dist_thresh: float = SHAPE_FITTING_MERGE_DIST_THRESH,
        size_thresh: float = SHAPE_FITTING_MERGE_SIZE_THRESH
    ) -> List[CircleParams]:
        """
        Merge concentric or near-concentric circles.
        
        Circles are merged if their centers are within dist_thresh
        and their radii differ by less than size_thresh.
        
        Args:
            shapes: List of CircleParams to merge
            dist_thresh: Maximum distance between centers to merge
            size_thresh: Maximum radius difference to merge
            
        Returns:
            List of merged CircleParams (averaged parameters)
        """
        if len(shapes) == 0:
            return []
        
        merged: List[CircleParams] = []
        used = [False] * len(shapes)
        
        for i in range(len(shapes)):
            if used[i]:
                continue
            
            group = [shapes[i]]
            used[i] = True
            
            xc1, yc1, r1 = shapes[i].xc, shapes[i].yc, shapes[i].r
            
            for j in range(i + 1, len(shapes)):
                if used[j]:
                    continue
                
                xc2, yc2, r2 = shapes[j].xc, shapes[j].yc, shapes[j].r
                center_dist = np.hypot(xc1 - xc2, yc1 - yc2)
                radius_diff = abs(r1 - r2)
                
                if center_dist < dist_thresh and radius_diff < size_thresh:
                    used[j] = True
                    group.append(shapes[j])
            
            xc_avg = np.mean([c.xc for c in group])
            yc_avg = np.mean([c.yc for c in group])
            r_avg = np.mean([c.r for c in group])
            
            merged.append(CircleParams(
                xc=float(xc_avg),
                yc=float(yc_avg),
                r=float(r_avg)
            ))
        
        return merged
    
    def is_size_valid(self, params: CircleParams) -> bool:
        """
        Check if the circle radius is within valid bounds.
        
        Args:
            params: Circle parameters
            
        Returns:
            True if radius is within [min_radius, max_radius]
        """
        return self.min_radius < params.r < self.max_radius


def fit_circle(x: NDArray, y: NDArray) -> Tuple[float, float, float]:
    """
    Fit a circle using the Kasa method.
    
    Args:
        x: X coordinates of points
        y: Y coordinates of points
        
    Returns:
        Tuple of (xc, yc, r) - center and radius
    """
    fitter = CircleFitter()
    params = fitter.fit(x, y)
    return params.to_tuple()


def circle_fit_error(
    x: NDArray, 
    y: NDArray, 
    xc: float, 
    yc: float, 
    r: float
) -> Tuple[float, float]:
    """
    Compute circle fitting error.
    
    Args:
        x: X coordinates of points
        y: Y coordinates of points
        xc: Circle center X
        yc: Circle center Y
        r: Circle radius
        
    Returns:
        Tuple of (relative_error, arc_fraction)
    """
    fitter = CircleFitter()
    params = CircleParams(xc=xc, yc=yc, r=r)
    return fitter.compute_error(x, y, params)


def merge_circles(
    circle_list: List[Tuple[float, float, float]],
    dist_thresh: float = SHAPE_FITTING_MERGE_DIST_THRESH,
    r_thresh: float = SHAPE_FITTING_MERGE_SIZE_THRESH
) -> List[Tuple[float, float, float]]:
    """
    Merge concentric circles.
    
    Args:
        circle_list: List of (xc, yc, r) tuples
        dist_thresh: Maximum center distance to merge
        r_thresh: Maximum radius difference to merge
        
    Returns:
        List of merged (xc, yc, r) tuples
    """
    fitter = CircleFitter()
    params_list = [CircleParams.from_tuple(c) for c in circle_list]
    merged = fitter.merge(params_list, dist_thresh, r_thresh)
    return [m.to_tuple() for m in merged]

