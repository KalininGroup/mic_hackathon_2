"""
Ellipse fitting using the Fitzgibbon method.

Provides ellipse fitting, error calculation, and merging of similar ellipses.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray

from nanorange.image_analyzer.geometry_fitter.fit_shapes.base_shape import BaseShapeFitter, FitResult
from nanorange.settings import (
    SHAPE_FITTING_MIN_RADIUS,
    SHAPE_FITTING_MAX_RADIUS,
    SHAPE_FITTING_ELLIPSE_ERROR_THRESH,
    SHAPE_FITTING_MIN_ARC_FRACTION,
    SHAPE_FITTING_MIN_CONTOUR_POINTS,
    SHAPE_FITTING_MERGE_DIST_THRESH,
    SHAPE_FITTING_MERGE_SIZE_THRESH,
    ELLIPSE_FITTER_MAX_ECCENTRICITY,
)


@dataclass(frozen=True)
class EllipseParams:
    """Parameters defining an ellipse."""
    xc: float
    yc: float
    a: float
    b: float
    theta: float
    
    def to_tuple(self) -> Tuple[float, float, float, float, float]:
        """Convert to tuple (xc, yc, a, b, theta)."""
        return (self.xc, self.yc, self.a, self.b, self.theta)
    
    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float, float, float]) -> 'EllipseParams':
        """Create from tuple (xc, yc, a, b, theta)."""
        return cls(xc=t[0], yc=t[1], a=t[2], b=t[3], theta=t[4])
    
    @property
    def mean_radius(self) -> float:
        """Average of semi-axes (for size comparisons)."""
        return (self.a + self.b) / 2
    
    @property
    def eccentricity(self) -> float:
        """Eccentricity of the ellipse (0 = circle, 1 = parabola)."""
        if self.a >= self.b:
            return np.sqrt(1 - (self.b / self.a) ** 2)
        return np.sqrt(1 - (self.a / self.b) ** 2)


class EllipseFitter(BaseShapeFitter[EllipseParams]):
    """
    Ellipse fitter using the Fitzgibbon method.
    
    The Fitzgibbon method fits an ellipse using a direct least squares
    approach with an ellipse-specific constraint (4AC - B² > 0).
    This ensures the result is always an ellipse, not a hyperbola.
    """
    
    def __init__(
        self,
        min_radius: float = SHAPE_FITTING_MIN_RADIUS,
        max_radius: float = SHAPE_FITTING_MAX_RADIUS,
        max_rel_error: float = SHAPE_FITTING_ELLIPSE_ERROR_THRESH,
        min_arc_fraction: float = SHAPE_FITTING_MIN_ARC_FRACTION,
        min_contour_points: int = SHAPE_FITTING_MIN_CONTOUR_POINTS,
        max_eccentricity: float = ELLIPSE_FITTER_MAX_ECCENTRICITY
    ):
        """
        Initialize the ellipse fitter.
        
        Args:
            min_radius: Minimum acceptable mean radius
            max_radius: Maximum acceptable mean radius
            max_rel_error: Maximum relative error for valid fit
            min_arc_fraction: Minimum arc coverage fraction
            min_contour_points: Minimum points needed for fitting
            max_eccentricity: Maximum acceptable eccentricity (0-1)
        """
        super().__init__(
            min_radius=min_radius,
            max_radius=max_radius,
            max_rel_error=max_rel_error,
            min_arc_fraction=min_arc_fraction,
            min_contour_points=min_contour_points
        )
        self.max_eccentricity = max_eccentricity
    
    def fit(self, x: NDArray, y: NDArray) -> EllipseParams:
        """
        Fit an ellipse to the given points using the Fitzgibbon method.
        
        The general conic equation:
            A*x² + B*x*y + C*y² + D*x + E*y + F = 0
        
        With constraint 4AC - B² > 0 for ellipse.
        
        Args:
            x: X coordinates of points
            y: Y coordinates of points
            
        Returns:
            EllipseParams with center, axes, and rotation
        """
        x = x.flatten()[:, np.newaxis]
        y = y.flatten()[:, np.newaxis]
        
        D = np.hstack([
            x * x,
            x * y,
            y * y,
            x,
            y,
            np.ones_like(x)
        ])
        
        S = np.dot(D.T, D)
        
        C = np.zeros([6, 6])
        C[0, 2] = C[2, 0] = 2
        C[1, 1] = -1
        
        try:
            eigvals, eigvecs = np.linalg.eig(np.dot(np.linalg.inv(S), C))
        except np.linalg.LinAlgError:
            raise ValueError("Cannot fit ellipse: singular scatter matrix")
        
        real_eigvals = np.real(eigvals)
        positive_idx = np.argmax(real_eigvals)
        
        a = np.real(eigvecs[:, positive_idx])
        A, B, Cc, Dd, Ee, Ff = a
        
        return self._conic_to_geometric(A, B, Cc, Dd, Ee, Ff)
    
    def _conic_to_geometric(
        self, 
        A: float, 
        B: float, 
        C: float, 
        D: float, 
        E: float, 
        F: float
    ) -> EllipseParams:
        """
        Convert conic coefficients to geometric ellipse parameters.
        
        Args:
            A, B, C, D, E, F: Conic equation coefficients
            
        Returns:
            EllipseParams with geometric representation
        """
        denom = B * B - 4 * A * C
        
        if abs(denom) < 1e-10:
            raise ValueError("Degenerate conic (not an ellipse)")
        
        xc = (2 * C * D - B * E) / denom
        yc = (2 * A * E - B * D) / denom
        
        up = 2 * (A * xc * xc + C * yc * yc + B * xc * yc - F)
        down1 = (A + C) + np.sqrt((A - C) ** 2 + B * B)
        down2 = (A + C) - np.sqrt((A - C) ** 2 + B * B)
        
        if up / down1 < 0 or up / down2 < 0:
            a_axis = np.sqrt(abs(up / down1))
            b_axis = np.sqrt(abs(up / down2))
        else:
            a_axis = np.sqrt(up / down1)
            b_axis = np.sqrt(up / down2)
        
        if a_axis < b_axis:
            a_axis, b_axis = b_axis, a_axis
        
        theta = 0.5 * np.arctan2(B, A - C)
        
        return EllipseParams(
            xc=float(xc),
            yc=float(yc),
            a=float(a_axis),
            b=float(b_axis),
            theta=float(theta)
        )
    
    def compute_error(
        self, 
        x: NDArray, 
        y: NDArray, 
        params: EllipseParams
    ) -> Tuple[float, float]:
        """
        Compute the fitting error and arc coverage.
        
        Error is computed as the standard deviation of the normalized
        algebraic distance from points to the ellipse.
        
        Args:
            x: X coordinates of points
            y: Y coordinates of points
            params: Fitted ellipse parameters
            
        Returns:
            Tuple of (relative_error, arc_fraction)
        """
        xc, yc = params.xc, params.yc
        a, b = params.a, params.b
        theta = params.theta
        
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        xr = (x - xc) * cos_t + (y - yc) * sin_t
        yr = -(x - xc) * sin_t + (y - yc) * cos_t
        
        vals = (xr / a) ** 2 + (yr / b) ** 2
        rel_error = np.std(vals - 1)
        
        angles = np.arctan2(yr / b, xr / a)
        angles = np.unwrap(angles)
        arc_fraction = (angles.max() - angles.min()) / (2 * np.pi)
        
        return float(rel_error), float(arc_fraction)
    
    def merge(
        self, 
        shapes: List[EllipseParams],
        dist_thresh: float = SHAPE_FITTING_MERGE_DIST_THRESH,
        size_thresh: float = SHAPE_FITTING_MERGE_SIZE_THRESH
    ) -> List[EllipseParams]:
        """
        Merge similar ellipses based on center distance and size.
        
        Args:
            shapes: List of EllipseParams to merge
            dist_thresh: Maximum center distance to merge
            size_thresh: Maximum mean radius difference to merge
            
        Returns:
            List of merged EllipseParams
        """
        if len(shapes) == 0:
            return []
        
        merged: List[EllipseParams] = []
        used = [False] * len(shapes)
        
        for i in range(len(shapes)):
            if used[i]:
                continue
            
            group = [shapes[i]]
            used[i] = True
            
            e1 = shapes[i]
            
            for j in range(i + 1, len(shapes)):
                if used[j]:
                    continue
                
                e2 = shapes[j]
                center_dist = np.hypot(e1.xc - e2.xc, e1.yc - e2.yc)
                size_diff = abs(e1.mean_radius - e2.mean_radius)
                
                if center_dist < dist_thresh and size_diff < size_thresh:
                    used[j] = True
                    group.append(shapes[j])
            
            merged.append(self._average_ellipses(group))
        
        return merged
    
    def _average_ellipses(self, ellipses: List[EllipseParams]) -> EllipseParams:
        """
        Compute average ellipse parameters from a group.
        
        Args:
            ellipses: List of ellipses to average
            
        Returns:
            Averaged EllipseParams
        """
        n = len(ellipses)
        
        xc_avg = sum(e.xc for e in ellipses) / n
        yc_avg = sum(e.yc for e in ellipses) / n
        a_avg = sum(e.a for e in ellipses) / n
        b_avg = sum(e.b for e in ellipses) / n
        
        sin_sum = sum(np.sin(2 * e.theta) for e in ellipses)
        cos_sum = sum(np.cos(2 * e.theta) for e in ellipses)
        theta_avg = 0.5 * np.arctan2(sin_sum, cos_sum)
        
        return EllipseParams(
            xc=float(xc_avg),
            yc=float(yc_avg),
            a=float(a_avg),
            b=float(b_avg),
            theta=float(theta_avg)
        )
    
    def is_size_valid(self, params: EllipseParams) -> bool:
        """
        Check if the ellipse size and eccentricity are valid.
        
        Args:
            params: Ellipse parameters
            
        Returns:
            True if size and eccentricity are within bounds
        """
        mean_r = params.mean_radius
        if not (self.min_radius < mean_r < self.max_radius):
            return False
        
        if params.eccentricity > self.max_eccentricity:
            return False
        
        return True


def fit_ellipse(x: NDArray, y: NDArray) -> Tuple[float, float, float, float, float]:
    """
    Fit an ellipse using the Fitzgibbon method.
    
    Args:
        x: X coordinates of points
        y: Y coordinates of points
        
    Returns:
        Tuple of (xc, yc, a, b, theta) - center, axes, and rotation
    """
    fitter = EllipseFitter()
    params = fitter.fit(x, y)
    return params.to_tuple()


def ellipse_fit_error(
    x: NDArray, 
    y: NDArray, 
    xc: float, 
    yc: float, 
    a: float, 
    b: float, 
    theta: float
) -> float:
    """
    Compute ellipse fitting error (standard deviation).
    
    Args:
        x: X coordinates of points
        y: Y coordinates of points
        xc: Ellipse center X
        yc: Ellipse center Y
        a: Semi-major axis
        b: Semi-minor axis
        theta: Rotation angle
        
    Returns:
        Relative fitting error
    """
    fitter = EllipseFitter()
    params = EllipseParams(xc=xc, yc=yc, a=a, b=b, theta=theta)
    rel_error, _ = fitter.compute_error(x, y, params)
    return rel_error


def merge_ellipses(
    ellipse_list: List[Tuple[float, float, float, float, float]],
    dist_thresh: float = SHAPE_FITTING_MERGE_DIST_THRESH,
    size_thresh: float = SHAPE_FITTING_MERGE_SIZE_THRESH
) -> List[Tuple[float, float, float, float, float]]:
    """
    Merge similar ellipses.
    
    Args:
        ellipse_list: List of (xc, yc, a, b, theta) tuples
        dist_thresh: Maximum center distance to merge
        size_thresh: Maximum mean radius difference to merge
        
    Returns:
        List of merged (xc, yc, a, b, theta) tuples
    """
    fitter = EllipseFitter()
    params_list = [EllipseParams.from_tuple(e) for e in ellipse_list]
    merged = fitter.merge(params_list, dist_thresh, size_thresh)
    return [m.to_tuple() for m in merged]

