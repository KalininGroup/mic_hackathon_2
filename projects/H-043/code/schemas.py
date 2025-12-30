"""
Pydantic schemas for configuration and data validation.
"""

from pydantic import BaseModel, Field

from nanorange.settings import (
    SHAPE_FITTING_MIN_OBJECT_SIZE,
    SHAPE_FITTING_N_COLORS,
    SHAPE_FITTING_CIRCLE_REL_ERROR_THRESH,
    SHAPE_FITTING_MIN_RADIUS,
    SHAPE_FITTING_MAX_RADIUS,
    SHAPE_FITTING_MIN_CONTOUR_POINTS,
    SHAPE_FITTING_MIN_ARC_FRACTION,
    SHAPE_FITTING_ELLIPSE_ERROR_THRESH,
    SHAPE_FITTING_MERGE_DIST_THRESH,
    SHAPE_FITTING_MERGE_SIZE_THRESH,
)


class ShapeFittingConfig(BaseModel):
    """Configuration parameters for shape fitting."""
    
    min_object_size: int = Field(
        default=SHAPE_FITTING_MIN_OBJECT_SIZE,
        description="Minimum object size in pixels"
    )
    n_colors: int = Field(
        default=SHAPE_FITTING_N_COLORS,
        description="Number of colors for segmentation"
    )
    
    circle_rel_error_thresh: float = Field(
        default=SHAPE_FITTING_CIRCLE_REL_ERROR_THRESH,
        description="Maximum relative error for circle fitting"
    )
    min_radius: float = Field(
        default=SHAPE_FITTING_MIN_RADIUS,
        description="Minimum radius for shape detection"
    )
    max_radius: float = Field(
        default=SHAPE_FITTING_MAX_RADIUS,
        description="Maximum radius for shape detection"
    )
    min_contour_points: int = Field(
        default=SHAPE_FITTING_MIN_CONTOUR_POINTS,
        description="Minimum number of contour points"
    )
    min_arc_fraction: float = Field(
        default=SHAPE_FITTING_MIN_ARC_FRACTION,
        description="Minimum arc fraction for valid shapes"
    )
    
    ellipse_error_thresh: float = Field(
        default=SHAPE_FITTING_ELLIPSE_ERROR_THRESH,
        description="Maximum error for ellipse fitting"
    )
    
    merge_dist_thresh: float = Field(
        default=SHAPE_FITTING_MERGE_DIST_THRESH,
        description="Distance threshold for merging shapes"
    )
    merge_size_thresh: float = Field(
        default=SHAPE_FITTING_MERGE_SIZE_THRESH,
        description="Size threshold for merging shapes"
    )


class SuggestedParameters(BaseModel):
    """AI-suggested parameters for Cryo-TEM analysis based on image characteristics."""
    
    min_object_size: int = Field(
        description="Minimum object size in pixels to retain. Smaller values detect finer particles."
    )
    n_colors: int = Field(
        description="Number of color clusters for segmentation. More colors for complex images."
    )
    merge_dist_thresh: float = Field(
        description="Maximum distance between shape centers to merge duplicate detections."
    )
    merge_size_thresh: float = Field(
        description="Maximum size difference between shapes to allow merging."
    )
    circle_rel_error_thresh: float = Field(
        description="Maximum relative error for circle fitting. Higher for irregular shapes."
    )
    min_radius: float = Field(
        description="Minimum acceptable radius in pixels."
    )
    max_radius: float = Field(
        description="Maximum acceptable radius in pixels."
    )
    min_contour_points: int = Field(
        description="Minimum contour points required for shape fitting."
    )
    min_arc_fraction: float = Field(
        description="Minimum arc coverage fraction (0.0-1.0) for valid shapes."
    )
    ellipse_error_thresh: float = Field(
        description="Maximum error for ellipse fitting."
    )
    reasoning: str = Field(
        description="Brief explanation of why these parameters were chosen based on image analysis."
    )
