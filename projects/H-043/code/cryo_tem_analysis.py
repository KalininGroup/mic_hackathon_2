"""
Cryo-TEM Analysis Tool

A simple function to run the complete Cryo-TEM image analysis pipeline.
"""

from nanorange.pipelines.cryo_tem_pipeline import CryoTEMPipeline
from nanorange.settings import (
    SHAPE_FITTING_MIN_OBJECT_SIZE,
    SHAPE_FITTING_N_COLORS,
    SHAPE_FITTING_MERGE_DIST_THRESH,
    SHAPE_FITTING_MERGE_SIZE_THRESH,
    SHAPE_FITTING_CIRCLE_REL_ERROR_THRESH,
    SHAPE_FITTING_MIN_RADIUS,
    SHAPE_FITTING_MAX_RADIUS,
    SHAPE_FITTING_MIN_CONTOUR_POINTS,
    SHAPE_FITTING_MIN_ARC_FRACTION,
    SHAPE_FITTING_ELLIPSE_ERROR_THRESH,
)


def analyze_cryo_tem(
    image_path: str,
    rows: int = 1,
    cols: int = 1,
    min_object_size: int = SHAPE_FITTING_MIN_OBJECT_SIZE,
    n_colors: int = SHAPE_FITTING_N_COLORS,
    merge_dist_thresh: float = SHAPE_FITTING_MERGE_DIST_THRESH,
    merge_size_thresh: float = SHAPE_FITTING_MERGE_SIZE_THRESH,
    circle_rel_error_thresh: float = SHAPE_FITTING_CIRCLE_REL_ERROR_THRESH,
    min_radius: float = SHAPE_FITTING_MIN_RADIUS,
    max_radius: float = SHAPE_FITTING_MAX_RADIUS,
    min_contour_points: int = SHAPE_FITTING_MIN_CONTOUR_POINTS,
    min_arc_fraction: float = SHAPE_FITTING_MIN_ARC_FRACTION,
    ellipse_error_thresh: float = SHAPE_FITTING_ELLIPSE_ERROR_THRESH,
) -> dict:
    """
    Analyze a Cryo-TEM image using the full processing pipeline.
    
    This function runs the complete Cryo-TEM analysis workflow:
        1. Split image into tiles (if rows/cols > 1)
        2. Enhance contrast using AI
        3. Apply thresholding
        4. Colorize boundaries using AI
        5. Fit shapes (circles/ellipses)
        6. Generate visualizations
        7. Plot size distribution histogram
    
    Args:
        image_path: Path to the input Cryo-TEM image
        rows: Number of rows to split the image into
        cols: Number of columns to split the image into
        min_object_size: Minimum object size in pixels to retain
        n_colors: Number of color clusters for segmentation
        merge_dist_thresh: Max distance between centers to merge shapes
        merge_size_thresh: Max size difference to merge shapes
        circle_rel_error_thresh: Max relative error for circle fitting
        min_radius: Minimum acceptable radius
        max_radius: Maximum acceptable radius
        min_contour_points: Minimum points needed for fitting
        min_arc_fraction: Minimum arc coverage fraction
        ellipse_error_thresh: Max error for ellipse fitting
    
    Returns:
        Dictionary containing:
            - success: Boolean indicating if the pipeline completed successfully
            - tile_paths: List of paths to tile images
            - enhanced_paths: List of paths to enhanced images
            - thresholded_paths: List of paths to thresholded images
            - colorized_paths: List of paths to colorized images
            - final_thresholded_path: Path to final thresholded image
            - final_colorized_path: Path to final colorized image
            - final_original_path: Path to original image copy
            - csv_path: Path to detected shapes CSV file
            - visualization_paths: Dictionary of visualization file paths
            - size_distribution_paths: Dictionary of size distribution plot paths
    
    Example:
        >>> results = analyze_cryo_tem(
        ...     image_path="images/sample.jpg",
        ...     rows=2,
        ...     cols=2
        ... )
        >>> if results["success"]:
        ...     print(f"Found shapes saved to: {results['csv_path']}")
    """
    pipeline = CryoTEMPipeline()
    
    return pipeline.run(
        image_path=image_path,
        rows=rows,
        cols=cols,
        min_object_size=min_object_size,
        n_colors=n_colors,
        merge_dist_thresh=merge_dist_thresh,
        merge_size_thresh=merge_size_thresh,
        circle_rel_error_thresh=circle_rel_error_thresh,
        min_radius=min_radius,
        max_radius=max_radius,
        min_contour_points=min_contour_points,
        min_arc_fraction=min_arc_fraction,
        ellipse_error_thresh=ellipse_error_thresh,
    )


# --------- TESTING ---------
if __name__ == "__main__":
    image_path = "/Users/mjraei/Desktop/Projects/NanORange-dev/examples/data/Synergy-2-31.jpg"
    results = analyze_cryo_tem(image_path=image_path)
    print(results)