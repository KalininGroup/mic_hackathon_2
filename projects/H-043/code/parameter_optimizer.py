"""
Parameter Optimizer for Cryo-TEM Analysis

Uses Gemini AI to analyze an image and suggest optimal parameters
for the shape fitting pipeline based on visual characteristics.
"""

from nanorange.image_analyzer.analyzer_agent.parameter_optimizer import (
    ParameterOptimizerAgent,
)


def optimize_parameters(image_path: str) -> dict:
    """
    Analyze a Cryo-TEM image and suggest optimal analysis parameters.
    
    This function uses AI vision to examine the image characteristics 
    (particle sizes, density, contrast, noise, etc.) and recommends 
    optimal parameters for the shape detection pipeline.
    
    Args:
        image_path: Path to the Cryo-TEM image to analyze
        
    Returns:
        Dictionary containing:
            - min_object_size: Minimum object size in pixels
            - n_colors: Number of color clusters for segmentation
            - merge_dist_thresh: Distance threshold for merging shapes
            - merge_size_thresh: Size threshold for merging shapes
            - circle_rel_error_thresh: Error threshold for circle fitting
            - min_radius: Minimum acceptable radius
            - max_radius: Maximum acceptable radius
            - min_contour_points: Minimum contour points for fitting
            - min_arc_fraction: Minimum arc coverage fraction
            - ellipse_error_thresh: Error threshold for ellipse fitting
            - reasoning: Explanation of the parameter choices
            
    Example:
        >>> params = optimize_parameters("images/sample.jpg")
        >>> print(f"Suggested min_object_size: {params['min_object_size']}")
        >>> print(f"Reasoning: {params['reasoning']}")
        
        # Use with analyze_cryo_tem:
        >>> from nanorange.agent_tools import analyze_cryo_tem
        >>> results = analyze_cryo_tem(
        ...     image_path="images/sample.jpg",
        ...     min_object_size=params['min_object_size'],
        ...     n_colors=params['n_colors'],
        ...     # ... other parameters
        ... )
    """
    optimizer = ParameterOptimizerAgent()
    suggested = optimizer.analyze(image_path)
    
    # Print AI reasoning
    print("\n🔬 Parameter Optimizer Analysis")
    print("=" * 50)
    print(f"📝 {suggested.reasoning}")
    print("=" * 50)
    print("\n📊 Suggested Parameters:")
    print(f"  min_object_size:        {suggested.min_object_size}")
    print(f"  n_colors:               {suggested.n_colors}")
    print(f"  merge_dist_thresh:      {suggested.merge_dist_thresh}")
    print(f"  merge_size_thresh:      {suggested.merge_size_thresh}")
    print(f"  circle_rel_error_thresh: {suggested.circle_rel_error_thresh}")
    print(f"  min_radius:             {suggested.min_radius}")
    print(f"  max_radius:             {suggested.max_radius}")
    print(f"  min_contour_points:     {suggested.min_contour_points}")
    print(f"  min_arc_fraction:       {suggested.min_arc_fraction}")
    print(f"  ellipse_error_thresh:   {suggested.ellipse_error_thresh}")
    print()
    
    optimizer.print_usage_stats()
    
    return suggested.model_dump()


# --------- TESTING ---------
if __name__ == "__main__":
    image_path = "/Users/mjraei/Desktop/Projects/NanORange-dev/examples/data/Synergy-2-31.jpg"
    params = optimize_parameters(image_path)
    print("\nReturned dictionary:")
    print(params)
