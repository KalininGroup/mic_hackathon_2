"""
Example script for visualizing detected shapes on images.

This script demonstrates how to use the ShapePlotter to draw
detected shapes (circles, ellipses) on microscopy images.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from nanorange.image_analyzer.plotter.shape_plotter import (
    ShapePlotter,
    PlotterConfig,
    DrawStyle,
    plot_shapes
)


def main():
    """Run the shape visualization example."""
    shapes_folder = os.path.join(
        os.path.dirname(__file__), "..", "..", "results", "5_shapes"
    )
    shapes_folder = os.path.abspath(shapes_folder)
    
    print("\n" + "=" * 60)
    print("SHAPE VISUALIZATION")
    print("=" * 60)
    print(f"\nProcessing folder: {shapes_folder}")
    
    # Option 1: Use the convenience function
    # results = plot_shapes(shapes_folder)
    
    # Option 2: Use the ShapePlotter class for more control
    config = PlotterConfig(
        circle_style=DrawStyle(
            line_color="lime",
            line_width=2.0,
            opacity=0.9
        ),
        ellipse_style=DrawStyle(
            line_color="cyan",
            line_width=2.0,
            opacity=0.9
        ),
        show_axis=False
    )
    
    plotter = ShapePlotter(config)
    results = plotter.process_shapes_folder(shapes_folder)
    
    print("\n" + "-" * 40)
    print("OUTPUT FILES:")
    print("-" * 40)
    
    for image_type, files in results.items():
        print(f"\n{image_type.upper()}:")
        for file_type, path in files.items():
            print(f"   {file_type}: {path}")
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60 + "\n")
    
    return results


if __name__ == "__main__":
    main()
