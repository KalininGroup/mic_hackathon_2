"""
Example: Shape Fitter Pipeline

This example extends the BoundaryColorizerPipeline with shape fitting and final image steps:
    1. Split image into tiles
    2. Enhance contrast using AI (per tile)
    3. Apply thresholding (per tile)
    4. Colorize boundaries using AI (on FULL stitched image, then split back)
    5. Fit shapes (circles/ellipses) on each colorized tile and export to CSV
    6. Save final images to shapes folder (stitch if multiple tiles)
    7. Plot fitted shapes on images (PNG + interactive HTML)
    8. Plot particle size distribution histogram

Usage:
    python shape_fitter.py <image_path> [options]

Example:
    python shape_fitter.py sample.png --rows 3 --cols 3 --output results
    python shape_fitter.py sample.png --n-colors 15 --min-radius 5 --max-radius 200
"""

import os
import sys
import shutil
import argparse
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from boundary_colorizer import BoundaryColorizerPipeline
from nanorange.image_analyzer.tools.image_stitcher import ImageStitcher
from nanorange.image_analyzer.geometry_fitter.shape_fitter import (
    ShapeFitter,
    ShapeFittingConfig,
    ShapeFittingResult
)
from nanorange.image_analyzer.geometry_fitter.fit_shapes.fit_circle import CircleParams
from nanorange.image_analyzer.geometry_fitter.fit_shapes.fit_ellipse import EllipseParams
from nanorange.image_analyzer.plotter.shape_plotter import (
    ShapePlotter, 
    PlotterConfig, 
    DrawStyle,
    calculate_tile_offsets
)
from nanorange.image_analyzer.plotter.stats_plotter import SizeDistributionPlotter


class ShapeFitterPipeline(BoundaryColorizerPipeline):
    """
    Extends BoundaryColorizerPipeline with shape fitting step.
    
    Steps:
        1. Split image into tiles (inherited)
        2. Enhance contrast using AI (inherited, per tile)
        3. Apply thresholding (inherited, per tile)
        4. Colorize boundaries using AI (inherited, on full image then split)
        5. Fit shapes (circles/ellipses) on each colorized tile and export to CSV
        6. Save final images to shapes folder (stitch if multiple tiles)
        7. Plot fitted shapes on images (PNG + interactive HTML)
        8. Plot particle size distribution histogram
    """
    
    def __init__(self, output_base_dir: str = "output") -> None:
        """Initialize the shape fitter pipeline."""
        super().__init__(output_base_dir)
        self.shapes_dir = os.path.join(output_base_dir, "5_shapes")
    
    def run(
        self,
        image_path: str,
        rows: int,
        cols: int,
        min_object_size: int = 30,
        n_colors: int = 10,
        merge_dist_thresh: float = 15.0,
        merge_size_thresh: float = 15.0,
        circle_rel_error_thresh: float = 0.20,
        min_radius: float = 0.5,
        max_radius: float = 400.0,
        min_contour_points: int = 25,
        min_arc_fraction: float = 0.5,
        ellipse_error_thresh: float = 0.05
    ) -> dict:
        """
        Run the pipeline with shape fitting.
        
        Args:
            image_path: Path to the input image
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
            Dictionary containing paths from each step and success status
        """
        print("\n" + "=" * 60)
        print("🔬 SHAPE FITTER PIPELINE STARTED")
        print("=" * 60)
        
        results = {
            "success": False,
            "tile_paths": [],
            "enhanced_paths": [],
            "thresholded_paths": [],
            "colorized_paths": [],
            "csv_path": None,
            "final_thresholded_path": None,
            "final_colorized_path": None,
            "final_original_path": None,
            "visualization_paths": {},
            "size_distribution_paths": {}
        }
        
        shape_config = ShapeFittingConfig(
            min_object_size=min_object_size,
            n_colors=n_colors,
            circle_rel_error_thresh=circle_rel_error_thresh,
            min_radius=min_radius,
            max_radius=max_radius,
            min_contour_points=min_contour_points,
            min_arc_fraction=min_arc_fraction,
            ellipse_error_thresh=ellipse_error_thresh,
            merge_dist_thresh=merge_dist_thresh,
            merge_size_thresh=merge_size_thresh
        )
        
        # Steps 1-3: Run parent pipeline steps
        tile_paths = self._step_split_image(image_path, rows, cols)
        if not tile_paths:
            print("❌ Pipeline failed at Step 1: Image splitting")
            return results
        results["tile_paths"] = tile_paths
        
        enhanced_paths = self._step_enhance_contrast(tile_paths)
        if not enhanced_paths:
            print("❌ Pipeline failed at Step 2: Contrast enhancement")
            return results
        results["enhanced_paths"] = enhanced_paths
        
        thresholded_paths = self._step_threshold_images(enhanced_paths)
        if not thresholded_paths:
            print("❌ Pipeline failed at Step 3: Thresholding")
            return results
        results["thresholded_paths"] = thresholded_paths
        
        # Step 4: Colorize boundaries (on full image, then split back)
        colorized_paths = self._step_colorize_boundaries(thresholded_paths, rows, cols)
        if not colorized_paths:
            print("❌ Pipeline failed at Step 4: Boundary colorization")
            return results
        results["colorized_paths"] = colorized_paths
        
        # Step 5: Fit shapes and export CSV
        shape_results, csv_path = self._step_fit_shapes(colorized_paths, shape_config)
        if not shape_results:
            print("❌ Pipeline failed at Step 5: Shape fitting")
            return results
        results["csv_path"] = csv_path
        
        # Step 6: Save final images (stitch if multiple tiles, copy if single)
        final_thresholded, final_colorized, final_original = self._step_save_final_images(
            thresholded_paths, colorized_paths, rows, cols, image_path
        )
        results["final_thresholded_path"] = final_thresholded
        results["final_colorized_path"] = final_colorized
        results["final_original_path"] = final_original
        
        # Step 7: Plot shapes on images
        if csv_path:
            visualization_paths = self._step_plot_shapes()
            results["visualization_paths"] = visualization_paths
        
        # Step 8: Plot size distribution
        if csv_path:
            size_distribution_paths = self._step_plot_size_distribution()
            results["size_distribution_paths"] = size_distribution_paths
        
        results["success"] = True
        self._print_summary(results)
        
        return results
    
    def _step_fit_shapes(
        self, 
        colorized_paths: List[str],
        config: ShapeFittingConfig
    ) -> tuple[List[ShapeFittingResult], Optional[str]]:
        """
        Step 5: Fit shapes to colorized tiles and export results.
        
        Args:
            colorized_paths: List of paths to colorized tile images
            config: Shape fitting configuration
            
        Returns:
            Tuple of (list of ShapeFittingResult, path to combined CSV)
        """
        print("\n📐 Step 5: Fitting shapes...")
        
        os.makedirs(self.shapes_dir, exist_ok=True)
        
        shape_fitter = ShapeFitter(config)
        all_results: List[ShapeFittingResult] = []
        total_circles = 0
        total_ellipses = 0
        
        for i, colorized_path in enumerate(colorized_paths):
            tile_name = os.path.basename(colorized_path)
            print(f"   Processing tile {i + 1}/{len(colorized_paths)}: {tile_name}")
            
            try:
                result = shape_fitter.fit_shapes(colorized_path)
                all_results.append(result)
                
                total_circles += len(result.circles)
                total_ellipses += len(result.ellipses)
                
                print(f"      Found {len(result.circles)} circles, "
                      f"{len(result.ellipses)} ellipses")
                
            except Exception as e:
                print(f"   ⚠️  Failed to fit shapes: {tile_name} - {str(e)}")
                all_results.append(ShapeFittingResult())
        
        csv_path = os.path.join(self.shapes_dir, "detected_shapes.csv")
        combined_result = self._combine_results(all_results, colorized_paths)
        
        if ShapeFitter.save_results_csv(combined_result, csv_path):
            print(f"   ✓ CSV saved to: {csv_path}")
        else:
            csv_path = None
        
        print(f"   ✓ Total: {total_circles} circles, {total_ellipses} ellipses")
        
        return all_results, csv_path
    
    def _combine_results(
        self, 
        results: List[ShapeFittingResult],
        tile_paths: List[str]
    ) -> ShapeFittingResult:
        """
        Combine shape fitting results from all tiles with coordinate offsets.
        
        Each tile's shapes have coordinates relative to that tile. This method
        adjusts the coordinates to the global image coordinate system by adding
        the appropriate offset based on each tile's position in the grid.
        
        Args:
            results: List of results from each tile
            tile_paths: List of tile paths (for coordinate offset calculation)
            
        Returns:
            Combined ShapeFittingResult with globally adjusted coordinates
        """
        combined = ShapeFittingResult()
        
        if not tile_paths:
            return combined
        
        tile_offsets = calculate_tile_offsets(tile_paths)
        
        for result, tile_path in zip(results, tile_paths):
            offset_x, offset_y = tile_offsets.get(tile_path, (0, 0))
            
            for circle in result.circles:
                adjusted_circle = CircleParams(
                    xc=circle.xc + offset_x,
                    yc=circle.yc + offset_y,
                    r=circle.r
                )
                combined.circles.append(adjusted_circle)
            
            for ellipse in result.ellipses:
                adjusted_ellipse = EllipseParams(
                    xc=ellipse.xc + offset_x,
                    yc=ellipse.yc + offset_y,
                    a=ellipse.a,
                    b=ellipse.b,
                    theta=ellipse.theta
                )
                combined.ellipses.append(adjusted_ellipse)
        
        return combined
    
    def _step_save_final_images(
        self,
        thresholded_paths: List[str],
        colorized_paths: List[str],
        rows: int,
        cols: int,
        original_image_path: Optional[str] = None
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Step 6: Save final images (stitch if multiple tiles, copy if single).
        
        Args:
            thresholded_paths: List of paths to thresholded tile images
            colorized_paths: List of paths to colorized tile images
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            original_image_path: Path to the original input image
            
        Returns:
            Tuple of (final_thresholded_path, final_colorized_path, original_path)
        """
        is_single_tile = (rows == 1 and cols == 1)
        action = "Copying" if is_single_tile else "Stitching"
        print(f"\n🧩 Step 6: {action} final images...")
        
        os.makedirs(self.shapes_dir, exist_ok=True)
        
        final_thresholded_path = os.path.join(self.shapes_dir, "final_thresholded.png")
        final_colorized_path = os.path.join(self.shapes_dir, "final_colorized.png")
        final_original_path = os.path.join(self.shapes_dir, "original.png")
        
        try:
            if is_single_tile:
                shutil.copy(thresholded_paths[0], final_thresholded_path)
            else:
                ImageStitcher.stitch_tiles(
                    thresholded_paths, rows, cols, final_thresholded_path
                )
        except Exception as e:
            print(f"   ⚠️  Failed to save thresholded image: {str(e)}")
            final_thresholded_path = None
        
        try:
            if is_single_tile:
                shutil.copy(colorized_paths[0], final_colorized_path)
            else:
                ImageStitcher.stitch_tiles(
                    colorized_paths, rows, cols, final_colorized_path
                )
        except Exception as e:
            print(f"   ⚠️  Failed to save colorized image: {str(e)}")
            final_colorized_path = None
        
        if original_image_path and os.path.exists(original_image_path):
            try:
                shutil.copy(original_image_path, final_original_path)
                print(f"   ✓ Original image copied")
            except Exception as e:
                print(f"   ⚠️  Failed to copy original image: {str(e)}")
                final_original_path = None
        else:
            final_original_path = None
        
        if final_thresholded_path and final_colorized_path:
            print(f"   ✓ Final images saved to {self.shapes_dir}")
        
        return final_thresholded_path, final_colorized_path, final_original_path
    
    def _step_plot_shapes(self) -> dict:
        """
        Step 7: Plot fitted shapes on images.
        
        Creates visualizations with shapes overlaid on both thresholded
        and colorized images. Outputs PNG and interactive HTML files.
        
        Returns:
            Dictionary with paths to visualization files
        """
        print("\n🎨 Step 7: Plotting shapes on images...")
        
        config = PlotterConfig(
            circle_style=DrawStyle(line_color="lime", line_width=2.0),
            ellipse_style=DrawStyle(line_color="cyan", line_width=2.0)
        )
        
        plotter = ShapePlotter(config)
        
        try:
            visualization_paths = plotter.process_shapes_folder(self.shapes_dir)
            
            for image_type, files in visualization_paths.items():
                if isinstance(files, dict):
                    for file_type, path in files.items():
                        print(f"      {image_type}_{file_type}: {path}")
                else:
                    print(f"      {image_type}: {files}")
            
            return visualization_paths
            
        except Exception as e:
            print(f"   ⚠️  Error plotting shapes: {str(e)}")
            return {}
    
    def _step_plot_size_distribution(self) -> dict:
        """
        Step 8: Plot particle size distribution histogram.
        
        Creates a histogram showing the distribution of particle diameters
        based on the detected shapes CSV file.
        
        Returns:
            Dictionary with paths to saved plot files
        """
        print("\n📊 Step 8: Plotting size distribution...")
        
        plotter = SizeDistributionPlotter()
        
        try:
            saved_files = plotter.plot_from_shapes_folder(
                self.shapes_dir,
                csv_filename="detected_shapes_original.csv",
                output_filename="size_distribution"
            )
            
            if saved_files:
                for file_type, path in saved_files.items():
                    print(f"   ✓ size_distribution.{file_type}: {path}")
            
            return saved_files
            
        except Exception as e:
            print(f"   ⚠️  Error plotting size distribution: {str(e)}")
            return {}
    
    def _print_summary(self, results: dict) -> None:
        """Print a summary of the pipeline execution."""
        print("\n" + "=" * 60)
        print("📊 PIPELINE SUMMARY")
        print("=" * 60)
        print(f"   Status: {'✅ SUCCESS' if results['success'] else '❌ FAILED'}")
        print(f"   Tiles created:     {len(results['tile_paths'])}")
        print(f"   Tiles enhanced:    {len(results['enhanced_paths'])}")
        print(f"   Tiles thresholded: {len(results['thresholded_paths'])}")
        print(f"   Tiles colorized:   {len(results['colorized_paths'])}")
        
        if results['csv_path']:
            print(f"   CSV output:        {results['csv_path']}")
        
        if results.get('final_thresholded_path'):
            print(f"   Final thresholded: {results['final_thresholded_path']}")
        if results.get('final_colorized_path'):
            print(f"   Final colorized:   {results['final_colorized_path']}")
        if results.get('final_original_path'):
            print(f"   Final original:    {results['final_original_path']}")
        
        if results.get('visualization_paths'):
            print("\n   📊 Visualizations:")
            for img_type, files in results['visualization_paths'].items():
                for file_type, path in files.items():
                    print(f"      {img_type}_{file_type}: {os.path.basename(path)}")
        
        if results.get('size_distribution_paths'):
            print("\n   📈 Size Distribution:")
            for file_type, path in results['size_distribution_paths'].items():
                print(f"      size_distribution.{file_type}: {os.path.basename(path)}")
        
        print(f"\n   Output directory:  {self.output_base_dir}")
        print("=" * 60 + "\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process images with shape fitting (circles and ellipses)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("image_path", type=str, help="Path to the input image")
    
    parser.add_argument("--rows", type=int, default=2, help="Number of rows")
    parser.add_argument("--cols", type=int, default=2, help="Number of columns")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    
    parser.add_argument(
        "--min-object-size", type=int, default=30,
        help="Minimum object size in pixels"
    )
    parser.add_argument(
        "--n-colors", type=int, default=10,
        help="Number of color clusters for segmentation"
    )
    parser.add_argument(
        "--merge-dist-thresh", type=float, default=15.0,
        help="Max distance between centers to merge shapes"
    )
    parser.add_argument(
        "--merge-size-thresh", type=float, default=15.0,
        help="Max size difference to merge shapes"
    )
    
    parser.add_argument(
        "--circle-error-thresh", type=float, default=0.20,
        help="Max relative error for circle fitting"
    )
    parser.add_argument(
        "--min-radius", type=float, default=0.5,
        help="Minimum acceptable radius"
    )
    parser.add_argument(
        "--max-radius", type=float, default=400.0,
        help="Maximum acceptable radius"
    )
    parser.add_argument(
        "--min-contour-points", type=int, default=25,
        help="Minimum points needed for fitting"
    )
    parser.add_argument(
        "--min-arc-fraction", type=float, default=0.5,
        help="Minimum arc coverage fraction"
    )
    
    parser.add_argument(
        "--ellipse-error-thresh", type=float, default=0.05,
        help="Max error for ellipse fitting"
    )
    
    return parser.parse_args()


def main():
    """Run the shape fitter pipeline."""
    args = parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"❌ Error: Image not found: {args.image_path}")
        sys.exit(1)
    
    pipeline = ShapeFitterPipeline(output_base_dir=args.output)
    
    results = pipeline.run(
        image_path=args.image_path,
        rows=args.rows,
        cols=args.cols,
        min_object_size=args.min_object_size,
        n_colors=args.n_colors,
        merge_dist_thresh=args.merge_dist_thresh,
        merge_size_thresh=args.merge_size_thresh,
        circle_rel_error_thresh=args.circle_error_thresh,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        min_contour_points=args.min_contour_points,
        min_arc_fraction=args.min_arc_fraction,
        ellipse_error_thresh=args.ellipse_error_thresh
    )
    
    if results["success"]:
        print("✅ Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("❌ Pipeline failed. Check the logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
