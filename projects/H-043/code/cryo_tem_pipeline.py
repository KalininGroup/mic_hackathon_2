import os
import shutil
from typing import List, Optional

from nanorange.image_analyzer.tools.image_splitter import ImageSplitter
from nanorange.image_analyzer.tools.image_stitcher import ImageStitcher
from nanorange.image_analyzer.tools.image_thresholder import ImageThresholder
from nanorange.image_analyzer.analyzer_agent.contrast_enhancer.agent import ContrastEnhancerAgent
from nanorange.image_analyzer.analyzer_agent.boundary_colorizer.agent import BoundaryColorizerAgent
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
from nanorange.image_analyzer.plotter.stats_plotter import (
    SizeDistributionPlotter,
    SizeDistributionConfig
)
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


class CryoTEMPipeline:
    """
    Pipeline for processing Cryo-TEM images.
    
    Steps:
        1. Split image into tiles
        2. Enhance contrast using AI
        3. Apply thresholding
        4. Colorize boundaries using AI
        5. Fit shapes (circles/ellipses) on each tile and export to CSV
        6. Save final images to shapes folder (stitch if multiple tiles)
        7. Plot fitted shapes on images (PNG + interactive HTML)
        8. Plot particle size distribution histogram
    """
    
    def __init__(self, output_base_dir: str = "output") -> None:
        """
        Initialize the Cryo-TEM pipeline.
        
        Args:
            output_base_dir: Base directory for all pipeline outputs
        """
        self.output_base_dir = output_base_dir
        self.contrast_agent = ContrastEnhancerAgent()
        self.colorizer_agent = BoundaryColorizerAgent()
        
        self.tiles_dir = os.path.join(output_base_dir, "1_tiles")
        self.enhanced_dir = os.path.join(output_base_dir, "2_enhanced")
        self.thresholded_dir = os.path.join(output_base_dir, "3_thresholded")
        self.colorized_dir = os.path.join(output_base_dir, "4_colorized")
        self.shapes_dir = os.path.join(output_base_dir, "5_shapes")
    
    def run(
        self,
        image_path: str,
        rows: int,
        cols: int,
        min_object_size: int = SHAPE_FITTING_MIN_OBJECT_SIZE,
        n_colors: int = SHAPE_FITTING_N_COLORS,
        merge_dist_thresh: float = SHAPE_FITTING_MERGE_DIST_THRESH,
        merge_size_thresh: float = SHAPE_FITTING_MERGE_SIZE_THRESH,
        circle_rel_error_thresh: float = SHAPE_FITTING_CIRCLE_REL_ERROR_THRESH,
        min_radius: float = SHAPE_FITTING_MIN_RADIUS,
        max_radius: float = SHAPE_FITTING_MAX_RADIUS,
        min_contour_points: int = SHAPE_FITTING_MIN_CONTOUR_POINTS,
        min_arc_fraction: float = SHAPE_FITTING_MIN_ARC_FRACTION,
        ellipse_error_thresh: float = SHAPE_FITTING_ELLIPSE_ERROR_THRESH
    ) -> dict:
        """
        Run the full Cryo-TEM processing pipeline.
        
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
        print("CRYO-TEM PIPELINE STARTED")
        print("=" * 60)
        
        results = {
            "success": False,
            "tile_paths": [],
            "enhanced_paths": [],
            "thresholded_paths": [],
            "colorized_paths": [],
            "final_thresholded_path": None,
            "final_colorized_path": None,
            "final_original_path": None,
            "csv_path": None,
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
        
        # Step 1: Split image
        tile_paths = self._step_split_image(image_path, rows, cols)
        if not tile_paths:
            print("❌ Pipeline failed at Step 1: Image splitting")
            return results
        results["tile_paths"] = tile_paths
        
        # Step 2: Enhance contrast
        enhanced_paths = self._step_enhance_contrast(tile_paths)
        if not enhanced_paths:
            print("❌ Pipeline failed at Step 2: Contrast enhancement")
            return results
        results["enhanced_paths"] = enhanced_paths
        
        # Step 3: Threshold images
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
        
        print("\n" + "=" * 60)
        print("CRYO-TEM PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        return results
    
    def _step_split_image(self, image_path: str, rows: int, cols: int) -> List[str]:
        """
        Step 1: Split the input image into tiles.
        
        Args:
            image_path: Path to the input image
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            
        Returns:
            List of tile paths, empty list if failed
        """
        print("\n Step 1: Splitting image into tiles...")
        print(f"   Grid: {rows}x{cols}")
        
        try:
            tile_paths = ImageSplitter.split_image(
                image_path=image_path,
                rows=rows,
                cols=cols,
                output_dir=self.tiles_dir
            )
            print(f"   ✓ Created {len(tile_paths)} tiles")
            return tile_paths
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return []
    
    def _step_enhance_contrast(self, tile_paths: List[str]) -> List[str]:
        """
        Step 2: Enhance contrast of each tile using AI.
        
        Args:
            tile_paths: List of paths to tile images
            
        Returns:
            List of enhanced tile paths, empty list if failed
        """
        print("\n Step 2: Enhancing contrast...")
        
        os.makedirs(self.enhanced_dir, exist_ok=True)
        enhanced_paths = []
        
        for i, tile_path in enumerate(tile_paths):
            tile_name = os.path.basename(tile_path)
            output_path = os.path.join(self.enhanced_dir, tile_name)
            
            print(f"   Processing tile {i + 1}/{len(tile_paths)}: {tile_name}")
            
            success = self.contrast_agent.process_tile(tile_path, output_path)
            
            if success:
                enhanced_paths.append(output_path)
            else:
                print(f"   ⚠️  Failed to enhance: {tile_name}")
        
        print(f"   ✓ Enhanced {len(enhanced_paths)}/{len(tile_paths)} tiles")
        
        # Print token usage
        self.contrast_agent.print_usage_stats()
        
        return enhanced_paths if enhanced_paths else []
    
    def _step_threshold_images(self, enhanced_paths: List[str]) -> List[str]:
        """
        Step 3: Apply thresholding to enhanced tiles.
        
        Args:
            enhanced_paths: List of paths to enhanced tile images
            
        Returns:
            List of thresholded tile paths, empty list if failed
        """
        print("\n Step 3: Applying thresholding...")
        
        os.makedirs(self.thresholded_dir, exist_ok=True)
        thresholded_paths = []
        
        for i, enhanced_path in enumerate(enhanced_paths):
            tile_name = os.path.basename(enhanced_path)
            output_path = os.path.join(self.thresholded_dir, tile_name)
            
            success = ImageThresholder.threshold_tile(enhanced_path, output_path)
            
            if success:
                thresholded_paths.append(output_path)
            else:
                print(f"   ⚠️  Failed to threshold: {tile_name}")
        
        print(f"   ✓ Thresholded {len(thresholded_paths)}/{len(enhanced_paths)} tiles")
        
        return thresholded_paths if thresholded_paths else []
    
    def _step_colorize_boundaries(
        self, 
        thresholded_paths: List[str],
        rows: int,
        cols: int
    ) -> List[str]:
        """
        Step 4: Colorize boundaries using AI on the FULL image.
        
        This step stitches all thresholded tiles into a full image,
        colorizes the full image, then splits it back into tiles
        for subsequent processing steps.
        
        Args:
            thresholded_paths: List of paths to thresholded tile images
            rows: Number of rows in the tile grid
            cols: Number of columns in the tile grid
            
        Returns:
            List of colorized tile paths, empty list if failed
        """
        print("\n Step 4: Colorizing boundaries (full image)...")
        
        os.makedirs(self.colorized_dir, exist_ok=True)
        
        is_single_tile = (rows == 1 and cols == 1)
        
        # Step 4a: Stitch thresholded tiles into full image (if multiple tiles)
        if is_single_tile:
            full_thresholded_path = thresholded_paths[0]
            print("   Single tile - using directly")
        else:
            full_thresholded_path = os.path.join(self.colorized_dir, "_temp_full_thresholded.png")
            print(f"   Stitching {len(thresholded_paths)} tiles into full image...")
            try:
                ImageStitcher.stitch_tiles(
                    thresholded_paths, rows, cols, full_thresholded_path
                )
            except Exception as e:
                print(f"   ❌ Failed to stitch thresholded tiles: {str(e)}")
                return []
        
        # Step 4b: Colorize the full image
        full_colorized_path = os.path.join(self.colorized_dir, "_full_colorized.png")
        print("   Colorizing full image...")
        
        success = self.colorizer_agent.process_tile(full_thresholded_path, full_colorized_path)
        
        if not success:
            print("   ❌ Failed to colorize full image")
            return []
        
        print("   ✓ Full image colorized")
        self.colorizer_agent.print_usage_stats()
        
        # Step 4c: Split colorized image back into tiles (if multiple tiles)
        if is_single_tile:
            # For single tile, just rename/copy the output
            tile_name = os.path.basename(thresholded_paths[0])
            final_output_path = os.path.join(self.colorized_dir, tile_name)
            shutil.copy(full_colorized_path, final_output_path)
            
            colorized_paths = [final_output_path]
        else:
            print(f"   Splitting colorized image back into {rows}x{cols} tiles...")
            try:
                colorized_paths = ImageSplitter.split_image(
                    image_path=full_colorized_path,
                    rows=rows,
                    cols=cols,
                    output_dir=self.colorized_dir
                )
                print(f"   ✓ Created {len(colorized_paths)} colorized tiles")
            except Exception as e:
                print(f"   ❌ Failed to split colorized image: {str(e)}")
                return []
            
            # Clean up temporary stitched thresholded image
            try:
                os.remove(full_thresholded_path)
            except OSError:
                pass
        
        return colorized_paths if colorized_paths else []
    
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
        print(f"\n Step 6: {action} final images...")
        
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
        print("\n Step 7: Plotting shapes on images...")
        
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
        print("\n Step 8: Plotting size distribution...")
        
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
        print("\n Step 5: Fitting shapes...")
        
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
    