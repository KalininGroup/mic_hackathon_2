"""
Example: Boundary Colorizer Pipeline

This example extends the ImageEnhancerPipeline with a 4th step:
    1. Split image into tiles
    2. Enhance contrast using AI (per tile)
    3. Apply thresholding (per tile)
    4. Colorize boundaries using AI (on FULL stitched image, then split back)

Note: Step 4 processes the full stitched image for better boundary detection,
then splits the colorized result back into tiles for downstream processing.

Usage:
    python boundary_colorizer.py <image_path> [--rows ROWS] [--cols COLS] [--output OUTPUT_DIR]

Example:
    python boundary_colorizer.py sample.png --rows 3 --cols 3 --output results
"""

import os
import sys
import shutil
import argparse
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from image_enhancer import ImageEnhancerPipeline
from nanorange.image_analyzer.analyzer_agent.boundary_colorizer.agent import BoundaryColorizerAgent
from nanorange.image_analyzer.tools.image_splitter import ImageSplitter
from nanorange.image_analyzer.tools.image_stitcher import ImageStitcher


class BoundaryColorizerPipeline(ImageEnhancerPipeline):
    """
    Extends ImageEnhancerPipeline with boundary colorization step.
    
    Steps:
        1. Split image into tiles (inherited)
        2. Enhance contrast using AI (inherited, per tile)
        3. Apply thresholding (inherited, per tile)
        4. Colorize boundaries using AI (on full image, then split back)
    """
    
    def __init__(self, output_base_dir: str = "output") -> None:
        """Initialize the boundary colorizer pipeline."""
        super().__init__(output_base_dir)
        self.colorizer_agent = BoundaryColorizerAgent()
        self.colorized_dir = os.path.join(output_base_dir, "4_colorized")
    
    def run(self, image_path: str, rows: int, cols: int) -> dict:
        """Run the pipeline with boundary colorization."""
        print("\n" + "=" * 60)
        print(" BOUNDARY COLORIZER PIPELINE STARTED")
        print("=" * 60)
        
        results = {
            "success": False,
            "tile_paths": [],
            "enhanced_paths": [],
            "thresholded_paths": [],
            "colorized_paths": []
        }
        
        # Steps 1-3: Run parent pipeline steps
        tile_paths = self._step_split_image(image_path, rows, cols)
        if not tile_paths:
            print("Pipeline failed at Step 1: Image splitting")
            return results
        results["tile_paths"] = tile_paths
        
        enhanced_paths = self._step_enhance_contrast(tile_paths)
        if not enhanced_paths:
            print("Pipeline failed at Step 2: Contrast enhancement")
            return results
        results["enhanced_paths"] = enhanced_paths
        
        thresholded_paths = self._step_threshold_images(enhanced_paths)
        if not thresholded_paths:
            print("Pipeline failed at Step 3: Thresholding")
            return results
        results["thresholded_paths"] = thresholded_paths
        
        # Step 4: Colorize boundaries (on full image, then split back)
        colorized_paths = self._step_colorize_boundaries(thresholded_paths, rows, cols)
        if not colorized_paths:
            print("Pipeline failed at Step 4: Boundary colorization")
            return results
        results["colorized_paths"] = colorized_paths
        
        results["success"] = True
        self._print_summary(results)
        
        return results
    
    def _step_colorize_boundaries(
        self,
        thresholded_paths: List[str],
        rows: int,
        cols: int
    ) -> List[str]:
        """
        Step 4: Colorize boundaries using AI on the FULL image.
        
        This step stitches all thresholded tiles into a full image,
        colorizes the full image, then splits it back into tiles.
        
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
            
            try:
                os.remove(full_thresholded_path)
            except OSError:
                pass
        
        return colorized_paths if colorized_paths else []
    
    def _print_summary(self, results: dict) -> None:
        """Print a summary of the pipeline execution."""
        print("\n" + "=" * 60)
        print(" PIPELINE SUMMARY")
        print("=" * 60)
        print(f"   Status: {'SUCCESS' if results['success'] else 'FAILED'}")
        print(f"   Tiles created:     {len(results['tile_paths'])}")
        print(f"   Tiles enhanced:    {len(results['enhanced_paths'])}")
        print(f"   Tiles thresholded: {len(results['thresholded_paths'])}")
        print(f"   Tiles colorized:   {len(results['colorized_paths'])}")
        print(f"   Output directory:  {self.output_base_dir}")
        print("=" * 60 + "\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process images with contrast enhancement, thresholding, and boundary colorization"
    )
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--rows", type=int, default=2, help="Number of rows (default: 2)")
    parser.add_argument("--cols", type=int, default=2, help="Number of columns (default: 2)")
    parser.add_argument("--output", type=str, default="output", help="Output directory (default: output)")
    
    return parser.parse_args()


def main():
    """Run the boundary colorizer pipeline."""
    args = parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        sys.exit(1)
    
    pipeline = BoundaryColorizerPipeline(output_base_dir=args.output)
    results = pipeline.run(image_path=args.image_path, rows=args.rows, cols=args.cols)
    
    if results["success"]:
        print(" Pipeline completed successfully!")
        sys.exit(0)
    else:
        print(" Pipeline failed. Check the logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
