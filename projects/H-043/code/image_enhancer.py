"""
Example: Image Enhancement Pipeline

This example demonstrates a 3-step image enhancement workflow:
    1. Split image into tiles
    2. Enhance contrast using AI
    3. Apply thresholding

Usage:
    python image_enhancer.py <image_path> [--rows ROWS] [--cols COLS] [--output OUTPUT_DIR]

Example:
    python image_enhancer.py sample.png --rows 3 --cols 3 --output results
"""

import os
import sys
import argparse
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from nanorange.image_analyzer.tools.image_splitter import ImageSplitter
from nanorange.image_analyzer.tools.image_thresholder import ImageThresholder
from nanorange.image_analyzer.analyzer_agent.contrast_enhancer.agent import ContrastEnhancerAgent


class ImageEnhancerPipeline:
    """
    Pipeline for image enhancement with 3 steps:
        1. Split image into tiles
        2. Enhance contrast using AI
        3. Apply thresholding
    """
    
    def __init__(self, output_base_dir: str = "output") -> None:
        """
        Initialize the image enhancer pipeline.
        
        Args:
            output_base_dir: Base directory for all pipeline outputs
        """
        self.output_base_dir = output_base_dir
        self.contrast_agent = ContrastEnhancerAgent()
        
        self.tiles_dir = os.path.join(output_base_dir, "1_tiles")
        self.enhanced_dir = os.path.join(output_base_dir, "2_enhanced")
        self.thresholded_dir = os.path.join(output_base_dir, "3_thresholded")
    
    def run(self, image_path: str, rows: int, cols: int) -> dict:
        """
        Run the image enhancement pipeline.
        
        Args:
            image_path: Path to the input image
            rows: Number of rows to split the image into
            cols: Number of columns to split the image into
            
        Returns:
            Dictionary containing paths from each step and success status
        """
        print("\n" + "=" * 60)
        print(" IMAGE ENHANCER PIPELINE STARTED")
        print("=" * 60)
        
        results = {
            "success": False,
            "tile_paths": [],
            "enhanced_paths": [],
            "thresholded_paths": []
        }
        
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
        
        results["success"] = True
        self._print_summary(results)
        
        return results
    
    def _step_split_image(self, image_path: str, rows: int, cols: int) -> List[str]:
        """Step 1: Split the input image into tiles."""
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
        """Step 2: Enhance contrast of each tile using AI."""
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
        self.contrast_agent.print_usage_stats()
        
        return enhanced_paths if enhanced_paths else []
    
    def _step_threshold_images(self, enhanced_paths: List[str]) -> List[str]:
        """Step 3: Apply thresholding to enhanced tiles."""
        print("\n Step 3: Applying thresholding...")
        
        os.makedirs(self.thresholded_dir, exist_ok=True)
        thresholded_paths = []
        
        for enhanced_path in enhanced_paths:
            tile_name = os.path.basename(enhanced_path)
            output_path = os.path.join(self.thresholded_dir, tile_name)
            
            success = ImageThresholder.threshold_tile(enhanced_path, output_path)
            
            if success:
                thresholded_paths.append(output_path)
            else:
                print(f"   ⚠️  Failed to threshold: {tile_name}")
        
        print(f"   ✓ Thresholded {len(thresholded_paths)}/{len(enhanced_paths)} tiles")
        
        return thresholded_paths if thresholded_paths else []
    
    def _print_summary(self, results: dict) -> None:
        """Print a summary of the pipeline execution."""
        print("\n" + "=" * 60)
        print(" PIPELINE SUMMARY")
        print("=" * 60)
        print(f"   Status: {'✅ SUCCESS' if results['success'] else '❌ FAILED'}")
        print(f"   Tiles created:     {len(results['tile_paths'])}")
        print(f"   Tiles enhanced:    {len(results['enhanced_paths'])}")
        print(f"   Tiles thresholded: {len(results['thresholded_paths'])}")
        print(f"   Output directory:  {self.output_base_dir}")
        print("=" * 60 + "\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhance images through split, contrast enhancement, and thresholding"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input image"
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=2,
        help="Number of rows to split the image (default: 2)"
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=2,
        help="Number of columns to split the image (default: 2)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for results (default: output)"
    )
    
    return parser.parse_args()


def main():
    """Run the image enhancement pipeline."""
    args = parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"❌ Error: Image not found: {args.image_path}")
        sys.exit(1)
    
    pipeline = ImageEnhancerPipeline(output_base_dir=args.output)
    
    results = pipeline.run(
        image_path=args.image_path,
        rows=args.rows,
        cols=args.cols
    )
    
    if results["success"]:
        print(" Pipeline completed successfully!")
        sys.exit(0)
    else:
        print(" Pipeline failed. Check the logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
