import os
from typing import List
from PIL import Image


class ImageStitcher:
    """Handles stitching processed tiles back into a single image."""
    
    @staticmethod
    def stitch_tiles(tile_paths: List[str], rows: int, cols: int, output_path: str) -> None:
        """
        Stitch tiles back into a single image.
        
        Args:
            tile_paths: List of paths to tile images (in row-major order)
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            output_path: Path to save the stitched image
            
        Raises:
            ValueError: If number of tiles doesn't match rows * cols
        """
        if len(tile_paths) != rows * cols:
            raise ValueError(f"Expected {rows * cols} tiles, got {len(tile_paths)}")
        
        sample = Image.open(tile_paths[0])
        tile_w, tile_h = sample.size
        
        out_w = tile_w * cols
        out_h = tile_h * rows
        stitched = Image.new("RGB", (out_w, out_h))
        
        idx = 0
        for r in range(rows):
            for c in range(cols):
                tile = Image.open(tile_paths[idx])
                stitched.paste(tile, (c * tile_w, r * tile_h))
                idx += 1
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        stitched.save(output_path)
        print(f"✅ Saved stitched image → {output_path}")