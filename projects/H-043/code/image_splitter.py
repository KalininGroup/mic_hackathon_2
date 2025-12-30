import os
from typing import List
import cv2


class ImageSplitter:
    """Handles splitting images into tiles for processing."""
    
    @staticmethod
    def split_image(image_path: str, rows: int, cols: int, output_dir: str) -> List[str]:
        """
        Split an image into a grid of tiles.
        
        Args:
            image_path: Path to the input image
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            output_dir: Directory to save the tiles
            
        Returns:
            List of paths to the saved tile images
            
        Raises:
            FileNotFoundError: If the input image doesn't exist
            ValueError: If rows or cols are less than 1
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if rows < 1 or cols < 1:
            raise ValueError("Rows and cols must be at least 1")
        
        os.makedirs(output_dir, exist_ok=True)
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        h, w, _ = img.shape
        tile_h = h // rows
        tile_w = w // cols
        
        tile_paths = []
        
        for i in range(rows):
            for j in range(cols):
                y0 = i * tile_h
                x0 = j * tile_w
                
                y1 = y0 + tile_h if i < rows - 1 else h
                x1 = x0 + tile_w if j < cols - 1 else w
                
                block = img[y0:y1, x0:x1]
                tile_path = os.path.join(output_dir, f"tile_{i}_{j}.png")
                cv2.imwrite(tile_path, block)
                tile_paths.append(tile_path)
        
        print(f"✅ Split image into {len(tile_paths)} tiles")
        return tile_paths

