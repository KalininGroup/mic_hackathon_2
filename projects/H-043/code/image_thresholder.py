import os
import cv2


class ImageThresholder:
    """Handles thresholding operations on images."""
    
    @staticmethod
    def threshold_tile(input_path: str, output_path: str) -> bool:
        """
        Apply Otsu thresholding to a tile.
        
        Args:
            input_path: Path to the input tile
            output_path: Path to save the thresholded tile
            
        Returns:
            True if successful, False otherwise
        """
        try:
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"⚠️  Failed to load image: {input_path}")
                return False
            
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            cv2.imwrite(output_path, binary)
            return True
            
        except Exception as e:
            print(f"❌ Error thresholding {input_path}: {str(e)}")
            return False