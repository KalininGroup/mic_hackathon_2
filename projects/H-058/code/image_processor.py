"""
Intelligent Image Processing for VantaScope - Lab-grade preprocessing
"""

import numpy as np
import cv2
from PIL import Image
import torch
from scipy import ndimage
from skimage import morphology, measure

class IntelligentImageProcessor:
    """Professional-grade image preprocessing for microscopy data."""
    
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size
        
    def process_image(self, image):
        """
        Complete intelligent preprocessing pipeline.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            torch.Tensor: Preprocessed tensor ready for model inference
        """
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Intelligent cropping to remove legends/axes
        cropped_img = self.intelligent_crop(img_array)
        
        # Normalize and resize
        processed_img = self.normalize_and_resize(cropped_img)
        
        # Convert to tensor
        tensor = torch.from_numpy(processed_img).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        return tensor
    
    def intelligent_crop(self, img_array):
        """
        Intelligently crop image to remove legends, axes, and whitespace.
        
        Args:
            img_array: 2D numpy array
            
        Returns:
            numpy.ndarray: Cropped image
        """
        try:
            # Convert to uint8 if needed
            if img_array.dtype != np.uint8:
                img_normalized = ((img_array - img_array.min()) / 
                                (img_array.max() - img_array.min()) * 255).astype(np.uint8)
            else:
                img_normalized = img_array
            
            # Edge detection to find content boundaries
            edges = cv2.Canny(img_normalized, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return img_array  # Return original if no contours found
            
            # Find the largest rectangular region (likely the main image)
            largest_area = 0
            best_rect = None
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Filter out small regions and very thin rectangles (likely axes/legends)
                min_size = min(img_array.shape) * 0.3
                aspect_ratio = max(w, h) / min(w, h)
                
                if area > largest_area and w > min_size and h > min_size and aspect_ratio < 3:
                    largest_area = area
                    best_rect = (x, y, w, h)
            
            # Crop to best rectangle with padding
            if best_rect is not None:
                x, y, w, h = best_rect
                padding = 20  # Add small padding
                
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(img_array.shape[1], x + w + padding)
                y2 = min(img_array.shape[0], y + h + padding)
                
                return img_array[y1:y2, x1:x2]
            
            return img_array
            
        except Exception as e:
            print(f"Intelligent crop failed: {e}")
            return img_array  # Return original on failure
    
    def normalize_and_resize(self, img_array):
        """
        Normalize intensity and resize to target dimensions.
        
        Args:
            img_array: 2D numpy array
            
        Returns:
            numpy.ndarray: Normalized and resized image
        """
        try:
            # Percentile-based normalization to handle outliers
            p2, p98 = np.percentile(img_array, (2, 98))
            img_clipped = np.clip(img_array, p2, p98)
            
            # Normalize to [0, 1]
            img_normalized = (img_clipped - p2) / (p98 - p2)
            
            # Resize to target size
            img_resized = cv2.resize(img_normalized, self.target_size, interpolation=cv2.INTER_CUBIC)
            
            return img_resized
            
        except Exception as e:
            print(f"Normalization failed: {e}")
            # Fallback: simple normalization and resize
            img_norm = (img_array - img_array.min()) / (img_array.max() - img_array.min())
            return cv2.resize(img_norm, self.target_size, interpolation=cv2.INTER_CUBIC)
    
    def enhance_contrast(self, img_array):
        """Apply CLAHE for better contrast."""
        try:
            # Convert to uint8 for CLAHE
            img_uint8 = (img_array * 255).astype(np.uint8)
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(img_uint8)
            
            # Convert back to float
            return enhanced.astype(np.float32) / 255.0
            
        except Exception as e:
            print(f"Contrast enhancement failed: {e}")
            return img_array
    
    def detect_scale_bar(self, img_array):
        """Detect and extract scale bar information."""
        try:
            # Look for horizontal lines (scale bars) in bottom portion
            bottom_region = img_array[-img_array.shape[0]//4:, :]
            
            # Detect horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            detected_lines = cv2.morphologyEx(bottom_region, cv2.MORPH_OPEN, horizontal_kernel)
            
            # Find scale bar candidates
            contours, _ = cv2.findContours(
                (detected_lines * 255).astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            scale_bars = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 20 and h < 10:  # Likely scale bar dimensions
                    scale_bars.append((x, y, w, h))
            
            return scale_bars
            
        except Exception as e:
            print(f"Scale bar detection failed: {e}")
            return []
