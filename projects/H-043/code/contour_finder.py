"""Image preprocessing utilities for contour extraction and analysis."""

from dataclasses import dataclass
from typing import List
import numpy as np
from skimage import io, color, filters, morphology, measure
from sklearn.cluster import KMeans

from nanorange.settings import CONTOUR_FINDER_MIN_OBJECT_SIZE, SHAPE_FITTING_N_COLORS


@dataclass
class PreprocessingResult:
    """Container for preprocessing results."""
    gray: np.ndarray
    binary: np.ndarray
    contours: List[np.ndarray]


@dataclass
class ColorSegmentationResult:
    """Container for color segmentation results."""
    cluster_map: np.ndarray
    cluster_centers: np.ndarray
    cluster_colors: np.ndarray
    

class ContourFinder:
    """Handles contour finding operations for image analysis."""
    
    def __init__(
        self,
        min_object_size: int = CONTOUR_FINDER_MIN_OBJECT_SIZE,
        contour_level: float = 0.5
    ):
        """
        Initialize the contour finder with configuration.
        
        Args:
            min_object_size: Minimum object size in pixels to retain (removes noise)
            contour_level: Level at which to find contours (0.5 for binary images)
        """
        self.min_object_size = min_object_size
        self.contour_level = contour_level
    
    @staticmethod
    def load_image(path: str) -> np.ndarray:
        """
        Load an image from disk.
        
        Args:
            path: Path to the image file
            
        Returns:
            Image as numpy array
            
        Raises:
            FileNotFoundError: If the image doesn't exist
            ValueError: If the image cannot be loaded
        """
        try:
            img = io.imread(path)
            if img is None:
                raise ValueError(f"Failed to load image: {path}")
            return img
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found: {path}")
    
    @staticmethod
    def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
        """
        Convert an image to grayscale.
        
        Args:
            img: Input image (RGB or grayscale)
            
        Returns:
            Grayscale image as float array (0.0 to 1.0)
        """
        if img.ndim == 3:
            return color.rgb2gray(img)
        return img.astype(float) / 255.0
    
    @staticmethod
    def apply_otsu_threshold(gray: np.ndarray) -> np.ndarray:
        """
        Apply Otsu's thresholding to a grayscale image.
        
        Args:
            gray: Grayscale image (float, 0.0 to 1.0)
            
        Returns:
            Binary mask (boolean array)
        """
        threshold_value = filters.threshold_otsu(gray)
        return gray > threshold_value
    
    def remove_noise(self, binary: np.ndarray) -> np.ndarray:
        """
        Remove small objects from a binary mask.
        
        Args:
            binary: Binary mask (boolean array)
            
        Returns:
            Cleaned binary mask with small objects removed
        """
        return morphology.remove_small_objects(binary, min_size=self.min_object_size)
    
    def extract_contours(self, binary: np.ndarray) -> List[np.ndarray]:
        """
        Extract contours from a binary mask.
        
        Args:
            binary: Binary mask (boolean array)
            
        Returns:
            List of contour arrays, each with shape (N, 2) for N points
        """
        return measure.find_contours(binary, self.contour_level)
    
    def preprocess(self, path: str) -> PreprocessingResult:
        """
        Full preprocessing pipeline: load, grayscale, threshold, clean, and extract contours.
        
        Args:
            path: Path to the input image
            
        Returns:
            PreprocessingResult containing gray image, binary mask, and contours
        """
        img = self.load_image(path)
        gray = self.convert_to_grayscale(img)
        binary = self.apply_otsu_threshold(gray)
        clean = self.remove_noise(binary)
        contours = self.extract_contours(clean)
        
        return PreprocessingResult(
            gray=gray,
            binary=clean,
            contours=contours
        )
    
    def preprocess_from_array(self, img: np.ndarray) -> PreprocessingResult:
        """
        Preprocess an already-loaded image array.
        
        Args:
            img: Input image as numpy array
            
        Returns:
            PreprocessingResult containing gray image, binary mask, and contours
        """
        gray = self.convert_to_grayscale(img)
        binary = self.apply_otsu_threshold(gray)
        clean = self.remove_noise(binary)
        contours = self.extract_contours(clean)
        
        return PreprocessingResult(
            gray=gray,
            binary=clean,
            contours=contours
        )

    def color_segment(
        self,
        path: str,
        n_colors: int = SHAPE_FITTING_N_COLORS,
        n_init: int = 5,
        random_state: int = 0
    ) -> ColorSegmentationResult:
        """
        Perform color-based segmentation using KMeans clustering in LAB color space.
        
        Args:
            path: Path to the input image
            n_colors: Number of color clusters to find
            n_init: Number of KMeans initializations
            random_state: Random seed for reproducibility
            
        Returns:
            ColorSegmentationResult containing cluster map, centers, and RGB colors
        """
        img = io.imread(path)[:, :, :3]
        H, W = img.shape[:2]

        img_lab = color.rgb2lab(img)
        pixels = img_lab.reshape(-1, 3)

        kmeans = KMeans(n_clusters=n_colors, n_init=n_init, random_state=random_state)
        labels = kmeans.fit_predict(pixels)

        cluster_map = labels.reshape(H, W)
        cluster_centers = kmeans.cluster_centers_
        cluster_colors = color.lab2rgb(
            cluster_centers.reshape(-1, 1, 1, 3)
        ).reshape(-1, 3)

        return ColorSegmentationResult(
            cluster_map=cluster_map,
            cluster_centers=cluster_centers,
            cluster_colors=cluster_colors
        )
