"""
Shape plotter for visualizing detected shapes on images.

Reads shape data from CSV files and draws them on images using Plotly.
Supports multiple output formats (PNG, HTML) and multiple image types.
Includes automatic coordinate scaling when target image differs from reference.
"""

import csv
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from PIL import Image

import cv2
import plotly.graph_objects as go

from nanorange.image_analyzer.plotter.shape_plotter.base_drawer import BaseShapeDrawer, DrawStyle
from nanorange.image_analyzer.plotter.shape_plotter.circle_drawer import CircleDrawer
from nanorange.image_analyzer.plotter.shape_plotter.ellipse_drawer import EllipseDrawer


def calculate_tile_offsets(tile_paths: List[str]) -> Dict[str, Tuple[int, int]]:
    """
    Calculate the pixel offset for each tile based on its grid position.
    
    Tile filenames follow the pattern 'tile_{row}_{col}.png', where
    row and col indicate the tile's position in the grid.
    
    Args:
        tile_paths: List of paths to tile images
        
    Returns:
        Dictionary mapping tile_path to (offset_x, offset_y) tuple
    """
    offsets: Dict[str, Tuple[int, int]] = {}
    
    if not tile_paths:
        return offsets
    
    first_tile = cv2.imread(tile_paths[0])
    if first_tile is None:
        print("   ⚠️ Warning: Could not read tile to determine offsets")
        return {path: (0, 0) for path in tile_paths}
    
    tile_height, tile_width = first_tile.shape[:2]
    
    tile_pattern = re.compile(r'tile_(\d+)_(\d+)')
    
    for tile_path in tile_paths:
        filename = os.path.basename(tile_path)
        match = tile_pattern.search(filename)
        
        if match:
            row = int(match.group(1))
            col = int(match.group(2))
            offset_x = col * tile_width
            offset_y = row * tile_height
            offsets[tile_path] = (offset_x, offset_y)
        else:
            offsets[tile_path] = (0, 0)
    
    return offsets


@dataclass
class ScaleFactors:
    """Scale factors for coordinate transformation between images."""
    scale_x: float = 1.0
    scale_y: float = 1.0
    
    @classmethod
    def from_dimensions(
        cls,
        reference_size: Tuple[int, int],
        target_size: Tuple[int, int]
    ) -> "ScaleFactors":
        """
        Calculate scale factors from reference to target dimensions.
        
        Args:
            reference_size: (width, height) of reference image (where shapes were fitted)
            target_size: (width, height) of target image (where shapes will be drawn)
            
        Returns:
            ScaleFactors instance
        """
        ref_w, ref_h = reference_size
        tgt_w, tgt_h = target_size
        
        return cls(
            scale_x=tgt_w / ref_w if ref_w > 0 else 1.0,
            scale_y=tgt_h / ref_h if ref_h > 0 else 1.0
        )
    
    def scale_point(self, x: float, y: float) -> Tuple[float, float]:
        """Scale a point's coordinates."""
        return x * self.scale_x, y * self.scale_y
    
    def scale_radius(self, radius: float) -> float:
        """Scale a radius using the average of x and y scales."""
        return radius * (self.scale_x + self.scale_y) / 2
    
    def scale_axes(self, a: float, b: float) -> Tuple[float, float]:
        """Scale ellipse axes using the average scale factor."""
        avg_scale = (self.scale_x + self.scale_y) / 2
        return a * avg_scale, b * avg_scale
    
    @property
    def is_identity(self) -> bool:
        """Check if this is effectively an identity transform (no scaling)."""
        return abs(self.scale_x - 1.0) < 0.001 and abs(self.scale_y - 1.0) < 0.001


@dataclass
class PlotterConfig:
    """Configuration for the shape plotter."""
    circle_style: DrawStyle = field(default_factory=lambda: DrawStyle(
        line_color="lime",
        line_width=2.0
    ))
    ellipse_style: DrawStyle = field(default_factory=lambda: DrawStyle(
        line_color="cyan",
        line_width=2.0
    ))
    figure_margin: int = 0
    show_axis: bool = False


class ShapePlotter:
    """
    Plots detected shapes on images using Plotly.
    
    Features:
    - Modular drawer system for easy extension
    - Reads shapes from CSV files
    - Outputs both PNG and interactive HTML
    - Processes multiple image types (thresholded, colorized)
    """
    
    def __init__(self, config: Optional[PlotterConfig] = None):
        """
        Initialize the shape plotter.
        
        Args:
            config: Plotter configuration
        """
        self.config = config or PlotterConfig()
        self._drawers: Dict[str, BaseShapeDrawer] = {}
        
        self.register_drawer(CircleDrawer(style=self.config.circle_style))
        self.register_drawer(EllipseDrawer(style=self.config.ellipse_style))
    
    def register_drawer(self, drawer: BaseShapeDrawer) -> None:
        """
        Register a shape drawer.
        
        Args:
            drawer: Shape drawer instance to register
        """
        self._drawers[drawer.shape_type] = drawer
    
    def get_drawer(self, shape_type: str) -> Optional[BaseShapeDrawer]:
        """
        Get the drawer for a specific shape type.
        
        Args:
            shape_type: Type of shape (e.g., 'Circle', 'Ellipse')
            
        Returns:
            Drawer instance or None if not found
        """
        return self._drawers.get(shape_type)
    
    def load_shapes_from_csv(self, csv_path: str) -> List[Dict]:
        """
        Load shape data from a CSV file.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            List of dictionaries, each containing shape data
        """
        shapes = []
        
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                shapes.append(dict(row))
        
        return shapes
    
    def create_figure_with_image(
        self, 
        image_path: str
    ) -> go.Figure:
        """
        Create a Plotly figure with an image as background.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Plotly figure with image layout
        """
        img = Image.open(image_path)
        width, height = img.size
        
        fig = go.Figure()
        
        fig.add_layout_image(
            dict(
                source=img,
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=width,
                sizey=height,
                sizing="stretch",
                layer="below"
            )
        )
        
        fig.update_xaxes(
            range=[0, width],
            showgrid=False,
            zeroline=False,
            showticklabels=self.config.show_axis,
            visible=self.config.show_axis
        )
        
        fig.update_yaxes(
            range=[height, 0],
            showgrid=False,
            zeroline=False,
            showticklabels=self.config.show_axis,
            visible=self.config.show_axis,
            scaleanchor="x",
            scaleratio=1
        )
        
        fig.update_layout(
            width=width,
            height=height,
            margin=dict(
                l=self.config.figure_margin,
                r=self.config.figure_margin,
                t=self.config.figure_margin,
                b=self.config.figure_margin
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        
        return fig
    
    def draw_shapes_on_figure(
        self, 
        fig: go.Figure, 
        shapes: List[Dict],
        scale_factors: Optional[ScaleFactors] = None
    ) -> go.Figure:
        """
        Draw all shapes on a figure with optional coordinate scaling.
        
        Args:
            fig: Plotly figure to draw on
            shapes: List of shape dictionaries from CSV
            scale_factors: Optional scale factors for coordinate transformation
            
        Returns:
            Updated figure with shapes drawn
        """
        scale = scale_factors or ScaleFactors()
        
        for shape_data in shapes:
            shape_type = shape_data.get("Type", "")
            drawer = self.get_drawer(shape_type)
            
            if drawer is None:
                print(f"   ⚠️  No drawer registered for shape type: {shape_type}")
                continue
            
            try:
                params = drawer.from_csv_row(shape_data)
                
                params = self._scale_shape_params(params, shape_type, scale)
                
                fig = drawer.draw(fig, **params)
            except (ValueError, KeyError) as e:
                print(f"   ⚠️  Error drawing {shape_type}: {e}")
        
        return fig
    
    def _scale_shape_params(
        self,
        params: Dict,
        shape_type: str,
        scale: ScaleFactors
    ) -> Dict:
        """
        Scale shape parameters based on scale factors.
        
        Args:
            params: Original shape parameters from CSV
            shape_type: Type of shape ('Circle' or 'Ellipse')
            scale: Scale factors to apply
            
        Returns:
            Scaled parameters dictionary
        """
        if scale.is_identity:
            return params
        
        scaled = params.copy()
        
        if "x" in scaled and "y" in scaled:
            scaled["x"], scaled["y"] = scale.scale_point(scaled["x"], scaled["y"])
        
        if shape_type == "Circle" and "radius" in scaled:
            scaled["radius"] = scale.scale_radius(scaled["radius"])
        elif shape_type == "Ellipse":
            if "semi_major" in scaled and "semi_minor" in scaled:
                scaled["semi_major"], scaled["semi_minor"] = scale.scale_axes(
                    scaled["semi_major"], scaled["semi_minor"]
                )
        
        return scaled
    
    def plot_shapes_on_image(
        self, 
        image_path: str, 
        csv_path: str,
        reference_size: Optional[Tuple[int, int]] = None
    ) -> go.Figure:
        """
        Create a plot with shapes drawn on an image.
        
        When reference_size is provided and differs from the target image,
        shape coordinates are automatically scaled to match the target.
        
        Args:
            image_path: Path to the background image
            csv_path: Path to the CSV file with shape data
            reference_size: Optional (width, height) of the image where shapes
                           were originally fitted. If provided, shapes will be
                           scaled to match the target image dimensions.
            
        Returns:
            Plotly figure with image and shapes
        """
        shapes = self.load_shapes_from_csv(csv_path)
        fig = self.create_figure_with_image(image_path)
        
        scale_factors = None
        if reference_size is not None:
            target_img = Image.open(image_path)
            target_size = target_img.size
            scale_factors = ScaleFactors.from_dimensions(reference_size, target_size)
            
            if not scale_factors.is_identity:
                print(f"      Scaling shapes: {reference_size} → {target_size} "
                      f"(scale: {scale_factors.scale_x:.3f}x, {scale_factors.scale_y:.3f}x)")
        
        fig = self.draw_shapes_on_figure(fig, shapes, scale_factors)
        
        return fig
    
    def save_figure(
        self, 
        fig: go.Figure, 
        output_path: str,
        save_html: bool = True,
        save_png: bool = True
    ) -> Dict[str, str]:
        """
        Save the figure to file(s).
        
        Args:
            fig: Plotly figure to save
            output_path: Base output path (without extension)
            save_html: Whether to save HTML file
            save_png: Whether to save PNG file
            
        Returns:
            Dictionary with paths to saved files
        """
        saved_files = {}
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        if save_html:
            html_path = f"{output_path}.html"
            fig.write_html(html_path, include_plotlyjs=True, full_html=True)
            saved_files["html"] = html_path
        
        if save_png:
            png_path = f"{output_path}.png"
            fig.write_image(png_path, scale=1)
            saved_files["png"] = png_path
        
        return saved_files
    
    def save_scaled_csv(
        self,
        input_csv_path: str,
        output_csv_path: str,
        reference_size: Tuple[int, int],
        target_size: Tuple[int, int]
    ) -> bool:
        """
        Save a CSV with shape coordinates scaled to target image dimensions.
        
        Args:
            input_csv_path: Path to the original CSV file
            output_csv_path: Path for the scaled CSV file
            reference_size: (width, height) of the reference image
            target_size: (width, height) of the target image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            scale = ScaleFactors.from_dimensions(reference_size, target_size)
            
            if scale.is_identity:
                shutil.copy(input_csv_path, output_csv_path)
                return True
            
            shapes = self.load_shapes_from_csv(input_csv_path)
            
            os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
            
            with open(output_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Type", "X", "Y", "R_or_A", "B", "Theta"])
                
                for shape in shapes:
                    shape_type = shape.get("Type", "")
                    x = float(shape.get("X", 0))
                    y = float(shape.get("Y", 0))
                    
                    scaled_x, scaled_y = scale.scale_point(x, y)
                    
                    if shape_type == "Circle":
                        r = float(shape.get("R_or_A", 0))
                        scaled_r = scale.scale_radius(r)
                        writer.writerow(["Circle", scaled_x, scaled_y, scaled_r, "", ""])
                    elif shape_type == "Ellipse":
                        a = float(shape.get("R_or_A", 0))
                        b = float(shape.get("B", 0))
                        theta = shape.get("Theta", 0)
                        scaled_a, scaled_b = scale.scale_axes(a, b)
                        writer.writerow(["Ellipse", scaled_x, scaled_y, scaled_a, scaled_b, theta])
            
            return True
        except Exception as e:
            print(f"   ⚠️ Error saving scaled CSV: {e}")
            return False
    
    def process_shapes_folder(
        self,
        shapes_folder: str,
        csv_filename: str = "detected_shapes.csv",
        thresholded_filename: str = "final_thresholded.png",
        colorized_filename: str = "final_colorized.png",
        original_filename: str = "original.png"
    ) -> Dict[str, Dict[str, str]]:
        """
        Process a shapes folder and generate visualizations.
        
        The colorized image is used as the reference since shapes are fitted
        on it. When drawing on other images (like thresholded or original), 
        coordinates are automatically scaled if dimensions differ.
        Also creates scaled CSV files for images with different dimensions.
        
        Args:
            shapes_folder: Path to the folder containing images and CSV
            csv_filename: Name of the CSV file
            thresholded_filename: Name of the thresholded image
            colorized_filename: Name of the colorized image (reference)
            original_filename: Name of the original image
            
        Returns:
            Dictionary with output paths for each processed image
        """
        csv_path = os.path.join(shapes_folder, csv_filename)
        colorized_path = os.path.join(shapes_folder, colorized_filename)
        results = {}
        
        reference_size = None
        if os.path.exists(colorized_path):
            ref_img = Image.open(colorized_path)
            reference_size = ref_img.size
            print(f"   Reference image size (colorized): {reference_size[0]}×{reference_size[1]}")
        
        if os.path.exists(colorized_path):
            print(f"   Processing colorized image...")
            fig = self.plot_shapes_on_image(colorized_path, csv_path)
            output_base = os.path.join(shapes_folder, "colorized_with_shapes")
            results["colorized"] = self.save_figure(fig, output_base)
            print(f"   ✓ Saved colorized visualizations")
        
        thresholded_path = os.path.join(shapes_folder, thresholded_filename)
        if os.path.exists(thresholded_path):
            print(f"   Processing thresholded image...")
            fig = self.plot_shapes_on_image(
                thresholded_path, 
                csv_path, 
                reference_size=reference_size
            )
            output_base = os.path.join(shapes_folder, "thresholded_with_shapes")
            results["thresholded"] = self.save_figure(fig, output_base)
            print(f"   ✓ Saved thresholded visualizations")
        
        original_path = os.path.join(shapes_folder, original_filename)
        if os.path.exists(original_path) and reference_size is not None:
            print(f"   Processing original image...")
            
            original_img = Image.open(original_path)
            original_size = original_img.size
            print(f"      Original image size: {original_size[0]}×{original_size[1]}")
            
            original_csv_path = os.path.join(shapes_folder, "detected_shapes_original.csv")
            if self.save_scaled_csv(csv_path, original_csv_path, reference_size, original_size):
                print(f"   ✓ Saved scaled CSV: {os.path.basename(original_csv_path)}")
                results["original_csv"] = original_csv_path
            
            fig = self.plot_shapes_on_image(
                original_path,
                csv_path,
                reference_size=reference_size
            )
            output_base = os.path.join(shapes_folder, "original_with_shapes")
            results["original"] = self.save_figure(fig, output_base)
            print(f"   ✓ Saved original visualizations")
        
        return results


def plot_shapes(
    shapes_folder: str,
    circle_color: str = "lime",
    ellipse_color: str = "cyan",
    line_width: float = 2.0
) -> Dict[str, Dict[str, str]]:
    """
    Convenience function to plot shapes on images in a folder.
    
    Args:
        shapes_folder: Path to folder with images and CSV
        circle_color: Color for circles
        ellipse_color: Color for ellipses
        line_width: Line width for shapes
        
    Returns:
        Dictionary with output paths
    """
    config = PlotterConfig(
        circle_style=DrawStyle(line_color=circle_color, line_width=line_width),
        ellipse_style=DrawStyle(line_color=ellipse_color, line_width=line_width)
    )
    
    plotter = ShapePlotter(config)
    return plotter.process_shapes_folder(shapes_folder)
