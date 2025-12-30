"""
Size distribution plotter for visualizing particle diameter distributions.

Creates histogram plots of particle sizes (diameters) from detected shapes CSV files.
Uses Plotly for interactive visualizations with export to PNG and HTML.
"""

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class SizeDistributionConfig:
    """Configuration for size distribution plots."""
    
    bin_size: Optional[float] = None
    num_bins: int = 20
    outlier_iqr_multiplier: float = 1.5
    
    bar_color: str = "#3498db"
    bar_color_filtered: str = "#2ecc71"
    bar_line_color: str = "#2980b9"
    bar_line_color_filtered: str = "#27ae60"
    bar_line_width: float = 1.0
    
    x_axis_title: str = "Particle Diameter (pixels)"
    y_axis_title: str = "Number of Particles"
    title: str = "Particle Size Distribution"
    title_filtered: str = "Particle Size Distribution (Outliers Removed)"
    
    width: int = 900
    height: int = 800
    
    margin: Dict[str, int] = field(default_factory=lambda: {
        "l": 70, "r": 40, "t": 80, "b": 60
    })


class SizeDistributionPlotter:
    """
    Creates size distribution histograms from detected shapes CSV files.
    
    Reads particle data from CSV and generates histogram plots showing
    the distribution of particle diameters (2 * radius for circles,
    2 * semi-major axis for ellipses).
    
    Creates two subplots:
    - Top: Full distribution with all data
    - Bottom: Filtered distribution with outliers removed
    """
    
    def __init__(self, config: Optional[SizeDistributionConfig] = None):
        """
        Initialize the size distribution plotter.
        
        Args:
            config: Plot configuration. If None, uses defaults.
        """
        self.config = config or SizeDistributionConfig()
    
    def load_diameters_from_csv(self, csv_path: str) -> List[float]:
        """
        Load particle diameters from a detected shapes CSV file.
        
        For circles: diameter = 2 * radius
        For ellipses: diameter = 2 * semi-major axis (a)
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            List of particle diameters
        """
        diameters = []
        
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r_or_a = row.get("R_or_A", "")
                
                if not r_or_a or r_or_a.strip() == "":
                    continue
                
                try:
                    radius_or_a = float(r_or_a)
                    diameter = 2 * radius_or_a
                    diameters.append(diameter)
                except ValueError:
                    continue
        
        return diameters
    
    def filter_outliers_iqr(
        self, 
        diameters: List[float]
    ) -> Tuple[List[float], float, float]:
        """
        Filter outliers using the IQR (Interquartile Range) method.
        
        Values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are considered outliers.
        
        Args:
            diameters: List of particle diameters
            
        Returns:
            Tuple of (filtered_diameters, lower_bound, upper_bound)
        """
        if not diameters:
            return [], 0, 0
        
        arr = np.array(diameters)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        
        multiplier = self.config.outlier_iqr_multiplier
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        filtered = [d for d in diameters if lower_bound <= d <= upper_bound]
        
        return filtered, lower_bound, upper_bound
    
    def calculate_bin_size(self, diameters: List[float]) -> float:
        """
        Calculate bin size for histogram.
        
        Args:
            diameters: List of particle diameters
            
        Returns:
            Calculated bin size
        """
        if not diameters:
            return 1.0
        
        if self.config.bin_size is not None:
            return self.config.bin_size
        
        data_range = max(diameters) - min(diameters)
        if data_range == 0:
            return 1.0
        
        return data_range / self.config.num_bins
    
    def create_dual_histogram(self, diameters: List[float]) -> go.Figure:
        """
        Create a figure with two histogram subplots.
        
        Top: Full distribution (all data)
        Bottom: Filtered distribution (outliers removed)
        
        Args:
            diameters: List of particle diameters
            
        Returns:
            Plotly figure with two subplots
        """
        filtered_diameters, lower_bound, upper_bound = self.filter_outliers_iqr(diameters)
        n_outliers = len(diameters) - len(filtered_diameters)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                f"All Particles (n={len(diameters)})",
                f"Outliers Removed (n={len(filtered_diameters)}, {n_outliers} excluded)"
            ),
            vertical_spacing=0.12
        )
        
        bin_size_full = self.calculate_bin_size(diameters)
        fig.add_trace(
            go.Histogram(
                x=diameters,
                xbins=dict(
                    start=min(diameters),
                    end=max(diameters),
                    size=bin_size_full
                ),
                marker=dict(
                    color=self.config.bar_color,
                    line=dict(
                        color=self.config.bar_line_color,
                        width=self.config.bar_line_width
                    )
                ),
                hovertemplate=(
                    "Diameter: %{x:.1f}<br>"
                    "Count: %{y}<extra></extra>"
                ),
                name="All"
            ),
            row=1, col=1
        )
        
        if filtered_diameters:
            bin_size_filtered = self.calculate_bin_size(filtered_diameters)
            fig.add_trace(
                go.Histogram(
                    x=filtered_diameters,
                    xbins=dict(
                        start=min(filtered_diameters),
                        end=max(filtered_diameters),
                        size=bin_size_filtered
                    ),
                    marker=dict(
                        color=self.config.bar_color_filtered,
                        line=dict(
                            color=self.config.bar_line_color_filtered,
                            width=self.config.bar_line_width
                        )
                    ),
                    hovertemplate=(
                        "Diameter: %{x:.1f}<br>"
                        "Count: %{y}<extra></extra>"
                    ),
                    name="Filtered"
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title=dict(
                text=self.config.title,
                x=0.5,
                xanchor="center",
                font=dict(size=18)
            ),
            width=self.config.width,
            height=self.config.height,
            margin=self.config.margin,
            showlegend=False,
            plot_bgcolor="white"
        )
        
        for row in [1, 2]:
            fig.update_xaxes(
                title_text=self.config.x_axis_title,
                showgrid=True,
                gridcolor="lightgray",
                zeroline=True,
                zerolinecolor="gray",
                row=row, col=1
            )
            fig.update_yaxes(
                title_text=self.config.y_axis_title,
                showgrid=True,
                gridcolor="lightgray",
                zeroline=True,
                zerolinecolor="gray",
                row=row, col=1
            )
        
        fig.add_annotation(
            text=f"Outlier bounds: {lower_bound:.1f} - {upper_bound:.1f} px",
            xref="paper", yref="paper",
            x=0.98, y=0.48,
            showarrow=False,
            font=dict(size=10, color="gray"),
            xanchor="right"
        )
        
        return fig
    
    def save_figure(
        self,
        fig: go.Figure,
        output_path: str,
        save_html: bool = True,
        save_png: bool = True
    ) -> Dict[str, str]:
        """
        Save the histogram figure to file(s).
        
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
            fig.write_image(png_path, scale=2)
            saved_files["png"] = png_path
        
        return saved_files
    
    def plot_from_csv(
        self,
        csv_path: str,
        output_base: str,
        save_html: bool = True,
        save_png: bool = True
    ) -> Dict[str, str]:
        """
        Generate size distribution plot from a CSV file.
        
        Args:
            csv_path: Path to the detected shapes CSV file
            output_base: Base path for output files (without extension)
            save_html: Whether to save HTML file
            save_png: Whether to save PNG file
            
        Returns:
            Dictionary with paths to saved files
        """
        diameters = self.load_diameters_from_csv(csv_path)
        
        if not diameters:
            print("   ⚠️  No valid diameter data found in CSV")
            return {}
        
        fig = self.create_dual_histogram(diameters)
        return self.save_figure(fig, output_base, save_html, save_png)
    
    def plot_from_shapes_folder(
        self,
        shapes_folder: str,
        csv_filename: str = "detected_shapes_original.csv",
        output_filename: str = "size_distribution"
    ) -> Dict[str, str]:
        """
        Generate size distribution plot from a shapes folder.
        
        Args:
            shapes_folder: Path to the folder containing the CSV file
            csv_filename: Name of the CSV file to read
            output_filename: Base name for output files (without extension)
            
        Returns:
            Dictionary with paths to saved files
        """
        csv_path = os.path.join(shapes_folder, csv_filename)
        
        if not os.path.exists(csv_path):
            csv_path = os.path.join(shapes_folder, "detected_shapes.csv")
            if not os.path.exists(csv_path):
                print(f"   ⚠️  No CSV file found in {shapes_folder}")
                return {}
        
        output_base = os.path.join(shapes_folder, output_filename)
        
        diameters = self.load_diameters_from_csv(csv_path)
        
        if not diameters:
            print("   ⚠️  No valid diameter data found in CSV")
            return {}
        
        filtered, lower, upper = self.filter_outliers_iqr(diameters)
        n_outliers = len(diameters) - len(filtered)
        
        print(f"   Found {len(diameters)} particles")
        print(f"   Diameter range: {min(diameters):.1f} - {max(diameters):.1f} pixels")
        print(f"   Outliers: {n_outliers} particles (bounds: {lower:.1f} - {upper:.1f} px)")
        
        fig = self.create_dual_histogram(diameters)
        saved_files = self.save_figure(fig, output_base)
        
        return saved_files


def plot_size_distribution(
    shapes_folder: str,
    csv_filename: str = "detected_shapes_original.csv",
    output_filename: str = "size_distribution",
    bin_size: Optional[float] = None,
    num_bins: int = 20,
    outlier_iqr_multiplier: float = 1.5
) -> Dict[str, str]:
    """
    Convenience function to plot size distribution from a shapes folder.
    
    Args:
        shapes_folder: Path to folder containing CSV file
        csv_filename: Name of the CSV file
        output_filename: Base name for output files
        bin_size: Optional fixed bin size (auto-calculated if None)
        num_bins: Number of bins when bin_size is None
        outlier_iqr_multiplier: IQR multiplier for outlier detection
        
    Returns:
        Dictionary with paths to saved files
    """
    config = SizeDistributionConfig(
        bin_size=bin_size,
        num_bins=num_bins,
        outlier_iqr_multiplier=outlier_iqr_multiplier
    )
    
    plotter = SizeDistributionPlotter(config)
    return plotter.plot_from_shapes_folder(
        shapes_folder,
        csv_filename,
        output_filename
    )
