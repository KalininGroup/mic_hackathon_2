# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mcp[cli]>=1.12.3",
#     "pydantic>=2.11.7",
#     "python-dotenv>=1.1.1",
#     "Pyro5>=5.14",
#     "numpy>=1.24.0",
#     "scikit-learn>=1.3.0",
#     "matplotlib>=3.7.2",
#     "requests>=2.32.0",
# ]
# ///

from mcp.server.fastmcp import FastMCP
from typing import List, Tuple, Optional
import numpy as np
import Pyro5.api
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json
import os
import requests
import sys
import matplotlib.pyplot as plt
from datetime import datetime

mcp = FastMCP("STEMMicroscope")

# Global variable to store the microscope connection
_mic_server = None
_current_data = {
    "spectra": [],
    "locations": [],
    "overview_image": None,
    "pca_results": None,
    "clusters": None,
    "saved_images": []  # Track all saved images
}

# Create output directory for images
OUTPUT_DIR = "stem_analysis_images"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def _save_image_info(filepath: str, description: str):
    """Track saved images"""
    _current_data["saved_images"].append({
        "filepath": filepath,
        "description": description,
        "timestamp": datetime.now().isoformat()
    })

def _get_microscope_connection(uri: str = "PYRO:microscope.server@localhost:9091"):
    """Get or create microscope server connection"""
    global _mic_server
    if _mic_server is None:
        _mic_server = Pyro5.api.Proxy(uri)
    return _mic_server

@mcp.tool(
    name="Download_Data_File",
    description="Download a data file from a URL to the current working directory"
)
def download_data_file(url: str, filename: Optional[str] = None) -> str:
    """
    Download a file from a URL.
    
    Args:
        url: URL of the file to download (e.g., https://github.com/.../test_stem.h5)
        filename: Optional custom filename. If not provided, extracts from URL
    """
    try:
        # Extract filename from URL if not provided
        if filename is None:
            filename = url.split('/')[-1]
        
        # Get absolute path for the file
        abs_path = os.path.abspath(filename)
        
        # Download the file
        print(f"Downloading from: {url}", file=sys.stderr)
        print(f"Saving to: {abs_path}", file=sys.stderr)
        
        response = requests.get(url, verify=False, stream=True)
        response.raise_for_status()
        
        # Save the file
        total_size = int(response.headers.get('content-length', 0))
        with open(abs_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
        
        # Verify file exists
        if os.path.exists(abs_path):
            file_size = os.path.getsize(abs_path)
            return f"File downloaded successfully!\n  URL: {url}\n  Filename: {filename}\n  Absolute path: {abs_path}\n  Size: {file_size:,} bytes"
        else:
            return f"Error: File download failed - file not found at {abs_path}"
            
    except requests.exceptions.RequestException as e:
        return f"Error downloading file: {str(e)}\n  URL: {url}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool(
    name="Initialize_Microscope",
    description="Initialize the STEM microscope connection and optionally register a data file"
)
def initialize_microscope(
    microscope_type: str = "STEM",
    data_file: Optional[str] = None,
    server_uri: str = "PYRO:microscope.server@localhost:9091"
) -> str:
    """
    Initialize microscope and optionally register data.
    
    Args:
        microscope_type: Type of microscope (default: "STEM")
        data_file: Optional H5 data file to register (e.g., "test_stem.h5" or full path)
        server_uri: Pyro5 server URI
    """
    try:
        mic_server = _get_microscope_connection(server_uri)
        mic_server.initialize_microscope(microscope_type)
        
        result = f"Microscope initialized successfully as {microscope_type}"
        
        if data_file:
            # Convert to absolute path
            abs_path = os.path.abspath(data_file)
            
            # Check if file exists
            if not os.path.exists(abs_path):
                return f"Error: File '{data_file}' not found at path: {abs_path}\n\nPlease ensure the file exists or provide the correct path."
            
            mic_server.register_data(abs_path)
            result += f"\nData file registered successfully:\n  Original: {data_file}\n  Absolute path: {abs_path}"
        
        return result
    except Exception as e:
        return f"Error initializing microscope: {str(e)}"

@mcp.tool(
    name="Register_Data_File",
    description="Register a data file (H5 format) with the microscope"
)
def register_data_file(data_file: str) -> str:
    """
    Register a data file with the microscope.
    
    Args:
        data_file: Path to H5 data file (e.g., "test_stem.h5" or full path)
    """
    try:
        # Convert to absolute path
        abs_path = os.path.abspath(data_file)
        
        # Check if file exists
        if not os.path.exists(abs_path):
            return f"Error: File '{data_file}' not found at path: {abs_path}\n\nPlease ensure the file exists or provide the correct path."
        
        mic_server = _get_microscope_connection()
        mic_server.register_data(abs_path)
        
        return f"Data file registered successfully:\n  Original: {data_file}\n  Absolute path: {abs_path}"
    except Exception as e:
        return f"Error registering data file: {str(e)}"

@mcp.tool(
    name="Get_Overview_Image",
    description="Get the overview image from the microscope and store it for visualization"
)
def get_overview_image() -> str:
    """
    Retrieve the overview image from the microscope.
    Returns information about the image shape and dtype.
    """
    try:
        mic_server = _get_microscope_connection()
        array_list, shape, dtype = mic_server.get_overview_image()
        
        # Store the image in global data
        im_array = np.array(array_list, dtype=dtype).reshape(shape)
        _current_data["overview_image"] = im_array
        
        # Save the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(OUTPUT_DIR, f"overview_image_{timestamp}.png")
        
        plt.figure(figsize=(10, 8))
        plt.imshow(im_array, cmap='gray')
        plt.colorbar(label='Intensity')
        plt.title('Overview Image')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        _save_image_info(filepath, "Overview image from microscope")
        
        abs_path = os.path.abspath(filepath)
        return f"Overview image retrieved successfully.\nShape: {shape}\nDtype: {dtype}\nImage stored in memory for analysis.\n\n📸 Image saved to: {abs_path}"
    except Exception as e:
        return f"Error getting overview image: {str(e)}"

@mcp.tool(
    name="Get_Point_Spectrum",
    description="Get the spectrum from a specific point (x, y) on the sample"
)
def get_point_spectrum(x: int, y: int, channel: str = "Channel_001") -> str:
    """
    Get spectrum from a specific location.
    
    Args:
        x: X coordinate
        y: Y coordinate
        channel: Channel name (default: "Channel_001")
    """
    try:
        mic_server = _get_microscope_connection()
        array_list, shape, dtype = mic_server.get_point_data(channel, x, y)
        spectrum = np.array(array_list, dtype=dtype).reshape(shape)
        
        # Save spectrum plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(OUTPUT_DIR, f"spectrum_point_{x}_{y}_{timestamp}.png")
        
        plt.figure(figsize=(10, 6))
        plt.plot(spectrum.flatten())
        plt.title(f'Spectrum at Point ({x}, {y})')
        plt.xlabel('Energy Channel')
        plt.ylabel('Intensity')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        _save_image_info(filepath, f"Spectrum from point ({x}, {y})")
        
        abs_path = os.path.abspath(filepath)
        return f"Spectrum retrieved from point ({x}, {y}).\nShape: {shape}\nDtype: {dtype}\nSpectrum length: {len(spectrum.flatten())}\n\n📸 Spectrum plot saved to: {abs_path}"
    except Exception as e:
        return f"Error getting spectrum: {str(e)}"

@mcp.tool(
    name="Collect_Grid_Spectra",
    description="Collect spectra from a grid of points. This is useful for clustering and analysis."
)
def collect_grid_spectra(
    grid_size_x: int = 10,
    grid_size_y: int = 10,
    channel: str = "Channel_001"
) -> str:
    """
    Collect spectra from a grid of points.
    
    Args:
        grid_size_x: Number of points in X direction (default: 10)
        grid_size_y: Number of points in Y direction (default: 10)
        channel: Channel name (default: "Channel_001")
    """
    try:
        mic_server = _get_microscope_connection()
        
        spectra = []
        locations = []
        
        for x in range(grid_size_x):
            for y in range(grid_size_y):
                array_list, shape, dtype = mic_server.get_point_data(channel, x, y)
                spectrum = np.array(array_list, dtype=dtype).reshape(shape)
                spectra.append(spectrum.flatten())
                locations.append((x, y))
        
        # Store in global data
        _current_data["spectra"] = np.array(spectra)
        _current_data["locations"] = locations
        
        # Save grid collection visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(OUTPUT_DIR, f"grid_collection_{grid_size_x}x{grid_size_y}_{timestamp}.png")
        
        # Create a visualization showing sample spectra
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Grid Collection: {grid_size_x}x{grid_size_y} points', fontsize=14)
        
        # Plot 4 sample spectra from different positions
        sample_indices = [0, len(spectra)//3, 2*len(spectra)//3, len(spectra)-1]
        for idx, ax in enumerate(axes.flat):
            spec_idx = sample_indices[idx]
            loc = locations[spec_idx]
            ax.plot(spectra[spec_idx])
            ax.set_title(f'Point ({loc[0]}, {loc[1]})')
            ax.set_xlabel('Energy Channel')
            ax.set_ylabel('Intensity')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        _save_image_info(filepath, f"Grid collection {grid_size_x}x{grid_size_y} sample spectra")
        
        total_points = grid_size_x * grid_size_y
        abs_path = os.path.abspath(filepath)
        return f"Collected {total_points} spectra from {grid_size_x}x{grid_size_y} grid.\nSpectra shape: {_current_data['spectra'].shape}\nData stored in memory for analysis.\n\n📸 Sample spectra visualization saved to: {abs_path}"
    except Exception as e:
        return f"Error collecting spectra: {str(e)}"

@mcp.tool(
    name="Perform_PCA_Analysis",
    description="Perform Principal Component Analysis (PCA) on collected spectra to reduce dimensionality"
)
def perform_pca_analysis(n_components: int = 2) -> str:
    """
    Perform PCA on collected spectra.
    
    Args:
        n_components: Number of principal components (default: 2)
    """
    try:
        if len(_current_data["spectra"]) == 0:
            return "No spectra data available. Please collect spectra first using Collect_Grid_Spectra."
        
        pca = PCA(n_components=n_components)
        data_pca = pca.fit_transform(_current_data["spectra"])
        
        # Store PCA results
        _current_data["pca_results"] = data_pca
        
        explained_variance = pca.explained_variance_ratio_
        
        # Save PCA visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if n_components == 2:
            # 2D scatter plot
            filepath = os.path.join(OUTPUT_DIR, f"pca_analysis_{n_components}components_{timestamp}.png")
            
            plt.figure(figsize=(10, 8))
            plt.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.6, s=50)
            plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
            plt.title(f'PCA Analysis - {n_components} Components')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            _save_image_info(filepath, f"PCA scatter plot ({n_components} components)")
        else:
            # Multiple component visualization
            filepath = os.path.join(OUTPUT_DIR, f"pca_analysis_{n_components}components_{timestamp}.png")
            
            n_plots = min(n_components, 4)
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'PCA Analysis - First {n_plots} Components', fontsize=14)
            
            for idx, ax in enumerate(axes.flat):
                if idx < n_components:
                    ax.hist(data_pca[:, idx], bins=30, alpha=0.7)
                    ax.set_title(f'PC{idx+1} ({explained_variance[idx]:.2%} variance)')
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Frequency')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            _save_image_info(filepath, f"PCA component distributions ({n_components} components)")
        
        result = f"PCA completed successfully.\n"
        result += f"Reduced from {_current_data['spectra'].shape[1]} to {n_components} dimensions.\n"
        result += f"PCA shape: {data_pca.shape}\n"
        result += f"Explained variance ratio: {explained_variance}\n"
        result += f"Total variance explained: {sum(explained_variance):.2%}\n"
        
        abs_path = os.path.abspath(filepath)
        result += f"\n📸 PCA visualization saved to: {abs_path}"
        
        return result
    except Exception as e:
        return f"Error performing PCA: {str(e)}"

@mcp.tool(
    name="Perform_Clustering",
    description="Perform K-means clustering on PCA-reduced data or original spectra"
)
def perform_clustering(
    n_clusters: int = 3,
    use_pca: bool = True,
    random_state: int = 42
) -> str:
    """
    Perform K-means clustering.
    
    Args:
        n_clusters: Number of clusters (default: 3)
        use_pca: Use PCA-reduced data if available (default: True)
        random_state: Random state for reproducibility (default: 42)
    """
    try:
        # Determine which data to use
        if use_pca and _current_data["pca_results"] is not None:
            data = _current_data["pca_results"]
            data_type = "PCA-reduced data"
        elif len(_current_data["spectra"]) > 0:
            data = _current_data["spectra"]
            data_type = "original spectra"
        else:
            return "No data available. Please collect spectra first using Collect_Grid_Spectra."
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        clusters = kmeans.fit_predict(data)
        
        # Store results
        _current_data["clusters"] = clusters
        
        # Count samples per cluster
        cluster_counts = {}
        for i in range(n_clusters):
            cluster_counts[f"Cluster {i}"] = int(np.sum(clusters == i))
        
        # Save clustering visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(OUTPUT_DIR, f"clustering_{n_clusters}clusters_{timestamp}.png")
        
        fig = plt.figure(figsize=(14, 6))
        
        # Left plot: Scatter plot if using PCA with 2 components
        if use_pca and data.shape[1] >= 2:
            ax1 = plt.subplot(1, 2, 1)
            scatter = ax1.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', 
                                alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            ax1.set_xlabel('PC1')
            ax1.set_ylabel('PC2')
            ax1.set_title(f'K-means Clustering ({n_clusters} clusters)')
            ax1.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax1, label='Cluster')
        else:
            ax1 = plt.subplot(1, 2, 1)
            ax1.text(0.5, 0.5, 'Clustering performed\non high-dimensional data', 
                    ha='center', va='center', fontsize=12)
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.axis('off')
        
        # Right plot: Cluster distribution bar chart
        ax2 = plt.subplot(1, 2, 2)
        cluster_ids = list(range(n_clusters))
        counts = [cluster_counts[f"Cluster {i}"] for i in cluster_ids]
        bars = ax2.bar(cluster_ids, counts, color=plt.cm.viridis(np.linspace(0, 1, n_clusters)))
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Number of Points')
        ax2.set_title('Cluster Distribution')
        ax2.set_xticks(cluster_ids)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        _save_image_info(filepath, f"K-means clustering ({n_clusters} clusters)")
        
        # Also save spatial map if we have locations
        if len(_current_data["locations"]) > 0 and _current_data["overview_image"] is not None:
            filepath_map = os.path.join(OUTPUT_DIR, f"clustering_spatial_map_{n_clusters}clusters_{timestamp}.png")
            
            # Create spatial cluster map
            grid_x = max([loc[0] for loc in _current_data["locations"]]) + 1
            grid_y = max([loc[1] for loc in _current_data["locations"]]) + 1
            cluster_map = np.zeros((grid_x, grid_y))
            
            for idx, (x, y) in enumerate(_current_data["locations"]):
                cluster_map[x, y] = clusters[idx]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Overview image
            im1 = ax1.imshow(_current_data["overview_image"], cmap='gray')
            ax1.set_title('Overview Image')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            plt.colorbar(im1, ax=ax1, label='Intensity')
            
            # Cluster map
            im2 = ax2.imshow(cluster_map.T, cmap='viridis', interpolation='nearest')
            ax2.set_title(f'Spatial Cluster Map ({n_clusters} clusters)')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            plt.colorbar(im2, ax=ax2, label='Cluster ID')
            
            plt.tight_layout()
            plt.savefig(filepath_map, dpi=150, bbox_inches='tight')
            plt.close()
            
            _save_image_info(filepath_map, f"Spatial cluster map ({n_clusters} clusters)")
            abs_path_map = os.path.abspath(filepath_map)
        
        result = f"K-means clustering completed successfully.\n"
        result += f"Data used: {data_type}\n"
        result += f"Number of clusters: {n_clusters}\n"
        result += f"Data shape: {data.shape}\n"
        result += f"\nCluster distribution:\n"
        for cluster, count in cluster_counts.items():
            result += f"  {cluster}: {count} samples\n"
        
        abs_path = os.path.abspath(filepath)
        result += f"\n📸 Clustering visualization saved to: {abs_path}"
        
        if len(_current_data["locations"]) > 0 and _current_data["overview_image"] is not None:
            result += f"\n📸 Spatial cluster map saved to: {abs_path_map}"
        
        return result
    except Exception as e:
        return f"Error performing clustering: {str(e)}"

@mcp.tool(
    name="Get_Analysis_Summary",
    description="Get a summary of all current analysis results including spectra, PCA, and clustering"
)
def get_analysis_summary() -> str:
    """
    Get a comprehensive summary of current analysis state.
    """
    summary = "=== STEM Analysis Summary ===\n\n"
    
    # Overview image
    if _current_data["overview_image"] is not None:
        summary += f"Overview Image: Available (shape: {_current_data['overview_image'].shape})\n\n"
    else:
        summary += "Overview Image: Not loaded\n\n"
    
    # Spectra
    if len(_current_data["spectra"]) > 0:
        summary += f"Spectra Data:\n"
        summary += f"  - Number of spectra: {len(_current_data['spectra'])}\n"
        summary += f"  - Spectrum length: {_current_data['spectra'].shape[1]}\n"
        summary += f"  - Locations: {len(_current_data['locations'])} points\n\n"
    else:
        summary += "Spectra Data: No spectra collected\n\n"
    
    # PCA
    if _current_data["pca_results"] is not None:
        summary += f"PCA Results:\n"
        summary += f"  - Shape: {_current_data['pca_results'].shape}\n"
        summary += f"  - Components: {_current_data['pca_results'].shape[1]}\n\n"
    else:
        summary += "PCA Results: Not performed\n\n"
    
    # Clustering
    if _current_data["clusters"] is not None:
        n_clusters = len(np.unique(_current_data["clusters"]))
        summary += f"Clustering Results:\n"
        summary += f"  - Number of clusters: {n_clusters}\n"
        summary += f"  - Samples clustered: {len(_current_data['clusters'])}\n\n"
    else:
        summary += "Clustering Results: Not performed\n\n"
    
    # Saved images
    if len(_current_data["saved_images"]) > 0:
        summary += f"📸 Saved Images ({len(_current_data['saved_images'])} total):\n"
        for img_info in _current_data["saved_images"]:
            summary += f"  - {img_info['description']}\n"
            summary += f"    Path: {os.path.abspath(img_info['filepath'])}\n"
            summary += f"    Time: {img_info['timestamp']}\n"
    else:
        summary += "📸 Saved Images: None\n"
    
    return summary

@mcp.tool(
    name="Export_Analysis_Data",
    description="Export current analysis data as JSON for further processing"
)
def export_analysis_data() -> str:
    """
    Export analysis data in JSON format.
    Returns cluster assignments and locations if available.
    """
    try:
        export_data = {}
        
        if _current_data["clusters"] is not None and len(_current_data["locations"]) > 0:
            export_data["cluster_map"] = [
                {
                    "location": {"x": loc[0], "y": loc[1]},
                    "cluster": int(_current_data["clusters"][i])
                }
                for i, loc in enumerate(_current_data["locations"])
            ]
        
        if _current_data["pca_results"] is not None:
            export_data["pca_summary"] = {
                "shape": list(_current_data["pca_results"].shape),
                "n_components": _current_data["pca_results"].shape[1]
            }
        
        if len(export_data) == 0:
            return "No analysis data available to export."
        
        return json.dumps(export_data, indent=2)
    except Exception as e:
        return f"Error exporting data: {str(e)}"

@mcp.tool(
    name="Reset_Analysis",
    description="Clear all stored analysis data and start fresh"
)
def reset_analysis() -> str:
    """
    Reset all stored analysis data.
    """
    _current_data["spectra"] = []
    _current_data["locations"] = []
    _current_data["overview_image"] = None
    _current_data["pca_results"] = None
    _current_data["clusters"] = None
    _current_data["saved_images"] = []
    
    return "All analysis data has been reset. Ready for new analysis.\n\n(Note: Previously saved image files in the 'stem_analysis_images' folder have NOT been deleted)"

if __name__ == "__main__":
    mcp.run(transport="stdio")