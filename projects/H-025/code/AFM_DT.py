# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mcp>=1.0.0",
#     "Pyro5>=5.14",
#     "numpy>=1.24.0",
# ]
# ///

from mcp.server.fastmcp import FastMCP
import Pyro5.api
import numpy as np
import os

# Create the MCP Server
mcp = FastMCP("AFM_Digital_Twin")

# CONFIGURATION - Set your default path here so it can auto-initialize
DEFAULT_H5_PATH = "dset_spm1.h5" 

def get_initialized_mic(data_path=DEFAULT_H5_PATH, data_source="Compound_Dataset_1"):
    """
    Connects to the existing server and initializes it immediately.
    This bypasses the 'forgotten object' bug by ensuring the object 
    is created right before we use it.
    """
    uri = "PYRO:microscope.server@localhost:9092"
    mic = Pyro5.api.Proxy(uri)
    
    # Force initialization every time a tool is called 
    # to ensure the 'microscope' attribute exists in the current session.
    mic.initialize_microscope("AFM", data_path=data_path)
    mic.setup_microscope(data_source=data_source)
    return mic

@mcp.tool()
def initialize_afm(data_path: str, data_source: str = "Compound_Dataset_1"):
    """Initialize the AFM microscope and return dataset info."""
    mic = get_initialized_mic(data_path, data_source)
    info = mic.get_dataset_info()
    return f"✅ AFM Initialized!\nData: {data_path}\nInfo: {info}"

@mcp.tool()
def get_full_scan(channels: list[str], direction: str = "horizontal", trace: str = "forward"):
    """Get the complete 2D scan image. Auto-initializes if server forgot state."""
    # We use the helper to ensure the server has the microscope object ready
    mic = get_initialized_mic() 
    
    array_list, shape, dtype = mic.get_scan(
        channels=channels, 
        direction=direction, 
        trace=trace
    )
    
    dat = np.array(array_list, dtype=dtype).reshape(shape)
    
    result = f"✅ 2D Scan Completed (Shape: {shape})\n"
    for i, ch in enumerate(channels):
        result += f"- {ch}: [Mean: {np.mean(dat[i]):.3e}, Std: {np.std(dat[i]):.3e}]\n"
    return result

@mcp.tool()
def scan_individual_line(direction: str, coord: float, channels: list[str]):
    """Scan a single line at a specific coordinate."""
    mic = get_initialized_mic()
    array_list, shape, dtype = mic.scan_individual_line(direction, coord=coord, channels=channels)
    return f"✅ Line Scan at {coord}m complete. Points: {shape[-1]}"

if __name__ == "__main__":
    mcp.run(transport="stdio")