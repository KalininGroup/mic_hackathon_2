# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mcp[cli]>=1.12.3",
#     "pydantic>=2.11.7",
#     "python-dotenv>=1.1.1",
#     "numpy>=1.24.0",
#     "torch>=2.0.0",
#     "matplotlib>=3.7.2",
#     "Pillow>=10.0.0",
# ]
# ///

from mcp.server.fastmcp import FastMCP
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
from PIL import Image
import torch

mcp = FastMCP("ImageSegmentation")

# Global variables
_model = None
_model_path = None

# Create output directory
OUTPUT_DIR = "segmentation_results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Model path
DEFAULT_MODEL_PATH = r"C:\Users\jcruanes\Documents\GitHub\Hackathon-2025_MCP-Server-With-Claude\ML_Model\model.pt"

def _load_model():
    """Load model on demand"""
    global _model, _model_path
    
    if _model is not None:
        return _model  # Already loaded
    
    try:
        print(f"Loading model from: {DEFAULT_MODEL_PATH}", file=sys.stderr)
        
        if not os.path.exists(DEFAULT_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {DEFAULT_MODEL_PATH}")
        
        _model = torch.load(DEFAULT_MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
        _model.eval()  # Set to evaluation mode
        _model_path = DEFAULT_MODEL_PATH
        
        print(f"✓ Model loaded successfully", file=sys.stderr)
        return _model
        
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}", file=sys.stderr)
        raise

@mcp.tool(
    name="Check_Model_Status",
    description="Check if the segmentation model is loaded and ready"
)
def check_model_status() -> str:
    """
    Check the status of the loaded model.
    """
    if _model is None:
        return f"⏸️ Model NOT yet loaded (will load on first use)\n\nModel path: {DEFAULT_MODEL_PATH}\nFile exists: {os.path.exists(DEFAULT_MODEL_PATH)}"
    else:
        return f"✅ Model IS loaded\n\nModel path: {_model_path}\nModel type: {type(_model)}\n\nModel is ready for segmentation!"

@mcp.tool(
    name="Segment_Image",
    description="Apply segmentation model to an image and return the original and predicted segmentation mask"
)
def segment_image(image_path: str) -> str:
    """
    Apply segmentation to an image and save both original and prediction.
    
    Args:
        image_path: Path to the input image file
    """
    try:
        # Load model on demand
        model = _load_model()
        
        # Convert to absolute path
        abs_path = os.path.abspath(image_path)
        
        # Check if file exists
        if not os.path.exists(abs_path):
            return f"Error: Image file not found at path: {abs_path}"
        
        # Load image
        print(f"Loading image from: {abs_path}", file=sys.stderr)
        image = Image.open(abs_path)
        image_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            image_array = np.mean(image_array, axis=2)
        
        # Run model prediction
        print("Running model prediction...", file=sys.stderr)
        with torch.no_grad():
            pred, peaks = model.predict(image_array)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(OUTPUT_DIR, f"segmentation_{timestamp}.png")
        
        # Create side-by-side visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        axes[0].imshow(image_array, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Prediction
        axes[1].imshow(pred, cmap='jet')
        axes[1].set_title('Segmentation Prediction')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        abs_filepath = os.path.abspath(filepath)
        
        result = f"✅ Segmentation complete!\n\n"
        result += f"📸 Results saved to:\n{abs_filepath}\n\n"
        result += f"Detected {len(peaks)} peaks in segmentation"
        
        return result
    
    except Exception as e:
        return f"Error during segmentation: {str(e)}\n\nPlease check that the model file exists at: {DEFAULT_MODEL_PATH}"

if __name__ == "__main__":
    mcp.run(transport="stdio")