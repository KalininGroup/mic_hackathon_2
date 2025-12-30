"""
Explore the downloaded graphene dataset structure.
"""

import h5py
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from vantascope.utils.logging import logger

def explore_graphene_folders():
    """Explore graphene dataset structure."""
    data_dir = Path("data/graphene")
    
    if not data_dir.exists():
        logger.error(f"Graphene data not found at {data_dir}")
        return
    
    logger.info(f"🔍 Exploring graphene data at {data_dir}")
    
    for folder in data_dir.iterdir():
        if folder.is_dir():
            logger.info(f"\n📁 Folder: {folder.name}")
            
            # Count file types
            h5_files = list(folder.glob("*.h5"))
            png_files = list(folder.glob("*.png"))
            
            logger.info(f"   H5 files: {len(h5_files)}")
            logger.info(f"   PNG files: {len(png_files)}")
            
            # Examine first H5 file
            if h5_files:
                examine_h5_file(h5_files[0])

def examine_h5_file(h5_path: Path):
    """Examine structure of an H5 file."""
    logger.info(f"   🔬 Examining: {h5_path.name}")
    
    try:
        with h5py.File(h5_path, 'r') as f:
            logger.info(f"      Keys: {list(f.keys())}")
            
            # Look at each dataset
            def print_dataset_info(name, obj):
                if isinstance(obj, h5py.Dataset):
                    logger.info(f"         {name}: {obj.shape} {obj.dtype}")
                    
                    # Sample a small region if it's an image
                    if len(obj.shape) >= 2 and obj.shape[-1] > 10 and obj.shape[-2] > 10:
                        sample = obj[..., :5, :5] if len(obj.shape) > 2 else obj[:5, :5]
                        logger.info(f"            Sample: {sample.flatten()[:3]}...")
            
            f.visititems(print_dataset_info)
            
    except Exception as e:
        logger.error(f"      Failed to read {h5_path.name}: {e}")

if __name__ == "__main__":
    explore_graphene_folders()
