"""
Debug: Explore actual structure of graphene and PZT data files.
"""

import h5py
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from vantascope.utils.logging import logger

def debug_graphene_files():
    """Debug graphene folder structure."""
    graphene_dir = Path("data/graphene")
    
    if not graphene_dir.exists():
        logger.error(f"Graphene directory not found: {graphene_dir}")
        return
    
    logger.info(f"🔍 Debugging graphene data structure...")
    
    for folder in sorted(graphene_dir.iterdir()):
        if folder.is_dir():
            logger.info(f"\n📁 Folder: {folder.name}")
            
            # List all files
            h5_files = list(folder.glob("*.h5"))
            png_files = list(folder.glob("*.png"))
            other_files = list(folder.glob("*"))
            
            logger.info(f"   Files: H5={len(h5_files)}, PNG={len(png_files)}, Total={len(other_files)}")
            
            # Examine first H5 file in detail
            if h5_files:
                debug_h5_file(h5_files[0])

def debug_h5_file(h5_path: Path):
    """Debug a single H5 file in detail."""
    logger.info(f"   🔬 Detailed examination: {h5_path.name}")
    
    try:
        with h5py.File(h5_path, 'r') as f:
            logger.info(f"      Root keys: {list(f.keys())}")
            
            # Recursively explore all datasets
            def explore_item(name, obj):
                if isinstance(obj, h5py.Dataset):
                    logger.info(f"         Dataset '{name}': shape={obj.shape}, dtype={obj.dtype}")
                    
                    # Check if it looks like image data
                    if len(obj.shape) >= 2:
                        sample = np.array(obj[..., :3, :3])  # Small sample
                        logger.info(f"           Sample values: {sample.flatten()[:5]}")
                elif isinstance(obj, h5py.Group):
                    logger.info(f"         Group '{name}': {len(obj)} items")
            
            f.visititems(explore_item)
            
    except Exception as e:
        logger.error(f"      Error reading {h5_path.name}: {e}")

def debug_pzt_file():
    """Debug PZT file structure."""
    pzt_path = Path("data/PZT_final0001.h5")
    
    if not pzt_path.exists():
        logger.warning(f"PZT file not found: {pzt_path}")
        return
    
    logger.info(f"\n🔬 Debugging PZT file: {pzt_path}")
    
    try:
        with h5py.File(pzt_path, 'r') as f:
            logger.info(f"Root keys: {list(f.keys())}")
            
            def explore_pzt(name, obj):
                if isinstance(obj, h5py.Dataset):
                    logger.info(f"   Dataset '{name}': shape={obj.shape}, dtype={obj.dtype}")
                    if obj.attrs:
                        logger.info(f"      Attributes: {dict(obj.attrs)}")
                elif isinstance(obj, h5py.Group):
                    logger.info(f"   Group '{name}': {len(obj)} items")
                    if obj.attrs:
                        logger.info(f"      Attributes: {dict(obj.attrs)}")
            
            f.visititems(explore_pzt)
            
    except Exception as e:
        logger.error(f"Error reading PZT file: {e}")

def list_actual_files():
    """List what files actually exist."""
    logger.info(f"\n📋 Listing actual files...")
    
    # Check graphene
    graphene_dir = Path("data/graphene")
    if graphene_dir.exists():
        logger.info(f"Graphene folder exists: {graphene_dir}")
        for item in graphene_dir.iterdir():
            if item.is_dir():
                h5_count = len(list(item.glob("*.h5")))
                logger.info(f"   {item.name}: {h5_count} H5 files")
    else:
        logger.info(f"Graphene folder missing: {graphene_dir}")
    
    # Check PZT
    pzt_path = Path("data/PZT_final0001.h5")
    logger.info(f"PZT file exists: {pzt_path.exists()}")

if __name__ == "__main__":
    list_actual_files()
    debug_graphene_files()
    debug_pzt_file()
