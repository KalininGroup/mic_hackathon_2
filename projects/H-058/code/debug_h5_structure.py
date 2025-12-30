"""
Debug the actual H5 file structure
"""

import h5py
import numpy as np
from pathlib import Path
from vantascope.utils.logging import logger

def examine_h5_file():
    """Look at the actual structure of an H5 file."""
    
    # Get a sample H5 file
    train_dir = Path("data/train")
    h5_files = list(train_dir.glob("*.h5"))
    
    if not h5_files:
        logger.error("No H5 files found!")
        return
    
    sample_file = h5_files[0]
    logger.info(f"🔍 Examining: {sample_file.name}")
    
    with h5py.File(sample_file, 'r') as f:
        logger.info("📋 H5 File Contents:")
        
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                logger.info(f"   Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}")
                if obj.size < 10:  # Small datasets, show values
                    logger.info(f"      Values: {obj[...]}")
                else:  # Large datasets, show sample
                    logger.info(f"      Sample: {obj[...].flat[:5]}")
            else:
                logger.info(f"   Group: {name}")
        
        f.visititems(print_structure)
        
        # Try to access specific fields
        logger.info("\n🎯 Specific Field Access:")
        
        if 'energy' in f:
            energy = f['energy'][...]
            logger.info(f"   Energy: {energy} (type: {type(energy)})")
        else:
            logger.info("   ❌ No 'energy' field found")
            
        if 'coordinates' in f:
            coords = f['coordinates'][...]
            logger.info(f"   Coordinates shape: {coords.shape}")
            logger.info(f"   Coordinates sample: {coords[:3]}")
        else:
            logger.info("   ❌ No 'coordinates' field found")
            
        if 'cell' in f:
            cell = f['cell'][...]
            logger.info(f"   Cell: {cell}")
        else:
            logger.info("   ❌ No 'cell' field found")

if __name__ == "__main__":
    examine_h5_file()
