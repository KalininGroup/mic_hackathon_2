import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import h5py
import torch
from pathlib import Path
from vantascope.data.gaussian_splatting import GaussianSplatter
from vantascope.utils.logging import logger

def test_direct_loading():
    logger.info("🚀 Testing direct loading with Gaussian splatting...")
    
    # Initialize Gaussian splatting
    splatter = GaussianSplatter(image_size=512)
    
    # Get some H5 files
    train_dir = Path("data/train")
    h5_files = list(train_dir.glob("*.h5"))[:50]  # Test 50 files
    
    logger.info(f"Found {len(h5_files)} files to test")
    
    start_time = time.time()
    
    for i, h5_path in enumerate(h5_files):
        # Load H5 data
        with h5py.File(h5_path, 'r') as h5_file:
            coords = h5_file['coordinates'][:]
            energy = h5_file['energy'][()]
            cell = h5_file['cell'][:]
        
        # Convert to image using Gaussian splatting
        image = splatter.coordinates_to_image(coords, cell)
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            logger.info(f"   Processed {i+1} files in {elapsed:.1f}s ({rate:.1f} samples/sec)")
    
    total_time = time.time() - start_time
    final_rate = len(h5_files) / total_time
    
    logger.info(f"📊 Full pipeline: {final_rate:.1f} samples/sec")
    logger.info(f"📊 Estimated for full dataset: {501473/final_rate/3600:.1f} hours")

if __name__ == "__main__":
    test_direct_loading()
