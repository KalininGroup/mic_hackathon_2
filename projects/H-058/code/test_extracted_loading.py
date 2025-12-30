import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import h5py
from pathlib import Path
from vantascope.utils.logging import logger

def test_extracted_loading():
    logger.info("🚀 Testing loading speed from extracted files...")
    
    train_dir = Path("data/train")
    h5_files = list(train_dir.glob("*.h5"))[:1000]  # Test first 1000
    
    logger.info(f"Found {len(h5_files)} files to test")
    
    start_time = time.time()
    
    for i, h5_path in enumerate(h5_files):
        with h5py.File(h5_path, 'r') as h5_file:
            coords = h5_file['coordinates'][:]
            energy = h5_file['energy'][()]
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            logger.info(f"   Processed {i+1} files in {elapsed:.1f}s ({rate:.1f} files/sec)")
    
    total_time = time.time() - start_time
    final_rate = len(h5_files) / total_time
    
    logger.info(f"📊 Extracted file reading: {final_rate:.1f} files/sec")
    logger.info(f"📊 Estimated for 100k files: {100000/final_rate/60:.1f} minutes")
    logger.info(f"📊 Estimated for full 501k files: {501473/final_rate/60:.1f} minutes")

if __name__ == "__main__":
    test_extracted_loading()
