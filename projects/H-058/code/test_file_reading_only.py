import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import tarfile
import h5py
import io
from vantascope.utils.logging import logger

def test_raw_file_reading():
    logger.info("🚀 Testing raw file reading speed...")
    
    start_time = time.time()
    count = 0
    
    with tarfile.open("data/BigGrapheneDataset.tar.gz", 'r:gz') as tar:
        # Get first 100 train files
        train_files = [f for f in tar.getnames() if f.endswith('.h5') and f.startswith('train/')][:100]
        
        logger.info(f"Found {len(train_files)} train files")
        
        for filename in train_files:
            # Extract and read H5 file
            file_obj = tar.extractfile(filename)
            file_data = file_obj.read()
            
            # Parse H5 in memory
            with h5py.File(io.BytesIO(file_data), 'r') as h5_file:
                coords = h5_file['coordinates'][:]
                energy = h5_file['energy'][()]
                
            count += 1
            if count % 10 == 0:
                elapsed = time.time() - start_time
                logger.info(f"   Processed {count} files in {elapsed:.1f}s ({count/elapsed:.1f} files/sec)")
    
    total_time = time.time() - start_time
    logger.info(f"📊 Raw file reading: {count/total_time:.1f} files/sec")
    logger.info(f"📊 Estimated for 100k files: {100000/(count/total_time)/60:.1f} minutes")

if __name__ == "__main__":
    test_raw_file_reading()
