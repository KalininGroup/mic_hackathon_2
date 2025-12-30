"""
Debug the actual structure of BigGrapheneDataset files.
"""

import tarfile
import h5py
import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vantascope.utils.logging import logger

def debug_dataset_structure():
    """Examine the actual structure of the dataset."""
    data_path = "data/BigGrapheneDataset.tar.gz"
    
    logger.info("🔍 Debugging BigGrapheneDataset structure...")
    
    with tarfile.open(data_path, 'r:gz') as tar:
        # Get file list
        all_files = tar.getnames()
        h5_files = [f for f in all_files if f.endswith('.h5')]
        
        logger.info(f"📦 Total files: {len(all_files)}")
        logger.info(f"📄 H5 files: {len(h5_files)}")
        logger.info(f"📂 Sample files: {h5_files[:5]}")
        
        # Examine first H5 file
        if h5_files:
            first_file = h5_files[0]
            logger.info(f"\n🔬 Examining: {first_file}")
            
            try:
                # Extract and examine
                h5_bytes = tar.extractfile(first_file).read()
                with h5py.File(io.BytesIO(h5_bytes), 'r') as f:
                    logger.info(f"📋 Root keys: {list(f.keys())}")
                    
                    # Examine each key
                    for key in f.keys():
                        try:
                            data = f[key]
                            if hasattr(data, 'shape'):
                                logger.info(f"   {key}: shape={data.shape}, dtype={data.dtype}")
                                if data.size < 20:  # Small arrays
                                    logger.info(f"      Values: {data[...]}")
                                else:
                                    logger.info(f"      Range: [{data[...].min():.3f}, {data[...].max():.3f}]")
                            else:
                                logger.info(f"   {key}: (group) -> {list(data.keys())}")
                        except Exception as e:
                            logger.info(f"   {key}: Error reading - {e}")
                            
            except Exception as e:
                logger.error(f"Failed to read {first_file}: {e}")

if __name__ == "__main__":
    debug_dataset_structure()
