"""
Check the actual size of the BigGrapheneDataset.
"""

import tarfile
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vantascope.utils.logging import logger

def check_dataset_size():
    """Check actual dataset size."""
    data_path = "data/BigGrapheneDataset.tar.gz"
    
    logger.info("📊 Checking BigGrapheneDataset size...")
    
    with tarfile.open(data_path, 'r:gz') as tar:
        all_files = tar.getnames()
        h5_files = [f for f in all_files if f.endswith('.h5')]
        
        # Count by split
        train_files = [f for f in h5_files if f.startswith('train/')]
        test_files = [f for f in h5_files if f.startswith('test/')]
        
        logger.info(f"📦 Total files: {len(all_files):,}")
        logger.info(f"📄 H5 files: {len(h5_files):,}")
        logger.info(f"🚂 Training files: {len(train_files):,}")
        logger.info(f"🧪 Testing files: {len(test_files):,}")
        
        # Show folder structure
        folders = set()
        for f in all_files[:100]:  # First 100 files
            if '/' in f:
                folders.add(f.split('/')[0])
        
        logger.info(f"📁 Top-level folders: {sorted(folders)}")

if __name__ == "__main__":
    check_dataset_size()
