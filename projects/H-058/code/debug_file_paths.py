"""
Debug the file path construction in _load_h5_file
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pathlib import Path
from vantascope.data.dft_graphene_loader import DFTGrapheneDataset
from vantascope.utils.logging import logger

def debug_file_paths():
    """Debug how file paths are being constructed."""
    
    logger.info("🔍 Debugging file paths...")
    
    # Create dataset
    dataset = DFTGrapheneDataset(
        data_path="data/train",
        split='train',
        max_samples=3
    )
    
    logger.info(f"📊 Dataset data_path: {dataset.data_path}")
    logger.info(f"📊 Dataset split: {dataset.split}")
    logger.info(f"📊 Dataset h5_files: {dataset.h5_files[:3]}")
    
    # Test the _load_h5_file method directly
    for i, h5_file in enumerate(dataset.h5_files[:3]):
        logger.info(f"\n🧪 Testing file {i}: {h5_file}")
        
        # Show what path would be constructed
        if dataset.split == 'train':
            full_path = Path(dataset.data_path).parent / 'train' / h5_file
        else:
            full_path = Path(dataset.data_path).parent / 'test' / h5_file
        
        logger.info(f"   Constructed path: {full_path}")
        logger.info(f"   Path exists: {full_path.exists()}")
        
        # Test the actual _load_h5_file method
        data = dataset._load_h5_file(h5_file)
        logger.info(f"   _load_h5_file result: {data}")
        
        if data:
            logger.info(f"   Energy in data: {data.get('energy', 'MISSING')}")
        
        # Test direct file access
        if full_path.exists():
            try:
                import h5py
                with h5py.File(full_path, 'r') as f:
                    energy = f['energy'][...]
                    logger.info(f"   Direct H5 energy: {energy}")
            except Exception as e:
                logger.error(f"   Direct H5 failed: {e}")

if __name__ == "__main__":
    debug_file_paths()
