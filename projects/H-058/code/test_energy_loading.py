"""
Test energy loading directly from H5 files
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import h5py
import torch
import numpy as np
from pathlib import Path
from vantascope.utils.logging import logger

def test_energy_conversion():
    """Test how energy gets converted."""
    
    # Get a sample H5 file
    train_dir = Path("data/train")
    h5_files = list(train_dir.glob("*.h5"))[:5]  # Test 5 files
    
    for i, h5_file in enumerate(h5_files):
        logger.info(f"🔍 Testing file {i+1}: {h5_file.name[:20]}...")
        
        try:
            with h5py.File(h5_file, 'r') as f:
                # Read energy directly
                energy_raw = f['energy'][...]
                logger.info(f"   Raw energy: {energy_raw} (type: {type(energy_raw)}, shape: {energy_raw.shape})")
                
                # Convert to tensor (current method)
                energy_tensor = torch.from_numpy(energy_raw).float()
                logger.info(f"   Tensor energy: {energy_tensor} (shape: {energy_tensor.shape})")
                
                # Alternative conversion
                energy_scalar = float(energy_raw)
                energy_tensor_alt = torch.tensor(energy_scalar).float()
                logger.info(f"   Alternative: {energy_tensor_alt} (shape: {energy_tensor_alt.shape})")
                
        except Exception as e:
            logger.error(f"   Error loading {h5_file.name}: {e}")

if __name__ == "__main__":
    test_energy_conversion()
