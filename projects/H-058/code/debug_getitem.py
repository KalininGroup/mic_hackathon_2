"""
Debug the __getitem__ method to see where it fails
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import h5py
import torch
import numpy as np
from pathlib import Path
from vantascope.utils.logging import logger

def debug_getitem_manually():
    """Manually step through the __getitem__ logic."""
    
    logger.info("🔍 Debugging __getitem__ manually...")
    
    # Get the first H5 file
    train_dir = Path("data/train")
    h5_files = list(train_dir.glob("*.h5"))
    h5_file = h5_files[0]
    
    logger.info(f"📁 Testing file: {h5_file.name}")
    
    try:
        # Step 1: Load H5 data
        logger.info("Step 1: Loading H5 data...")
        with h5py.File(h5_file, 'r') as f:
            coordinates = f['coordinates'][...]
            energy = f['energy'][...]
            cell = f['cell'][...]
        
        logger.info(f"   ✅ Loaded - Energy: {energy}, Coords: {coordinates.shape}, Cell: {cell.shape}")
        
        # Step 2: Create density map (this might be failing)
        logger.info("Step 2: Creating density map...")
        # We need to import the actual dataset class to use its methods
        from vantascope.data.dft_graphene_loader import DFTGrapheneDataset
        
        # Create a dataset instance to access its methods
        dataset = DFTGrapheneDataset("data/train", max_samples=1)
        
        try:
            density_map = dataset._create_density_map(coordinates, cell)
            logger.info(f"   ✅ Density map created: {density_map.shape}")
        except Exception as e:
            logger.error(f"   ❌ Density map failed: {e}")
            return
        
        # Step 3: Extract defect info
        logger.info("Step 3: Extracting defect info...")
        try:
            defect_info = dataset._extract_defect_info(coordinates, cell)
            logger.info(f"   ✅ Defect info: {defect_info}")
        except Exception as e:
            logger.error(f"   ❌ Defect info failed: {e}")
            return
        
        # Step 4: Convert to tensors
        logger.info("Step 4: Converting to tensors...")
        try:
            image_tensor = torch.from_numpy(density_map).unsqueeze(0).float()
            energy_tensor = torch.from_numpy(energy).float()
            coordinates_tensor = torch.from_numpy(coordinates).float()
            
            logger.info(f"   ✅ Tensors created:")
            logger.info(f"      Image: {image_tensor.shape}")
            logger.info(f"      Energy: {energy_tensor} (shape: {energy_tensor.shape})")
            logger.info(f"      Coordinates: {coordinates_tensor.shape}")
            
        except Exception as e:
            logger.error(f"   ❌ Tensor conversion failed: {e}")
            return
        
        logger.info("🎉 All steps successful!")
        
    except Exception as e:
        logger.error(f"❌ Failed at H5 loading: {e}")

if __name__ == "__main__":
    debug_getitem_manually()
