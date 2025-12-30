"""
Test DFT Dataset Loader
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vantascope.data.dft_dataset import create_dft_dataset, create_dft_dataloader
from vantascope.utils.logging import logger

def test_dft_dataset():
    """Test DFT dataset loading."""
    logger.info("🧪 Testing DFT Dataset Loader")
    
    # Test with dummy tar path (will use fallback dummy data)
    dummy_tar_path = "dummy_dataset.tar.gz"
    
    # Create dataset
    dataset = create_dft_dataset(
        tar_path=dummy_tar_path,
        split='train',
        max_samples=10  # Small test
    )
    
    logger.info(f"   Dataset length: {len(dataset)}")
    
    # Test single sample
    sample = dataset[0]
    
    logger.info(f"   Sample keys: {list(sample.keys())}")
    logger.info(f"   Image shape: {sample['image'].shape}")
    logger.info(f"   Energy: {sample['energy'].item():.1f} eV")
    logger.info(f"   Coordinates shape: {sample['coordinates'].shape}")
    logger.info(f"   Metadata: {sample['metadata']}")
    
    # Test dataloader
    dataloader = create_dft_dataloader(
        tar_path=dummy_tar_path,
        batch_size=4,
        max_samples=10,
        num_workers=0  # Avoid multiprocessing issues in test
    )
    
    logger.info(f"   DataLoader created with {len(dataloader)} batches")
    
    # Test batch loading
    for i, batch in enumerate(dataloader):
        logger.info(f"   Batch {i}:")
        logger.info(f"     Images: {batch['image'].shape}")
        logger.info(f"     Energies: {batch['energy'].shape}")
        logger.info(f"     Coordinates: {len(batch['coordinates'])} samples")
        logger.info(f"     Energy range: [{batch['energy'].min().item():.1f}, {batch['energy'].max().item():.1f}] eV")
        
        if i >= 1:  # Test first 2 batches
            break
    
    logger.info("🎉 DFT Dataset test completed successfully!")
    
    return dataset, dataloader

if __name__ == "__main__":
    test_dft_dataset()
