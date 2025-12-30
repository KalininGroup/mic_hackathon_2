"""
Test the full data pipeline to find where energies become zero
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from torch.utils.data import DataLoader

from vantascope.data.dft_graphene_loader import DFTGrapheneDataset, collate_dft_batch
from vantascope.utils.logging import logger

def test_full_pipeline():
    """Test the complete pipeline from dataset to batch."""
    
    logger.info("🔍 Testing full data pipeline...")
    
    # Create dataset
    dataset = DFTGrapheneDataset(
        data_path="data/train",
        split='train',
        max_samples=10,
        grid_size=512
    )
    
    logger.info(f"📊 Dataset created with {len(dataset)} samples")
    
    # Test individual samples first
    logger.info("\n🧪 Testing individual samples:")
    for i in range(min(3, len(dataset))):
        try:
            sample = dataset[i]
            logger.info(f"   Sample {i}:")
            logger.info(f"     Energy: {sample['energy']} (shape: {sample['energy'].shape})")
            logger.info(f"     Image: {sample['image'].shape}")
            logger.info(f"     Coordinates: {sample['coordinates'].shape}")
        except Exception as e:
            logger.error(f"   Sample {i} failed: {e}")
    
    # Test with DataLoader
    logger.info("\n🔄 Testing with DataLoader:")
    dataloader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        collate_fn=collate_dft_batch,
        num_workers=0
    )
    
    for batch_idx, batch in enumerate(dataloader):
        logger.info(f"   Batch {batch_idx}:")
        logger.info(f"     Energy batch: {batch['energy']} (shape: {batch['energy'].shape})")
        logger.info(f"     Image batch: {batch['image'].shape}")
        
        # Check individual energies in batch
        for i, energy in enumerate(batch['energy']):
            logger.info(f"       Sample {i} energy: {energy.item()}")
        
        if batch_idx >= 1:  # Test 2 batches
            break

if __name__ == "__main__":
    test_full_pipeline()
