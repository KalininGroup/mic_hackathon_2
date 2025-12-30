import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
from torch.utils.data import DataLoader
from vantascope.data.dft_graphene_loader import DFTGrapheneDataset, collate_dft_batch
from vantascope.utils.logging import logger

def test_fixed_dataset():
    logger.info("🚀 Testing fixed dataset with extracted files...")
    
    # Create dataset pointing to extracted directory
    dataset = DFTGrapheneDataset(
        data_path="data/train",  # Point to extracted directory
        split='train',
        max_samples=100  # Test with 100 samples
    )
    
    logger.info(f"📊 Dataset loaded: {len(dataset)} samples")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=False, 
        collate_fn=collate_dft_batch,
        num_workers=2
    )
    
    start_time = time.time()
    
    for i, batch in enumerate(dataloader):
        if i >= 5:  # Test 5 batches
            break
        logger.info(f"   Batch {i}: Images {batch['image'].shape}, Energy {batch['energy'].shape}")
    
    elapsed = time.time() - start_time
    samples_processed = min(len(dataset), 5 * 8)
    samples_per_sec = samples_processed / elapsed
    
    logger.info(f"📊 Full pipeline with DataLoader: {samples_per_sec:.1f} samples/sec")
    logger.info(f"📊 Estimated full training time: {501473/samples_per_sec/3600:.1f} hours")

if __name__ == "__main__":
    test_fixed_dataset()
