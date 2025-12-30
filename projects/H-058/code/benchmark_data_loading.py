import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import torch
from torch.utils.data import DataLoader
from vantascope.data.dft_graphene_loader import DFTGrapheneDataset, collate_dft_batch
from vantascope.utils.logging import logger

def benchmark_loading():
    logger.info("🚀 Benchmarking data loading speed...")
    
    # Test loading speed
    dataset = DFTGrapheneDataset(
        data_path="data/BigGrapheneDataset.tar.gz",
        split='train',
        max_samples=1000  # Just test first 1000
    )
    
    logger.info(f"📊 Dataset size: {len(dataset)} samples")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=False, 
        collate_fn=collate_dft_batch,
        num_workers=2
    )
    
    start_time = time.time()
    
    # Load first few batches to estimate speed
    for i, batch in enumerate(dataloader):
        if i >= 10:  # Test 10 batches
            break
        logger.info(f"   Batch {i}: {batch['image'].shape}")
    
    elapsed = time.time() - start_time
    samples_processed = min(len(dataset), 10 * 4)  # 10 batches * batch_size
    samples_per_sec = samples_processed / elapsed
    
    logger.info(f"📊 Loading speed: {samples_per_sec:.1f} samples/sec")
    logger.info(f"📊 Estimated time for 100k samples: {100000/samples_per_sec/60:.1f} minutes")
    logger.info(f"📊 Estimated time for full dataset: {500000/samples_per_sec/60:.1f} minutes")

if __name__ == "__main__":
    benchmark_loading()
