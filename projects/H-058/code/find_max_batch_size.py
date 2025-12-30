"""
Find Maximum Viable Batch Size for VantaScope Training
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import gc

from vantascope.models.integrated_model import create_vantascope_model_with_uncertainty
from vantascope.data.dft_graphene_loader import DFTGrapheneDataset, collate_dft_batch
from vantascope.utils.logging import logger
from vantascope.utils.helpers import get_device

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def test_batch_size(batch_size, with_graph=True):
    """Test if a specific batch size works."""
    logger.info(f"🧪 Testing batch size {batch_size} (graph={with_graph})...")
    
    device = get_device()
    clear_memory()
    
    try:
        # Create dataset
        dataset = DFTGrapheneDataset(
            data_path="data/train",
            split='train',
            max_samples=batch_size * 3  # Just enough for testing
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_dft_batch,
            num_workers=0
        )
        
        # Create model
        model = create_vantascope_model_with_uncertainty()
        model = model.to(device)
        model.eval()
        
        # Test forward pass
        for batch in dataloader:
            images = batch['image'].to(device)
            
            memory_before = torch.cuda.memory_allocated() / 1024**3
            
            with torch.no_grad():
                with autocast():
                    outputs = model(images, return_graph=with_graph)
            
            memory_after = torch.cuda.memory_allocated() / 1024**3
            memory_used = memory_after - memory_before
            
            logger.info(f"   ✅ Batch size {batch_size} works!")
            logger.info(f"   Memory used: {memory_used:.2f} GB")
            logger.info(f"   Total memory: {memory_after:.2f} GB")
            
            clear_memory()
            return True
            
    except torch.cuda.OutOfMemoryError:
        logger.info(f"   ❌ Batch size {batch_size} failed (OOM)")
        clear_memory()
        return False
    except Exception as e:
        logger.error(f"   ❌ Batch size {batch_size} failed: {e}")
        clear_memory()
        return False

def find_optimal_batch_size():
    """Binary search for optimal batch size."""
    logger.info("🔍 Finding optimal batch size...")
    
    # Test without graph first (uses less memory)
    logger.info("\n📊 Testing WITHOUT graph processing:")
    max_batch_no_graph = 1
    for batch_size in [1, 2, 3, 4, 6, 8, 12, 16]:
        if test_batch_size(batch_size, with_graph=False):
            max_batch_no_graph = batch_size
        else:
            break
    
    logger.info(f"\n🎯 Max batch size WITHOUT graph: {max_batch_no_graph}")
    
    # Test with graph
    logger.info("\n📊 Testing WITH graph processing:")
    max_batch_with_graph = 1
    for batch_size in [1, 2, 3, 4, 6, 8]:
        if test_batch_size(batch_size, with_graph=True):
            max_batch_with_graph = batch_size
        else:
            break
    
    logger.info(f"\n🎯 Max batch size WITH graph: {max_batch_with_graph}")
    
    return max_batch_no_graph, max_batch_with_graph

def test_training_loop(batch_size):
    """Test actual training loop with gradients."""
    logger.info(f"\n🔥 Testing training loop with batch size {batch_size}...")
    
    device = get_device()
    clear_memory()
    
    try:
        dataset = DFTGrapheneDataset(
            data_path="data/train",
            split='train',
            max_samples=50
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_dft_batch,
            num_workers=0
        )
        
        model = create_vantascope_model_with_uncertainty()
        model = model.to(device)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        
        for i, batch in enumerate(dataloader):
            if i >= 5:  # Test 5 batches
                break
                
            images = batch['image'].to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images, return_graph=True)
                loss = criterion(outputs['reconstruction'], images)
            
            loss.backward()
            optimizer.step()
            
            memory_used = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"   Batch {i}: Loss {loss.item():.4f}, Memory {memory_used:.2f} GB")
            
            clear_memory()
        
        logger.info(f"   ✅ Training loop with batch size {batch_size} works!")
        return True
        
    except torch.cuda.OutOfMemoryError:
        logger.info(f"   ❌ Training loop with batch size {batch_size} failed (OOM)")
        clear_memory()
        return False

if __name__ == "__main__":
    # Find optimal batch sizes
    max_no_graph, max_with_graph = find_optimal_batch_size()
    
    # Test training loop with the max viable batch size
    test_training_loop(max_with_graph)
    
    logger.info(f"\n🎉 RESULTS:")
    logger.info(f"   Max batch (no graph): {max_no_graph}")
    logger.info(f"   Max batch (with graph): {max_with_graph}")
    logger.info(f"   Recommended for training: {max_with_graph}")
