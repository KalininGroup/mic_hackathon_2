"""
Ultra-Conservative Training - Start with batch size 1
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import gc

from vantascope.models.integrated_model import create_vantascope_model_with_uncertainty
from vantascope.data.dft_graphene_loader import DFTGrapheneDataset, collate_dft_batch
from vantascope.training.enhanced_losses import EnhancedVantaScopeLoss
from vantascope.utils.logging import logger
from vantascope.utils.helpers import get_device

def clear_memory():
    """Aggressively clear memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def test_minimal_batch():
    """Test with absolute minimal memory usage."""
    logger.info("🔍 Testing minimal batch size...")
    
    device = get_device()
    
    # Create tiny dataset
    dataset = DFTGrapheneDataset(
        data_path="data/train",
        split='train',
        max_samples=10  # Just 10 samples for testing
    )
    
    # Batch size 1
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_dft_batch,
        num_workers=0  # No multiprocessing
    )
    
    # Create model
    model = create_vantascope_model_with_uncertainty()
    model = model.to(device)
    model.eval()
    
    logger.info("🧠 Model loaded, testing forward pass...")
    
    # Test one batch
    for i, batch in enumerate(dataloader):
        clear_memory()
        
        logger.info(f"   Testing sample {i+1}/10...")
        
        try:
            images = batch['image'].to(device)
            logger.info(f"   Input shape: {images.shape}")
            
            # Check memory before
            memory_before = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"   Memory before: {memory_before:.2f} GB")
            
            with torch.no_grad():
                with autocast():
                    outputs = model(images, return_graph=False)  # No graph to save memory
            
            # Check memory after
            memory_after = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"   Memory after: {memory_after:.2f} GB")
            logger.info(f"   Memory used: {memory_after - memory_before:.2f} GB")
            
            logger.info(f"   ✅ Sample {i+1} successful!")
            
            clear_memory()
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"   ❌ OOM on sample {i+1}: {e}")
            clear_memory()
            break
        
        if i >= 2:  # Test just 3 samples
            break
    
    logger.info("🎯 Minimal test completed!")

def train_minimal():
    """Ultra-minimal training loop."""
    logger.info("🚀 Starting ultra-minimal training...")
    
    device = get_device()
    
    # Tiny dataset
    dataset = DFTGrapheneDataset(
        data_path="data/train",
        split='train',
        max_samples=100
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_dft_batch,
        num_workers=0
    )
    
    # Model
    model = create_vantascope_model_with_uncertainty()
    model = model.to(device)
    
    # Simple loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    
    logger.info("🔥 Starting training loop...")
    
    for epoch in range(2):  # Just 2 epochs
        logger.info(f"Epoch {epoch+1}/2")
        
        for batch_idx, batch in enumerate(dataloader):
            clear_memory()
            
            try:
                images = batch['image'].to(device)
                
                optimizer.zero_grad()
                
                with autocast():
                    outputs = model(images, return_graph=False)
                    # Simple reconstruction loss only
                    loss = criterion(outputs['reconstruction'], images)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                if batch_idx % 10 == 0:
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    logger.info(f"   Batch {batch_idx}: Loss {loss.item():.4f}, Memory {memory_used:.2f} GB")
                
                clear_memory()
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"   OOM at batch {batch_idx}: {e}")
                clear_memory()
                break
            
            if batch_idx >= 20:  # Just 20 batches per epoch
                break
    
    logger.info("🎉 Minimal training completed!")

if __name__ == "__main__":
    # First test if anything works
    test_minimal_batch()
    
    # Then try minimal training
    train_minimal()
