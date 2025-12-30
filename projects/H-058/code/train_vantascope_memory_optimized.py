"""
VantaScope Memory-Optimized Training for RTX 5090
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import time
from pathlib import Path

from vantascope.models.integrated_model import create_vantascope_model_with_uncertainty
from vantascope.data.dft_graphene_loader import DFTGrapheneDataset, collate_dft_batch
from vantascope.training.enhanced_losses import EnhancedVantaScopeLoss
from vantascope.utils.logging import logger
from vantascope.utils.helpers import set_seed, ensure_dir, get_device

def find_optimal_batch_size(model, device):
    """Find the largest batch size that fits in VRAM."""
    logger.info("🔍 Finding optimal batch size...")
    
    model.eval()
    batch_sizes = [32, 24, 16, 12, 8, 4]
    
    for batch_size in batch_sizes:
        try:
            # Test with dummy data
            dummy_input = torch.randn(batch_size, 1, 512, 512).to(device)
            
            with torch.no_grad():
                with autocast():
                    outputs = model(dummy_input, return_graph=True)
            
            # Clear cache
            torch.cuda.empty_cache()
            
            logger.info(f"✅ Batch size {batch_size} works!")
            return batch_size
            
        except torch.cuda.OutOfMemoryError:
            logger.info(f"❌ Batch size {batch_size} too large")
            torch.cuda.empty_cache()
            continue
    
    logger.error("Could not find a working batch size!")
    return 2

class MemoryOptimizedTrainer:
    """Memory-optimized trainer for RTX 5090."""
    
    def __init__(self, config):
        self.config = config
        self.device = get_device()
        
        # Memory optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Enable memory-efficient attention
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Mixed precision
        self.scaler = GradScaler()
        
        logger.info("🚀 Memory-Optimized Trainer initialized")
    
    def create_datasets(self):
        """Create datasets with memory considerations."""
        logger.info("📊 Creating datasets...")
        
        # Use subset for initial training to test memory usage
        train_samples = self.config.get('train_samples', 50000)  # Start with 50k
        
        full_dataset = DFTGrapheneDataset(
            data_path="data/train",
            split='train',
            max_samples=train_samples,
            grid_size=512
        )
        
        test_dataset = DFTGrapheneDataset(
            data_path="data/test",
            split='test', 
            max_samples=5000,  # Smaller test set
            grid_size=512
        )
        
        # 90/10 split for memory efficiency
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        logger.info(f"   Train: {len(train_dataset):,} samples")
        logger.info(f"   Val: {len(val_dataset):,} samples")
        logger.info(f"   Test: {len(test_dataset):,} samples")
        
        return train_dataset, val_dataset, test_dataset
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Memory-efficient training epoch."""
        model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # Clear cache periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
            
            images = batch['image'].to(self.device, non_blocking=True)
            energies = batch['energy'].to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision forward
            with autocast():
                outputs = model(images, return_graph=True)
                loss_dict = criterion(outputs, batch, images)
                loss = loss_dict['total_loss']
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            # Progress logging
            if batch_idx % 50 == 0:
                elapsed = time.time() - start_time
                samples_processed = (batch_idx + 1) * self.config['batch_size']
                samples_per_sec = samples_processed / elapsed
                
                # Memory usage
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_cached = torch.cuda.memory_reserved() / 1024**3
                
                logger.info(f"   Epoch {epoch} [{batch_idx:4d}/{num_batches}] "
                          f"Loss: {loss.item():.4f} | "
                          f"Speed: {samples_per_sec:.0f} samples/sec | "
                          f"GPU: {memory_used:.1f}/{memory_cached:.1f} GB")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self):
        """Memory-optimized training pipeline."""
        logger.info("🚀 Starting Memory-Optimized Training!")
        
        # Create model first to test memory
        model = create_vantascope_model_with_uncertainty()
        model = model.to(self.device)
        
        # Find optimal batch size
        optimal_batch_size = find_optimal_batch_size(model, self.device)
        self.config['batch_size'] = optimal_batch_size
        
        logger.info(f"🎯 Using batch size: {optimal_batch_size}")
        
        # Create datasets and loaders
        train_dataset, val_dataset, test_dataset = self.create_datasets()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=optimal_batch_size,
            shuffle=True,
            collate_fn=collate_dft_batch,
            num_workers=8,  # Reduced workers
            pin_memory=True,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=optimal_batch_size,
            shuffle=False,
            collate_fn=collate_dft_batch,
            num_workers=4,
            pin_memory=True
        )
        
        # Loss and optimizer
        criterion = EnhancedVantaScopeLoss().to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.config['learning_rate'])
        
        # Training loop
        for epoch in range(1, self.config['epochs'] + 1):
            logger.info(f"🔥 Epoch {epoch}/{self.config['epochs']}")
            
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer, epoch)
            
            logger.info(f"✅ Epoch {epoch} completed | Loss: {train_loss:.4f}")
            
            # Save checkpoint
            if epoch % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': self.config
                }, f'models/vantascope_checkpoint_epoch_{epoch}.pth')
        
        return model

def main():
    config = {
        'learning_rate': 1e-4,
        'epochs': 10,
        'train_samples': 50000,  # Start with subset
        'save_dir': 'models/',
    }
    
    set_seed(42)
    ensure_dir(config['save_dir'])
    
    trainer = MemoryOptimizedTrainer(config)
    model = trainer.train()
    
    logger.info("🎉 Training complete!")

if __name__ == "__main__":
    main()
