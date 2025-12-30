"""
Ultra-Safe VantaScope Training - Batch size 2 with aggressive memory management
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
import gc

from vantascope.models.integrated_model import create_vantascope_model_with_uncertainty
from vantascope.data.dft_graphene_loader import DFTGrapheneDataset, collate_dft_batch
from vantascope.training.enhanced_losses import EnhancedVantaScopeLoss
from vantascope.utils.logging import logger
from vantascope.utils.helpers import set_seed, ensure_dir, get_device

class UltraSafeTrainer:
    """Ultra-safe trainer with aggressive memory management."""
    
    def __init__(self, config):
        self.config = config
        self.device = get_device()
        
        # Memory optimizations
        torch.backends.cudnn.benchmark = False  # Disable for memory consistency
        torch.backends.cuda.matmul.allow_tf32 = True
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Large gradient accumulation for effective batch size
        self.accumulation_steps = config['accumulation_steps']
        self.effective_batch_size = config['batch_size'] * self.accumulation_steps
        
        logger.info("🛡️ Ultra-Safe Trainer initialized")
        logger.info(f"   Batch size: {config['batch_size']}")
        logger.info(f"   Accumulation steps: {self.accumulation_steps}")
        logger.info(f"   Effective batch size: {self.effective_batch_size}")
    
    def aggressive_memory_clear(self):
        """Aggressively clear all possible memory."""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Ultra-safe training epoch."""
        model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # Clear memory every single batch
            self.aggressive_memory_clear()
            
            try:
                images = batch['image'].to(self.device, non_blocking=True)
                
                # Check memory before forward pass
                memory_before = torch.cuda.memory_allocated() / 1024**3
                
                # Mixed precision forward pass
                with autocast():
                    outputs = model(images, return_graph=True)
                    loss_dict = criterion(outputs, batch, images)
                    loss = loss_dict['total_loss'] / self.accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Update weights every accumulation_steps
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
                    self.aggressive_memory_clear()
                
                total_loss += loss.item() * self.accumulation_steps
                
                # Progress logging every 20 batches
                if batch_idx % 20 == 0:
                    elapsed = time.time() - start_time
                    samples_processed = (batch_idx + 1) * self.config['batch_size']
                    samples_per_sec = samples_processed / elapsed
                    
                    memory_after = torch.cuda.memory_allocated() / 1024**3
                    memory_peak = torch.cuda.max_memory_allocated() / 1024**3
                    
                    logger.info(f"   Epoch {epoch} [{batch_idx:4d}/{num_batches}] "
                              f"Loss: {loss.item() * self.accumulation_steps:.4f} | "
                              f"Speed: {samples_per_sec:.0f} samples/sec | "
                              f"GPU: {memory_after:.1f}GB (peak: {memory_peak:.1f}GB)")
                    
                    # Reset peak memory stats
                    torch.cuda.reset_peak_memory_stats()
                
                # Clear memory after every batch
                self.aggressive_memory_clear()
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"   OOM at batch {batch_idx}: {e}")
                self.aggressive_memory_clear()
                # Skip this batch and continue
                continue
            except Exception as e:
                logger.error(f"   Error at batch {batch_idx}: {e}")
                self.aggressive_memory_clear()
                continue
        
        # Final gradient update if needed
        if num_batches % self.accumulation_steps != 0:
            try:
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
            except:
                pass
        
        avg_loss = total_loss / max(num_batches, 1)
        epoch_time = time.time() - start_time
        
        logger.info(f"✅ Epoch {epoch} completed in {epoch_time:.1f}s | Avg Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def train(self):
        """Ultra-safe training pipeline."""
        logger.info("🛡️ Starting Ultra-Safe VantaScope Training!")
        
        # Smaller dataset for safety
        train_samples = self.config.get('train_samples', 10000)  # Start with 10k
        
        full_dataset = DFTGrapheneDataset(
            data_path="data/train",
            split='train',
            max_samples=train_samples,
            grid_size=512
        )
        
        # 90/10 split
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Ultra-safe dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=collate_dft_batch,
            num_workers=0,  # No multiprocessing to save memory
            pin_memory=False  # Disable pin_memory
        )
        
        logger.info(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val")
        logger.info(f"📊 Batches per epoch: {len(train_loader)}")
        
        # Create model
        model = create_vantascope_model_with_uncertainty()
        model = model.to(self.device)
        
        # Simple loss and optimizer
        criterion = EnhancedVantaScopeLoss().to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.config['learning_rate'])
        
        # Training loop
        for epoch in range(1, self.config['epochs'] + 1):
            logger.info(f"🔥 Epoch {epoch}/{self.config['epochs']}")
            
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer, epoch)
            
            # Save checkpoint every epoch
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': self.config
                }, f'models/vantascope_safe_epoch_{epoch}.pth')
                logger.info(f"💾 Checkpoint saved for epoch {epoch}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
        
        logger.info("🎉 Ultra-safe training completed!")
        return model

def main():
    """Main training function."""
    
    # Ultra-conservative config
    config = {
        'batch_size': 2,           # Very safe batch size
        'accumulation_steps': 12,  # Effective batch size 24
        'learning_rate': 1e-4,
        'epochs': 5,               # Fewer epochs for testing
        'train_samples': 10000,    # Smaller dataset
        'save_dir': 'models/',
    }
    
    set_seed(42)
    ensure_dir(config['save_dir'])
    
    trainer = UltraSafeTrainer(config)
    model = trainer.train()
    
    logger.info("🛡️ Ultra-safe training complete!")

if __name__ == "__main__":
    main()
