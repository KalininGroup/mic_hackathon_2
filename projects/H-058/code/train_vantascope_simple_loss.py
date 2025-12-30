"""
VantaScope Training with Simple Loss Function
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
from vantascope.utils.logging import logger
from vantascope.utils.helpers import set_seed, ensure_dir, get_device

class SimpleLoss(nn.Module):
    """Simple loss function that works."""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, outputs, batch):
        """Simple reconstruction + energy loss."""
        images = batch['image'].to(outputs['reconstruction'].device)
        energies = batch['energy'].to(outputs['energy_mean'].device)
        
        # Reconstruction loss
        recon_loss = self.mse(outputs['reconstruction'], images)
        
        # Energy loss
        energy_loss = self.mse(outputs['energy_mean'], energies)
        
        # Total loss
        total_loss = recon_loss + 0.1 * energy_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'energy_loss': energy_loss
        }

class SimpleTrainer:
    """Simple trainer with working loss function."""
    
    def __init__(self, config):
        self.config = config
        self.device = get_device()
        
        # Memory optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Gradient accumulation
        self.accumulation_steps = config['accumulation_steps']
        
        logger.info("🚀 Simple Trainer initialized")
        logger.info(f"   Batch size: {config['batch_size']}")
        logger.info(f"   Accumulation steps: {self.accumulation_steps}")
    
    def clear_memory(self):
        """Clear GPU memory."""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Simple training epoch."""
        model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # Clear memory every 10 batches
            if batch_idx % 10 == 0:
                self.clear_memory()
            
            try:
                images = batch['image'].to(self.device, non_blocking=True)
                
                # Mixed precision forward pass
                with autocast():
                    outputs = model(images, return_graph=True)
                    loss_dict = criterion(outputs, batch)
                    loss = loss_dict['total_loss'] / self.accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Update weights every accumulation_steps
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * self.accumulation_steps
                
                # Progress logging
                if batch_idx % 50 == 0:
                    elapsed = time.time() - start_time
                    samples_processed = (batch_idx + 1) * self.config['batch_size']
                    samples_per_sec = samples_processed / elapsed
                    
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    
                    logger.info(f"   Epoch {epoch} [{batch_idx:4d}/{num_batches}] "
                              f"Loss: {loss.item() * self.accumulation_steps:.4f} "
                              f"(Recon: {loss_dict['recon_loss']:.4f}, Energy: {loss_dict['energy_loss']:.4f}) | "
                              f"Speed: {samples_per_sec:.0f} samples/sec | "
                              f"GPU: {memory_used:.1f}GB")
                
            except Exception as e:
                logger.error(f"   Error at batch {batch_idx}: {e}")
                self.clear_memory()
                continue
        
        # Final gradient update
        if num_batches % self.accumulation_steps != 0:
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
        
        avg_loss = total_loss / max(num_batches, 1)
        epoch_time = time.time() - start_time
        
        logger.info(f"✅ Epoch {epoch} completed in {epoch_time:.1f}s | Avg Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def train(self):
        """Simple training pipeline."""
        logger.info("🚀 Starting Simple VantaScope Training!")
        
        # Dataset
        train_samples = self.config.get('train_samples', 5000)  # Start small
        
        full_dataset = DFTGrapheneDataset(
            data_path="data/train",
            split='train',
            max_samples=train_samples,
            grid_size=512
        )
        
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Simple dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=collate_dft_batch,
            num_workers=0,
            pin_memory=False
        )
        
        logger.info(f"📊 Dataset: {len(train_dataset)} train samples")
        logger.info(f"📊 Batches per epoch: {len(train_loader)}")
        
        # Create model
        model = create_vantascope_model_with_uncertainty()
        model = model.to(self.device)
        
        # Simple loss and optimizer
        criterion = SimpleLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.config['learning_rate'])
        
        # Training loop
        for epoch in range(1, self.config['epochs'] + 1):
            logger.info(f"🔥 Epoch {epoch}/{self.config['epochs']}")
            
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer, epoch)
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'config': self.config
            }, f'models/vantascope_simple_epoch_{epoch}.pth')
            
            logger.info(f"💾 Checkpoint saved for epoch {epoch}")
        
        logger.info("🎉 Simple training completed!")
        return model

def main():
    """Main training function."""
    
    config = {
        'batch_size': 2,
        'accumulation_steps': 8,   # Effective batch size 16
        'learning_rate': 1e-4,
        'epochs': 3,               # Just 3 epochs for testing
        'train_samples': 5000,     # Small dataset
        'save_dir': 'models/',
    }
    
    set_seed(42)
    ensure_dir(config['save_dir'])
    
    trainer = SimpleTrainer(config)
    model = trainer.train()
    
    logger.info("🎉 Simple training complete!")

if __name__ == "__main__":
    main()
