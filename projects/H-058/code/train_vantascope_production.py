"""
VantaScope Production Training - Optimized for RTX 5090
Batch size 6 with gradient accumulation for effective batch size 24
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
from pathlib import Path

from vantascope.models.integrated_model import create_vantascope_model_with_uncertainty
from vantascope.data.dft_graphene_loader import DFTGrapheneDataset, collate_dft_batch
from vantascope.training.enhanced_losses import EnhancedVantaScopeLoss
from vantascope.utils.logging import logger
from vantascope.utils.helpers import set_seed, ensure_dir, get_device

class ProductionTrainer:
    """Production trainer optimized for RTX 5090 with batch size 6."""
    
    def __init__(self, config):
        self.config = config
        self.device = get_device()
        
        # RTX 5090 optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Gradient accumulation for effective larger batch size
        self.accumulation_steps = config['accumulation_steps']
        self.effective_batch_size = config['batch_size'] * self.accumulation_steps
        
        logger.info("🚀 Production Trainer initialized")
        logger.info(f"   Batch size: {config['batch_size']}")
        logger.info(f"   Accumulation steps: {self.accumulation_steps}")
        logger.info(f"   Effective batch size: {self.effective_batch_size}")
    
    def clear_memory(self):
        """Clear GPU memory."""
        gc.collect()
        torch.cuda.empty_cache()
    
    def create_datasets(self):
        """Create production datasets."""
        logger.info("📊 Creating production datasets...")
        
        # Use substantial portion of dataset
        train_samples = self.config.get('train_samples', 100000)  # 100k samples
        
        full_dataset = DFTGrapheneDataset(
            data_path="data/train",
            split='train',
            max_samples=train_samples,
            grid_size=512
        )
        
        test_dataset = DFTGrapheneDataset(
            data_path="data/test",
            split='test',
            max_samples=10000,  # 10k test samples
            grid_size=512
        )
        
        # 95/5 train/val split
        train_size = int(0.95 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        logger.info(f"   Train: {len(train_dataset):,} samples")
        logger.info(f"   Val: {len(val_dataset):,} samples")
        logger.info(f"   Test: {len(test_dataset):,} samples")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(self, train_dataset, val_dataset, test_dataset):
        """Create optimized dataloaders."""
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=collate_dft_batch,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            collate_fn=collate_dft_batch,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            collate_fn=collate_dft_batch,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"🔥 DataLoaders created:")
        logger.info(f"   Train batches: {len(train_loader):,}")
        logger.info(f"   Val batches: {len(val_loader):,}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Train one epoch with gradient accumulation."""
        model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # Clear memory periodically
            if batch_idx % 100 == 0:
                self.clear_memory()
            
            images = batch['image'].to(self.device, non_blocking=True)
            
            # Mixed precision forward pass
            with autocast():
                outputs = model(images, return_graph=True)
                loss_dict = criterion(outputs, batch, images)
                loss = loss_dict['total_loss'] / self.accumulation_steps  # Scale loss
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps  # Unscale for logging
            
            # Progress logging
            if batch_idx % 100 == 0:
                elapsed = time.time() - start_time
                samples_processed = (batch_idx + 1) * self.config['batch_size']
                samples_per_sec = samples_processed / elapsed
                
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_cached = torch.cuda.memory_reserved() / 1024**3
                
                logger.info(f"   Epoch {epoch} [{batch_idx:4d}/{num_batches}] "
                          f"Loss: {loss.item() * self.accumulation_steps:.4f} | "
                          f"Speed: {samples_per_sec:.0f} samples/sec | "
                          f"GPU: {memory_used:.1f}/{memory_cached:.1f} GB")
        
        # Final gradient update if needed
        if num_batches % self.accumulation_steps != 0:
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
        
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - start_time
        
        logger.info(f"✅ Epoch {epoch} completed in {epoch_time:.1f}s | Avg Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def validate(self, model, val_loader, criterion):
        """Validate the model."""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device, non_blocking=True)
                
                with autocast():
                    outputs = model(images, return_graph=True)
                    loss_dict = criterion(outputs, batch, images)
                    loss = loss_dict['total_loss']
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(self):
        """Full production training pipeline."""
        logger.info("🚀 Starting VantaScope Production Training!")
        
        # Create datasets
        train_dataset, val_dataset, test_dataset = self.create_datasets()
        train_loader, val_loader, test_loader = self.create_dataloaders(
            train_dataset, val_dataset, test_dataset
        )
        
        # Create model
        model = create_vantascope_model_with_uncertainty()
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"🧠 Model loaded:")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        
        # Loss function and optimizer
        criterion = EnhancedVantaScopeLoss(
            lambda_recon=1.0,
            lambda_energy=10.0,
            lambda_fuzzy=5.0,
            lambda_disentangle=3.0
        ).to(self.device)
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config['epochs']
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config['epochs'] + 1):
            logger.info(f"🔥 Epoch {epoch}/{self.config['epochs']}")
            
            # Train
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer, epoch)
            
            # Validate
            val_loss = self.validate(model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step()
            
            logger.info(f"📊 Epoch {epoch} Results:")
            logger.info(f"   Train Loss: {train_loss:.4f}")
            logger.info(f"   Val Loss: {val_loss:.4f}")
            logger.info(f"   LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }, 'models/vantascope_production_best.pth')
                
                logger.info(f"💾 New best model saved! Val Loss: {val_loss:.4f}")
            
            # Save checkpoint every 5 epochs
            if epoch % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': self.config
                }, f'models/vantascope_production_epoch_{epoch}.pth')
        
        logger.info("🎉 Production training completed!")
        return model

def main():
    """Main training function."""
    
    # Production config optimized for RTX 5090
    config = {
        'batch_size': 6,           # Safe batch size
        'accumulation_steps': 4,   # Effective batch size 24
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'epochs': 15,
        'train_samples': 100000,   # 100k samples
        'save_dir': 'models/',
    }
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Ensure save directory exists
    ensure_dir(config['save_dir'])
    
    # Create trainer and start training
    trainer = ProductionTrainer(config)
    model = trainer.train()
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, 'models/vantascope_production_final.pth')
    
    logger.info("🚀 VantaScope production training complete!")

if __name__ == "__main__":
    main()
