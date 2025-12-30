"""
VantaScope Full-Scale Training - Optimized for RTX 5090
Updated with gradient accumulation and your config
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
import json
import argparse

from vantascope.models.integrated_model import create_vantascope_model_with_uncertainty
from vantascope.data.dft_graphene_loader import DFTGrapheneDataset, collate_dft_batch
from vantascope.training.enhanced_losses import EnhancedVantaScopeLoss
from vantascope.utils.logging import logger
from vantascope.utils.helpers import set_seed, ensure_dir, get_device

class RTX5090Trainer:
    """Optimized trainer for RTX 5090 with gradient accumulation."""
    
    def __init__(self, config):
        self.config = config
        self.device = get_device()
        
        # RTX 5090 optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Gradient accumulation
        self.accumulation_steps = config['accumulation_steps']
        self.effective_batch_size = config['batch_size'] * self.accumulation_steps
        
        logger.info("🚀 RTX 5090 Trainer initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Batch size: {config['batch_size']} (effective: {self.effective_batch_size})")
        logger.info(f"   Mixed precision: Enabled")
        logger.info(f"   Gradient accumulation steps: {self.accumulation_steps}")
    
    def create_datasets(self):
        """Create datasets with specified sample limit."""
        logger.info("📊 Creating datasets...")
        
        # Training dataset with sample limit
        full_dataset = DFTGrapheneDataset(
            data_path="data/train",
            split='train',
            max_samples=self.config['train_samples'],  # Your 100k limit
            grid_size=512
        )
        
        # Test dataset
        test_dataset = DFTGrapheneDataset(
            data_path="data/test", 
            split='test',
            max_samples=20000,  # Large test set
            grid_size=512
        )
        
        # Split training into train/val
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
        
        batch_size = self.config['batch_size']
        num_workers = self.config['num_workers']
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_dft_batch,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            drop_last=True  # Consistent batch sizes for accumulation
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_dft_batch,
            num_workers=num_workers//2,
            pin_memory=True,
            persistent_workers=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_dft_batch,
            num_workers=num_workers//2,
            pin_memory=True
        )
        
        logger.info(f"🔥 DataLoaders created:")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Workers: {num_workers}")
        logger.info(f"   Train batches: {len(train_loader):,}")
        logger.info(f"   Effective batch size: {self.effective_batch_size}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Train one epoch with gradient accumulation."""
        model.train()
        total_loss = 0
        total_energy_mae = 0
        num_batches = len(train_loader)
        
        start_time = time.time()
        samples_processed = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to GPU
            images = batch['image'].to(self.device, non_blocking=True)
            
            # Mixed precision forward pass
            with autocast():
                outputs = model(images, return_graph=True)
                loss_dict = criterion(outputs, batch, images)
                # Scale loss by accumulation steps
                loss = loss_dict['total_loss'] / self.accumulation_steps
            
            # Scaled backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping for stability
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
            
            # Metrics tracking
            total_loss += loss.item() * self.accumulation_steps
            total_energy_mae += loss_dict.get('energy_mae', 0)
            samples_processed += self.config['batch_size']
            
            # Progress logging
            if batch_idx % 50 == 0:
                elapsed = time.time() - start_time
                samples_per_sec = samples_processed / elapsed
                
                # Memory stats
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                
                logger.info(f"   Epoch {epoch} [{batch_idx:4d}/{num_batches}] "
                          f"Loss: {loss.item() * self.accumulation_steps:.4f} "
                          f"(R: {loss_dict.get('recon_loss', 0):.4f}, E: {loss_dict.get('energy_loss', 0):.4f}) | "
                          f"Energy MAE: {loss_dict.get('energy_mae', 0):.1f} eV | "
                          f"Speed: {samples_per_sec:.0f} samples/sec | "
                          f"GPU: {memory_allocated:.1f}GB")
        
        # Final gradient update if needed
        if num_batches % self.accumulation_steps != 0:
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_energy_mae = total_energy_mae / max(num_batches, 1)
        epoch_time = time.time() - start_time
        
        logger.info(f"✅ Epoch {epoch} completed in {epoch_time:.1f}s | Avg Loss: {avg_loss:.4f} | Avg Energy MAE: {avg_energy_mae:.1f} eV")
        
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
        """Full training pipeline."""
        logger.info("🚀 Starting VantaScope Full-Scale Training!")
        
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
            
            # Save checkpoint every 3 epochs
            if epoch % 3 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }, f'models/vantascope_full_scale_epoch_{epoch}.pth')
                
                logger.info(f"💾 Checkpoint saved for epoch {epoch}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }, 'models/vantascope_full_scale_best.pth')
                
                logger.info(f"🌟 New best model saved! Val Loss: {val_loss:.4f}")
        
        logger.info("🎉 Training completed!")
        return model

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='VantaScope Full-Scale Training')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--train_samples', type=int, default=100000, help='Training samples')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    # Your optimized config
    config = {
        'batch_size': args.batch_size,
        'accumulation_steps': 4,      # Effective batch size 24
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-5,
        'epochs': args.epochs,
        'train_samples': args.train_samples,
        'num_workers': 16,
        'save_dir': 'models/',
    }
    
    logger.info("🔥 BEAST MODE TRAINING CONFIG:")
    logger.info(f"   Batch size: {config['batch_size']} (effective: {config['batch_size'] * config['accumulation_steps']})")
    logger.info(f"   Learning rate: {config['learning_rate']}")
    logger.info(f"   Epochs: {config['epochs']}")
    logger.info(f"   Training samples: {config['train_samples']:,}")
    logger.info(f"   Workers: {config['num_workers']}")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Ensure save directory exists
    ensure_dir(config['save_dir'])
    
    try:
        # Create trainer and start training
        trainer = RTX5090Trainer(config)
        model = trainer.train()
        
        # Save final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config
        }, 'models/vantascope_full_scale_final.pth')
        
        logger.info("🚀 VantaScope training complete! Ready for inference.")
        
    except Exception as e:
        logger.error(f"💥 Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
