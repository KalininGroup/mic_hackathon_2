"""
VantaScope Training with Properly Scaled Loss
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

class FixedLoss(nn.Module):
    """Loss function with proper energy scaling."""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
        # Energy normalization (DFT energies are ~-3000 eV)
        self.energy_mean = -3000.0  # Approximate mean
        self.energy_std = 500.0     # Approximate std
    
    def forward(self, outputs, batch):
        """Properly scaled reconstruction + energy loss."""
        images = batch['image'].to(outputs['reconstruction'].device)
        energies = batch['energy'].to(outputs['energy_mean'].device)
        
        # Reconstruction loss (already normalized 0-1)
        recon_loss = self.mse(outputs['reconstruction'], images)
        
        # Normalize energies to ~0 mean, ~1 std
        normalized_true = (energies - self.energy_mean) / self.energy_std
        normalized_pred = (outputs['energy_mean'] - self.energy_mean) / self.energy_std
        
        # Energy loss on normalized values
        energy_loss = self.mse(normalized_pred, normalized_true)
        
        # Balanced total loss
        total_loss = recon_loss + energy_loss  # Equal weighting
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'energy_loss': energy_loss,
            'energy_mae': torch.mean(torch.abs(outputs['energy_mean'] - energies))  # Real MAE in eV
        }

class FixedTrainer:
    """Trainer with fixed loss function."""
    
    def __init__(self, config):
        self.config = config
        self.device = get_device()
        
        torch.backends.cuda.matmul.allow_tf32 = True
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        self.scaler = GradScaler()
        self.accumulation_steps = config['accumulation_steps']
        
        logger.info("🔧 Fixed Trainer initialized")
        logger.info(f"   Batch size: {config['batch_size']}")
        logger.info(f"   Accumulation steps: {self.accumulation_steps}")
    
    def clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        model.train()
        total_loss = 0
        total_energy_mae = 0
        num_batches = len(train_loader)
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx % 10 == 0:
                self.clear_memory()
            
            try:
                images = batch['image'].to(self.device, non_blocking=True)
                
                with autocast():
                    outputs = model(images, return_graph=True)
                    loss_dict = criterion(outputs, batch)
                    loss = loss_dict['total_loss'] / self.accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * self.accumulation_steps
                total_energy_mae += loss_dict['energy_mae'].item()
                
                if batch_idx % 50 == 0:
                    elapsed = time.time() - start_time
                    samples_processed = (batch_idx + 1) * self.config['batch_size']
                    samples_per_sec = samples_processed / elapsed
                    
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    
                    logger.info(f"   Epoch {epoch} [{batch_idx:4d}/{num_batches}] "
                              f"Loss: {loss.item() * self.accumulation_steps:.4f} "
                              f"(Recon: {loss_dict['recon_loss']:.4f}, Energy: {loss_dict['energy_loss']:.4f}) | "
                              f"Energy MAE: {loss_dict['energy_mae'].item():.1f} eV | "
                              f"Speed: {samples_per_sec:.0f} samples/sec | GPU: {memory_used:.1f}GB")
                
            except Exception as e:
                logger.error(f"   Error at batch {batch_idx}: {e}")
                self.clear_memory()
                continue
        
        if num_batches % self.accumulation_steps != 0:
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_energy_mae = total_energy_mae / max(num_batches, 1)
        epoch_time = time.time() - start_time
        
        logger.info(f"✅ Epoch {epoch} completed in {epoch_time:.1f}s | "
                   f"Avg Loss: {avg_loss:.4f} | Avg Energy MAE: {avg_energy_mae:.1f} eV")
        
        return avg_loss
    
    def train(self):
        logger.info("🔧 Starting Fixed VantaScope Training!")
        
        train_samples = self.config.get('train_samples', 5000)
        
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
        
        model = create_vantascope_model_with_uncertainty()
        model = model.to(self.device)
        
        criterion = FixedLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.config['learning_rate'])
        
        for epoch in range(1, self.config['epochs'] + 1):
            logger.info(f"🔥 Epoch {epoch}/{self.config['epochs']}")
            
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer, epoch)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'config': self.config
            }, f'models/vantascope_fixed_epoch_{epoch}.pth')
            
            logger.info(f"💾 Checkpoint saved for epoch {epoch}")
        
        logger.info("🎉 Fixed training completed!")
        return model

def main():
    config = {
        'batch_size': 2,
        'accumulation_steps': 8,
        'learning_rate': 1e-4,
        'epochs': 3,
        'train_samples': 5000,
        'save_dir': 'models/',
    }
    
    set_seed(42)
    ensure_dir(config['save_dir'])
    
    trainer = FixedTrainer(config)
    model = trainer.train()
    
    logger.info("🔧 Fixed training complete!")

if __name__ == "__main__":
    main()
