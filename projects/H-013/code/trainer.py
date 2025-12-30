"""
Training manager for SE(3)-Equivariant AFM Reconstruction
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional
import time

from models.joint_model import JointTipSurfaceModel
from training.losses import AFMReconstructionLoss, SSIMLoss


class AFMTrainer:
    """Training manager for joint reconstruction"""
    
    def __init__(self, model: JointTipSurfaceModel, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Loss functions
        self.loss_fn = AFMReconstructionLoss(config)
        self.ssim_loss = SSIMLoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if config.training.mixed_precision else None
        
        # Tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Tensorboard
        log_dir = Path(config.log_dir) / config.experiment_name
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir) / config.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Trainer initialized")
        print(f"  Experiment: {config.experiment_name}")
        print(f"  Checkpoints: {self.checkpoint_dir}")
        print(f"  Logs: {log_dir}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.training.scheduler_type == 'cosine_warmup':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.training.t_0,
                T_mult=self.config.training.t_mult
            )
        elif self.config.training.scheduler_type == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=50,
                gamma=0.5
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = {
            'total': [], 'tip_mse': [], 'surface_mse': [],
            'consistency': [], 'smoothness': []
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            afm_image = batch['image'].to(self.config.training.device)
            tip_gt = batch['tip'].to(self.config.training.device)
            surface_gt = batch['surface'].to(self.config.training.device)
            
            # Forward pass
            if self.scaler is not None:
                with autocast():
                    predictions = self.model(afm_image)
                    
                    ground_truth = {
                        'tip': tip_gt,
                        'surface': surface_gt,
                        'image': afm_image
                    }
                    
                    loss, loss_dict = self.loss_fn(predictions, ground_truth)
            else:
                predictions = self.model(afm_image)
                
                ground_truth = {
                    'tip': tip_gt,
                    'surface': surface_gt,
                    'image': afm_image
                }
                
                loss, loss_dict = self.loss_fn(predictions, ground_truth)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.max_grad_norm
                )
                self.optimizer.step()
            
            # Track losses
            epoch_losses['total'].append(loss_dict['total'])
            epoch_losses['tip_mse'].append(loss_dict['tip_mse'].item() if isinstance(loss_dict['tip_mse'], torch.Tensor) else loss_dict['tip_mse'])
            epoch_losses['surface_mse'].append(loss_dict['surface_mse'].item() if isinstance(loss_dict['surface_mse'], torch.Tensor) else loss_dict['surface_mse'])
            epoch_losses['consistency'].append(loss_dict['consistency'].item() if isinstance(loss_dict['consistency'], torch.Tensor) else loss_dict['consistency'])
            
            smoothness = loss_dict['tip_smoothness'] + loss_dict['surface_smoothness']
            epoch_losses['smoothness'].append(smoothness.item() if isinstance(smoothness, torch.Tensor) else smoothness)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'tip': f"{loss_dict['tip_mse']:.4f}" if isinstance(loss_dict['tip_mse'], (int, float)) else f"{loss_dict['tip_mse'].item():.4f}",
                'surf': f"{loss_dict['surface_mse']:.4f}" if isinstance(loss_dict['surface_mse'], (int, float)) else f"{loss_dict['surface_mse'].item():.4f}"
            })
            
            # Log to tensorboard
            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            if batch_idx % 50 == 0:
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        self.writer.add_scalar(f'Train/{key}', value.item(), global_step)
                    else:
                        self.writer.add_scalar(f'Train/{key}', value, global_step)
        
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        return avg_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()
        
        val_losses = {
            'total': [], 'tip_mse': [], 'surface_mse': [],
            'tip_mae': [], 'surface_mae': [], 'consistency': [],
            'ssim': [], 'psnr': []
        }
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            afm_image = batch['image'].to(self.config.training.device)
            tip_gt = batch['tip'].to(self.config.training.device)
            surface_gt = batch['surface'].to(self.config.training.device)
            
            # Forward pass
            predictions = self.model(afm_image, monte_carlo_samples=1)
            
            ground_truth = {
                'tip': tip_gt,
                'surface': surface_gt,
                'image': afm_image
            }
            
            loss, loss_dict = self.loss_fn(predictions, ground_truth)
            
            # Compute additional metrics
            ssim = 1 - self.ssim_loss(predictions['surface'], surface_gt)
            
            mse = torch.mean((predictions['surface'] - surface_gt) ** 2)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-10))
            
            # Track
            val_losses['total'].append(loss_dict['total'])
            val_losses['tip_mse'].append(loss_dict['tip_mse'].item() if isinstance(loss_dict['tip_mse'], torch.Tensor) else loss_dict['tip_mse'])
            val_losses['surface_mse'].append(loss_dict['surface_mse'].item() if isinstance(loss_dict['surface_mse'], torch.Tensor) else loss_dict['surface_mse'])
            val_losses['tip_mae'].append(loss_dict['tip_mae'].item() if isinstance(loss_dict['tip_mae'], torch.Tensor) else loss_dict['tip_mae'])
            val_losses['surface_mae'].append(loss_dict['surface_mae'].item() if isinstance(loss_dict['surface_mae'], torch.Tensor) else loss_dict['surface_mae'])
            val_losses['consistency'].append(loss_dict['consistency'].item() if isinstance(loss_dict['consistency'], torch.Tensor) else loss_dict['consistency'])
            val_losses['ssim'].append(ssim.item())
            val_losses['psnr'].append(psnr.item())
        
        # Average
        avg_losses = {k: np.mean(v) for k, v in val_losses.items()}
        
        # Log to tensorboard
        for key, value in avg_losses.items():
            self.writer.add_scalar(f'Val/{key}', value, self.current_epoch)
        
        return avg_losses
    
    def train(self, num_epochs: Optional[int] = None):
        """Full training loop"""
        if num_epochs is None:
            num_epochs = self.config.training.epochs
        
        print(f"\n{'='*70}")
        print(f"Starting Training: {num_epochs} epochs")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses)
            
            # Validate
            if epoch % self.config.training.validate_every == 0:
                val_losses = self.validate()
                self.val_losses.append(val_losses)
                
                print(f"\nEpoch {epoch}:")
                print(f"  Train Loss: {train_losses['total']:.4f}")
                print(f"  Val Loss: {val_losses['total']:.4f}")
                print(f"  Val PSNR: {val_losses['psnr']:.2f} dB")
                print(f"  Val SSIM: {val_losses['ssim']:.4f}")
                
                # Save best model
                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    self.save_checkpoint('best_model.pt', val_losses)
                    print(f"  ✓ Saved best model (loss: {self.best_val_loss:.4f})")
            
            # Save checkpoint
            if epoch % self.config.training.save_every == 0 and epoch > 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
            
            # Update scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_losses['total'])
            else:
                self.scheduler.step()
            
            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/learning_rate', current_lr, epoch)
        
        # Training complete
        elapsed_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"  Total time: {elapsed_time/3600:.2f} hours")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"{'='*70}\n")
        
        self.writer.close()
    
    def save_checkpoint(self, filename: str, metrics: Optional[Dict] = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.config.training.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")


def test_trainer():
    """Test trainer with small dataset"""
    from config import get_config, set_seed
    from data.dataset import create_dataloaders
    
    # Set seed
    set_seed(42)
    
    # Config
    config = get_config()
    config.data.train_size = 100
    config.data.val_size = 20
    config.data.test_size = 20
    config.training.epochs = 2
    config.training.batch_size = 4
    
    print("Creating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(config, use_cached=False)
    
    print("Creating model...")
    model = JointTipSurfaceModel(config).to(config.training.device)
    
    print("Creating trainer...")
    trainer = AFMTrainer(model, train_loader, val_loader, config)
    
    print("\nStarting test training...")
    trainer.train(num_epochs=2)
    
    print("\n✓ Trainer test completed successfully")

if __name__ == '__main__':
    test_trainer()
