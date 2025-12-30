"""
physics_diffusion/training/diffusion_trainer.py

Complete training framework with physics-constrained losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional, Tuple
import wandb
from tqdm import tqdm
import numpy as np


class PhysicsLoss(nn.Module):
    """Physics consistency losses."""
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        x_pred: torch.Tensor,
        x_noisy: torch.Tensor,
        physics_operator,
        modality_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute physics consistency loss.
        
        Args:
            x_pred: Predicted clean image [B, C, H, W]
            x_noisy: Original noisy observation [B, C, H, W]
            physics_operator: Physics operator module
            modality_ids: Modality indices [B]
            
        Returns:
            Physics loss scalar
        """
        B = x_pred.shape[0]
        total_loss = 0.0
        
        # Apply forward physics model and compare
        for b in range(B):
            # Forward model: y_pred = P(x_pred)
            x_pred_single = x_pred[b:b+1]
            y_pred = physics_operator(x_pred_single)
            
            # Compare with actual noisy observation
            loss = F.l1_loss(y_pred, x_noisy[b:b+1])
            total_loss += loss
        
        return total_loss / B


class CycleLoss(nn.Module):
    """Cycle consistency loss for unpaired data."""
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        x_noisy: torch.Tensor,
        x_denoised: torch.Tensor,
        physics_operator
    ) -> torch.Tensor:
        """Compute cycle consistency: noisy → clean → noisy.
        
        Args:
            x_noisy: Original noisy image
            x_denoised: Denoised prediction
            physics_operator: Physics operator
            
        Returns:
            Cycle loss
        """
        # Apply physics to denoised to get back noisy
        x_renoised = physics_operator(x_denoised)
        
        # Should match original noisy image
        return F.l1_loss(x_renoised, x_noisy)


class DiffusionTrainer:
    """Complete training framework for physics-constrained diffusion."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: Dict,
        device: str = 'cuda',
        use_wandb: bool = True
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        
        # Loss components
        self.physics_loss_fn = PhysicsLoss()
        self.cycle_loss_fn = CycleLoss()
        
        # Loss weights (from config)
        self.lambda_diffusion = config.get('lambda_diffusion', 1.0)
        self.lambda_physics = config.get('lambda_physics', 0.5)
        self.lambda_cycle = config.get('lambda_cycle', 0.5)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.999),
            weight_decay=config.get('weight_decay', 0.05)
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['total_steps'],
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # Gradient clipping
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        
        if use_wandb:
            wandb.init(project='physics-diffusion', config=config)
    
    def compute_total_loss(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total weighted loss.
        
        Args:
            batch: Dictionary with 'noisy_images', 'modality_ids', 'physics_params'
            
        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Individual loss components
        """
        x_noisy = batch['noisy_images'].to(self.device)
        modality_ids = batch['modality_ids'].to(self.device)
        physics_params = batch.get('physics_params', None)
        
        # Forward pass through diffusion model
        results = self.model(
            x_noisy,
            modality_ids,
            physics_params,
            return_dict=True
        )
        
        # 1. Diffusion loss (noise prediction)
        loss_diffusion = results['loss_diffusion']
        
        # 2. Physics consistency loss
        loss_physics = results['loss_physics']
        
        # 3. Cycle consistency loss (optional)
        if self.lambda_cycle > 0:
            x_0_pred = results['x_0_pred']
            loss_cycle = 0.0
            
            # Compute per modality
            for b in range(x_noisy.shape[0]):
                mod_id = modality_ids[b].item()
                mod_name = self.model.modality_names[mod_id]
                
                if self.model.physics_operators is not None:
                    physics_op = self.model.physics_operators[mod_name]
                    loss_cycle += self.cycle_loss_fn(
                        x_noisy[b:b+1],
                        x_0_pred[b:b+1],
                        physics_op
                    )
            
            loss_cycle = loss_cycle / x_noisy.shape[0]
        else:
            loss_cycle = torch.tensor(0.0, device=self.device)
        
        # Total weighted loss
        total_loss = (
            self.lambda_diffusion * loss_diffusion +
            self.lambda_physics * loss_physics +
            self.lambda_cycle * loss_cycle
        )
        
        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_diffusion': loss_diffusion.item(),
            'loss_physics': loss_physics.item(),
            'loss_cycle': loss_cycle.item()
        }
        
        return total_loss, loss_dict
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary of losses
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Compute loss
        total_loss, loss_dict = self.compute_total_loss(batch)
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        # Update metrics
        loss_dict['learning_rate'] = self.scheduler.get_last_lr()[0]
        
        return loss_dict
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        val_losses = []
        
        for batch in tqdm(self.val_dataloader, desc='Validation'):
            _, loss_dict = self.compute_total_loss(batch)
            val_losses.append(loss_dict)
        
        # Average losses
        avg_losses = {
            key: np.mean([d[key] for d in val_losses])
            for key in val_losses[0].keys()
        }
        
        return avg_losses
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            path: Save path
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
    
    def train(self, total_steps: int, val_interval: int = 1000, 
             save_interval: int = 5000):
        """Main training loop.
        
        Args:
            total_steps: Total training steps
            val_interval: Steps between validation
            save_interval: Steps between checkpoints
        """
        print(f"Starting training for {total_steps} steps...")
        print(f"Device: {self.device}")
        print(f"Loss weights: λ_diff={self.lambda_diffusion}, "
              f"λ_phys={self.lambda_physics}, λ_cycle={self.lambda_cycle}")
        
        pbar = tqdm(total=total_steps, desc='Training')
        
        while self.global_step < total_steps:
            for batch in self.train_dataloader:
                # Training step
                loss_dict = self.train_step(batch)
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log(loss_dict, step=self.global_step)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{loss_dict['loss_total']:.4f}",
                    'lr': f"{loss_dict['learning_rate']:.2e}"
                })
                
                self.global_step += 1
                
                # Validation
                if self.global_step % val_interval == 0:
                    val_losses = self.validate()
                    
                    print(f"\nStep {self.global_step} Validation:")
                    for key, val in val_losses.items():
                        print(f"  {key}: {val:.4f}")
                    
                    if self.use_wandb:
                        wandb.log({f'val/{k}': v for k, v in val_losses.items()},
                                 step=self.global_step)
                    
                    # Save best model
                    if val_losses['loss_total'] < self.best_val_loss:
                        self.best_val_loss = val_losses['loss_total']
                        self.save_checkpoint(
                            f'checkpoints/step_{self.global_step}.pt',
                            is_best=True
                        )
                        print(f"  ✓ New best model saved!")
                
                # Periodic checkpoint
                if self.global_step % save_interval == 0:
                    self.save_checkpoint(f'checkpoints/step_{self.global_step}.pt')
                
                if self.global_step >= total_steps:
                    break
            
            self.epoch += 1
        
        pbar.close()
        print(f"\n✓ Training complete! Best val loss: {self.best_val_loss:.4f}")


# Example training configuration
def get_default_config() -> Dict:
    """Get default training configuration."""
    return {
        'learning_rate': 1e-4,
        'weight_decay': 0.05,
        'total_steps': 200000,
        'batch_size': 32,
        'max_grad_norm': 1.0,
        'min_lr': 1e-6,
        'lambda_diffusion': 1.0,
        'lambda_physics': 0.5,
        'lambda_cycle': 0.5,
        'val_interval': 1000,
        'save_interval': 5000,
        'num_workers': 4,
        'mixed_precision': True
    }


if __name__ == '__main__':
    print("Training framework loaded successfully!")
    print("\nDefault config:")
    config = get_default_config()
    for key, val in config.items():
        print(f"  {key}: {val}")
