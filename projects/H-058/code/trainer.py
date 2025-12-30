"""
VantaScope Training Pipeline: Complete implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
import numpy as np
from pathlib import Path
import time

from ..models.autoencoder import DINOv2CAE
from ..models.fuzzy_gat import FuzzyGAT
from ..utils.logging import logger
from ..utils.helpers import get_device


class VantaScopeTrainer:
    """Complete training pipeline for CAE + Fuzzy-GAT."""
    
    def __init__(self, 
                 autoencoder: DINOv2CAE,
                 fuzzy_gat: FuzzyGAT,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict[str, Any]):
        
        self.autoencoder = autoencoder
        self.fuzzy_gat = fuzzy_gat
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.device = get_device()
        self.autoencoder.to(self.device)
        self.fuzzy_gat.to(self.device)
        
        # Multi-objective loss weights
        self.loss_weights = {
            'reconstruction': config.get('reconstruction_weight', 1.0),
            'kld_beta': config.get('kld_beta_weight', 0.02),
            'fuzzy_consistency': config.get('fuzzy_consistency_weight', 0.1),
            'graph_sparsity': config.get('graph_sparsity_weight', 0.05),
        }
        
        # Optimizers
        self.optimizer = optim.AdamW(
            list(self.autoencoder.parameters()) + list(self.fuzzy_gat.parameters()),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.05)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=config.get('scheduler_T0', 10),
            T_mult=2
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        
        # Training tracking
        self.metrics_history = {
            'train_loss': [], 'val_loss': [],
            'reconstruction_loss': [], 'fuzzy_loss': [], 'graph_loss': []
        }
        
        logger.info(f"🚀 VantaScope trainer initialized")
        logger.info(f"   Loss weights: {self.loss_weights}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.autoencoder.parameters()) + sum(p.numel() for p in self.fuzzy_gat.parameters())
        logger.info(f"   Parameters: {total_params:,}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.autoencoder.train()
        self.fuzzy_gat.train()
        
        epoch_metrics = {
            'total_loss': 0, 'reconstruction_loss': 0, 
            'kld_loss': 0, 'fuzzy_loss': 0, 'graph_loss': 0
        }
        
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass through autoencoder
            cae_outputs = self.autoencoder(images)
            
            # Forward pass through Fuzzy-GAT
            gat_outputs = self.fuzzy_gat(cae_outputs['patch_embeddings'])
            
            # Compute losses
            losses = self._compute_losses(images, cae_outputs, gat_outputs)
            total_loss = losses['total_loss']
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                list(self.autoencoder.parameters()) + list(self.fuzzy_gat.parameters()),
                max_norm=1.0
            )
            
            self.optimizer.step()
            
            # Update metrics
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    epoch_metrics[key] += value.item()
            
            # Log progress
            if batch_idx % 5 == 0:
                progress = batch_idx / num_batches * 100
                logger.info(f"   Epoch {epoch}, Batch {batch_idx}/{num_batches} ({progress:.1f}%) - Loss: {total_loss.item():.6f}")
        
        # Average metrics over epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.autoencoder.eval()
        self.fuzzy_gat.eval()
        
        val_metrics = {
            'total_loss': 0, 'reconstruction_loss': 0,
            'kld_loss': 0, 'fuzzy_loss': 0, 'graph_loss': 0
        }
        
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                
                # Forward passes
                cae_outputs = self.autoencoder(images)
                gat_outputs = self.fuzzy_gat(cae_outputs['patch_embeddings'])
                
                # Compute losses
                losses = self._compute_losses(images, cae_outputs, gat_outputs)
                
                # Update metrics
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor):
                        val_metrics[key] += value.item()
        
        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        return val_metrics
    
    def _compute_losses(self, images: torch.Tensor, cae_outputs: Dict, gat_outputs: Dict) -> Dict[str, torch.Tensor]:
        """Compute multi-objective loss."""
        losses = {}
        
        # 1. Reconstruction Loss
        reconstruction = cae_outputs['reconstruction']
        reconstruction_loss = self.mse_loss(reconstruction, images)
        losses['reconstruction_loss'] = reconstruction_loss
        
        # 2. KLD Loss for disentanglement (simplified)
        latent = cae_outputs['latent']
        kld_loss = 0.5 * torch.sum(latent.pow(2), dim=1).mean()
        losses['kld_loss'] = kld_loss
        
        # 3. Fuzzy Consistency Loss
        fuzzy_memberships = gat_outputs['fuzzy_memberships']
        node_predictions = gat_outputs['node_predictions']
        fuzzy_loss = self.mse_loss(fuzzy_memberships, torch.sigmoid(node_predictions))
        losses['fuzzy_loss'] = fuzzy_loss
        
        # 4. Graph Sparsity Loss
        edge_weights = gat_outputs['edge_weights']
        graph_loss = torch.abs(edge_weights).mean()
        losses['graph_loss'] = graph_loss
        
        # 5. Total weighted loss
        total_loss = (
            self.loss_weights['reconstruction'] * reconstruction_loss +
            self.loss_weights['kld_beta'] * kld_loss +
            self.loss_weights['fuzzy_consistency'] * fuzzy_loss +
            self.loss_weights['graph_sparsity'] * graph_loss
        )
        
        losses['total_loss'] = total_loss
        return losses
    
    def train(self, num_epochs: int, save_dir: Optional[Path] = None) -> None:
        """Complete training loop."""
        logger.info(f"🚀 Starting VantaScope training for {num_epochs} epochs")
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate epoch
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step()
            
            # Update metrics history
            self.metrics_history['train_loss'].append(train_metrics['total_loss'])
            self.metrics_history['val_loss'].append(val_metrics['total_loss'])
            self.metrics_history['reconstruction_loss'].append(train_metrics['reconstruction_loss'])
            self.metrics_history['fuzzy_loss'].append(train_metrics['fuzzy_loss'])
            self.metrics_history['graph_loss'].append(train_metrics['graph_loss'])
            
            epoch_time = time.time() - start_time
            
            # Log epoch summary
            logger.info(f"🔥 Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            logger.info(f"   Train Loss: {train_metrics['total_loss']:.6f}")
            logger.info(f"   Val Loss: {val_metrics['total_loss']:.6f}")
            logger.info(f"   Reconstruction: {train_metrics['reconstruction_loss']:.6f}")
            logger.info(f"   Fuzzy: {train_metrics['fuzzy_loss']:.6f}")
            logger.info(f"   Graph: {train_metrics['graph_loss']:.6f}")
            logger.info(f"   LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if save_dir and val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                self.save_checkpoint(save_dir / "best_model.pth", epoch, val_metrics)
                logger.info(f"💾 Saved best model (val_loss: {best_val_loss:.6f})")
        
        logger.info("🎉 Training complete!")
        
        # Save final model
        if save_dir:
            self.save_checkpoint(save_dir / "final_model.pth", num_epochs-1, val_metrics)
            logger.info("💾 Saved final model")
    
    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'autoencoder_state': self.autoencoder.state_dict(),
            'fuzzy_gat_state': self.fuzzy_gat.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'metrics': metrics,
            'metrics_history': self.metrics_history,
            'config': self.config
        }
        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved to {path}")
