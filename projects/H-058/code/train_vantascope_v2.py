"""
Improved training with better fuzzy classification learning.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vantascope.models.autoencoder import create_autoencoder
from vantascope.models.fuzzy_gat import create_fuzzy_gat
from vantascope.data.training_dataset import TrainingGrapheneDataset, custom_collate_fn
from vantascope.config import VantaScopeConfig
from vantascope.utils.helpers import set_seed, ensure_dir, get_device
from vantascope.utils.logging import logger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import time

def create_robust_dataloader(dataset, batch_size, split="train", val_split=0.2):
    """Create DataLoader with robust batch handling."""
    
    if len(dataset) == 0:
        raise ValueError("Empty dataset")
    
    if len(dataset) == 1:
        return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    
    set_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    target_dataset = train_dataset if split == "train" else val_dataset
    target_shuffle = True if split == "train" else False
    
    logger.info(f"📊 {split.title()} split: {len(target_dataset)} samples")
    
    return DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=target_shuffle,
        num_workers=0,
        collate_fn=custom_collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )


class ImprovedTrainer:
    """Standalone improved trainer with entropy regularization."""
    
    def __init__(self, autoencoder, fuzzy_gat, train_loader, val_loader, config):
        self.autoencoder = autoencoder
        self.fuzzy_gat = fuzzy_gat
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.device = get_device()
        self.autoencoder.to(self.device)
        self.fuzzy_gat.to(self.device)
        
        # Optimizers
        self.optimizer = optim.AdamW(
            list(self.autoencoder.parameters()) + list(self.fuzzy_gat.parameters()),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=config.get('scheduler_T0', 5), T_mult=2
        )
        
        self.mse_loss = nn.MSELoss()
        self.metrics_history = {'train_loss': [], 'val_loss': []}
        
        logger.info("🚀 Improved trainer initialized with entropy regularization")
    
    def compute_losses(self, images, cae_outputs, gat_outputs):
        """Compute losses with entropy regularization."""
        
        # 1. Reconstruction Loss
        reconstruction_loss = self.mse_loss(cae_outputs['reconstruction'], images)
        
        # 2. KLD Loss
        latent = cae_outputs['latent']
        kld_loss = 0.5 * torch.sum(latent.pow(2), dim=1).mean()
        
        # 3. Fuzzy Consistency Loss
        fuzzy_memberships = gat_outputs['fuzzy_memberships']
        node_predictions = gat_outputs['node_predictions']
        fuzzy_loss = self.mse_loss(fuzzy_memberships, torch.sigmoid(node_predictions))
        
        # 4. ENTROPY REGULARIZATION - Encourage diverse classifications!
        membership_probs = fuzzy_memberships / (fuzzy_memberships.sum(dim=1, keepdim=True) + 1e-8)
        entropy = -(membership_probs * torch.log(membership_probs + 1e-8)).sum(dim=1).mean()
        entropy_loss = -entropy  # Minimize negative entropy = maximize entropy
        
        # 5. DIVERSITY LOSS - Ensure balanced category usage
        mean_membership = fuzzy_memberships.mean(dim=0)
        target_uniform = torch.ones_like(mean_membership) / 5.0
        diversity_loss = self.mse_loss(mean_membership, target_uniform)
        
        # 6. Graph Sparsity Loss
        edge_weights = gat_outputs['edge_weights']
        graph_loss = torch.abs(edge_weights).mean()
        
        # Total weighted loss
        total_loss = (
            1.0 * reconstruction_loss +
            0.005 * kld_loss +
            0.1 * fuzzy_loss +
            0.01 * graph_loss +
            0.5 * entropy_loss +
            0.3 * diversity_loss
        )
        
        return {
            'total': total_loss,
            'reconstruction': reconstruction_loss.item(),
            'entropy': entropy_loss.item(),
            'diversity': diversity_loss.item()
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.autoencoder.train()
        self.fuzzy_gat.train()
        
        total_loss = 0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            
            self.optimizer.zero_grad()
            
            cae_outputs = self.autoencoder(images)
            gat_outputs = self.fuzzy_gat(cae_outputs['patch_embeddings'])
            
            losses = self.compute_losses(images, cae_outputs, gat_outputs)
            losses['total'].backward()
            
            torch.nn.utils.clip_grad_norm_(
                list(self.autoencoder.parameters()) + list(self.fuzzy_gat.parameters()),
                max_norm=1.0
            )
            
            self.optimizer.step()
            total_loss += losses['total'].item()
            
            if batch_idx % 5 == 0:
                logger.info(f"   Epoch {epoch}, Batch {batch_idx}/{num_batches} - "
                           f"Loss: {losses['total'].item():.4f}, "
                           f"Recon: {losses['reconstruction']:.4f}, "
                           f"Entropy: {losses['entropy']:.4f}, "
                           f"Diversity: {losses['diversity']:.4f}")
        
        return total_loss / num_batches
    
    def validate_epoch(self):
        """Validate."""
        self.autoencoder.eval()
        self.fuzzy_gat.eval()
        
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                cae_outputs = self.autoencoder(images)
                gat_outputs = self.fuzzy_gat(cae_outputs['patch_embeddings'])
                losses = self.compute_losses(images, cae_outputs, gat_outputs)
                total_loss += losses['total'].item()
        
        return total_loss / num_batches
    
    def train(self, num_epochs, save_dir):
        """Complete training loop."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            start = time.time()
            
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch()
            
            self.scheduler.step()
            
            elapsed = time.time() - start
            logger.info(f"🔥 Epoch {epoch+1}/{num_epochs} ({elapsed:.1f}s) - "
                       f"Train: {train_loss:.4f}, Val: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(save_dir / "best_model.pth", epoch, val_loss)
                logger.info(f"💾 Saved best model (val_loss: {val_loss:.4f})")
        
        self.save_checkpoint(save_dir / "final_model.pth", num_epochs-1, val_loss)
        logger.info("🎉 Training complete!")
    
    def save_checkpoint(self, path, epoch, val_loss):
        """Save checkpoint."""
        torch.save({
            'epoch': epoch,
            'autoencoder_state': self.autoencoder.state_dict(),
            'fuzzy_gat_state': self.fuzzy_gat.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'metrics': {'total_loss': val_loss}
        }, path)


def train_improved_system():
    """Train with improved fuzzy learning."""
    logger.info("🚀 Training VantaScope v2 with entropy regularization")
    
    set_seed(42)
    
    config = VantaScopeConfig.from_yaml("config/datasets_real.yaml")
    dataset_config = config.datasets["graphene_stem"]
    dataset = TrainingGrapheneDataset(dataset_config, enable_intelligent_crop=False)
    
    if len(dataset) == 0:
        logger.error("No training data found!")
        return
    
    logger.info(f"📊 Training on {len(dataset)} graphene images")
    
    train_loader = create_robust_dataloader(dataset, batch_size=2, split="train")
    val_loader = create_robust_dataloader(dataset, batch_size=2, split="val")
    
    logger.info("🧠 Initializing fresh models...")
    autoencoder = create_autoencoder()
    fuzzy_gat = create_fuzzy_gat()
    
    training_config = {
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'scheduler_T0': 5
    }
    
    trainer = ImprovedTrainer(
        autoencoder=autoencoder,
        fuzzy_gat=fuzzy_gat,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config
    )
    
    save_dir = ensure_dir("models/trained_v2")
    
    num_epochs = 30
    logger.info(f"🔥 Starting improved training for {num_epochs} epochs...")
    
    try:
        trainer.train(num_epochs=num_epochs, save_dir=save_dir)
    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_improved_system()
