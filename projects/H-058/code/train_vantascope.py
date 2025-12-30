"""
Train the complete VantaScope system with robust data handling.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vantascope.models.autoencoder import create_autoencoder
from vantascope.models.fuzzy_gat import create_fuzzy_gat
from vantascope.data.training_dataset import TrainingGrapheneDataset, custom_collate_fn
from vantascope.config import VantaScopeConfig
from vantascope.training.trainer import VantaScopeTrainer
from vantascope.utils.helpers import set_seed, ensure_dir
from vantascope.utils.logging import logger
import torch
from torch.utils.data import DataLoader, random_split

def create_robust_dataloader(dataset, batch_size, split="train", val_split=0.2):
    """Create DataLoader with robust batch handling."""
    
    if len(dataset) == 0:
        raise ValueError("Empty dataset")
    
    # Handle single sample case
    if len(dataset) == 1:
        logger.warning("Single sample dataset - no validation split")
        return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    
    # Train/validation split
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    
    # Reproducible split
    set_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    target_dataset = train_dataset if split == "train" else val_dataset
    target_shuffle = True if split == "train" else False
    
    logger.info(f"📊 {split.title()} split: {len(target_dataset)} samples")
    
    return DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=target_shuffle,
        num_workers=0,  # Keep at 0 for stability
        collate_fn=custom_collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # Drop incomplete batches
    )

def train_complete_system():
    """Train the complete VantaScope system."""
    logger.info("🚀 Training VantaScope on real microscopy data")
    
    set_seed(42)
    
    # Load configuration
    config = VantaScopeConfig.from_yaml("config/datasets_real.yaml")
    
    # Create training dataset (disable intelligent cropping to avoid CV2 dependency for now)
    dataset_config = config.datasets["graphene_stem"]
    dataset = TrainingGrapheneDataset(dataset_config, enable_intelligent_crop=False)
    
    if len(dataset) == 0:
        logger.error("No training data found!")
        return
    
    logger.info(f"📊 Training on {len(dataset)} graphene images")
    
    # Create data loaders with smaller batch size and custom collate
    train_loader = create_robust_dataloader(dataset, batch_size=2, split="train")
    val_loader = create_robust_dataloader(dataset, batch_size=2, split="val")
    
    # Create models
    logger.info("🧠 Initializing models...")
    autoencoder = create_autoencoder()
    fuzzy_gat = create_fuzzy_gat()
    
    # Training configuration optimized for stability
    training_config = {
        'learning_rate': 1e-4,  # Conservative learning rate
        'weight_decay': 0.01,
        'reconstruction_weight': 1.0,
        'kld_beta_weight': 0.01,  # Lower beta for better reconstruction
        'fuzzy_consistency_weight': 0.05,  
        'graph_sparsity_weight': 0.01,
        'scheduler_T0': 5
    }
    
    # Create trainer
    trainer = VantaScopeTrainer(
        autoencoder=autoencoder,
        fuzzy_gat=fuzzy_gat,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config
    )
    
    # Create save directory
    save_dir = ensure_dir("models/trained")
    
    # Train for fewer epochs initially
    num_epochs = 10  
    logger.info(f"🔥 Starting training for {num_epochs} epochs...")
    
    try:
        trainer.train(num_epochs=num_epochs, save_dir=save_dir)
        logger.info("🎉 Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_complete_system()
