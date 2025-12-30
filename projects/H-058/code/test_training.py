"""
Test the complete VantaScope training pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vantascope.models.autoencoder import create_autoencoder
from vantascope.models.fuzzy_gat import create_fuzzy_gat
from vantascope.data.factory import VantaScopeDataFactory
from vantascope.config import VantaScopeConfig
from vantascope.training.trainer import VantaScopeTrainer
from vantascope.utils.helpers import set_seed
from vantascope.utils.logging import logger

def test_training():
    """Test training pipeline with real data."""
    logger.info("🚀 Testing VantaScope training pipeline")
    
    set_seed(42)
    
    # Load config and data
    config = VantaScopeConfig.from_yaml("config/datasets_real.yaml")
    dataset = VantaScopeDataFactory.create_dataset(config.datasets["graphene_stem"])
    
    # Create train/val loaders
    train_loader = VantaScopeDataFactory.create_dataloader(dataset, batch_size=2, split="train")
    val_loader = VantaScopeDataFactory.create_dataloader(dataset, batch_size=2, split="val")
    
    # Create models
    autoencoder = create_autoencoder()
    fuzzy_gat = create_fuzzy_gat()
    
    # Training configuration
    training_config = {
        'learning_rate': 1e-4,
        'weight_decay': 0.05,
        'reconstruction_weight': 1.0,
        'kld_beta_weight': 0.02,
        'fuzzy_consistency_weight': 0.1,
        'graph_sparsity_weight': 0.05,
        'scheduler_T0': 10
    }
    
    # Create trainer
    trainer = VantaScopeTrainer(
        autoencoder=autoencoder,
        fuzzy_gat=fuzzy_gat,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config
    )
    
    # Test one training step
    logger.info("🔥 Testing one training epoch...")
    train_metrics = trainer.train_epoch(epoch=0)
    
    logger.info("📊 Training metrics:")
    for key, value in train_metrics.items():
        logger.info(f"   {key}: {value:.6f}")
    
    # Test validation
    val_metrics = trainer.validate_epoch()
    logger.info("📊 Validation metrics:")
    for key, value in val_metrics.items():
        logger.info(f"   {key}: {value:.6f}")
    
    logger.info("🎉 Training pipeline test complete!")

if __name__ == "__main__":
    test_training()
