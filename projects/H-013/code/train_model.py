"""
Main training script for SE(3)-Equivariant AFM Reconstruction
Run with: python scripts/train_model.py --config config.yaml
"""

import sys
sys.path.append('.')

import argparse
import torch
from pathlib import Path

from config import get_config, set_seed
from data.dataset import create_dataloaders
from models.joint_model import JointTipSurfaceModel
from training.trainer import AFMTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train SE(3)-Equivariant AFM Model')
    
    # Data
    parser.add_argument('--train_size', type=int, default=80000,
                       help='Training set size')
    parser.add_argument('--val_size', type=int, default=10000,
                       help='Validation set size')
    parser.add_argument('--use_cached', action='store_true',
                       help='Use cached datasets')
    parser.add_argument('--cache_dir', type=str, default='./data/cache',
                       help='Cache directory')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=300,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                       help='Weight decay')
    
    # Model
    parser.add_argument('--num_se3_layers', type=int, default=4,
                       help='Number of SE(3) layers')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension')
    
    # Experiment
    parser.add_argument('--experiment_name', type=str, default='se3_afm_v1',
                       help='Experiment name')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Log directory')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get configuration
    config = get_config()
    
    # Override config with command line arguments
    config.data.train_size = args.train_size
    config.data.val_size = args.val_size
    config.training.batch_size = args.batch_size
    config.training.epochs = args.epochs
    config.training.learning_rate = args.lr
    config.training.weight_decay = args.weight_decay
    config.training.num_workers = args.num_workers
    config.model.num_se3_layers = args.num_se3_layers
    config.model.hidden_dim = args.hidden_dim
    config.experiment_name = args.experiment_name
    config.checkpoint_dir = args.checkpoint_dir
    config.log_dir = args.log_dir
    config.seed = args.seed
    
    print("="*70)
    print("SE(3)-Equivariant AFM Reconstruction - Training")
    print("="*70)
    print(f"\nExperiment: {config.experiment_name}")
    print(f"Device: {config.training.device}")
    print(f"Mixed Precision: {config.training.mixed_precision}")
    print(f"\nDataset:")
    print(f"  Train size: {config.data.train_size}")
    print(f"  Val size: {config.data.val_size}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"\nModel:")
    print(f"  SE(3) layers: {config.model.num_se3_layers}")
    print(f"  Hidden dim: {config.model.hidden_dim}")
    print(f"  Max degree: {config.model.max_degree}")
    print(f"\nTraining:")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Weight decay: {config.training.weight_decay}")
    print("="*70 + "\n")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(
        config, 
        use_cached=args.use_cached,
        cache_dir=args.cache_dir
    )
    
    # Create model
    print("\nCreating model...")
    model = JointTipSurfaceModel(config).to(config.training.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Memory (FP32): ~{total_params * 4 / 1e6:.1f} MB")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = AFMTrainer(model, train_loader, val_loader, config)
    
    # Resume from checkpoint if specified
    if args.resume is not None:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70 + "\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint('interrupted.pt')
        print("✓ Checkpoint saved")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"\nBest validation loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {trainer.checkpoint_dir}")
    print(f"Logs saved to: {config.log_dir}/{config.experiment_name}")
    print("\nTo evaluate the model, run:")
    print(f"  python scripts/evaluate_model.py --checkpoint {trainer.checkpoint_dir}/best_model.pt")


if __name__ == '__main__':
    main()
