"""
Main execution script for Neural CDE 4D-STEM Reconstruction

Usage:
    python main.py --mode train
    python main.py --mode eval --checkpoint checkpoints/best_model.pt
    python main.py --mode both
"""

import torch
import argparse
from pathlib import Path
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig
from data.dataset import Simulated4DSTEMDataset
from models.neural_cde import NeuralCDE4DSTEM
from training.trainer import Trainer
from evaluation.evaluator import Evaluator


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Neural CDE for 4D-STEM Reconstruction')
    
    parser.add_argument('--mode', type=str, default='both', 
                       choices=['train', 'eval', 'both'],
                       help='Mode: train, eval, or both')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')
    
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for evaluation or resuming training')
    
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    
    parser.add_argument('--sparse_rate', type=float, default=None,
                       help='Override sparse sampling rate')
    
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: auto-detect)')
    
    parser.add_argument('--n_train', type=int, default=800,
                       help='Number of training sequences')
    
    parser.add_argument('--n_val', type=int, default=100,
                       help='Number of validation sequences')
    
    parser.add_argument('--n_test', type=int, default=100,
                       help='Number of test sequences')
    
    return parser.parse_args()


def create_config(args):
    """Create configuration from arguments"""
    if args.config:
        print(f"Loading configuration from {args.config}")
        config = ModelConfig.from_yaml(args.config)
    else:
        print("Using default configuration")
        config = ModelConfig()
    
    # Override with command line arguments
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.sparse_rate is not None:
        config.sparse_sampling_rate = args.sparse_rate
    
    return config


def setup_directories(config):
    """Create necessary directories"""
    Path(config.data_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    print("✓ Directories created")


def create_datasets(config, args):
    """Create train/val/test datasets"""
    print("\nCreating datasets...")
    print(f"  Training sequences: {args.n_train}")
    print(f"  Validation sequences: {args.n_val}")
    print(f"  Test sequences: {args.n_test}")
    
    train_dataset = Simulated4DSTEMDataset(config, mode='train', n_sequences=args.n_train)
    val_dataset = Simulated4DSTEMDataset(config, mode='val', n_sequences=args.n_val)
    test_dataset = Simulated4DSTEMDataset(config, mode='test', n_sequences=args.n_test)
    
    print("✓ Datasets created")
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset, config):
    """Create data loaders"""
    print("\nCreating data loaders...")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    print("✓ Data loaders created")
    
    return train_loader, val_loader, test_loader


def create_model(config, device):
    """Create Neural CDE model"""
    print("\nCreating model...")
    model = NeuralCDE4DSTEM(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print("✓ Model created")
    
    return model


def train_model(model, train_loader, val_loader, config, device, checkpoint_path=None):
    """Train the model"""
    print(f"\n{'='*60}")
    print("TRAINING PHASE")
    print(f"{'='*60}")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config, device)
    
    # Load checkpoint if provided
    if checkpoint_path:
        print(f"\nLoading checkpoint from {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)
    
    # Save configuration
    config_save_path = Path(config.checkpoint_dir) / 'config.yaml'
    config.to_yaml(str(config_save_path))
    print(f"Configuration saved to {config_save_path}")
    
    # Train
    trainer.train()
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"{'='*60}")
    
    return trainer


def evaluate_model(model, test_loader, config, device, checkpoint_path=None):
    """Evaluate the model"""
    print(f"\n{'='*60}")
    print("EVALUATION PHASE")
    print(f"{'='*60}")
    
    # Load checkpoint if provided
    if checkpoint_path:
        print(f"\nLoading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from epoch {checkpoint['epoch']}")
        print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
    else:
        print("\n⚠ Warning: No checkpoint provided, using current model state")
    
    # Create evaluator
    evaluator = Evaluator(model, config, device)
    
    # Evaluate
    results = evaluator.evaluate_model(test_loader, save_plots=True)
    
    return results


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_args()
    
    # Print header
    print("\n" + "="*60)
    print("NEURAL CDE FOR 4D-STEM RECONSTRUCTION")
    print("="*60)
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Create configuration
    config = create_config(args)
    print(f"\n{config}")
    
    # Setup directories
    setup_directories(config)
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(config, args)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, config
    )
    
    # Create model
    model = create_model(config, device)
    
    # Training
    if args.mode in ['train', 'both']:
        trainer = train_model(
            model, train_loader, val_loader, config, device, args.checkpoint
        )
        # Update checkpoint path to best model for evaluation
        best_checkpoint = Path(config.checkpoint_dir) / 'best_model.pt'
        if best_checkpoint.exists():
            args.checkpoint = str(best_checkpoint)
    
    # Evaluation
    if args.mode in ['eval', 'both']:
        # Use best checkpoint if we just trained
        if args.mode == 'both':
            checkpoint_path = Path(config.checkpoint_dir) / 'best_model.pt'
            if checkpoint_path.exists():
                args.checkpoint = str(checkpoint_path)
        
        results = evaluate_model(
            model, test_loader, config, device, args.checkpoint
        )
    
    # Final summary
    print(f"\n{'='*60}")
    print("EXECUTION COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {config.results_dir}")
    print(f"Checkpoints saved to: {config.checkpoint_dir}")
    print("\n✓ All done!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
