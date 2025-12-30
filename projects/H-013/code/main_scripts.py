"""
physics_diffusion/scripts/train.py
physics_diffusion/scripts/evaluate.py
physics_diffusion/scripts/reproduce_paper.py

Main execution scripts for training, evaluation, and reproduction.
"""

# ============================================================================
# TRAIN.PY - Main Training Script
# ============================================================================

"""
Usage:
    python scripts/train.py --config configs/full_model.yaml
    python scripts/train.py --config configs/ablation_no_physics.yaml
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.diffusion_model import PhysicsConstrainedDDPM
from training.diffusion_trainer import DiffusionTrainer, get_default_config
from data.dataloaders import create_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='Train physics-constrained diffusion model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--data_dir', type=str, default='data/simulated',
                       help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Path to save outputs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print("="*80)
    print("Configuration:")
    print("="*80)
    for key, val in config.items():
        print(f"  {key}: {val}")
    print("="*80 + "\n")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=Path(args.data_dir),
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 4)
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}\n")
    
    # Create model
    print("Creating model...")
    model = PhysicsConstrainedDDPM(
        image_size=config.get('image_size', 256),
        in_channels=config.get('in_channels', 1),
        model_channels=config.get('model_channels', 128),
        num_modalities=config.get('num_modalities', 4),
        timesteps=config.get('timesteps', 1000),
        use_physics=config.get('use_physics', True),
        modality_names=config.get('modality_names', ['AFM', 'TEM', 'SEM', 'STEM'])
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}\n")
    
    # Create trainer
    trainer = DiffusionTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=config,
        device=args.device,
        use_wandb=args.use_wandb
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(
        total_steps=config['total_steps'],
        val_interval=config.get('val_interval', 1000),
        save_interval=config.get('save_interval', 5000)
    )
    
    print("\n" + "="*80)
    print("Training completed successfully!")
    print("="*80)


if __name__ == '__main__':
    main()


# ============================================================================
# EVALUATE.PY - Evaluation Script
# ============================================================================

"""
Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best.pt \
                               --data_dir data/real/AFM \
                               --output_dir results/afm
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.diffusion_model import PhysicsConstrainedDDPM
from evaluation.metrics_calculator import MetricsCalculator
from data.dataloaders import SimulatedMicroscopyDataset
from torch.utils.data import DataLoader


def parse_eval_args():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to save results')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to evaluate (None = all)')
    parser.add_argument('--save_images', action='store_true',
                       help='Save comparison images')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    return parser.parse_args()


def evaluate_model(
    model,
    dataloader,
    metrics_calc,
    device,
    output_dir,
    save_images=False
):
    """Evaluate model and save results."""
    model.eval()
    
    all_metrics = []
    saved_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Evaluating')):
            clean = batch['clean_images'].to(device)
            noisy = batch['noisy_images'].to(device)
            modality_ids = batch['modality_ids'].to(device)
            
            # Generate predictions
            pred = model.sample(noisy, modality_ids, num_inference_steps=50)
            
            # Calculate metrics
            batch_metrics = metrics_calc.calculate_all_metrics(pred, clean)
            all_metrics.append(batch_metrics)
            
            # Save images
            if save_images and saved_count < 20:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                img_idx = 0
                axes[0].imshow(noisy[img_idx, 0].cpu(), cmap='gray')
                axes[0].set_title('Noisy Input')
                axes[0].axis('off')
                
                axes[1].imshow(pred[img_idx, 0].cpu(), cmap='gray')
                axes[1].set_title('Prediction')
                axes[1].axis('off')
                
                axes[2].imshow(clean[img_idx, 0].cpu(), cmap='gray')
                axes[2].set_title('Ground Truth')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(output_dir / f'comparison_{saved_count:03d}.png', 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
                saved_count += 1
    
    # Aggregate metrics
    aggregated = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        aggregated[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    return aggregated


def main_evaluate():
    args = parse_eval_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint['config']
    
    # Create model
    model = PhysicsConstrainedDDPM(
        image_size=config.get('image_size', 256),
        in_channels=config.get('in_channels', 1),
        model_channels=config.get('model_channels', 128),
        num_modalities=config.get('num_modalities', 4),
        timesteps=config.get('timesteps', 1000),
        use_physics=config.get('use_physics', True)
    ).to(args.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dataset and dataloader
    dataset = SimulatedMicroscopyDataset(
        clean_image_dir=Path(args.data_dir),
        num_samples=args.num_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Create metrics calculator
    metrics_calc = MetricsCalculator(device=args.device)
    
    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_model(
        model,
        dataloader,
        metrics_calc,
        args.device,
        output_dir,
        save_images=args.save_images
    )
    
    # Print results
    print("\n" + "="*80)
    print("Evaluation Results:")
    print("="*80)
    for metric, values in results.items():
        print(f"\n{metric.upper()}:")
        for stat, val in values.items():
            print(f"  {stat}: {val:.4f}")
    
    # Save results to JSON
    results_file = output_dir / 'metrics.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    print("="*80)


if __name__ == '__main__':
    main_evaluate()


# ============================================================================
# REPRODUCE_PAPER.PY - One-Click Reproduction Script
# ============================================================================

"""
Usage:
    python scripts/reproduce_paper.py --quick_test  # Fast test run
    python scripts/reproduce_paper.py --full        # Full reproduction
"""

import argparse
import subprocess
import sys
from pathlib import Path
import yaml


def run_command(cmd: str):
    """Run shell command and print output."""
    print(f"\n{'='*80}")
    print(f"Running: {cmd}")
    print('='*80)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error running command: {cmd}")
        sys.exit(1)


def reproduce_paper(quick_test=False):
    """Run all experiments to reproduce paper results."""
    
    print("="*80)
    print("REPRODUCING PAPER RESULTS")
    print("="*80)
    
    # Determine configuration based on mode
    if quick_test:
        print("\nRunning in QUICK TEST mode")
        print("This will use reduced settings for fast testing\n")
        configs = ['configs/quick_test.yaml']
        num_steps = 1000
    else:
        print("\nRunning in FULL mode")
        print("This will reproduce all paper experiments\n")
        configs = [
            'configs/full_model.yaml',
            'configs/ablation_no_physics.yaml',
            'configs/ablation_no_modality.yaml',
            'configs/ablation_only_diffusion.yaml'
        ]
        num_steps = 200000
    
    # Step 1: Train all models
    print("\n" + "="*80)
    print("STEP 1: Training Models")
    print("="*80)
    
    for config in configs:
        cmd = f"python scripts/train.py --config {config} --use_wandb"
        run_command(cmd)
    
    # Step 2: Evaluate all models
    print("\n" + "="*80)
    print("STEP 2: Evaluating Models")
    print("="*80)
    
    checkpoints = [
        'outputs/checkpoints/full_model_best.pt',
        'outputs/checkpoints/no_physics_best.pt',
        'outputs/checkpoints/no_modality_best.pt',
        'outputs/checkpoints/only_diffusion_best.pt'
    ]
    
    for checkpoint in checkpoints:
        cmd = (f"python scripts/evaluate.py "
               f"--checkpoint {checkpoint} "
               f"--data_dir data/test "
               f"--output_dir results/{Path(checkpoint).stem} "
               f"--save_images")
        run_command(cmd)
    
    # Step 3: Generate tables and figures
    print("\n" + "="*80)
    print("STEP 3: Generating Tables and Figures")
    print("="*80)
    
    cmd = "python scripts/generate_results.py --results_dir results --output_dir paper_outputs"
    run_command(cmd)
    
    print("\n" + "="*80)
    print("✓ REPRODUCTION COMPLETE!")
    print("="*80)
    print("\nResults are saved in:")
    print("  - Trained models: outputs/checkpoints/")
    print("  - Evaluation metrics: results/")
    print("  - Paper figures: paper_outputs/figures/")
    print("  - Paper tables: paper_outputs/tables/")


def main_reproduce():
    parser = argparse.ArgumentParser(description='Reproduce paper results')
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick test with reduced settings')
    parser.add_argument('--full', action='store_true',
                       help='Run full reproduction')
    args = parser.parse_args()
    
    if not (args.quick_test or args.full):
        print("Please specify either --quick_test or --full")
        sys.exit(1)
    
    reproduce_paper(quick_test=args.quick_test)


if __name__ == '__main__':
    main_reproduce()
