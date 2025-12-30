"""
Main evaluation script for SE(3)-Equivariant AFM Reconstruction
Run with: python scripts/evaluate_model.py --checkpoint path/to/checkpoint.pt
"""

import sys
sys.path.append('.')

import argparse
import torch
from pathlib import Path
import json

from config import get_config, set_seed
from data.dataset import create_dataloaders
from models.joint_model import JointTipSurfaceModel
from evaluation.evaluator import AFMEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SE(3)-Equivariant AFM Model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_size', type=int, default=10000,
                       help='Test set size')
    parser.add_argument('--use_cached', action='store_true',
                       help='Use cached test dataset')
    parser.add_argument('--cache_dir', type=str, default='./data/cache',
                       help='Cache directory')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Results directory')
    parser.add_argument('--num_visualizations', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--mc_samples', type=int, default=20,
                       help='Monte Carlo samples for uncertainty')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        print("Warning: Config not found in checkpoint, using default config")
        config = get_config()
    
    # Create model
    model = JointTipSurfaceModel(config).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'best_val_loss' in checkpoint:
        print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    return model, config


def print_results_table(results: dict):
    """Print formatted results table"""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    # Overall results
    print("\nOVERALL PERFORMANCE:")
    print("-"*80)
    print(f"{'Metric':<25} {'Mean':<15} {'Std':<15} {'Median':<15}")
    print("-"*80)
    
    for metric, stats in results['overall'].items():
        print(f"{metric:<25} {stats['mean']:>14.4f} {stats['std']:>14.4f} {stats['median']:>14.4f}")
    
    # Results by tip type
    print("\n" + "="*80)
    print("RESULTS BY TIP TYPE:")
    print("="*80)
    
    for tip_type, metrics in results['by_tip_type'].items():
        print(f"\n{tip_type.upper()}:")
        print("-"*80)
        print(f"{'Metric':<25} {'Mean':<15} {'Std':<15}")
        print("-"*80)
        for metric, stats in metrics.items():
            print(f"{metric:<25} {stats['mean']:>14.4f} {stats['std']:>14.4f}")
    
    # Results by surface type
    print("\n" + "="*80)
    print("RESULTS BY SURFACE TYPE:")
    print("="*80)
    
    for surface_type, metrics in results['by_surface_type'].items():
        print(f"\n{surface_type.upper()}:")
        print("-"*80)
        print(f"{'Metric':<25} {'Mean':<15} {'Std':<15}")
        print("-"*80)
        for metric, stats in metrics.items():
            print(f"{metric:<25} {stats['mean']:>14.4f} {stats['std']:>14.4f}")
    
    print("\n" + "="*80)


def generate_paper_table(results: dict, save_path: Path):
    """Generate LaTeX table for paper"""
    latex = []
    
    # Table 1: Tip Reconstruction Error by Type
    latex.append("% Table 1: Tip Reconstruction Error")
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Tip Reconstruction Error (nm) by Tip Type}")
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\hline")
    latex.append("Tip Type & RMSE & MAE & PSNR (dB) & Mean \\\\")
    latex.append("\\hline")
    
    for tip_type in ['pyramidal', 'conical', 'spherical', 'blunt']:
        if tip_type in results['by_tip_type']:
            metrics = results['by_tip_type'][tip_type]
            rmse = metrics['tip_rmse']['mean']
            mae = metrics['tip_mae']['mean']
            psnr = metrics['tip_psnr']['mean']
            latex.append(f"{tip_type.capitalize()} & {rmse:.2f} & {mae:.2f} & {psnr:.1f} & - \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    latex.append("")
    
    # Table 2: Surface Reconstruction Error by Type
    latex.append("% Table 2: Surface Reconstruction Error")
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Surface Reconstruction Performance by Surface Type}")
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\hline")
    latex.append("Surface Type & RMSE (nm) & SSIM & PSNR (dB) & Improvement \\\\")
    latex.append("\\hline")
    
    for surf_type in ['random', 'nanoparticles', 'steps', 'periodic']:
        if surf_type in results['by_surface_type']:
            metrics = results['by_surface_type'][surf_type]
            rmse = metrics['surface_rmse']['mean']
            ssim = metrics['ssim']['mean']
            psnr = metrics['surface_psnr']['mean']
            latex.append(f"{surf_type.capitalize()} & {rmse:.2f} & {ssim:.3f} & {psnr:.1f} & - \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    # Save to file
    with open(save_path / 'paper_tables.tex', 'w') as f:
        f.write('\n'.join(latex))
    
    print(f"✓ LaTeX tables saved to {save_path / 'paper_tables.tex'}")


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    print("="*80)
    print("SE(3)-Equivariant AFM Reconstruction - Evaluation")
    print("="*80)
    
    # Load model
    model, config = load_model_from_checkpoint(
        args.checkpoint, 
        device=config.training.device if 'config' in locals() else 'cuda'
    )
    model.eval()
    
    # Update config with eval parameters
    config.data.test_size = args.test_size
    config.model.monte_carlo_samples = args.mc_samples
    
    print(f"\nTest size: {args.test_size}")
    print(f"MC samples: {args.mc_samples}")
    print(f"Results directory: {args.results_dir}")
    
    # Create test dataloader
    print("\nCreating test dataloader...")
    _, _, test_loader = create_dataloaders(
        config,
        use_cached=args.use_cached,
        cache_dir=args.cache_dir
    )
    
    # Create evaluator
    evaluator = AFMEvaluator(model, config)
    
    # Run evaluation
    print("\n" + "="*80)
    print("Running Comprehensive Evaluation")
    print("="*80 + "\n")
    
    results = evaluator.evaluate_dataset(
        test_loader, 
        save_dir=args.results_dir
    )
    
    # Print results
    print_results_table(results)
    
    # Generate visualizations
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80 + "\n")
    
    viz_dir = Path(args.results_dir) / 'visualizations'
    evaluator.visualize_samples(
        test_loader,
        num_samples=args.num_visualizations,
        save_dir=str(viz_dir)
    )
    
    # Generate paper-ready tables
    print("\n" + "="*80)
    print("Generating Paper Tables")
    print("="*80 + "\n")
    
    generate_paper_table(results, Path(args.results_dir))
    
    # Final summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.results_dir}")
    print(f"  - evaluation_results.json: Full numerical results")
    print(f"  - evaluation_results.txt: Human-readable summary")
    print(f"  - paper_tables.tex: LaTeX tables for paper")
    print(f"  - *.png: Performance plots")
    print(f"  - visualizations/: Sample reconstructions")
    
    # Key metrics for paper
    print("\n" + "="*80)
    print("KEY METRICS FOR PAPER:")
    print("="*80)
    overall = results['overall']
    print(f"Tip RMSE: {overall['tip_rmse']['mean']:.3f} ± {overall['tip_rmse']['std']:.3f} nm")
    print(f"Surface RMSE: {overall['surface_rmse']['mean']:.3f} ± {overall['surface_rmse']['std']:.3f} nm")
    print(f"Surface PSNR: {overall['surface_psnr']['mean']:.2f} ± {overall['surface_psnr']['std']:.2f} dB")
    print(f"SSIM: {overall['ssim']['mean']:.4f} ± {overall['ssim']['std']:.4f}")
    if 'tip_ece' in overall:
        print(f"Tip ECE: {overall['tip_ece']['mean']:.4f} ± {overall['tip_ece']['std']:.4f}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
