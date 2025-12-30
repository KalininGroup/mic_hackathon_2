"""
Comprehensive evaluation suite for AFM reconstruction
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from models.joint_model import JointTipSurfaceModel
from training.losses import SSIMLoss


class AFMEvaluator:
    """Comprehensive evaluation of reconstruction quality"""
    
    def __init__(self, model: JointTipSurfaceModel, config):
        self.model = model
        self.config = config
        self.device = config.training.device
        
        # Metrics
        self.ssim_loss = SSIMLoss()
        
    def evaluate_dataset(self, test_loader, save_dir: str = './results') -> Dict[str, Dict]:
        """
        Comprehensive evaluation on test dataset
        
        Returns:
            Dictionary with results for each tip/surface type combination
        """
        self.model.eval()
        
        results = {
            'overall': {'tip_errors': [], 'surface_errors': [], 
                       'tip_psnr': [], 'surface_psnr': [], 'ssim': []},
            'by_tip_type': {},
            'by_surface_type': {}
        }
        
        print("Evaluating on test set...")
        
        with torch.no_grad():
            for batch in tqdm(test_loader):
                afm_image = batch['image'].to(self.device)
                tip_gt = batch['tip'].to(self.device)
                surface_gt = batch['surface'].to(self.device)
                tip_type = batch['tip_type'][0]  # batch size 1 for test
                surface_type = batch['surface_type'][0]
                
                # Forward pass with uncertainty
                predictions = self.model(afm_image, monte_carlo_samples=10)
                
                # Compute metrics
                metrics = self._compute_metrics(
                    predictions['tip'], tip_gt,
                    predictions['surface'], surface_gt,
                    predictions['tip_uncertainty'],
                    predictions['surface_uncertainty']
                )
                
                # Store overall results
                for key, value in metrics.items():
                    if key in results['overall']:
                        results['overall'][key].append(value)
                
                # Store by tip type
                if tip_type not in results['by_tip_type']:
                    results['by_tip_type'][tip_type] = {k: [] for k in metrics.keys()}
                for key, value in metrics.items():
                    results['by_tip_type'][tip_type][key].append(value)
                
                # Store by surface type
                if surface_type not in results['by_surface_type']:
                    results['by_surface_type'][surface_type] = {k: [] for k in metrics.keys()}
                for key, value in metrics.items():
                    results['by_surface_type'][surface_type][key].append(value)
        
        # Compute statistics
        final_results = self._compute_statistics(results)
        
        # Save results
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self._save_results(final_results, save_path)
        
        # Generate plots
        self._plot_results(final_results, save_path)
        
        return final_results
    
    def _compute_metrics(self, tip_pred, tip_gt, surface_pred, surface_gt,
                        tip_unc=None, surface_unc=None) -> Dict[str, float]:
        """Compute all evaluation metrics"""
        metrics = {}
        
        # Tip reconstruction metrics
        tip_mse = torch.mean((tip_pred - tip_gt) ** 2)
        tip_rmse = torch.sqrt(tip_mse)
        tip_mae = torch.mean(torch.abs(tip_pred - tip_gt))
        tip_psnr = 20 * torch.log10(1.0 / torch.sqrt(tip_mse + 1e-10))
        
        metrics['tip_rmse'] = tip_rmse.item()
        metrics['tip_mae'] = tip_mae.item()
        metrics['tip_psnr'] = tip_psnr.item()
        
        # Surface reconstruction metrics
        surface_mse = torch.mean((surface_pred - surface_gt) ** 2)
        surface_rmse = torch.sqrt(surface_mse)
        surface_mae = torch.mean(torch.abs(surface_pred - surface_gt))
        surface_psnr = 20 * torch.log10(1.0 / torch.sqrt(surface_mse + 1e-10))
        
        metrics['surface_rmse'] = surface_rmse.item()
        metrics['surface_mae'] = surface_mae.item()
        metrics['surface_psnr'] = surface_psnr.item()
        
        # SSIM
        ssim = 1 - self.ssim_loss(surface_pred, surface_gt)
        metrics['ssim'] = ssim.item()
        
        # Uncertainty calibration (if available)
        if tip_unc is not None:
            tip_error = torch.abs(tip_pred - tip_gt)
            tip_calibration = self._compute_calibration(tip_error, tip_unc)
            metrics['tip_ece'] = tip_calibration
        
        if surface_unc is not None:
            surface_error = torch.abs(surface_pred - surface_gt)
            surface_calibration = self._compute_calibration(surface_error, surface_unc)
            metrics['surface_ece'] = surface_calibration
        
        return metrics
    
    def _compute_calibration(self, errors: torch.Tensor, 
                           uncertainties: torch.Tensor,
                           n_bins: int = 10) -> float:
        """Compute Expected Calibration Error (ECE)"""
        errors_flat = errors.flatten().cpu().numpy()
        uncertainties_flat = uncertainties.flatten().cpu().numpy()
        
        # Create bins
        bin_edges = np.linspace(0, uncertainties_flat.max(), n_bins + 1)
        
        ece = 0.0
        for i in range(n_bins):
            mask = (uncertainties_flat >= bin_edges[i]) & (uncertainties_flat < bin_edges[i+1])
            if mask.sum() > 0:
                mean_error = errors_flat[mask].mean()
                mean_uncertainty = uncertainties_flat[mask].mean()
                ece += np.abs(mean_error - mean_uncertainty) * mask.sum() / len(errors_flat)
        
        return ece
    
    def _compute_statistics(self, results: Dict) -> Dict:
        """Compute mean, std, median, etc. for all metrics"""
        final_results = {}
        
        # Overall statistics
        final_results['overall'] = {}
        for metric, values in results['overall'].items():
            final_results['overall'][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'p25': np.percentile(values, 25),
                'p75': np.percentile(values, 75),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        # By tip type
        final_results['by_tip_type'] = {}
        for tip_type, metrics in results['by_tip_type'].items():
            final_results['by_tip_type'][tip_type] = {}
            for metric, values in metrics.items():
                final_results['by_tip_type'][tip_type][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        
        # By surface type
        final_results['by_surface_type'] = {}
        for surface_type, metrics in results['by_surface_type'].items():
            final_results['by_surface_type'][surface_type] = {}
            for metric, values in metrics.items():
                final_results['by_surface_type'][surface_type][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        
        return final_results
    
    def _save_results(self, results: Dict, save_path: Path):
        """Save results to file"""
        import json
        
        # Convert to JSON-serializable format
        with open(save_path / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save as text
        with open(save_path / 'evaluation_results.txt', 'w') as f:
            f.write("="*70 + "\n")
            f.write("SE(3)-Equivariant AFM Reconstruction - Evaluation Results\n")
            f.write("="*70 + "\n\n")
            
            f.write("OVERALL RESULTS:\n")
            f.write("-"*70 + "\n")
            for metric, stats in results['overall'].items():
                f.write(f"{metric}:\n")
                f.write(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                f.write(f"  Median: {stats['median']:.4f}\n")
                f.write(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("RESULTS BY TIP TYPE:\n")
            f.write("="*70 + "\n")
            for tip_type, metrics in results['by_tip_type'].items():
                f.write(f"\n{tip_type.upper()}:\n")
                f.write("-"*70 + "\n")
                for metric, stats in metrics.items():
                    f.write(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("RESULTS BY SURFACE TYPE:\n")
            f.write("="*70 + "\n")
            for surface_type, metrics in results['by_surface_type'].items():
                f.write(f"\n{surface_type.upper()}:\n")
                f.write("-"*70 + "\n")
                for metric, stats in metrics.items():
                    f.write(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
        
        print(f"✓ Results saved to {save_path}")
    
    def _plot_results(self, results: Dict, save_path: Path):
        """Generate visualization plots"""
        sns.set_style("whitegrid")
        
        # Plot 1: Overall metrics
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Overall Reconstruction Performance', fontsize=16)
        
        metrics_to_plot = ['tip_rmse', 'surface_rmse', 'tip_psnr', 'surface_psnr', 'ssim']
        
        for idx, metric in enumerate(metrics_to_plot):
            if idx < 6:
                ax = axes[idx // 3, idx % 3]
                stats = results['overall'][metric]
                
                # Bar plot with error bars
                ax.bar([0], [stats['mean']], yerr=[stats['std']], 
                      capsize=5, alpha=0.7, color='steelblue')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_xticks([])
                ax.set_title(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        plt.tight_layout()
        plt.savefig(save_path / 'overall_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Comparison by tip type
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Performance by Tip Type', fontsize=16)
        
        tip_types = list(results['by_tip_type'].keys())
        tip_rmse = [results['by_tip_type'][t]['tip_rmse']['mean'] for t in tip_types]
        surface_rmse = [results['by_tip_type'][t]['surface_rmse']['mean'] for t in tip_types]
        
        axes[0].bar(tip_types, tip_rmse, alpha=0.7, color='coral')
        axes[0].set_ylabel('Tip RMSE')
        axes[0].set_title('Tip Reconstruction Error')
        axes[0].tick_params(axis='x', rotation=45)
        
        axes[1].bar(tip_types, surface_rmse, alpha=0.7, color='teal')
        axes[1].set_ylabel('Surface RMSE')
        axes[1].set_title('Surface Reconstruction Error')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path / 'by_tip_type.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Comparison by surface type
        fig, ax = plt.subplots(figsize=(10, 6))
        
        surface_types = list(results['by_surface_type'].keys())
        surface_rmse = [results['by_surface_type'][t]['surface_rmse']['mean'] for t in surface_types]
        ssim = [results['by_surface_type'][t]['ssim']['mean'] for t in surface_types]
        
        x = np.arange(len(surface_types))
        width = 0.35
        
        ax.bar(x - width/2, surface_rmse, width, label='RMSE', alpha=0.7, color='purple')
        ax2 = ax.twinx()
        ax2.bar(x + width/2, ssim, width, label='SSIM', alpha=0.7, color='green')
        
        ax.set_xlabel('Surface Type')
        ax.set_ylabel('Surface RMSE', color='purple')
        ax2.set_ylabel('SSIM', color='green')
        ax.set_xticks(x)
        ax.set_xticklabels(surface_types, rotation=45)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.title('Performance by Surface Type')
        plt.tight_layout()
        plt.savefig(save_path / 'by_surface_type.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plots saved to {save_path}")
    
    @torch.no_grad()
    def visualize_samples(self, test_loader, num_samples: int = 5, 
                         save_dir: str = './results/visualizations'):
        """Visualize reconstruction samples"""
        self.model.eval()
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        samples = []
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
            
            afm_image = batch['image'].to(self.device)
            tip_gt = batch['tip'].to(self.device)
            surface_gt = batch['surface'].to(self.device)
            
            predictions = self.model(afm_image, monte_carlo_samples=10)
            
            samples.append({
                'image': afm_image[0].cpu(),
                'tip_gt': tip_gt[0].cpu(),
                'tip_pred': predictions['tip'][0].cpu(),
                'tip_unc': predictions['tip_uncertainty'][0].cpu(),
                'surface_gt': surface_gt[0].cpu(),
                'surface_pred': predictions['surface'][0].cpu(),
                'surface_unc': predictions['surface_uncertainty'][0].cpu(),
                'simulated': predictions['simulated_image'][0].cpu()
            })
        
        # Create visualization
        for idx, sample in enumerate(samples):
            fig = plt.figure(figsize=(20, 10))
            gs = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.3)
            
            # Row 1: Images
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(sample['image'][0], cmap='afmhot')
            ax1.set_title('Input AFM Image')
            ax1.axis('off')
            
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(sample['surface_gt'][0], cmap='viridis')
            ax2.set_title('Ground Truth Surface')
            ax2.axis('off')
            
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.imshow(sample['surface_pred'][0], cmap='viridis')
            ax3.set_title('Predicted Surface')
            ax3.axis('off')
            
            ax4 = fig.add_subplot(gs[0, 3])
            error = torch.abs(sample['surface_pred'][0] - sample['surface_gt'][0])
            im4 = ax4.imshow(error, cmap='hot')
            ax4.set_title('Surface Error')
            ax4.axis('off')
            plt.colorbar(im4, ax=ax4, fraction=0.046)
            
            ax5 = fig.add_subplot(gs[0, 4])
            im5 = ax5.imshow(sample['surface_unc'][0], cmap='plasma')
            ax5.set_title('Surface Uncertainty')
            ax5.axis('off')
            plt.colorbar(im5, ax=ax5, fraction=0.046)
            
            # Row 2: Tip visualizations (center slices)
            ax6 = fig.add_subplot(gs[1, 0])
            tip_gt_slice = sample['tip_gt'][0, :, :, sample['tip_gt'].shape[3]//2]
            ax6.imshow(tip_gt_slice, cmap='viridis')
            ax6.set_title('Ground Truth Tip (slice)')
            ax6.axis('off')
            
            ax7 = fig.add_subplot(gs[1, 1])
            tip_pred_slice = sample['tip_pred'][0, :, :, sample['tip_pred'].shape[3]//2]
            ax7.imshow(tip_pred_slice, cmap='viridis')
            ax7.set_title('Predicted Tip (slice)')
            ax7.axis('off')
            
            ax8 = fig.add_subplot(gs[1, 2])
            tip_error = torch.abs(tip_pred_slice - tip_gt_slice)
            im8 = ax8.imshow(tip_error, cmap='hot')
            ax8.set_title('Tip Error (slice)')
            ax8.axis('off')
            plt.colorbar(im8, ax=ax8, fraction=0.046)
            
            ax9 = fig.add_subplot(gs[1, 3])
            im9 = ax9.imshow(sample['simulated'][0], cmap='afmhot')
            ax9.set_title('Simulated Image')
            ax9.axis('off')
            
            ax10 = fig.add_subplot(gs[1, 4])
            consistency_error = torch.abs(sample['simulated'][0] - sample['image'][0])
            im10 = ax10.imshow(consistency_error, cmap='hot')
            ax10.set_title('Consistency Error')
            ax10.axis('off')
            plt.colorbar(im10, ax=ax10, fraction=0.046)
            
            plt.suptitle(f'Reconstruction Sample {idx+1}', fontsize=16)
            plt.savefig(save_path / f'sample_{idx+1}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✓ Visualizations saved to {save_path}")


def test_evaluator():
    """Test evaluator"""
    from config import get_config, set_seed
    from data.dataset import create_dataloaders
    from models.joint_model import JointTipSurfaceModel
    
    set_seed(42)
    
    config = get_config()
    config.data.train_size = 50
    config.data.val_size = 20
    config.data.test_size = 20
    
    _, _, test_loader = create_dataloaders(config, use_cached=False)
    
    model = JointTipSurfaceModel(config).to(config.training.device)
    
    evaluator = AFMEvaluator(model, config)
    
    print("\nRunning evaluation...")
    results = evaluator.evaluate_dataset(test_loader, save_dir='./test_results')
    
    print("\nGenerating visualizations...")
    evaluator.visualize_samples(test_loader, num_samples=2, 
                               save_dir='./test_results/viz')
    
    print("\n✓ Evaluator test completed")

if __name__ == '__main__':
    test_evaluator()
