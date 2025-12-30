"""
physics_diffusion/evaluation/metrics_calculator.py
physics_diffusion/experiments/ablation_studies.py

Comprehensive evaluation metrics and automated ablation studies.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from pathlib import Path
import json
import yaml
from dataclasses import dataclass, asdict
from tqdm import tqdm


# ============================================================================
# Metrics Calculator
# ============================================================================

class MetricsCalculator:
    """Comprehensive metrics for microscopy image evaluation."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
        # Load LPIPS model
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        except ImportError:
            print("Warning: lpips not installed. LPIPS metric unavailable.")
            self.lpips_fn = None
    
    def calculate_psnr(self, pred: torch.Tensor, target: torch.Tensor,
                       data_range: float = 1.0) -> float:
        """Calculate Peak Signal-to-Noise Ratio.
        
        Args:
            pred: Predicted image [B, C, H, W] or [C, H, W]
            target: Ground truth image
            data_range: Maximum possible pixel value
            
        Returns:
            PSNR in dB
        """
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(torch.tensor(data_range) / torch.sqrt(mse)).item()
    
    def calculate_ssim(self, pred: torch.Tensor, target: torch.Tensor,
                       window_size: int = 11, data_range: float = 1.0) -> float:
        """Calculate Structural Similarity Index.
        
        Uses Gaussian window for local statistics.
        """
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        
        # Create Gaussian window
        sigma = 1.5
        gauss = torch.Tensor([
            np.exp(-(x - window_size//2)**2 / (2 * sigma**2))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        
        # 2D window
        window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)
        window = window.to(pred.device)
        
        # Ensure 4D tensors
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)
        
        # Calculate statistics
        mu1 = F.conv2d(pred, window, padding=window_size//2, groups=pred.shape[1])
        mu2 = F.conv2d(target, window, padding=window_size//2, groups=target.shape[1])
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred**2, window, padding=window_size//2, 
                            groups=pred.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(target**2, window, padding=window_size//2,
                            groups=target.shape[1]) - mu2_sq
        sigma12 = F.conv2d(pred*target, window, padding=window_size//2,
                          groups=pred.shape[1]) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean().item()
    
    def calculate_lpips(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Learned Perceptual Image Patch Similarity.
        
        Requires lpips library and pretrained network.
        """
        if self.lpips_fn is None:
            return float('nan')
        
        # Ensure correct format [-1, 1] range and RGB
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)
        
        with torch.no_grad():
            lpips_value = self.lpips_fn(pred, target)
        
        return lpips_value.mean().item()
    
    def calculate_frc(self, pred: torch.Tensor, target: torch.Tensor,
                     threshold: float = 0.5) -> Dict[str, float]:
        """Calculate Fourier Ring Correlation.
        
        Measures resolution via correlation in Fourier space.
        
        Returns:
            Dictionary with FRC curve and cutoff resolution
        """
        # Ensure 2D
        if pred.dim() == 4:
            pred = pred[0, 0]
        elif pred.dim() == 3:
            pred = pred[0]
        
        if target.dim() == 4:
            target = target[0, 0]
        elif target.dim() == 3:
            target = target[0]
        
        # FFT
        fft_pred = torch.fft.fft2(pred)
        fft_target = torch.fft.fft2(target)
        
        H, W = pred.shape
        cy, cx = H // 2, W // 2
        
        # Create radial bins
        y = torch.arange(H) - cy
        x = torch.arange(W) - cx
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        radius = torch.sqrt(xx**2 + yy**2).to(pred.device)
        
        max_radius = int(min(H, W) / 2)
        frc_curve = []
        
        for r in range(1, max_radius):
            mask = (radius >= r-0.5) & (radius < r+0.5)
            
            if mask.sum() == 0:
                continue
            
            # Correlation in this ring
            numerator = (fft_pred[mask] * torch.conj(fft_target[mask])).real.sum()
            denom1 = (torch.abs(fft_pred[mask])**2).sum()
            denom2 = (torch.abs(fft_target[mask])**2).sum()
            
            if denom1 * denom2 > 0:
                frc = numerator / torch.sqrt(denom1 * denom2)
                frc_curve.append(frc.item())
            else:
                frc_curve.append(0.0)
        
        # Find cutoff resolution
        cutoff_idx = next((i for i, v in enumerate(frc_curve) if v < threshold), 
                         len(frc_curve))
        
        return {
            'frc_curve': frc_curve,
            'cutoff_resolution': cutoff_idx,
            'mean_frc': np.mean(frc_curve)
        }
    
    def calculate_edge_preservation(self, pred: torch.Tensor, 
                                   target: torch.Tensor) -> float:
        """Calculate edge preservation index.
        
        Measures how well edges are preserved.
        """
        # Sobel edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                              dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
        sobel_y = sobel_x.transpose(-2, -1)
        
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)
        
        # Compute edges
        pred_gx = F.conv2d(pred, sobel_x, padding=1)
        pred_gy = F.conv2d(pred, sobel_y, padding=1)
        pred_edges = torch.sqrt(pred_gx**2 + pred_gy**2)
        
        target_gx = F.conv2d(target, sobel_x, padding=1)
        target_gy = F.conv2d(target, sobel_y, padding=1)
        target_edges = torch.sqrt(target_gx**2 + target_gy**2)
        
        # Correlation of edge maps
        correlation = F.cosine_similarity(
            pred_edges.flatten(1),
            target_edges.flatten(1),
            dim=1
        )
        
        return correlation.mean().item()
    
    def calculate_all_metrics(self, pred: torch.Tensor, 
                             target: torch.Tensor) -> Dict[str, float]:
        """Calculate all metrics at once.
        
        Args:
            pred: Predicted images [B, C, H, W]
            target: Ground truth images [B, C, H, W]
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {
            'psnr': self.calculate_psnr(pred, target),
            'ssim': self.calculate_ssim(pred, target),
            'lpips': self.calculate_lpips(pred, target),
            'edge_preservation': self.calculate_edge_preservation(pred, target)
        }
        
        # FRC on first image
        frc_results = self.calculate_frc(pred[0], target[0])
        metrics['frc_cutoff'] = frc_results['cutoff_resolution']
        metrics['mean_frc'] = frc_results['mean_frc']
        
        return metrics


# ============================================================================
# Ablation Study Framework
# ============================================================================

@dataclass
class AblationConfig:
    """Configuration for ablation study."""
    name: str
    use_physics: bool
    lambda_diffusion: float
    lambda_physics: float
    lambda_cycle: float
    modality_conditioning: bool
    num_diffusion_steps: int
    description: str


class AblationRunner:
    """Automated ablation study runner."""
    
    def __init__(
        self,
        base_model_class,
        base_config: Dict,
        output_dir: Path,
        device: str = 'cuda'
    ):
        self.base_model_class = base_model_class
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        self.metrics_calc = MetricsCalculator(device)
        
    def create_ablation_configs(self) -> List[AblationConfig]:
        """Create all ablation configurations.
        
        Returns:
            List of ablation configurations to test
        """
        configs = [
            # 1. Full model (baseline)
            AblationConfig(
                name='full_model',
                use_physics=True,
                lambda_diffusion=1.0,
                lambda_physics=0.5,
                lambda_cycle=0.5,
                modality_conditioning=True,
                num_diffusion_steps=1000,
                description='Full model with all components'
            ),
            
            # 2. No physics constraint
            AblationConfig(
                name='no_physics',
                use_physics=False,
                lambda_diffusion=1.0,
                lambda_physics=0.0,
                lambda_cycle=0.0,
                modality_conditioning=True,
                num_diffusion_steps=1000,
                description='Remove physics constraint module'
            ),
            
            # 3. No modality conditioning
            AblationConfig(
                name='no_modality',
                use_physics=True,
                lambda_diffusion=1.0,
                lambda_physics=0.5,
                lambda_cycle=0.5,
                modality_conditioning=False,
                num_diffusion_steps=1000,
                description='Single model for all modalities'
            ),
            
            # 4. Only diffusion loss
            AblationConfig(
                name='only_diffusion',
                use_physics=True,
                lambda_diffusion=1.0,
                lambda_physics=0.0,
                lambda_cycle=0.0,
                modality_conditioning=True,
                num_diffusion_steps=1000,
                description='Only denoising diffusion loss'
            ),
            
            # 5. Reduced diffusion steps
            AblationConfig(
                name='fast_sampling',
                use_physics=True,
                lambda_diffusion=1.0,
                lambda_physics=0.5,
                lambda_cycle=0.5,
                modality_conditioning=True,
                num_diffusion_steps=100,
                description='Fast sampling with 100 steps'
            ),
            
            # 6. Different loss weights
            AblationConfig(
                name='high_physics_weight',
                use_physics=True,
                lambda_diffusion=1.0,
                lambda_physics=1.0,
                lambda_cycle=1.0,
                modality_conditioning=True,
                num_diffusion_steps=1000,
                description='Higher weight on physics losses'
            ),
        ]
        
        return configs
    
    def run_ablation(
        self,
        config: AblationConfig,
        train_loader,
        val_loader,
        test_loader,
        num_epochs: int = 50
    ) -> Dict[str, any]:
        """Run single ablation experiment.
        
        Args:
            config: Ablation configuration
            train_loader: Training dataloader
            val_loader: Validation dataloader
            test_loader: Test dataloader
            num_epochs: Number of training epochs
            
        Returns:
            Results dictionary
        """
        print(f"\n{'='*80}")
        print(f"Running ablation: {config.name}")
        print(f"Description: {config.description}")
        print(f"{'='*80}\n")
        
        # Create model with ablation settings
        model_config = self.base_config.copy()
        model_config['use_physics'] = config.use_physics
        model_config['timesteps'] = config.num_diffusion_steps
        
        model = self.base_model_class(**model_config).to(self.device)
        
        # Train model (simplified - full training loop in separate script)
        print(f"Training for {num_epochs} epochs...")
        # ... training code here ...
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_metrics = self.evaluate_model(model, test_loader)
        
        # Save results
        results = {
            'config': asdict(config),
            'test_metrics': test_metrics,
            'model_params': sum(p.numel() for p in model.parameters())
        }
        
        # Save to file
        results_file = self.output_dir / f'{config.name}_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    @torch.no_grad()
    def evaluate_model(self, model, test_loader) -> Dict[str, float]:
        """Evaluate model on test set.
        
        Args:
            model: Trained model
            test_loader: Test dataloader
            
        Returns:
            Dictionary of averaged metrics
        """
        model.eval()
        
        all_metrics = []
        
        for batch in tqdm(test_loader, desc='Testing'):
            clean = batch['clean_images'].to(self.device)
            noisy = batch['noisy_images'].to(self.device)
            modality_ids = batch['modality_ids'].to(self.device)
            
            # Generate prediction
            pred = model.sample(noisy, modality_ids, num_inference_steps=50)
            
            # Calculate metrics
            metrics = self.metrics_calc.calculate_all_metrics(pred, clean)
            all_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        
        # Add standard deviations
        for key in all_metrics[0].keys():
            avg_metrics[f'{key}_std'] = np.std([m[key] for m in all_metrics])
        
        return avg_metrics
    
    def run_all_ablations(
        self,
        train_loader,
        val_loader,
        test_loader,
        num_epochs: int = 50
    ) -> Dict[str, Dict]:
        """Run all ablation studies.
        
        Returns:
            Dictionary mapping ablation names to results
        """
        configs = self.create_ablation_configs()
        
        all_results = {}
        
        for config in configs:
            results = self.run_ablation(
                config,
                train_loader,
                val_loader,
                test_loader,
                num_epochs
            )
            all_results[config.name] = results
        
        # Save combined results
        combined_file = self.output_dir / 'all_ablations.json'
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Generate comparison tables
        self.generate_comparison_table(all_results)
        
        return all_results
    
    def generate_comparison_table(self, results: Dict[str, Dict]):
        """Generate LaTeX comparison table.
        
        Args:
            results: Dictionary of all ablation results
        """
        print("\nGenerating comparison table...")
        
        # Extract metrics
        metrics_to_compare = ['psnr', 'ssim', 'lpips', 'edge_preservation']
        
        # Create LaTeX table
        latex = ["\\begin{table}[h]"]
        latex.append("\\centering")
        latex.append("\\begin{tabular}{l" + "c" * len(metrics_to_compare) + "}")
        latex.append("\\toprule")
        
        # Header
        header = "Ablation & " + " & ".join([m.upper() for m in metrics_to_compare]) + " \\\\"
        latex.append(header)
        latex.append("\\midrule")
        
        # Find best values for each metric
        best_values = {
            metric: max(results[name]['test_metrics'][metric] 
                       for name in results)
            for metric in metrics_to_compare
        }
        
        # Rows
        for name, result in results.items():
            row = [name.replace('_', ' ').title()]
            
            for metric in metrics_to_compare:
                value = result['test_metrics'][metric]
                std = result['test_metrics'].get(f'{metric}_std', 0)
                
                # Bold if best
                if abs(value - best_values[metric]) < 1e-6:
                    row.append(f"\\textbf{{{value:.3f}}} $\\pm$ {std:.3f}")
                else:
                    row.append(f"{value:.3f} $\\pm$ {std:.3f}")
            
            latex.append(" & ".join(row) + " \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\caption{Ablation study results}")
        latex.append("\\label{tab:ablation}")
        latex.append("\\end{table}")
        
        # Save to file
        table_file = self.output_dir / 'ablation_table.tex'
        with open(table_file, 'w') as f:
            f.write('\n'.join(latex))
        
        print(f"✓ Table saved to {table_file}")


# Example usage
if __name__ == '__main__':
    print("Evaluation and ablation framework loaded!")
    
    # Test metrics
    print("\nTesting metrics...")
    calc = MetricsCalculator(device='cpu')
    
    pred = torch.rand(2, 1, 256, 256)
    target = torch.rand(2, 1, 256, 256)
    
    metrics = calc.calculate_all_metrics(pred, target)
    print("Metrics calculated:")
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}")
