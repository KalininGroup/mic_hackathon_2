"""
physics_diffusion/evaluation/expert_eval.py
physics_diffusion/experiments/baseline_comparison.py

Expert evaluation interface and baseline method implementations.
"""

# ============================================================================
# EXPERT_EVAL.PY
# ============================================================================

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class ExpertEvaluationInterface:
    """GUI for expert evaluation of microscopy images."""
    
    def __init__(
        self,
        image_paths: List[Path],
        output_file: Path,
        criteria: List[str] = None
    ):
        """
        Args:
            image_paths: List of image file paths to evaluate
            output_file: Where to save evaluation results
            criteria: List of evaluation criteria
        """
        self.image_paths = image_paths
        self.output_file = Path(output_file)
        self.current_idx = 0
        
        if criteria is None:
            self.criteria = [
                'Artifact Removal',
                'Detail Preservation',
                'Physical Plausibility',
                'Overall Quality'
            ]
        else:
            self.criteria = criteria
        
        # Results storage
        self.results = []
        
        # Load existing results if available
        if self.output_file.exists():
            with open(self.output_file, 'r') as f:
                self.results = json.load(f)
        
        # Create GUI
        self.root = tk.Tk()
        self.root.title("Expert Microscopy Evaluation")
        self.root.geometry("1200x800")
        
        self._create_widgets()
        self._load_image()
    
    def _create_widgets(self):
        """Create GUI widgets."""
        # Top frame - Image display
        self.image_frame = ttk.Frame(self.root)
        self.image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack()
        
        # Middle frame - Criteria ratings
        self.rating_frame = ttk.Frame(self.root)
        self.rating_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        ttk.Label(self.rating_frame, text="Rate each criterion (1-5):", 
                 font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=6, pady=10)
        
        self.rating_vars = {}
        for i, criterion in enumerate(self.criteria, start=1):
            ttk.Label(self.rating_frame, text=criterion).grid(row=i, column=0, sticky=tk.W, padx=5)
            
            var = tk.IntVar(value=3)
            self.rating_vars[criterion] = var
            
            for j in range(1, 6):
                ttk.Radiobutton(
                    self.rating_frame, 
                    text=str(j), 
                    variable=var, 
                    value=j
                ).grid(row=i, column=j, padx=5)
        
        # Comments box
        ttk.Label(self.rating_frame, text="Comments:").grid(
            row=len(self.criteria)+1, column=0, sticky=tk.W, padx=5, pady=(10, 0)
        )
        
        self.comments_text = tk.Text(self.rating_frame, height=4, width=60)
        self.comments_text.grid(
            row=len(self.criteria)+2, column=0, columnspan=6, 
            padx=5, pady=5, sticky=tk.W
        )
        
        # Bottom frame - Navigation
        self.nav_frame = ttk.Frame(self.root)
        self.nav_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        self.progress_label = ttk.Label(
            self.nav_frame, 
            text=f"Image 1 of {len(self.image_paths)}"
        )
        self.progress_label.pack(side=tk.LEFT)
        
        ttk.Button(self.nav_frame, text="Previous", 
                  command=self._prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.nav_frame, text="Next", 
                  command=self._next_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.nav_frame, text="Save & Exit", 
                  command=self._save_and_exit).pack(side=tk.RIGHT, padx=5)
    
    def _load_image(self):
        """Load and display current image."""
        if self.current_idx >= len(self.image_paths):
            return
        
        img_path = self.image_paths[self.current_idx]
        
        # Load image
        img = Image.open(img_path).convert('L')
        
        # Resize to fit display
        img.thumbnail((800, 600), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.photo)
        
        # Update progress
        self.progress_label.config(
            text=f"Image {self.current_idx + 1} of {len(self.image_paths)}"
        )
        
        # Load existing rating if available
        existing = next(
            (r for r in self.results if r['image_path'] == str(img_path)),
            None
        )
        
        if existing:
            for criterion, var in self.rating_vars.items():
                var.set(existing['ratings'].get(criterion, 3))
            self.comments_text.delete('1.0', tk.END)
            self.comments_text.insert('1.0', existing.get('comments', ''))
    
    def _save_current(self):
        """Save current ratings."""
        if self.current_idx >= len(self.image_paths):
            return
        
        img_path = self.image_paths[self.current_idx]
        
        ratings = {
            criterion: var.get()
            for criterion, var in self.rating_vars.items()
        }
        
        comments = self.comments_text.get('1.0', tk.END).strip()
        
        result = {
            'image_path': str(img_path),
            'ratings': ratings,
            'comments': comments,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update or append
        existing_idx = next(
            (i for i, r in enumerate(self.results) 
             if r['image_path'] == str(img_path)),
            None
        )
        
        if existing_idx is not None:
            self.results[existing_idx] = result
        else:
            self.results.append(result)
    
    def _next_image(self):
        """Go to next image."""
        self._save_current()
        
        if self.current_idx < len(self.image_paths) - 1:
            self.current_idx += 1
            self._load_image()
    
    def _prev_image(self):
        """Go to previous image."""
        self._save_current()
        
        if self.current_idx > 0:
            self.current_idx -= 1
            self._load_image()
    
    def _save_and_exit(self):
        """Save results and close."""
        self._save_current()
        
        with open(self.output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.root.destroy()
    
    def run(self):
        """Run the evaluation interface."""
        self.root.mainloop()


def analyze_expert_evaluations(results_file: Path) -> Dict:
    """Analyze expert evaluation results.
    
    Args:
        results_file: JSON file with evaluation results
        
    Returns:
        Dictionary with statistics
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Aggregate statistics
    criteria_scores = {}
    
    for result in results:
        for criterion, score in result['ratings'].items():
            if criterion not in criteria_scores:
                criteria_scores[criterion] = []
            criteria_scores[criterion].append(score)
    
    stats = {
        'num_images': len(results),
        'criteria_stats': {}
    }
    
    for criterion, scores in criteria_scores.items():
        stats['criteria_stats'][criterion] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'median': np.median(scores),
            'min': np.min(scores),
            'max': np.max(scores)
        }
    
    return stats


# ============================================================================
# BASELINE_COMPARISON.PY
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import cv2


class BaselineMethod:
    """Base class for baseline methods."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
    
    def denoise(self, noisy_image: torch.Tensor) -> torch.Tensor:
        """Denoise image.
        
        Args:
            noisy_image: [B, C, H, W] noisy input
            
        Returns:
            Denoised image [B, C, H, W]
        """
        raise NotImplementedError


class WienerFilter(BaselineMethod):
    """Wiener deconvolution baseline."""
    
    def __init__(self, noise_variance: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.noise_variance = noise_variance
    
    def denoise(self, noisy_image: torch.Tensor) -> torch.Tensor:
        """Apply Wiener filter."""
        # Estimate PSF (simple Gaussian blur)
        kernel_size = 5
        sigma = 1.0
        
        # Create Gaussian kernel
        ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size).to(self.device)
        
        B, C, H, W = noisy_image.shape
        denoised = torch.zeros_like(noisy_image)
        
        for b in range(B):
            for c in range(C):
                img = noisy_image[b:b+1, c:c+1]
                
                # FFT
                fft_img = torch.fft.rfft2(img)
                fft_kernel = torch.fft.rfft2(kernel, s=(H, W))
                
                # Wiener filter
                H_conj = torch.conj(fft_kernel)
                H_abs_sq = torch.abs(fft_kernel) ** 2
                
                wiener = H_conj / (H_abs_sq + self.noise_variance)
                
                fft_denoised = fft_img * wiener
                denoised[b, c] = torch.fft.irfft2(fft_denoised, s=(H, W))
        
        return denoised


class BM3D(BaselineMethod):
    """BM3D denoising baseline (uses OpenCV)."""
    
    def __init__(self, sigma: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
    
    def denoise(self, noisy_image: torch.Tensor) -> torch.Tensor:
        """Apply BM3D denoising."""
        B, C, H, W = noisy_image.shape
        denoised = torch.zeros_like(noisy_image)
        
        for b in range(B):
            for c in range(C):
                # Convert to numpy
                img_np = noisy_image[b, c].cpu().numpy()
                
                # Normalize to [0, 1]
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                
                # Convert to uint8
                img_uint8 = (img_np * 255).astype(np.uint8)
                
                # Apply BM3D (using fastNlMeansDenoising as proxy)
                denoised_np = cv2.fastNlMeansDenoising(
                    img_uint8, 
                    None, 
                    h=self.sigma
                )
                
                # Convert back
                denoised_np = denoised_np.astype(np.float32) / 255.0
                denoised_np = denoised_np * (img_np.max() - img_np.min()) + img_np.min()
                
                denoised[b, c] = torch.from_numpy(denoised_np).to(self.device)
        
        return denoised


class Noise2Noise(BaselineMethod):
    """Noise2Noise baseline (simplified training)."""
    
    def __init__(self, model_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        
        # Simple U-Net for Noise2Noise
        self.model = self._build_unet().to(self.device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        
        self.model.eval()
    
    def _build_unet(self) -> nn.Module:
        """Build simple U-Net for Noise2Noise."""
        from ..models.unet_arch import UNetWithAttention
        
        return UNetWithAttention(
            in_channels=1,
            model_channels=64,
            out_channels=1,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            channel_mult=(1, 2, 4, 8),
            num_heads=4,
            use_modality_conditioning=False
        )
    
    def denoise(self, noisy_image: torch.Tensor) -> torch.Tensor:
        """Apply Noise2Noise denoising."""
        with torch.no_grad():
            # Noise2Noise trains on pairs of noisy images
            # At test time, just run single forward pass
            denoised = self.model(
                noisy_image, 
                torch.zeros(noisy_image.shape[0], dtype=torch.long, device=self.device),
                None
            )
        
        return denoised


class DnCNN(BaselineMethod):
    """DnCNN baseline."""
    
    def __init__(self, depth: int = 17, **kwargs):
        super().__init__(**kwargs)
        
        # Build DnCNN
        layers = []
        layers.append(nn.Conv2d(1, 64, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(64, 64, 3, padding=1))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(64, 1, 3, padding=1))
        
        self.model = nn.Sequential(*layers).to(self.device)
        self.model.eval()
    
    def denoise(self, noisy_image: torch.Tensor) -> torch.Tensor:
        """Apply DnCNN denoising."""
        with torch.no_grad():
            # DnCNN predicts noise
            noise = self.model(noisy_image)
            denoised = noisy_image - noise
        
        return denoised


class BaselineComparator:
    """Compare all baseline methods."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
        # Initialize all baselines
        self.baselines = {
            'Wiener': WienerFilter(device=device),
            'BM3D': BM3D(device=device),
            'DnCNN': DnCNN(device=device),
        }
    
    def run_comparison(
        self,
        noisy_images: torch.Tensor,
        clean_images: torch.Tensor,
        metrics_calculator
    ) -> Dict[str, Dict[str, float]]:
        """Run all baselines and compute metrics.
        
        Args:
            noisy_images: [B, C, H, W] noisy inputs
            clean_images: [B, C, H, W] clean targets
            metrics_calculator: MetricsCalculator instance
            
        Returns:
            Dictionary of results per method
        """
        results = {}
        
        for name, baseline in self.baselines.items():
            print(f"Running {name}...")
            
            try:
                # Denoise
                denoised = baseline.denoise(noisy_images)
                
                # Calculate metrics
                metrics = metrics_calculator.calculate_all_metrics(
                    denoised, clean_images
                )
                
                results[name] = metrics
                
            except Exception as e:
                print(f"  Error with {name}: {e}")
                results[name] = {
                    'psnr': 0.0,
                    'ssim': 0.0,
                    'error': str(e)
                }
        
        return results


# Testing
if __name__ == '__main__':
    print("Testing baseline methods...\n")
    
    # Test Wiener filter
    print("1. Wiener Filter")
    wiener = WienerFilter(device='cpu')
    noisy = torch.randn(2, 1, 128, 128)
    denoised = wiener.denoise(noisy)
    print(f"   Input: {noisy.shape}")
    print(f"   Output: {denoised.shape}")
    
    # Test DnCNN
    print("\n2. DnCNN")
    dncnn = DnCNN(device='cpu')
    denoised = dncnn.denoise(noisy)
    print(f"   Input: {noisy.shape}")
    print(f"   Output: {denoised.shape}")
    
    print("\n✓ All tests passed!")
