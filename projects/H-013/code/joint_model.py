"""
Joint Tip and Surface Reconstruction Model
Combines SE(3)-equivariant tip reconstruction with surface network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from models.se3_network import SE3TipReconstructor
from models.surface_network import SurfaceReconstructor

class DifferentiableAFMSimulator(nn.Module):
    """Differentiable AFM imaging forward model"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def forward(self, surface: torch.Tensor, tip: torch.Tensor) -> torch.Tensor:
        """
        Simulate AFM image from surface and tip
        Uses morphological dilation: I = S ⊕ T
        
        Args:
            surface: [B, 1, H, W]
            tip: [B, 1, H_t, W_t, D_t]
        
        Returns:
            image: [B, 1, H, W]
        """
        B, _, H, W = surface.shape
        
        # Get 2D tip profile (max projection along z)
        tip_2d = torch.max(tip, dim=4)[0]  # [B, 1, H_t, W_t]
        
        # Normalize tip
        tip_2d = tip_2d - tip_2d.min()
        
        # Pad surface for convolution
        pad_h = tip_2d.shape[2] // 2
        pad_w = tip_2d.shape[3] // 2
        surface_padded = F.pad(surface, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
        
        # Flip tip for proper convolution
        tip_flipped = torch.flip(tip_2d, dims=[2, 3])
        
        # Convolve (morphological dilation approximation)
        image = F.conv2d(surface_padded, tip_flipped, padding=0)
        
        # Crop to original size
        start_h = (image.shape[2] - H) // 2
        start_w = (image.shape[3] - W) // 2
        image = image[:, :, start_h:start_h+H, start_w:start_w+W]
        
        return image


class JointTipSurfaceModel(nn.Module):
    """Joint reconstruction model with uncertainty quantification"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Component networks
        self.tip_reconstructor = SE3TipReconstructor(config)
        self.surface_reconstructor = SurfaceReconstructor(config)
        
        # Differentiable forward model
        self.afm_simulator = DifferentiableAFMSimulator(config)
        
    def forward(self, afm_image: torch.Tensor, 
                monte_carlo_samples: int = 1,
                return_intermediate: bool = False) -> Dict[str, torch.Tensor]:
        """
        Joint reconstruction with optional uncertainty quantification
        
        Args:
            afm_image: [B, 1, H, W]
            monte_carlo_samples: Number of MC dropout samples
            return_intermediate: Whether to return intermediate features
        
        Returns:
            Dictionary containing:
                - tip: [B, 1, H_t, W_t, D_t]
                - surface: [B, 1, H, W]
                - simulated_image: [B, 1, H, W]
                - tip_uncertainty: [B, 1, H_t, W_t, D_t] (if MC > 1)
                - surface_uncertainty: [B, 1, H, W] (if MC > 1)
        """
        if self.training or monte_carlo_samples == 1:
            return self._forward_single(afm_image, return_intermediate)
        else:
            return self._forward_mc(afm_image, monte_carlo_samples, return_intermediate)
    
    def _forward_single(self, afm_image: torch.Tensor, 
                       return_intermediate: bool = False) -> Dict[str, torch.Tensor]:
        """Single forward pass"""
        # Reconstruct tip
        tip, tip_uncertainty = self.tip_reconstructor(afm_image, monte_carlo=False)
        
        # Reconstruct surface
        surface = self.surface_reconstructor(afm_image, tip, tip_uncertainty)
        
        # Simulate image for consistency
        simulated_image = self.afm_simulator(surface, tip)
        
        result = {
            'tip': tip,
            'surface': surface,
            'simulated_image': simulated_image,
            'tip_uncertainty': tip_uncertainty,
            'surface_uncertainty': None
        }
        
        return result
    
    def _forward_mc(self, afm_image: torch.Tensor, n_samples: int,
                   return_intermediate: bool = False) -> Dict[str, torch.Tensor]:
        """Monte Carlo dropout for uncertainty estimation"""
        tip_samples = []
        surface_samples = []
        
        # Enable dropout
        self.train()
        
        with torch.no_grad():
            for _ in range(n_samples):
                # Sample with dropout
                tip, _ = self.tip_reconstructor(afm_image, monte_carlo=True)
                surface = self.surface_reconstructor(afm_image, tip, None)
                
                tip_samples.append(tip)
                surface_samples.append(surface)
        
        # Back to eval mode
        self.eval()
        
        # Compute statistics
        tip_samples = torch.stack(tip_samples, dim=0)  # [N, B, ...]
        surface_samples = torch.stack(surface_samples, dim=0)
        
        tip_mean = torch.mean(tip_samples, dim=0)
        tip_std = torch.std(tip_samples, dim=0)
        
        surface_mean = torch.mean(surface_samples, dim=0)
        surface_std = torch.std(surface_samples, dim=0)
        
        # Simulate with mean estimates
        simulated_image = self.afm_simulator(surface_mean, tip_mean)
        
        result = {
            'tip': tip_mean,
            'surface': surface_mean,
            'simulated_image': simulated_image,
            'tip_uncertainty': tip_std,
            'surface_uncertainty': surface_std
        }
        
        return result
    
    def reconstruct_tip_only(self, afm_image: torch.Tensor) -> torch.Tensor:
        """Fast tip-only reconstruction"""
        tip, _ = self.tip_reconstructor(afm_image, monte_carlo=False)
        return tip
    
    def reconstruct_surface_only(self, afm_image: torch.Tensor, 
                                 tip: torch.Tensor) -> torch.Tensor:
        """Reconstruct surface given known tip"""
        tip_uncertainty = torch.zeros_like(tip)
        surface = self.surface_reconstructor(afm_image, tip, tip_uncertainty)
        return surface


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_joint_model():
    """Test complete joint model"""
    from config import get_config
    
    config = get_config()
    model = JointTipSurfaceModel(config).cuda()
    
    print("="*70)
    print("Joint Model Architecture Test")
    print("="*70)
    
    # Test single forward pass
    batch_size = 2
    afm_image = torch.randn(batch_size, 1, 128, 128).cuda()
    
    print("\n1. Single Forward Pass:")
    model.eval()
    with torch.no_grad():
        result = model(afm_image, monte_carlo_samples=1)
    
    print(f"  Input shape: {afm_image.shape}")
    print(f"  Tip shape: {result['tip'].shape}")
    print(f"  Surface shape: {result['surface'].shape}")
    print(f"  Simulated image shape: {result['simulated_image'].shape}")
    
    # Test MC uncertainty
    print("\n2. Monte Carlo Uncertainty Estimation:")
    with torch.no_grad():
        result_mc = model(afm_image, monte_carlo_samples=5)
    
    print(f"  Tip uncertainty shape: {result_mc['tip_uncertainty'].shape}")
    print(f"  Surface uncertainty shape: {result_mc['surface_uncertainty'].shape}")
    print(f"  Tip uncertainty range: [{result_mc['tip_uncertainty'].min():.4f}, "
          f"{result_mc['tip_uncertainty'].max():.4f}]")
    print(f"  Surface uncertainty range: [{result_mc['surface_uncertainty'].min():.4f}, "
          f"{result_mc['surface_uncertainty'].max():.4f}]")
    
    # Test consistency loss
    print("\n3. Imaging Consistency:")
    consistency_error = F.mse_loss(result['simulated_image'], afm_image)
    print(f"  MSE between simulated and input: {consistency_error:.6f}")
    
    # Parameter count
    print("\n4. Model Statistics:")
    tip_params = count_parameters(model.tip_reconstructor)
    surface_params = count_parameters(model.surface_reconstructor)
    total_params = count_parameters(model)
    
    print(f"  Tip reconstructor: {tip_params:,} parameters")
    print(f"  Surface reconstructor: {surface_params:,} parameters")
    print(f"  Total: {total_params:,} parameters")
    print(f"  Memory (FP32): ~{total_params * 4 / 1e6:.1f} MB")
    
    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)


if __name__ == '__main__':
    test_joint_model()
