"""
Physics-informed loss functions for AFM reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np

class AFMReconstructionLoss(nn.Module):
    """Multi-term loss for joint tip and surface reconstruction"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Loss weights from config
        self.tip_weight = config.training.tip_weight
        self.surface_weight = config.training.surface_weight
        self.consistency_weight = config.training.consistency_weight
        self.smoothness_weight = config.training.smoothness_weight
        self.uncertainty_weight = config.training.uncertainty_weight
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                ground_truth: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-term physics-informed loss
        
        Args:
            predictions: Dict with 'tip', 'surface', 'simulated_image', 'tip_uncertainty', etc.
            ground_truth: Dict with 'tip', 'surface', 'image'
        
        Returns:
            total_loss: Scalar loss for backprop
            loss_dict: Dictionary of individual loss terms
        """
        loss_dict = {}
        
        # 1. Reconstruction losses
        if 'tip' in ground_truth and ground_truth['tip'] is not None:
            loss_dict['tip_mse'] = F.mse_loss(predictions['tip'], ground_truth['tip'])
            loss_dict['tip_mae'] = F.l1_loss(predictions['tip'], ground_truth['tip'])
        else:
            loss_dict['tip_mse'] = torch.tensor(0.0, device=predictions['tip'].device)
            loss_dict['tip_mae'] = torch.tensor(0.0, device=predictions['tip'].device)
        
        if 'surface' in ground_truth and ground_truth['surface'] is not None:
            loss_dict['surface_mse'] = F.mse_loss(predictions['surface'], ground_truth['surface'])
            loss_dict['surface_mae'] = F.l1_loss(predictions['surface'], ground_truth['surface'])
        else:
            loss_dict['surface_mse'] = torch.tensor(0.0, device=predictions['surface'].device)
            loss_dict['surface_mae'] = torch.tensor(0.0, device=predictions['surface'].device)
        
        # 2. Imaging consistency loss (most important for blind reconstruction)
        loss_dict['consistency'] = F.mse_loss(
            predictions['simulated_image'], 
            ground_truth['image']
        )
        
        # Add perceptual loss component
        loss_dict['perceptual'] = self._perceptual_loss(
            predictions['simulated_image'],
            ground_truth['image']
        )
        
        # 3. Smoothness priors
        loss_dict['tip_smoothness'] = self._smoothness_loss_3d(predictions['tip'])
        loss_dict['surface_smoothness'] = self._smoothness_loss_2d(predictions['surface'])
        
        # 4. Physical plausibility constraints
        loss_dict['tip_positivity'] = self._positivity_constraint(predictions['tip'])
        loss_dict['tip_shape'] = self._tip_shape_prior(predictions['tip'])
        
        # 5. Uncertainty calibration (if available)
        if predictions.get('tip_uncertainty') is not None and 'tip' in ground_truth:
            loss_dict['uncertainty'] = self._uncertainty_calibration_loss(
                predictions, ground_truth
            )
        else:
            loss_dict['uncertainty'] = torch.tensor(0.0, device=predictions['tip'].device)
        
        # Compute weighted total loss
        total_loss = (
            self.tip_weight * (loss_dict['tip_mse'] + 0.5 * loss_dict['tip_mae']) +
            self.surface_weight * (loss_dict['surface_mse'] + 0.5 * loss_dict['surface_mae']) +
            self.consistency_weight * (loss_dict['consistency'] + 0.3 * loss_dict['perceptual']) +
            self.smoothness_weight * (loss_dict['tip_smoothness'] + loss_dict['surface_smoothness']) +
            0.1 * loss_dict['tip_positivity'] +
            0.1 * loss_dict['tip_shape'] +
            self.uncertainty_weight * loss_dict['uncertainty']
        )
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def _smoothness_loss_2d(self, tensor: torch.Tensor) -> torch.Tensor:
        """Total variation smoothness for 2D surfaces"""
        diff_x = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
        diff_y = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
        
        smoothness = torch.mean(diff_x ** 2) + torch.mean(diff_y ** 2)
        return smoothness
    
    def _smoothness_loss_3d(self, tensor: torch.Tensor) -> torch.Tensor:
        """Total variation smoothness for 3D tips"""
        # tensor shape: [B, 1, H, W, D]
        diff_x = tensor[:, :, 1:, :, :] - tensor[:, :, :-1, :, :]
        diff_y = tensor[:, :, :, 1:, :] - tensor[:, :, :, :-1, :]
        diff_z = tensor[:, :, :, :, 1:] - tensor[:, :, :, :, :-1]
        
        smoothness = (
            torch.mean(diff_x ** 2) + 
            torch.mean(diff_y ** 2) + 
            torch.mean(diff_z ** 2)
        )
        return smoothness
    
    def _perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Perceptual loss using gradient similarity"""
        # Compute gradients
        pred_grad_x = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        pred_grad_y = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        
        target_grad_x = target[:, :, 1:, :] - target[:, :, :-1, :]
        target_grad_y = target[:, :, :, 1:] - target[:, :, :, :-1]
        
        # MSE on gradients
        loss = (
            F.mse_loss(pred_grad_x, target_grad_x) +
            F.mse_loss(pred_grad_y, target_grad_y)
        )
        return loss
    
    def _positivity_constraint(self, tip: torch.Tensor) -> torch.Tensor:
        """Ensure tip has physically meaningful values"""
        # Penalize negative values (tips should be positive heights)
        negative_penalty = torch.mean(F.relu(-tip))
        return negative_penalty
    
    def _tip_shape_prior(self, tip: torch.Tensor) -> torch.Tensor:
        """Encourage physically plausible tip shapes"""
        # Tips should generally taper towards the apex
        # Check if tip tapers along z-axis (depth)
        B, C, H, W, D = tip.shape
        
        # Compute mean radius at each depth slice
        radii = []
        for d in range(D):
            slice_d = tip[:, :, :, :, d]
            # Compute center of mass
            threshold = torch.mean(slice_d, dim=(1, 2, 3), keepdim=True)
            mask = (slice_d > threshold).float()
            radius = torch.sum(mask, dim=(1, 2, 3)) / (H * W)
            radii.append(radius)
        
        radii = torch.stack(radii, dim=1)  # [B, D]
        
        # Penalize if radii don't decrease towards apex
        radius_diffs = radii[:, 1:] - radii[:, :-1]
        increasing_penalty = torch.mean(F.relu(radius_diffs))
        
        return increasing_penalty
    
    def _uncertainty_calibration_loss(self, predictions: Dict, 
                                     ground_truth: Dict) -> torch.Tensor:
        """Calibrate uncertainty estimates with actual errors"""
        # Compute reconstruction errors
        tip_error = torch.abs(predictions['tip'] - ground_truth['tip'])
        surface_error = torch.abs(predictions['surface'] - ground_truth['surface'])
        
        # Uncertainty should correlate with error
        # Using NLL-like formulation
        if predictions['tip_uncertainty'] is not None:
            tip_nll = torch.mean(
                (tip_error ** 2) / (2 * predictions['tip_uncertainty'] ** 2 + 1e-6) +
                torch.log(predictions['tip_uncertainty'] + 1e-6)
            )
        else:
            tip_nll = torch.tensor(0.0, device=tip_error.device)
        
        if predictions.get('surface_uncertainty') is not None:
            surface_nll = torch.mean(
                (surface_error ** 2) / (2 * predictions['surface_uncertainty'] ** 2 + 1e-6) +
                torch.log(predictions['surface_uncertainty'] + 1e-6)
            )
        else:
            surface_nll = torch.tensor(0.0, device=surface_error.device)
        
        return tip_nll + surface_nll


class SSIMLoss(nn.Module):
    """Structural Similarity Index loss"""
    
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)
    
    def _gaussian(self, window_size: int, sigma: float):
        """Create Gaussian window"""
        gauss = torch.Tensor([
            np.exp(-(x - window_size//2)**2 / (2.0 * sigma**2)) 
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def _create_window(self, window_size: int, channel: int):
        """Create 2D Gaussian window"""
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss"""
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(dim=(1, 2, 3))


def test_losses():
    """Test loss functions"""
    from config import get_config
    
    config = get_config()
    loss_fn = AFMReconstructionLoss(config)
    
    # Create dummy predictions and ground truth
    B = 2
    predictions = {
        'tip': torch.randn(B, 1, 32, 32, 32),
        'surface': torch.randn(B, 1, 128, 128),
        'simulated_image': torch.randn(B, 1, 128, 128),
        'tip_uncertainty': torch.abs(torch.randn(B, 1, 32, 32, 32)) * 0.1,
        'surface_uncertainty': None
    }
    
    ground_truth = {
        'tip': torch.randn(B, 1, 32, 32, 32),
        'surface': torch.randn(B, 1, 128, 128),
        'image': torch.randn(B, 1, 128, 128)
    }
    
    # Compute loss
    total_loss, loss_dict = loss_fn(predictions, ground_truth)
    
    print("Loss Function Test:")
    print(f"  Total loss: {total_loss.item():.6f}")
    print("\n  Individual losses:")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"    {key}: {value.item():.6f}")
        else:
            print(f"    {key}: {value:.6f}")
    
    # Test SSIM
    ssim_loss = SSIMLoss()
    ssim_val = ssim_loss(predictions['surface'], ground_truth['surface'])
    print(f"\n  SSIM loss: {ssim_val.item():.6f}")
    
    print("\n✓ Loss functions working correctly")

if __name__ == '__main__':
    test_losses()
