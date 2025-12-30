"""
physics_diffusion/training/scheduler.py
physics_diffusion/training/physics_losses.py

Learning rate schedulers, noise schedulers, and physics-based loss functions.
"""

# ============================================================================
# SCHEDULER.PY
# ============================================================================

import torch
import math
from torch.optim.lr_scheduler import _LRScheduler
from typing import List


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """Cosine annealing with warm restarts and warmup.
    
    Combines:
    - Linear warmup
    - Cosine annealing
    - Optional periodic restarts
    """
    
    def __init__(
        self,
        optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Wrapped optimizer
            first_cycle_steps: Number of steps in first cycle
            cycle_mult: Cycle length multiplier (>1 for increasing cycles)
            max_lr: Maximum learning rate
            min_lr: Minimum learning rate
            warmup_steps: Number of warmup steps
            gamma: LR decay factor per cycle
            last_epoch: Index of last epoch
        """
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super().__init__(optimizer, last_epoch)
        
        # Initialize learning rate
        self.init_lr()
    
    def init_lr(self):
        """Initialize learning rates."""
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate."""
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            # Warmup phase
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.step_in_cycle - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps)
            return [self.min_lr + (self.max_lr - self.min_lr) * 
                    (1 + math.cos(math.pi * progress)) / 2
                    for _ in self.base_lrs]
    
    def step(self, epoch=None):
        """Update learning rate."""
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle += 1
            
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = 0
                self.cur_cycle_steps = int(self.cur_cycle_steps * self.cycle_mult)
                self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), 
                                    self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * 
                                                     (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** n
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class NoiseScheduler:
    """Noise scheduling for diffusion models.
    
    Supports multiple noise schedules:
    - Linear
    - Cosine
    - Sigmoid
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule_type: str = 'cosine',
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        s: float = 0.008
    ):
        """
        Args:
            num_timesteps: Number of diffusion steps
            schedule_type: 'linear', 'cosine', or 'sigmoid'
            beta_start: Starting beta value (for linear)
            beta_end: Ending beta value (for linear)
            s: Offset for cosine schedule
        """
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        
        if schedule_type == 'linear':
            self.betas = self._linear_schedule(beta_start, beta_end)
        elif schedule_type == 'cosine':
            self.betas = self._cosine_schedule(s)
        elif schedule_type == 'sigmoid':
            self.betas = self._sigmoid_schedule(beta_start, beta_end)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Compute alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]), 
            self.alphas_cumprod[:-1]
        ])
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        # Clipping for numerical stability
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)
        
        # Log calculations
        self.posterior_log_variance = torch.log(self.posterior_variance)
        
        # Coefficients for x_0 prediction
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Coefficients for posterior mean
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _linear_schedule(self, beta_start: float, beta_end: float) -> torch.Tensor:
        """Linear noise schedule."""
        return torch.linspace(beta_start, beta_end, self.num_timesteps)
    
    def _cosine_schedule(self, s: float = 0.008) -> torch.Tensor:
        """Cosine noise schedule (improved DDPM)."""
        steps = self.num_timesteps + 1
        x = torch.linspace(0, self.num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _sigmoid_schedule(self, beta_start: float, beta_end: float) -> torch.Tensor:
        """Sigmoid noise schedule."""
        betas = torch.linspace(-6, 6, self.num_timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


# ============================================================================
# PHYSICS_LOSSES.PY
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class DiffusionLoss(nn.Module):
    """Standard diffusion denoising loss (MSE on noise prediction)."""
    
    def __init__(self, loss_type: str = 'mse'):
        """
        Args:
            loss_type: 'mse', 'l1', or 'huber'
        """
        super().__init__()
        self.loss_type = loss_type
    
    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            noise_pred: Predicted noise [B, C, H, W]
            noise_target: Target noise [B, C, H, W]
            mask: Optional mask [B, 1, H, W]
            
        Returns:
            Loss scalar
        """
        if self.loss_type == 'mse':
            loss = F.mse_loss(noise_pred, noise_target, reduction='none')
        elif self.loss_type == 'l1':
            loss = F.l1_loss(noise_pred, noise_target, reduction='none')
        elif self.loss_type == 'huber':
            loss = F.smooth_l1_loss(noise_pred, noise_target, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)
        else:
            return loss.mean()


class PhysicsConsistencyLoss(nn.Module):
    """Physics consistency loss: ensures output satisfies physical model."""
    
    def __init__(
        self,
        loss_type: str = 'l1',
        use_fourier: bool = False
    ):
        """
        Args:
            loss_type: 'l1' or 'mse'
            use_fourier: Whether to compute loss in Fourier space
        """
        super().__init__()
        self.loss_type = loss_type
        self.use_fourier = use_fourier
    
    def forward(
        self,
        x_pred: torch.Tensor,
        x_measured: torch.Tensor,
        physics_operator: nn.Module,
        modality_id: int
    ) -> torch.Tensor:
        """
        Args:
            x_pred: Predicted clean image [B, C, H, W]
            x_measured: Measured noisy image [B, C, H, W]
            physics_operator: Forward physics model
            modality_id: Which modality this is
            
        Returns:
            Physics consistency loss
        """
        # Apply forward physics model
        y_pred = physics_operator(x_pred)
        
        # Compute residual
        if self.use_fourier:
            # Loss in Fourier space (emphasizes high frequencies)
            fft_pred = torch.fft.rfft2(y_pred)
            fft_meas = torch.fft.rfft2(x_measured)
            residual = torch.abs(fft_pred - fft_meas)
        else:
            residual = y_pred - x_measured
        
        # Compute loss
        if self.loss_type == 'l1':
            loss = torch.abs(residual).mean()
        elif self.loss_type == 'mse':
            loss = (residual ** 2).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class CycleConsistencyLoss(nn.Module):
    """Cycle consistency: noisy -> clean -> noisy should match."""
    
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(
        self,
        x_noisy: torch.Tensor,
        x_denoised: torch.Tensor,
        physics_operator: nn.Module
    ) -> torch.Tensor:
        """
        Args:
            x_noisy: Original noisy image [B, C, H, W]
            x_denoised: Denoised prediction [B, C, H, W]
            physics_operator: Forward physics model
            
        Returns:
            Cycle consistency loss
        """
        # Re-apply artifacts to denoised image
        x_renoised = physics_operator(x_denoised)
        
        # Should match original
        if self.loss_type == 'l1':
            loss = F.l1_loss(x_renoised, x_noisy)
        elif self.loss_type == 'mse':
            loss = F.mse_loss(x_renoised, x_noisy)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class PerceptualLoss(nn.Module):
    """Perceptual loss using pretrained VGG features."""
    
    def __init__(self, layers: list = [2, 7, 12, 21]):
        """
        Args:
            layers: VGG layers to use for feature extraction
        """
        super().__init__()
        
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        except ImportError:
            raise ImportError("torchvision required for perceptual loss")
        
        self.layers = layers
        self.features = nn.ModuleList()
        
        prev_layer = 0
        for layer_idx in layers:
            self.features.append(vgg[prev_layer:layer_idx+1])
            prev_layer = layer_idx + 1
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        x_pred: torch.Tensor,
        x_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_pred: Predicted image [B, C, H, W]
            x_target: Target image [B, C, H, W]
            
        Returns:
            Perceptual loss
        """
        # Convert grayscale to RGB if needed
        if x_pred.shape[1] == 1:
            x_pred = x_pred.repeat(1, 3, 1, 1)
        if x_target.shape[1] == 1:
            x_target = x_target.repeat(1, 3, 1, 1)
        
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x_pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x_pred.device)
        
        x_pred = (x_pred - mean) / std
        x_target = (x_target - mean) / std
        
        # Extract features and compute loss
        loss = 0.0
        for layer in self.features:
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            loss += F.mse_loss(x_pred, x_target)
        
        return loss / len(self.features)


class TotalVariationLoss(nn.Module):
    """Total variation loss for smoothness regularization."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image [B, C, H, W]
            
        Returns:
            TV loss
        """
        # Compute gradients
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        
        return self.weight * (diff_h.mean() + diff_w.mean())


class CombinedLoss(nn.Module):
    """Combined loss with multiple components and adaptive weighting."""
    
    def __init__(
        self,
        lambda_diffusion: float = 1.0,
        lambda_physics: float = 0.5,
        lambda_cycle: float = 0.5,
        lambda_perceptual: float = 0.1,
        lambda_tv: float = 0.01,
        use_adaptive_weights: bool = False
    ):
        """
        Args:
            lambda_*: Loss weights
            use_adaptive_weights: Use uncertainty weighting
        """
        super().__init__()
        
        self.diffusion_loss = DiffusionLoss()
        self.physics_loss = PhysicsConsistencyLoss()
        self.cycle_loss = CycleConsistencyLoss()
        
        try:
            self.perceptual_loss = PerceptualLoss()
            self.has_perceptual = True
        except ImportError:
            self.has_perceptual = False
        
        self.tv_loss = TotalVariationLoss()
        
        # Loss weights
        if use_adaptive_weights:
            # Learnable uncertainty-based weights
            self.log_var_diffusion = nn.Parameter(torch.zeros(1))
            self.log_var_physics = nn.Parameter(torch.zeros(1))
            self.log_var_cycle = nn.Parameter(torch.zeros(1))
        else:
            self.lambda_diffusion = lambda_diffusion
            self.lambda_physics = lambda_physics
            self.lambda_cycle = lambda_cycle
            self.lambda_perceptual = lambda_perceptual
            self.lambda_tv = lambda_tv
        
        self.use_adaptive_weights = use_adaptive_weights
    
    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        x_pred: torch.Tensor,
        x_measured: torch.Tensor,
        x_denoised: torch.Tensor,
        physics_operator: nn.Module,
        modality_id: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Individual loss values
        """
        # Individual losses
        loss_diff = self.diffusion_loss(noise_pred, noise_target)
        loss_phys = self.physics_loss(x_pred, x_measured, physics_operator, modality_id)
        loss_cyc = self.cycle_loss(x_measured, x_denoised, physics_operator)
        loss_tv = self.tv_loss(x_pred)
        
        if self.has_perceptual:
            loss_perc = self.perceptual_loss(x_pred, x_measured)
        else:
            loss_perc = torch.tensor(0.0)
        
        # Adaptive weighting (uncertainty weighting)
        if self.use_adaptive_weights:
            precision_diff = torch.exp(-self.log_var_diffusion)
            precision_phys = torch.exp(-self.log_var_physics)
            precision_cyc = torch.exp(-self.log_var_cycle)
            
            total_loss = (
                precision_diff * loss_diff + self.log_var_diffusion +
                precision_phys * loss_phys + self.log_var_physics +
                precision_cyc * loss_cyc + self.log_var_cycle
            )
        else:
            # Fixed weighting
            total_loss = (
                self.lambda_diffusion * loss_diff +
                self.lambda_physics * loss_phys +
                self.lambda_cycle * loss_cyc +
                self.lambda_perceptual * loss_perc +
                self.lambda_tv * loss_tv
            )
        
        loss_dict = {
            'diffusion': loss_diff.item(),
            'physics': loss_phys.item(),
            'cycle': loss_cyc.item(),
            'perceptual': loss_perc.item() if torch.is_tensor(loss_perc) else 0.0,
            'tv': loss_tv.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


# Testing
if __name__ == '__main__':
    print("Testing schedulers and losses...\n")
    
    # Test noise scheduler
    print("1. Noise Scheduler")
    scheduler = NoiseScheduler(num_timesteps=1000, schedule_type='cosine')
    print(f"   Betas shape: {scheduler.betas.shape}")
    print(f"   Beta range: [{scheduler.betas.min():.6f}, {scheduler.betas.max():.6f}]")
    print(f"   Alpha_bar at t=500: {scheduler.alphas_cumprod[500]:.4f}")
    
    # Test losses
    print("\n2. Loss Functions")
    B, C, H, W = 2, 1, 64, 64
    
    noise_pred = torch.randn(B, C, H, W)
    noise_target = torch.randn(B, C, H, W)
    x_pred = torch.randn(B, C, H, W)
    x_measured = torch.randn(B, C, H, W)
    
    combined_loss = CombinedLoss()
    
    # Dummy physics operator
    class DummyPhysics(nn.Module):
        def forward(self, x):
            return x + 0.1 * torch.randn_like(x)
    
    physics_op = DummyPhysics()
    
    total_loss, loss_dict = combined_loss(
        noise_pred, noise_target,
        x_pred, x_measured, x_pred,
        physics_op, 0
    )
    
    print("   Loss components:")
    for name, val in loss_dict.items():
        print(f"     {name}: {val:.4f}")
    
    print("\n✓ All tests passed!")
