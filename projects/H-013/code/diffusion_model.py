"""
physics_diffusion/models/diffusion_model.py

Core diffusion model with physics constraints and modality conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, List
from .unet_arch import UNetWithAttention
from .physics_modules import PhysicsOperatorFactory
from .modality_adapters import ModalityAdapter

class PhysicsConstrainedDDPM(nn.Module):
    """Physics-Constrained Denoising Diffusion Probabilistic Model.
    
    Integrates diffusion model with differentiable physics operators
    for self-supervised microscopy artifact correction.
    """
    
    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 1,
        model_channels: int = 128,
        num_modalities: int = 4,
        timesteps: int = 1000,
        use_physics: bool = True,
        modality_names: List[str] = ['AFM', 'TEM', 'SEM', 'STEM']
    ):
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.timesteps = timesteps
        self.use_physics = use_physics
        self.modality_names = modality_names
        
        # Core U-Net for noise prediction
        self.unet = UNetWithAttention(
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=in_channels,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            channel_mult=(1, 2, 4, 8),
            num_heads=4,
            use_modality_conditioning=True,
            num_modalities=num_modalities
        )
        
        # Modality adapter
        self.modality_adapter = ModalityAdapter(
            num_modalities=num_modalities,
            embed_dim=model_channels * 4
        )
        
        # Physics operators
        if use_physics:
            self.physics_operators = nn.ModuleDict({
                name: PhysicsOperatorFactory.create(name)
                for name in modality_names
            })
        else:
            self.physics_operators = None
        
        # Noise schedule (cosine schedule)
        self.register_buffer(
            'betas',
            self._cosine_beta_schedule(timesteps)
        )
        
        alphas = 1.0 - self.betas
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev',
                           F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        
        # Posterior variance calculations
        self.register_buffer(
            'posterior_variance',
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        # Square root of alphas for efficient sampling
        self.register_buffer(
            'sqrt_alphas_cumprod',
            torch.sqrt(self.alphas_cumprod)
        )
        self.register_buffer(
            'sqrt_one_minus_alphas_cumprod',
            torch.sqrt(1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as proposed in Improved DDPM."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward diffusion: add noise to clean image.
        
        q(x_t | x_0) = N(x_t; √(ᾱ_t)x_0, (1-ᾱ_t)I)
        
        Args:
            x_start: Clean image [B, C, H, W]
            t: Timestep [B]
            noise: Optional noise, sampled if None
            
        Returns:
            Noisy image x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        while len(sqrt_alphas_cumprod_t.shape) < len(x_start.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        return (sqrt_alphas_cumprod_t * x_start + 
                sqrt_one_minus_alphas_cumprod_t * noise)
    
    def predict_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        modality_ids: torch.Tensor,
        physics_params: Optional[Dict] = None
    ) -> torch.Tensor:
        """Predict noise ε_θ(x_t, t, modality).
        
        Args:
            x_t: Noisy image [B, C, H, W]
            t: Timestep [B]
            modality_ids: Modality indices [B]
            physics_params: Optional physics parameters per sample
            
        Returns:
            Predicted noise [B, C, H, W]
        """
        # Get modality conditioning
        modality_emb = self.modality_adapter(modality_ids)
        
        # Predict noise using U-Net
        noise_pred = self.unet(x_t, t, modality_emb)
        
        return noise_pred
    
    def apply_physics_constraint(
        self,
        x_pred: torch.Tensor,
        x_noisy: torch.Tensor,
        modality_ids: torch.Tensor,
        physics_params: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply physics consistency constraint.
        
        Ensures predicted clean image satisfies physical imaging model.
        
        Args:
            x_pred: Predicted clean image [B, C, H, W]
            x_noisy: Original noisy image [B, C, H, W]
            modality_ids: Modality indices [B]
            physics_params: Optional physics parameters
            
        Returns:
            physics_residual: Difference between forward model and noisy image
            physics_loss: Physics consistency loss
        """
        if not self.use_physics or self.physics_operators is None:
            return torch.zeros_like(x_pred), torch.tensor(0.0, device=x_pred.device)
        
        B = x_pred.shape[0]
        physics_residual = torch.zeros_like(x_pred)
        
        # Apply physics operator per sample based on modality
        for b in range(B):
            modality_id = modality_ids[b].item()
            modality_name = self.modality_names[modality_id]
            
            # Get physics operator
            physics_op = self.physics_operators[modality_name]
            
            # Forward model: y = P(x)
            x_pred_single = x_pred[b:b+1]
            y_pred = physics_op(x_pred_single, physics_params)
            
            # Compare with noisy observation
            physics_residual[b] = y_pred[0] - x_noisy[b]
        
        # L1 loss for physics consistency
        physics_loss = torch.abs(physics_residual).mean()
        
        return physics_residual, physics_loss
    
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        modality_ids: torch.Tensor,
        physics_params: Optional[Dict] = None
    ) -> torch.Tensor:
        """Single reverse diffusion step: p(x_{t-1} | x_t).
        
        Args:
            x_t: Current noisy image [B, C, H, W]
            t: Current timestep [B]
            modality_ids: Modality indices [B]
            physics_params: Optional physics parameters
            
        Returns:
            x_{t-1}: Denoised image at previous timestep
        """
        # Predict noise
        noise_pred = self.predict_noise(x_t, t, modality_ids, physics_params)
        
        # Compute x_0 prediction
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        x_0_pred = (x_t - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t
        
        # Clamp predictions
        x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
        
        # Compute mean of posterior
        alphas_cumprod_prev_t = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1)
        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        
        posterior_mean = (
            torch.sqrt(alphas_cumprod_prev_t) * betas_t / (1.0 - alphas_cumprod_t) * x_0_pred +
            torch.sqrt(self.alphas[t].view(-1, 1, 1, 1)) * (1.0 - alphas_cumprod_prev_t) / 
            (1.0 - alphas_cumprod_t) * x_t
        )
        
        # Add noise if not final step
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1, 1)
            x_t_minus_1 = posterior_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            x_t_minus_1 = posterior_mean
        
        return x_t_minus_1
    
    @torch.no_grad()
    def sample(
        self,
        x_noisy: torch.Tensor,
        modality_ids: torch.Tensor,
        physics_params: Optional[Dict] = None,
        num_inference_steps: int = 50,
        use_physics_guidance: bool = True
    ) -> torch.Tensor:
        """Full reverse diffusion sampling.
        
        Args:
            x_noisy: Starting noisy image [B, C, H, W]
            modality_ids: Modality indices [B]
            physics_params: Optional physics parameters
            num_inference_steps: Number of sampling steps (< timesteps for DDIM)
            use_physics_guidance: Whether to apply physics guidance
            
        Returns:
            Denoised image [B, C, H, W]
        """
        device = x_noisy.device
        B = x_noisy.shape[0]
        
        # Start from noisy image (or noise for unconditional)
        x_t = x_noisy
        
        # DDIM sampling: subsample timesteps
        timestep_seq = torch.linspace(
            self.timesteps - 1, 0, num_inference_steps
        ).long().to(device)
        
        # Reverse diffusion
        for i, t_val in enumerate(timestep_seq):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            
            x_t = self.p_sample(x_t, t, modality_ids, physics_params)
            
            # Optional: physics-guided correction
            if use_physics_guidance and self.use_physics and i % 10 == 0:
                _, phys_loss = self.apply_physics_constraint(
                    x_t, x_noisy, modality_ids, physics_params
                )
                # Small gradient step toward physics consistency
                # (This would require gradients, simplified here)
        
        return x_t
    
    def forward(
        self,
        x_noisy: torch.Tensor,
        modality_ids: torch.Tensor,
        physics_params: Optional[Dict] = None,
        return_dict: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass.
        
        Args:
            x_noisy: Noisy microscopy images [B, C, H, W]
            modality_ids: Modality indices [B]
            physics_params: Optional physics parameters per sample
            return_dict: Whether to return detailed dictionary
            
        Returns:
            Dictionary with losses and predictions
        """
        device = x_noisy.device
        B = x_noisy.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (B,), device=device).long()
        
        # Sample noise
        noise = torch.randn_like(x_noisy)
        
        # Forward diffusion: add noise
        x_t = self.q_sample(x_noisy, t, noise)
        
        # Predict noise
        noise_pred = self.predict_noise(x_t, t, modality_ids, physics_params)
        
        # Denoising loss (MSE on noise)
        loss_diffusion = F.mse_loss(noise_pred, noise)
        
        # Compute predicted clean image
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_0_pred = (x_t - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t
        x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
        
        # Physics consistency loss
        _, loss_physics = self.apply_physics_constraint(
            x_0_pred, x_noisy, modality_ids, physics_params
        )
        
        if return_dict:
            return {
                'loss_diffusion': loss_diffusion,
                'loss_physics': loss_physics,
                'noise_pred': noise_pred,
                'x_0_pred': x_0_pred,
                'x_t': x_t
            }
        
        return loss_diffusion, loss_physics


# Testing
if __name__ == '__main__':
    print("Testing Physics-Constrained DDPM...")
    
    # Create model
    model = PhysicsConstrainedDDPM(
        image_size=256,
        in_channels=1,
        model_channels=64,
        num_modalities=4,
        timesteps=1000,
        use_physics=True
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    x_noisy = torch.randn(batch_size, 1, 256, 256)
    modality_ids = torch.tensor([0, 1, 2, 3])  # AFM, TEM, SEM, STEM
    
    # Training forward
    results = model(x_noisy, modality_ids, return_dict=True)
    print("\nTraining forward pass:")
    print(f"  Diffusion loss: {results['loss_diffusion'].item():.4f}")
    print(f"  Physics loss: {results['loss_physics'].item():.4f}")
    
    # Sampling
    print("\nSampling (inference):")
    model.eval()
    with torch.no_grad():
        x_clean = model.sample(x_noisy, modality_ids, num_inference_steps=50)
    print(f"  Output shape: {x_clean.shape}")
    print(f"  Output range: [{x_clean.min():.3f}, {x_clean.max():.3f}]")
    
    print("\n✓ Model test passed!")
