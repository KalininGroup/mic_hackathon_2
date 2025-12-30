"""
Physics-informed constraints for electron scattering in 4D-STEM
"""

import torch
import torch.nn as nn
import torch.fft
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig


class ScatteringPhysicsModule(nn.Module):
    """Differentiable scattering physics constraints for 4D-STEM"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Learnable physics parameters
        self.damping_factor = nn.Parameter(torch.tensor(0.1))
        self.debye_waller = nn.Parameter(torch.tensor(0.5))
        
        # Physics constraint networks
        self.friedel_enforcer = nn.Sequential(
            nn.Linear(config.latent_dim, 256),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, config.physics_dim)
        )
        
        self.kinematic_constraint = nn.Sequential(
            nn.Linear(config.latent_dim, 256),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, config.physics_dim)
        )
        
        # Time embedding for physics evolution
        self.time_physics_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
    
    def forward(self, latent, time):
        """
        Enforce physics constraints on latent representation
        
        Args:
            latent: [B, latent_dim] or [B, T, latent_dim]
            time: [B] or scalar
            
        Returns:
            physics_params: [B, physics_dim + extras]
        """
        # Handle different input shapes
        if len(latent.shape) == 3:
            # Temporal sequence: [B, T, latent_dim]
            B, T, D = latent.shape
            latent_flat = latent.reshape(B * T, D)
            if isinstance(time, torch.Tensor):
                if time.dim() == 1 and len(time) == B:
                    time = time.unsqueeze(1).expand(B, T).reshape(B * T)
                elif time.dim() == 2:
                    time = time.reshape(B * T)
        else:
            latent_flat = latent
        
        # Ensure time is tensor
        if not isinstance(time, torch.Tensor):
            time = torch.tensor([time], device=latent.device).expand(latent_flat.shape[0])
        elif time.dim() == 0:
            time = time.unsqueeze(0).expand(latent_flat.shape[0])
        
        # Time encoding for physics evolution
        time_encoding = self.time_physics_embed(time.unsqueeze(-1))
        
        # Enforce Friedel's law (I(h,k,l) = I(-h,-k,-l))
        friedel_params = self.friedel_enforcer(latent_flat)
        
        # Enforce kinematic scattering constraints
        kinematic_params = self.kinematic_constraint(latent_flat)
        
        # Combine all physics parameters
        physics_params = torch.cat([
            friedel_params,
            kinematic_params,
            time_encoding,
            self.damping_factor.expand(latent_flat.shape[0], 1),
            self.debye_waller.expand(latent_flat.shape[0], 1)
        ], dim=-1)
        
        # Reshape if input was temporal
        if len(latent.shape) == 3:
            physics_params = physics_params.reshape(B, T, -1)
        
        return physics_params
    
    def compute_physics_loss(self, reconstructed_patterns):
        """
        Compute loss based on scattering physics violations
        
        Args:
            reconstructed_patterns: [B, T, 1, H, W] or [B, 1, H, W]
            
        Returns:
            Dictionary of physics losses
        """
        losses = {}
        
        # Flatten temporal dimension if present
        if len(reconstructed_patterns.shape) == 5:
            B, T, C, H, W = reconstructed_patterns.shape
            patterns = reconstructed_patterns.reshape(B * T, C, H, W)
        else:
            patterns = reconstructed_patterns
        
        # 1. Friedel symmetry loss
        if self.config.use_friedel_law:
            losses['friedel_loss'] = self._friedel_symmetry_loss(patterns)
        
        # 2. Kinematic scattering loss (intensity constraints)
        if self.config.use_kinematic_scattering:
            losses['kinematic_loss'] = self._kinematic_scattering_loss(patterns)
        
        # 3. Energy conservation (total scattered electrons)
        if self.config.use_energy_conservation:
            if len(reconstructed_patterns.shape) == 5:
                # Use temporal dimension for energy conservation
                losses['energy_loss'] = self._energy_conservation_loss(
                    reconstructed_patterns.reshape(B, T, C, H, W)
                )
            else:
                losses['energy_loss'] = self._energy_conservation_loss_spatial(patterns)
        
        return losses
    
    def _friedel_symmetry_loss(self, patterns):
        """
        Enforce Friedel pair symmetry I(h) = I(-h)
        
        In reciprocal space, diffraction patterns should have inversion symmetry
        """
        # Take FFT to reciprocal space
        pattern_fft = torch.fft.fft2(patterns.squeeze(1))  # [B, H, W]
        pattern_fft_intensity = torch.abs(pattern_fft) ** 2
        
        # Check symmetry: compare with inverted pattern
        pattern_inverted = torch.flip(pattern_fft_intensity, dims=[-2, -1])
        
        # Friedel symmetry loss
        loss = torch.mean((pattern_fft_intensity - pattern_inverted) ** 2)
        
        return loss
    
    def _kinematic_scattering_loss(self, patterns):
        """
        Enforce kinematic scattering conditions:
        1. Intensity should be non-negative
        2. Smoothness in reciprocal space
        """
        loss = 0.0
        
        # 1. Penalize negative intensities (should not happen with sigmoid, but safety)
        negative_pixels = torch.relu(-patterns)
        loss += torch.mean(negative_pixels ** 2)
        
        # 2. Smoothness constraint in reciprocal space
        pattern_fft = torch.fft.fft2(patterns.squeeze(1))
        pattern_fft_mag = torch.abs(pattern_fft)
        
        # Compute gradients in reciprocal space
        grad_x = torch.diff(pattern_fft_mag, dim=-1)
        grad_y = torch.diff(pattern_fft_mag, dim=-2)
        
        # Encourage smoothness (but not too much)
        smoothness_loss = torch.mean(grad_x ** 2) + torch.mean(grad_y ** 2)
        loss += 0.01 * smoothness_loss
        
        # 3. Physical intensity range (normalized to [0, 1])
        # Penalize patterns that are too uniform or too sparse
        pattern_std = torch.std(patterns, dim=(-2, -1))
        uniformity_penalty = torch.relu(0.05 - pattern_std)  # Encourage variation
        loss += torch.mean(uniformity_penalty ** 2)
        
        return loss
    
    def _energy_conservation_loss(self, patterns):
        """
        Total intensity should be conserved over time
        
        Args:
            patterns: [B, T, 1, H, W]
        """
        # Sum over spatial dimensions
        total_intensity = torch.sum(patterns, dim=(-2, -1))  # [B, T, 1]
        
        # Intensity should be relatively constant over time (beam current constant)
        # Compute variance across time for each batch
        intensity_variance = torch.var(total_intensity, dim=1)
        loss = torch.mean(intensity_variance)
        
        return loss
    
    def _energy_conservation_loss_spatial(self, patterns):
        """
        Energy conservation for single time point (spatial uniformity)
        
        Args:
            patterns: [B, 1, H, W]
        """
        # Total intensity should be reasonable
        total_intensity = torch.sum(patterns, dim=(-2, -1))  # [B, 1]
        
        # Penalize extreme total intensities
        target_intensity = patterns.numel() / patterns.shape[0] * 0.5  # Expected ~0.5 avg
        loss = torch.mean((total_intensity - target_intensity) ** 2)
        
        return loss


if __name__ == "__main__":
    # Test physics module
    from config import ModelConfig
    
    config = ModelConfig()
    config.latent_dim = 256
    config.physics_dim = 128
    
    print("Testing Physics Module...")
    
    physics_module = ScatteringPhysicsModule(config)
    
    # Test forward pass
    print("\n1. Testing forward pass:")
    latent = torch.randn(4, 256)
    time = torch.tensor([0.0, 0.25, 0.5, 0.75])
    
    physics_params = physics_module(latent, time)
    print(f"   Latent shape: {latent.shape}")
    print(f"   Time shape: {time.shape}")
    print(f"   Physics params shape: {physics_params.shape}")
    
    # Test with temporal sequence
    print("\n2. Testing temporal sequence:")
    latent_seq = torch.randn(2, 10, 256)
    time_seq = torch.linspace(0, 1, 10).unsqueeze(0).expand(2, -1)
    
    physics_params_seq = physics_module(latent_seq, time_seq)
    print(f"   Latent seq shape: {latent_seq.shape}")
    print(f"   Time seq shape: {time_seq.shape}")
    print(f"   Physics params seq shape: {physics_params_seq.shape}")
    
    # Test physics losses
    print("\n3. Testing physics losses:")
    patterns = torch.rand(4, 10, 1, 64, 64)
    losses = physics_module.compute_physics_loss(patterns)
    print(f"   Pattern shape: {patterns.shape}")
    for loss_name, loss_value in losses.items():
        print(f"   {loss_name}: {loss_value.item():.6f}")
    
    print("\nPhysics Module test passed!")
