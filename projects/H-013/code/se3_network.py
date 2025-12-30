"""
SE(3)-Equivariant Neural Network for Tip Reconstruction
Uses spherical harmonics and equivariant convolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

# Note: For production, use e3nn library
# Here we provide a simplified but functional implementation

class SphericalHarmonics(nn.Module):
    """Compute spherical harmonics up to degree L"""
    
    def __init__(self, max_degree: int = 3):
        super().__init__()
        self.max_degree = max_degree
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute spherical harmonics
        Args:
            positions: [B, N, 3] - normalized direction vectors
        Returns:
            sh_features: [B, N, num_harmonics]
        """
        x, y, z = positions[..., 0], positions[..., 1], positions[..., 2]
        
        sh_features = []
        
        # L = 0 (1 component)
        sh_features.append(torch.ones_like(x) * 0.282095)  # Y_00
        
        if self.max_degree >= 1:
            # L = 1 (3 components)
            sh_features.append(0.488603 * y)  # Y_1-1
            sh_features.append(0.488603 * z)  # Y_10
            sh_features.append(0.488603 * x)  # Y_11
        
        if self.max_degree >= 2:
            # L = 2 (5 components)
            sh_features.append(1.092548 * x * y)  # Y_2-2
            sh_features.append(1.092548 * y * z)  # Y_2-1
            sh_features.append(0.315392 * (3*z*z - 1))  # Y_20
            sh_features.append(1.092548 * x * z)  # Y_21
            sh_features.append(0.546274 * (x*x - y*y))  # Y_22
        
        if self.max_degree >= 3:
            # L = 3 (7 components) - simplified
            sh_features.append(0.590044 * y * (3*x*x - y*y))  # Y_3-3
            sh_features.append(2.890611 * x * y * z)  # Y_3-2
            sh_features.append(0.457046 * y * (5*z*z - 1))  # Y_3-1
            sh_features.append(0.373176 * z * (5*z*z - 3))  # Y_30
            sh_features.append(0.457046 * x * (5*z*z - 1))  # Y_31
            sh_features.append(1.445306 * z * (x*x - y*y))  # Y_32
            sh_features.append(0.590044 * x * (x*x - 3*y*y))  # Y_33
        
        return torch.stack(sh_features, dim=-1)


class RadialBasisFunctions(nn.Module):
    """Learnable radial basis functions"""
    
    def __init__(self, num_radial: int = 16, cutoff: float = 5.0):
        super().__init__()
        self.num_radial = num_radial
        self.cutoff = cutoff
        
        # Learnable parameters
        self.centers = nn.Parameter(torch.linspace(0, cutoff, num_radial))
        self.widths = nn.Parameter(torch.ones(num_radial) * 0.5)
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: [B, N] - distances from origin
        Returns:
            rbf_features: [B, N, num_radial]
        """
        distances = distances.unsqueeze(-1)  # [B, N, 1]
        centers = self.centers.view(1, 1, -1)  # [1, 1, num_radial]
        widths = self.widths.view(1, 1, -1)
        
        # Gaussian RBF
        rbf = torch.exp(-((distances - centers) ** 2) / (2 * widths ** 2))
        
        # Smooth cutoff
        cutoff_mask = (distances <= self.cutoff).float()
        rbf = rbf * cutoff_mask
        
        return rbf


class SE3EquivariantConv(nn.Module):
    """SE(3)-equivariant convolution layer"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 max_degree: int = 3, num_radial: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_degree = max_degree
        
        # Compute number of spherical harmonics
        self.num_sh = (max_degree + 1) ** 2
        
        # Learnable weights for combining radial and angular parts
        self.weight = nn.Linear(in_channels * self.num_sh * num_radial, 
                               out_channels)
        
        self.radial_basis = RadialBasisFunctions(num_radial)
        self.spherical_harmonics = SphericalHarmonics(max_degree)
    
    def forward(self, features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N, in_channels]
            positions: [B, N, 3]
        Returns:
            out_features: [B, N, out_channels]
        """
        B, N, C = features.shape
        
        # Compute distances
        distances = torch.norm(positions, dim=-1, keepdim=False)  # [B, N]
        
        # Normalize positions for spherical harmonics
        positions_normalized = positions / (distances.unsqueeze(-1) + 1e-8)
        
        # Compute radial basis
        rbf = self.radial_basis(distances)  # [B, N, num_radial]
        
        # Compute spherical harmonics
        sh = self.spherical_harmonics(positions_normalized)  # [B, N, num_sh]
        
        # Combine features with geometric information
        # Outer product of features, rbf, and sh
        geo_features = torch.einsum('bnc,bnr,bns->bncrs', features, rbf, sh)
        geo_features = geo_features.reshape(B, N, -1)
        
        # Apply learnable transformation
        out = self.weight(geo_features)
        
        return out


class SE3EquivariantBlock(nn.Module):
    """Complete SE(3)-equivariant processing block"""
    
    def __init__(self, channels: int, max_degree: int = 3, num_radial: int = 16):
        super().__init__()
        
        self.conv = SE3EquivariantConv(channels, channels, max_degree, num_radial)
        self.norm = nn.LayerNorm(channels)
        self.activation = nn.SiLU()
        
        # Residual connection
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N, C]
            positions: [B, N, 3]
        Returns:
            out_features: [B, N, C]
        """
        identity = features
        
        out = self.conv(features, positions)
        out = self.norm(out)
        out = self.activation(out)
        
        # Residual connection
        out = identity + self.residual_weight * out
        
        return out


class SE3TipReconstructor(nn.Module):
    """SE(3)-equivariant tip reconstruction network"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Image encoder (2D CNN)
        self.image_encoder = self._build_image_encoder()
        
        # Project to 3D feature space
        self.to_3d = nn.Linear(config.model.image_features, 
                               config.model.hidden_dim)
        
        # SE(3)-equivariant blocks
        self.se3_blocks = nn.ModuleList([
            SE3EquivariantBlock(
                config.model.hidden_dim,
                config.model.max_degree,
                config.model.num_radial
            )
            for _ in range(config.model.num_se3_layers)
        ])
        
        # Output head for tip density
        self.tip_head = nn.Sequential(
            nn.Linear(config.model.hidden_dim, config.model.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(config.model.hidden_dim // 2, 1),
            nn.Sigmoid()  # Tip density in [0, 1]
        )
        
        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config.model.hidden_dim, config.model.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(config.model.hidden_dim // 2, 1),
            nn.Softplus()  # Positive uncertainty
        )
        
        # Dropout for uncertainty estimation
        self.dropout = nn.Dropout(config.model.dropout_rate)
    
    def _build_image_encoder(self):
        """2D CNN encoder for AFM images"""
        return nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.MaxPool2d(2),
            
            # 64x64 -> 32x32
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(16, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(16, 64),
            nn.SiLU(),
            nn.MaxPool2d(2),
            
            # 32x32 -> 16x16
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.MaxPool2d(2),
            
            # Global pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
    
    def _generate_tip_grid(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate 3D grid positions for tip"""
        H, W, D = self.config.data.tip_voxel_size
        
        # Create normalized grid [-1, 1]
        z = torch.linspace(-1, 1, H, device=device)
        y = torch.linspace(-1, 1, W, device=device)
        x = torch.linspace(-1, 1, D, device=device)
        
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # [H, W, D, 3]
        grid = grid.reshape(-1, 3)  # [H*W*D, 3]
        
        # Expand for batch
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, 3]
        
        return grid
    
    def forward(self, afm_image: torch.Tensor, 
                monte_carlo: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            afm_image: [B, 1, H, W]
            monte_carlo: Whether to enable dropout for uncertainty
        
        Returns:
            tip_density: [B, 1, H_t, W_t, D_t]
            uncertainty: [B, 1, H_t, W_t, D_t]
        """
        B = afm_image.shape[0]
        device = afm_image.device
        
        # Encode image
        image_features = self.image_encoder(afm_image)  # [B, 128]
        
        # Generate 3D grid
        grid_positions = self._generate_tip_grid(B, device)  # [B, N, 3]
        N = grid_positions.shape[1]
        
        # Project to 3D space
        features_3d = self.to_3d(image_features)  # [B, hidden_dim]
        features_3d = features_3d.unsqueeze(1).expand(-1, N, -1)  # [B, N, hidden_dim]
        
        # Apply SE(3)-equivariant processing
        if monte_carlo:
            features_3d = self.dropout(features_3d)
        
        for se3_block in self.se3_blocks:
            features_3d = se3_block(features_3d, grid_positions)
            if monte_carlo:
                features_3d = self.dropout(features_3d)
        
        # Predict tip density and uncertainty
        tip_density = self.tip_head(features_3d)  # [B, N, 1]
        uncertainty = self.uncertainty_head(features_3d)  # [B, N, 1]
        
        # Reshape to 3D volume
        H, W, D = self.config.data.tip_voxel_size
        tip_density = tip_density.reshape(B, H, W, D).unsqueeze(1)  # [B, 1, H, W, D]
        uncertainty = uncertainty.reshape(B, H, W, D).unsqueeze(1)  # [B, 1, H, W, D]
        
        return tip_density, uncertainty


def test_se3_network():
    """Test SE(3)-equivariant network"""
    from config import get_config
    
    config = get_config()
    model = SE3TipReconstructor(config).cuda()
    
    # Test forward pass
    batch_size = 2
    afm_image = torch.randn(batch_size, 1, 128, 128).cuda()
    
    tip_density, uncertainty = model(afm_image, monte_carlo=False)
    
    print("SE(3) Network Test:")
    print(f"  Input shape: {afm_image.shape}")
    print(f"  Tip density shape: {tip_density.shape}")
    print(f"  Uncertainty shape: {uncertainty.shape}")
    print(f"  Tip density range: [{tip_density.min():.3f}, {tip_density.max():.3f}]")
    print(f"  Uncertainty range: [{uncertainty.min():.3f}, {uncertainty.max():.3f}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

if __name__ == '__main__':
    test_se3_network()
