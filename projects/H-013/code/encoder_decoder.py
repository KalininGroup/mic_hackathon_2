"""
U-Net Encoder-Decoder with Fourier Features for 4D-STEM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig


class FourierFeatures(nn.Module):
    """Add Fourier features for better frequency representation in diffraction patterns"""
    
    def __init__(self, num_features=128, scale=10.0):
        super().__init__()
        self.num_features = num_features
        self.scale = scale
        # Fixed random frequencies
        self.register_buffer('freqs', torch.randn(num_features, 2) * scale)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C + 2*num_features, H, W]
        """
        B, C, H, W = x.shape
        
        # Create coordinate grid
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        coords = torch.stack([x_coords, y_coords], dim=-1)  # [H, W, 2]
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]
        
        # Compute Fourier features
        projected = torch.matmul(coords, self.freqs.T)  # [B, H, W, F]
        sin_features = torch.sin(2 * np.pi * projected)
        cos_features = torch.cos(2 * np.pi * projected)
        
        fourier_features = torch.cat([sin_features, cos_features], dim=-1)
        fourier_features = fourier_features.permute(0, 3, 1, 2)  # [B, 2F, H, W]
        
        # Concatenate with input
        return torch.cat([x, fourier_features], dim=1)


class DownBlock(nn.Module):
    """Downsampling block with residual connections"""
    
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
        
    def forward(self, x):
        # Store input for residual
        identity = self.residual_conv(x)
        
        # First conv block
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.norm2(x)
        
        # Residual connection
        x = x + identity
        x = self.activation(x)
        
        # Store for skip connection (before downsampling)
        skip = x
        
        # Downsample
        x = self.downsample(x)
        
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block with skip connections"""
    
    def __init__(self, in_channels, out_channels, skip_channels, dropout=0.1):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 
                                          kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x, skip):
        # Upsample
        x = self.upsample(x)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # First conv block
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        
        return x


class UNetEncoder(nn.Module):
    """U-Net encoder with Fourier features for diffraction patterns"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Fourier feature layer
        self.fourier = FourierFeatures(num_features=config.fourier_features)
        
        # Initial convolution (1 input channel + 2*fourier_features)
        initial_channels = 1 + 2 * config.fourier_features
        self.init_conv = nn.Sequential(
            nn.Conv2d(initial_channels, config.encoder_channels[0], 3, padding=1),
            nn.GroupNorm(min(8, config.encoder_channels[0]), config.encoder_channels[0]),
            nn.SiLU()
        )
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        in_channels = config.encoder_channels[0]
        
        for out_channels in config.encoder_channels[1:]:
            self.down_blocks.append(
                DownBlock(in_channels, out_channels, dropout=config.dropout)
            )
            in_channels = out_channels
        
        # Calculate spatial dimensions after downsampling
        self.final_spatial_dim = config.image_size[0] // (2 ** len(config.encoder_channels[1:]))
        
        # Final projection to latent space
        self.latent_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.encoder_channels[-1], config.latent_dim),
            nn.LayerNorm(config.latent_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, T, 1, H, W] or [B, 1, H, W]
        Returns:
            latent: [B, T, latent_dim] or [B, latent_dim]
            skip_connections: List of tensors for decoder
        """
        # Handle temporal dimension
        has_temp = len(x.shape) == 5
        if has_temp:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
        
        # Add Fourier features for diffraction patterns
        x = self.fourier(x)
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Store skip connections
        skip_connections = []
        
        # Downsample
        for down_block in self.down_blocks:
            x, skip = down_block(x)
            skip_connections.append(skip)
        
        # Project to latent space
        latent = self.latent_proj(x)
        
        if has_temp:
            latent = latent.reshape(B, T, -1)
            # Replicate skip connections for temporal dimension
            skip_connections_temp = []
            for skip in skip_connections:
                _, C, H, W = skip.shape
                skip_reshaped = skip.reshape(B, T, C, H, W)
                skip_connections_temp.append(skip_reshaped)
            skip_connections = skip_connections_temp
        
        return latent, skip_connections


class UNetDecoder(nn.Module):
    """U-Net decoder for reconstructing 4D-STEM patterns"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Calculate spatial dimensions after encoding
        self.final_spatial_dim = config.image_size[0] // (2 ** len(config.encoder_channels[1:]))
        
        # Initial projection from latent
        self.latent_expand = nn.Sequential(
            nn.Linear(config.latent_dim, 
                     config.decoder_channels[0] * self.final_spatial_dim * self.final_spatial_dim),
            nn.LayerNorm(config.decoder_channels[0] * self.final_spatial_dim * self.final_spatial_dim)
        )
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        in_channels = config.decoder_channels[0]
        
        for idx, out_channels in enumerate(config.decoder_channels[1:]):
            skip_channels = config.encoder_channels[-(idx+2)]
            self.up_blocks.append(
                UpBlock(in_channels, out_channels, skip_channels, dropout=config.dropout)
            )
            in_channels = out_channels
        
        # Final convolution to output
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()  # Ensure output is in [0, 1]
        )
    
    def forward(self, latent, skip_connections):
        """
        Args:
            latent: [B, T, latent_dim] or [B, latent_dim]
            skip_connections: List of tensors from encoder
        Returns:
            [B, T, 1, H, W] or [B, 1, H, W]
        """
        # Handle temporal dimension
        has_temp = len(latent.shape) == 3
        if has_temp:
            B, T, D = latent.shape
            latent = latent.reshape(B * T, D)
            # Flatten temporal dimension in skip connections
            skip_connections_flat = []
            for skip in skip_connections:
                _, _, C, H, W = skip.shape
                skip_flat = skip.reshape(B * T, C, H, W)
                skip_connections_flat.append(skip_flat)
            skip_connections = skip_connections_flat
        
        # Expand latent to spatial dimensions
        x = self.latent_expand(latent)
        x = x.reshape(-1, self.config.decoder_channels[0], 
                     self.final_spatial_dim, self.final_spatial_dim)
        
        # Upsample with skip connections
        for up_block, skip in zip(self.up_blocks, reversed(skip_connections)):
            x = up_block(x, skip)
        
        # Final convolution
        x = self.final_conv(x)
        
        if has_temp:
            x = x.reshape(B, T, 1, self.config.image_size[0], self.config.image_size[1])
        
        return x


if __name__ == "__main__":
    # Test encoder-decoder
    from config import ModelConfig
    
    config = ModelConfig()
    config.image_size = (64, 64)
    config.encoder_channels = [64, 128, 256]
    config.decoder_channels = [256, 128, 64]
    
    print("Testing U-Net Encoder-Decoder...")
    
    # Create encoder and decoder
    encoder = UNetEncoder(config)
    decoder = UNetDecoder(config)
    
    # Test with single image
    print("\n1. Testing single image:")
    x = torch.randn(4, 1, 64, 64)
    latent, skips = encoder(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Latent shape: {latent.shape}")
    print(f"   Number of skip connections: {len(skips)}")
    
    recon = decoder(latent, skips)
    print(f"   Reconstruction shape: {recon.shape}")
    
    # Test with temporal sequence
    print("\n2. Testing temporal sequence:")
    x_seq = torch.randn(2, 10, 1, 64, 64)
    latent_seq, skips_seq = encoder(x_seq)
    print(f"   Input shape: {x_seq.shape}")
    print(f"   Latent shape: {latent_seq.shape}")
    print(f"   Skip connection 0 shape: {skips_seq[0].shape}")
    
    recon_seq = decoder(latent_seq, skips_seq)
    print(f"   Reconstruction shape: {recon_seq.shape}")
    
    print("\nEncoder-Decoder test passed!")
