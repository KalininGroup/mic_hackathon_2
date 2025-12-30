"""
Surface Reconstruction Network with Geometric Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class UNetBlock(nn.Module):
    """U-Net building block"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.activation = nn.SiLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        return x


class GeometricAttention(nn.Module):
    """Attention mechanism with geometric priors"""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        # Learnable geometric convolution kernels
        self.geometric_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Output projection
        self.out_proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            attended: [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Apply geometric convolution
        geo_features = self.geometric_conv(x)  # [B, C, H, W]
        
        # Reshape for attention
        geo_flat = geo_features.view(B, C, H*W).transpose(1, 2)  # [B, H*W, C]
        x_flat = x.view(B, C, H*W).transpose(1, 2)  # [B, H*W, C]
        
        # Apply self-attention
        attended, _ = self.attention(x_flat, x_flat, x_flat)  # [B, H*W, C]
        
        # Reshape back
        attended = attended.transpose(1, 2).view(B, C, H, W)
        
        # Output projection with residual
        out = self.out_proj(attended) + x
        
        return out


class SurfaceUNet(nn.Module):
    """U-Net for surface reconstruction"""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 features: list = [64, 128, 256, 512]):
        super().__init__()
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        in_ch = in_channels
        for feat in features:
            self.encoder_blocks.append(UNetBlock(in_ch, feat))
            in_ch = feat
        
        # Bottleneck with attention
        self.bottleneck = UNetBlock(features[-1], features[-1] * 2)
        self.attention = GeometricAttention(features[-1] * 2, num_heads=8)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        for i in range(len(features) - 1, -1, -1):
            in_ch = features[i] * 2 if i == len(features) - 1 else features[i] * 3
            out_ch = features[i]
            
            self.upconvs.append(
                nn.ConvTranspose2d(in_ch if i == len(features)-1 else features[i+1], 
                                  out_ch, 2, stride=2)
            )
            self.decoder_blocks.append(UNetBlock(out_ch * 2, out_ch))
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, out_channels, H, W]
        """
        # Encoder
        encoder_features = []
        for enc_block in self.encoder_blocks:
            x = enc_block(x)
            encoder_features.append(x)
            x = self.pool(x)
        
        # Bottleneck with attention
        x = self.bottleneck(x)
        x = self.attention(x)
        
        # Decoder
        encoder_features = encoder_features[::-1]  # Reverse order
        
        for i, (upconv, dec_block) in enumerate(zip(self.upconvs, self.decoder_blocks)):
            x = upconv(x)
            # Concatenate with skip connection
            x = torch.cat([x, encoder_features[i]], dim=1)
            x = dec_block(x)
        
        # Final output
        out = self.final_conv(x)
        
        return out


class SurfaceReconstructor(nn.Module):
    """Complete surface reconstruction with tip features"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Tip feature extractor (3D to 2D projection)
        self.tip_projector = self._build_tip_projector()
        
        # Main U-Net
        self.unet = SurfaceUNet(
            in_channels=2,  # Image + tip projection
            out_channels=1,
            features=[64, 128, 256, 512]
        )
        
        # Refinement network
        self.refine_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 1, 1),
            nn.Tanh()  # Surface can have positive and negative deviations
        )
    
    def _build_tip_projector(self):
        """Project 3D tip to 2D features"""
        return nn.Sequential(
            # Pool along depth dimension
            # Input: [B, 1, H, W, D] -> Output: [B, 64, H_img, W_img]
            nn.Conv3d(1, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.GroupNorm(16, 64),
            nn.SiLU()
        )
    
    def forward(self, afm_image: torch.Tensor, estimated_tip: torch.Tensor,
                tip_uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Args:
            afm_image: [B, 1, H, W]
            estimated_tip: [B, 1, H_t, W_t, D_t]
            tip_uncertainty: [B, 1, H_t, W_t, D_t]
        
        Returns:
            surface: [B, 1, H, W]
        """
        B, _, H, W = afm_image.shape
        
        # Extract tip features
        tip_features = self.tip_projector(estimated_tip)  # [B, 64, H_t, W_t, D_t]
        
        # Max project along depth to get 2D representation
        tip_projection = torch.max(tip_features, dim=4)[0]  # [B, 64, H_t, W_t]
        
        # Resize tip projection to match image size
        tip_projection = F.interpolate(
            tip_projection, size=(H, W), mode='bilinear', align_corners=False
        )
        
        # Take mean across channels for single channel input
        tip_projection = torch.mean(tip_projection, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Concatenate image and tip features
        combined = torch.cat([afm_image, tip_projection], dim=1)  # [B, 2, H, W]
        
        # Reconstruct surface
        surface_coarse = self.unet(combined)  # [B, 1, H, W]
        
        # Refine
        surface_refined = self.refine_conv(surface_coarse)
        surface = surface_coarse + 0.1 * surface_refined
        
        # Apply sigmoid to get normalized output
        surface = torch.sigmoid(surface)
        
        return surface


def test_surface_network():
    """Test surface reconstruction network"""
    from config import get_config
    
    config = get_config()
    model = SurfaceReconstructor(config).cuda()
    
    # Test inputs
    batch_size = 2
    afm_image = torch.randn(batch_size, 1, 128, 128).cuda()
    tip = torch.randn(batch_size, 1, 32, 32, 32).cuda()
    tip_unc = torch.randn(batch_size, 1, 32, 32, 32).cuda()
    
    # Forward pass
    surface = model(afm_image, tip, tip_unc)
    
    print("Surface Network Test:")
    print(f"  Image shape: {afm_image.shape}")
    print(f"  Tip shape: {tip.shape}")
    print(f"  Surface shape: {surface.shape}")
    print(f"  Surface range: [{surface.min():.3f}, {surface.max():.3f}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

if __name__ == '__main__':
    test_surface_network()
