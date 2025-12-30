"""
physics_diffusion/models/unet_arch.py and modality_adapters.py combined

U-Net with self-attention and modality conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# ============================================================================
# Time and Modality Embeddings
# ============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embeddings for diffusion timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: [B] timesteps
        Returns:
            embeddings: [B, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ModalityAdapter(nn.Module):
    """Adapter that converts modality IDs to conditioning vectors."""
    
    def __init__(self, num_modalities: int, embed_dim: int):
        super().__init__()
        self.num_modalities = num_modalities
        self.embed_dim = embed_dim
        
        # Learnable modality embeddings
        self.modality_embeddings = nn.Embedding(num_modalities, embed_dim)
        
        # MLP to process embeddings
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
    
    def forward(self, modality_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            modality_ids: [B] integer modality indices
        Returns:
            modality_emb: [B, embed_dim]
        """
        emb = self.modality_embeddings(modality_ids)
        emb = self.mlp(emb)
        return emb


# ============================================================================
# U-Net Building Blocks
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with group normalization and conditioning."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First conv block
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # Second conv block
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            time_emb: [B, time_emb_dim]
        Returns:
            out: [B, out_channels, H, W]
        """
        h = self.conv1(F.silu(self.norm1(x)))
        
        # Add time conditioning
        time_proj = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_proj
        
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """Self-attention block for U-Net."""
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        assert channels % num_heads == 0
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        B, C, H, W = x.shape
        h = self.norm(x)
        
        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, heads, HW, C//heads]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(torch.einsum('bhqc,bhkc->bhqk', q, k) * scale, dim=-1)
        
        h = torch.einsum('bhqk,bhkc->bhqc', attn, v)
        h = h.reshape(B, C, H, W)
        h = self.proj(h)
        
        return x + h


class Downsample(nn.Module):
    """Downsampling layer."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# ============================================================================
# Complete U-Net Architecture
# ============================================================================

class UNetWithAttention(nn.Module):
    """U-Net with self-attention and time/modality conditioning.
    
    Architecture:
        - Encoder: 4 downsampling blocks (256→128→64→32→16)
        - Bottleneck: Transformer block with self-attention
        - Decoder: 4 upsampling blocks with skip connections
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        model_channels: int = 128,
        out_channels: int = 1,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (8, 16),
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_heads: int = 4,
        dropout: float = 0.1,
        use_modality_conditioning: bool = True,
        num_modalities: int = 4
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        
        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Modality conditioning (if used)
        self.use_modality_conditioning = use_modality_conditioning
        if use_modality_conditioning:
            self.modality_proj = nn.Linear(time_emb_dim, time_emb_dim)
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        
        ch = model_channels
        input_block_channels = [ch]
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(ch, model_channels * mult, time_emb_dim, dropout)]
                ch = model_channels * mult
                
                # Add attention at specified resolutions
                if 2 ** level in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                
                self.encoder_blocks.append(nn.ModuleList(layers))
                input_block_channels.append(ch)
            
            # Downsample (except last level)
            if level != len(channel_mult) - 1:
                self.downsample_blocks.append(Downsample(ch))
                input_block_channels.append(ch)
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResidualBlock(ch, ch, time_emb_dim, dropout),
            AttentionBlock(ch, num_heads),
            ResidualBlock(ch, ch, time_emb_dim, dropout)
        ])
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):
                # Skip connection from encoder
                skip_ch = input_block_channels.pop()
                layers = [ResidualBlock(ch + skip_ch, model_channels * mult, 
                                       time_emb_dim, dropout)]
                ch = model_channels * mult
                
                # Add attention
                if 2 ** level in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                
                # Upsample (except first block of each level and last level)
                if level != 0 and i == num_res_blocks:
                    layers.append(Upsample(ch))
                
                self.decoder_blocks.append(nn.ModuleList(layers))
        
        # Output
        self.out_norm = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        modality_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] input image
            timesteps: [B] diffusion timesteps
            modality_emb: [B, embed_dim] modality conditioning (optional)
            
        Returns:
            out: [B, out_channels, H, W] predicted noise
        """
        # Time embedding
        t_emb = self.time_embedding(timesteps)
        
        # Add modality conditioning
        if self.use_modality_conditioning and modality_emb is not None:
            mod_emb = self.modality_proj(modality_emb)
            t_emb = t_emb + mod_emb
        
        # Initial conv
        h = self.conv_in(x)
        
        # Encoder with skip connections
        encoder_outputs = [h]
        
        for i, block in enumerate(self.encoder_blocks):
            for layer in block:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
            encoder_outputs.append(h)
            
            # Downsample
            if i < len(self.downsample_blocks):
                h = self.downsample_blocks[i](h)
                encoder_outputs.append(h)
        
        # Bottleneck
        for layer in self.bottleneck:
            if isinstance(layer, ResidualBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
        
        # Decoder with skip connections
        for block in self.decoder_blocks:
            # Get skip connection
            skip = encoder_outputs.pop()
            h = torch.cat([h, skip], dim=1)
            
            for layer in block:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
        
        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h


# Testing
if __name__ == '__main__':
    print("Testing U-Net with Modality Conditioning...")
    
    # Create modality adapter
    modality_adapter = ModalityAdapter(num_modalities=4, embed_dim=512)
    
    # Create U-Net
    unet = UNetWithAttention(
        in_channels=1,
        model_channels=64,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        channel_mult=(1, 2, 4, 8),
        num_heads=4,
        use_modality_conditioning=True
    )
    
    print(f"U-Net parameters: {sum(p.numel() for p in unet.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 256, 256)
    t = torch.randint(0, 1000, (batch_size,))
    modality_ids = torch.tensor([0, 1, 2, 3])
    
    # Get modality embeddings
    modality_emb = modality_adapter(modality_ids)
    print(f"Modality embedding shape: {modality_emb.shape}")
    
    # Forward pass
    output = unet(x, t, modality_emb)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test gradient flow
    loss = output.mean()
    loss.backward()
    print("Gradient flow: ✓")
    
    print("\n✓ All tests passed!")
