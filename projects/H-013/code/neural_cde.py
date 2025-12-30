"""
Neural Controlled Differential Equation for 4D-STEM Reconstruction
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from typing import Optional, Tuple, Dict
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ModelConfig
from models.encoder_decoder import UNetEncoder, UNetDecoder
from models.physics_module import ScatteringPhysicsModule


class CDEFunction(nn.Module):
    """
    Neural network that defines the ODE: dz/dt = f(z, t, physics_params)
    """
    
    def __init__(self, config: ModelConfig, physics_module: ScatteringPhysicsModule):
        super().__init__()
        self.config = config
        self.physics_module = physics_module
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        
        # Calculate physics parameter dimension
        # physics_dim + physics_dim + 64 (time) + 1 (damping) + 1 (debye_waller)
        physics_total_dim = config.physics_dim * 2 + 64 + 2
        
        # Main CDE function network
        self.cde_net = nn.Sequential(
            nn.Linear(config.latent_dim + physics_total_dim + 64, config.hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[1], config.hidden_dims[2]),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[2], config.hidden_dims[3]),
            nn.SiLU(),
            nn.Linear(config.hidden_dims[3], config.latent_dim)
        )
    
    def forward(self, t, z):
        """
        Args:
            t: scalar time
            z: [B, latent_dim]
            
        Returns:
            dz/dt: [B, latent_dim]
        """
        B = z.shape[0]
        
        # Handle time
        if isinstance(t, torch.Tensor):
            if t.dim() == 0:
                t_expanded = t.unsqueeze(0).expand(B, 1)
            else:
                t_expanded = t.view(-1, 1)
        else:
            t_expanded = torch.tensor([[t]], device=z.device).expand(B, 1)
        
        # Time embedding
        time_feat = self.time_embed(t_expanded)
        
        # Get physics parameters at time t
        physics_params = self.physics_module(z, t_expanded.squeeze(-1))
        
        # Concatenate all features
        combined = torch.cat([z, physics_params, time_feat], dim=-1)
        
        # Compute derivative
        dzdt = self.cde_net(combined)
        
        return dzdt


class NeuralCDE4DSTEM(nn.Module):
    """
    Complete Neural CDE model for sparse 4D-STEM reconstruction
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Components
        self.encoder = UNetEncoder(config)
        self.physics_module = ScatteringPhysicsModule(config)
        self.decoder = UNetDecoder(config)
        
        # CDE function
        self.cde_func = CDEFunction(config, self.physics_module)
    
    def forward(
        self, 
        sparse_sequence: torch.Tensor, 
        sparse_times: torch.Tensor,
        query_times: Optional[torch.Tensor] = None,
        return_trajectory: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            sparse_sequence: [B, K, 1, H, W] - Sparse measurements
            sparse_times: [B, K] - Times of sparse measurements
            query_times: [T_query] - Times to query reconstruction (optional)
            return_trajectory: Whether to return full latent trajectory
        
        Returns:
            Dictionary containing:
                - reconstruction: [B, T_query, 1, H, W]
                - latent_trajectory: [B, T_query, latent_dim] (if return_trajectory=True)
                - physics_params: Physics parameters at query times
        """
        B, K, C, H, W = sparse_sequence.shape
        device = sparse_sequence.device
        
        # Encode sparse measurements to latent space
        latent_sparse, skip_connections = self.encoder(sparse_sequence)
        # latent_sparse: [B, K, latent_dim]
        
        # Setup query times
        if query_times is None:
            # Default: reconstruct at uniform time points
            query_times = torch.linspace(0, 1, self.config.max_time_points, device=device)
        
        # Solve ODE for each batch element
        # We'll use the sparse latent codes and times to guide the ODE
        all_latents = []
        
        for b in range(B):
            # Get sparse latent codes and times for this batch
            z_sparse_b = latent_sparse[b]  # [K, latent_dim]
            t_sparse_b = sparse_times[b]  # [K]
            
            # Solve ODE from first sparse point
            z0 = z_sparse_b[0:1]  # [1, latent_dim]
            
            # Create time points for ODE solver
            # We need to include sparse measurement times to enforce constraints
            all_times = torch.cat([t_sparse_b, query_times])
            all_times = torch.unique(all_times, sorted=True)
            
            # Solve ODE
            z_trajectory = odeint(
                self.cde_func,
                z0,
                all_times,
                method=self.config.ode_solver,
                rtol=self.config.ode_rtol,
                atol=self.config.ode_atol
            )
            # z_trajectory: [T_all, 1, latent_dim]
            
            # Extract latent codes at query times
            query_indices = torch.searchsorted(all_times, query_times)
            z_query = z_trajectory[query_indices].squeeze(1)  # [T_query, latent_dim]
            
            all_latents.append(z_query)
        
        # Stack all batches
        z_continuous = torch.stack(all_latents, dim=0)  # [B, T_query, latent_dim]
        
        # Get physics parameters at query times
        physics_params = self.physics_module(z_continuous, query_times.unsqueeze(0).expand(B, -1))
        
        # Decode to full 4D-STEM sequence
        reconstructed = self.decoder(z_continuous, skip_connections)
        # reconstructed: [B, T_query, 1, H, W]
        
        result = {
            'reconstruction': reconstructed,
            'physics_params': physics_params
        }
        
        if return_trajectory:
            result['latent_trajectory'] = z_continuous
        
        return result
    
    def compute_loss(self, predictions, targets, sparse_sequence, sparse_times):
        """
        Compute total loss including reconstruction and physics constraints
        
        Args:
            predictions: Dictionary from forward pass
            targets: [B, T, 1, H, W] - Ground truth sequence
            sparse_sequence: [B, K, 1, H, W] - Sparse input
            sparse_times: [B, K] - Sparse times
            
        Returns:
            total_loss: Scalar loss
            loss_dict: Dictionary of individual losses
        """
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(predictions['reconstruction'], targets)
        
        # Physics constraint losses
        physics_losses = self.physics_module.compute_physics_loss(predictions['reconstruction'])
        physics_total = sum(physics_losses.values())
        
        # Temporal smoothness loss
        temporal_loss = self._temporal_consistency_loss(predictions['reconstruction'])
        
        # Data consistency loss (reconstruction should match sparse observations)
        data_consistency_loss = self._data_consistency_loss(
            predictions['reconstruction'], 
            sparse_sequence, 
            sparse_times,
            targets.shape[1]
        )
        
        # Total loss
        total_loss = (
            recon_loss + 
            self.config.physics_weight * physics_total +
            self.config.temporal_smoothness_weight * temporal_loss +
            0.1 * data_consistency_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'reconstruction': recon_loss.item(),
            'physics': physics_total.item(),
            'temporal': temporal_loss.item(),
            'data_consistency': data_consistency_loss.item()
        }
        loss_dict.update({k: v.item() for k, v in physics_losses.items()})
        
        return total_loss, loss_dict
    
    def _temporal_consistency_loss(self, sequence):
        """Encourage smooth evolution over time"""
        # sequence: [B, T, C, H, W]
        time_diff = sequence[:, 1:] - sequence[:, :-1]
        return torch.mean(time_diff ** 2)
    
    def _data_consistency_loss(self, reconstruction, sparse_sequence, sparse_times, total_time_points):
        """
        Ensure reconstruction matches sparse observations at measurement times
        
        Args:
            reconstruction: [B, T_query, 1, H, W]
            sparse_sequence: [B, K, 1, H, W]
            sparse_times: [B, K]
            total_time_points: Number of time points in reconstruction
        """
        B, K = sparse_times.shape
        T_query = reconstruction.shape[1]
        
        # Map sparse times to indices in reconstruction
        # Assume query times are uniformly spaced from 0 to 1
        indices = (sparse_times * (T_query - 1)).long().clamp(0, T_query - 1)
        
        # Extract reconstructed values at sparse times
        loss = 0
        for b in range(B):
            for k in range(K):
                idx = indices[b, k]
                recon_at_sparse = reconstruction[b, idx]
                sparse_val = sparse_sequence[b, k]
                loss += torch.mean((recon_at_sparse - sparse_val) ** 2)
        
        return loss / (B * K)


if __name__ == "__main__":
    # Test Neural CDE model
    from config import ModelConfig
    
    config = ModelConfig()
    config.image_size = (64, 64)
    config.max_time_points = 20
    config.sparse_sampling_rate = 0.2
    config.encoder_channels = [64, 128, 256]
    config.decoder_channels = [256, 128, 64]
    
    print("Testing Neural CDE 4D-STEM Model...")
    
    # Create model
    model = NeuralCDE4DSTEM(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create test data
    B, K = 2, 4  # 2 batches, 4 sparse measurements each
    H, W = 64, 64
    
    sparse_sequence = torch.rand(B, K, 1, H, W)
    sparse_times = torch.sort(torch.rand(B, K))[0]  # Sorted times in [0, 1]
    
    print(f"\nInput shapes:")
    print(f"  Sparse sequence: {sparse_sequence.shape}")
    print(f"  Sparse times: {sparse_times.shape}")
    
    # Forward pass
    print("\nRunning forward pass...")
    query_times = torch.linspace(0, 1, 10)
    
    with torch.no_grad():
        predictions = model(sparse_sequence, sparse_times, query_times, return_trajectory=True)
    
    print(f"\nOutput shapes:")
    print(f"  Reconstruction: {predictions['reconstruction'].shape}")
    print(f"  Latent trajectory: {predictions['latent_trajectory'].shape}")
    print(f"  Physics params: {predictions['physics_params'].shape}")
    
    # Test loss computation
    print("\nTesting loss computation...")
    target = torch.rand(B, 10, 1, H, W)
    
    model.train()
    predictions = model(sparse_sequence, sparse_times, query_times)
    loss, loss_dict = model.compute_loss(predictions, target, sparse_sequence, sparse_times)
    
    print(f"\nLosses:")
    for name, value in loss_dict.items():
        print(f"  {name}: {value:.6f}")
    
    print("\nNeural CDE model test passed!")
