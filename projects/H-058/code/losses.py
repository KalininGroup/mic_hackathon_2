"""
Multi-Objective Loss Functions for VantaScope DFT Training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np

from ..utils.logging import logger


class VantaScopeLoss(nn.Module):
    """Multi-objective loss function for VantaScope training."""
    
    def __init__(self, 
                 lambda_recon: float = 1.0,
                 lambda_energy: float = 10.0,
                 lambda_fuzzy: float = 5.0,
                 lambda_disentangle: float = 2.0):
        super().__init__()
        
        self.lambda_recon = lambda_recon
        self.lambda_energy = lambda_energy
        self.lambda_fuzzy = lambda_fuzzy
        self.lambda_disentangle = lambda_disentangle
        
        logger.info(f"🎯 VantaScope Loss initialized")
    
    def forward(self, model_outputs: Dict, targets: Dict) -> Dict:
        """Compute multi-objective loss."""
        
        # 1. Reconstruction Loss
        recon_loss = F.mse_loss(model_outputs['reconstruction'], targets['image'])
        
        # 2. Energy Loss
        energy_loss = F.mse_loss(model_outputs['energy_mean'], targets['energy'])
        
        # 3. Simple fuzzy loss (placeholder)
        fuzzy_loss = torch.tensor(0.0, device=recon_loss.device)
        
        # 4. Simple disentanglement loss (placeholder)
        disentangle_loss = torch.tensor(0.0, device=recon_loss.device)
        
        # Total loss
        total_loss = (
            self.lambda_recon * recon_loss +
            self.lambda_energy * energy_loss +
            self.lambda_fuzzy * fuzzy_loss +
            self.lambda_disentangle * disentangle_loss
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'energy_loss': energy_loss,
            'fuzzy_loss': fuzzy_loss,
            'disentanglement_loss': disentangle_loss
        }
