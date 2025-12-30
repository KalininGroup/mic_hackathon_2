"""
Enhanced Multi-Objective Loss Functions with Full Disentanglement and Fuzzy Logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..utils.logging import logger


class EnhancedVantaScopeLoss(nn.Module):
    """
    Enhanced multi-objective loss with full disentanglement and fuzzy consistency.
    
    Loss Components:
    1. Reconstruction Loss (MSE + Perceptual)
    2. Energy Loss (NLL with uncertainty)
    3. Enhanced Fuzzy Consistency Loss
    4. Advanced Disentanglement Loss (KLD + Orthogonality + Supervision)
    5. Regularization Terms
    """
    
    def __init__(self, 
                 lambda_recon: float = 1.0,
                 lambda_energy: float = 10.0,
                 lambda_fuzzy: float = 5.0,
                 lambda_disentangle: float = 3.0,
                 lambda_regularization: float = 0.1,
                 energy_scale: float = 1000.0):
        super().__init__()
        
        # Loss weights
        self.lambda_recon = lambda_recon
        self.lambda_energy = lambda_energy
        self.lambda_fuzzy = lambda_fuzzy
        self.lambda_disentangle = lambda_disentangle
        self.lambda_regularization = lambda_regularization
        
        # Enhanced loss components
        self.reconstruction_loss = EnhancedReconstructionLoss()
        self.energy_loss = EnhancedEnergyLoss(energy_scale=energy_scale)
        self.fuzzy_loss = EnhancedFuzzyConsistencyLoss()
        self.disentanglement_loss = AdvancedDisentanglementLoss()
        self.regularization_loss = RegularizationLoss()
        
        logger.info(f"🎯 Enhanced VantaScope Loss initialized:")
        logger.info(f"   λ_recon: {lambda_recon}, λ_energy: {lambda_energy}")
        logger.info(f"   λ_fuzzy: {lambda_fuzzy}, λ_disentangle: {lambda_disentangle}")
        logger.info(f"   λ_regularization: {lambda_regularization}")
    
    def forward(self, 
                model_outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute enhanced multi-objective loss."""
        
        device = model_outputs['latent'].device
        
        # 1. Enhanced Reconstruction Loss
        loss_recon = self.reconstruction_loss(
            model_outputs['reconstruction'], 
            targets['image'],
            model_outputs.get('attention_maps')
        )
        
        # 2. Enhanced Energy Loss
        loss_energy = self.energy_loss(
            model_outputs['energy_mean'],
            model_outputs['energy_std'],
            targets['energy']
        )
        
        # 3. Enhanced Fuzzy Consistency Loss
        loss_fuzzy = self.fuzzy_loss(
            model_outputs,
            targets.get('coordinates'),
            targets.get('metadata')
        )
        
        # 4. Advanced Disentanglement Loss
        loss_disentangle = self.disentanglement_loss(
            model_outputs['geometric'],
            model_outputs['topological'],
            model_outputs['disorder'],
            targets.get('coordinates'),
            targets.get('metadata')
        )
        
        # 5. Regularization Loss
        loss_regularization = self.regularization_loss(model_outputs)
        
        # Total weighted loss
        total_loss = (
            self.lambda_recon * loss_recon +
            self.lambda_energy * loss_energy +
            self.lambda_fuzzy * loss_fuzzy +
            self.lambda_disentangle * loss_disentangle +
            self.lambda_regularization * loss_regularization
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': loss_recon,
            'energy_loss': loss_energy,
            'fuzzy_loss': loss_fuzzy,
            'disentanglement_loss': loss_disentangle,
            'regularization_loss': loss_regularization,
            'weighted_recon': self.lambda_recon * loss_recon,
            'weighted_energy': self.lambda_energy * loss_energy,
            'weighted_fuzzy': self.lambda_fuzzy * loss_fuzzy,
            'weighted_disentangle': self.lambda_disentangle * loss_disentangle,
            'weighted_regularization': self.lambda_regularization * loss_regularization
        }


class EnhancedReconstructionLoss(nn.Module):
    """Enhanced reconstruction loss with perceptual and structural components."""
    
    def __init__(self, 
                 mse_weight: float = 0.8,
                 ssim_weight: float = 0.2):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        
        self.mse_loss = nn.MSELoss()
    
    def forward(self, 
                reconstruction: torch.Tensor, 
                target: torch.Tensor,
                attention_maps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute enhanced reconstruction loss."""
        
        # 1. MSE Loss
        mse = self.mse_loss(reconstruction, target)
        
        # 2. Simple SSIM approximation (structural similarity)
        ssim = self._simple_ssim_loss(reconstruction, target)
        
        total_loss = self.mse_weight * mse + self.ssim_weight * ssim
        
        return total_loss
    
    def _simple_ssim_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Simple SSIM approximation using local statistics."""
        
        # Local means
        mu_x = F.avg_pool2d(x, 3, stride=1, padding=1)
        mu_y = F.avg_pool2d(y, 3, stride=1, padding=1)
        
        # Local variances and covariance
        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.avg_pool2d(x * x, 3, stride=1, padding=1) - mu_x_sq
        sigma_y_sq = F.avg_pool2d(y * y, 3, stride=1, padding=1) - mu_y_sq
        sigma_xy = F.avg_pool2d(x * y, 3, stride=1, padding=1) - mu_xy
        
        # SSIM constants
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        # SSIM formula
        numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        
        ssim_map = numerator / (denominator + 1e-8)
        
        # Return 1 - SSIM as loss (lower is better)
        return 1 - ssim_map.mean()


class EnhancedEnergyLoss(nn.Module):
    """Enhanced energy loss with calibration and physics constraints."""
    
    def __init__(self, 
                 energy_scale: float = 1000.0,
                 min_std: float = 0.1):
        super().__init__()
        self.energy_scale = energy_scale
        self.min_std = min_std
    
    def forward(self, 
                pred_mean: torch.Tensor, 
                pred_std: torch.Tensor, 
                target_energy: torch.Tensor) -> torch.Tensor:
        """Enhanced energy loss with uncertainty calibration."""
        
        # Clamp std to prevent numerical issues
        pred_std = torch.clamp(pred_std, min=self.min_std)
        
        # Negative log-likelihood
        squared_error = (target_energy - pred_mean) ** 2
        variance = pred_std ** 2
        nll = 0.5 * torch.log(2 * np.pi * variance) + squared_error / (2 * variance)
        
        return nll.mean()


class EnhancedFuzzyConsistencyLoss(nn.Module):
    """Enhanced fuzzy consistency with spatial and temporal coherence."""
    
    def __init__(self, defect_weight: float = 3.0):
        super().__init__()
        self.defect_weight = defect_weight
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, 
                model_outputs: Dict[str, torch.Tensor],
                coordinates: Optional[List[torch.Tensor]] = None,
                metadata: Optional[List[Dict]] = None) -> torch.Tensor:
        """Enhanced fuzzy consistency loss."""
        
        if 'properties' not in model_outputs:
            return torch.tensor(0.0, device=model_outputs['latent'].device)
        
        properties = model_outputs['properties']
        
        if 'defect_logits' not in properties:
            return torch.tensor(0.0, device=model_outputs['latent'].device)
        
        defect_logits = properties['defect_logits']
        device = defect_logits.device
        
        # Generate enhanced pseudo-labels
        if coordinates is not None and metadata is not None:
            defect_labels = self._generate_enhanced_pseudo_labels(coordinates, metadata, device)
        else:
            return torch.tensor(0.0, device=device)
        
        # Weighted cross-entropy
        ce_losses = self.ce_loss(defect_logits, defect_labels)
        weights = torch.where(defect_labels == 3, 1.0, self.defect_weight)
        weighted_ce = (ce_losses * weights).mean()
        
        return weighted_ce
    
    def _generate_enhanced_pseudo_labels(self, 
                                       coordinates: List[torch.Tensor],
                                       metadata: List[Dict],
                                       device: torch.device) -> torch.Tensor:
        """Generate enhanced pseudo-labels using multiple heuristics."""
        
        labels = []
        
        for coords, meta in zip(coordinates, metadata):
            num_atoms = meta.get('num_atoms', len(coords))
            defect_ratio = meta.get('defect_ratio', 0.0)
            
            # Enhanced heuristics
            expected_atoms = 468
            atom_deficit_ratio = (expected_atoms - num_atoms) / expected_atoms
            
            # Multi-criteria classification
            if defect_ratio > 0.1 or atom_deficit_ratio > 0.1:
                if atom_deficit_ratio > 0.05:
                    label = 1  # vacancy
                else:
                    label = 0  # stone_wales
            elif defect_ratio > 0.02 or atom_deficit_ratio > 0.02:
                label = 2  # grain_boundary
            else:
                label = 3  # pristine
            
            labels.append(label)
        
        return torch.tensor(labels, device=device)


class AdvancedDisentanglementLoss(nn.Module):
    """Advanced disentanglement with supervised learning and mutual information."""
    
    def __init__(self, 
                 kld_weight: float = 1.0,
                 orthogonal_weight: float = 0.8,
                 supervision_weight: float = 2.0):
        super().__init__()
        self.kld_weight = kld_weight
        self.orthogonal_weight = orthogonal_weight
        self.supervision_weight = supervision_weight
    
    def forward(self, 
                geometric: torch.Tensor,
                topological: torch.Tensor, 
                disorder: torch.Tensor,
                coordinates: Optional[List[torch.Tensor]] = None,
                metadata: Optional[List[Dict]] = None) -> torch.Tensor:
        """Advanced disentanglement loss."""
        
        total_loss = torch.tensor(0.0, device=geometric.device)
        
        # 1. KL Divergence - encourage unit Gaussian priors
        if self.kld_weight > 0:
            kld_geo = self._kl_divergence(geometric)
            kld_topo = self._kl_divergence(topological)
            kld_disorder = self._kl_divergence(disorder)
            
            kld_loss = (kld_geo + kld_topo + kld_disorder) / 3.0
            total_loss += self.kld_weight * kld_loss
        
        # 2. Orthogonality - encourage independence between components
        if self.orthogonal_weight > 0:
            ortho_loss = self._orthogonality_loss(geometric, topological, disorder)
            total_loss += self.orthogonal_weight * ortho_loss
        
        # 3. Supervised disentanglement using coordinates
        if self.supervision_weight > 0 and coordinates is not None and metadata is not None:
            supervision_loss = self._supervision_loss(geometric, topological, coordinates, metadata)
            total_loss += self.supervision_weight * supervision_loss
        
        return total_loss
    
    def _kl_divergence(self, latent: torch.Tensor) -> torch.Tensor:
        """KL divergence from unit Gaussian."""
        mu = latent.mean(dim=0)
        var = latent.var(dim=0, unbiased=False)
        
        # KL(N(μ,σ²)||N(0,1)) = 0.5 * (σ² + μ² - 1 - log(σ²))
        kld = 0.5 * (var + mu**2 - 1 - torch.log(var + 1e-8))
        return kld.mean()
    
    def _orthogonality_loss(self, 
                           geometric: torch.Tensor,
                           topological: torch.Tensor,
                           disorder: torch.Tensor) -> torch.Tensor:
        """Encourage orthogonality between latent components."""
        
        # Normalize features
        geo_norm = F.normalize(geometric, dim=1)
        topo_norm = F.normalize(topological, dim=1)
        disorder_norm = F.normalize(disorder, dim=1)
        
        # Compute cross-correlations (should be close to zero)
        corr_geo_topo = torch.abs(torch.mm(geo_norm.T, topo_norm)).mean()
        corr_geo_disorder = torch.abs(torch.mm(geo_norm.T, disorder_norm)).mean()
        corr_topo_disorder = torch.abs(torch.mm(topo_norm.T, disorder_norm)).mean()
        
        return (corr_geo_topo + corr_geo_disorder + corr_topo_disorder) / 3.0
    
    def _supervision_loss(self, 
                         geometric: torch.Tensor,
                         topological: torch.Tensor,
                         coordinates: List[torch.Tensor],
                         metadata: List[Dict]) -> torch.Tensor:
        """Supervised disentanglement using ground truth coordinates."""
        
        batch_size = len(coordinates)
        geometric_targets = []
        topological_targets = []
        
        for i, (coords, meta) in enumerate(zip(coordinates, metadata)):
            # Extract geometric properties from coordinates
            if len(coords) > 1:
                # Compute average bond length (geometric property)
                dists = torch.cdist(coords[:, :2], coords[:, :2])
                dists = dists + torch.eye(len(coords), device=coords.device) * 1e6
                min_dists = dists.min(dim=1)[0]
                avg_bond_length = min_dists.mean()
                geometric_target = avg_bond_length / 1.42  # Normalize by ideal graphene bond length
            else:
                geometric_target = torch.tensor(1.0, device=coords.device)
            
            # Extract topological properties (defect density)
            num_atoms = meta.get('num_atoms', len(coords))
            expected_atoms = 468
            defect_density = max(0, (expected_atoms - num_atoms) / expected_atoms)
            topological_target = torch.tensor(defect_density, device=coords.device)
            
            geometric_targets.append(geometric_target)
            topological_targets.append(topological_target)
        
        # Convert to tensors
        geometric_targets = torch.stack(geometric_targets).unsqueeze(1)  # [B, 1]
        topological_targets = torch.stack(topological_targets).unsqueeze(1)  # [B, 1]
        
        # Supervision loss: encourage geometric features to predict geometric properties
        geo_pred = geometric[:, 0:1]  # Use first dimension as bond length predictor
        topo_pred = topological[:, 0:1]  # Use first dimension as defect density predictor
        
        geo_loss = F.mse_loss(geo_pred, geometric_targets)
        topo_loss = F.mse_loss(topo_pred, topological_targets)
        
        return (geo_loss + topo_loss) / 2.0


class RegularizationLoss(nn.Module):
    """Additional regularization terms for model stability."""
    
    def __init__(self, 
                 sparsity_weight: float = 0.1,
                 smoothness_weight: float = 0.1):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.smoothness_weight = smoothness_weight
    
    def forward(self, model_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute regularization loss."""
        
        total_loss = torch.tensor(0.0, device=model_outputs['latent'].device)
        
        # 1. Sparsity regularization on latent features
        if self.sparsity_weight > 0:
            latent = model_outputs['latent']
            sparsity_loss = torch.abs(latent).mean()
            total_loss += self.sparsity_weight * sparsity_loss
        
        # 2. Smoothness regularization on reconstruction
        if self.smoothness_weight > 0 and 'reconstruction' in model_outputs:
            reconstruction = model_outputs['reconstruction']
            smoothness_loss = self._total_variation_loss(reconstruction)
            total_loss += self.smoothness_weight * smoothness_loss
        
        return total_loss
    
    def _total_variation_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Total variation loss for smoothness."""
        
        # Compute gradients
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        
        # Total variation
        tv_loss = diff_h.mean() + diff_w.mean()
        
        return tv_loss


# Factory functions
def create_enhanced_loss(**kwargs) -> EnhancedVantaScopeLoss:
    """Factory function to create enhanced loss."""
    return EnhancedVantaScopeLoss(**kwargs)


def create_basic_loss(**kwargs) -> EnhancedVantaScopeLoss:
    """Factory function to create basic loss with reduced complexity."""
    return EnhancedVantaScopeLoss(
        lambda_fuzzy=1.0,  # Reduced
        lambda_disentangle=1.0,  # Reduced
        lambda_regularization=0.05,  # Reduced
        **kwargs
    )


class AdvancedDisentanglementLoss(nn.Module):
    """Advanced disentanglement with supervised learning and mutual information."""
    
    def __init__(self, 
                 kld_weight: float = 1.0,
                 orthogonal_weight: float = 0.8,
                 supervision_weight: float = 2.0):
        super().__init__()
        self.kld_weight = kld_weight
        self.orthogonal_weight = orthogonal_weight
        self.supervision_weight = supervision_weight
    
    def forward(self, 
                geometric: torch.Tensor,
                topological: torch.Tensor, 
                disorder: torch.Tensor,
                coordinates: Optional[List[torch.Tensor]] = None,
                metadata: Optional[List[Dict]] = None) -> torch.Tensor:
        """Advanced disentanglement loss computation."""
        
        total_loss = torch.tensor(0.0, device=geometric.device)
        
        # 1. KL Divergence - encourage unit Gaussian priors for each component
        if self.kld_weight > 0:
            kld_geometric = self._compute_kl_divergence(geometric)
            kld_topological = self._compute_kl_divergence(topological)
            kld_disorder = self._compute_kl_divergence(disorder)
            
            kld_loss = (kld_geometric + kld_topological + kld_disorder) / 3.0
            total_loss += self.kld_weight * kld_loss
        
        # 2. Orthogonality constraint - encourage independence between components
        if self.orthogonal_weight > 0:
            orthogonality_loss = self._compute_orthogonality_loss(geometric, topological, disorder)
            total_loss += self.orthogonal_weight * orthogonality_loss
        
        # 3. Supervised disentanglement using ground truth atomic coordinates
        if self.supervision_weight > 0 and coordinates is not None and metadata is not None:
            supervision_loss = self._compute_supervision_loss(geometric, topological, coordinates, metadata)
            total_loss += self.supervision_weight * supervision_loss
        
        return total_loss
    
    def _compute_kl_divergence(self, latent_features: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence from unit Gaussian distribution."""
        batch_mean = latent_features.mean(dim=0)
        batch_variance = latent_features.var(dim=0, unbiased=False)
        
        # KL divergence formula: KL(N(μ,σ²)||N(0,1)) = 0.5 * (σ² + μ² - 1 - log(σ²))
        kl_divergence = 0.5 * (batch_variance + batch_mean**2 - 1 - torch.log(batch_variance + 1e-8))
        return kl_divergence.mean()
    
    def _compute_orthogonality_loss(self, 
                                   geometric: torch.Tensor,
                                   topological: torch.Tensor,
                                   disorder: torch.Tensor) -> torch.Tensor:
        """Encourage orthogonality between different latent components."""
        
        # Normalize feature vectors to unit length
        geometric_normalized = F.normalize(geometric, dim=1)
        topological_normalized = F.normalize(topological, dim=1)
        disorder_normalized = F.normalize(disorder, dim=1)
        
        # Compute cross-correlations between different components (should be close to zero)
        correlation_geometric_topological = torch.abs(torch.mm(geometric_normalized.T, topological_normalized)).mean()
        correlation_geometric_disorder = torch.abs(torch.mm(geometric_normalized.T, disorder_normalized)).mean()
        correlation_topological_disorder = torch.abs(torch.mm(topological_normalized.T, disorder_normalized)).mean()
        
        average_correlation = (correlation_geometric_topological + correlation_geometric_disorder + correlation_topological_disorder) / 3.0
        return average_correlation
    
    def _compute_supervision_loss(self, 
                                 geometric: torch.Tensor,
                                 topological: torch.Tensor,
                                 coordinates: List[torch.Tensor],
                                 metadata: List[Dict]) -> torch.Tensor:
        """Supervised disentanglement using ground truth atomic coordinates."""
        
        batch_size = len(coordinates)
        geometric_property_targets = []
        topological_property_targets = []
        
        for batch_index, (atomic_coordinates, sample_metadata) in enumerate(zip(coordinates, metadata)):
            # Extract geometric properties from atomic coordinates
            if len(atomic_coordinates) > 1:
                # Compute average bond length as geometric property
                pairwise_distances = torch.cdist(atomic_coordinates[:, :2], atomic_coordinates[:, :2])
                # Mask diagonal elements to avoid zero distances
                pairwise_distances = pairwise_distances + torch.eye(len(atomic_coordinates), device=atomic_coordinates.device) * 1e6
                minimum_distances = pairwise_distances.min(dim=1)[0]
                average_bond_length = minimum_distances.mean()
                # Normalize by ideal graphene bond length (1.42 Angstroms)
                normalized_geometric_target = average_bond_length / 1.42
            else:
                normalized_geometric_target = torch.tensor(1.0, device=atomic_coordinates.device)
            
            # Extract topological properties (defect density from atom count)
            actual_atom_count = sample_metadata.get('num_atoms', len(atomic_coordinates))
            expected_atom_count = 468  # Expected atoms for 3.5nm x 3.5nm graphene
            defect_density_ratio = max(0, (expected_atom_count - actual_atom_count) / expected_atom_count)
            topological_property_target = torch.tensor(defect_density_ratio, device=atomic_coordinates.device)
            
            geometric_property_targets.append(normalized_geometric_target)
            topological_property_targets.append(topological_property_target)
        
        # Convert lists to tensors
        geometric_targets_tensor = torch.stack(geometric_property_targets).unsqueeze(1)  # Shape: [batch_size, 1]
        topological_targets_tensor = torch.stack(topological_property_targets).unsqueeze(1)  # Shape: [batch_size, 1]
        
        # Use first dimension of each component as predictor for respective property
        geometric_prediction = geometric[:, 0:1]  # Use first dimension as bond length predictor
        topological_prediction = topological[:, 0:1]  # Use first dimension as defect density predictor
        
        # Compute mean squared error between predictions and targets
        geometric_supervision_loss = F.mse_loss(geometric_prediction, geometric_targets_tensor)
        topological_supervision_loss = F.mse_loss(topological_prediction, topological_targets_tensor)
        
        combined_supervision_loss = (geometric_supervision_loss + topological_supervision_loss) / 2.0
        return combined_supervision_loss


class RegularizationLoss(nn.Module):
    """Additional regularization terms for model stability and generalization."""
    
    def __init__(self, 
                 sparsity_weight: float = 0.1,
                 smoothness_weight: float = 0.1):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.smoothness_weight = smoothness_weight
    
    def forward(self, model_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute regularization loss terms."""
        
        total_regularization_loss = torch.tensor(0.0, device=model_outputs['latent'].device)
        
        # 1. Sparsity regularization on latent feature representations
        if self.sparsity_weight > 0:
            latent_representations = model_outputs['latent']
            sparsity_loss = torch.abs(latent_representations).mean()
            total_regularization_loss += self.sparsity_weight * sparsity_loss
        
        # 2. Smoothness regularization on reconstruction output
        if self.smoothness_weight > 0 and 'reconstruction' in model_outputs:
            reconstruction_output = model_outputs['reconstruction']
            smoothness_loss = self._compute_total_variation_loss(reconstruction_output)
            total_regularization_loss += self.smoothness_weight * smoothness_loss
        
        return total_regularization_loss
    
    def _compute_total_variation_loss(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Compute total variation loss for image smoothness."""
        
        # Compute horizontal gradients (differences between adjacent pixels)
        horizontal_differences = torch.abs(image_tensor[:, :, 1:, :] - image_tensor[:, :, :-1, :])
        # Compute vertical gradients (differences between adjacent pixels)
        vertical_differences = torch.abs(image_tensor[:, :, :, 1:] - image_tensor[:, :, :, :-1])
        
        # Total variation is sum of horizontal and vertical gradients
        total_variation_loss = horizontal_differences.mean() + vertical_differences.mean()
        
        return total_variation_loss


# Factory functions for creating loss instances
def create_enhanced_loss(**kwargs) -> EnhancedVantaScopeLoss:
    """Factory function to create enhanced VantaScope loss with full complexity."""
    return EnhancedVantaScopeLoss(**kwargs)


def create_basic_loss(**kwargs) -> EnhancedVantaScopeLoss:
    """Factory function to create basic VantaScope loss with reduced complexity for initial training."""
    return EnhancedVantaScopeLoss(
        lambda_fuzzy=1.0,  # Reduced from default 5.0
        lambda_disentangle=1.0,  # Reduced from default 3.0
        lambda_regularization=0.05,  # Reduced from default 0.1
        **kwargs
    )
