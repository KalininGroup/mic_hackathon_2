"""
Test Enhanced Loss Functions
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vantascope.models.dft_autoencoder import create_dft_autoencoder
from vantascope.training.enhanced_losses import create_enhanced_loss, create_basic_loss
from vantascope.data.dft_dataset import create_dft_dataloader
from vantascope.utils.logging import logger

def test_enhanced_losses():
    """Test the enhanced loss functions."""
    logger.info("🧪 Testing Enhanced Loss Functions")
    
    # Create model
    model = create_dft_autoencoder()
    model.train()
    
    # Create enhanced loss function
    enhanced_loss_fn = create_enhanced_loss(
        lambda_recon=1.0,
        lambda_energy=10.0,
        lambda_fuzzy=5.0,
        lambda_disentangle=3.0,
        lambda_regularization=0.1
    )
    
    # Create basic loss function for comparison
    basic_loss_fn = create_basic_loss(
        lambda_recon=1.0,
        lambda_energy=10.0
    )
    
    # Create test data
    dataloader = create_dft_dataloader(
        tar_path="dummy_dataset.tar.gz",
        batch_size=2,
        max_samples=4,
        num_workers=0
    )
    
    logger.info("✅ All components initialized")
    
    # Test with sample batch
    sample_batch = next(iter(dataloader))
    
    # Forward pass
    outputs = model(sample_batch['image'])
    
    # Prepare targets
    targets = {
        'image': sample_batch['image'],
        'energy': sample_batch['energy'],
        'coordinates': sample_batch['coordinates'],
        'metadata': sample_batch['metadata']
    }
    
    # Test enhanced loss
    logger.info("🔬 Testing Enhanced Loss Function...")
    enhanced_losses = enhanced_loss_fn(outputs, targets)
    
    logger.info("   Enhanced Loss Components:")
    logger.info(f"     Total Loss: {enhanced_losses['total_loss'].item():.2f}")
    logger.info(f"     Reconstruction: {enhanced_losses['reconstruction_loss'].item():.4f}")
    logger.info(f"     Energy: {enhanced_losses['energy_loss'].item():.2f}")
    logger.info(f"     Fuzzy Consistency: {enhanced_losses['fuzzy_loss'].item():.4f}")
    logger.info(f"     Disentanglement: {enhanced_losses['disentanglement_loss'].item():.4f}")
    logger.info(f"     Regularization: {enhanced_losses['regularization_loss'].item():.4f}")
    
    # Test basic loss
    logger.info("🔬 Testing Basic Loss Function...")
    basic_losses = basic_loss_fn(outputs, targets)
    
    logger.info("   Basic Loss Components:")
    logger.info(f"     Total Loss: {basic_losses['total_loss'].item():.2f}")
    logger.info(f"     Reconstruction: {basic_losses['reconstruction_loss'].item():.4f}")
    logger.info(f"     Energy: {basic_losses['energy_loss'].item():.2f}")
    logger.info(f"     Fuzzy Consistency: {basic_losses['fuzzy_loss'].item():.4f}")
    logger.info(f"     Disentanglement: {basic_losses['disentanglement_loss'].item():.4f}")
    logger.info(f"     Regularization: {basic_losses['regularization_loss'].item():.4f}")
    
    # Test backward pass
    logger.info("🔬 Testing Backward Pass...")
    
    enhanced_total_loss = enhanced_losses['total_loss']
    enhanced_total_loss.backward()
    
    # Check gradients
    gradient_count = 0
    total_parameters = 0
    
    for name, parameter in model.named_parameters():
        total_parameters += 1
        if parameter.grad is not None:
            gradient_count += 1
    
    logger.info(f"   Gradients computed: {gradient_count}/{total_parameters} parameters")
    
    # Test individual loss components
    logger.info("🔬 Testing Individual Loss Components...")
    
    # Test SSIM loss component
    reconstruction = outputs['reconstruction']
    target_image = targets['image']
    ssim_component = enhanced_loss_fn.reconstruction_loss._simple_ssim_loss(reconstruction, target_image)
    logger.info(f"   SSIM Loss: {ssim_component.item():.4f}")
    
    # Test KL divergence components
    geometric = outputs['geometric']
    topological = outputs['topological']
    disorder = outputs['disorder']
    
    kld_geo = enhanced_loss_fn.disentanglement_loss._compute_kl_divergence(geometric)
    kld_topo = enhanced_loss_fn.disentanglement_loss._compute_kl_divergence(topological)
    kld_disorder = enhanced_loss_fn.disentanglement_loss._compute_kl_divergence(disorder)
    
    logger.info(f"   KL Divergences - Geometric: {kld_geo.item():.4f}, Topological: {kld_topo.item():.4f}, Disorder: {kld_disorder.item():.4f}")
    
    # Test orthogonality loss
    ortho_loss = enhanced_loss_fn.disentanglement_loss._compute_orthogonality_loss(geometric, topological, disorder)
    logger.info(f"   Orthogonality Loss: {ortho_loss.item():.4f}")
    
    # Test supervision loss
    supervision_loss = enhanced_loss_fn.disentanglement_loss._compute_supervision_loss(
        geometric, topological, targets['coordinates'], targets['metadata']
    )
    logger.info(f"   Supervision Loss: {supervision_loss.item():.4f}")
    
    logger.info("🎉 Enhanced Loss Functions test completed successfully!")
    
    return enhanced_loss_fn, basic_loss_fn, enhanced_losses, basic_losses

if __name__ == "__main__":
    test_enhanced_losses()
