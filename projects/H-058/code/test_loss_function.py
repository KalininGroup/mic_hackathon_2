"""
Test the multi-objective loss function.
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vantascope.models.dft_autoencoder import create_dft_autoencoder
from vantascope.training.losses import VantaScopeLoss
from vantascope.utils.logging import logger

def test_loss_function():
    """Test the loss function with model outputs."""
    logger.info("🧪 Testing VantaScope Loss Function")
    
    # Create model and loss
    model = create_dft_autoencoder()
    loss_fn = VantaScopeLoss()
    
    # Create dummy data
    batch_size = 2
    dummy_input = torch.randn(batch_size, 1, 512, 512)
    dummy_energy = torch.randn(batch_size, 1) * 1000 - 3000  # DFT-like energies
    
    # Forward pass WITH gradients for backward test
    model.train()  # Enable training mode
    outputs = model(dummy_input)
    
    # Create targets
    targets = {
        'image': dummy_input,  # Self-reconstruction for test
        'energy': dummy_energy
    }
    
    # Compute loss
    losses = loss_fn(outputs, targets)
    
    logger.info("✅ Loss computation successful!")
    logger.info(f"   Total loss: {losses['total_loss'].item():.4f}")
    logger.info(f"   Reconstruction: {losses['reconstruction_loss'].item():.4f}")
    logger.info(f"   Energy: {losses['energy_loss'].item():.4f}")
    logger.info(f"   Fuzzy: {losses['fuzzy_loss'].item():.4f}")
    logger.info(f"   Disentanglement: {losses['disentanglement_loss'].item():.4f}")
    
    # Test backward pass
    total_loss = losses['total_loss']
    total_loss.backward()
    
    logger.info("✅ Backward pass successful!")
    
    # Check that gradients were computed
    grad_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_count += 1
    
    logger.info(f"✅ Gradients computed for {grad_count} parameters")
    logger.info("🎉 Loss function test passed!")

if __name__ == "__main__":
    test_loss_function()
