"""
Test Complete Training Pipeline: DFT Dataset → Model → Loss → Optimization
"""

import torch
import torch.optim as optim
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vantascope.models.dft_autoencoder import create_dft_autoencoder
from vantascope.training.losses import VantaScopeLoss
from vantascope.data.dft_dataset import create_dft_dataloader
from vantascope.utils.logging import logger

def test_complete_pipeline():
    """Test the complete training pipeline."""
    logger.info("🚀 Testing Complete Training Pipeline")
    
    # 1. Create model
    model = create_dft_autoencoder()
    model.train()
    
    # 2. Create loss function
    loss_fn = VantaScopeLoss(
        lambda_recon=1.0,
        lambda_energy=10.0,
        lambda_fuzzy=5.0,
        lambda_disentangle=2.0
    )
    
    # 3. Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # 4. Create dataloader
    dataloader = create_dft_dataloader(
        tar_path="dummy_dataset.tar.gz",
        batch_size=2,  # Small batch for test
        max_samples=6,
        num_workers=0
    )
    
    logger.info("✅ All components initialized")
    
    # 5. Test training loop
    logger.info("🏋️ Testing training loop...")
    
    for epoch in range(2):  # Test 2 epochs
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            # Forward pass
            outputs = model(batch['image'])
            
            # Prepare targets
            targets = {
                'image': batch['image'],
                'energy': batch['energy'],
                'coordinates': batch['coordinates']
            }
            
            # Compute loss
            losses = loss_fn(outputs, targets)
            total_loss = losses['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_losses.append(total_loss.item())
            
            logger.info(f"   Epoch {epoch}, Batch {batch_idx}:")
            logger.info(f"     Total Loss: {total_loss.item():.2f}")
            logger.info(f"     Reconstruction: {losses['reconstruction_loss'].item():.4f}")
            logger.info(f"     Energy: {losses['energy_loss'].item():.2f}")
            
            if batch_idx >= 1:  # Test first 2 batches per epoch
                break
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(f"   Epoch {epoch} Average Loss: {avg_loss:.2f}")
    
    logger.info("🎉 Complete training pipeline test successful!")
    
    # Test model outputs
    logger.info("🔍 Testing model outputs...")
    
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(dataloader))
        outputs = model(sample_batch['image'])
        
        logger.info(f"   Input shape: {sample_batch['image'].shape}")
        logger.info(f"   Reconstruction shape: {outputs['reconstruction'].shape}")
        logger.info(f"   Energy prediction: {outputs['energy_mean'].squeeze().tolist()}")
        logger.info(f"   Energy uncertainty: {outputs['energy_std'].squeeze().tolist()}")
        
        # Test properties
        props = outputs['properties']
        logger.info(f"   Crystallinity: {props['crystallinity'].squeeze().tolist()}")
        logger.info(f"   Defect density: {props['defect_density'].squeeze().tolist()}")
        logger.info(f"   Defect probabilities: {props['defect_probs'].squeeze().tolist()}")
    
    return model, loss_fn, optimizer, dataloader

if __name__ == "__main__":
    test_complete_pipeline()
