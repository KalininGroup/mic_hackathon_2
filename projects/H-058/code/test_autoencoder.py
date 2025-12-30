"""
Test complete autoencoder with real data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vantascope.models.autoencoder import create_autoencoder
from vantascope.data.factory import VantaScopeDataFactory
from vantascope.config import VantaScopeConfig
from vantascope.utils.helpers import set_seed, get_device
from vantascope.utils.logging import logger
import torch
import matplotlib.pyplot as plt

def test_autoencoder():
    """Test autoencoder reconstruction."""
    logger.info("🔄 Testing complete autoencoder")
    
    set_seed(42)
    device = get_device()
    
    # Load real data
    config = VantaScopeConfig.from_yaml("config/datasets_real.yaml")
    dataset = VantaScopeDataFactory.create_dataset(config.datasets["graphene_stem"])
    dataloader = VantaScopeDataFactory.create_dataloader(dataset, batch_size=2, split="train")
    
    # Create autoencoder
    autoencoder = create_autoencoder().to(device)
    autoencoder.eval()
    
    # Test forward pass
    batch = next(iter(dataloader))
    images = batch['image'].to(device)
    
    logger.info(f"📊 Input images: {images.shape}")
    
    with torch.no_grad():
        outputs = autoencoder(images)
    
    # Log outputs
    for key, tensor in outputs.items():
        if tensor is not None:
            logger.info(f"✅ {key}: {tensor.shape}")
    
    # Check reconstruction quality
    reconstruction = outputs['reconstruction']
    mse_loss = torch.nn.functional.mse_loss(reconstruction, images)
    logger.info(f"📊 Reconstruction MSE: {mse_loss.item():.6f}")
    
    # Visualize original vs reconstruction
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
    for i in range(2):
        # Original
        axes[0, i].imshow(images[i, 0].cpu().numpy(), cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstruction
        axes[1, i].imshow(reconstruction[i, 0].cpu().numpy(), cmap='gray')
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('data/autoencoder_test.png', dpi=150)
    logger.info("💾 Saved reconstruction comparison to data/autoencoder_test.png")
    
    logger.info("🎉 Autoencoder test complete!")

if __name__ == "__main__":
    test_autoencoder()
