"""
Integration test: Real data -> DINOv2 backbone
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vantascope.data.factory import VantaScopeDataFactory
from vantascope.models.dinov2_backbone import create_dinov2_backbone
from vantascope.config import VantaScopeConfig
from vantascope.utils.helpers import set_seed, get_device
from vantascope.utils.logging import logger
import torch

def test_real_data_through_model():
    """Test real microscopy data through DINOv2."""
    logger.info("🔗 Testing real data -> DINOv2 pipeline")
    
    set_seed(42)
    device = get_device()
    
    # Load real config
    config = VantaScopeConfig.from_yaml("config/datasets_real.yaml")
    
    # Test graphene data
    graphene_dataset = VantaScopeDataFactory.create_dataset(config.datasets["graphene_stem"])
    graphene_loader = VantaScopeDataFactory.create_dataloader(graphene_dataset, batch_size=2, split="train")
    
    # Create DINOv2 model
    model = create_dinov2_backbone().to(device)
    model.eval()
    
    # Forward pass with real graphene data
    batch = next(iter(graphene_loader))
    images = batch['image'].to(device)
    
    logger.info(f"📊 Real graphene batch: {images.shape}")
    logger.info(f"   Value range: [{images.min():.3f}, {images.max():.3f}]")
    
    with torch.no_grad():
        outputs = model(images)
    
    # Log all outputs
    for key, tensor in outputs.items():
        if tensor is not None:
            logger.info(f"✅ {key}: {tensor.shape}")
            if key == 'global_features':
                logger.info(f"   Feature range: [{tensor.min():.3f}, {tensor.max():.3f}]")
    
    # Test PZT data
    logger.info("\n🔬 Testing PZT data...")
    pzt_dataset = VantaScopeDataFactory.create_dataset(config.datasets["pzt_ferroelectric"])
    pzt_loader = VantaScopeDataFactory.create_dataloader(pzt_dataset, batch_size=2, split="train")
    
    pzt_batch = next(iter(pzt_loader))
    pzt_images = pzt_batch['image'].to(device)
    
    logger.info(f"📊 Real PZT batch: {pzt_images.shape}")
    
    with torch.no_grad():
        pzt_outputs = model(pzt_images)
    
    logger.info(f"✅ PZT features: {pzt_outputs['global_features'].shape}")
    
    logger.info("🎉 Real data integration test passed!")

if __name__ == "__main__":
    test_real_data_through_model()
