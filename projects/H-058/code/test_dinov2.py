"""
Test the DINOv2 backbone with dummy microscopy data.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vantascope.models.dinov2_backbone import create_dinov2_backbone
from vantascope.utils.helpers import set_seed, get_device
from vantascope.utils.logging import logger

def test_dinov2():
    """Test DINOv2 backbone."""
    logger.info("🧪 Testing DINOv2 backbone")
    
    # Setup
    set_seed(42)
    device = get_device()
    
    # Create dummy microscopy image
    batch_size = 2
    dummy_image = torch.randn(batch_size, 1, 512, 512).to(device)
    logger.info(f"📊 Input shape: {dummy_image.shape}")
    
    # Create backbone
    backbone = create_dinov2_backbone().to(device)
    backbone.eval()
    
    logger.log_model_summary(backbone, (1, 512, 512))
    
    # Test forward pass
    with torch.no_grad():
        outputs = backbone(dummy_image)
    
    # Log outputs
    for key, tensor in outputs.items():
        if tensor is not None:
            logger.info(f"✅ {key}: {tensor.shape}")
        else:
            logger.info(f"❌ {key}: None")
    
    logger.info("🎉 DINOv2 backbone test passed!")

if __name__ == "__main__":
    test_dinov2()
