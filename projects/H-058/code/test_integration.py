"""
Integration test: Data loader -> DINOv2 backbone
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vantascope.data.loader import DataFactory
from vantascope.data.preprocessing import get_preset_preprocessor  
from vantascope.models.dinov2_backbone import create_dinov2_backbone
from vantascope.config import DatasetConfig
from vantascope.utils.helpers import set_seed, get_device
from vantascope.utils.logging import logger
import numpy as np
import tempfile

def test_data_to_model():
    """Test data pipeline -> model forward pass."""
    logger.info("🔗 Testing data loading -> DINOv2 integration")
    
    # Create dummy data
    with tempfile.TemporaryDirectory() as temp_dir:
        dummy_data = np.random.rand(3, 256, 256).astype(np.float32)
        data_path = Path(temp_dir) / "test.npy"
        np.save(data_path, dummy_data)
        
        # Create dataset
        config = DatasetConfig(
            name="integration_test",
            data_path=data_path,
            format="numpy", 
            channels=["test"],
            complexity="basic",
            source="test"
        )
        
        dataset = DataFactory.create_dataset(config, get_preset_preprocessor("graphene"))
        dataloader = DataFactory.create_dataloader(dataset, batch_size=2, split="train")
        
        # Create model
        model = create_dinov2_backbone().to(get_device())
        model.eval()
        
        # Test forward pass
        batch = next(iter(dataloader))
        images = batch['image'].to(get_device())
        
        logger.info(f"📊 Batch shape: {images.shape}")
        
        with torch.no_grad():
            outputs = model(images)
        
        for key, tensor in outputs.items():
            if tensor is not None:
                logger.info(f"✅ {key}: {tensor.shape}")
        
        logger.info("🎉 Integration test passed!")

if __name__ == "__main__":
    import torch
    set_seed(42)
    test_data_to_model()
