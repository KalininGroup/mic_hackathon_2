"""
Quick test of the data loading system with dummy data.
"""

import numpy as np
import h5py
import torch
import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vantascope.data.loader import MicroscopyDataset, DataFactory
from vantascope.data.preprocessing import MicroscopyPreprocessor, get_preset_preprocessor
from vantascope.config import DatasetConfig
from vantascope.utils.helpers import set_seed
from vantascope.utils.logging import logger

def create_dummy_data(temp_dir: Path):
    """Create dummy microscopy data files."""
    
    # Create dummy numpy data (time series)
    graphene_data = np.random.rand(10, 256, 256).astype(np.float32)  # 10 frames
    graphene_path = temp_dir / "graphene.npy"
    np.save(graphene_path, graphene_data)
    
    # Create dummy HDF5 data
    pzt_path = temp_dir / "pzt.h5"
    with h5py.File(pzt_path, 'w') as f:
        # Simulate PZT structure
        height_data = np.random.rand(512, 512).astype(np.float32)
        phase_data = np.random.rand(512, 512).astype(np.float32)
        
        f.create_dataset('HeightRetrace', data=height_data)
        f.create_dataset('Phase1Retrace', data=phase_data)
        f.attrs['instrument'] = 'AFM'
        f.attrs['scan_size'] = '1 um'
    
    return graphene_path, pzt_path

def test_preprocessor():
    """Test preprocessing pipeline."""
    logger.info("🔧 Testing preprocessor")
    
    # Test different input shapes
    test_images = [
        np.random.rand(256, 256),           # 2D
        np.random.rand(1, 256, 256),        # 3D channel-first
        np.random.rand(256, 256, 1),        # 3D channel-last
    ]
    
    preprocessor = MicroscopyPreprocessor(target_size=128)
    
    for i, img in enumerate(test_images):
        result = preprocessor(img)
        logger.info(f"   Input {img.shape} -> Output {result.shape}")
        assert result.shape == (1, 128, 128), f"Wrong output shape: {result.shape}"
        assert result.dtype == torch.float32, f"Wrong dtype: {result.dtype}"
        assert 0 <= result.min() and result.max() <= 1, f"Values not in [0,1]: [{result.min():.3f}, {result.max():.3f}]"
    
    logger.info("✅ Preprocessor test passed")

def test_datasets(graphene_path: Path, pzt_path: Path):
    """Test dataset loading."""
    logger.info("📁 Testing dataset loading")
    
    # Test numpy dataset
    graphene_config = DatasetConfig(
        name="test_graphene",
        data_path=graphene_path,
        format="numpy",
        channels=["HAADF"],
        complexity="basic",
        source="test"
    )
    
    graphene_dataset = DataFactory.create_dataset(graphene_config)
    logger.info(f"   Graphene dataset: {len(graphene_dataset)} samples")
    
    # Test HDF5 dataset  
    pzt_config = DatasetConfig(
        name="test_pzt",
        data_path=pzt_path,
        format="hdf5", 
        channels=["HeightRetrace", "Phase1Retrace"],
        complexity="advanced",
        source="test"
    )
    
    pzt_dataset = DataFactory.create_dataset(pzt_config)
    logger.info(f"   PZT dataset: {len(pzt_dataset)} samples")
    
    # Test sample loading
    graphene_sample = graphene_dataset[0]
    pzt_sample = pzt_dataset[0]
    
    logger.info(f"   Graphene sample shape: {graphene_sample['image'].shape}")
    logger.info(f"   PZT sample shape: {pzt_sample['image'].shape}")
    
    assert graphene_sample['image'].shape == (1, 512, 512), "Wrong graphene shape"
    assert pzt_sample['image'].shape == (1, 512, 512), "Wrong PZT shape"
    
    logger.info("✅ Dataset loading test passed")
    
    return graphene_dataset, pzt_dataset

def test_dataloader(dataset):
    """Test DataLoader creation."""
    logger.info("🔄 Testing DataLoader")
    
    dataloader = DataFactory.create_dataloader(
        dataset, 
        batch_size=2,
        split="train",
        num_workers=0
    )
    
    # Test one batch
    batch = next(iter(dataloader))
    logger.info(f"   Batch image shape: {batch['image'].shape}")
    logger.info(f"   Batch metadata keys: {list(batch['metadata'].keys())}")
    
    assert batch['image'].shape[0] <= 2, "Batch size too large"
    assert batch['image'].shape[1:] == (1, 512, 512), "Wrong image dimensions"
    
    logger.info("✅ DataLoader test passed")

def main():
    """Run all tests."""
    logger.info("🚀 Testing VantaScope data loading system")
    
    set_seed(42)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Create test data
        graphene_path, pzt_path = create_dummy_data(temp_dir)
        
        # Run tests
        test_preprocessor()
        graphene_dataset, pzt_dataset = test_datasets(graphene_path, pzt_path)
        test_dataloader(graphene_dataset)
        
        logger.info("🎉 All data loading tests passed!")

if __name__ == "__main__":
    main()
