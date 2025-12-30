"""
Test loading real graphene and PZT data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vantascope.data.factory import VantaScopeDataFactory
from vantascope.config import VantaScopeConfig
from vantascope.utils.logging import logger
from vantascope.utils.viz import plot_dataset_samples
import matplotlib.pyplot as plt

def test_real_datasets():
    """Test loading real datasets."""
    
    # Load config
    config = VantaScopeConfig.from_yaml("config/datasets_real.yaml")
    
    # Test graphene dataset
    if Path(config.datasets["graphene_stem"].data_path).exists():
        logger.info("🔬 Testing Graphene dataset...")
        
        graphene_dataset = VantaScopeDataFactory.create_dataset(config.datasets["graphene_stem"])
        logger.info(f"   Loaded {len(graphene_dataset)} graphene images")
        
        if len(graphene_dataset) > 0:
            # Load a few samples
            samples = []
            for i in range(min(6, len(graphene_dataset))):
                sample = graphene_dataset[i]
                samples.append(sample['image'].squeeze().numpy())
            
            # Plot samples
            fig = plot_dataset_samples(samples, titles=[f"Sample {i}" for i in range(len(samples))])
            plt.savefig("data/graphene_samples.png")
            logger.info("   Saved sample images to data/graphene_samples.png")
    
    # Test PZT dataset
    if Path(config.datasets["pzt_ferroelectric"].data_path).exists():
        logger.info("🔬 Testing PZT dataset...")
        
        pzt_dataset = VantaScopeDataFactory.create_dataset(config.datasets["pzt_ferroelectric"])
        logger.info(f"   Loaded {len(pzt_dataset)} PZT channels")
        
        if len(pzt_dataset) > 0:
            sample = pzt_dataset[0]
            logger.info(f"   PZT sample shape: {sample['image'].shape}")
            logger.info(f"   Metadata: {sample['metadata']}")

if __name__ == "__main__":
    test_real_datasets()
