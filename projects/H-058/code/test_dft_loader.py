"""
Test DFT Graphene dataset loader with BigGrapheneDataset.tar.gz
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vantascope.data.dft_graphene_loader import create_dft_loaders
from vantascope.utils.logging import logger
import matplotlib.pyplot as plt
import torch

def test_dft_loader():
    """Test DFT dataset loading."""
    logger.info("🧪 Testing DFT Graphene loader with BigGrapheneDataset")
    
    # Path to your dataset
    data_path = "data/BigGrapheneDataset.tar.gz"
    
    # Create loaders with small sample for testing
    train_loader, test_loader = create_dft_loaders(
        data_path=data_path,
        batch_size=4,
        max_samples=20,   # Start small to test structure
        num_workers=0     # Single thread for testing
    )
    
    logger.info(f"📊 Training samples: {len(train_loader.dataset):,}")
    logger.info(f"📊 Testing samples: {len(test_loader.dataset):,}")
    
    # Get a batch
    try:
        batch = next(iter(train_loader))
        
        logger.info(f"📊 Batch shapes:")
        logger.info(f"   Images: {batch['image'].shape}")
        logger.info(f"   Energies: {batch['energy'].shape}")
        logger.info(f"   Coordinates: {batch['coordinates'].shape}")  # Fixed: coordinates not positions
        logger.info(f"   Geometric GT: {batch['geometric_gt'].shape}")
        logger.info(f"   Topological GT: {batch['topological_gt'].shape}")
        
        # Show energy and defect statistics
        energies = batch['energy'].numpy()
        defect_ratios = [m['defect_ratio'] for m in batch['metadata']]
        num_atoms = [m['num_atoms'] for m in batch['metadata']]
        
        logger.info(f"📈 Sample statistics:")
        logger.info(f"   Energy range: [{energies.min():.1f}, {energies.max():.1f}] eV")
        logger.info(f"   Defect ratios: {[f'{d:.3f}' for d in defect_ratios]}")
        logger.info(f"   Atom counts: {num_atoms}")
        
        # Plot samples
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()
        
        for i in range(4):
            axes[i].imshow(batch['image'][i, 0].numpy(), cmap='viridis')
            energy = batch['energy'][i].item()
            defect = batch['metadata'][i]['defect_ratio']
            atoms = batch['metadata'][i]['num_atoms']
            axes[i].set_title(f"E: {energy:.0f} eV\nDefects: {defect:.3f} | Atoms: {atoms}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('data/dft_samples.png', dpi=150)
        logger.info("💾 Saved sample visualization to data/dft_samples.png")
        
        logger.info("🎉 DFT loader test successful!")
        
    except Exception as e:
        logger.error(f"❌ Error loading batch: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dft_loader()
