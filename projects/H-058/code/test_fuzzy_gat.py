"""
Test Fuzzy-GAT with real patch embeddings from autoencoder.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vantascope.models.autoencoder import create_autoencoder
from vantascope.models.fuzzy_gat import create_fuzzy_gat
from vantascope.data.factory import VantaScopeDataFactory
from vantascope.config import VantaScopeConfig
from vantascope.utils.helpers import set_seed, get_device
from vantascope.utils.logging import logger
import torch

def test_fuzzy_gat():
    """Test Fuzzy-GAT with real microscopy data."""
    logger.info("🧠 Testing Fuzzy-GAT with real patch embeddings")
    
    set_seed(42)
    device = get_device()
    
    # Load real data
    config = VantaScopeConfig.from_yaml("config/datasets_real.yaml")
    dataset = VantaScopeDataFactory.create_dataset(config.datasets["graphene_stem"])
    dataloader = VantaScopeDataFactory.create_dataloader(dataset, batch_size=2, split="train")
    
    # Create models
    autoencoder = create_autoencoder().to(device)
    fuzzy_gat = create_fuzzy_gat().to(device)
    
    autoencoder.eval()
    fuzzy_gat.eval()
    
    # Get real data
    batch = next(iter(dataloader))
    images = batch['image'].to(device)
    
    logger.info(f"📊 Input images: {images.shape}")
    
    # Extract patch embeddings using autoencoder
    with torch.no_grad():
        cae_outputs = autoencoder(images)
        patch_embeddings = cae_outputs['patch_embeddings']  # [batch_size, num_patches, feature_dim]
        
        logger.info(f"📊 Patch embeddings: {patch_embeddings.shape}")
        
        # Process with Fuzzy-GAT
        gat_outputs = fuzzy_gat(patch_embeddings)
    
    # Log all outputs
    logger.info("\n🔬 Fuzzy-GAT outputs:")
    for key, tensor in gat_outputs.items():
        if isinstance(tensor, torch.Tensor):
            logger.info(f"   {key}: {tensor.shape}")
            
            # Special analysis for key outputs
            if key == 'fuzzy_memberships':
                # Show fuzzy membership distributions
                mean_memberships = tensor.mean(dim=0)
                logger.info(f"      Mean memberships: {mean_memberships.cpu().numpy()}")
            elif key == 'edge_weights':
                logger.info(f"      Edge weight range: [{tensor.min():.3f}, {tensor.max():.3f}]")
        elif isinstance(tensor, list):
            logger.info(f"   {key}: {len(tensor)} attention layers")
    
    # Analyze graph structure
    num_nodes = gat_outputs['nodes'].shape[0]
    num_edges = gat_outputs['edge_index'].shape[1]
    avg_degree = (num_edges * 2) / num_nodes  # Undirected graph
    
    logger.info(f"\n📊 Graph statistics:")
    logger.info(f"   Nodes: {num_nodes}")
    logger.info(f"   Edges: {num_edges}")
    logger.info(f"   Average degree: {avg_degree:.1f}")
    
    # Test fuzzy reasoning
    fuzzy_memberships = gat_outputs['fuzzy_memberships']
    dominant_categories = torch.argmax(fuzzy_memberships, dim=1)
    category_counts = torch.bincount(dominant_categories, minlength=5)
    
    categories = ['perfect_lattice', 'defect', 'grain_boundary', 'amorphous', 'noise']
    logger.info(f"\n🔮 Fuzzy category distribution:")
    for i, (cat, count) in enumerate(zip(categories, category_counts)):
        percentage = (count.float() / num_nodes * 100).item()
        logger.info(f"   {cat}: {count} nodes ({percentage:.1f}%)")
    
    logger.info("🎉 Fuzzy-GAT test complete!")

if __name__ == "__main__":
    test_fuzzy_gat()
