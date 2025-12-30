"""
Test the graph overlay visualization with real data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vantascope.models.autoencoder import create_autoencoder
from vantascope.models.fuzzy_gat import create_fuzzy_gat
from vantascope.data.factory import VantaScopeDataFactory
from vantascope.config import VantaScopeConfig
from vantascope.visualization.graph_overlay import create_graph_overlay
from vantascope.utils.helpers import set_seed, get_device
from vantascope.utils.logging import logger
import torch
import matplotlib.pyplot as plt

def test_graph_visualization():
    """Test graph overlay on real microscopy data."""
    logger.info("🎨 Testing graph overlay visualization")
    
    set_seed(42)
    device = get_device()
    
    # Load real data
    config = VantaScopeConfig.from_yaml("config/datasets_real.yaml")
    dataset = VantaScopeDataFactory.create_dataset(config.datasets["graphene_stem"])
    dataloader = VantaScopeDataFactory.create_dataloader(dataset, batch_size=1, split="train")
    
    # Create models
    autoencoder = create_autoencoder().to(device)
    fuzzy_gat = create_fuzzy_gat().to(device)
    
    autoencoder.eval()
    fuzzy_gat.eval()
    
    # Get a sample
    batch = next(iter(dataloader))
    images = batch['image'].to(device)
    
    logger.info(f"📊 Processing image: {images.shape}")
    
    # Forward pass
    with torch.no_grad():
        cae_outputs = autoencoder(images)
        gat_outputs = fuzzy_gat(cae_outputs['patch_embeddings'])
    
    # Extract first image and convert to numpy
    image_np = images[0, 0].cpu().numpy()  # [H, W]
    
    # Create visualization
    visualizer = create_graph_overlay()
    
    logger.info("🎨 Creating graph overlay...")
    fig = visualizer.create_overlay(
        image=image_np,
        gat_outputs=gat_outputs,
        title="VantaScope: Fuzzy-GAT Reasoning on Graphene STEM"
    )
    
    # Save visualization
    output_path = "data/graph_overlay_demo.png"
    visualizer.save_overlay(fig, output_path, dpi=150)
    
    logger.info(f"🎉 Graph visualization complete! Check {output_path}")
    
    # Show plot (optional)
    # plt.show()  # Commented out for terminal mode

if __name__ == "__main__":
    test_graph_visualization()
