"""
Test the complete trained VantaScope system.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import matplotlib.pyplot as plt
import numpy as np

from vantascope.models.autoencoder import create_autoencoder
from vantascope.models.fuzzy_gat import create_fuzzy_gat
from vantascope.data.factory import VantaScopeDataFactory
from vantascope.config import VantaScopeConfig
from vantascope.visualization.graph_overlay import create_graph_overlay
from vantascope.utils.logging import logger
from vantascope.utils.helpers import get_device

def load_trained_models():
    """Load the trained models."""
    device = get_device()
    
    # Load checkpoint
    checkpoint = torch.load("models/trained/best_model.pth", map_location=device)
    logger.info(f"📂 Loaded model from epoch {checkpoint['epoch']}")
    
    # Create models
    autoencoder = create_autoencoder()
    fuzzy_gat = create_fuzzy_gat()
    
    # Load trained weights
    autoencoder.load_state_dict(checkpoint['autoencoder_state'])
    fuzzy_gat.load_state_dict(checkpoint['fuzzy_gat_state'])
    
    # Move to device and eval mode
    autoencoder.to(device).eval()
    fuzzy_gat.to(device).eval()
    
    logger.info("✅ Trained models loaded successfully")
    return autoencoder, fuzzy_gat, device

def test_trained_system():
    """Test the trained VantaScope system."""
    logger.info("🧪 Testing trained VantaScope system")
    
    # Load trained models
    autoencoder, fuzzy_gat, device = load_trained_models()
    
    # Load test data
    config = VantaScopeConfig.from_yaml("config/datasets_real.yaml")
    dataset = VantaScopeDataFactory.create_dataset(config.datasets["graphene_stem"])
    test_loader = VantaScopeDataFactory.create_dataloader(dataset, batch_size=1, split="val")
    
    # Get a test sample
    batch = next(iter(test_loader))
    images = batch['image'].to(device)
    
    logger.info(f"📊 Testing on image: {images.shape}")
    
    # Forward pass with trained models
    with torch.no_grad():
        cae_outputs = autoencoder(images)
        gat_outputs = fuzzy_gat(cae_outputs['patch_embeddings'])
    
    # Check reconstruction quality
    reconstruction = cae_outputs['reconstruction'][0, 0].cpu().numpy()
    original = images[0, 0].cpu().numpy()
    mse = torch.nn.functional.mse_loss(cae_outputs['reconstruction'], images)
    
    logger.info(f"🎭 Reconstruction MSE: {mse.item():.6f}")
    
    # Check fuzzy classifications
    fuzzy_memberships = gat_outputs['fuzzy_memberships'].cpu().numpy()
    dominant_categories = torch.argmax(gat_outputs['fuzzy_memberships'], dim=1)
    category_counts = torch.bincount(dominant_categories, minlength=5)
    
    categories = ['perfect_lattice', 'defect', 'grain_boundary', 'amorphous', 'noise']
    logger.info("🔮 Material classification results:")
    total_nodes = len(fuzzy_memberships)
    for i, (cat, count) in enumerate(zip(categories, category_counts)):
        percentage = (count.float() / total_nodes * 100).item()
        logger.info(f"   {cat}: {count} nodes ({percentage:.1f}%)")
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Input')
    axes[0, 0].axis('off')
    
    # Reconstruction
    axes[0, 1].imshow(reconstruction, cmap='gray')
    axes[0, 1].set_title(f'Trained Reconstruction\nMSE: {mse.item():.6f}')
    axes[0, 1].axis('off')
    
    # Difference
    diff = np.abs(original - reconstruction)
    im = axes[0, 2].imshow(diff, cmap='hot')
    axes[0, 2].set_title('Reconstruction Error')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
    
    # Attention map
    attention = cae_outputs['attention_maps'][0, 0].cpu().numpy()
    axes[1, 0].imshow(original, cmap='gray', alpha=0.7)
    axes[1, 0].imshow(attention, cmap='hot', alpha=0.5)
    axes[1, 0].set_title('Attention Overlay')
    axes[1, 0].axis('off')
    
    # Latent space histogram
    latent = cae_outputs['latent'][0].cpu().numpy()
    axes[1, 1].hist(latent, bins=30, color='purple', alpha=0.7)
    axes[1, 1].set_title('Latent Space Distribution')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Frequency')
    
    # Material composition bar chart
    percentages = [(count.float() / total_nodes * 100).item() for count in category_counts]
    colors = ['green', 'red', 'yellow', 'blue', 'magenta']
    bars = axes[1, 2].bar(range(5), percentages, color=colors, alpha=0.7)
    axes[1, 2].set_xticks(range(5))
    axes[1, 2].set_xticklabels(['Lattice', 'Defect', 'Grain\nBoundary', 'Amorphous', 'Noise'], fontsize=8)
    axes[1, 2].set_ylabel('Percentage (%)')
    axes[1, 2].set_title('Material Composition')
    
    plt.tight_layout()
    plt.savefig('data/trained_system_test.png', dpi=150, bbox_inches='tight')
    logger.info("💾 Saved test results to data/trained_system_test.png")
    
    # Create graph overlay
    visualizer = create_graph_overlay()
    graph_fig = visualizer.create_overlay(
        image=original,
        gat_outputs=gat_outputs,
        title="Trained VantaScope: Material Structure Analysis"
    )
    visualizer.save_overlay(graph_fig, "data/trained_graph_overlay.png", dpi=150)
    logger.info("💾 Saved graph overlay to data/trained_graph_overlay.png")
    
    logger.info("🎉 Trained system test complete!")

if __name__ == "__main__":
    test_trained_system()
