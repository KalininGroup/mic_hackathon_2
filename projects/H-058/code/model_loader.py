"""
Utility to load trained VantaScope models for inference.
"""

import torch
from pathlib import Path
from typing import Tuple, Optional

from ..models.autoencoder import create_autoencoder
from ..models.fuzzy_gat import create_fuzzy_gat
from .logging import logger
from .helpers import get_device


def load_trained_vantascope(model_path: Optional[str] = None) -> Tuple:
    """
    Load trained VantaScope models for inference.
    
    Args:
        model_path: Path to trained model, or None for best available
        
    Returns:
        (autoencoder, fuzzy_gat, device) tuple
    """
    device = get_device()
    
    # Find best available model
    if model_path is None:
        model_dir = Path("models/trained")
        
        # Look for best model first
        best_path = model_dir / "best_model.pth"
        final_path = model_dir / "vantascope_final.pth"
        
        if best_path.exists():
            model_path = best_path
            logger.info("📥 Loading best trained model")
        elif final_path.exists():
            model_path = final_path
            logger.info("📥 Loading final trained model")
        else:
            logger.warning("⚠️ No trained model found, using untrained weights")
            return _load_untrained_models(device)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        logger.info(f"📂 Loading model from {model_path}")
        
        # Create models
        autoencoder = create_autoencoder()
        fuzzy_gat = create_fuzzy_gat()
        
        # Load trained weights
        autoencoder.load_state_dict(checkpoint['autoencoder_state'])
        fuzzy_gat.load_state_dict(checkpoint['fuzzy_gat_state'])
        
        # Move to device and set eval mode
        autoencoder.to(device).eval()
        fuzzy_gat.to(device).eval()
        
        # Log training info
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            logger.info(f"✅ Loaded trained model - Val Loss: {metrics.get('total_loss', 'unknown'):.4f}")
        
        logger.info("🎯 Models ready for inference with trained weights")
        return autoencoder, fuzzy_gat, device
        
    except Exception as e:
        logger.error(f"❌ Failed to load trained model: {e}")
        logger.info("🔄 Falling back to untrained models")
        return _load_untrained_models(device)


def _load_untrained_models(device):
    """Load untrained models as fallback."""
    autoencoder = create_autoencoder().to(device).eval()
    fuzzy_gat = create_fuzzy_gat().to(device).eval()
    
    logger.warning("⚠️ Using untrained models - results will be random!")
    logger.info("💡 Run 'python scripts/train_vantascope.py' to train the system")
    
    return autoencoder, fuzzy_gat, device
