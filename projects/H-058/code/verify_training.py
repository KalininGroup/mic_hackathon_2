"""
Verify that training actually worked.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vantascope.utils.logging import logger
import torch

def verify_training():
    """Check if training actually completed."""
    
    models_dir = Path("models/trained")
    
    # Check for saved models
    best_model = models_dir / "best_model.pth"
    final_model = models_dir / "final_model.pth"
    
    logger.info("🔍 Verifying training results...")
    
    if best_model.exists():
        logger.info(f"✅ Best model found: {best_model}")
        
        # Load and inspect checkpoint
        try:
            checkpoint = torch.load(best_model, map_location='cpu')
            
            epoch = checkpoint.get('epoch', 'unknown')
            metrics = checkpoint.get('metrics', {})
            
            logger.info(f"📊 Training completed through epoch: {epoch}")
            logger.info(f"📈 Final validation metrics:")
            for key, value in metrics.items():
                logger.info(f"   {key}: {value:.6f}")
            
            # Check if losses make sense
            if metrics.get('total_loss', 999) > 50:
                logger.warning("⚠️ Very high loss - training might not have converged")
            elif metrics.get('reconstruction_loss', 999) < 0.001:
                logger.warning("⚠️ Very low loss - might indicate issue")
            else:
                logger.info("✅ Loss values look reasonable")
                
        except Exception as e:
            logger.error(f"❌ Failed to load model checkpoint: {e}")
    
    else:
        logger.error("❌ No trained model found!")
        logger.info("💡 Training might have failed or not completed")
    
    if final_model.exists():
        logger.info(f"✅ Final model also saved: {final_model}")
    
    # Check model file sizes
    if best_model.exists():
        size_mb = best_model.stat().st_size / (1024*1024)
        logger.info(f"📦 Model size: {size_mb:.1f} MB")
        
        if size_mb < 100:
            logger.warning("⚠️ Model file seems small - might not have saved properly")

if __name__ == "__main__":
    verify_training()
