"""
Test Integrated VantaScope Model
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vantascope.models.integrated_model import create_vantascope_model, create_vantascope_model_with_uncertainty
from vantascope.utils.logging import logger

def test_integrated_model():
    """Test the integrated VantaScope model."""
    logger.info("🧪 Testing Integrated VantaScope Model")
    
    # Create models
    model = create_vantascope_model()
    model_with_uncertainty = create_vantascope_model_with_uncertainty()
    
    # Test input
    batch_size = 2
    test_input = torch.randn(batch_size, 1, 512, 512)
    
    logger.info(f"   Input shape: {test_input.shape}")
    
    # Test basic model
    logger.info("🔬 Testing Basic Integrated Model...")
    model.eval()
    
    with torch.no_grad():
        outputs = model(test_input, return_graph=True)
    
    logger.info(f"   Output keys: {list(outputs.keys())}")
    
    # Check all expected outputs
    expected_keys = [
        'reconstruction', 'latent', 'geometric', 'topological', 'disorder',
        'energy_mean', 'energy_std', 'properties', 'graph_outputs',
        'enhanced_energy_mean', 'enhanced_energy_std', 
        'patch_defect_logits', 'patch_defect_probs'
    ]
    
    for key in expected_keys:
        if key in outputs:
            if isinstance(outputs[key], torch.Tensor):
                logger.info(f"   ✅ {key}: {outputs[key].shape}")
            elif isinstance(outputs[key], dict):
                logger.info(f"   ✅ {key}: dict with keys {list(outputs[key].keys())}")
        else:
            logger.warning(f"   ⚠️  Missing key: {key}")
    
    # Test graph outputs
    if 'graph_outputs' in outputs:
        graph_outputs = outputs['graph_outputs']
        logger.info("   Graph outputs:")
        logger.info(f"     Node features: {graph_outputs['node_features'].shape}")
        logger.info(f"     Attention weights: {graph_outputs['attention_weights'].shape}")
        logger.info(f"     Graph embedding: {graph_outputs['graph_embedding'].shape}")
        logger.info(f"     Graph statistics: {list(graph_outputs['graph_statistics'].keys())}")
    
    # Test without graph
    logger.info("🔬 Testing Model without Graph...")
    outputs_no_graph = model(test_input, return_graph=False)
    logger.info(f"   Output keys (no graph): {list(outputs_no_graph.keys())}")
    
    # Test model with uncertainty
    logger.info("🔬 Testing Model with Uncertainty...")
    model_with_uncertainty.eval()
    
    with torch.no_grad():
        uncertainty_outputs = model_with_uncertainty(test_input)
    
    logger.info("   Additional uncertainty outputs:")
    logger.info(f"     Property predictions: {uncertainty_outputs['property_predictions'].shape}")
    logger.info(f"     Property uncertainty: {uncertainty_outputs['property_uncertainty'].shape}")
    logger.info(f"     Bandgap: {uncertainty_outputs['bandgap_mean'].squeeze().tolist()} ± {uncertainty_outputs['bandgap_std'].squeeze().tolist()}")
    logger.info(f"     Modulus: {uncertainty_outputs['modulus_mean'].squeeze().tolist()} ± {uncertainty_outputs['modulus_std'].squeeze().tolist()}")
    
    # Test training mode
    logger.info("🔬 Testing Training Mode...")
    model.train()
    outputs_train = model(test_input)
    
    # Compute dummy loss and check gradients
    loss = outputs_train['reconstruction'].mean()
    loss.backward()
    
    grad_count = 0
    for param in model.parameters():
        if param.grad is not None:
            grad_count += 1
    
    logger.info(f"   Gradients computed: {grad_count} parameters")
    
    logger.info("🎉 Integrated Model test completed successfully!")
    
    return model, model_with_uncertainty, outputs

if __name__ == "__main__":
    test_integrated_model()
