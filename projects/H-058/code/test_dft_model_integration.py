"""
Comprehensive DFT Model Integration Test
Tests the complete pipeline: DFT Autoencoder → Linguistic Engine
"""

import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vantascope.models.dft_autoencoder import create_dft_autoencoder
from vantascope.analysis.linguistic_engine import create_linguistic_engine
from vantascope.utils.logging import logger

def test_model_architecture():
    """Test 1: Verify model architecture and forward pass."""
    logger.info("🧪 Test 1: Model Architecture & Forward Pass")
    
    # Create model
    model = create_dft_autoencoder()
    model.eval()
    
    # Create dummy input - SINGLE CHANNEL for microscopy
    batch_size = 2
    dummy_input = torch.randn(batch_size, 1, 512, 512)  # Single channel grayscale
    
    logger.info(f"   Input shape: {dummy_input.shape}")
    
    # Forward pass
    try:
        with torch.no_grad():
            outputs = model(dummy_input)
        
        logger.info("   ✅ Forward pass successful")
        
        # Check tensor outputs
        tensor_keys = ['reconstruction', 'latent', 'geometric', 'topological', 'disorder', 'features', 'energy_mean', 'energy_std']
        
        for key in tensor_keys:
            if key in outputs and isinstance(outputs[key], torch.Tensor):
                logger.info(f"   ✅ {key}: {outputs[key].shape}")
            else:
                logger.error(f"   ❌ Missing tensor output: {key}")
                return False
        
        # Check properties dict
        if 'properties' in outputs and isinstance(outputs['properties'], dict):
            logger.info(f"   ✅ properties: dict with {len(outputs['properties'])} keys")
            prop_keys = ['crystallinity', 'defect_density', 'stability', 'defect_probs']
            for key in prop_keys:
                if key in outputs['properties']:
                    logger.info(f"   ✅ properties.{key}: {outputs['properties'][key].shape}")
                else:
                    logger.error(f"   ❌ Missing property key: {key}")
                    return False
        else:
            logger.error("   ❌ Properties output missing or not a dict")
            return False
        
        # Check other outputs (can be dict/list/etc)
        other_keys = ['attention_maps', 'patch_embeddings']
        for key in other_keys:
            if key in outputs:
                value = outputs[key]
                if isinstance(value, torch.Tensor):
                    logger.info(f"   ✅ {key}: {value.shape}")
                elif isinstance(value, dict):
                    logger.info(f"   ✅ {key}: dict with keys {list(value.keys())}")
                elif isinstance(value, list):
                    logger.info(f"   ✅ {key}: list with {len(value)} items")
                else:
                    logger.info(f"   ✅ {key}: {type(value)}")
            else:
                logger.warning(f"   ⚠️  Optional output missing: {key}")
        
        return True, outputs
        
    except Exception as e:
        logger.error(f"   ❌ Forward pass failed: {e}")
        import traceback
        logger.error(f"   Full traceback: {traceback.format_exc()}")
        return False, None

def test_output_shapes_and_ranges(outputs):
    """Test 2: Verify output shapes and value ranges."""
    logger.info("🧪 Test 2: Output Shapes & Value Ranges")
    
    batch_size = outputs['latent'].shape[0]
    logger.info(f"   Batch size: {batch_size}")
    
    # Test shapes
    shape_tests = [
        ('reconstruction', (batch_size, 1, 512, 512)),
        ('latent', (batch_size, 64)),
        ('geometric', (batch_size, 16)),
        ('topological', (batch_size, 32)),
        ('disorder', (batch_size, 16)),
        ('energy_mean', (batch_size, 1)),
        ('energy_std', (batch_size, 1)),
    ]
    
    for key, expected_shape in shape_tests:
        if key in outputs and isinstance(outputs[key], torch.Tensor):
            actual_shape = outputs[key].shape
            if actual_shape == expected_shape:
                logger.info(f"   ✅ {key} shape: {actual_shape}")
            else:
                logger.error(f"   ❌ {key} shape mismatch: got {actual_shape}, expected {expected_shape}")
                return False
    
    # Test value ranges
    range_tests = [
        ('energy_mean', -5000.0, 0.0),  # DFT energies are negative
        ('energy_std', 0.0, 100.0),  # Positive uncertainty
    ]
    
    for key, min_val, max_val in range_tests:
        if key in outputs and isinstance(outputs[key], torch.Tensor):
            tensor = outputs[key]
            actual_min, actual_max = tensor.min().item(), tensor.max().item()
            
            if min_val <= actual_min and actual_max <= max_val:
                logger.info(f"   ✅ {key} range: [{actual_min:.3f}, {actual_max:.3f}]")
            else:
                logger.warning(f"   ⚠️  {key} range: [{actual_min:.3f}, {actual_max:.3f}] (expected [{min_val}, {max_val}])")
    
    # Test properties ranges
    if 'properties' in outputs:
        props = outputs['properties']
        prop_range_tests = [
            ('crystallinity', 0.0, 1.0),
            ('defect_density', 0.0, 1.0),
            ('stability', 0.0, 1.0),
        ]
        
        for key, min_val, max_val in prop_range_tests:
            if key in props:
                tensor = props[key]
                actual_min, actual_max = tensor.min().item(), tensor.max().item()
                
                if min_val <= actual_min and actual_max <= max_val:
                    logger.info(f"   ✅ properties.{key} range: [{actual_min:.3f}, {actual_max:.3f}]")
                else:
                    logger.warning(f"   ⚠️  properties.{key} range: [{actual_min:.3f}, {actual_max:.3f}] (expected [{min_val}, {max_val}])")
        
        # Test defect probabilities sum to 1
        if 'defect_probs' in props:
            defect_probs = props['defect_probs']
            prob_sums = defect_probs.sum(dim=1)
            
            if torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-3):
                logger.info(f"   ✅ defect_probs sum to 1: {prob_sums.tolist()}")
            else:
                logger.warning(f"   ⚠️  defect_probs don't sum to 1: {prob_sums.tolist()}")
    
    return True

def test_linguistic_integration(outputs):
    """Test 3: Integration with Linguistic Engine."""
    logger.info("🧪 Test 3: Linguistic Engine Integration")
    
    # Create linguistic engine
    linguistic_engine = create_linguistic_engine()
    
    # Test with first sample from batch
    sample_outputs = {}
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            sample_outputs[key] = value[0:1]
        elif isinstance(value, dict):
            sample_outputs[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, torch.Tensor):
                    sample_outputs[key][subkey] = subvalue[0:1]
    
    # Create dummy metadata
    metadata = {
        'num_atoms': 450,
        'defect_ratio': 0.02,
        'filename': 'test_sample.h5'
    }
    
    try:
        # Generate analysis
        analysis = linguistic_engine.analyze(sample_outputs, metadata)
        
        logger.info("   ✅ Linguistic analysis generated successfully")
        logger.info(f"   Quality Grade: {analysis.quality_grade}")
        logger.info(f"   Primary Defect: {analysis.primary_defect_type}")
        logger.info(f"   Energy: {analysis.energy:.1f} ± {analysis.energy_uncertainty:.1f} eV")
        
        # Test full report generation
        full_report = linguistic_engine.format_full_report(analysis)
        
        logger.info("   ✅ Full report generated successfully")
        logger.info(f"   Report length: {len(full_report)} characters")
        
        return True, analysis, full_report
        
    except Exception as e:
        logger.error(f"   ❌ Linguistic integration failed: {e}")
        import traceback
        logger.error(f"   Full traceback: {traceback.format_exc()}")
        return False, None, None

def test_disentanglement_properties(outputs):
    """Test 4: Verify disentanglement properties."""
    logger.info("🧪 Test 4: Disentanglement Properties")
    
    geometric = outputs['geometric']
    topological = outputs['topological'] 
    disorder = outputs['disorder']
    
    # Test 1: Dimensions sum to total latent dimension
    total_dims = geometric.shape[1] + topological.shape[1] + disorder.shape[1]
    expected_dims = outputs['latent'].shape[1]
    
    if total_dims == expected_dims:
        logger.info(f"   ✅ Dimension consistency: {geometric.shape[1]}+{topological.shape[1]}+{disorder.shape[1]} = {total_dims}")
    else:
        logger.error(f"   ❌ Dimension mismatch: {total_dims} != {expected_dims}")
        return False
    
    # Test 2: Concatenation matches latent
    reconstructed_latent = torch.cat([geometric, topological, disorder], dim=1)
    
    if torch.allclose(reconstructed_latent, outputs['latent'], atol=1e-5):
        logger.info("   ✅ Latent reconstruction matches concatenation")
    else:
        logger.error("   ❌ Latent reconstruction mismatch")
        return False
    
    # Test 3: Statistical independence (correlation test)
    def correlation_matrix(x, y):
        """Compute correlation between all pairs of dimensions."""
        x_flat = x.view(-1, x.shape[-1])
        y_flat = y.view(-1, y.shape[-1])
        
        # Normalize
        x_norm = (x_flat - x_flat.mean(0)) / (x_flat.std(0) + 1e-8)
        y_norm = (y_flat - y_flat.mean(0)) / (y_flat.std(0) + 1e-8)
        
        # Correlation
        corr = torch.mm(x_norm.T, y_norm) / x_norm.shape[0]
        return corr.abs().mean().item()
    
    geo_topo_corr = correlation_matrix(geometric, topological)
    geo_disorder_corr = correlation_matrix(geometric, disorder)
    topo_disorder_corr = correlation_matrix(topological, disorder)
    
    logger.info(f"   Cross-correlations (lower is better):")
    logger.info(f"     Geometric ↔ Topological: {geo_topo_corr:.4f}")
    logger.info(f"     Geometric ↔ Disorder: {geo_disorder_corr:.4f}")
    logger.info(f"     Topological ↔ Disorder: {topo_disorder_corr:.4f}")
    
    return True

def test_gradient_flow():
    """Test 5: Gradient flow and backpropagation."""
    logger.info("🧪 Test 5: Gradient Flow")
    
    model = create_dft_autoencoder()
    model.train()  # Enable gradients
    
    # Create dummy input and targets
    dummy_input = torch.randn(1, 1, 512, 512, requires_grad=True)
    target_energy = torch.tensor([[-2500.0]])  # Dummy DFT energy
    
    # Forward pass
    try:
        outputs = model(dummy_input)
        
        # Simple loss (just energy for this test)
        energy_loss = torch.nn.functional.mse_loss(outputs['energy_mean'], target_energy)
        
        # Backward pass
        energy_loss.backward()
        logger.info("   ✅ Backward pass successful")
        
        # Check gradients exist
        grad_count = 0
        total_params = 0
        non_zero_grads = 0
        
        for name, param in model.named_parameters():
            total_params += 1
            if param.grad is not None:
                grad_count += 1
                grad_norm = param.grad.norm().item()
                if grad_norm > 1e-6:  # Non-zero gradient
                    non_zero_grads += 1
        
        logger.info(f"   Gradient coverage: {grad_count}/{total_params} parameters have gradients")
        logger.info(f"   Non-zero gradients: {non_zero_grads}/{grad_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Gradient flow failed: {e}")
        import traceback
        logger.error(f"   Full traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all integration tests."""
    logger.info("🚀 Starting DFT Model Integration Tests")
    logger.info("=" * 60)
    
    # Test 1: Architecture
    success, outputs = test_model_architecture()
    if not success:
        logger.error("❌ Architecture test failed - stopping")
        return
    
    # Test 2: Shapes and ranges
    if not test_output_shapes_and_ranges(outputs):
        logger.error("❌ Shape/range test failed - stopping")
        return
    
    # Test 3: Linguistic integration
    success, analysis, report = test_linguistic_integration(outputs)
    if not success:
        logger.error("❌ Linguistic integration failed - stopping")
        return
    
    # Test 4: Disentanglement
    if not test_disentanglement_properties(outputs):
        logger.error("❌ Disentanglement test failed - stopping")
        return
    
    # Test 5: Gradients
    if not test_gradient_flow():
        logger.error("❌ Gradient flow test failed - stopping")
        return
    
    logger.info("=" * 60)
    logger.info("🎉 ALL TESTS PASSED!")
    logger.info("=" * 60)
    
    # Print sample report
    logger.info("\n📋 SAMPLE LINGUISTIC REPORT:")
    logger.info("-" * 40)
    print(report[:500] + "..." if len(report) > 500 else report)

if __name__ == "__main__":
    main()
