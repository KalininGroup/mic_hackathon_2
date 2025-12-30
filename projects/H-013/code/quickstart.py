"""
Quick Start Script for SE(3)-Equivariant AFM Reconstruction

This script runs a quick test to ensure everything is working correctly.
Run with: python scripts/quickstart.py
"""

import sys
sys.path.append('.')

import torch
import time
from pathlib import Path

print("="*80)
print("SE(3)-Equivariant AFM Reconstruction - Quick Start Test")
print("="*80)


def test_imports():
    """Test all imports"""
    print("\n1. Testing imports...")
    
    try:
        from config import get_config, set_seed
        from data.simulator import AFMSimulator
        from data.dataset import create_dataloaders
        from models.se3_network import SE3TipReconstructor
        from models.surface_network import SurfaceReconstructor
        from models.joint_model import JointTipSurfaceModel
        from training.losses import AFMReconstructionLoss
        from training.trainer import AFMTrainer
        from evaluation.evaluator import AFMEvaluator
        
        print("   ✓ All imports successful")
        return True
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        return False


def test_configuration():
    """Test configuration"""
    print("\n2. Testing configuration...")
    
    try:
        from config import get_config, set_seed
        
        set_seed(42)
        config = get_config()
        
        print(f"   Device: {config.training.device}")
        print(f"   Batch size: {config.training.batch_size}")
        print(f"   Image size: {config.data.image_size}")
        print(f"   Tip voxel size: {config.data.tip_voxel_size}")
        print("   ✓ Configuration loaded successfully")
        return True, config
    except Exception as e:
        print(f"   ✗ Configuration failed: {e}")
        return False, None


def test_simulator(config):
    """Test AFM simulator"""
    print("\n3. Testing AFM simulator...")
    
    try:
        from data.simulator import AFMSimulator
        
        simulator = AFMSimulator(config)
        
        # Generate sample
        tip = simulator.generate_tip('pyramidal')
        surface = simulator.generate_surface('random')
        image = simulator.simulate_image(surface, tip)
        noisy_image = simulator.add_noise(image)
        
        print(f"   Tip shape: {tip.shape}")
        print(f"   Surface shape: {surface.shape}")
        print(f"   Image shape: {image.shape}")
        print("   ✓ Simulator working correctly")
        return True
    except Exception as e:
        print(f"   ✗ Simulator failed: {e}")
        return False


def test_dataset(config):
    """Test dataset creation"""
    print("\n4. Testing dataset creation (small sample)...")
    
    try:
        from data.dataset import create_dataloaders
        
        # Create small test dataset
        config.data.train_size = 50
        config.data.val_size = 10
        config.data.test_size = 10
        config.training.batch_size = 4
        
        train_loader, val_loader, test_loader = create_dataloaders(
            config, use_cached=False
        )
        
        # Get a batch
        batch = next(iter(train_loader))
        
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        print(f"   Batch image shape: {batch['image'].shape}")
        print(f"   Batch surface shape: {batch['surface'].shape}")
        print(f"   Batch tip shape: {batch['tip'].shape}")
        print("   ✓ Dataset creation successful")
        return True, train_loader, val_loader
    except Exception as e:
        print(f"   ✗ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_model(config):
    """Test model creation"""
    print("\n5. Testing model creation...")
    
    try:
        from models.joint_model import JointTipSurfaceModel
        
        model = JointTipSurfaceModel(config).to(config.training.device)
        
        # Test forward pass
        batch_size = 2
        test_input = torch.randn(batch_size, 1, 128, 128).to(config.training.device)
        
        with torch.no_grad():
            output = model(test_input, monte_carlo_samples=1)
        
        print(f"   Input shape: {test_input.shape}")
        print(f"   Tip output shape: {output['tip'].shape}")
        print(f"   Surface output shape: {output['surface'].shape}")
        print(f"   Simulated image shape: {output['simulated_image'].shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
        print(f"   Memory (FP32): ~{total_params * 4 / 1e6:.1f} MB")
        
        print("   ✓ Model creation and forward pass successful")
        return True, model
    except Exception as e:
        print(f"   ✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_training_step(config, model, train_loader):
    """Test a single training step"""
    print("\n6. Testing training step...")
    
    try:
        from training.losses import AFMReconstructionLoss
        
        loss_fn = AFMReconstructionLoss(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
        
        # Get a batch
        batch = next(iter(train_loader))
        afm_image = batch['image'].to(config.training.device)
        tip_gt = batch['tip'].to(config.training.device)
        surface_gt = batch['surface'].to(config.training.device)
        
        # Forward pass
        model.train()
        predictions = model(afm_image)
        
        ground_truth = {
            'tip': tip_gt,
            'surface': surface_gt,
            'image': afm_image
        }
        
        loss, loss_dict = loss_fn(predictions, ground_truth)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"   Total loss: {loss.item():.4f}")
        print(f"   Tip MSE: {loss_dict['tip_mse']:.4f}")
        print(f"   Surface MSE: {loss_dict['surface_mse']:.4f}")
        print(f"   Consistency: {loss_dict['consistency']:.4f}")
        print("   ✓ Training step successful")
        return True
    except Exception as e:
        print(f"   ✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation(config, model, val_loader):
    """Test evaluation"""
    print("\n7. Testing evaluation...")
    
    try:
        from evaluation.evaluator import AFMEvaluator
        
        evaluator = AFMEvaluator(model, config)
        
        model.eval()
        batch = next(iter(val_loader))
        afm_image = batch['image'].to(config.training.device)
        tip_gt = batch['tip'].to(config.training.device)
        surface_gt = batch['surface'].to(config.training.device)
        
        with torch.no_grad():
            predictions = model(afm_image, monte_carlo_samples=5)
        
        # Compute metrics
        metrics = evaluator._compute_metrics(
            predictions['tip'], tip_gt,
            predictions['surface'], surface_gt,
            predictions['tip_uncertainty'],
            predictions['surface_uncertainty']
        )
        
        print(f"   Tip RMSE: {metrics['tip_rmse']:.4f}")
        print(f"   Surface RMSE: {metrics['surface_rmse']:.4f}")
        print(f"   Surface PSNR: {metrics['surface_psnr']:.2f} dB")
        print(f"   SSIM: {metrics['ssim']:.4f}")
        print("   ✓ Evaluation successful")
        return True
    except Exception as e:
        print(f"   ✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    start_time = time.time()
    
    # Test 1: Imports
    if not test_imports():
        print("\n✗ Quickstart failed at imports")
        return
    
    # Test 2: Configuration
    success, config = test_configuration()
    if not success:
        print("\n✗ Quickstart failed at configuration")
        return
    
    # Test 3: Simulator
    if not test_simulator(config):
        print("\n✗ Quickstart failed at simulator")
        return
    
    # Test 4: Dataset
    success, train_loader, val_loader = test_dataset(config)
    if not success:
        print("\n✗ Quickstart failed at dataset")
        return
    
    # Test 5: Model
    success, model = test_model(config)
    if not success:
        print("\n✗ Quickstart failed at model")
        return
    
    # Test 6: Training
    if not test_training_step(config, model, train_loader):
        print("\n✗ Quickstart failed at training")
        return
    
    # Test 7: Evaluation
    if not test_evaluation(config, model, val_loader):
        print("\n✗ Quickstart failed at evaluation")
        return
    
    # Success!
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED!")
    print("="*80)
    print(f"\nTotal time: {elapsed_time:.2f} seconds")
    print("\nYour environment is ready! Next steps:")
    print("  1. Train a model:")
    print("     python scripts/train_model.py --experiment_name my_experiment")
    print("\n  2. Or run a full small-scale test:")
    print("     python scripts/train_model.py --train_size 1000 --val_size 200 --epochs 5")
    print("\n  3. Monitor training:")
    print("     tensorboard --logdir logs/")
    print("="*80)


if __name__ == '__main__':
    main()
