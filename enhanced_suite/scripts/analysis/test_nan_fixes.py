#!/usr/bin/env python3
"""
Test script to validate NaN fixes for RadioDiff training resume functionality.
This script tests the enhanced checkpoint loading and NaN detection mechanisms.
"""

import os
import sys
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/home/cine/Documents/Github/RadioDiff')

def create_test_checkpoint():
    """Create a test checkpoint with various edge cases"""
    print("Creating test checkpoint...")
    
    # Create a temporary directory for testing
    test_dir = Path(tempfile.mkdtemp(prefix="radiodiff_test_"))
    print(f"Test directory: {test_dir}")
    
    # Create a simple model state with potential issues
    model_state = {
        'model.conv.weight': torch.randn(64, 3, 3, 3),
        'model.conv.bias': torch.randn(64),
        'scale_factor': 0.18215,  # Standard VAE scaling factor
        # Add some problematic values for testing
        'problematic_param': torch.tensor([float('nan'), float('inf')]),
    }
    
    # Create optimizer state
    optimizer_state = {
        'state': {},
        'param_groups': [{'lr': 1e-4, 'weight_decay': 1e-4}]
    }
    
    # Create checkpoint data
    checkpoint_data = {
        'step': 1000,
        'model': model_state,
        'opt': optimizer_state,
        'lr_scheduler': {},
        'ema': {},
        'scaler': {},
    }
    
    # Save checkpoint
    checkpoint_path = test_dir / 'model-5.pt'
    torch.save(checkpoint_data, checkpoint_path)
    
    return test_dir, checkpoint_path

def test_checkpoint_loading():
    """Test the enhanced checkpoint loading functionality"""
    print("\n" + "="*60)
    print("TESTING CHECKPOINT LOADING")
    print("="*60)
    
    test_dir, checkpoint_path = create_test_checkpoint()
    
    try:
        # Import the trainer
        from train_cond_ldm import Trainer
        
        # Mock configuration for testing
        class MockConfig:
            def __init__(self):
                self.trainer = MockTrainerConfig()
        
        class MockTrainerConfig:
            def __init__(self):
                self.enable_resume = True
                self.ema_update_after_step = 10000
                self.ema_update_every = 10
                self.min_lr = 1e-6
        
        # Mock model and accelerator
        class MockModel:
            def __init__(self):
                self.scale_factor = None
                self.parameters = lambda: []
                
            def load_state_dict(self, state_dict, strict=True):
                print(f"Mock model loading state dict with keys: {list(state_dict.keys())}")
                
            def named_parameters(self):
                return []
        
        class MockAccelerator:
            def __init__(self):
                self.device = torch.device('cpu')
                self.is_main_process = True
                self.scaler = None
                
            def unwrap_model(self, model):
                return model
                
            def prepare(self, *args):
                return args
        
        class MockOptimizer:
            def __init__(self):
                self.param_groups = [{'lr': 1e-4}]
                
            def load_state_dict(self, state):
                print(f"Mock optimizer loading state")
                
            def state_dict(self):
                return {}
        
        class MockScheduler:
            def load_state_dict(self, state):
                print(f"Mock scheduler loading state")
                
            def state_dict(self):
                return {}
        
        class MockEMA:
            def load_state_dict(self, state):
                print(f"Mock EMA loading state")
                
            def state_dict(self):
                return {}
        
        # Create trainer instance
        trainer = Trainer.__new__(Trainer)
        trainer.enable_resume = True
        trainer.accelerator = MockAccelerator()
        trainer.model = MockModel()
        trainer.opt = MockOptimizer()
        trainer.lr_scheduler = MockScheduler()
        trainer.ema = MockEMA()
        trainer.results_folder = test_dir
        trainer.step = 0
        
        # Test the enhanced load method
        print(f"Testing enhanced load method with checkpoint: {checkpoint_path}")
        trainer.load(5)
        
        print("✓ Checkpoint loading test passed")
        result = True
        
    except Exception as e:
        print(f"✗ Checkpoint loading test failed: {e}")
        result = False
        
    finally:
        # Cleanup
        shutil.rmtree(test_dir)
    
    return result

def test_nan_detection():
    """Test NaN detection functionality"""
    print("\n" + "="*60)
    print("TESTING NaN DETECTION")
    print("="*60)
    
    try:
        # Import trainer components
        from train_cond_ldm import Trainer
        
        # Create mock trainer instance
        trainer = Trainer.__new__(Trainer)
        trainer.step = 100
        
        # Test batch validation
        batch_with_nan = {
            'image': torch.tensor([[[[1.0, float('nan'), 2.0]]]]),
            'cond': torch.tensor([[[[1.0, 2.0, 3.0]]]]),
        }
        
        print("Testing batch validation with NaN...")
        trainer._validate_batch_data(batch_with_nan)
        
        # Test loss detection
        nan_loss = torch.tensor(float('nan'))
        log_dict = {'train/loss_simple': 0.1, 'train/loss_vlb': 0.05}
        
        print("Testing loss NaN detection...")
        issues = trainer._detect_training_issues(nan_loss, log_dict, batch_with_nan)
        
        if issues:
            print("✓ NaN detection working correctly")
        else:
            print("✗ NaN detection failed to identify NaN loss")
            return False
            
        # Test Inf detection
        inf_loss = torch.tensor(float('inf'))
        print("Testing loss Inf detection...")
        issues = trainer._detect_training_issues(inf_loss, log_dict, batch_with_nan)
        
        if issues:
            print("✓ Inf detection working correctly")
        else:
            print("✗ Inf detection failed to identify Inf loss")
            return False
            
        print("✓ NaN detection tests passed")
        return True
        
    except Exception as e:
        print(f"✗ NaN detection test failed: {e}")
        return False

def test_gradient_stability():
    """Test gradient stability measures"""
    print("\n" + "="*60)
    print("TESTING GRADIENT STABILITY")
    print("="*60)
    
    try:
        from train_cond_ldm import Trainer
        
        # Create mock trainer
        trainer = Trainer.__new__(Trainer)
        trainer.step = 100
        
        # Create mock model with problematic gradients
        class MockParam:
            def __init__(self, grad_data):
                self.grad = grad_data
        
        class MockModel:
            def __init__(self):
                self.param1 = MockParam(torch.tensor([1.0, float('nan'), 2.0]))
                self.param2 = MockParam(torch.tensor([float('inf'), 1.0, 2.0]))
                
            def parameters(self):
                return [self.param1, self.param2]
        
        trainer.model = MockModel()
        
        print("Testing gradient stability measures...")
        grad_norm = trainer._apply_gradient_stability_measures()
        
        # Check if NaN gradients were handled
        if not torch.isnan(trainer.model.param1.grad).any():
            print("✓ NaN gradients handled correctly")
        else:
            print("✗ NaN gradients not handled")
            return False
            
        # Check if Inf gradients were handled
        if not torch.isinf(trainer.model.param2.grad).any():
            print("✓ Inf gradients handled correctly")
        else:
            print("✗ Inf gradients not handled")
            return False
            
        print("✓ Gradient stability tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Gradient stability test failed: {e}")
        return False

def test_model_validation():
    """Test model parameter validation"""
    print("\n" + "="*60)
    print("TESTING MODEL VALIDATION")
    print("="*60)
    
    try:
        from train_cond_ldm import Trainer
        
        # Create mock trainer
        trainer = Trainer.__new__(Trainer)
        trainer.step = 100
        
        # Create mock model with problematic parameters
        class MockModel:
            def __init__(self):
                self.good_param = torch.nn.Parameter(torch.randn(10))
                self.nan_param = torch.nn.Parameter(torch.tensor([1.0, float('nan'), 2.0]))
                self.inf_param = torch.nn.Parameter(torch.tensor([float('inf'), 1.0, 2.0]))
                
            def named_parameters(self):
                return [
                    ('good_param', self.good_param),
                    ('nan_param', self.nan_param),
                    ('inf_param', self.inf_param),
                ]
        
        model = MockModel()
        
        print("Testing model parameter validation...")
        trainer._validate_model_parameters(model)
        
        print("✓ Model validation test passed")
        return True
        
    except Exception as e:
        print(f"✗ Model validation test failed: {e}")
        return False

def run_all_tests():
    """Run all test cases"""
    print("RadioDiff NaN Fix Validation Tests")
    print("="*60)
    
    tests = [
        ("Checkpoint Loading", test_checkpoint_loading),
        ("NaN Detection", test_nan_detection),
        ("Gradient Stability", test_gradient_stability),
        ("Model Validation", test_model_validation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The NaN fixes are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)