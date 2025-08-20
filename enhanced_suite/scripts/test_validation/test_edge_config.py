#!/usr/bin/env python3
"""
Edge Detection Dataset Configuration Tester

This script helps users test their dataset configuration before training.
It verifies the dataset structure and provides sample outputs for debugging.
"""

import os
import sys
import yaml
import numpy as np
from PIL import Image
import torch
from pathlib import Path

def test_dataset_structure(data_root, verbose=True):
    """Test if the dataset structure is correct"""
    if verbose:
        print(f"Testing dataset structure at: {data_root}")
    
    required_paths = [
        os.path.join(data_root, 'image', 'raw'),
        os.path.join(data_root, 'edge', 'raw')
    ]
    
    missing_paths = []
    for path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(path)
    
    if missing_paths:
        print(f"❌ Missing required paths:")
        for path in missing_paths:
            print(f"   - {path}")
        return False
    
    if verbose:
        print("✅ Dataset structure is correct")
    
    return True

def test_file_extensions(data_root, verbose=True):
    """Test if files have correct extensions"""
    if verbose:
        print("Testing file extensions...")
    
    image_raw_path = os.path.join(data_root, 'image', 'raw')
    edge_raw_path = os.path.join(data_root, 'edge', 'raw')
    
    supported_extensions = {'.jpg', '.jpeg', '.png', '.pgm', '.ppm'}
    
    issues = []
    
    # Check image files
    for subset_dir in os.listdir(image_raw_path):
        subset_path = os.path.join(image_raw_path, subset_dir)
        if os.path.isdir(subset_path):
            for filename in os.listdir(subset_path):
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext not in supported_extensions:
                    issues.append(f"Unsupported image file: {filename}")
    
    # Check edge files
    for subset_dir in os.listdir(edge_raw_path):
        subset_path = os.path.join(edge_raw_path, subset_dir)
        if os.path.isdir(subset_path):
            for filename in os.listdir(subset_path):
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext not in supported_extensions:
                    issues.append(f"Unsupported edge file: {filename}")
    
    if issues:
        print("❌ File extension issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    if verbose:
        print("✅ All files have supported extensions")
    
    return True

def test_image_loading(data_root, verbose=True):
    """Test if images can be loaded correctly"""
    if verbose:
        print("Testing image loading...")
    
    image_raw_path = os.path.join(data_root, 'image', 'raw')
    edge_raw_path = os.path.join(data_root, 'edge', 'raw')
    
    # Find first subset directory
    subsets = [d for d in os.listdir(image_raw_path) if os.path.isdir(os.path.join(image_raw_path, d))]
    if not subsets:
        print("❌ No subset directories found")
        return False
    
    first_subset = subsets[0]
    image_files = os.listdir(os.path.join(image_raw_path, first_subset))
    edge_files = os.listdir(os.path.join(edge_raw_path, first_subset))
    
    if not image_files:
        print("❌ No image files found")
        return False
    
    if not edge_files:
        print("❌ No edge files found")
        return False
    
    # Test loading first image-edge pair
    try:
        # Test image loading
        image_path = os.path.join(image_raw_path, first_subset, image_files[0])
        img = Image.open(image_path).convert('RGB')
        if verbose:
            print(f"✅ Image loaded successfully: {img.size} {img.mode}")
        
        # Test edge loading
        edge_path = os.path.join(edge_raw_path, first_subset, edge_files[0])
        edge = Image.open(edge_path).convert('L')
        if verbose:
            print(f"✅ Edge map loaded successfully: {edge.size} {edge.mode}")
        
        # Test conversion to numpy
        img_array = np.array(img)
        edge_array = np.array(edge)
        
        if verbose:
            print(f"✅ Arrays created: img={img_array.shape}, edge={edge_array.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading images: {e}")
        return False

def test_config_file(config_path, verbose=True):
    """Test if configuration file is valid"""
    if verbose:
        print(f"Testing configuration file: {config_path}")
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ['data']
        for field in required_fields:
            if field not in config:
                print(f"❌ Missing required field: {field}")
                return False
        
        # Check data configuration
        data_config = config['data']
        if 'name' not in data_config or data_config['name'] != 'edge':
            print("❌ Data configuration must have name: 'edge'")
            return False
        
        if verbose:
            print("✅ Configuration file is valid")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False

def test_dataset_import(verbose=True):
    """Test if dataset classes can be imported"""
    if verbose:
        print("Testing dataset imports...")
    
    try:
        sys.path.append('.')
        from denoising_diffusion_pytorch.data import AdaptEdgeDataset, EdgeDataset, EdgeDatasetTest
        
        if verbose:
            print("✅ Dataset classes imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main testing function"""
    print("🔍 Edge Detection Dataset Configuration Tester")
    print("=" * 50)
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python test_edge_config.py <data_root> [config_path]")
        print("Example: python test_edge_config.py /path/to/dataset configs_edge/edge_vae_train.yaml")
        sys.exit(1)
    
    data_root = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Run tests
    tests = [
        ("Dataset Structure", lambda: test_dataset_structure(data_root)),
        ("File Extensions", lambda: test_file_extensions(data_root)),
        ("Image Loading", lambda: test_image_loading(data_root)),
        ("Dataset Import", test_dataset_import),
    ]
    
    if config_path:
        tests.append(("Configuration File", lambda: test_config_file(config_path)))
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("🎉 All tests passed! Your dataset is ready for training.")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues before training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())