#!/usr/bin/env python3
"""
Test script for RadioMapSeer edge detection implementation

This script tests the edge detection functionality on sample DPM images
from the RadioMapSeer dataset and visualizes the results.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from radiomapseer_edge_detection import RadioMapSeerEdgeDataset, EdgeDetector, SyntheticEdgeGenerator

def test_edge_detection(data_root: str, output_dir: str, num_samples: int = 5):
    """Test edge detection on sample images"""
    
    print("Testing edge detection on RadioMapSeer dataset...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Test different edge detection methods
    methods = ['canny', 'sobel', 'laplacian', 'prewitt']
    
    # Initialize dataset
    try:
        dataset = RadioMapSeerEdgeDataset(data_root, image_size=(256, 256))
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return 1
    
    # Test each method
    for method in methods:
        print(f"\nTesting {method} edge detection...")
        
        method_output = output_path / method
        method_output.mkdir(exist_ok=True)
        
        # Create edge detector
        edge_detector = EdgeDetector(method)
        
        # Process sample images
        for i in range(min(num_samples, len(dataset.dpm_files))):
            dpm_file = dataset.dpm_files[i]
            
            try:
                # Load image
                dpm_image = np.array(Image.open(dpm_file))
                
                # Detect edges
                edges = edge_detector.detect_edges(dpm_image)
                
                # Create visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                
                ax1.imshow(dpm_image, cmap='gray')
                ax1.set_title('Original DPM Image')
                ax1.axis('off')
                
                ax2.imshow(edges, cmap='gray')
                ax2.set_title(f'{method.capitalize()} Edges')
                ax2.axis('off')
                
                plt.tight_layout()
                
                # Save visualization
                viz_path = method_output / f'sample_{i}_comparison.png'
                plt.savefig(viz_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Save individual images
                orig_path = method_output / f'sample_{i}_original.png'
                edge_path = method_output / f'sample_{i}_edges.png'
                
                Image.fromarray(dpm_image).save(orig_path)
                Image.fromarray(edges).save(edge_path)
                
                print(f"  Processed sample {i+1}/{num_samples}")
                
            except Exception as e:
                print(f"  Error processing sample {i}: {e}")
                continue
    
    print(f"\nEdge detection test completed!")
    print(f"Results saved to: {output_path}")
    
    return 0

def test_synthetic_edges(data_root: str, output_dir: str, num_samples: int = 3):
    """Test synthetic edge generation"""
    
    print("Testing synthetic edge generation...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Test different synthetic edge types
    edge_types = ['gradient', 'contour', 'ridge']
    
    # Initialize dataset
    try:
        dataset = RadioMapSeerEdgeDataset(data_root, image_size=(256, 256))
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return 1
    
    # Test each synthetic edge type
    for edge_type in edge_types:
        print(f"\nTesting {edge_type} synthetic edges...")
        
        type_output = output_path / edge_type
        type_output.mkdir(exist_ok=True)
        
        # Create synthetic edge generator
        generator = SyntheticEdgeGenerator((256, 256))
        
        # Process sample images
        for i in range(min(num_samples, len(dataset.dpm_files))):
            dpm_file = dataset.dpm_files[i]
            
            try:
                # Load image
                dpm_image = np.array(Image.open(dpm_file))
                
                # Generate synthetic edges
                edges = generator.generate_synthetic_edges(dpm_image, edge_type)
                
                # Create visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                
                ax1.imshow(dpm_image, cmap='gray')
                ax1.set_title('Original DPM Image')
                ax1.axis('off')
                
                ax2.imshow(edges, cmap='gray')
                ax2.set_title(f'{edge_type.capitalize()} Synthetic Edges')
                ax2.axis('off')
                
                plt.tight_layout()
                
                # Save visualization
                viz_path = type_output / f'sample_{i}_comparison.png'
                plt.savefig(viz_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Save individual images
                orig_path = type_output / f'sample_{i}_original.png'
                edge_path = type_output / f'sample_{i}_synthetic_edges.png'
                
                Image.fromarray(dpm_image).save(orig_path)
                Image.fromarray(edges).save(edge_path)
                
                print(f"  Processed sample {i+1}/{num_samples}")
                
            except Exception as e:
                print(f"  Error processing sample {i}: {e}")
                continue
    
    print(f"\nSynthetic edge generation test completed!")
    print(f"Results saved to: {output_path}")
    
    return 0

def create_sample_dataset(data_root: str, output_dir: str, num_samples: int = 10):
    """Create a small sample dataset for testing"""
    
    print("Creating sample edge detection dataset...")
    
    # Initialize dataset
    try:
        dataset = RadioMapSeerEdgeDataset(data_root, image_size=(256, 256), edge_method='canny')
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return 1
    
    # Create sample dataset
    dataset.create_edge_dataset(output_dir, split_ratio=0.8)
    
    print(f"Sample dataset created at: {output_dir}")
    
    # Verify dataset structure
    expected_dirs = [
        Path(output_dir) / 'image' / 'raw' / 'train',
        Path(output_dir) / 'image' / 'raw' / 'val',
        Path(output_dir) / 'edge' / 'raw' / 'train',
        Path(output_dir) / 'edge' / 'raw' / 'val'
    ]
    
    for dir_path in expected_dirs:
        if dir_path.exists():
            files = list(dir_path.glob('*.png'))
            print(f"  {dir_path}: {len(files)} files")
        else:
            print(f"  {dir_path}: MISSING")
    
    return 0

def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description='Test RadioMapSeer edge detection')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to RadioMapSeer dataset')
    parser.add_argument('--output_dir', type=str, default='./enhanced_suite/archive/radiomapseer_edge_test_results',
                       help='Output directory for test results')
    parser.add_argument('--test_type', type=str, default='all',
                       choices=['all', 'edge_detection', 'synthetic', 'dataset'],
                       help='Type of test to run')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to test')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data_root):
        print(f"Error: Dataset directory {args.data_root} does not exist")
        return 1
    
    try:
        if args.test_type == 'all':
            print("Running all tests...")
            
            # Test edge detection
            edge_output = Path(args.output_dir) / 'edge_detection'
            test_edge_detection(args.data_root, edge_output, args.num_samples)
            
            # Test synthetic edges
            synthetic_output = Path(args.output_dir) / 'synthetic'
            test_synthetic_edges(args.data_root, synthetic_output, args.num_samples)
            
            # Create sample dataset
            dataset_output = Path(args.output_dir) / 'sample_dataset'
            create_sample_dataset(args.data_root, dataset_output, args.num_samples)
            
        elif args.test_type == 'edge_detection':
            test_edge_detection(args.data_root, args.output_dir, args.num_samples)
            
        elif args.test_type == 'synthetic':
            test_synthetic_edges(args.data_root, args.output_dir, args.num_samples)
            
        elif args.test_type == 'dataset':
            create_sample_dataset(args.data_root, args.output_dir, args.num_samples)
        
        print(f"\nAll tests completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())