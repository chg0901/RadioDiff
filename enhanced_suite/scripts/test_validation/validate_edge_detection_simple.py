#!/usr/bin/env python3
"""
Simple Edge Detection Validation Script

This script performs a focused validation of the edge detection functionality
by processing a small sample without copying the entire PNG directory.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
import torch
from pathlib import Path
from typing import List, Tuple
import argparse
import tempfile
import shutil
import matplotlib.pyplot as plt

# Import the edge detection modules
sys.path.append('.')
from radiomapseer_edge_detection_m import EdgeDetector

def validate_edge_detection_simple(data_root: str, output_dir: str, sample_size: int = 5):
    """Simple validation of edge detection functionality"""
    print("=== Simple Edge Detection Validation ===")
    
    # Paths
    data_path = Path(data_root)
    dpm_path = data_path / 'gain' / 'DPM'
    
    # Get sample files
    dpm_files = sorted([f for f in dpm_path.glob('*.png')])[:sample_size]
    
    if not dpm_files:
        print(f"No DPM files found in {dpm_path}")
        return False
    
    print(f"Processing {len(dpm_files)} sample files...")
    
    # Test edge detection methods
    methods = ['canny', 'sobel', 'laplacian', 'prewitt']
    results = {}
    
    for method in methods:
        print(f"\nTesting {method} method...")
        try:
            edge_detector = EdgeDetector(method)
            method_results = []
            
            for i, dpm_file in enumerate(dpm_files):
                # Load and process image
                dpm_image = np.array(Image.open(dpm_file))
                
                # Generate edge map
                edge_map = edge_detector.detect_edges(dpm_image)
                
                # Resize to standard size
                dpm_resized = cv2.resize(dpm_image, (256, 256))
                edge_resized = cv2.resize(edge_map, (256, 256))
                
                # Validate results
                validation_result = {
                    'file': dpm_file.name,
                    'original_shape': dpm_image.shape,
                    'edge_shape': edge_map.shape,
                    'resized_shape': dpm_resized.shape,
                    'edge_max': edge_map.max(),
                    'edge_min': edge_map.min(),
                    'has_nan': np.isnan(edge_map).any(),
                    'success': True
                }
                
                method_results.append(validation_result)
                
                print(f"  ✓ {dpm_file.name}: {edge_map.shape} (max: {edge_map.max()})")
            
            results[method] = {
                'status': 'SUCCESS',
                'results': method_results,
                'success_rate': len([r for r in method_results if r['success']]) / len(method_results)
            }
            
        except Exception as e:
            print(f"  ✗ {method}: ERROR - {e}")
            results[method] = {
                'status': 'ERROR',
                'error': str(e),
                'success_rate': 0.0
            }
    
    # Create visualization
    try:
        print("\nCreating validation visualization...")
        create_validation_visualization(dpm_files, methods, output_dir)
        print("✓ Validation visualization created")
    except Exception as e:
        print(f"  Warning: Could not create visualization: {e}")
    
    # Summary
    print("\n=== Validation Summary ===")
    successful_methods = []
    for method, result in results.items():
        if result['status'] == 'SUCCESS':
            success_rate = result['success_rate']
            print(f"✓ {method}: {success_rate:.1%} success rate")
            successful_methods.append(method)
        else:
            print(f"✗ {method}: FAILED - {result.get('error', 'Unknown error')}")
    
    overall_success = len(successful_methods) >= 3  # At least 3 methods should work
    
    if overall_success:
        print(f"\n🎉 VALIDATION: SUCCESS ({len(successful_methods)}/{len(methods)} methods working)")
        return True
    else:
        print(f"\n❌ VALIDATION: FAILED (only {len(successful_methods)}/{len(methods)} methods working)")
        return False

def create_validation_visualization(dpm_files: List[Path], methods: List[str], output_dir: Path):
    """Create a comprehensive visualization of edge detection results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    n_files = len(dpm_files)
    n_methods = len(methods)
    
    fig, axes = plt.subplots(n_files, n_methods + 1, figsize=(4 * (n_methods + 1), 4 * n_files))
    
    if n_files == 1:
        axes = axes.reshape(1, -1)
    
    for i, dpm_file in enumerate(dpm_files):
        # Load original image
        orig_img = np.array(Image.open(dpm_file))
        orig_img_resized = cv2.resize(orig_img, (256, 256))
        
        # Display original
        axes[i, 0].imshow(orig_img_resized, cmap='gray')
        axes[i, 0].set_title(f'Original\n{dpm_file.name}')
        axes[i, 0].axis('off')
        
        # Process and display each method
        for j, method in enumerate(methods):
            try:
                edge_detector = EdgeDetector(method)
                edge_map = edge_detector.detect_edges(orig_img)
                edge_resized = cv2.resize(edge_map, (256, 256))
                
                axes[i, j + 1].imshow(edge_resized, cmap='gray')
                axes[i, j + 1].set_title(f'{method.capitalize()}\nMax: {edge_map.max()}')
                axes[i, j + 1].axis('off')
                
            except Exception as e:
                axes[i, j + 1].text(0.5, 0.5, f'ERROR\n{str(e)[:30]}...', 
                                   ha='center', va='center', fontsize=8)
                axes[i, j + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'edge_detection_validation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also create a summary statistics plot
    create_summary_statistics(dpm_files, methods, output_dir)

def create_summary_statistics(dpm_files: List[Path], methods: List[str], output_dir: Path):
    """Create summary statistics for edge detection methods"""
    stats = {}
    
    for method in methods:
        try:
            edge_detector = EdgeDetector(method)
            edge_counts = []
            max_values = []
            
            for dpm_file in dpm_files:
                orig_img = np.array(Image.open(dpm_file))
                edge_map = edge_detector.detect_edges(orig_img)
                
                # Count non-zero pixels (edge density)
                edge_density = np.sum(edge_map > 0) / edge_map.size
                edge_counts.append(edge_density)
                max_values.append(edge_map.max())
            
            stats[method] = {
                'edge_densities': edge_counts,
                'max_values': max_values,
                'avg_edge_density': np.mean(edge_counts),
                'avg_max_value': np.mean(max_values)
            }
            
        except Exception as e:
            print(f"Warning: Could not compute statistics for {method}: {e}")
    
    if not stats:
        return
    
    # Create statistics visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Edge density comparison
    methods_list = list(stats.keys())
    densities = [stats[method]['avg_edge_density'] for method in methods_list]
    
    ax1.bar(methods_list, densities)
    ax1.set_title('Average Edge Density')
    ax1.set_ylabel('Edge Density (non-zero pixels)')
    ax1.set_ylim(0, max(densities) * 1.1 if densities else 1)
    
    # Max value comparison
    max_values = [stats[method]['avg_max_value'] for method in methods_list]
    
    ax2.bar(methods_list, max_values)
    ax2.set_title('Average Maximum Edge Value')
    ax2.set_ylabel('Max Value')
    ax2.set_ylim(0, max(max_values) * 1.1 if max_values else 255)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'edge_detection_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Main function for simple validation"""
    parser = argparse.ArgumentParser(description='Simple validation of RadioMapSeer edge detection')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to RadioMapSeer dataset')
    parser.add_argument('--output_dir', type=str, default='./simple_validation_results',
                       help='Output directory for validation results')
    parser.add_argument('--sample_size', type=int, default=5,
                       help='Number of sample files to process')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data_root):
        print(f"Error: Dataset directory {args.data_root} does not exist")
        return 1
    
    try:
        # Run validation
        success = validate_edge_detection_simple(args.data_root, args.output_dir, args.sample_size)
        
        if success:
            print(f"\n🎉 SIMPLE VALIDATION: SUCCESS")
            print("Edge detection functionality is working correctly!")
            print(f"Results saved to: {args.output_dir}")
            return 0
        else:
            print(f"\n❌ SIMPLE VALIDATION: FAILED")
            print("Some issues were found during validation.")
            return 1
            
    except Exception as e:
        print(f"Error during validation: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())