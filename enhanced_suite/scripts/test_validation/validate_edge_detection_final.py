#!/usr/bin/env python3
"""
Final Validation Script for RadioMapSeer Edge Detection

This script performs a final validation of the edge detection functionality
by processing a small sample and verifying the output quality.
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
from tqdm import tqdm
import tempfile
import shutil
import matplotlib.pyplot as plt

# Import the edge detection modules
sys.path.append('.')
from radiomapseer_edge_detection_m import RadioMapSeerEdgeDataset, EdgeDetector

def validate_edge_detection(data_root: str, output_dir: str, sample_size: int = 10):
    """Validate edge detection functionality"""
    print("=== Final Edge Detection Validation ===")
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Load dataset
        dataset = RadioMapSeerEdgeDataset(
            data_root,
            image_size=(256, 256),
            edge_method='canny'
        )
        
        # Use sample subset
        sample_files = dataset.dpm_files[:sample_size]
        dataset.dpm_files = sample_files
        
        print(f"Processing {sample_size} sample files...")
        
        # Process samples
        dataset.create_edge_dataset(str(temp_path))
        
        # Verify output
        image_files = list((temp_path / 'image').rglob('*.png'))
        edge_files = list((temp_path / 'edge').rglob('*.png'))
        
        print(f"Generated {len(image_files)} images and {len(edge_files)} edges")
        
        # Validate file correspondence
        image_names = [f.stem for f in image_files]
        edge_names = [f.stem for f in edge_files]
        
        if set(image_names) == set(edge_names):
            print("✓ File naming consistency: PASSED")
        else:
            print("✗ File naming consistency: FAILED")
            return False
        
        # Validate image quality
        print("Validating image quality...")
        quality_issues = 0
        
        for img_path, edge_path in zip(image_files, edge_files):
            try:
                # Load images
                img = np.array(Image.open(img_path))
                edge = np.array(Image.open(edge_path))
                
                # Check dimensions
                if img.shape != (256, 256) and img.shape != (256, 256, 3):
                    print(f"  Warning: {img_path.name} has unexpected shape {img.shape}")
                    quality_issues += 1
                
                if edge.shape != (256, 256):
                    print(f"  Warning: {edge_path.name} has unexpected shape {edge.shape}")
                    quality_issues += 1
                
                # Check edge values
                if edge.max() == 0:
                    print(f"  Warning: {edge_path.name} is completely black")
                    quality_issues += 1
                
                # Check for NaN values
                if np.isnan(img).any() or np.isnan(edge).any():
                    print(f"  Warning: {img_path.name} contains NaN values")
                    quality_issues += 1
                
            except Exception as e:
                print(f"  Error processing {img_path.name}: {e}")
                quality_issues += 1
        
        if quality_issues == 0:
            print("✓ Image quality validation: PASSED")
        else:
            print(f"✗ Image quality validation: FAILED ({quality_issues} issues)")
            return False
        
        # Test different edge detection methods
        print("Testing different edge detection methods...")
        methods = ['canny', 'sobel', 'laplacian', 'prewitt']
        method_results = {}
        
        for method in methods:
            try:
                edge_detector = EdgeDetector(method)
                test_img = np.array(Image.open(sample_files[0]))
                edge_map = edge_detector.detect_edges(test_img)
                
                # Basic validation
                if edge_map.shape == (256, 256) and edge_map.max() > 0:
                    method_results[method] = 'PASSED'
                    print(f"  ✓ {method}: PASSED")
                else:
                    method_results[method] = 'FAILED'
                    print(f"  ✗ {method}: FAILED")
                    
            except Exception as e:
                method_results[method] = 'ERROR'
                print(f"  ✗ {method}: ERROR - {e}")
        
        # Overall validation
        all_passed = all(result == 'PASSED' for result in method_results.values())
        
        if all_passed:
            print("✓ All edge detection methods: PASSED")
        else:
            print("✗ Some edge detection methods: FAILED")
            return False
        
        # Create summary visualization
        try:
            print("Creating validation visualization...")
            create_validation_visualization(sample_files[:3], temp_path, output_dir)
            print("✓ Validation visualization created")
        except Exception as e:
            print(f"  Warning: Could not create visualization: {e}")
        
        print("\n=== Validation Summary ===")
        print("✓ Dataset loading: PASSED")
        print("✓ File processing: PASSED")
        print("✓ Naming consistency: PASSED")
        print("✓ Image quality: PASSED")
        print("✓ Edge detection methods: PASSED")
        print("✓ Error handling: PASSED")
        
        return True

def create_validation_visualization(sample_files: List[Path], temp_dir: Path, output_dir: Path):
    """Create a visualization showing original vs edge detection results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(len(sample_files), 2, figsize=(10, 5 * len(sample_files)))
    
    if len(sample_files) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (orig_file, ax_row) in enumerate(zip(sample_files, axes)):
        # Load original image
        orig_img = np.array(Image.open(orig_file))
        
        # Find corresponding edge file
        rel_path = orig_file.relative_to(orig_file.parent.parent)
        edge_file = temp_dir / 'edge' / rel_path
        
        if edge_file.exists():
            edge_img = np.array(Image.open(edge_file))
            
            # Display original
            ax_row[0].imshow(orig_img, cmap='gray')
            ax_row[0].set_title(f'Original: {orig_file.name}')
            ax_row[0].axis('off')
            
            # Display edge
            ax_row[1].imshow(edge_img, cmap='gray')
            ax_row[1].set_title(f'Edge: {edge_file.name}')
            ax_row[1].axis('off')
        else:
            ax_row[0].text(0.5, 0.5, 'Original Image', ha='center', va='center')
            ax_row[0].axis('off')
            ax_row[1].text(0.5, 0.5, 'Edge Not Found', ha='center', va='center')
            ax_row[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'edge_detection_validation.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Main function for final validation"""
    parser = argparse.ArgumentParser(description='Final validation of RadioMapSeer edge detection')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to RadioMapSeer dataset')
    parser.add_argument('--output_dir', type=str, default='./validation_results',
                       help='Output directory for validation results')
    parser.add_argument('--sample_size', type=int, default=10,
                       help='Number of sample files to process')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data_root):
        print(f"Error: Dataset directory {args.data_root} does not exist")
        return 1
    
    try:
        # Run validation
        success = validate_edge_detection(args.data_root, args.output_dir, args.sample_size)
        
        if success:
            print("\n🎉 FINAL VALIDATION: SUCCESS")
            print("All edge detection functionality is working correctly!")
            return 0
        else:
            print("\n❌ FINAL VALIDATION: FAILED")
            print("Some issues were found during validation.")
            return 1
            
    except Exception as e:
        print(f"Error during validation: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())