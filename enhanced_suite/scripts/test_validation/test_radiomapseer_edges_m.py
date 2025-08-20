#!/usr/bin/env python3
"""
Test script for RadioMapSeer edge detection with preserved structure

This script tests the modified edge detection functionality that preserves
the original RadioMapSeer dataset structure without train/validation splits.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from radiomapseer_edge_detection_m import RadioMapSeerEdgeDataset, EdgeDetector, SyntheticEdgeGenerator

def test_flat_structure(data_root: str, output_dir: str, max_files: int = 30):
    """Test edge detection with flat structure"""
    
    print("Testing edge detection with flat structure...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset
    try:
        dataset = RadioMapSeerEdgeDataset(data_root, image_size=(256, 256))
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return 1
    
    print(f"Found {len(dataset.dpm_files)} DPM images")
    
    # Limit to first few files for testing
    test_files = dataset.dpm_files[:max_files]
    print(f"Testing {len(test_files)} files")
    
    # Create flat directory structure
    (output_path / 'image').mkdir(parents=True, exist_ok=True)
    (output_path / 'edge').mkdir(parents=True, exist_ok=True)
    
    # Process each file
    for i, dpm_file in enumerate(test_files):
        try:
            # Load DPM image
            dpm_image = np.array(Image.open(dpm_file))
            
            # Generate edge map
            edge_map = dataset.edge_detector.detect_edges(dpm_image)
            
            # Resize images
            dpm_resized = cv2.resize(dpm_image, (256, 256))
            edge_resized = cv2.resize(edge_map, (256, 256))
            
            # Save images
            image_path = output_path / 'image' / f'{dpm_file.stem}.png'
            edge_path = output_path / 'edge' / f'{dpm_file.stem}.png'
            
            Image.fromarray(dpm_resized).save(image_path)
            Image.fromarray(edge_resized).save(edge_path)
            
            print(f"  Processed {dpm_file.name}")
            
        except Exception as e:
            print(f"  Error processing {dpm_file}: {e}")
            continue
    
    # Verify structure
    print("\nVerifying created structure:")
    image_count = len(list((output_path / 'image').glob('*.png')))
    edge_count = len(list((output_path / 'edge').glob('*.png')))
    print(f"  Total: {image_count} images, {edge_count} edges")
    
    print(f"\nTest completed! Results saved to: {output_path}")
    return 0

def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description='Test RadioMapSeer edge detection with flat structure')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to RadioMapSeer dataset')
    parser.add_argument('--output_dir', type=str, default='./radiomapseer_edge_test_flat',
                       help='Output directory for test results')
    parser.add_argument('--max_files', type=int, default=30,
                       help='Maximum number of files to test')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data_root):
        print(f"Error: Dataset directory {args.data_root} does not exist")
        return 1
    
    try:
        return test_flat_structure(args.data_root, args.output_dir, args.max_files)
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())