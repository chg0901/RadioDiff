#!/usr/bin/env python3
"""
Edge Detection Implementation for RadioMapSeer Dataset

This script implements edge detection functionality for the RadioMapSeer dataset
without requiring a specific edge dataset. It uses the DPM (Gain) data as input
and generates edge maps using various edge detection algorithms.

The script can:
1. Process DPM images to generate edge maps
2. Create a compatible dataset structure for edge detection training
3. Support multiple edge detection algorithms (Canny, Sobel, Laplacian)
4. Generate synthetic edge datasets from existing radio maps
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import argparse
from tqdm import tqdm
import shutil

class EdgeDetector:
    """Edge detection class with multiple algorithms"""
    
    def __init__(self, method: str = 'canny', **kwargs):
        """
        Initialize edge detector
        
        Args:
            method: Edge detection method ('canny', 'sobel', 'laplacian', 'prewitt')
            **kwargs: Method-specific parameters
        """
        self.method = method
        self.params = kwargs
        
        # Set default parameters
        if method == 'canny':
            self.params.setdefault('threshold1', 50)
            self.params.setdefault('threshold2', 150)
            self.params.setdefault('aperture_size', 3)
        elif method == 'sobel':
            self.params.setdefault('ksize', 3)
            self.params.setdefault('scale', 1)
            self.params.setdefault('delta', 0)
        elif method == 'laplacian':
            self.params.setdefault('ksize', 3)
            self.params.setdefault('scale', 1)
            self.params.setdefault('delta', 0)
        elif method == 'prewitt':
            self.params.setdefault('ksize', 3)
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges in an image
        
        Args:
            image: Input image (grayscale or RGB)
            
        Returns:
            Edge map as binary image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Normalize to 0-255 range
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply edge detection
        if self.method == 'canny':
            edges = cv2.Canny(gray, 
                           self.params['threshold1'], 
                           self.params['threshold2'],
                           apertureSize=self.params['aperture_size'])
        
        elif self.method == 'sobel':
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.params['ksize'],
                              scale=self.params['scale'], delta=self.params['delta'])
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.params['ksize'],
                              scale=self.params['scale'], delta=self.params['delta'])
            edges = np.sqrt(sobel_x**2 + sobel_y**2)
            edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        elif self.method == 'laplacian':
            laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=self.params['ksize'],
                                   scale=self.params['scale'], delta=self.params['delta'])
            edges = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        elif self.method == 'prewitt':
            kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
            kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
            
            prewitt_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
            prewitt_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
            edges = np.sqrt(prewitt_x**2 + prewitt_y**2)
            edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        else:
            raise ValueError(f"Unknown edge detection method: {self.method}")
        
        return edges

class RadioMapSeerEdgeDataset:
    """Dataset class for RadioMapSeer edge detection"""
    
    def __init__(self, data_root: str, image_size: Tuple[int, int] = (256, 256),
                 edge_method: str = 'canny', **kwargs):
        """
        Initialize dataset
        
        Args:
            data_root: Path to RadioMapSeer dataset
            image_size: Target image size
            edge_method: Edge detection method
            **kwargs: Edge detection parameters
        """
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.edge_detector = EdgeDetector(edge_method, **kwargs)
        
        # Dataset paths
        self.dpm_path = self.data_root / 'gain' / 'DPM'
        self.png_path = self.data_root / 'png'
        self.csv_path = self.data_root / 'dataset.csv'
        
        # Check if dataset exists
        if not self.dpm_path.exists():
            raise FileNotFoundError(f"DPM dataset not found at {self.dpm_path}")
        
        # Get list of DPM images
        self.dpm_files = sorted([f for f in self.dpm_path.glob('*.png')])
        if not self.dpm_files:
            raise FileNotFoundError(f"No DPM images found in {self.dpm_path}")
        
        print(f"Found {len(self.dpm_files)} DPM images")
    
    def create_edge_dataset(self, output_dir: str, split_ratio: float = 0.8) -> None:
        """
        Create edge detection dataset from DPM images
        
        Args:
            output_dir: Output directory for edge dataset
            split_ratio: Train/validation split ratio
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        for subset in ['train', 'val']:
            for data_type in ['image', 'edge']:
                (output_path / data_type / 'raw' / subset).mkdir(parents=True, exist_ok=True)
        
        # Split dataset
        split_idx = int(len(self.dpm_files) * split_ratio)
        train_files = self.dpm_files[:split_idx]
        val_files = self.dpm_files[split_idx:]
        
        print(f"Creating edge dataset...")
        print(f"Train: {len(train_files)} images")
        print(f"Val: {len(val_files)} images")
        
        # Process training images
        for i, dpm_file in enumerate(tqdm(train_files, desc="Processing training images")):
            self._process_image_pair(dpm_file, output_path / 'image' / 'raw' / 'train',
                                   output_path / 'edge' / 'raw' / 'train', f'train_{i}')
        
        # Process validation images
        for i, dpm_file in enumerate(tqdm(val_files, desc="Processing validation images")):
            self._process_image_pair(dpm_file, output_path / 'image' / 'raw' / 'val',
                                   output_path / 'edge' / 'raw' / 'val', f'val_{i}')
        
        print(f"Edge dataset created at {output_path}")
    
    def _process_image_pair(self, dpm_file: Path, image_output_dir: Path, 
                          edge_output_dir: Path, base_name: str) -> None:
        """Process a single DPM image to create image-edge pair"""
        try:
            # Load DPM image
            dpm_image = np.array(Image.open(dpm_file))
            
            # Generate edge map
            edge_map = self.edge_detector.detect_edges(dpm_image)
            
            # Resize images to target size
            dpm_resized = cv2.resize(dpm_image, self.image_size)
            edge_resized = cv2.resize(edge_map, self.image_size)
            
            # Save images
            image_path = image_output_dir / f'{base_name}.png'
            edge_path = edge_output_dir / f'{base_name}.png'
            
            Image.fromarray(dpm_resized).save(image_path)
            Image.fromarray(edge_resized).save(edge_path)
            
        except Exception as e:
            print(f"Error processing {dpm_file}: {e}")
    
    def get_sample_pair(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a sample image-edge pair
        
        Args:
            index: Index of the sample
            
        Returns:
            Tuple of (image, edge_map)
        """
        if index >= len(self.dpm_files):
            raise IndexError("Index out of range")
        
        dpm_file = self.dpm_files[index]
        dpm_image = np.array(Image.open(dpm_file))
        edge_map = self.edge_detector.detect_edges(dpm_image)
        
        return dpm_image, edge_map

class SyntheticEdgeGenerator:
    """Generate synthetic edge maps from radio map characteristics"""
    
    def __init__(self, image_size: Tuple[int, int] = (256, 256)):
        self.image_size = image_size
    
    def generate_synthetic_edges(self, dpm_image: np.ndarray, 
                                edge_type: str = 'gradient') -> np.ndarray:
        """
        Generate synthetic edge maps from DPM images
        
        Args:
            dpm_image: Input DPM image
            edge_type: Type of synthetic edges ('gradient', 'contour', 'ridge')
            
        Returns:
            Synthetic edge map
        """
        if len(dpm_image.shape) == 3:
            gray = cv2.cvtColor(dpm_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = dpm_image.copy()
        
        # Normalize to 0-255 range
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        if edge_type == 'gradient':
            # Gradient-based edges
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            edges = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        elif edge_type == 'contour':
            # Contour-based edges
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            edges = np.zeros_like(gray)
            cv2.drawContours(edges, contours, -1, 255, 1)
        
        elif edge_type == 'ridge':
            # Ridge detection using morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            edges = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        else:
            raise ValueError(f"Unknown synthetic edge type: {edge_type}")
        
        return edges

def main():
    """Main function for edge detection dataset generation"""
    parser = argparse.ArgumentParser(description='Generate edge detection dataset from RadioMapSeer')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to RadioMapSeer dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for edge dataset')
    parser.add_argument('--method', type=str, default='canny',
                       choices=['canny', 'sobel', 'laplacian', 'prewitt'],
                       help='Edge detection method')
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256],
                       help='Target image size (width height)')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                       help='Train/validation split ratio')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic edge generation')
    parser.add_argument('--synthetic_type', type=str, default='gradient',
                       choices=['gradient', 'contour', 'ridge'],
                       help='Synthetic edge generation type')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data_root):
        print(f"Error: Dataset directory {args.data_root} does not exist")
        return 1
    
    if args.split_ratio <= 0 or args.split_ratio >= 1:
        print("Error: split_ratio must be between 0 and 1")
        return 1
    
    try:
        # Create dataset
        if args.synthetic:
            print(f"Creating synthetic edge dataset using {args.synthetic_type} method...")
            # For synthetic edges, we'll use the regular edge detector with custom parameters
            dataset = RadioMapSeerEdgeDataset(
                args.data_root, 
                tuple(args.image_size),
                args.method
            )
            # Override edge detector with synthetic generator
            dataset.edge_detector = SyntheticEdgeGenerator(tuple(args.image_size))
            dataset.edge_detector.detect_edges = lambda img: dataset.edge_detector.generate_synthetic_edges(img, args.synthetic_type)
        else:
            print(f"Creating edge dataset using {args.method} method...")
            dataset = RadioMapSeerEdgeDataset(
                args.data_root, 
                tuple(args.image_size),
                args.method
            )
        
        # Generate edge dataset
        dataset.create_edge_dataset(args.output_dir, args.split_ratio)
        
        print("\nDataset generation completed successfully!")
        print(f"Edge dataset saved to: {args.output_dir}")
        print(f"Method: {args.method}")
        print(f"Image size: {args.image_size}")
        print(f"Train/Val split: {args.split_ratio:.1f}/{1-args.split_ratio:.1f}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())