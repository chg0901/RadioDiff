#!/usr/bin/env python3
"""
RadioMapSeer Edge Detection Dataset Generator (Modified Version)

This script generates edge detection datasets from RadioMapSeer while maintaining
the original dataset structure without any train/validation splitting.

Features:
1. Process DPM images to generate edge maps using Canny edge detection
2. Maintain original RadioMapSeer folder structure exactly
3. No dataset splitting - preserves all files in original organization
4. Direct structure preservation from source to destination
5. Configurable image size and edge detection parameters

Usage:
    python radiomapseer_edge_detection_m.py \
        --data_root /home/cine/Documents/dataset/RadioMapSeer \
        --output_dir ./radiomapseer_edge_dataset \
        --method canny \
        --image_size 256 256

Expected Output Structure:
    output_dir/
    ├── image/           # RGB images (copied from original PNG files)
    │   ├── subset1/
    │   │   ├── img1.jpg
    │   │   └── img2.png
    │   └── subset2/
    │       ├── img3.jpg
    │       └── img4.png
    └── edge/           # Edge labels (generated from DPM)
        ├── subset1/
        │   ├── img1.jpg
        │   └── img2.png
        └── subset2/
            ├── img3.jpg
            └── img4.png
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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        
        logger.info(f"Initialized {method} edge detector with parameters: {self.params}")
    
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
    """Dataset class for RadioMapSeer edge detection with exact structure preservation"""
    
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
        
        # Get list of DPM images recursively to preserve original structure
        self.dpm_files = sorted([f for f in self.dpm_path.rglob('*.png')])
        
        if not self.dpm_files:
            raise FileNotFoundError(f"No DPM images found in {self.dpm_path}")
        
        logger.info(f"Found {len(self.dpm_files)} DPM images")
        logger.info(f"DPM path: {self.dpm_path}")
        logger.info(f"PNG path: {self.png_path}")
        logger.info(f"CSV path: {self.csv_path}")
    
    def create_edge_dataset(self, output_dir: str) -> None:
        """
        Create edge detection dataset from DPM images preserving exact structure
        
        Args:
            output_dir: Output directory for edge dataset
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure matching original RadioMapSeer
        # output_dir/
        # ├── image/           # RGB images (copied from original PNG files)
        # │   ├── subset1/
        # │   │   ├── img1.jpg
        # │   │   └── img2.png
        # │   └── subset2/
        # │       ├── img3.jpg
        # │       └── img4.png
        # └── edge/           # Edge labels (generated from DPM)
        #     ├── subset1/
        #     │   ├── img1.jpg
        #     │   └── img2.png
        #     └── subset2/
        #         ├── img3.jpg
        #         └── img4.png
        
        # Create image and edge directories
        image_dir = output_path / 'image'
        edge_dir = output_path / 'edge'
        
        logger.info(f"Creating edge dataset with exact structure preservation...")
        logger.info(f"Output directory: {output_path}")
        
        # Process all DPM files while preserving structure
        processed_count = 0
        for dpm_file in tqdm(self.dpm_files, desc="Processing DPM images"):
            # Get relative path from DPM directory
            rel_path = dpm_file.relative_to(self.dpm_path)
            
            # Create corresponding output paths
            edge_output_path = edge_dir / rel_path
            
            # Create parent directories
            edge_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Process the DPM file to generate edge map
            try:
                self._process_dpm_to_edge(dpm_file, edge_output_path)
                processed_count += 1
            except Exception as e:
                logger.error(f"Error processing {dpm_file}: {e}")
                continue
        
        # Copy PNG images to image directory preserving structure
        if self.png_path.exists():
            logger.info("Copying PNG images to image directory...")
            png_files = sorted([f for f in self.png_path.rglob('*.png')])
            for png_file in tqdm(png_files, desc="Copying PNG images"):
                rel_path = png_file.relative_to(self.png_path)
                image_output_path = image_dir / rel_path
                image_output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Load, resize, and save PNG image
                try:
                    png_image = np.array(Image.open(png_file))
                    png_resized = cv2.resize(png_image, self.image_size)
                    Image.fromarray(png_resized).save(image_output_path)
                except Exception as e:
                    logger.error(f"Error processing PNG {png_file}: {e}")
                    continue
        
        # Copy dataset.csv if it exists
        if self.csv_path.exists():
            shutil.copy2(self.csv_path, output_path / 'dataset.csv')
            logger.info("Copied dataset.csv")
        
        logger.info(f"Edge dataset created at {output_path}")
        logger.info(f"Successfully processed {processed_count}/{len(self.dpm_files)} DPM files")
        logger.info("Original structure preserved exactly!")
    
    def _process_dpm_to_edge(self, dpm_file: Path, edge_output_path: Path) -> None:
        """Process a single DPM image to create edge map"""
        # Load DPM image
        dpm_image = np.array(Image.open(dpm_file))
        
        # Generate edge map
        edge_map = self.edge_detector.detect_edges(dpm_image)
        
        # Resize edge map to target size
        edge_resized = cv2.resize(edge_map, self.image_size)
        
        # Save edge map
        Image.fromarray(edge_resized).save(edge_output_path)
    
    def analyze_structure(self) -> Dict[str, any]:
        """
        Analyze the dataset structure
        
        Returns:
            Dictionary with structure analysis
        """
        analysis = {
            'total_dpm_files': len(self.dpm_files),
            'dpm_path': str(self.dpm_path),
            'png_path': str(self.png_path),
            'csv_path': str(self.csv_path),
            'has_png_images': self.png_path.exists(),
            'has_csv_file': self.csv_path.exists()
        }
        
        if self.png_path.exists():
            png_files = sorted([f for f in self.png_path.rglob('*.png')])
            analysis['png_files_count'] = len(png_files)
        
        # Analyze directory structure
        dpm_structure = {}
        for dpm_file in self.dpm_files:
            rel_path = dpm_file.relative_to(self.dpm_path)
            parent_dir = str(rel_path.parent)
            if parent_dir not in dpm_structure:
                dpm_structure[parent_dir] = []
            dpm_structure[parent_dir].append(str(rel_path))
        
        analysis['dpm_structure'] = dpm_structure
        
        if self.png_path.exists():
            png_structure = {}
            png_files = sorted([f for f in self.png_path.rglob('*.png')])
            for png_file in png_files:
                rel_path = png_file.relative_to(self.png_path)
                parent_dir = str(rel_path.parent)
                if parent_dir not in png_structure:
                    png_structure[parent_dir] = []
                png_structure[parent_dir].append(str(rel_path))
            analysis['png_structure'] = png_structure
        
        return analysis

def main():
    """Main function for edge detection dataset generation"""
    parser = argparse.ArgumentParser(
        description='Generate edge detection dataset from RadioMapSeer preserving exact structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with Canny edge detection
    python radiomapseer_edge_detection_m.py \\
        --data_root /home/cine/Documents/dataset/RadioMapSeer \\
        --output_dir ./radiomapseer_edge_dataset \\
        --method canny \\
        --image_size 256 256

    # Analyze dataset structure only
    python radiomapseer_edge_detection_m.py \\
        --data_root /home/cine/Documents/dataset/RadioMapSeer \\
        --analyze_only

    # Use Sobel edge detection with custom parameters
    python radiomapseer_edge_detection_m.py \\
        --data_root /home/cine/Documents/dataset/RadioMapSeer \\
        --output_dir ./radiomapseer_edge_dataset \\
        --method sobel \\
        --image_size 512 512
        """
    )
    
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to RadioMapSeer dataset')
    parser.add_argument('--output_dir', type=str, required=False,
                       help='Output directory for edge dataset (required unless --analyze_only)')
    parser.add_argument('--method', type=str, default='canny',
                       choices=['canny', 'sobel', 'laplacian', 'prewitt'],
                       help='Edge detection method (default: canny)')
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256],
                       help='Target image size (width height) (default: 256 256)')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze dataset structure without processing')
    parser.add_argument('--canny_threshold1', type=int, default=50,
                       help='Canny edge detection threshold1 (default: 50)')
    parser.add_argument('--canny_threshold2', type=int, default=150,
                       help='Canny edge detection threshold2 (default: 150)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate inputs
    if not os.path.exists(args.data_root):
        logger.error(f"Dataset directory {args.data_root} does not exist")
        return 1
    
    if not args.analyze_only and not args.output_dir:
        logger.error("Output directory is required unless --analyze_only is specified")
        return 1
    
    try:
        # Create dataset
        logger.info(f"Creating edge dataset using {args.method} method...")
        
        # Update edge detection parameters if specified
        edge_params = {}
        if args.method == 'canny':
            edge_params = {
                'threshold1': args.canny_threshold1,
                'threshold2': args.canny_threshold2
            }
        
        dataset = RadioMapSeerEdgeDataset(
            args.data_root, 
            tuple(args.image_size),
            args.method,
            **edge_params
        )
        
        # Analyze structure
        if args.analyze_only:
            logger.info("Analyzing dataset structure...")
            analysis = dataset.analyze_structure()
            
            print("\n" + "="*60)
            print("RADIOMAPSEER DATASET STRUCTURE ANALYSIS")
            print("="*60)
            print(f"Total DPM files: {analysis['total_dpm_files']}")
            print(f"DPM path: {analysis['dpm_path']}")
            print(f"Has PNG images: {analysis['has_png_images']}")
            if analysis['has_png_images']:
                print(f"PNG files count: {analysis['png_files_count']}")
                print(f"PNG path: {analysis['png_path']}")
            print(f"Has CSV file: {analysis['has_csv_file']}")
            if analysis['has_csv_file']:
                print(f"CSV path: {analysis['csv_path']}")
            
            print("\nDPM directory structure:")
            for subset_name, subset_files in analysis['dpm_structure'].items():
                print(f"  - {subset_name}: {len(subset_files)} files")
            
            if analysis['has_png_images']:
                print("\nPNG directory structure:")
                for subset_name, subset_files in analysis['png_structure'].items():
                    print(f"  - {subset_name}: {len(subset_files)} files")
            
            print("="*60)
            return 0
        
        # Generate edge dataset
        dataset.create_edge_dataset(args.output_dir)
        
        print("\n" + "="*60)
        print("EDGE DATASET GENERATION COMPLETED")
        print("="*60)
        print(f"Edge dataset saved to: {args.output_dir}")
        print(f"Method: {args.method}")
        print(f"Image size: {args.image_size}")
        print("Structure: Exact preservation from original dataset")
        print("No train/validation splitting performed")
        print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())