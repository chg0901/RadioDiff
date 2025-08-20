#!/usr/bin/env python3
"""
RadioDiff Image Comparison Script

This script compares generated images with ground truth images using various metrics
and statistical analysis. It supports different RadioDiff models and configurations.

Usage:
    python compare_images.py --config configs/radio_sample_m.yaml --gt_dir /path/to/gt --gen_dir /path/to/generated
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import pandas as pd
from skimage import metrics
from torchvision import transforms
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ImageComparator:
    """Comprehensive image comparison class for RadioDiff"""
    
    def __init__(self, config_path: str, gt_dir: str, gen_dir: str, output_dir: str = "./enhanced_suite/archive/comparison_results"):
        """
        Initialize the image comparator
        
        Args:
            config_path: Path to configuration YAML file
            gt_dir: Directory containing ground truth images
            gen_dir: Directory containing generated images
            output_dir: Directory to save comparison results
        """
        self.config = self._load_config(config_path)
        self.gt_dir = Path(gt_dir)
        self.gen_dir = Path(gen_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Image size from config
        self.image_size = self.config.get('model', {}).get('image_size', [256, 256])
        
        # Initialize metrics storage
        self.metrics = {
            'nmse': [],
            'rmse': [],
            'ssim': [],
            'psnr': [],
            'mae': [],
            'relative_error': [],
            'brightest_point_distance': [],
            'sharpness_ratio': []
        }
        
        # Load images
        self.gt_images, self.gen_images, self.image_pairs = self._load_image_pairs()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _load_image_pairs(self) -> Tuple[Dict, Dict, List]:
        """Load and pair ground truth and generated images"""
        gt_images = {}
        gen_images = {}
        image_pairs = []
        
        # Load ground truth images
        for img_path in self.gt_dir.glob("*.png"):
            if img_path.stem.startswith('gt-sample-'):
                img_num = img_path.stem.split('-')[-1]
                gt_images[img_num] = self._load_image(img_path)
        
        # Load generated images
        for img_path in self.gen_dir.glob("*.png"):
            if img_path.stem.startswith('sample-') and not img_path.stem.startswith('sample-0_'):
                img_num = img_path.stem.split('-')[-1]
                gen_images[img_num] = self._load_image(img_path)
        
        # Create pairs
        for img_num in gt_images:
            if img_num in gen_images:
                image_pairs.append((img_num, gt_images[img_num], gen_images[img_num]))
        
        print(f"Found {len(image_pairs)} matching image pairs")
        return gt_images, gen_images, image_pairs
    
    def _load_image(self, img_path: Path) -> torch.Tensor:
        """Load and preprocess image"""
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Resize if needed
        if img.size != tuple(self.image_size):
            img = img.resize(tuple(self.image_size), Image.BILINEAR)
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        return transform(img).to(device)
    
    def variance_of_laplacian(self, img: torch.Tensor) -> torch.Tensor:
        """Calculate variance of Laplacian for sharpness assessment"""
        laplacian_kernel = torch.tensor([[0, 1, 0],
                                       [1, -4, 1],
                                       [0, 1, 0]], dtype=img.dtype, device=img.device)
        laplacian_kernel = laplacian_kernel.unsqueeze(0).unsqueeze(0)
        
        lap = F.conv2d(img, laplacian_kernel, padding=1)
        return torch.var(lap)
    
    def brightest_point_distance(self, gt: torch.Tensor, pred: torch.Tensor) -> float:
        """Calculate Euclidean distance between brightest points"""
        gt_2d = gt.squeeze()
        pred_2d = pred.squeeze()
        
        gt_idx = torch.argmax(gt_2d)
        pred_idx = torch.argmax(pred_2d)
        
        gt_y, gt_x = divmod(gt_idx.item(), gt_2d.shape[1])
        pred_y, pred_x = divmod(pred_idx.item(), pred_2d.shape[1])
        
        distance = torch.sqrt(torch.tensor((gt_x - pred_x) ** 2 + (gt_y - pred_y) ** 2))
        return distance.item()
    
    def calculate_relative_error(self, pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
        """Calculate relative error with sqrt transformation"""
        relative_error = torch.abs(pred - target) / (torch.abs(target) + 1e-8)
        relative_error = torch.sqrt(relative_error)
        clipped_error = torch.clamp(relative_error, min=0.0, max=1)
        return clipped_error.detach().cpu().numpy().flatten()
    
    def calculate_nmse(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Normalized Mean Squared Error"""
        mse = F.mse_loss(pred, target)
        target_variance = F.mse_loss(target, torch.zeros_like(target))
        return (mse / target_variance).item()
    
    def calculate_all_metrics(self, gt: torch.Tensor, gen: torch.Tensor) -> Dict[str, float]:
        """Calculate all comparison metrics for a single image pair"""
        metrics_dict = {}
        
        # Ensure images are in the right format
        if gt.dim() == 3:
            gt = gt.unsqueeze(0)
        if gen.dim() == 3:
            gen = gen.unsqueeze(0)
        
        # Normalize to [0, 1] range for some metrics
        gt_norm = (gt + 1) / 2
        gen_norm = (gen + 1) / 2
        
        # Basic metrics
        metrics_dict['nmse'] = self.calculate_nmse(gen, gt)
        metrics_dict['rmse'] = torch.sqrt(F.mse_loss(gen, gt)).item()
        metrics_dict['mae'] = F.l1_loss(gen, gt).item()
        
        # SSIM and PSNR (require [0, 1] range)
        metrics_dict['ssim'] = ssim(gen_norm, gt_norm).item()
        metrics_dict['psnr'] = psnr(gen_norm, gt_norm).item()
        
        # Relative error
        rel_error = self.calculate_relative_error(gen, gt)
        metrics_dict['relative_error'] = np.mean(rel_error)
        
        # Brightest point distance
        metrics_dict['brightest_point_distance'] = self.brightest_point_distance(gt_norm, gen_norm)
        
        # Sharpness comparison
        gt_sharpness = self.variance_of_laplacian(gt_norm).item()
        gen_sharpness = self.variance_of_laplacian(gen_norm).item()
        metrics_dict['sharpness_ratio'] = gen_sharpness / (gt_sharpness + 1e-8)
        
        return metrics_dict
    
    def compare_all_images(self):
        """Compare all image pairs and calculate metrics"""
        print("Comparing images...")
        
        for img_num, gt_img, gen_img in tqdm(self.image_pairs):
            try:
                metrics_dict = self.calculate_all_metrics(gt_img, gen_img)
                
                # Store metrics
                for key, value in metrics_dict.items():
                    self.metrics[key].append(value)
            except Exception as e:
                print(f"Error processing image {img_num}: {e}")
                continue
        
        print("Comparison complete!")
        
        # Debug: Print metric lengths
        for key, values in self.metrics.items():
            print(f"{key}: {len(values)} values")
    
    def generate_statistics(self) -> Dict[str, Dict]:
        """Generate statistical summary of all metrics"""
        stats = {}
        for metric_name, values in self.metrics.items():
            if values:
                values_array = np.array(values)
                stats[metric_name] = {
                    'mean': np.mean(values_array),
                    'std': np.std(values_array),
                    'min': np.min(values_array),
                    'max': np.max(values_array),
                    'median': np.median(values_array)
                }
        
        return stats
    
    def save_results(self):
        """Save comparison results and visualizations"""
        stats = self.generate_statistics()
        
        # Save statistics to CSV
        df_stats = pd.DataFrame(stats).T
        df_stats.to_csv(self.output_dir / "comparison_statistics.csv")
        print(f"Statistics saved to {self.output_dir / 'comparison_statistics.csv'}")
        
        # Save detailed metrics
        try:
            df_detailed = pd.DataFrame(self.metrics)
            df_detailed.to_csv(self.output_dir / "detailed_metrics.csv", index=False)
            print(f"Detailed metrics saved to {self.output_dir / 'detailed_metrics.csv'}")
        except ValueError as e:
            print(f"Warning: Could not save detailed metrics due to inconsistent array lengths: {e}")
            # Save individual metrics as separate files
            for metric_name, values in self.metrics.items():
                if values:
                    pd.Series(values).to_csv(self.output_dir / f"{metric_name}_values.csv", index=False)
            print("Individual metric values saved as separate CSV files")
        
        # Generate visualizations
        self._create_visualizations(stats)
        
        # Print summary
        self._print_summary(stats)
    
    def _create_visualizations(self, stats: Dict):
        """Create comparison visualizations"""
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        
        # 1. Metrics bar plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics_to_plot = ['nmse', 'rmse', 'ssim', 'psnr', 'mae', 'brightest_point_distance']
        for i, metric in enumerate(metrics_to_plot):
            if metric in self.metrics and self.metrics[metric]:
                axes[i].hist(self.metrics[metric], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].axvline(stats[metric]['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats[metric]["mean"]:.4f}')
                axes[i].set_title(f'{metric.upper()} Distribution')
                axes[i].set_xlabel(metric)
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "metrics_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Relative error distribution
        if self.metrics['relative_error']:
            plt.figure(figsize=(10, 6))
            all_errors = np.concatenate([np.array(self.metrics['relative_error']).flatten()])
            filtered_errors = all_errors[all_errors > 1e-3]
            
            plt.hist(filtered_errors, bins=100, color='blue', alpha=0.4, density=True, label='Histogram')
            sns.kdeplot(filtered_errors, color='red', linewidth=2, label='KDE')
            
            plt.xlabel("Relative Error")
            plt.ylabel("Density")
            plt.title("Relative Error Distribution")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(self.output_dir / "relative_error_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Box plot comparison
        if self.metrics:
            plt.figure(figsize=(12, 8))
            metrics_for_boxplot = ['nmse', 'rmse', 'ssim', 'psnr', 'mae']
            data_for_boxplot = [self.metrics[m] for m in metrics_for_boxplot if m in self.metrics]
            labels_for_boxplot = [m.upper() for m in metrics_for_boxplot if m in self.metrics]
            
            if data_for_boxplot:
                plt.boxplot(data_for_boxplot, tick_labels=labels_for_boxplot)
                plt.title("Metrics Distribution Comparison")
                plt.ylabel("Value")
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.output_dir / "metrics_boxplot.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        # 4. Sample comparison images
        self._create_sample_comparison()
        
        print(f"Visualizations saved to {self.output_dir}")
    
    def _create_sample_comparison(self):
        """Create side-by-side comparison of sample images"""
        # Select a few sample images for visual comparison
        sample_indices = [0, len(self.image_pairs)//4, len(self.image_pairs)//2, 
                         3*len(self.image_pairs)//4, -1]
        
        fig, axes = plt.subplots(len(sample_indices), 3, figsize=(15, 5*len(sample_indices)))
        
        for i, idx in enumerate(sample_indices):
            if idx >= len(self.image_pairs):
                continue
                
            img_num, gt_img, gen_img = self.image_pairs[idx]
            
            # Convert to numpy for plotting
            gt_np = gt_img.squeeze().cpu().numpy()
            gen_np = gen_img.squeeze().cpu().numpy()
            diff_np = np.abs(gt_np - gen_np)
            
            # Plot ground truth
            axes[i, 0].imshow(gt_np, cmap='viridis')
            axes[i, 0].set_title(f'Ground Truth {img_num}')
            axes[i, 0].axis('off')
            
            # Plot generated
            axes[i, 1].imshow(gen_np, cmap='viridis')
            axes[i, 1].set_title(f'Generated {img_num}')
            axes[i, 1].axis('off')
            
            # Plot difference
            axes[i, 2].imshow(diff_np, cmap='hot')
            axes[i, 2].set_title(f'Difference {img_num}')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "sample_comparisons.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _print_summary(self, stats: Dict):
        """Print a summary of comparison results"""
        print("\n" + "="*60)
        print("RADIO DIFF IMAGE COMPARISON SUMMARY")
        print("="*60)
        print(f"Total images compared: {len(self.image_pairs)}")
        print(f"Image size: {self.image_size}")
        print("-"*60)
        
        for metric_name, stat_dict in stats.items():
            print(f"{metric_name.upper():<25}: "
                  f"Mean={stat_dict['mean']:.6f}, "
                  f"Std={stat_dict['std']:.6f}, "
                  f"Min={stat_dict['min']:.6f}, "
                  f"Max={stat_dict['max']:.6f}")
        
        print("="*60)
        print(f"Results saved to: {self.output_dir}")
        print("="*60)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="RadioDiff Image Comparison Tool")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to configuration YAML file")
    parser.add_argument("--gt_dir", type=str, required=True,
                       help="Directory containing ground truth images")
    parser.add_argument("--gen_dir", type=str, required=True,
                       help="Directory containing generated images")
    parser.add_argument("--output_dir", type=str, default="./enhanced_suite/archive/comparison_results",
                       help="Directory to save comparison results")
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Initialize comparator
    comparator = ImageComparator(
        config_path=args.config,
        gt_dir=args.gt_dir,
        gen_dir=args.gen_dir,
        output_dir=args.output_dir
    )
    
    # Run comparison
    comparator.compare_all_images()
    
    # Save results
    comparator.save_results()


if __name__ == "__main__":
    main()