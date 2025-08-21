#!/usr/bin/env python3
"""
Script to compare IRT4 ground truth and generated images.
Creates three-column visualizations showing GT, generated, and difference map.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from pathlib import Path
import random


def load_image(image_path):
    """Load image as numpy array."""
    return np.array(Image.open(image_path))


def compute_difference_map(img1, img2):
    """Compute absolute difference between two images."""
    # Ensure images are the same size
    if img1.shape[:2] != img2.shape[:2]:
        img2 = np.array(Image.fromarray(img2).resize((img1.shape[1], img1.shape[0])))
    
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1 = np.mean(img1, axis=2)
    if len(img2.shape) == 3:
        img2 = np.mean(img2, axis=2)
    
    # Compute absolute difference
    diff = np.abs(img1.astype(float) - img2.astype(float))
    
    # Normalize to 0-255 range
    if diff.max() > 0:
        diff = (diff / diff.max()) * 255
    
    return diff.astype(np.uint8)


def create_comparison_figure(gt_images, gen_images, output_path, samples_per_figure=5):
    """Create a comparison figure with GT, generated, and difference map."""
    n_samples = min(len(gt_images), len(gen_images), samples_per_figure)
    
    if n_samples == 0:
        print("No matching images found!")
        return
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        gt_img = load_image(gt_images[i])
        gen_img = load_image(gen_images[i])
        diff_img = compute_difference_map(gt_img, gen_img)
        
        # Ground truth
        if len(gt_img.shape) == 3:
            axes[i, 0].imshow(gt_img)
        else:
            axes[i, 0].imshow(gt_img, cmap='gray')
        axes[i, 0].set_title(f'Ground Truth - {os.path.basename(gt_images[i])}')
        axes[i, 0].axis('off')
        
        # Generated
        if len(gen_img.shape) == 3:
            axes[i, 1].imshow(gen_img)
        else:
            axes[i, 1].imshow(gen_img, cmap='gray')
        axes[i, 1].set_title(f'Generated - {os.path.basename(gen_images[i])}')
        axes[i, 1].axis('off')
        
        # Difference map
        axes[i, 2].imshow(diff_img, cmap='hot')
        axes[i, 2].set_title('Difference Map')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison figure saved to: {output_path}")


def find_matching_images(gt_dir, gen_dir):
    """Find matching images between ground truth and generated directories."""
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.png')]
    gen_files = [f for f in os.listdir(gen_dir) if f.endswith('.png')]
    
    gt_files_set = set(gt_files)
    gen_files_set = set(gen_files)
    
    matching_files = gt_files_set.intersection(gen_files_set)
    
    gt_paths = [os.path.join(gt_dir, f) for f in matching_files]
    gen_paths = [os.path.join(gen_dir, f) for f in matching_files]
    
    return gt_paths, gen_paths, matching_files


def main():
    parser = argparse.ArgumentParser(description='Compare IRT4 ground truth and generated images')
    parser.add_argument('--gt_dir', type=str, required=True, 
                       help='Path to ground truth images directory')
    parser.add_argument('--gen_dir', type=str, required=True,
                       help='Path to generated images directory')
    parser.add_argument('--output_dir', type=str, default='./irt4_comparison_results',
                       help='Output directory for comparison figures')
    parser.add_argument('--samples_per_figure', type=int, default=5,
                       help='Number of samples per figure')
    parser.add_argument('--num_figures', type=int, default=1,
                       help='Number of figures to create')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for sampling')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find matching images
    gt_paths, gen_paths, matching_files = find_matching_images(args.gt_dir, args.gen_dir)
    
    print(f"Found {len(matching_files)} matching images")
    
    if len(matching_files) == 0:
        print("No matching images found between directories!")
        return
    
    # Set random seed for reproducibility
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Create multiple figures if needed
    total_samples = min(len(matching_files), args.samples_per_figure * args.num_figures)
    
    for fig_idx in range(args.num_figures):
        start_idx = fig_idx * args.samples_per_figure
        end_idx = min(start_idx + args.samples_per_figure, total_samples)
        
        if start_idx >= total_samples:
            break
        
        # Sample images
        sampled_indices = list(range(start_idx, end_idx))
        
        gt_sampled = [gt_paths[i] for i in sampled_indices]
        gen_sampled = [gen_paths[i] for i in sampled_indices]
        
        # Create comparison figure
        output_path = os.path.join(args.output_dir, f'irt4_comparison_figure_{fig_idx + 1}.png')
        create_comparison_figure(gt_sampled, gen_sampled, output_path, args.samples_per_figure)
    
    print(f"Created {min(args.num_figures, (total_samples + args.samples_per_figure - 1) // args.samples_per_figure)} comparison figures")


if __name__ == "__main__":
    main()