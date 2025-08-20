#!/usr/bin/env python3
"""
RadioDiff VAE Sample Visualization Script

This script creates a comprehensive visualization of RadioDiff VAE sample images
arranged in a 3x10 grid layout for easy analysis and comparison.

Usage:
    python radiodiff_vae_sample_visualization.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from pathlib import Path

def load_sample_images(image_dir, pattern="sample-*.png"):
    """
    Load sample images from the specified directory.
    
    Args:
        image_dir (str): Directory containing sample images
        pattern (str): File pattern to match (default: "sample-*.png")
    
    Returns:
        list: List of loaded PIL Images
        list: List of corresponding file paths
    """
    image_paths = sorted(glob.glob(os.path.join(image_dir, pattern)))
    
    if not image_paths:
        raise FileNotFoundError(f"No images found matching pattern '{pattern}' in {image_dir}")
    
    images = []
    valid_paths = []
    
    for path in image_paths:
        try:
            img = Image.open(path)
            images.append(img)
            valid_paths.append(path)
            print(f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
    
    print(f"Successfully loaded {len(images)} images")
    return images, valid_paths

def create_visualization_grid(images, output_path, rows=3, cols=10, figsize=(20, 6)):
    """
    Create a grid visualization of the sample images.
    
    Args:
        images (list): List of PIL Images to display
        output_path (str): Path to save the visualization
        rows (int): Number of rows in the grid (default: 3)
        cols (int): Number of columns in the grid (default: 10)
        figsize (tuple): Figure size (default: (20, 6))
    """
    if len(images) == 0:
        raise ValueError("No images to visualize")
    
    # Create figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle('RadioDiff VAE Sample Images Visualization\n(3×10 Grid Layout)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Flatten axes for easy iteration
    axes_flat = axes.flatten()
    
    # Display images
    for i, (img, ax) in enumerate(zip(images, axes_flat)):
        if i < len(images):
            # Convert PIL Image to numpy array for matplotlib
            img_array = np.array(img)
            
            # Handle different image modes
            if len(img_array.shape) == 3:  # RGB image
                ax.imshow(img_array)
            elif len(img_array.shape) == 2:  # Grayscale image
                ax.imshow(img_array, cmap='gray')
            else:
                ax.imshow(img_array)
            
            # Set title with sample number
            sample_num = i + 1
            ax.set_title(f'Sample {sample_num}', fontsize=8, pad=2)
            
            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Hide empty subplots
            ax.set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.1)
    
    # Save the visualization
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Visualization saved to: {output_path}")
    
    # Also show the plot
    plt.show()
    
    return fig

def create_detailed_analysis(images, output_dir):
    """
    Create detailed analysis visualizations of the sample images.
    
    Args:
        images (list): List of PIL Images
        output_dir (str): Directory to save analysis plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Analysis 1: Image statistics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('RadioDiff VAE Sample Images - Statistical Analysis', fontsize=14, fontweight='bold')
    
    # Collect statistics
    shapes = []
    modes = []
    min_values = []
    max_values = []
    mean_values = []
    
    for img in images:
        img_array = np.array(img)
        shapes.append(img_array.shape)
        modes.append(img.mode)
        min_values.append(img_array.min())
        max_values.append(img_array.max())
        mean_values.append(img_array.mean())
    
    # Plot 1: Image dimensions distribution
    unique_shapes = list(set(shapes))
    shape_counts = [shapes.count(shape) for shape in unique_shapes]
    ax1.bar(range(len(unique_shapes)), shape_counts)
    ax1.set_title('Image Shape Distribution')
    ax1.set_xlabel('Shape Index')
    ax1.set_ylabel('Count')
    
    # Plot 2: Value range distribution
    ax2.scatter(range(len(images)), min_values, label='Min', alpha=0.7, s=20)
    ax2.scatter(range(len(images)), max_values, label='Max', alpha=0.7, s=20)
    ax2.scatter(range(len(images)), mean_values, label='Mean', alpha=0.7, s=20)
    ax2.set_title('Pixel Value Statistics')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Pixel Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Image mode distribution
    unique_modes = list(set(modes))
    mode_counts = [modes.count(mode) for mode in unique_modes]
    ax3.pie(mode_counts, labels=unique_modes, autopct='%1.1f%%')
    ax3.set_title('Image Mode Distribution')
    
    # Plot 4: Mean value histogram
    ax4.hist(mean_values, bins=15, alpha=0.7, edgecolor='black')
    ax4.set_title('Mean Pixel Value Distribution')
    ax4.set_xlabel('Mean Pixel Value')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    stats_path = os.path.join(output_dir, 'sample_statistics.png')
    plt.savefig(stats_path, dpi=150, bbox_inches='tight')
    print(f"Statistical analysis saved to: {stats_path}")
    plt.close()
    
    # Analysis 2: Sample quality metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('RadioDiff VAE Sample Images - Quality Analysis', fontsize=14, fontweight='bold')
    
    # Calculate simple quality metrics
    std_values = [np.array(img).std() for img in images]
    contrast_values = [np.array(img).max() - np.array(img).min() for img in images]
    
    # Plot 1: Standard deviation (image complexity)
    ax1.plot(range(len(images)), std_values, 'b-o', markersize=4, linewidth=1)
    ax1.set_title('Image Complexity (Standard Deviation)')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Standard Deviation')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Contrast range
    ax2.plot(range(len(images)), contrast_values, 'r-s', markersize=4, linewidth=1)
    ax2.set_title('Image Contrast (Max-Min)')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Contrast Range')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    quality_path = os.path.join(output_dir, 'sample_quality_analysis.png')
    plt.savefig(quality_path, dpi=150, bbox_inches='tight')
    print(f"Quality analysis saved to: {quality_path}")
    plt.close()

def main():
    """Main function to execute the visualization pipeline."""
    # Configuration
    image_dir = "/home/cine/Documents/Github/RadioDiff/radiodiff_Vae"
    output_dir = "/home/cine/Documents/Github/RadioDiff/enhanced_suite/visualization"
    main_output_path = os.path.join(output_dir, "radiodiff_vae_samples_grid.png")
    analysis_dir = os.path.join(output_dir, "vae_sample_analysis")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    print("RadioDiff VAE Sample Visualization Script")
    print("=" * 50)
    
    try:
        # Load images
        print(f"\nLoading images from: {image_dir}")
        images, image_paths = load_sample_images(image_dir)
        
        # Create main visualization grid
        print(f"\nCreating 3×10 grid visualization...")
        create_visualization_grid(images, main_output_path, rows=3, cols=10)
        
        # Create detailed analysis
        print(f"\nCreating detailed analysis...")
        create_detailed_analysis(images, analysis_dir)
        
        # Print summary
        print(f"\n{'='*50}")
        print("VISUALIZATION COMPLETE")
        print(f"{'='*50}")
        print(f"Total images processed: {len(images)}")
        print(f"Main visualization: {main_output_path}")
        print(f"Analysis directory: {analysis_dir}")
        print(f"Analysis files:")
        print(f"  - Statistical analysis: {os.path.join(analysis_dir, 'sample_statistics.png')}")
        print(f"  - Quality analysis: {os.path.join(analysis_dir, 'sample_quality_analysis.png')}")
        
        # Additional information
        print(f"\nImage Details:")
        print(f"  - First image: {os.path.basename(image_paths[0])}")
        print(f"  - Last image: {os.path.basename(image_paths[-1])}")
        print(f"  - Image sizes: {[img.size for img in images[:3]]}... (showing first 3)")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())