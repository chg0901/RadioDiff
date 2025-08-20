#!/usr/bin/env python3
"""
Script to visualize IRT4 training progress through gt-sample pairs
"""
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from datetime import datetime
import re

def find_image_pairs(directory):
    """Find gt-sample pairs in directory"""
    gt_files = sorted(glob.glob(os.path.join(directory, "gt-sample-*.png")))
    sample_files = sorted(glob.glob(os.path.join(directory, "sample-*.png")))
    
    # Extract numbers from filenames
    gt_numbers = []
    for f in gt_files:
        match = re.search(r'gt-sample-(\d+)\.png', f)
        if match:
            gt_numbers.append(int(match.group(1)))
    
    sample_numbers = []
    for f in sample_files:
        match = re.search(r'sample-(\d+)\.png', f)
        if match:
            sample_numbers.append(int(match.group(1)))
    
    # Find common numbers
    common_numbers = sorted(set(gt_numbers) & set(sample_numbers))
    
    pairs = []
    for num in common_numbers:
        gt_path = os.path.join(directory, f"gt-sample-{num}.png")
        sample_path = os.path.join(directory, f"sample-{num}.png")
        if os.path.exists(gt_path) and os.path.exists(sample_path):
            pairs.append((num, gt_path, sample_path))
    
    return sorted(pairs)

def create_pair_visualization(pairs, title, output_path, figsize=(15, 10)):
    """Create visualization of gt-sample pairs"""
    n_pairs = len(pairs)
    if n_pairs == 0:
        print(f"No pairs found for {title}")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_pairs, 2, figsize=figsize)
    if n_pairs == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for i, (step, gt_path, sample_path) in enumerate(pairs):
        # Load images
        gt_img = mpimg.imread(gt_path)
        sample_img = mpimg.imread(sample_path)
        
        # Plot ground truth
        axes[i, 0].imshow(gt_img, cmap='viridis')
        axes[i, 0].set_title(f'Ground Truth (Step {step})', fontweight='bold')
        axes[i, 0].axis('off')
        
        # Plot sample
        axes[i, 1].imshow(sample_img, cmap='viridis')
        axes[i, 1].set_title(f'Generated Sample (Step {step})', fontweight='bold')
        axes[i, 1].axis('off')
        
        # Add step information
        axes[i, 0].text(0.02, 0.98, f'Step {step}', transform=axes[i, 0].transAxes, 
                      fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set column labels
    axes[0, 0].set_xlabel('Ground Truth', fontweight='bold', labelpad=20)
    axes[0, 1].set_xlabel('Generated Sample', fontweight='bold', labelpad=20)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created visualization: {output_path}")

def create_learning_progress_visualization(all_pairs, output_path):
    """Create a comprehensive learning progress visualization"""
    # Select key points to show learning progression
    if len(all_pairs) <= 10:
        selected_pairs = all_pairs
    else:
        # Select first 3, middle 3, and last 4
        first_part = all_pairs[:3]
        last_part = all_pairs[-4:]
        middle_start = len(all_pairs) // 2 - 1
        middle_part = all_pairs[middle_start:middle_start + 3]
        selected_pairs = first_part + middle_part + last_part
    
    n_pairs = len(selected_pairs)
    fig, axes = plt.subplots(n_pairs, 3, figsize=(18, 4 * n_pairs))
    if n_pairs == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('IRT4 Training Progress: Learning Process Visualization', fontsize=18, fontweight='bold')
    
    for i, (step, gt_path, sample_path) in enumerate(selected_pairs):
        # Load images
        gt_img = mpimg.imread(gt_path)
        sample_img = mpimg.imread(sample_path)
        
        # Plot ground truth
        axes[i, 0].imshow(gt_img, cmap='viridis')
        axes[i, 0].set_title(f'Ground Truth', fontweight='bold')
        axes[i, 0].axis('off')
        
        # Plot sample
        axes[i, 1].imshow(sample_img, cmap='viridis')
        axes[i, 1].set_title(f'Generated Sample', fontweight='bold')
        axes[i, 1].axis('off')
        
        # Plot difference
        diff_img = np.abs(gt_img - sample_img)
        axes[i, 2].imshow(diff_img, cmap='hot')
        axes[i, 2].set_title(f'Difference Map', fontweight='bold')
        axes[i, 2].axis('off')
        
        # Add step information
        axes[i, 0].text(0.02, 0.98, f'Step {step}', transform=axes[i, 0].transAxes, 
                      fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Calculate and display MSE
        mse = np.mean((gt_img - sample_img) ** 2)
        axes[i, 1].text(0.02, 0.98, f'MSE: {mse:.4f}', transform=axes[i, 1].transAxes, 
                      fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Set column labels
    axes[0, 0].set_xlabel('Ground Truth', fontweight='bold', labelpad=20)
    axes[0, 1].set_xlabel('Generated Sample', fontweight='bold', labelpad=20)
    axes[0, 2].set_xlabel('Difference Map', fontweight='bold', labelpad=20)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created learning progress visualization: {output_path}")

def main():
    # Directories
    irt4_train2_dir = '/mnt/hdd/IRT4_Train2'
    irt4_train_dir = '/mnt/hdd/IRT4_Train'
    output_dir = '/home/cine/Documents/Github/RadioDiff'
    
    print("=== IRT4 Training Progress Visualization ===")
    
    # Process IRT4_Train2 (Session 2)
    print("\n1. Processing IRT4_Train2 (Session 2)...")
    train2_pairs = find_image_pairs(irt4_train2_dir)
    print(f"Found {len(train2_pairs)} pairs in IRT4_Train2")
    
    if train2_pairs:
        # First 5 pairs
        first5_train2 = train2_pairs[:5]
        create_pair_visualization(first5_train2, 
                                'IRT4 Session 2: First 5 Training Steps (Early Learning)',
                                os.path.join(output_dir, 'irt4_train2_first5_pairs.png'),
                                figsize=(15, 12))
    
    # Process IRT4_Train (Session 1)
    print("\n2. Processing IRT4_Train (Session 1)...")
    train_pairs = find_image_pairs(irt4_train_dir)
    print(f"Found {len(train_pairs)} pairs in IRT4_Train")
    
    if train_pairs:
        # First 5 pairs
        first5_train = train_pairs[:5]
        create_pair_visualization(first5_train,
                                'IRT4 Session 1: First 5 Training Steps (Early Learning)',
                                os.path.join(output_dir, 'irt4_train_first5_pairs.png'),
                                figsize=(15, 12))
        
        # Last 5 pairs
        last5_train = train_pairs[-5:]
        create_pair_visualization(last5_train,
                                'IRT4 Session 1: Last 5 Training Steps (Final Performance)',
                                os.path.join(output_dir, 'irt4_train_last5_pairs.png'),
                                figsize=(15, 12))
        
        # Learning progress visualization
        create_learning_progress_visualization(train_pairs,
                                             os.path.join(output_dir, 'irt4_learning_progress.png'))
    
    # Summary statistics
    print("\n=== Summary ===")
    print(f"IRT4_Train2: {len(train2_pairs)} pairs")
    print(f"IRT4_Train: {len(train_pairs)} pairs")
    
    if train_pairs:
        steps = [pair[0] for pair in train_pairs]
        print(f"Training steps range: {min(steps)} to {max(steps)}")
        print(f"Total training checkpoints: {len(steps)}")

if __name__ == "__main__":
    main()