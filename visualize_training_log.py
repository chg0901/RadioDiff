#!/usr/bin/env python3
"""
Visualize training log data from RadioDiff VAE training.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def parse_log_file(log_file_path):
    """Parse the training log file and extract metrics."""
    
    # Pattern to match training step lines
    pattern = r'\[Train Step\] (\d+)/\d+: (.+?)(?= lr: 0\.0+,\s*$)'
    
    steps = []
    metrics = {
        'total_loss': [],
        'logvar': [],
        'kl_loss': [],
        'nll_loss': [],
        'rec_loss': [],
        'd_weight': [],
        'disc_factor': [],
        'g_loss': []
    }
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        match = re.search(pattern, line)
        if match:
            step = int(match.group(1))
            metrics_str = match.group(2)
            
            # Parse individual metrics
            metrics_dict = {}
            for metric in metrics_str.split(', '):
                if ': ' in metric:
                    key, value = metric.split(': ')
                    try:
                        metrics_dict[key] = float(value)
                    except ValueError:
                        continue
            
            steps.append(step)
            for key in metrics:
                if key in metrics_dict:
                    metrics[key].append(metrics_dict[key])
                else:
                    metrics[key].append(np.nan)
    
    return steps, metrics

def create_visualizations(steps, metrics, output_dir='radiodiff_Vae'):
    """Create visualizations for the training metrics."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Main loss metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('RadioDiff VAE Training Metrics', fontsize=16)
    
    # Total Loss
    axes[0, 0].plot(steps, metrics['total_loss'], 'b-', linewidth=2)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # KL Loss
    axes[0, 1].plot(steps, metrics['kl_loss'], 'r-', linewidth=2)
    axes[0, 1].set_title('KL Loss')
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Reconstruction Loss
    axes[1, 0].plot(steps, metrics['rec_loss'], 'g-', linewidth=2)
    axes[1, 0].set_title('Reconstruction Loss')
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Generator Loss
    axes[1, 1].plot(steps, metrics['g_loss'], 'm-', linewidth=2)
    axes[1, 1].set_title('Generator Loss')
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Combined loss plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Normalize losses for better visualization
    total_loss_norm = np.array(metrics['total_loss']) / np.max(metrics['total_loss'])
    kl_loss_norm = np.array(metrics['kl_loss']) / np.max(metrics['kl_loss'])
    rec_loss_norm = np.array(metrics['rec_loss']) / np.max(metrics['rec_loss'])
    
    ax.plot(steps, total_loss_norm, 'b-', linewidth=2, label='Total Loss (normalized)')
    ax.plot(steps, kl_loss_norm, 'r-', linewidth=2, label='KL Loss (normalized)')
    ax.plot(steps, rec_loss_norm, 'g-', linewidth=2, label='Reconstruction Loss (normalized)')
    
    ax.set_title('Normalized Training Losses Comparison')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Normalized Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'normalized_losses.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Loss progression (first 1000 steps for detailed view)
    if len(steps) > 1000:
        early_steps = steps[:1000]
        early_metrics = {key: values[:1000] for key, values in metrics.items()}
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(early_steps, early_metrics['total_loss'], 'b-', linewidth=2, label='Total Loss')
        ax.set_title('Training Progress - First 1000 Steps')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'early_training.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    print(f"Visualizations saved to {output_dir}/")

def main():
    log_file_path = '/home/cine/Documents/Github/RadioDiff/radiodiff_Vae/2025-08-15-17-21_.log'
    
    print("Parsing training log...")
    steps, metrics = parse_log_file(log_file_path)
    
    print(f"Found {len(steps)} training steps")
    print(f"Step range: {min(steps)} to {max(steps)}")
    
    # Print some statistics
    for key, values in metrics.items():
        if values and not all(np.isnan(v) for v in values):
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                print(f"{key}: min={min(valid_values):.2f}, max={max(valid_values):.2f}, mean={np.mean(valid_values):.2f}")
    
    print("\nCreating visualizations...")
    create_visualizations(steps, metrics)
    
    print("Visualization complete!")

if __name__ == "__main__":
    main()