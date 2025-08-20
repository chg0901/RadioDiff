#!/usr/bin/env python3
"""
Improved visualization of RadioDiff VAE training log with proper y-axis scaling.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os

def parse_log_file_detailed(log_file_path):
    """Parse the training log file with detailed metrics."""
    
    # Pattern to match training step lines
    pattern = r'\[Train Step\] (\d+)/\d+: (.+?)(?= lr: 0\.0+,\s*$)'
    
    data = []
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        match = re.search(pattern, line)
        if match:
            step = int(match.group(1))
            metrics_str = match.group(2)
            
            # Parse individual metrics
            metrics_dict = {'step': step}
            for metric in metrics_str.split(', '):
                if ': ' in metric:
                    key, value = metric.split(': ')
                    try:
                        metrics_dict[key] = float(value)
                    except ValueError:
                        continue
            
            data.append(metrics_dict)
    
    return pd.DataFrame(data)

def create_improved_visualizations(df, output_dir='radiodiff_Vae'):
    """Create improved visualizations with proper y-axis scaling."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Individual metric plots with proper scaling
    metrics_to_plot = [
        ('train/total_loss', 'Total Loss', 'blue'),
        ('train/kl_loss', 'KL Loss', 'red'),
        ('train/rec_loss', 'Reconstruction Loss', 'green'),
        ('train/g_loss', 'Generator Loss', 'purple')
    ]
    
    # Create individual plots for each metric
    for metric, title, color in metrics_to_plot:
        if metric in df.columns:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot raw data
            ax.plot(df['step'], df[metric], color=color, alpha=0.6, linewidth=1, label='Raw')
            
            # Add moving average for trend
            window_size = max(5, len(df) // 20)  # Adaptive window size
            moving_avg = df[metric].rolling(window=window_size, center=True).mean()
            ax.plot(df['step'], moving_avg, color=color, linewidth=3, label=f'Moving Avg (window={window_size})')
            
            # Add trend line
            if len(df) > 2:
                z = np.polyfit(df['step'], df[metric], 1)
                p = np.poly1d(z)
                ax.plot(df['step'], p(df['step']), '--', color='black', linewidth=2, label='Trend')
            
            ax.set_title(f'{title} - Training Progress', fontsize=14, fontweight='bold')
            ax.set_xlabel('Training Step', fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Adjust y-axis to show the trend better
            y_min, y_max = df[metric].min(), df[metric].max()
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            
            plt.tight_layout()
            filename = f'{metric.replace("/", "_")}_improved.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            plt.show()
    
    # Create a comprehensive overview with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('RadioDiff VAE Training Metrics Overview', fontsize=16, fontweight='bold')
    
    # Plot each metric in its own subplot with proper scaling
    for idx, (metric, title, color) in enumerate(metrics_to_plot):
        if metric in df.columns:
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # Plot raw data with low alpha
            ax.plot(df['step'], df[metric], color=color, alpha=0.4, linewidth=1)
            
            # Plot moving average
            window_size = max(5, len(df) // 20)
            moving_avg = df[metric].rolling(window=window_size, center=True).mean()
            ax.plot(df['step'], moving_avg, color=color, linewidth=2.5)
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Training Step', fontsize=10)
            ax.set_ylabel(title, fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Auto-scale y-axis for each subplot
            ax.relim()
            ax.autoscale_view()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_overview_improved.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a loss progression comparison (normalized)
    if all(metric in df.columns for metric, _, _ in metrics_to_plot[:3]):  # total, kl, rec
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Normalize each metric to [0, 1] range for comparison
        for metric, title, color in metrics_to_plot[:3]:
            if metric in df.columns:
                normalized = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
                ax.plot(df['step'], normalized, color=color, linewidth=2, label=title)
        
        ax.set_title('Normalized Loss Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Normalized Loss', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'normalized_comparison_improved.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # Create training progress summary
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot multiple metrics with different y-axes
    ax1 = ax
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    
    # Offset the third y-axis
    ax3.spines['right'].set_position(('outward', 60))
    
    if 'train/total_loss' in df.columns:
        line1 = ax1.plot(df['step'], df['train/total_loss'], 'b-', linewidth=2, label='Total Loss')
        ax1.set_ylabel('Total Loss', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
    
    if 'train/kl_loss' in df.columns:
        line2 = ax2.plot(df['step'], df['train/kl_loss'], 'r-', linewidth=2, label='KL Loss')
        ax2.set_ylabel('KL Loss', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
    
    if 'train/rec_loss' in df.columns:
        line3 = ax3.plot(df['step'], df['train/rec_loss'], 'g-', linewidth=2, label='Reconstruction Loss')
        ax3.set_ylabel('Reconstruction Loss', color='g')
        ax3.tick_params(axis='y', labelcolor='g')
    
    ax1.set_xlabel('Training Step')
    ax1.set_title('Training Losses with Multiple Y-Axes', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multi_axis_losses.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Improved visualizations saved to {output_dir}/")

def main():
    log_file_path = '/home/cine/Documents/Github/RadioDiff/radiodiff_Vae/2025-08-15-17-21_.log'
    
    print("Parsing training log for improved visualization...")
    df = parse_log_file_detailed(log_file_path)
    
    if len(df) == 0:
        print("No training data found in log file.")
        return
    
    print(f"Found {len(df)} training steps")
    print(f"Step range: {df['step'].min()} to {df['step'].max()}")
    
    create_improved_visualizations(df)
    
    print("Improved visualization complete!")

if __name__ == "__main__":
    main()