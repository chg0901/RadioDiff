#!/usr/bin/env python3
"""
Fixed visualization of RadioDiff VAE training log with proper parsing and y-axis scaling.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Set matplotlib to use non-interactive backend
plt.switch_backend('Agg')

def parse_log_file_fixed(log_file_path):
    """Parse the training log file, merging duplicate step entries."""
    
    # Pattern to match training step lines
    pattern = r'\[Train Step\] (\d+)/\d+: (.+?)(?= lr: 0\.0+,\s*$)'
    
    # Dictionary to store merged data by step
    step_data = {}
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        match = re.search(pattern, line)
        if match:
            step = int(match.group(1))
            metrics_str = match.group(2)
            
            # Initialize step entry if not exists
            if step not in step_data:
                step_data[step] = {'step': step}
            
            # Parse individual metrics and merge
            for metric in metrics_str.split(', '):
                if ': ' in metric:
                    key, value = metric.split(': ')
                    try:
                        step_data[step][key] = float(value)
                    except ValueError:
                        continue
    
    # Convert to DataFrame and sort by step
    df = pd.DataFrame(list(step_data.values()))
    df = df.sort_values('step').reset_index(drop=True)
    
    return df

def create_fixed_visualizations(df, output_dir='radiodiff_Vae'):
    """Create fixed visualizations with proper y-axis scaling."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating visualizations with {len(df)} data points...")
    
    # Individual metric plots with proper scaling
    metrics_to_plot = [
        ('train/total_loss', 'Total Loss', 'blue'),
        ('train/kl_loss', 'KL Loss', 'red'),
        ('train/rec_loss', 'Reconstruction Loss', 'green'),
        ('train/g_loss', 'Generator Loss', 'purple'),
        ('train/disc_loss', 'Discriminator Loss', 'orange')
    ]
    
    print("Creating individual metric plots...")
    
    # Create individual plots for each metric
    for metric, title, color in metrics_to_plot:
        if metric in df.columns:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Remove NaN values for plotting
            valid_data = df.dropna(subset=[metric])
            
            if len(valid_data) > 0:
                # Plot raw data
                ax.plot(valid_data['step'], valid_data[metric], color=color, alpha=0.6, linewidth=1, label='Raw')
                
                # Add moving average for trend
                window_size = max(5, len(valid_data) // 20)  # Adaptive window size
                moving_avg = valid_data[metric].rolling(window=window_size, center=True).mean()
                ax.plot(valid_data['step'], moving_avg, color=color, linewidth=3, label=f'Moving Avg (window={window_size})')
                
                # Add trend line
                if len(valid_data) > 2:
                    z = np.polyfit(valid_data['step'], valid_data[metric], 1)
                    p = np.poly1d(z)
                    ax.plot(valid_data['step'], p(valid_data['step']), '--', color='black', linewidth=2, label='Trend')
                
                ax.set_title(f'{title} - Training Progress', fontsize=14, fontweight='bold')
                ax.set_xlabel('Training Step', fontsize=12)
                ax.set_ylabel(title, fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Adjust y-axis to show the trend better
                y_min, y_max = valid_data[metric].min(), valid_data[metric].max()
                y_range = y_max - y_min
                if y_range > 0:
                    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
                
                plt.tight_layout()
                filename = f'{metric.replace("/", "_")}_fixed.png'
                plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved: {filename} ({len(valid_data)} data points)")
            else:
                print(f"  No valid data for {metric}")
    
    print("Creating comprehensive overview...")
    
    # Create a comprehensive overview with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('RadioDiff VAE Training Metrics Overview', fontsize=16, fontweight='bold')
    
    # Plot each metric in its own subplot with proper scaling
    for idx, (metric, title, color) in enumerate(metrics_to_plot[:4]):  # Limit to 4 plots
        if metric in df.columns:
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # Remove NaN values
            valid_data = df.dropna(subset=[metric])
            
            if len(valid_data) > 0:
                # Plot raw data with low alpha
                ax.plot(valid_data['step'], valid_data[metric], color=color, alpha=0.4, linewidth=1)
                
                # Plot moving average
                window_size = max(5, len(valid_data) // 20)
                moving_avg = valid_data[metric].rolling(window=window_size, center=True).mean()
                ax.plot(valid_data['step'], moving_avg, color=color, linewidth=2.5)
                
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.set_xlabel('Training Step', fontsize=10)
                ax.set_ylabel(title, fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Auto-scale y-axis for each subplot
                ax.relim()
                ax.autoscale_view()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_overview_fixed.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: metrics_overview_fixed.png")
    
    print("Creating normalized comparison...")
    
    # Create a loss progression comparison (normalized)
    key_metrics = [('train/total_loss', 'Total Loss', 'blue'), 
                   ('train/kl_loss', 'KL Loss', 'red'), 
                   ('train/rec_loss', 'Reconstruction Loss', 'green')]
    
    # Check if we have data for all key metrics
    if all(metric in df.columns for metric, _, _ in key_metrics):
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Normalize each metric to [0, 1] range for comparison
        for metric, title, color in key_metrics:
            if metric in df.columns:
                valid_data = df.dropna(subset=[metric])
                if len(valid_data) > 0:
                    min_val = valid_data[metric].min()
                    max_val = valid_data[metric].max()
                    if max_val > min_val:
                        normalized = (valid_data[metric] - min_val) / (max_val - min_val)
                        ax.plot(valid_data['step'], normalized, color=color, linewidth=2, label=title)
        
        ax.set_title('Normalized Loss Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Normalized Loss', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'normalized_comparison_fixed.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("  Saved: normalized_comparison_fixed.png")
    
    print("Creating multi-axis plot...")
    
    # Create training progress summary with multiple y-axes
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot multiple metrics with different y-axes
    ax1 = ax
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    
    # Offset the third y-axis
    ax3.spines['right'].set_position(('outward', 60))
    
    # Get valid data for each metric
    if 'train/total_loss' in df.columns:
        valid_total = df.dropna(subset=['train/total_loss'])
        if len(valid_total) > 0:
            ax1.plot(valid_total['step'], valid_total['train/total_loss'], 'b-', linewidth=2, label='Total Loss')
            ax1.set_ylabel('Total Loss', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
    
    if 'train/kl_loss' in df.columns:
        valid_kl = df.dropna(subset=['train/kl_loss'])
        if len(valid_kl) > 0:
            ax2.plot(valid_kl['step'], valid_kl['train/kl_loss'], 'r-', linewidth=2, label='KL Loss')
            ax2.set_ylabel('KL Loss', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
    
    if 'train/rec_loss' in df.columns:
        valid_rec = df.dropna(subset=['train/rec_loss'])
        if len(valid_rec) > 0:
            ax3.plot(valid_rec['step'], valid_rec['train/rec_loss'], 'g-', linewidth=2, label='Reconstruction Loss')
            ax3.set_ylabel('Reconstruction Loss', color='g')
            ax3.tick_params(axis='y', labelcolor='g')
    
    ax1.set_xlabel('Training Step')
    ax1.set_title('Training Losses with Multiple Y-Axes', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multi_axis_losses_fixed.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: multi_axis_losses_fixed.png")
    
    print(f"\nAll fixed visualizations saved to {output_dir}/")

def main():
    log_file_path = '/home/cine/Documents/Github/RadioDiff/radiodiff_Vae/2025-08-15-17-21_.log'
    
    print("Parsing training log with fixed function...")
    df = parse_log_file_fixed(log_file_path)
    
    if len(df) == 0:
        print("No training data found in log file.")
        return
    
    print(f"Found {len(df)} unique training steps")
    print(f"Step range: {df['step'].min()} to {df['step'].max()}")
    
    # Print basic statistics
    print("\n=== Training Statistics ===")
    metrics_to_analyze = ['train/total_loss', 'train/kl_loss', 'train/rec_loss', 'train/g_loss', 'train/disc_loss']
    for metric in metrics_to_analyze:
        if metric in df.columns:
            values = df[metric].dropna()
            if len(values) > 0:
                print(f"\n{metric}:")
                print(f"  Count: {len(values)}")
                print(f"  Min: {values.min():.2f}")
                print(f"  Max: {values.max():.2f}")
                print(f"  Mean: {values.mean():.2f}")
                print(f"  Std: {values.std():.2f}")
                print(f"  Latest: {values.iloc[-1]:.2f}")
    
    create_fixed_visualizations(df)
    
    print("Fixed visualization complete!")

if __name__ == "__main__":
    main()