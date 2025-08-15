#!/usr/bin/env python3
"""
Improved visualization of RadioDiff VAE training log with proper multi-axis display.
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

def create_improved_visualizations(df, output_dir='radiodiff_Vae'):
    """Create improved visualizations with proper multi-axis display."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating improved visualizations with {len(df)} data points...")
    
    # 1. Create improved normalized comparison with total loss
    print("Creating improved normalized comparison...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    key_metrics = [
        ('train/total_loss', 'Total Loss', 'blue'),
        ('train/kl_loss', 'KL Loss', 'red'), 
        ('train/rec_loss', 'Reconstruction Loss', 'green')
    ]
    
    # Plot each normalized metric
    for metric, title, color in key_metrics:
        if metric in df.columns:
            valid_data = df.dropna(subset=[metric])
            if len(valid_data) > 0:
                min_val = valid_data[metric].min()
                max_val = valid_data[metric].max()
                if max_val > min_val:
                    normalized = (valid_data[metric] - min_val) / (max_val - min_val)
                    ax.plot(valid_data['step'], normalized, color=color, linewidth=3, label=title)
                    print(f"  Added {title}: {len(valid_data)} points, range [{min_val:.2f}, {max_val:.2f}]")
    
    ax.set_title('Normalized Loss Comparison - All Components', fontsize=16, fontweight='bold')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Normalized Loss [0,1]', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'normalized_comparison_improved.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: normalized_comparison_improved.png")
    
    # 2. Create improved multi-axis plot with proper legend
    print("Creating improved multi-axis plot...")
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Setup multiple y-axes
    ax1 = ax
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    
    # Offset the third y-axis
    ax3.spines['right'].set_position(('outward', 60))
    
    # Plot Total Loss
    if 'train/total_loss' in df.columns:
        valid_total = df.dropna(subset=['train/total_loss'])
        if len(valid_total) > 0:
            line1 = ax1.plot(valid_total['step'], valid_total['train/total_loss'], 
                           'b-', linewidth=3, label='Total Loss')
            ax1.set_ylabel('Total Loss', color='b', fontsize=12)
            ax1.tick_params(axis='y', labelcolor='b')
            print(f"  Added Total Loss: {len(valid_total)} points")
    
    # Plot KL Loss
    if 'train/kl_loss' in df.columns:
        valid_kl = df.dropna(subset=['train/kl_loss'])
        if len(valid_kl) > 0:
            line2 = ax2.plot(valid_kl['step'], valid_kl['train/kl_loss'], 
                           'r-', linewidth=3, label='KL Loss')
            ax2.set_ylabel('KL Loss', color='r', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='r')
            print(f"  Added KL Loss: {len(valid_kl)} points")
    
    # Plot Reconstruction Loss
    if 'train/rec_loss' in df.columns:
        valid_rec = df.dropna(subset=['train/rec_loss'])
        if len(valid_rec) > 0:
            line3 = ax3.plot(valid_rec['step'], valid_rec['train/rec_loss'], 
                           'g-', linewidth=3, label='Reconstruction Loss')
            ax3.set_ylabel('Reconstruction Loss', color='g', fontsize=12)
            ax3.tick_params(axis='y', labelcolor='g')
            print(f"  Added Reconstruction Loss: {len(valid_rec)} points")
    
    # Create custom legend
    lines = []
    labels = []
    
    if 'train/total_loss' in df.columns:
        lines.append(line1[0])
        labels.append('Total Loss')
    if 'train/kl_loss' in df.columns:
        lines.append(line2[0])
        labels.append('KL Loss')
    if 'train/rec_loss' in df.columns:
        lines.append(line3[0])
        labels.append('Reconstruction Loss')
    
    if lines:
        ax1.legend(lines, labels, loc='upper right', fontsize=12)
    
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_title('Multi-axis Loss Analysis - All Components', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multi_axis_losses_improved.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: multi_axis_losses_improved.png")
    
    # 3. Create comprehensive overview with total loss
    print("Creating comprehensive overview...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('RadioDiff VAE Training Metrics - Complete Overview', fontsize=16, fontweight='bold')
    
    metrics_to_plot = [
        ('train/total_loss', 'Total Loss', 'blue', 0, 0),
        ('train/kl_loss', 'KL Loss', 'red', 0, 1),
        ('train/rec_loss', 'Reconstruction Loss', 'green', 1, 0),
        ('train/disc_loss', 'Discriminator Loss', 'orange', 1, 1)
    ]
    
    for metric, title, color, row, col in metrics_to_plot:
        if metric in df.columns:
            ax = axes[row, col]
            valid_data = df.dropna(subset=[metric])
            
            if len(valid_data) > 0:
                ax.plot(valid_data['step'], valid_data[metric], color=color, linewidth=2.5)
                ax.set_title(f'{title} ({len(valid_data)} points)', fontsize=12, fontweight='bold')
                ax.set_xlabel('Training Step', fontsize=10)
                ax.set_ylabel(title, fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.relim()
                ax.autoscale_view()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_overview_improved.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: metrics_overview_improved.png")
    
    print(f"\nAll improved visualizations saved to {output_dir}/")

def main():
    log_file_path = '/home/cine/Documents/Github/RadioDiff/radiodiff_Vae/2025-08-15-17-21_.log'
    
    print("Parsing training log...")
    df = parse_log_file_fixed(log_file_path)
    
    if len(df) == 0:
        print("No training data found in log file.")
        return
    
    print(f"Found {len(df)} unique training steps")
    print(f"Step range: {df['step'].min()} to {df['step'].max()}")
    
    # Print data verification
    print("\n=== Data Verification ===")
    key_metrics = ['train/total_loss', 'train/kl_loss', 'train/rec_loss', 'train/disc_loss']
    for metric in key_metrics:
        if metric in df.columns:
            values = df[metric].dropna()
            print(f"{metric}: {len(values)} non-null values")
    
    create_improved_visualizations(df)
    
    print("Improved visualization complete!")

if __name__ == "__main__":
    main()