#!/usr/bin/env python3
"""
Detailed analysis of RadioDiff VAE training progress.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os

def parse_log_file_detailed(log_file_path):
    """Parse the training log file with more detailed metrics."""
    
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

def analyze_training_progress(df, output_dir='radiodiff_Vae'):
    """Create detailed analysis of training progress."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Print summary statistics
    print("=== Training Progress Analysis ===")
    print(f"Total steps analyzed: {len(df)}")
    print(f"Step range: {df['step'].min()} to {df['step'].max()}")
    print(f"Total training steps configured: 150000")
    print(f"Progress: {(df['step'].max() / 150000) * 100:.1f}%")
    
    # Loss statistics
    loss_columns = ['train/total_loss', 'train/kl_loss', 'train/rec_loss', 'train/g_loss']
    for col in loss_columns:
        if col in df.columns:
            values = df[col].dropna()
            if len(values) > 0:
                print(f"\n{col}:")
                print(f"  Min: {values.min():.2f}")
                print(f"  Max: {values.max():.2f}")
                print(f"  Mean: {values.mean():.2f}")
                print(f"  Std: {values.std():.2f}")
                print(f"  Latest: {values.iloc[-1]:.2f}")
    
    # Create detailed plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('RadioDiff VAE Training Progress - Detailed Analysis', fontsize=16)
    
    # Plot 1: Total Loss over time
    if 'train/total_loss' in df.columns:
        axes[0, 0].plot(df['step'], df['train/total_loss'], 'b-', linewidth=2)
        axes[0, 0].set_title('Total Loss Over Training')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: KL Loss over time
    if 'train/kl_loss' in df.columns:
        axes[0, 1].plot(df['step'], df['train/kl_loss'], 'r-', linewidth=2)
        axes[0, 1].set_title('KL Loss Over Training')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('KL Loss')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Reconstruction Loss over time
    if 'train/rec_loss' in df.columns:
        axes[0, 2].plot(df['step'], df['train/rec_loss'], 'g-', linewidth=2)
        axes[0, 2].set_title('Reconstruction Loss Over Training')
        axes[0, 2].set_xlabel('Training Step')
        axes[0, 2].set_ylabel('Reconstruction Loss')
        axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Generator Loss over time
    if 'train/g_loss' in df.columns:
        axes[1, 0].plot(df['step'], df['train/g_loss'], 'm-', linewidth=2)
        axes[1, 0].set_title('Generator Loss Over Training')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Generator Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Loss comparison (normalized)
    if all(col in df.columns for col in ['train/total_loss', 'train/kl_loss', 'train/rec_loss']):
        # Normalize losses for comparison
        total_norm = df['train/total_loss'] / df['train/total_loss'].max()
        kl_norm = df['train/kl_loss'] / df['train/kl_loss'].max()
        rec_norm = df['train/rec_loss'] / df['train/rec_loss'].max()
        
        axes[1, 1].plot(df['step'], total_norm, 'b-', linewidth=2, label='Total Loss (norm)')
        axes[1, 1].plot(df['step'], kl_norm, 'r-', linewidth=2, label='KL Loss (norm)')
        axes[1, 1].plot(df['step'], rec_norm, 'g-', linewidth=2, label='Rec Loss (norm)')
        axes[1, 1].set_title('Normalized Loss Comparison')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Normalized Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Loss trends (moving average)
    if 'train/total_loss' in df.columns:
        window = min(10, len(df) // 10)
        if window > 1:
            moving_avg = df['train/total_loss'].rolling(window=window).mean()
            axes[1, 2].plot(df['step'], df['train/total_loss'], 'b-', alpha=0.3, label='Raw')
            axes[1, 2].plot(df['step'], moving_avg, 'b-', linewidth=2, label=f'MA ({window})')
            axes[1, 2].set_title('Total Loss with Moving Average')
            axes[1, 2].set_xlabel('Training Step')
            axes[1, 2].set_ylabel('Total Loss')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_training_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Early training analysis (first 1000 steps)
    early_df = df[df['step'] <= 1000]
    if len(early_df) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Early Training Progress (First 1000 Steps)', fontsize=16)
        
        if 'train/total_loss' in early_df.columns:
            axes[0, 0].plot(early_df['step'], early_df['train/total_loss'], 'b-', linewidth=2)
            axes[0, 0].set_title('Total Loss - Early Training')
            axes[0, 0].set_xlabel('Training Step')
            axes[0, 0].set_ylabel('Total Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        if 'train/kl_loss' in early_df.columns:
            axes[0, 1].plot(early_df['step'], early_df['train/kl_loss'], 'r-', linewidth=2)
            axes[0, 1].set_title('KL Loss - Early Training')
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('KL Loss')
            axes[0, 1].grid(True, alpha=0.3)
        
        if 'train/rec_loss' in early_df.columns:
            axes[1, 0].plot(early_df['step'], early_df['train/rec_loss'], 'g-', linewidth=2)
            axes[1, 0].set_title('Reconstruction Loss - Early Training')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Reconstruction Loss')
            axes[1, 0].grid(True, alpha=0.3)
        
        if 'train/g_loss' in early_df.columns:
            axes[1, 1].plot(early_df['step'], early_df['train/g_loss'], 'm-', linewidth=2)
            axes[1, 1].set_title('Generator Loss - Early Training')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Generator Loss')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'early_training_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    print(f"\nDetailed analysis plots saved to {output_dir}/")

def main():
    log_file_path = '/home/cine/Documents/Github/RadioDiff/radiodiff_Vae/2025-08-15-17-21_.log'
    
    print("Parsing training log for detailed analysis...")
    df = parse_log_file_detailed(log_file_path)
    
    if len(df) == 0:
        print("No training data found in log file.")
        return
    
    analyze_training_progress(df)
    
    print("Detailed analysis complete!")

if __name__ == "__main__":
    main()