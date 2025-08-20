#!/usr/bin/env python3
"""
Comprehensive VAE Training Visualization and Analysis Script
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Set matplotlib to use non-interactive backend
plt.switch_backend('Agg')

def parse_vae_log_file(log_file_path):
    """Parse VAE training log file with comprehensive metrics extraction"""
    
    pattern = r'\[Train Step\] (\d+)/\d+: (.+?)(?= lr: 0\.0+,\s*$)'
    step_data = {}
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        match = re.search(pattern, line)
        if match:
            step = int(match.group(1))
            metrics_str = match.group(2)
            
            if step not in step_data:
                step_data[step] = {'step': step}
            
            for metric in metrics_str.split(', '):
                if ': ' in metric:
                    key, value = metric.split(': ')
                    try:
                        step_data[step][key] = float(value)
                    except ValueError:
                        continue
    
    df = pd.DataFrame(list(step_data.values()))
    df = df.sort_values('step').reset_index(drop=True)
    return df

def create_vae_comprehensive_visualizations(df, output_dir='radiodiff_Vae'):
    """Create comprehensive VAE training visualizations"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating comprehensive VAE visualizations with {len(df)} data points...")
    
    # 1. Main Loss Components Analysis
    print("Creating main loss components analysis...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('RadioDiff VAE Training - Loss Components Analysis', fontsize=16, fontweight='bold')
    
    # Total Loss
    ax1 = axes[0, 0]
    ax1.plot(df['step'], df['train/total_loss'], 'b-', linewidth=2, label='Total Loss')
    latest_total = df["train/total_loss"].iloc[-1]
    ax1.set_title(f'Total Loss Progression\n(Latest: {latest_total:.0f})', fontsize=12)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Total Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # KL Loss
    ax2 = axes[0, 1]
    ax2.plot(df['step'], df['train/kl_loss'], 'r-', linewidth=2, label='KL Loss')
    latest_kl = df["train/kl_loss"].iloc[-1]
    ax2.set_title(f'KL Loss Development\n(Latest: {latest_kl:.0f})', fontsize=12)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('KL Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Reconstruction Loss
    ax3 = axes[1, 0]
    ax3.plot(df['step'], df['train/rec_loss'], 'g-', linewidth=2, label='Reconstruction Loss')
    latest_rec = df["train/rec_loss"].iloc[-1]
    ax3.set_title(f'Reconstruction Loss Quality\n(Latest: {latest_rec:.3f})', fontsize=12)
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Reconstruction Loss')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # NLL Loss
    ax4 = axes[1, 1]
    ax4.plot(df['step'], df['train/nll_loss'], 'm-', linewidth=2, label='NLL Loss')
    latest_nll = df["train/nll_loss"].iloc[-1]
    ax4.set_title(f'Negative Log Likelihood Loss\n(Latest: {latest_nll:.0f})', fontsize=12)
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('NLL Loss')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vae_loss_components_comprehensive.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: vae_loss_components_comprehensive.png")
    
    # 2. Multi-axis Loss Analysis
    print("Creating multi-axis loss analysis...")
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Setup multiple y-axes
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    
    # Plot losses
    line1 = ax1.plot(df['step'], df['train/total_loss'], 'b-', linewidth=3, label='Total Loss')
    line2 = ax2.plot(df['step'], df['train/kl_loss'], 'r-', linewidth=3, label='KL Loss')
    line3 = ax3.plot(df['step'], df['train/rec_loss'], 'g-', linewidth=3, label='Reconstruction Loss')
    
    # Customize axes
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Total Loss', color='b', fontsize=12)
    ax2.set_ylabel('KL Loss', color='r', fontsize=12)
    ax3.set_ylabel('Reconstruction Loss', color='g', fontsize=12)
    
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax3.tick_params(axis='y', labelcolor='g')
    
    # Create legend
    lines = [line1[0], line2[0], line3[0]]
    labels = ['Total Loss', 'KL Loss', 'Reconstruction Loss']
    ax1.legend(lines, labels, loc='upper right', fontsize=12)
    
    ax1.set_title('VAE Multi-axis Loss Analysis - Scale Differences', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vae_multi_axis_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: vae_multi_axis_analysis.png")
    
    # 3. Normalized Loss Comparison
    print("Creating normalized loss comparison...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Normalize each metric to [0,1] range
    metrics_to_normalize = [
        ('train/total_loss', 'Total Loss', 'blue'),
        ('train/kl_loss', 'KL Loss', 'red'),
        ('train/rec_loss', 'Reconstruction Loss', 'green'),
        ('train/nll_loss', 'NLL Loss', 'magenta')
    ]
    
    for metric, title, color in metrics_to_normalize:
        if metric in df.columns:
            valid_data = df.dropna(subset=[metric])
            if len(valid_data) > 0:
                min_val = valid_data[metric].min()
                max_val = valid_data[metric].max()
                if max_val > min_val:
                    normalized = (valid_data[metric] - min_val) / (max_val - min_val)
                    ax.plot(valid_data['step'], normalized, color=color, linewidth=3, label=title)
                    print(f"  Normalized {title}: range [{min_val:.2f}, {max_val:.2f}]")
    
    ax.set_title('VAE Normalized Loss Comparison - Relative Trends', fontsize=16, fontweight='bold')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Normalized Loss [0,1]', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vae_normalized_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: vae_normalized_comparison.png")
    
    # 4. Training Progress Dashboard
    print("Creating training progress dashboard...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('RadioDiff VAE Training Progress Dashboard', fontsize=16, fontweight='bold')
    
    # Loss Evolution
    axes[0, 0].plot(df['step'], df['train/total_loss'], 'b-', linewidth=2)
    axes[0, 0].set_title('Total Loss Evolution')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # KL Divergence Progress
    axes[0, 1].plot(df['step'], df['train/kl_loss'], 'r-', linewidth=2)
    axes[0, 1].set_title('KL Divergence Progress')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('KL Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Reconstruction Quality
    axes[0, 2].plot(df['step'], df['train/rec_loss'], 'g-', linewidth=2)
    axes[0, 2].set_title('Reconstruction Quality')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('Rec Loss')
    axes[0, 2].grid(True, alpha=0.3)
    
    # NLL Loss Trend
    axes[1, 0].plot(df['step'], df['train/nll_loss'], 'm-', linewidth=2)
    axes[1, 0].set_title('NLL Loss Trend')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('NLL Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss Correlation Heatmap
    loss_data = df[['train/total_loss', 'train/kl_loss', 'train/rec_loss', 'train/nll_loss']].dropna()
    if len(loss_data) > 0:
        correlation_matrix = loss_data.corr()
        im = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 1].set_title('Loss Correlation Matrix')
        axes[1, 1].set_xticks(range(len(correlation_matrix.columns)))
        axes[1, 1].set_yticks(range(len(correlation_matrix.columns)))
        axes[1, 1].set_xticklabels([col.split('/')[-1] for col in correlation_matrix.columns], rotation=45)
        axes[1, 1].set_yticklabels([col.split('/')[-1] for col in correlation_matrix.columns])
        
        # Add correlation values
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                text = axes[1, 1].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=axes[1, 1])
    
    # Training Statistics
    axes[1, 2].axis('off')
    stats_text = f"""
    Training Statistics:
    
    Total Steps: {len(df)}
    Progress: {(df['step'].iloc[-1] / 150000) * 100:.1f}%
    Current Step: {df['step'].iloc[-1]:,}
    
    Latest Metrics:
    • Total Loss: {df['train/total_loss'].iloc[-1]:.0f}
    • KL Loss: {df['train/kl_loss'].iloc[-1]:.0f}
    • Rec Loss: {df['train/rec_loss'].iloc[-1]:.3f}
    • NLL Loss: {df['train/nll_loss'].iloc[-1]:.0f}
    
    Loss Ranges:
    • Total: [{df['train/total_loss'].min():.0f}, {df['train/total_loss'].max():.0f}]
    • KL: [{df['train/kl_loss'].min():.0f}, {df['train/kl_loss'].max():.0f}]
    • Rec: [{df['train/rec_loss'].min():.3f}, {df['train/rec_loss'].max():.3f}]
    """
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vae_training_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: vae_training_dashboard.png")
    
    # 5. Loss Distribution Analysis
    print("Creating loss distribution analysis...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('VAE Loss Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Total Loss Distribution
    axes[0, 0].hist(df['train/total_loss'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Total Loss Distribution')
    axes[0, 0].set_xlabel('Total Loss')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # KL Loss Distribution
    axes[0, 1].hist(df['train/kl_loss'], bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0, 1].set_title('KL Loss Distribution')
    axes[0, 1].set_xlabel('KL Loss')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Reconstruction Loss Distribution
    axes[1, 0].hist(df['train/rec_loss'], bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].set_title('Reconstruction Loss Distribution')
    axes[1, 0].set_xlabel('Reconstruction Loss')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # NLL Loss Distribution
    axes[1, 1].hist(df['train/nll_loss'], bins=50, alpha=0.7, color='magenta', edgecolor='black')
    axes[1, 1].set_title('NLL Loss Distribution')
    axes[1, 1].set_xlabel('NLL Loss')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vae_loss_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: vae_loss_distribution.png")
    
    print(f"\nAll comprehensive VAE visualizations saved to {output_dir}/")

def main():
    log_file_path = '/home/cine/Documents/Github/RadioDiff/radiodiff_Vae/2025-08-15-20-41_.log'
    
    print("Parsing VAE training log...")
    df = parse_vae_log_file(log_file_path)
    
    if len(df) == 0:
        print("No training data found in log file.")
        return
    
    print(f"Found {len(df)} unique training steps")
    print(f"Step range: {df['step'].min()} to {df['step'].max()}")
    
    create_vae_comprehensive_visualizations(df)
    
    print("Comprehensive VAE visualization complete!")

if __name__ == "__main__":
    main()