#!/usr/bin/env python3
"""
Script to generate visualization figures for VAE training progress
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def generate_training_visualizations():
    """
    Generate comprehensive visualizations for VAE training progress
    """
    print("Loading training data...")
    
    # Load the parsed training data
    df = pd.read_csv('radiodiff_Vae/training_data_parsed.csv')
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'])
    
    print(f"Loaded {len(df)} training steps")
    
    # Create output directory
    output_dir = Path('radiodiff_Vae/training_visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # 1. Loss evolution over time
    plt.figure(figsize=(12, 8))
    
    # Plot main losses
    plt.subplot(2, 2, 1)
    plt.plot(df['step'], df['train/total_loss'], label='Total Loss', linewidth=2)
    plt.plot(df['step'], df['train/kl_loss']/1000, label='KL Loss (÷1000)', linewidth=2)
    plt.plot(df['step'], df['train/nll_loss'], label='NLL Loss', linewidth=2)
    plt.xlabel('Training Step')
    plt.ylabel('Loss Value')
    plt.title('Loss Evolution During Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot reconstruction loss (scaled up for visibility)
    plt.subplot(2, 2, 2)
    plt.plot(df['step'], df['train/rec_loss'] * 10000, label='Reconstruction Loss (×10000)', linewidth=2, color='red')
    plt.xlabel('Training Step')
    plt.ylabel('Loss Value (×10000)')
    plt.title('Reconstruction Loss Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot discriminator metrics
    plt.subplot(2, 2, 3)
    plt.plot(df['step'], df['train/g_loss'], label='Generator Loss', linewidth=2)
    plt.plot(df['step'], df['train/disc_factor'], label='Discriminator Factor', linewidth=2)
    plt.xlabel('Training Step')
    plt.ylabel('Value')
    plt.title('Discriminator Metrics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot discriminator weight
    plt.subplot(2, 2, 4)
    plt.plot(df['step'], df['train/d_weight'], label='Discriminator Weight', linewidth=2, color='orange')
    plt.xlabel('Training Step')
    plt.ylabel('Weight Value')
    plt.title('Discriminator Weight Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved loss evolution plot to: {output_dir / 'loss_evolution.png'}")
    
    # 2. Training timeline
    plt.figure(figsize=(15, 6))
    
    # Create time-based plots
    plt.subplot(1, 2, 1)
    plt.plot(df['datetime'], df['train/total_loss'], linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Total Loss')
    plt.title('Training Progress Over Time')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Calculate rolling average for smoother visualization
    window_size = 50
    rolling_loss = df['train/total_loss'].rolling(window=window_size).mean()
    plt.plot(df['step'], rolling_loss, label=f'Rolling Avg (window={window_size})', linewidth=2)
    plt.plot(df['step'], df['train/total_loss'], alpha=0.3, label='Raw Loss', linewidth=1)
    plt.xlabel('Training Step')
    plt.ylabel('Total Loss')
    plt.title('Smoothed Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training timeline plot to: {output_dir / 'training_timeline.png'}")
    
    # 3. Loss distribution analysis
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['train/total_loss'], bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(df['train/total_loss'].mean(), color='red', linestyle='--', label=f'Mean: {df["train/total_loss"].mean():.2f}')
    plt.axvline(df['train/total_loss'].median(), color='green', linestyle='--', label=f'Median: {df["train/total_loss"].median():.2f}')
    plt.xlabel('Total Loss')
    plt.ylabel('Frequency')
    plt.title('Total Loss Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Box plot of different loss components
    loss_data = [
        df['train/total_loss'],
        df['train/kl_loss']/1000,  # Scale down for better visualization
        df['train/nll_loss'],
        df['train/rec_loss']*10000  # Scale up for better visualization
    ]
    labels = ['Total Loss', 'KL Loss (÷1000)', 'NLL Loss', 'Rec Loss (×10000)']
    
    plt.boxplot(loss_data, labels=labels)
    plt.ylabel('Loss Value')
    plt.title('Loss Component Distribution')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved loss distribution plot to: {output_dir / 'loss_distribution.png'}")
    
    # 4. Training progress summary
    plt.figure(figsize=(10, 8))
    
    # Create a comprehensive summary plot
    plt.subplot(2, 1, 1)
    
    # Normalize losses for comparison
    total_loss_norm = (df['train/total_loss'] - df['train/total_loss'].min()) / (df['train/total_loss'].max() - df['train/total_loss'].min())
    kl_loss_norm = (df['train/kl_loss'] - df['train/kl_loss'].min()) / (df['train/kl_loss'].max() - df['train/kl_loss'].min())
    nll_loss_norm = (df['train/nll_loss'] - df['train/nll_loss'].min()) / (df['train/nll_loss'].max() - df['train/nll_loss'].min())
    
    plt.plot(df['step'], total_loss_norm, label='Total Loss (normalized)', linewidth=2)
    plt.plot(df['step'], kl_loss_norm, label='KL Loss (normalized)', linewidth=2)
    plt.plot(df['step'], nll_loss_norm, label='NLL Loss (normalized)', linewidth=2)
    plt.xlabel('Training Step')
    plt.ylabel('Normalized Loss Value')
    plt.title('Normalized Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    
    # Training progress milestones
    milestones = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 104800]
    milestone_losses = []
    
    for milestone in milestones:
        if milestone in df['step'].values:
            loss_at_milestone = df[df['step'] == milestone]['train/total_loss'].iloc[0]
            milestone_losses.append(loss_at_milestone)
        else:
            # Find closest step
            closest_idx = (df['step'] - milestone).abs().idxmin()
            loss_at_milestone = df.loc[closest_idx, 'train/total_loss']
            milestone_losses.append(loss_at_milestone)
    
    plt.bar(range(len(milestones)), milestone_losses, alpha=0.7, color='skyblue', edgecolor='navy')
    plt.xticks(range(len(milestones)), [f'{m//1000}k' for m in milestones], rotation=45)
    plt.ylabel('Total Loss')
    plt.title('Training Loss at Key Milestones')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training summary plot to: {output_dir / 'training_summary.png'}")
    
    # 5. Individual loss components detailed view
    plt.figure(figsize=(16, 12))
    
    # Total Loss
    plt.subplot(3, 2, 1)
    plt.plot(df['step'], df['train/total_loss'], linewidth=2)
    plt.xlabel('Training Step')
    plt.ylabel('Total Loss')
    plt.title('Total Loss Evolution')
    plt.grid(True, alpha=0.3)
    
    # KL Loss
    plt.subplot(3, 2, 2)
    plt.plot(df['step'], df['train/kl_loss'], linewidth=2, color='red')
    plt.xlabel('Training Step')
    plt.ylabel('KL Loss')
    plt.title('KL Loss Evolution')
    plt.grid(True, alpha=0.3)
    
    # NLL Loss
    plt.subplot(3, 2, 3)
    plt.plot(df['step'], df['train/nll_loss'], linewidth=2, color='green')
    plt.xlabel('Training Step')
    plt.ylabel('NLL Loss')
    plt.title('NLL Loss Evolution')
    plt.grid(True, alpha=0.3)
    
    # Reconstruction Loss
    plt.subplot(3, 2, 4)
    plt.plot(df['step'], df['train/rec_loss'], linewidth=2, color='purple')
    plt.xlabel('Training Step')
    plt.ylabel('Reconstruction Loss')
    plt.title('Reconstruction Loss Evolution')
    plt.grid(True, alpha=0.3)
    
    # Generator Loss
    plt.subplot(3, 2, 5)
    plt.plot(df['step'], df['train/g_loss'], linewidth=2, color='orange')
    plt.xlabel('Training Step')
    plt.ylabel('Generator Loss')
    plt.title('Generator Loss Evolution')
    plt.grid(True, alpha=0.3)
    
    # Learning Rate (if available)
    plt.subplot(3, 2, 6)
    if 'lr' in df.columns and df['lr'].notna().any():
        plt.plot(df['step'], df['lr'], linewidth=2, color='brown')
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Evolution')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Learning Rate Data\nNot Available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Learning Rate Evolution')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'individual_losses.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved individual losses plot to: {output_dir / 'individual_losses.png'}")
    
    print(f"\nAll visualizations saved to: {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    output_dir = generate_training_visualizations()
    print(f"\nVisualization generation complete!")