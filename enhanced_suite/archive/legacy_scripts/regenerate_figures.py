#!/usr/bin/env python3
"""
Regenerate figures with latest RadioDiff VAE training data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from improved_visualization_final import parse_log_file_fixed, create_improved_visualizations

def regenerate_figures():
    """Regenerate all figures with the latest log data"""
    
    log_file = '/home/cine/Documents/Github/RadioDiff/radiodiff_Vae/2025-08-15-20-41_.log'
    output_dir = '/home/cine/Documents/Github/RadioDiff/radiodiff_Vae'
    
    print("=== Regenerating RadioDiff VAE Training Figures ===")
    print(f"Log file: {log_file}")
    print(f"Output directory: {output_dir}")
    
    # Parse the log file
    print("\n1. Parsing log file...")
    df = parse_log_file_fixed(log_file)
    
    if df is None or len(df) == 0:
        print("Error: No data found in log file")
        return False
    
    print(f"Found {len(df)} training steps")
    print(f"Step range: {df['step'].min()} to {df['step'].max()}")
    
    # Check for required columns
    required_columns = ['train/total_loss', 'train/kl_loss', 'train/rec_loss']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return False
    
    # Get current metrics
    latest_step = df['step'].max()
    latest_data = df[df['step'] == latest_step].iloc[0]
    
    print(f"\n2. Current training metrics:")
    print(f"   Step: {latest_step:,}")
    print(f"   Total Loss: {latest_data['train/total_loss']:,.2f}")
    print(f"   KL Loss: {latest_data['train/kl_loss']:,.2f}")
    print(f"   Reconstruction Loss: {latest_data['train/rec_loss']:.4f}")
    
    # Check discriminator status
    if 'train/disc_factor' in df.columns:
        disc_factors = df['train/disc_factor'].unique()
        print(f"   Discriminator Factors: {disc_factors}")
        active_discriminator = any(df > 0 for df in disc_factors)
        print(f"   Discriminator Active: {'Yes' if active_discriminator else 'No'}")
    
    # Generate improved visualizations
    print(f"\n3. Generating improved visualizations...")
    try:
        create_improved_visualizations(df, output_dir)
        print("   ✓ Improved visualizations created successfully")
    except Exception as e:
        print(f"   ✗ Error creating improved visualizations: {e}")
        return False
    
    # Also create individual loss plots
    print(f"\n4. Creating individual loss plots...")
    try:
        create_individual_plots(df, output_dir)
        print("   ✓ Individual plots created successfully")
    except Exception as e:
        print(f"   ✗ Error creating individual plots: {e}")
        return False
    
    print(f"\n5. Figure regeneration completed successfully!")
    print(f"   All figures saved to: {output_dir}")
    
    return True

def create_individual_plots(df, output_dir):
    """Create individual loss plots for detailed analysis"""
    
    # Set up the figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('RadioDiff VAE Training Progress - Individual Loss Analysis', fontsize=16, fontweight='bold')
    
    # Total Loss
    ax1 = axes[0, 0]
    ax1.plot(df['step'], df['train/total_loss'], 'b-', linewidth=2, label='Total Loss')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Loss Progression')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # KL Loss
    ax2 = axes[0, 1]
    ax2.plot(df['step'], df['train/kl_loss'], 'r-', linewidth=2, label='KL Loss')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('KL Loss')
    ax2.set_title('KL Loss Development')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Reconstruction Loss
    ax3 = axes[1, 0]
    ax3.plot(df['step'], df['train/rec_loss'], 'g-', linewidth=2, label='Reconstruction Loss')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Reconstruction Loss')
    ax3.set_title('Reconstruction Loss Quality')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Discriminator Loss (if available)
    ax4 = axes[1, 1]
    if 'train/disc_loss' in df.columns:
        ax4.plot(df['step'], df['train/disc_loss'], 'm-', linewidth=2, label='Discriminator Loss')
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Discriminator Loss')
        ax4.set_title('Discriminator Loss Status')
    elif 'train/disc_factor' in df.columns:
        # Show disc_factor instead
        ax4.plot(df['step'], df['train/disc_factor'], 'm-', linewidth=2, label='Discriminator Factor')
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Discriminator Factor')
        ax4.set_title('Discriminator Activation Status')
    else:
        ax4.text(0.5, 0.5, 'Discriminator data\nnot available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Discriminator Status')
    
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'individual_losses_detailed.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create discriminator activation plot
    if 'train/disc_factor' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color code by discriminator status
        disc_active = df['train/disc_factor'] > 0
        disc_inactive = ~disc_active
        
        if disc_inactive.any():
            ax.scatter(df[disc_inactive]['step'], df[disc_inactive]['train/total_loss'], 
                      c='blue', alpha=0.6, s=20, label='VAE Pre-training')
        
        if disc_active.any():
            ax.scatter(df[disc_active]['step'], df[disc_active]['train/total_loss'], 
                      c='red', alpha=0.6, s=20, label='VAE-GAN Training')
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Total Loss')
        ax.set_title('Training Phases: VAE Pre-training vs VAE-GAN Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add vertical line at discriminator activation
        if disc_active.any():
            first_active = df[disc_active]['step'].min()
            ax.axvline(x=first_active, color='red', linestyle='--', alpha=0.7, 
                      label=f'Discriminator Activation (Step {first_active:,})')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_phases_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    success = regenerate_figures()
    if success:
        print("\n✅ All figures regenerated successfully!")
    else:
        print("\n❌ Figure regeneration failed!")
        exit(1)