#!/usr/bin/env python3
"""
Task 3 ICASSP Path Loss Results Visualization

This script creates comprehensive visualizations of the Task 3 ICASSP dataset
showing path loss results that were created from Task 1 data using newdata_convert.py.

Author: Claude Code Assistant
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from pathlib import Path
import os
from tqdm.auto import tqdm
import seaborn as sns

def create_task3_visualization(data_root, output_dir='./task3_visualizations'):
    """Create comprehensive visualization of Task 3 ICASSP dataset"""
    
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get sample files from Task 3 path loss results
    input_dir = data_root / 'Inputs/Task_1_ICASSP'
    task3_dir = data_root / 'Inputs/Task_3_ICASSP_path_loss_results_replaced'
    
    input_files = list(input_dir.glob('*.png'))
    task3_files = list(task3_dir.glob('*.png'))
    
    print(f"Original input files: {len(input_files)}")
    print(f"Task 3 path loss files: {len(task3_files)}")
    
    # Create matched samples
    samples = []
    for task3_file in task3_files:
        input_file = input_dir / task3_file.name
        if input_file.exists():
            samples.append({
                'input_path': input_file,
                'task3_path': task3_file,
                'stem': task3_file.stem
            })
    
    print(f"Matched samples: {len(samples)}")
    
    # 1. Create detailed comparison visualization
    print("Creating detailed comparison visualization...")
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Show 3 different examples
    for i in range(3):
        if i < len(samples):
            sample = samples[i]
            
            try:
                # Load images
                input_image = np.array(Image.open(sample['input_path']))
                task3_image = np.array(Image.open(sample['task3_path']))
                
                # Extract channels from original input
                reflectance = input_image[:, :, 0]
                transmittance = input_image[:, :, 1]
                distance = input_image[:, :, 2]
                
                # Find Tx position
                tx_y, tx_x = np.unravel_index(np.argmin(distance), distance.shape)
                
                # Parse filename
                parts = sample['stem'].split('_')
                building_id = int(parts[0][1:])
                antenna_id = int(parts[1][3:])
                freq_id = int(parts[2][1:])
                sample_id = int(parts[3][1:])
                
                # Plot original input channels
                axes[i, 0].imshow(reflectance, cmap='viridis')
                axes[i, 0].plot(tx_x, tx_y, 'r*', markersize=10)
                axes[i, 0].set_title(f'Reflectance (B{building_id})')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(transmittance, cmap='viridis')
                axes[i, 1].plot(tx_x, tx_y, 'r*', markersize=10)
                axes[i, 1].set_title(f'Transmittance (Ant{antenna_id})')
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(distance, cmap='viridis')
                axes[i, 2].plot(tx_x, tx_y, 'r*', markersize=10)
                axes[i, 2].set_title(f'Distance (f{freq_id})')
                axes[i, 2].axis('off')
                
                # Plot Task 3 path loss result
                axes[i, 3].imshow(task3_image, cmap='jet')
                axes[i, 3].plot(tx_x, tx_y, 'r*', markersize=10)
                axes[i, 3].set_title(f'Path Loss Result (S{sample_id})')
                axes[i, 3].axis('off')
                
                print(f"Sample {i+1}: {sample['stem']}")
                print(f"  Input size: {input_image.shape}, Task 3 size: {task3_image.shape}")
                print(f"  Tx position: ({tx_x}, {tx_y})")
                print(f"  Distance range: {distance.min():.1f}-{distance.max():.1f}")
                print(f"  Path loss range: {task3_image.min():.1f}-{task3_image.max():.1f}")
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                for j in range(4):
                    axes[i, j].text(0.5, 0.5, 'Error', ha='center', va='center', transform=axes[i, j].transAxes)
                    axes[i, j].axis('off')
    
    plt.suptitle('Task 3 ICASSP Path Loss Results\n(Original Input → Generated Path Loss)', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'task3_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2. Create dataset statistics
    print("Creating dataset statistics...")
    
    # Analyze building, antenna, and frequency distribution
    building_counts = {}
    antenna_counts = {}
    freq_counts = {}
    sample_sizes = []
    path_loss_stats = []
    
    for sample in tqdm(samples[:200]):  # Sample first 200 for speed
        try:
            input_image = np.array(Image.open(sample['input_path']))
            task3_image = np.array(Image.open(sample['task3_path']))
            
            sample_sizes.append(input_image.shape)
            path_loss_stats.append({
                'min': task3_image.min(),
                'max': task3_image.max(),
                'mean': task3_image.mean(),
                'std': task3_image.std()
            })
            
            parts = sample['stem'].split('_')
            building_id = parts[0][1:]
            antenna_id = parts[1][3:]
            freq_id = parts[2][1:]
            
            building_counts[building_id] = building_counts.get(building_id, 0) + 1
            antenna_counts[antenna_id] = antenna_counts.get(antenna_id, 0) + 1
            freq_counts[freq_id] = freq_counts.get(freq_id, 0) + 1
            
        except Exception as e:
            continue
    
    # Create statistics visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Building distribution
    buildings = sorted(building_counts.keys())
    counts = [building_counts[b] for b in buildings]
    ax1.bar(buildings, counts, color='skyblue')
    ax1.set_title('Building Distribution')
    ax1.set_xlabel('Building ID')
    ax1.set_ylabel('Sample Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Antenna distribution
    antennas = sorted(antenna_counts.keys())
    ant_counts = [antenna_counts[a] for a in antennas]
    ax2.bar(antennas, ant_counts, color='lightcoral')
    ax2.set_title('Antenna Distribution')
    ax2.set_xlabel('Antenna ID')
    ax2.set_ylabel('Sample Count')
    
    # Frequency distribution
    freq_labels = ['868 MHz', '1.8 GHz', '3.5 GHz']
    freq_values = [freq_counts.get('1', 0), freq_counts.get('2', 0), freq_counts.get('3', 0)]
    colors = ['lightgreen', 'gold', 'orange']
    ax3.bar(freq_labels, freq_values, color=colors)
    ax3.set_title('Frequency Distribution')
    ax3.set_ylabel('Sample Count')
    
    # Path loss statistics
    if path_loss_stats:
        df_stats = pd.DataFrame(path_loss_stats)
        ax4.hist(df_stats['mean'], bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax4.set_title('Path Loss Mean Value Distribution')
        ax4.set_xlabel('Mean Path Loss Value')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Task 3 ICASSP Dataset Statistics', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'task3_statistics.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 3. Create path loss analysis visualization
    print("Creating path loss analysis visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Select diverse examples
    selected_samples = []
    if len(samples) >= 6:
        # Try to get different buildings and frequencies
        selected_samples = samples[:6]
    
    for i in range(2):
        for j in range(3):
            idx = i * 3 + j
            if idx < len(selected_samples):
                sample = selected_samples[idx]
                
                try:
                    # Load images
                    input_image = np.array(Image.open(sample['input_path']))
                    task3_image = np.array(Image.open(sample['task3_path']))
                    
                    # Extract distance channel
                    distance = input_image[:, :, 2]
                    
                    # Find Tx position
                    tx_y, tx_x = np.unravel_index(np.argmin(distance), distance.shape)
                    
                    # Create path loss heatmap with distance overlay
                    axes[i, j].imshow(task3_image, cmap='jet', alpha=0.8)
                    axes[i, j].contour(distance, levels=10, colors='white', alpha=0.3, linewidths=0.5)
                    axes[i, j].plot(tx_x, tx_y, 'r*', markersize=15)
                    
                    # Parse filename for title
                    parts = sample['stem'].split('_')
                    building_id = parts[0][1:]
                    antenna_id = parts[1][3:]
                    freq_id = parts[2][1:]
                    sample_id = parts[3][1:]
                    
                    freq_map = {'1': '868MHz', '2': '1.8GHz', '3': '3.5GHz'}
                    title = f'B{building_id}-Ant{antenna_id}\n{freq_map.get(freq_id, freq_id)}-S{sample_id}'
                    axes[i, j].set_title(title, fontsize=10)
                    axes[i, j].axis('off')
                    
                except Exception as e:
                    print(f"Error processing sample {idx}: {e}")
                    axes[i, j].text(0.5, 0.5, 'Error', ha='center', va='center', transform=axes[i, j].transAxes)
                    axes[i, j].axis('off')
    
    plt.suptitle('Task 3 Path Loss Analysis\n(Path Loss with Distance Contours)', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'task3_pathloss_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 4. Create detailed statistics summary
    print("Creating detailed statistics summary...")
    
    if path_loss_stats:
        df_stats = pd.DataFrame(path_loss_stats)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Min values distribution
        ax1.hist(df_stats['min'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title('Minimum Path Loss Values')
        ax1.set_xlabel('Min Value')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Max values distribution
        ax2.hist(df_stats['max'], bins=20, alpha=0.7, color='red', edgecolor='black')
        ax2.set_title('Maximum Path Loss Values')
        ax2.set_xlabel('Max Value')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Standard deviation distribution
        ax3.hist(df_stats['std'], bins=20, alpha=0.7, color='green', edgecolor='black')
        ax3.set_title('Path Loss Standard Deviation')
        ax3.set_xlabel('Std Dev')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Summary statistics text
        ax4.axis('off')
        summary_text = (
            f"Task 3 Dataset Summary\n"
            f"{'='*40}\n"
            f"Total Processed Samples: {len(path_loss_stats)}\n"
            f"Buildings: {len(building_counts)}\n"
            f"Antennas: {len(antenna_counts)}\n"
            f"Frequencies: {len(freq_counts)}\n"
            f"\nPath Loss Statistics:\n"
            f"Mean Range: {df_stats['mean'].min():.2f} - {df_stats['mean'].max():.2f}\n"
            f"Std Range: {df_stats['std'].min():.2f} - {df_stats['std'].max():.2f}\n"
            f"Average Mean: {df_stats['mean'].mean():.2f}\n"
            f"Average Std: {df_stats['std'].mean():.2f}\n"
            f"\nImage Characteristics:\n"
            f"Input: 3-channel (R, T, D)\n"
            f"Output: 1-channel (Path Loss)\n"
            f"Typical Size: {sample_sizes[0] if sample_sizes else 'Unknown'}\n"
            f"Value Range: 0-255 (uint8)"
        )
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Task 3 Detailed Statistics', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'task3_detailed_stats.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"All visualizations saved to: {output_dir}")
    return {
        'total_samples': len(samples),
        'buildings': len(building_counts),
        'antennas': len(antenna_counts),
        'frequencies': len(freq_counts),
        'output_dir': output_dir,
        'path_loss_stats': path_loss_stats
    }

def main():
    """Main function"""
    
    # Configuration
    data_root = '/home/cine/Documents/Github/RadioDiff/datasets/ICASSP2025_Dataset'
    
    print("=== Task 3 ICASSP Path Loss Results Visualization ===")
    print(f"Data root: {data_root}")
    
    # Create comprehensive visualization
    results = create_task3_visualization(data_root)
    
    print("\n=== Visualization completed! ===")
    print(f"Total samples processed: {results['total_samples']}")
    print(f"Buildings represented: {results['buildings']}")
    print(f"Antennas represented: {results['antennas']}")
    print(f"Frequencies represented: {results['frequencies']}")
    print(f"Output directory: {results['output_dir']}")
    
    print("\nGenerated files:")
    print("- task3_comparison.png: Shows 3 examples with original input and generated path loss")
    print("- task3_statistics.png: Dataset statistics and distribution")
    print("- task3_pathloss_analysis.png: Path loss analysis with distance contours")
    print("- task3_detailed_stats.png: Detailed path loss statistics")

if __name__ == "__main__":
    main()