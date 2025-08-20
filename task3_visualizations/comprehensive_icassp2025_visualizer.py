#!/usr/bin/env python3
"""
ICASSP2025 Dataset Examples Visualization

This script creates comprehensive visualizations of the ICASSP2025 dataset showing:
1. Original input images (reflectance, transmittance, distance)
2. Output images (path loss)
3. Multiple examples with different scenarios
4. Dataset statistics

Author: Claude Code Assistant
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from pathlib import Path
import os
from tqdm.auto import tqdm

def create_comprehensive_visualization(data_root, output_dir='./icassp2025_visualizations'):
    """Create comprehensive visualization of ICASSP2025 dataset"""
    
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get sample files
    input_dir = data_root / 'Inputs/Task_1_ICASSP'
    output_dir_data = data_root / 'Outputs/Task_3_ICASSP'
    
    input_files = list(input_dir.glob('*.png'))
    output_files = {f.stem: f for f in output_dir_data.glob('*.png')}
    
    print(f"Input files: {len(input_files)}")
    print(f"Output files: {len(output_files)}")
    
    # Create matched samples
    samples = []
    for input_file in input_files:
        if input_file.stem in output_files:
            samples.append({
                'input_path': input_file,
                'output_path': output_files[input_file.stem],
                'stem': input_file.stem
            })
    
    print(f"Matched samples: {len(samples)}")
    
    # 1. Create detailed example visualization
    print("Creating detailed example visualization...")
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Show 3 different examples
    for i in range(3):
        if i < len(samples):
            sample = samples[i]
            
            try:
                # Load images
                input_image = np.array(Image.open(sample['input_path']))
                output_image = np.array(Image.open(sample['output_path']))
                
                # Extract channels
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
                
                # Plot input channels
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
                
                # Plot output
                axes[i, 3].imshow(output_image, cmap='jet')
                axes[i, 3].plot(tx_x, tx_y, 'r*', markersize=10)
                axes[i, 3].set_title(f'Path Loss (S{sample_id})')
                axes[i, 3].axis('off')
                
                print(f"Sample {i+1}: {sample['stem']}")
                print(f"  Size: {input_image.shape}, Tx: ({tx_x}, {tx_y})")
                print(f"  Distance range: {distance.min():.1f}-{distance.max():.1f}")
                print(f"  Path loss range: {output_image.min():.1f}-{output_image.max():.1f}")
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                for j in range(4):
                    axes[i, j].text(0.5, 0.5, 'Error', ha='center', va='center', transform=axes[i, j].transAxes)
                    axes[i, j].axis('off')
    
    plt.suptitle('ICASSP2025 Dataset Examples\n(Reflectance + Transmittance + Distance → Path Loss)', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_examples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2. Create dataset statistics
    print("Creating dataset statistics...")
    
    # Analyze building and frequency distribution
    building_counts = {}
    freq_counts = {}
    sample_sizes = []
    
    for sample in tqdm(samples[:100]):  # Sample first 100 for speed
        try:
            input_image = np.array(Image.open(sample['input_path']))
            sample_sizes.append(input_image.shape)
            
            parts = sample['stem'].split('_')
            building_id = parts[0][1:]
            freq_id = parts[2][1:]
            
            building_counts[building_id] = building_counts.get(building_id, 0) + 1
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
    
    # Frequency distribution
    freq_labels = ['868 MHz', '1.8 GHz', '3.5 GHz']
    freq_values = [freq_counts.get('1', 0), freq_counts.get('2', 0), freq_counts.get('3', 0)]
    colors = ['lightcoral', 'lightgreen', 'gold']
    ax2.bar(freq_labels, freq_values, color=colors)
    ax2.set_title('Frequency Distribution')
    ax2.set_ylabel('Sample Count')
    
    # Sample size distribution
    if sample_sizes:
        heights = [s[0] for s in sample_sizes]
        widths = [s[1] for s in sample_sizes]
        
        ax3.scatter(widths, heights, alpha=0.6)
        ax3.set_title('Image Size Distribution')
        ax3.set_xlabel('Width')
        ax3.set_ylabel('Height')
        ax3.grid(True, alpha=0.3)
    
    # Dataset summary
    ax4.axis('off')
    summary_text = (
        f"ICASSP2025 Dataset Summary\n"
        f"{'='*40}\n"
        f"Total Input Files: {len(input_files)}\n"
        f"Total Output Files: {len(output_files)}\n"
        f"Matched Pairs: {len(samples)}\n"
        f"Buildings: {len(building_counts)}\n"
        f"Frequencies: {len(freq_counts)}\n"
        f"Samples per Building: ~{len(samples)//len(building_counts) if building_counts else 0}\n"
        f"\nImage Characteristics:\n"
        f"Input: 3-channel (Reflectance, Transmittance, Distance)\n"
        f"Output: 1-channel (Path Loss)\n"
        f"Typical Size: {sample_sizes[0] if sample_sizes else 'Unknown'}\n"
        f"\nData Format:\n"
        f"Naming: B{{building}}_Ant{{antenna}}_f{{freq}}_S{{sample}}.png\n"
        f"Tx Position: Minimum value in distance channel\n"
        f"Value Range: 0-255 (uint8)"
    )
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('ICASSP2025 Dataset Statistics', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_statistics.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 3. Create three-channel input preview
    print("Creating three-channel input preview...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i in range(2):
        if i < len(samples):
            sample = samples[i]
            
            try:
                # Load images
                input_image = np.array(Image.open(sample['input_path']))
                output_image = np.array(Image.open(sample['output_path']))
                
                # Extract channels
                reflectance = input_image[:, :, 0]
                transmittance = input_image[:, :, 1]
                distance = input_image[:, :, 2]
                
                # Find Tx position
                tx_y, tx_x = np.unravel_index(np.argmin(distance), distance.shape)
                
                # Create three-channel preview (replace distance with normalized version for visualization)
                distance_norm = ((distance - distance.min()) / (distance.max() - distance.min()) * 255).astype(np.uint8)
                three_channel = np.stack([reflectance, transmittance, distance_norm], axis=-1)
                
                # Plot channels
                axes[i, 0].imshow(reflectance, cmap='viridis')
                axes[i, 0].plot(tx_x, tx_y, 'r*', markersize=10)
                axes[i, 0].set_title('Channel 0: Reflectance')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(transmittance, cmap='viridis')
                axes[i, 1].plot(tx_x, tx_y, 'r*', markersize=10)
                axes[i, 1].set_title('Channel 1: Transmittance')
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(three_channel)
                axes[i, 2].plot(tx_x, tx_y, 'r*', markersize=10)
                axes[i, 2].set_title('3-Channel Input Preview')
                axes[i, 2].axis('off')
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                for j in range(3):
                    axes[i, j].text(0.5, 0.5, 'Error', ha='center', va='center', transform=axes[i, j].transAxes)
                    axes[i, j].axis('off')
    
    plt.suptitle('Three-Channel Input Preview for VAE Training', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'three_channel_preview.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"All visualizations saved to: {output_dir}")
    return {
        'total_samples': len(samples),
        'buildings': len(building_counts),
        'frequencies': len(freq_counts),
        'output_dir': output_dir
    }

def main():
    """Main function"""
    
    # Configuration
    data_root = '/home/cine/Documents/Github/RadioDiff/datasets/ICASSP2025_Dataset'
    
    print("=== ICASSP2025 Dataset Visualization ===")
    print(f"Data root: {data_root}")
    
    # Create comprehensive visualization
    results = create_comprehensive_visualization(data_root)
    
    print("\n=== Visualization completed! ===")
    print(f"Total samples processed: {results['total_samples']}")
    print(f"Buildings represented: {results['buildings']}")
    print(f"Frequencies represented: {results['frequencies']}")
    print(f"Output directory: {results['output_dir']}")
    
    print("\nGenerated files:")
    print("- dataset_examples.png: Shows 3 examples with all channels")
    print("- dataset_statistics.png: Dataset statistics and distribution")
    print("- three_channel_preview.png: Three-channel input preview")

if __name__ == "__main__":
    main()