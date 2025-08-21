#!/usr/bin/env python3
"""
Simple ICASSP2025 Dataset Examples Visualizer

This script creates simple visualizations showing examples of:
- Original input images with 3 channels
- Output images (path loss)
- Multiple examples with different scenarios

Author: Claude Code Assistant
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from pathlib import Path
import os
from tqdm.auto import tqdm

# Import the path loss calculator
import sys
sys.path.append('.')
from newdata_convert import PathLossCalculator

def visualize_dataset_examples(data_root, num_examples=5):
    """Create simple visualization of dataset examples"""
    
    data_root = Path(data_root)
    calculator = PathLossCalculator()
    
    # Get sample files
    input_dir = data_root / 'Inputs/Task_1_ICASSP'
    output_dir = data_root / 'Outputs/Task_3_ICASSP'
    
    # Find matching input-output pairs
    input_files = list(input_dir.glob('*.png'))
    output_files = {f.stem: f for f in output_dir.glob('*.png')}
    
    samples = []
    for input_file in input_files[:num_examples*2]:  # Get more files to ensure we have matches
        if input_file.stem in output_files:
            samples.append({
                'input_path': input_file,
                'output_path': output_files[input_file.stem],
                'stem': input_file.stem
            })
        if len(samples) >= num_examples:
            break
    
    print(f"Found {len(samples)} samples to visualize")
    
    # Create visualization
    fig, axes = plt.subplots(len(samples), 4, figsize=(20, 5*len(samples)))
    
    if len(samples) == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(samples):
        print(f"Processing sample {i+1}: {sample['stem']}")
        
        try:
            # Parse filename
            parts = sample['stem'].split('_')
            building_id = int(parts[0][1:])
            antenna_id = int(parts[1][3:])
            freq_id = int(parts[2][1:])
            sample_id = int(parts[3][1:])
            
            # Load images
            input_image = np.array(Image.open(sample['input_path']))
            output_image = np.array(Image.open(sample['output_path']))
            
            # Calculate FSPL
            fspl_map, _ = calculator.create_path_loss_map(
                building_id, antenna_id, freq_id, sample_id
            )
            
            # Extract channels
            reflectance = input_image[:, :, 0]
            transmittance = input_image[:, :, 1]
            distance = input_image[:, :, 2]
            
            # Find Tx position
            tx_y, tx_x = np.unravel_index(np.argmin(distance), distance.shape)
            
            # Plot original input channels
            axes[i, 0].imshow(reflectance, cmap='viridis')
            axes[i, 0].plot(tx_x, tx_y, 'r*', markersize=15)
            axes[i, 0].set_title(f'Reflectance (B{building_id})')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(transmittance, cmap='viridis')
            axes[i, 1].plot(tx_x, tx_y, 'r*', markersize=15)
            axes[i, 1].set_title(f'Transmittance (Ant{antenna_id})')
            axes[i, 1].axis('off')
            
            # Plot FSPL and ground truth
            vmin = min(np.min(output_image), np.min(fspl_map))
            vmax = max(np.max(output_image), np.max(fspl_map))
            
            axes[i, 2].imshow(fspl_map, cmap='jet', vmin=vmin, vmax=vmax)
            axes[i, 2].plot(tx_x, tx_y, 'r*', markersize=15)
            axes[i, 2].set_title(f'Calculated FSPL (f{freq_id})')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(output_image, cmap='jet', vmin=vmin, vmax=vmax)
            axes[i, 3].plot(tx_x, tx_y, 'r*', markersize=15)
            axes[i, 3].set_title(f'Ground Truth (S{sample_id})')
            axes[i, 3].axis('off')
            
            # Calculate and print error metrics
            error_metrics = calculator.analyze_error(fspl_map, output_image)
            print(f"  RMSE: {error_metrics['RMSE']:.2f} dB, MAE: {error_metrics['MAE']:.2f} dB")
            
        except Exception as e:
            print(f"  Error processing sample: {e}")
            # Show error message in plots
            for j in range(4):
                axes[i, j].text(0.5, 0.5, 'Error', ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].set_title(f'Channel {j}')
                axes[i, j].axis('off')
    
    plt.suptitle('ICASSP2025 Dataset Examples\n(Original Input → Calculated FSPL → Ground Truth)', fontsize=16)
    plt.tight_layout()
    
    # Save the visualization
    output_path = './icassp2025_dataset_examples.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to: {output_path}")
    return samples

def create_three_channel_examples(data_root, num_examples=3):
    """Create examples showing three-channel input images"""
    
    data_root = Path(data_root)
    calculator = PathLossCalculator()
    
    # Get sample files
    input_dir = data_root / 'Inputs/Task_1_ICASSP'
    output_dir = data_root / 'Outputs/Task_3_ICASSP'
    
    # Find matching input-output pairs
    input_files = list(input_dir.glob('*.png'))
    output_files = {f.stem: f for f in output_dir.glob('*.png')}
    
    samples = []
    for input_file in input_files:
        if input_file.stem in output_files:
            samples.append({
                'input_path': input_file,
                'output_path': output_files[input_file.stem],
                'stem': input_file.stem
            })
        if len(samples) >= num_examples:
            break
    
    print(f"Creating three-channel examples for {len(samples)} samples")
    
    # Create visualization
    fig, axes = plt.subplots(len(samples), 3, figsize=(15, 5*len(samples)))
    
    if len(samples) == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(samples):
        try:
            # Parse filename
            parts = sample['stem'].split('_')
            building_id = int(parts[0][1:])
            antenna_id = int(parts[1][3:])
            freq_id = int(parts[2][1:])
            sample_id = int(parts[3][1:])
            
            # Load images
            input_image = np.array(Image.open(sample['input_path']))
            output_image = np.array(Image.open(sample['output_path']))
            
            # Calculate FSPL
            fspl_map, _ = calculator.create_path_loss_map(
                building_id, antenna_id, freq_id, sample_id
            )
            
            # Extract channels
            reflectance = input_image[:, :, 0]
            transmittance = input_image[:, :, 1]
            distance = input_image[:, :, 2]
            
            # Find Tx position
            tx_y, tx_x = np.unravel_index(np.argmin(distance), distance.shape)
            
            # Normalize FSPL to 0-255 range for visualization
            fspl_normalized = ((fspl_map - fspl_map.min()) / (fspl_map.max() - fspl_map.min()) * 255).astype(np.uint8)
            
            # Create three-channel input (reflectance, transmittance, FSPL)
            three_channel_input = np.stack([reflectance, transmittance, fspl_normalized], axis=-1)
            
            # Plot the three channels and combined image
            axes[i, 0].imshow(reflectance, cmap='viridis')
            axes[i, 0].plot(tx_x, tx_y, 'r*', markersize=15)
            axes[i, 0].set_title('Channel 0: Reflectance')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(transmittance, cmap='viridis')
            axes[i, 1].plot(tx_x, tx_y, 'r*', markersize=15)
            axes[i, 1].set_title('Channel 1: Transmittance')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(three_channel_input)
            axes[i, 2].plot(tx_x, tx_y, 'r*', markersize=15)
            axes[i, 2].set_title('3-Channel Input\n(Ref+Trans+FSPL)')
            axes[i, 2].axis('off')
            
        except Exception as e:
            print(f"Error processing sample {sample['stem']}: {e}")
            for j in range(3):
                axes[i, j].text(0.5, 0.5, 'Error', ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].axis('off')
    
    plt.suptitle('Three-Channel Input Examples for VAE Training', fontsize=16)
    plt.tight_layout()
    
    # Save the visualization
    output_path = './icassp2025_three_channel_examples.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Three-channel examples saved to: {output_path}")

def main():
    """Main function"""
    
    # Configuration
    data_root = '/home/cine/Documents/Github/RadioDiff/datasets/ICASSP2025_Dataset'
    
    print("=== ICASSP2025 Dataset Examples Visualization ===")
    print(f"Data root: {data_root}")
    
    # Create output directory
    output_dir = Path('./icassp2025_examples')
    output_dir.mkdir(exist_ok=True)
    
    # 1. Basic dataset examples
    print("\n1. Creating basic dataset examples...")
    samples = visualize_dataset_examples(data_root, num_examples=5)
    
    # 2. Three-channel examples
    print("\n2. Creating three-channel examples...")
    create_three_channel_examples(data_root, num_examples=3)
    
    print("\n=== Visualization completed! ===")
    print("Check the generated PNG files in the current directory.")

if __name__ == "__main__":
    main()