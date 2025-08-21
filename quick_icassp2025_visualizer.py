#!/usr/bin/env python3
"""
ICASSP2025 Dataset Quick Visualizer

This script quickly visualizes the existing dataset without FSPL calculation:
- Shows original input images (3 channels)
- Shows output images (path loss)
- Shows multiple examples

Author: Claude Code Assistant
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from pathlib import Path
import os

def visualize_existing_examples(data_root, num_examples=5):
    """Quick visualization of existing dataset examples"""
    
    data_root = Path(data_root)
    
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
    
    print(f"Found {len(samples)} samples to visualize")
    
    # Create visualization
    fig, axes = plt.subplots(len(samples), 4, figsize=(20, 5*len(samples)))
    
    if len(samples) == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(samples):
        print(f"Processing sample {i+1}: {sample['stem']}")
        
        try:
            # Load images
            input_image = np.array(Image.open(sample['input_path']))
            output_image = np.array(Image.open(sample['output_path']))
            
            # Extract input channels
            reflectance = input_image[:, :, 0]
            transmittance = input_image[:, :, 1]
            distance = input_image[:, :, 2]
            
            # Find Tx position from distance channel
            tx_y, tx_x = np.unravel_index(np.argmin(distance), distance.shape)
            
            # Parse filename for info
            parts = sample['stem'].split('_')
            building_id = int(parts[0][1:])
            antenna_id = int(parts[1][3:])
            freq_id = int(parts[2][1:])
            sample_id = int(parts[3][1:])
            
            # Plot input channels
            axes[i, 0].imshow(reflectance, cmap='viridis')
            axes[i, 0].plot(tx_x, tx_y, 'r*', markersize=15)
            axes[i, 0].set_title(f'Reflectance (B{building_id})')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(transmittance, cmap='viridis')
            axes[i, 1].plot(tx_x, tx_y, 'r*', markersize=15)
            axes[i, 1].set_title(f'Transmittance (Ant{antenna_id})')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(distance, cmap='viridis')
            axes[i, 2].plot(tx_x, tx_y, 'r*', markersize=15)
            axes[i, 2].set_title(f'Distance (f{freq_id})')
            axes[i, 2].axis('off')
            
            # Plot output (ground truth path loss)
            axes[i, 3].imshow(output_image, cmap='jet')
            axes[i, 3].plot(tx_x, tx_y, 'r*', markersize=15)
            axes[i, 3].set_title(f'Path Loss Output (S{sample_id})')
            axes[i, 3].axis('off')
            
            # Print some statistics
            print(f"  Image size: {input_image.shape}")
            print(f"  Tx position: ({tx_x}, {tx_y})")
            print(f"  Distance range: {distance.min():.1f} - {distance.max():.1f}")
            print(f"  Path loss range: {output_image.min():.1f} - {output_image.max():.1f}")
            
        except Exception as e:
            print(f"  Error processing sample: {e}")
            # Show error message in plots
            for j in range(4):
                axes[i, j].text(0.5, 0.5, 'Error', ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].set_title(f'Channel {j}')
                axes[i, j].axis('off')
    
    plt.suptitle('ICASSP2025 Dataset Examples\n(Input: Reflectance + Transmittance + Distance → Output: Path Loss)', fontsize=16)
    plt.tight_layout()
    
    # Save the visualization
    output_path = './icassp2025_existing_examples.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to: {output_path}")
    return samples

def create_dataset_summary(data_root):
    """Create a summary of the dataset"""
    
    data_root = Path(data_root)
    
    # Count files
    input_dir = data_root / 'Inputs/Task_1_ICASSP'
    output_dir = data_root / 'Outputs/Task_3_ICASSP'
    
    input_files = list(input_dir.glob('*.png'))
    output_files = list(output_dir.glob('*.png'))
    
    print(f"Dataset Summary:")
    print(f"Input files: {len(input_files)}")
    print(f"Output files: {len(output_files)}")
    
    # Analyze building distribution
    building_counts = {}
    freq_counts = {}
    
    for file_path in input_files:
        parts = file_path.stem.split('_')
        building_id = parts[0][1:]
        freq_id = parts[2][1:]
        
        building_counts[building_id] = building_counts.get(building_id, 0) + 1
        freq_counts[freq_id] = freq_counts.get(freq_id, 0) + 1
    
    print(f"Buildings: {sorted(building_counts.keys())}")
    print(f"Frequencies: {sorted(freq_counts.keys())}")
    
    # Check file matching
    output_stems = {f.stem for f in output_files}
    input_stems = {f.stem for f in input_files}
    
    matched = input_stems & output_stems
    unmatched_input = input_stems - output_stems
    unmatched_output = output_stems - input_stems
    
    print(f"Matched pairs: {len(matched)}")
    print(f"Unmatched input files: {len(unmatched_input)}")
    print(f"Unmatched output files: {len(unmatched_output)}")
    
    return {
        'input_files': len(input_files),
        'output_files': len(output_files),
        'matched_pairs': len(matched),
        'buildings': sorted(building_counts.keys()),
        'frequencies': sorted(freq_counts.keys())
    }

def main():
    """Main function"""
    
    # Configuration
    data_root = '/home/cine/Documents/Github/RadioDiff/datasets/ICASSP2025_Dataset'
    
    print("=== ICASSP2025 Dataset Quick Visualization ===")
    print(f"Data root: {data_root}")
    
    # Create dataset summary
    print("\n1. Dataset Summary:")
    summary = create_dataset_summary(data_root)
    
    # 2. Visualize examples
    print("\n2. Visualizing examples...")
    samples = visualize_existing_examples(data_root, num_examples=5)
    
    print("\n=== Visualization completed! ===")
    print(f"Generated visualization: ./icassp2025_existing_examples.png")
    print(f"Total samples shown: {len(samples)}")

if __name__ == "__main__":
    main()