#!/usr/bin/env python3
"""
ICASSP2025 3-Channel Dataset Visualization

This script creates comprehensive visualizations of the ICASSP2025 dataset with FSPL channel:
1. Original input images (reflectance, transmittance)
2. FSPL path loss channel (replacing distance)
3. Comparison with ground truth outputs
4. Multiple examples with different scenarios
5. Dataset statistics

Author: Claude Code Assistant
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from pathlib import Path
import os
from tqdm.auto import tqdm
import cv2

def create_comprehensive_visualization(data_root, output_dir='./task3_visualizations'):
    """Create comprehensive visualization of ICASSP2025 dataset with FSPL"""
    
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get sample files
    input_dir = data_root / 'Inputs/Task_3_ICASSP_path_loss_results_replaced'
    output_dir_data = data_root / 'Outputs/Task_3_ICASSP'
    
    input_files = list(input_dir.glob('*.png'))
    output_files = {f.stem: f for f in output_dir_data.glob('*.png')}
    
    print(f"Input files (with FSPL): {len(input_files)}")
    print(f"Output files (ground truth): {len(output_files)}")
    
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
                fspl_path_loss = input_image[:, :, 2]  # FSPL calculated path loss
                
                # Find Tx position (from minimum distance values in original dataset)
                # For FSPL dataset, we'll estimate Tx position from path loss pattern
                tx_y, tx_x = np.unravel_index(np.argmin(fspl_path_loss), fspl_path_loss.shape)
                
                # Parse filename
                parts = sample['stem'].split('_')
                building_id = int(parts[0][1:])
                antenna_id = int(parts[1][3:])
                freq_id = int(parts[2][1:])
                sample_id = int(parts[3][1:])
                
                # Map frequency ID to actual frequency
                freq_map = {1: '868 MHz', 2: '1.8 GHz', 3: '3.5 GHz'}
                freq_str = freq_map.get(freq_id, f'f{freq_id}')
                
                # Plot input channels
                axes[i, 0].imshow(reflectance, cmap='viridis')
                axes[i, 0].plot(tx_x, tx_y, 'r*', markersize=10)
                axes[i, 0].set_title(f'Reflectance (B{building_id})')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(transmittance, cmap='viridis')
                axes[i, 1].plot(tx_x, tx_y, 'r*', markersize=10)
                axes[i, 1].set_title(f'Transmittance (Ant{antenna_id})')
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(fspl_path_loss, cmap='jet', vmin=0, vmax=160)
                axes[i, 2].plot(tx_x, tx_y, 'r*', markersize=10)
                axes[i, 2].set_title(f'FSPL Path Loss ({freq_str})')
                axes[i, 2].axis('off')
                
                # Plot ground truth output
                axes[i, 3].imshow(output_image, cmap='jet', vmin=0, vmax=160)
                axes[i, 3].plot(tx_x, tx_y, 'r*', markersize=10)
                axes[i, 3].set_title(f'Ground Truth (S{sample_id})')
                axes[i, 3].axis('off')
                
                print(f"Sample {i+1}: {sample['stem']}")
                print(f"  Size: {input_image.shape}, Tx: ({tx_x}, {tx_y})")
                print(f"  FSPL range: {fspl_path_loss.min():.1f}-{fspl_path_loss.max():.1f}")
                print(f"  Ground truth range: {output_image.min():.1f}-{output_image.max():.1f}")
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                for j in range(4):
                    axes[i, j].text(0.5, 0.5, 'Error', ha='center', va='center', transform=axes[i, j].transAxes)
                    axes[i, j].axis('off')
    
    plt.suptitle('ICASSP2025 Dataset with FSPL Channel\n(Reflectance + Transmittance + FSPL Path Loss → Ground Truth)', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'fspl_dataset_examples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2. Create FSPL vs Ground Truth comparison
    print("Creating FSPL vs Ground Truth comparison...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Calculate error metrics
    all_errors = []
    all_fspl_values = []
    all_gt_values = []
    
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
                fspl_path_loss = input_image[:, :, 2]
                
                # Find Tx position
                tx_y, tx_x = np.unravel_index(np.argmin(fspl_path_loss), fspl_path_loss.shape)
                
                # Calculate error metrics
                error = fspl_path_loss - output_image
                mae = np.mean(np.abs(error))
                rmse = np.sqrt(np.mean(error**2))
                
                all_errors.extend(error.flatten())
                all_fspl_values.extend(fspl_path_loss.flatten())
                all_gt_values.extend(output_image.flatten())
                
                # Parse filename
                parts = sample['stem'].split('_')
                building_id = int(parts[0][1:])
                antenna_id = int(parts[1][3:])
                freq_id = int(parts[2][1:])
                sample_id = int(parts[3][1:])
                
                freq_map = {1: '868 MHz', 2: '1.8 GHz', 3: '3.5 GHz'}
                freq_str = freq_map.get(freq_id, f'f{freq_id}')
                
                # Plot FSPL
                im1 = axes[i, 0].imshow(fspl_path_loss, cmap='jet', vmin=0, vmax=160)
                axes[i, 0].plot(tx_x, tx_y, 'r*', markersize=10)
                axes[i, 0].set_title(f'FSPL Path Loss ({freq_str})')
                axes[i, 0].axis('off')
                plt.colorbar(im1, ax=axes[i, 0])
                
                # Plot Ground Truth
                im2 = axes[i, 1].imshow(output_image, cmap='jet', vmin=0, vmax=160)
                axes[i, 1].plot(tx_x, tx_y, 'r*', markersize=10)
                axes[i, 1].set_title('Ground Truth')
                axes[i, 1].axis('off')
                plt.colorbar(im2, ax=axes[i, 1])
                
                # Plot Error
                im3 = axes[i, 2].imshow(error, cmap='RdBu_r', vmin=-50, vmax=50)
                axes[i, 2].plot(tx_x, tx_y, 'k*', markersize=10)
                axes[i, 2].set_title(f'Error (MAE: {mae:.1f} dB)')
                axes[i, 2].axis('off')
                plt.colorbar(im3, ax=axes[i, 2])
                
                print(f"Sample {i+1} - MAE: {mae:.2f} dB, RMSE: {rmse:.2f} dB")
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                for j in range(3):
                    axes[i, j].text(0.5, 0.5, 'Error', ha='center', va='center', transform=axes[i, j].transAxes)
                    axes[i, j].axis('off')
    
    plt.suptitle('FSPL vs Ground Truth Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'fspl_vs_ground_truth.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 3. Create statistical analysis
    print("Creating statistical analysis...")
    
    if all_errors:
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
        
        # Create error distribution plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Error histogram
        ax1.hist(all_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(all_errors), color='red', linestyle='--', label=f'Mean: {np.mean(all_errors):.1f}')
        ax1.axvline(np.median(all_errors), color='green', linestyle='--', label=f'Median: {np.median(all_errors):.1f}')
        ax1.set_xlabel('Error (dB)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Error Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot: FSPL vs Ground Truth
        ax2.scatter(all_gt_values[:10000], all_fspl_values[:10000], alpha=0.5, s=1)
        ax2.plot([0, 160], [0, 160], 'r--', label='Perfect Match')
        ax2.set_xlabel('Ground Truth (dB)')
        ax2.set_ylabel('FSPL (dB)')
        ax2.set_title('FSPL vs Ground Truth')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Building distribution
        buildings = sorted(building_counts.keys())
        counts = [building_counts[b] for b in buildings]
        ax3.bar(buildings, counts, color='lightcoral')
        ax3.set_title('Building Distribution')
        ax3.set_xlabel('Building ID')
        ax3.set_ylabel('Sample Count')
        ax3.tick_params(axis='x', rotation=45)
        
        # Summary statistics
        ax4.axis('off')
        summary_text = (
            f"FSPL Dataset Summary\n"
            f"{'='*40}\n"
            f"Total Input Files: {len(input_files)}\n"
            f"Total Output Files: {len(output_files)}\n"
            f"Matched Pairs: {len(samples)}\n"
            f"Buildings: {len(building_counts)}\n"
            f"Frequencies: {len(freq_counts)}\n"
            f"\nError Statistics:\n"
            f"MAE: {np.mean(np.abs(all_errors)):.2f} dB\n"
            f"RMSE: {np.sqrt(np.mean(np.array(all_errors)**2)):.2f} dB\n"
            f"Mean Error: {np.mean(all_errors):.2f} dB\n"
            f"Std Error: {np.std(all_errors):.2f} dB\n"
            f"\nImage Characteristics:\n"
            f"Input: 3-channel (Reflectance, Transmittance, FSPL)\n"
            f"Output: 1-channel (Ground Truth Path Loss)\n"
            f"FSPL Range: {min(all_fspl_values):.1f}-{max(all_fspl_values):.1f} dB\n"
            f"Ground Truth Range: {min(all_gt_values):.1f}-{max(all_gt_values):.1f} dB\n"
            f"\nFSPL Calculation:\n"
            f"Free Space Path Loss with antenna pattern\n"
            f"Frequency: 868 MHz, 1.8 GHz, 3.5 GHz\n"
            f"Resolution: 0.25 m per pixel"
        )
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('FSPL Dataset Statistical Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / 'fspl_statistical_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # 4. Create radiation pattern analysis
    print("Creating radiation pattern analysis...")
    
    # Load radiation pattern data
    pattern_dir = data_root / 'Radiation_Patterns'
    patterns = {}
    pattern_characteristics = {}
    
    for ant_id in range(1, 6):
        pattern_file = pattern_dir / f'Ant{ant_id}_Pattern.csv'
        if pattern_file.exists():
            pattern_data = pd.read_csv(pattern_file, header=None).values.flatten()
            patterns[ant_id] = pattern_data
            
            # Analyze pattern characteristics
            gain_range = np.max(pattern_data) - np.min(pattern_data)
            if gain_range < 1:
                pattern_type = "Omnidirectional"
            elif gain_range < 5:
                pattern_type = "Weakly Directional"
            elif gain_range < 15:
                pattern_type = "Moderately Directional"
            elif gain_range < 25:
                pattern_type = "Highly Directional"
            else:
                pattern_type = "Extremely Directional"
            
            pattern_characteristics[ant_id] = {
                'type': pattern_type,
                'max_gain': np.max(pattern_data),
                'min_gain': np.min(pattern_data),
                'gain_range': gain_range
            }
    
    # Create radiation pattern visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot all patterns in polar coordinates
    ax_polar = plt.subplot(2, 3, 1, projection='polar')
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    angles = np.arange(360)
    angles_rad = np.deg2rad(angles)
    
    for ant_id, pattern in patterns.items():
        pattern_linear = 10**(pattern/10)
        pattern_normalized = pattern_linear / np.max(pattern_linear)
        ax_polar.plot(angles_rad, pattern_normalized, color=colors[ant_id-1], 
                     linewidth=2, label=f'Ant{ant_id}')
    
    ax_polar.set_title('Radiation Patterns (Polar)', fontsize=14, fontweight='bold')
    ax_polar.legend(loc='upper right')
    ax_polar.grid(True, alpha=0.3)
    
    # Plot all patterns in cartesian coordinates
    ax_cart = plt.subplot(2, 3, 2)
    for ant_id, pattern in patterns.items():
        ax_cart.plot(angles, pattern, color=colors[ant_id-1], 
                    linewidth=2, label=f'Ant{ant_id}')
    
    ax_cart.set_title('Radiation Patterns (Cartesian)', fontsize=14, fontweight='bold')
    ax_cart.set_xlabel('Angle (degrees)')
    ax_cart.set_ylabel('Gain (dB)')
    ax_cart.legend()
    ax_cart.grid(True, alpha=0.3)
    
    # Show individual patterns
    for i, (ant_id, pattern) in enumerate(patterns.items()):
        if i < 3:  # Show first 3 patterns
            ax_individual = plt.subplot(2, 3, 4 + i, projection='polar')
            pattern_linear = 10**(pattern/10)
            pattern_normalized = pattern_linear / np.max(pattern_linear)
            ax_individual.plot(angles_rad, pattern_normalized, color=colors[ant_id-1], linewidth=2)
            ax_individual.fill(angles_rad, pattern_normalized, alpha=0.3, color=colors[ant_id-1])
            ax_individual.set_title(f'Ant{ant_id} - {pattern_characteristics[ant_id]["type"]}', 
                                   fontsize=12, fontweight='bold')
            ax_individual.grid(True, alpha=0.3)
    
    # Characteristics summary
    ax_summary = plt.subplot(2, 3, 6)
    ax_summary.axis('off')
    
    summary_text = "Radiation Pattern Characteristics\n" + "="*40 + "\n\n"
    
    for ant_id, chars in pattern_characteristics.items():
        summary_text += f"Antenna {ant_id} ({chars['type']}):\n"
        summary_text += f"  Max Gain: {chars['max_gain']:.2f} dB\n"
        summary_text += f"  Min Gain: {chars['min_gain']:.2f} dB\n"
        summary_text += f"  Gain Range: {chars['gain_range']:.2f} dB\n\n"
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('ICASSP2025 Radiation Pattern Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'radiation_pattern_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 5. Create frequency-specific analysis
    print("Creating frequency-specific analysis...")
    
    freq_samples = {1: [], 2: [], 3: []}
    
    for sample in samples:
        parts = sample['stem'].split('_')
        freq_id = int(parts[2][1:])
        if freq_id in freq_samples:
            freq_samples[freq_id].append(sample)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    freq_labels = ['868 MHz', '1.8 GHz', '3.5 GHz']
    
    for i, (freq_id, freq_label) in enumerate(zip([1, 2, 3], freq_labels)):
        if freq_samples[freq_id]:
            sample = freq_samples[freq_id][0]  # Use first sample for each frequency
            
            try:
                input_image = np.array(Image.open(sample['input_path']))
                output_image = np.array(Image.open(sample['output_path']))
                
                # Extract channels
                reflectance = input_image[:, :, 0]
                transmittance = input_image[:, :, 1]
                fspl_path_loss = input_image[:, :, 2]
                
                # Find Tx position
                tx_y, tx_x = np.unravel_index(np.argmin(fspl_path_loss), fspl_path_loss.shape)
                
                # Plot channels
                axes[i, 0].imshow(reflectance, cmap='viridis')
                axes[i, 0].plot(tx_x, tx_y, 'r*', markersize=10)
                axes[i, 0].set_title(f'Reflectance ({freq_label})')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(transmittance, cmap='viridis')
                axes[i, 1].plot(tx_x, tx_y, 'r*', markersize=10)
                axes[i, 1].set_title(f'Transmittance ({freq_label})')
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(fspl_path_loss, cmap='jet', vmin=0, vmax=160)
                axes[i, 2].plot(tx_x, tx_y, 'r*', markersize=10)
                axes[i, 2].set_title(f'FSPL Path Loss ({freq_label})')
                axes[i, 2].axis('off')
                
            except Exception as e:
                print(f"Error processing {freq_label} sample: {e}")
                for j in range(3):
                    axes[i, j].text(0.5, 0.5, 'Error', ha='center', va='center', transform=axes[i, j].transAxes)
                    axes[i, j].axis('off')
    
    plt.suptitle('Frequency-Specific Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'frequency_specific_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"All visualizations saved to: {output_dir}")
    return {
        'total_samples': len(samples),
        'buildings': len(building_counts),
        'frequencies': len(freq_counts),
        'output_dir': output_dir,
        'mae': np.mean(np.abs(all_errors)) if all_errors else None,
        'rmse': np.sqrt(np.mean(np.array(all_errors)**2)) if all_errors else None
    }

def main():
    """Main function"""
    
    # Configuration
    data_root = '/home/cine/Documents/Github/RadioDiff/datasets/ICASSP2025_Dataset'
    
    print("=== ICASSP2025 4-Channel Dataset Visualization ===")
    print(f"Data root: {data_root}")
    
    # Create comprehensive visualization
    results = create_comprehensive_visualization(data_root)
    
    print("\n=== Visualization completed! ===")
    print(f"Total samples processed: {results['total_samples']}")
    print(f"Buildings represented: {results['buildings']}")
    print(f"Frequencies represented: {results['frequencies']}")
    print(f"Output directory: {results['output_dir']}")
    
    if results['mae'] is not None:
        print(f"Mean Absolute Error: {results['mae']:.2f} dB")
        print(f"Root Mean Square Error: {results['rmse']:.2f} dB")
    
    print("\nGenerated files:")
    print("- fspl_dataset_examples.png: Shows 3 examples with all channels")
    print("- fspl_vs_ground_truth.png: FSPL vs Ground Truth comparison")
    print("- fspl_statistical_analysis.png: Statistical analysis and error metrics")
    print("- radiation_pattern_analysis.png: 5 antenna radiation patterns analysis")
    print("- frequency_specific_analysis.png: Frequency-specific examples")

if __name__ == "__main__":
    main()