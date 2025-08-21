#!/usr/bin/env python3
"""
ICASSP2025 Dataset Arrangement Script for VAE Training

This script arranges the ICASSP2025 dataset into the proper structure for VAE training:
- Creates three-channel input images (reflectance, transmittance, FSPL)
- Creates single-channel output images (path loss)
- Crops images to 256x256 while preserving Tx position
- Organizes data in RadioMapSeer-compatible structure

Author: Claude Code Assistant
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import shutil
from pathlib import Path
import cv2

class ICASSPDatasetArranger:
    """Dataset arranger for ICASSP2025 dataset"""
    
    def __init__(self, data_root, output_root, target_size=(256, 256)):
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.target_size = target_size
        
        # Frequency dictionary
        self.FREQ_DICT = {
            'f1': 868e6,   # 868 MHz
            'f2': 1.8e9,   # 1.8 GHz
            'f3': 3.5e9    # 3.5 GHz
        }
        self.PIXEL_SIZE = 0.25  # Spatial resolution (m)
        
        # Create output directories
        self.create_output_structure()
        
    def create_output_structure(self):
        """Create the output directory structure"""
        # Main directories
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Training and validation directories
        for split in ['train', 'val']:
            split_dir = self.output_root / split
            split_dir.mkdir(exist_ok=True)
            
            # Image and edge directories
            (split_dir / 'image').mkdir(exist_ok=True)
            (split_dir / 'edge').mkdir(exist_ok=True)
    
    def load_antenna_pattern(self, antenna_id):
        """Load antenna radiation pattern"""
        pattern_file = self.data_root / 'Radiation_Patterns' / f'Ant{antenna_id}_Pattern.csv'
        pattern_data = pd.read_csv(pattern_file, header=None)
        return pattern_data.values.flatten()
    
    def load_tx_position(self, building_id, antenna_id, freq_id, sample_id):
        """Load transmitter position information"""
        pos_file = self.data_root / 'Positions' / f'Positions_B{building_id}_Ant{antenna_id}_f{freq_id}.csv'
        positions = pd.read_csv(pos_file)
        pos = positions.iloc[sample_id]
        return pos
    
    def load_input_image(self, building_id, antenna_id, freq_id, sample_id):
        """Load input image"""
        image_name = f'B{building_id}_Ant{antenna_id}_f{freq_id}_S{sample_id}'
        image_path = self.data_root / 'Inputs' / 'Task_1_ICASSP' / f'{image_name}.png'
        return np.array(Image.open(image_path))
    
    def load_output_image(self, building_id, antenna_id, freq_id, sample_id):
        """Load output image"""
        image_name = f'B{building_id}_Ant{antenna_id}_f{freq_id}_S{sample_id}'
        image_path = self.data_root / 'Outputs' / 'Task_1_ICASSP' / f'{image_name}.png'
        return np.array(Image.open(image_path))
    
    def calculate_fspl(self, frequency_hz, grid_coords, tx_pos, tx_azimuth):
        """Calculate Free Space Path Loss"""
        c = 3e8  # Speed of light
        wavelength = c / frequency_hz
        x, y = grid_coords
        tx_x, tx_y = tx_pos
        
        x_meters = x * self.PIXEL_SIZE
        y_meters = y * self.PIXEL_SIZE
        tx_x_meters = tx_x * self.PIXEL_SIZE
        tx_y_meters = tx_y * self.PIXEL_SIZE
        
        dx = x_meters - tx_x_meters
        dy = y_meters - tx_y_meters
        distances = np.sqrt(dx**2 + dy**2)
        distances = np.maximum(distances, 0.1)
        
        # Basic FSPL calculation
        fspl = 20 * np.log10(4 * np.pi * distances / wavelength)
        
        return fspl
    
    def calculate_fspl_with_pattern(self, power_levels, frequency_hz, grid_coords, tx_pos, tx_azimuth):
        """Calculate FSPL considering antenna radiation pattern"""
        c = 3e8  # Speed of light
        wavelength = c / frequency_hz
        x, y = grid_coords
        tx_x, tx_y = tx_pos
        
        x_meters = x * self.PIXEL_SIZE
        y_meters = y * self.PIXEL_SIZE
        tx_x_meters = tx_x * self.PIXEL_SIZE
        tx_y_meters = tx_y * self.PIXEL_SIZE
        
        dx = x_meters - tx_x_meters
        dy = y_meters - tx_y_meters
        distances = np.sqrt(dx**2 + dy**2)
        distances = np.maximum(distances, 0.1)
        
        angles = (np.degrees(np.arctan2(dy, dx)) + tx_azimuth) % 360
        angle_indices = np.round(angles).astype(int) % 360
        
        fspl_basic = 20 * np.log10(4 * np.pi * distances / wavelength)
        
        gain_linear = 10**(power_levels/10)
        max_gain = np.max(gain_linear)
        gain_normalized = gain_linear / max_gain
        
        pattern_factors = gain_normalized[angle_indices]
        pattern_factors = np.maximum(pattern_factors, 1e-10)
        
        total_path_loss = fspl_basic - 10 * np.log10(pattern_factors)
        
        return total_path_loss
    
    def crop_with_tx_center(self, image, tx_pos, target_size=(256, 256), border_margin=10):
        """Crop image to target size while keeping Tx away from borders"""
        h, w = image.shape[:2]
        tx_x, tx_y = tx_pos
        
        # Calculate crop boundaries
        crop_h, crop_w = target_size
        
        # Ensure Tx is not too close to borders
        min_x = border_margin
        max_x = w - border_margin - crop_w
        min_y = border_margin
        max_y = h - border_margin - crop_h
        
        # Center crop on Tx position
        center_x = tx_x - crop_w // 2
        center_y = tx_y - crop_h // 2
        
        # Clamp to valid boundaries
        center_x = np.clip(center_x, min_x, max_x)
        center_y = np.clip(center_y, min_y, max_y)
        
        # Perform crop
        cropped = image[center_y:center_y + crop_h, center_x:center_x + crop_w]
        
        return cropped, (center_x, center_y)
    
    def process_single_sample(self, building_id, antenna_id, freq_id, sample_id, split='train'):
        """Process a single sample and save to output directory"""
        try:
            # Load images
            input_image = self.load_input_image(building_id, antenna_id, freq_id, sample_id)
            output_image = self.load_output_image(building_id, antenna_id, freq_id, sample_id)
            
            # Load additional data
            tx_info = self.load_tx_position(building_id, antenna_id, freq_id, sample_id)
            power_levels = self.load_antenna_pattern(antenna_id)
            
            # Extract reflectance and transmittance channels
            reflectance = input_image[:, :, 0]
            transmittance = input_image[:, :, 1]
            
            # Calculate FSPL
            h, w = input_image.shape[:2]
            x = np.arange(w)
            y = np.arange(h)
            Y, X = np.meshgrid(x, y)
            
            frequency = self.FREQ_DICT[f'f{freq_id}']
            fspl = self.calculate_fspl_with_pattern(
                power_levels,
                frequency,
                (X, Y),
                (tx_info['X'], tx_info['Y']),
                (tx_info['Azimuth']-90)%360
            )
            
            # Find Tx position from distance channel
            distance = input_image[:, :, 2]
            tx_y, tx_x = np.unravel_index(np.argmin(distance), distance.shape)
            
            # Crop images
            cropped_reflectance, crop_offset = self.crop_with_tx_center(
                reflectance, (tx_x, tx_y), self.target_size
            )
            cropped_transmittance, _ = self.crop_with_tx_center(
                transmittance, (tx_x, tx_y), self.target_size
            )
            cropped_fspl, _ = self.crop_with_tx_center(
                fspl, (tx_x, tx_y), self.target_size
            )
            cropped_output, _ = self.crop_with_tx_center(
                output_image, (tx_x, tx_y), self.target_size
            )
            
            # Create three-channel input image
            three_channel_input = np.stack([
                cropped_reflectance,
                cropped_transmittance,
                cropped_fspl.astype(np.uint8)
            ], axis=-1)
            
            # Save images
            image_name = f'B{building_id}_Ant{antenna_id}_f{freq_id}_S{sample_id}'
            
            # Save three-channel input image
            input_path = self.output_root / split / 'image' / f'{image_name}.png'
            Image.fromarray(three_channel_input).save(input_path)
            
            # Save single-channel output image
            output_path = self.output_root / split / 'edge' / f'{image_name}.png'
            Image.fromarray(cropped_output).save(output_path)
            
            return True
            
        except Exception as e:
            print(f"Error processing sample {building_id}_{antenna_id}_{freq_id}_{sample_id}: {e}")
            return False
    
    def create_train_val_split(self, val_ratio=0.2):
        """Create train/validation split"""
        # Get all samples
        input_dir = self.data_root / 'Inputs' / 'Task_1_ICASSP'
        samples = []
        
        for file_name in os.listdir(input_dir):
            if file_name.endswith('.png'):
                parts = file_name.split('_')
                building_id = int(parts[0][1:])
                antenna_id = int(parts[1][3:])
                freq_id = int(parts[2][1:])
                sample_id = int(parts[3][1:].split('.')[0])
                samples.append((building_id, antenna_id, freq_id, sample_id))
        
        # Shuffle and split
        np.random.shuffle(samples)
        split_idx = int(len(samples) * (1 - val_ratio))
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]
        
        return train_samples, val_samples
    
    def process_dataset(self, val_ratio=0.2, max_samples=None):
        """Process the entire dataset"""
        print("Creating train/validation split...")
        train_samples, val_samples = self.create_train_val_split(val_ratio)
        
        print(f"Training samples: {len(train_samples)}")
        print(f"Validation samples: {len(val_samples)}")
        
        # Limit samples for testing
        if max_samples:
            train_samples = train_samples[:max_samples]
            val_samples = val_samples[:max_samples//4]
        
        # Process training samples
        print("Processing training samples...")
        train_success = 0
        for sample in tqdm(train_samples):
            if self.process_single_sample(*sample, split='train'):
                train_success += 1
        
        # Process validation samples
        print("Processing validation samples...")
        val_success = 0
        for sample in tqdm(val_samples):
            if self.process_single_sample(*sample, split='val'):
                val_success += 1
        
        print(f"Successfully processed {train_success}/{len(train_samples)} training samples")
        print(f"Successfully processed {val_success}/{len(val_samples)} validation samples")
        
        return train_success, val_success
    
    def create_sample_visualization(self, building_id=1, antenna_id=1, freq_id=1, sample_id=0):
        """Create visualization of a processed sample"""
        try:
            # Load original images
            input_image = self.load_input_image(building_id, antenna_id, freq_id, sample_id)
            output_image = self.load_output_image(building_id, antenna_id, freq_id, sample_id)
            
            # Load additional data
            tx_info = self.load_tx_position(building_id, antenna_id, freq_id, sample_id)
            power_levels = self.load_antenna_pattern(antenna_id)
            
            # Extract channels
            reflectance = input_image[:, :, 0]
            transmittance = input_image[:, :, 1]
            distance = input_image[:, :, 2]
            
            # Calculate FSPL
            h, w = input_image.shape[:2]
            x = np.arange(w)
            y = np.arange(h)
            Y, X = np.meshgrid(x, y)
            
            frequency = self.FREQ_DICT[f'f{freq_id}']
            fspl = self.calculate_fspl_with_pattern(
                power_levels,
                frequency,
                (X, Y),
                (tx_info['X'], tx_info['Y']),
                (tx_info['Azimuth']-90)%360
            )
            
            # Find Tx position
            tx_y, tx_x = np.unravel_index(np.argmin(distance), distance.shape)
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Original images
            axes[0, 0].imshow(reflectance, cmap='viridis')
            axes[0, 0].set_title('Reflectance')
            axes[0, 0].plot(tx_x, tx_y, 'r*', markersize=15)
            
            axes[0, 1].imshow(transmittance, cmap='viridis')
            axes[0, 1].set_title('Transmittance')
            axes[0, 1].plot(tx_x, tx_y, 'r*', markersize=15)
            
            axes[0, 2].imshow(distance, cmap='viridis')
            axes[0, 2].set_title('Distance')
            axes[0, 2].plot(tx_x, tx_y, 'r*', markersize=15)
            
            # Calculated FSPL and output
            axes[1, 0].imshow(fspl, cmap='jet', vmin=0, vmax=160)
            axes[1, 0].set_title('Calculated FSPL')
            axes[1, 0].plot(tx_x, tx_y, 'r*', markersize=15)
            
            axes[1, 1].imshow(output_image, cmap='jet', vmin=0, vmax=160)
            axes[1, 1].set_title('Ground Truth Output')
            axes[1, 1].plot(tx_x, tx_y, 'r*', markersize=15)
            
            # Three-channel input preview
            three_channel = np.stack([
                reflectance,
                transmittance,
                (fspl / 160 * 255).astype(np.uint8)
            ], axis=-1)
            axes[1, 2].imshow(three_channel)
            axes[1, 2].set_title('Three-Channel Input Preview')
            
            plt.suptitle(f'Sample B{building_id}_Ant{antenna_id}_f{freq_id}_S{sample_id}')
            plt.tight_layout()
            plt.savefig(self.output_root / 'sample_visualization.png', dpi=150, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error creating visualization: {e}")

def main():
    """Main function"""
    # Configuration
    data_root = '/home/cine/Documents/Github/RadioDiff/datasets/ICASSP2025_Dataset'
    output_root = './icassp2025_dataset_arranged'
    target_size = (256, 256)
    
    # Create arranger
    arranger = ICASSPDatasetArranger(data_root, output_root, target_size)
    
    # Create sample visualization
    print("Creating sample visualization...")
    arranger.create_sample_visualization()
    
    # Process dataset (use small number for testing)
    print("Processing dataset...")
    train_success, val_success = arranger.process_dataset(val_ratio=0.2, max_samples=100)
    
    print(f"Dataset processing complete!")
    print(f"Output saved to: {output_root}")
    
    # Print directory structure
    print("\nDirectory structure:")
    for root, dirs, files in os.walk(output_root):
        level = root.replace(output_root, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        if files:
            print(f"{subindent}{len(files)} files")

if __name__ == "__main__":
    main()