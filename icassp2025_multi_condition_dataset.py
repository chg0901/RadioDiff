#!/usr/bin/env python3
"""
Multi-Condition Dataset Arranger for ICASSP2025

This script prepares the ICASSP2025 dataset for multi-condition diffusion model training
by organizing the data into separate conditional inputs and target outputs.

Architecture:
- Condition 1: 2-channel (reflectance + transmittance)
- Condition 2: 1-channel (FSPL map) 
- Target: 1-channel (path loss output)

This creates the proper structure for training multi-condition VAEs and diffusion models.

Author: RadioDiff Team
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from tqdm.auto import tqdm
import yaml
from typing import Tuple, Dict, List, Optional
import cv2

class MultiConditionDatasetArranger:
    """
    Multi-condition dataset arranger for ICASSP2025
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the dataset arranger
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path) if config_path else self._default_config()
        
        # Create output directories
        self._create_directories()
        
        # Dataset statistics
        self.stats = {
            'total_samples': 0,
            'train_samples': 0,
            'val_samples': 0,
            'test_samples': 0,
            'buildings': [],
            'frequencies': [],
            'antennas': []
        }
        
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'dataset': {
                'name': 'ICASSP2025_MultiCondition',
                'version': '1.0',
                'description': 'Multi-condition dataset for radio map prediction with VAE'
            },
            'paths': {
                'input_root': '/home/cine/Documents/dataset/ICASSP2025_Dataset/Inputs/Task_1_ICASSP',
                'output_root': '/home/cine/Documents/dataset/ICASSP2025_Dataset/Outputs/Task_3_ICASSP',
                'output_dir': './icassp2025_multi_condition_vae',
                'train_dir': 'train',
                'val_dir': 'val',
                'test_dir': 'test'
            },
            'data_split': {
                'train_ratio': 0.8,
                'val_ratio': 0.1,
                'test_ratio': 0.1
            },
            'image_size': [320, 320],
            'vae_config': {
                'condition1_channels': 2,  # reflectance + transmittance
                'condition2_channels': 1,  # distance map
                'target_channels': 1,     # path loss output
                'latent_dim': 3,
                'use_pretrained': False
            },
            'preprocessing': {
                'normalize': True,
                'normalize_range': [-1, 1],
                'resize': True,
                'augmentation': False
            }
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_directories(self):
        """Create output directory structure"""
        base_dir = Path(self.config['paths']['output_dir'])
        
        # Main directories
        self.dirs = {
            'base': base_dir,
            'train': base_dir / self.config['paths']['train_dir'],
            'val': base_dir / self.config['paths']['val_dir'],
            'test': base_dir / self.config['paths']['test_dir'],
            # Training data
            'train_condition1': base_dir / self.config['paths']['train_dir'] / 'condition1',
            'train_condition2': base_dir / self.config['paths']['train_dir'] / 'condition2',
            'train_target': base_dir / self.config['paths']['train_dir'] / 'target',
            # Validation data
            'val_condition1': base_dir / self.config['paths']['val_dir'] / 'condition1',
            'val_condition2': base_dir / self.config['paths']['val_dir'] / 'condition2',
            'val_target': base_dir / self.config['paths']['val_dir'] / 'target',
            # Test data
            'test_condition1': base_dir / self.config['paths']['test_dir'] / 'condition1',
            'test_condition2': base_dir / self.config['paths']['test_dir'] / 'condition2',
            'test_target': base_dir / self.config['paths']['test_dir'] / 'target'
        }
        
        # Create directories
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print(f"Created multi-condition dataset structure at: {base_dir}")
    
    def parse_filename(self, filename: str) -> Dict[str, int]:
        """
        Parse ICASSP2025 filename to extract metadata
        
        Args:
            filename: Filename in format B{building_id}_Ant{antenna_id}_f{freq_id}_S{sample_id}.png
            
        Returns:
            Dictionary containing parsed metadata
        """
        parts = filename.split('_')
        building_id = int(parts[0][1:])
        antenna_id = int(parts[1][3:])
        freq_id = int(parts[2][1:])
        sample_id = int(parts[3][1:].split('.')[0])
        
        return {
            'building_id': building_id,
            'antenna_id': antenna_id,
            'freq_id': freq_id,
            'sample_id': sample_id
        }
    
    def load_and_preprocess_image(self, image_path: str, target_size: Optional[Tuple[int, int]] = None, is_rgb: bool = True) -> np.ndarray:
        """
        Load and preprocess image
        
        Args:
            image_path: Path to image file
            target_size: Target size for resizing (width, height)
            
        Returns:
            Preprocessed image as numpy array
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Load image
        image = Image.open(image_path)
        
        # Convert to RGB if needed (for input images)
        if is_rgb:
            if image.mode != 'RGB':
                image = image.convert('RGB')
        # Convert to grayscale if needed (for output images)
        else:
            if image.mode != 'L':
                image = image.convert('L')
        
        # Resize if needed
        if target_size:
            image = image.resize(target_size, Image.Resampling.BILINEAR)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize to [0, 1]
        if image_array.dtype == np.uint8:
            image_array = image_array.astype(np.float32) / 255.0
        
        return image_array
    
    def extract_conditions_and_target(self, input_image: np.ndarray, output_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract conditional inputs and target from images
        
        Args:
            input_image: 3-channel input image (reflectance, transmittance, distance)
            output_image: 1-channel output image (path loss)
            
        Returns:
            Tuple of (condition1, condition2, target) where:
            - condition1: 2-channel (reflectance + transmittance)
            - condition2: 1-channel (distance)
            - target: 1-channel (path loss)
        """
        # Condition 1: reflectance + transmittance from input image
        condition1 = input_image[:, :, :2]  # First two channels
        
        # Condition 2: distance from third channel of input image
        condition2 = input_image[:, :, 2:3]  # Third channel (distance)
        
        # Target: path loss output
        target = output_image
        
        return condition1, condition2, target
    
    def process_single_sample(self, input_path: str, output_path: str, 
                            split: str, sample_info: Dict[str, int]) -> bool:
        """
        Process a single sample and save to appropriate directories
        
        Args:
            input_path: Path to input image
            output_path: Path to output image
            split: Data split ('train', 'val', 'test')
            sample_info: Sample metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load images
            input_image = self.load_and_preprocess_image(input_path, self.config['image_size'], is_rgb=True)
            output_image = self.load_and_preprocess_image(output_path, self.config['image_size'], is_rgb=False)
            
            # Extract conditions and target
            condition1, condition2, target = self.extract_conditions_and_target(input_image, output_image)
            
            # Generate output filename
            base_name = f"B{sample_info['building_id']}_Ant{sample_info['antenna_id']}_f{sample_info['freq_id']}_S{sample_info['sample_id']}"
            
            # Save condition1 (2-channel)
            condition1_path = self.dirs[f'{split}_condition1'] / f'{base_name}.png'
            condition1_img = Image.fromarray((condition1 * 255).astype(np.uint8))
            condition1_img.save(condition1_path)
            
            # Save condition2 (1-channel)
            condition2_path = self.dirs[f'{split}_condition2'] / f'{base_name}.png'
            condition2_img = Image.fromarray((condition2.squeeze() * 255).astype(np.uint8), mode='L')
            condition2_img.save(condition2_path)
            
            # Save target (1-channel)
            target_path = self.dirs[f'{split}_target'] / f'{base_name}.png'
            target_img = Image.fromarray((target.squeeze() * 255).astype(np.uint8), mode='L')
            target_img.save(target_path)
            
            return True
            
        except Exception as e:
            print(f"Error processing sample {sample_info}: {str(e)}")
            return False
    
    def create_data_split(self, file_list: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        Create train/val/test split from file list
        
        Args:
            file_list: List of input file paths
            
        Returns:
            Tuple of (train_files, val_files, test_files)
        """
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(file_list)
        
        n_total = len(file_list)
        n_train = int(n_total * self.config['data_split']['train_ratio'])
        n_val = int(n_total * self.config['data_split']['val_ratio'])
        
        train_files = file_list[:n_train]
        val_files = file_list[n_train:n_train + n_val]
        test_files = file_list[n_train + n_val:]
        
        return train_files, val_files, test_files
    
    def process_dataset(self):
        """
        Process the entire dataset and create multi-condition arrangement
        """
        print("Starting ICASSP2025 multi-condition dataset arrangement...")
        
        # Get list of input files
        input_dir = Path(self.config['paths']['input_root'])
        input_files = list(input_dir.glob('*.png'))
        
        print(f"Found {len(input_files)} input files")
        
        # Create data split
        train_files, val_files, test_files = self.create_data_split(input_files)
        
        print(f"Data split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        
        # Process each split
        for split_name, file_list in [('train', train_files), ('val', val_files), ('test', test_files)]:
            print(f"Processing {split_name} split...")
            
            successful_samples = 0
            for input_file in tqdm(file_list):
                # Parse filename
                sample_info = self.parse_filename(input_file.name)
                
                # Construct paths
                output_file = Path(self.config['paths']['output_root']) / input_file.name
                
                # Check if files exist
                if not output_file.exists():
                    print(f"Warning: Output file not found: {output_file}")
                    continue
                
                # Process sample
                if self.process_single_sample(str(input_file), str(output_file), split_name, sample_info):
                    successful_samples += 1
                    
                    # Update statistics
                    if sample_info['building_id'] not in self.stats['buildings']:
                        self.stats['buildings'].append(sample_info['building_id'])
                    if sample_info['freq_id'] not in self.stats['frequencies']:
                        self.stats['frequencies'].append(sample_info['freq_id'])
                    if sample_info['antenna_id'] not in self.stats['antennas']:
                        self.stats['antennas'].append(sample_info['antenna_id'])
            
            self.stats[f'{split_name}_samples'] = successful_samples
            print(f"Successfully processed {successful_samples} samples for {split_name}")
        
        # Update total statistics
        self.stats['total_samples'] = sum([self.stats['train_samples'], self.stats['val_samples'], self.stats['test_samples']])
        
        # Save statistics
        self._save_statistics()
        
        print("Multi-condition dataset arrangement completed!")
        self._print_summary()
    
    def _save_statistics(self):
        """Save dataset statistics to file"""
        stats_file = self.dirs['base'] / 'dataset_statistics.yaml'
        
        with open(stats_file, 'w') as f:
            yaml.dump(self.stats, f, default_flow_style=False)
        
        print(f"Statistics saved to: {stats_file}")
    
    def _print_summary(self):
        """Print dataset summary"""
        print("\n" + "="*60)
        print("ICASSP2025 Multi-Condition Dataset Summary")
        print("="*60)
        print(f"Total samples: {self.stats['total_samples']}")
        print(f"Train samples: {self.stats['train_samples']}")
        print(f"Validation samples: {self.stats['val_samples']}")
        print(f"Test samples: {self.stats['test_samples']}")
        print(f"Buildings: {len(self.stats['buildings'])} ({sorted(self.stats['buildings'])})")
        print(f"Frequencies: {len(self.stats['frequencies'])} ({sorted(self.stats['frequencies'])})")
        print(f"Antennas: {len(self.stats['antennas'])} ({sorted(self.stats['antennas'])})")
        print(f"Image size: {self.config['image_size']}")
        print(f"Condition 1 channels: {self.config['vae_config']['condition1_channels']} (reflectance + transmittance)")
        print(f"Condition 2 channels: {self.config['vae_config']['condition2_channels']} (FSPL map)")
        print(f"Target channels: {self.config['vae_config']['target_channels']} (path loss)")
        print(f"Output directory: {self.dirs['base']}")
        print("="*60)
    
    def create_vae_config(self):
        """Create VAE configuration file for training"""
        vae_config = {
            'model': {
                'name': 'MultiConditionVAE',
                'embed_dim': self.config['vae_config']['latent_dim'],
                'lossconfig': {
                    'disc_start': 50001,
                    'kl_weight': 0.000001,
                    'disc_weight': 0.5,
                    'disc_in_channels': 1
                },
                'ddconfig': {
                    'double_z': True,
                    'z_channels': self.config['vae_config']['latent_dim'],
                    'resolution': self.config['image_size'],
                    'in_channels': 1,  # For single VAE approach
                    'out_ch': 1,
                    'ch': 128,
                    'ch_mult': [1, 2, 4],
                    'num_res_blocks': 2,
                    'attn_resolutions': [],
                    'dropout': 0.0
                },
                'condition_config': {
                    'condition1_channels': self.config['vae_config']['condition1_channels'],
                    'condition2_channels': self.config['vae_config']['condition2_channels'],
                    'fusion_method': 'concatenation',  # or 'cross_attention'
                    'use_condition_encoder': True
                }
            },
            'data': {
                'name': 'multi_condition_icassp2025',
                'batch_size': 16,
                'dataset_root': str(self.dirs['base']),
                'condition1_dir': 'condition1',
                'condition2_dir': 'condition2',
                'target_dir': 'target'
            },
            'training': {
                'gradient_accumulate_every': 2,
                'lr': 5e-6,
                'min_lr': 5e-7,
                'train_num_steps': 150000,
                'save_and_sample_every': 5000,
                'log_freq': 100,
                'results_folder': './results/multi_condition_vae',
                'amp': False,
                'fp16': False
            }
        }
        
        config_file = self.dirs['base'] / 'vae_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(vae_config, f, default_flow_style=False)
        
        print(f"VAE configuration saved to: {config_file}")
        return config_file
    
    def create_sample_visualization(self, num_samples=3):
        """Create visualization of processed samples"""
        print(f"Creating sample visualization with {num_samples} samples...")
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        # Get random samples from training set
        train_files = list((self.dirs['train_condition1']).glob('*.png'))
        np.random.shuffle(train_files)
        
        for i in range(min(num_samples, len(train_files))):
            sample_file = train_files[i]
            sample_name = sample_file.stem
            
            # Load processed images
            condition1_path = self.dirs['train_condition1'] / f'{sample_name}.png'
            condition2_path = self.dirs['train_condition2'] / f'{sample_name}.png'
            target_path = self.dirs['train_target'] / f'{sample_name}.png'
            
            condition1 = np.array(Image.open(condition1_path))
            condition2 = np.array(Image.open(condition2_path))
            target = np.array(Image.open(target_path))
            
            # Plot
            axes[i, 0].imshow(condition1)
            axes[i, 0].set_title(f'Condition 1: Reflectance+Transmittance')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(condition2, cmap='viridis')
            axes[i, 1].set_title(f'Condition 2: Distance Map')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(target, cmap='jet')
            axes[i, 2].set_title(f'Target: Path Loss')
            axes[i, 2].axis('off')
            
            # Combined visualization
            combined = np.zeros((target.shape[0], target.shape[1], 3))
            combined[:, :, 0] = condition1[:, :, 0] / 255.0  # reflectance
            combined[:, :, 1] = condition1[:, :, 1] / 255.0  # transmittance
            combined[:, :, 2] = condition2 / 255.0           # distance
            
            axes[i, 3].imshow(combined)
            axes[i, 3].set_title(f'Combined Conditions')
            axes[i, 3].axis('off')
            
            # Add sample name as row label
            axes[i, 0].set_ylabel(sample_name, fontsize=10, rotation=0, labelpad=50)
        
        plt.tight_layout()
        plt.savefig(self.dirs['base'] / 'sample_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Sample visualization saved to: {self.dirs['base'] / 'sample_visualization.png'}")


def main():
    """Main function to run the multi-condition dataset arrangement"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Arrange ICASSP2025 dataset for multi-condition VAE training")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--input_root', type=str, help='Override input directory path')
    parser.add_argument('--output_root', type=str, help='Override output directory path')
    parser.add_argument('--visualize_only', action='store_true', help='Only create visualization')
    parser.add_argument('--num_visualize', type=int, default=3, help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Initialize arranger
    arranger = MultiConditionDatasetArranger(args.config)
    
    # Override paths if provided
    if args.input_root:
        arranger.config['paths']['input_root'] = args.input_root
    if args.output_root:
        arranger.config['paths']['output_dir'] = args.output_root
    
    if args.visualize_only:
        # Only create visualization (for existing dataset)
        arranger.create_sample_visualization(args.num_visualize)
    else:
        # Process dataset
        arranger.process_dataset()
        
        # Create VAE configuration
        arranger.create_vae_config()
        
        # Create sample visualization
        arranger.create_sample_visualization(args.num_visualize)


if __name__ == "__main__":
    main()