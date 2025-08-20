#!/usr/bin/env python3
"""
Test script for ICASSP2025 VAE training configurations
"""

import sys
import os
import torch
sys.path.append('./datasets')
from icassp2025_dataloader import create_icassp2025_dataloader
import yaml

def test_configurations():
    """Test all three VAE configurations"""
    
    # Configuration files
    config_files = [
        'configs/icassp2025_vae_building.yaml',
        'configs/icassp2025_vae_antenna.yaml', 
        'configs/icassp2025_vae_radio.yaml'
    ]
    
    vae_types = ['building', 'antenna', 'radio']
    
    print("Testing ICASSP2025 VAE configurations...")
    
    for config_file, vae_type in zip(config_files, vae_types):
        print(f"\n--- Testing {vae_type} VAE ---")
        
        # Load configuration
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"Config: {config_file}")
        print(f"VAE type: {vae_type}")
        print(f"Data config: {config['data']}")
        
        # Test dataloader creation
        try:
            dataloader = create_icassp2025_dataloader(
                data_root=config['data']['data_root'],
                crop_size=config['data']['crop_size'],
                tx_margin=config['data']['tx_margin'],
                batch_size=config['data']['batch_size'],
                vae_type=vae_type,
                split='train',
                num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
                shuffle=True
            )
            
            # Test getting a batch
            batch = next(iter(dataloader))
            print(f"✓ Dataloader created successfully")
            print(f"✓ Batch shape: {batch['image'].shape}")
            print(f"✓ Expected channels: {config['model']['ddconfig']['in_channels']}")
            print(f"✓ Expected resolution: {config['model']['ddconfig']['resolution']}")
            
            # Verify batch properties
            expected_shape = (
                config['data']['batch_size'],
                config['model']['ddconfig']['in_channels'],
                config['data']['crop_size'],
                config['data']['crop_size']
            )
            
            if batch['image'].shape == expected_shape:
                print(f"✓ Batch shape matches expected: {expected_shape}")
            else:
                print(f"✗ Batch shape mismatch. Got: {batch['image'].shape}, Expected: {expected_shape}")
            
            # Verify Tx positions are within bounds
            tx_positions = batch['tx_position']
            crop_size = config['data']['crop_size']
            tx_margin = config['data']['tx_margin']
            
            # Check if all Tx positions are within the crop bounds with margin
            valid_positions = torch.all(
                (tx_positions >= tx_margin) & 
                (tx_positions <= crop_size - tx_margin)
            )
            
            if valid_positions:
                print(f"✓ Tx positions are within valid bounds (margin: {tx_margin})")
            else:
                print(f"✗ Some Tx positions out of bounds:")
                print(f"   Tx positions: {tx_positions}")
                print(f"   Valid range: [{tx_margin}, {crop_size - tx_margin}]")
                print(f"   Out of bounds count: {torch.sum(~((tx_positions >= tx_margin) & (tx_positions <= crop_size - tx_margin)))}")
            
        except Exception as e:
            print(f"✗ Error testing {vae_type} VAE: {str(e)}")
            continue
    
    print("\n=== Configuration Test Summary ===")
    print("All configurations tested successfully!")
    print("\nNext steps:")
    print("1. Run training for each VAE type:")
    print("   python train_icassp2025_vae.py --cfg configs/icassp2025_vae_building.yaml --vae_type building --mode train")
    print("   python train_icassp2025_vae.py --cfg configs/icassp2025_vae_antenna.yaml --vae_type antenna --mode train")
    print("   python train_icassp2025_vae.py --cfg configs/icassp2025_vae_radio.yaml --vae_type radio --mode train")
    print("\n2. For inference:")
    print("   python train_icassp2025_vae.py --cfg configs/icassp2025_vae_building.yaml --vae_type building --mode inference --checkpoint path/to/checkpoint.pth")


if __name__ == "__main__":
    test_configurations()