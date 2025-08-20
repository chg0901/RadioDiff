"""
Multi-Condition Dataset Loader for ICASSP2025

This dataset loader is designed to work with multi-condition VAE training,
supporting both single and dual VAE approaches.

Author: RadioDiff Team
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
from PIL import Image
import os
from typing import Tuple, Dict, Optional, List
import yaml

class MultiConditionICASSP2025Dataset(Dataset):
    """
    Multi-condition dataset for ICASSP2025 radio map prediction
    
    Loads:
    - Condition 1: 2-channel (reflectance + transmittance)
    - Condition 2: 1-channel (distance map)
    - Target: 1-channel (path loss)
    """
    
    def __init__(self, 
                 dataset_root: str,
                 split: str = 'train',
                 image_size: Tuple[int, int] = (320, 320),
                 normalize: bool = True,
                 augment: bool = False,
                 load_condition1: bool = True,
                 load_condition2: bool = True,
                 load_target: bool = True):
        """
        Initialize the dataset
        
        Args:
            dataset_root: Root directory of the dataset
            split: Data split ('train', 'val', 'test')
            image_size: Target image size (height, width)
            normalize: Whether to normalize images to [-1, 1]
            augment: Whether to apply data augmentation
            load_condition1: Whether to load condition 1
            load_condition2: Whether to load condition 2
            load_target: Whether to load target
        """
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment
        self.load_condition1 = load_condition1
        self.load_condition2 = load_condition2
        self.load_target = load_target
        
        # Set up directories
        self.condition1_dir = self.dataset_root / split / 'condition1'
        self.condition2_dir = self.dataset_root / split / 'condition2'
        self.target_dir = self.dataset_root / split / 'target'
        
        # Get file list
        self.file_list = self._get_file_list()
        
        # Set up transforms
        self.transform = self._get_transforms()
        
        print(f"Loaded {len(self.file_list)} samples for {split} split")
    
    def _get_file_list(self) -> List[str]:
        """Get list of sample files"""
        if not self.condition1_dir.exists():
            raise FileNotFoundError(f"Condition1 directory not found: {self.condition1_dir}")
        
        files = list(self.condition1_dir.glob('*.png'))
        files = [f.stem for f in files]
        files.sort()
        
        return files
    
    def _get_transforms(self) -> transforms.Compose:
        """Get image transforms"""
        transform_list = []
        
        # Resize
        if self.image_size != (320, 320):
            transform_list.append(transforms.Resize(self.image_size))
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalize
        if self.normalize:
            transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        
        return transforms.Compose(transform_list)
    
    def _load_image(self, image_path: Path, is_grayscale: bool = False) -> torch.Tensor:
        """Load and transform image"""
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = Image.open(image_path)
        
        # Convert to grayscale if needed
        if is_grayscale and image.mode != 'L':
            image = image.convert('L')
        elif not is_grayscale and image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        sample_name = self.file_list[idx]
        
        result = {'sample_name': sample_name}
        
        # Load condition 1 (2-channel RGB)
        if self.load_condition1:
            condition1_path = self.condition1_dir / f'{sample_name}.png'
            condition1 = self._load_image(condition1_path, is_grayscale=False)
            result['condition1'] = condition1
        
        # Load condition 2 (1-channel grayscale)
        if self.load_condition2:
            condition2_path = self.condition2_dir / f'{sample_name}.png'
            condition2 = self._load_image(condition2_path, is_grayscale=True)
            result['condition2'] = condition2
        
        # Load target (1-channel grayscale)
        if self.load_target:
            target_path = self.target_dir / f'{sample_name}.png'
            target = self._load_image(target_path, is_grayscale=True)
            result['target'] = target
        
        return result

class MultiConditionDataLoader:
    """
    Multi-condition data loader with various VAE training approaches
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the data loader
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_root = self.config['data']['dataset_root']
        self.batch_size = self.config['data']['batch_size']
        self.image_size = tuple(self.config['data']['image_size'])
        
        # VAE approach
        self.vae_approach = self.config['architecture']['vae_approach']
        
        print(f"Initialized MultiConditionDataLoader with {self.vae_approach} VAE approach")
    
    def get_dataloaders(self) -> Dict[str, DataLoader]:
        """Get train, validation, and test dataloaders"""
        dataloaders = {}
        
        for split in ['train', 'val', 'test']:
            # Check if split exists
            split_dir = Path(self.dataset_root) / split
            if not split_dir.exists():
                print(f"Warning: {split} split not found, skipping...")
                continue
            
            # Create dataset
            dataset = MultiConditionICASSP2025Dataset(
                dataset_root=self.dataset_root,
                split=split,
                image_size=self.image_size,
                normalize=True,
                augment=(split == 'train'),  # Only augment training data
                load_condition1=True,
                load_condition2=True,
                load_target=True
            )
            
            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=(split == 'train'),
                num_workers=self.config['data']['num_workers'],
                pin_memory=self.config['data']['pin_memory'],
                drop_last=(split == 'train')
            )
            
            dataloaders[split] = dataloader
            print(f"Created {split} dataloader with {len(dataset)} samples")
        
        return dataloaders
    
    def get_collate_fn(self) -> callable:
        """Get collate function based on VAE approach"""
        if self.vae_approach == 'single':
            return self._single_vae_collate
        elif self.vae_approach == 'dual':
            return self._dual_vae_collate
        else:
            raise ValueError(f"Unknown VAE approach: {self.vae_approach}")
    
    def _single_vae_collate(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate function for single VAE approach
        
        Concatenates all conditions and target into a single tensor
        """
        batch_size = len(batch)
        
        # Extract tensors
        condition1 = torch.stack([item['condition1'] for item in batch])
        condition2 = torch.stack([item['condition2'] for item in batch])
        target = torch.stack([item['target'] for item in batch])
        
        # For single VAE: concatenate condition1 (2ch) + condition2 (1ch) + target (1ch) = 4ch
        # But for training, we want to predict target from conditions
        input_tensor = torch.cat([condition1, condition2], dim=1)  # 3 channels total
        
        return {
            'input': input_tensor,  # 3-channel input for VAE
            'target': target,       # 1-channel target
            'condition1': condition1,
            'condition2': condition2,
            'sample_names': [item['sample_name'] for item in batch]
        }
    
    def _dual_vae_collate(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate function for dual VAE approach
        
        Keeps conditions and target separate for dual encoding
        """
        batch_size = len(batch)
        
        # Extract tensors
        condition1 = torch.stack([item['condition1'] for item in batch])
        condition2 = torch.stack([item['condition2'] for item in batch])
        target = torch.stack([item['target'] for item in batch])
        
        # For dual VAE: keep separate
        conditions = torch.cat([condition1, condition2], dim=1)  # 3 channels
        
        return {
            'conditions': conditions,  # 3-channel conditions
            'target': target,         # 1-channel target
            'condition1': condition1,
            'condition2': condition2,
            'sample_names': [item['sample_name'] for item in batch]
        }

class MultiConditionVAE(nn.Module):
    """
    Multi-Condition VAE model that can handle both single and dual approaches
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the multi-condition VAE
        
        Args:
            config_path: Path to configuration file
        """
        super().__init__()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.vae_approach = self.config['architecture']['vae_approach']
        self.embed_dim = self.config['model']['embed_dim']
        self.resolution = self.config['model']['ddconfig']['resolution']
        
        # Import VAE components
        from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
        
        if self.vae_approach == 'single':
            # Single VAE with concatenated input
            self.vae = AutoencoderKL(
                ddconfig=self.config['model']['ddconfig'],
                lossconfig=self.config['model']['lossconfig'],
                embed_dim=self.embed_dim,
                ckpt_path=self.config['model']['ckpt_path'],
            )
            
            # Modify input channels to accept 3 channels (conditions)
            self.vae.encoder.conv_in.in_channels = 3
            
        elif self.vae_approach == 'dual':
            # Dual VAE with separate encoders
            self.condition_encoder = AutoencoderKL(
                ddconfig=self.config['model']['ddconfig'],
                lossconfig=self.config['model']['lossconfig'],
                embed_dim=self.embed_dim,
                ckpt_path=self.config['model']['ckpt_path'],
            )
            
            self.target_encoder = AutoencoderKL(
                ddconfig=self.config['model']['ddconfig'],
                lossconfig=self.config['model']['lossconfig'],
                embed_dim=self.embed_dim,
                ckpt_path=self.config['model']['ckpt_path'],
            )
            
            # Modify input channels
            self.condition_encoder.encoder.conv_in.in_channels = 3  # conditions
            self.target_encoder.encoder.conv_in.in_channels = 1     # target
            
        print(f"Initialized MultiConditionVAE with {self.vae_approach} approach")
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        if self.vae_approach == 'single':
            return self._forward_single_vae(batch)
        elif self.vae_approach == 'dual':
            return self._forward_dual_vae(batch)
    
    def _forward_single_vae(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for single VAE"""
        input_tensor = batch['input']
        target = batch['target']
        
        # Encode and decode
        posterior = self.vae.encode(input_tensor)
        z = posterior.sample()
        recon = self.vae.decode(z)
        
        return {
            'reconstruction': recon,
            'latent': z,
            'posterior': posterior,
            'target': target
        }
    
    def _forward_dual_vae(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for dual VAE"""
        conditions = batch['conditions']
        target = batch['target']
        
        # Encode conditions and target separately
        condition_posterior = self.condition_encoder.encode(conditions)
        target_posterior = self.target_encoder.encode(target)
        
        condition_z = condition_posterior.sample()
        target_z = target_posterior.sample()
        
        # Fuse latents (simple concatenation for now)
        fused_z = torch.cat([condition_z, target_z], dim=1)
        
        # Decode (using condition decoder)
        recon = self.condition_encoder.decode(condition_z)
        
        return {
            'reconstruction': recon,
            'condition_latent': condition_z,
            'target_latent': target_z,
            'fused_latent': fused_z,
            'condition_posterior': condition_posterior,
            'target_posterior': target_posterior,
            'target': target
        }

def test_dataset():
    """Test the dataset loader"""
    print("Testing MultiCondition ICASSP2025 Dataset...")
    
    # Initialize data loader
    config_path = './configs/icassp2025_multi_condition_vae.yaml'
    data_loader = MultiConditionDataLoader(config_path)
    
    # Get dataloaders
    dataloaders = data_loader.get_dataloaders()
    
    # Test each split
    for split, dataloader in dataloaders.items():
        print(f"\nTesting {split} split:")
        
        # Get a batch
        try:
            batch = next(iter(dataloader))
            print(f"Batch keys: {batch.keys()}")
            
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape} {value.dtype}")
                else:
                    print(f"  {key}: {type(value)}")
            
            print(f"✓ {split} split loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading {split} split: {e}")
    
    print("\nDataset test completed!")

if __name__ == "__main__":
    test_dataset()