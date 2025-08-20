import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
from pathlib import Path
import pickle


class ICASSP2025Dataset(Dataset):
    """ICASSP2025 Dataset with Tx-aware cropping for specialized VAE training"""
    
    def __init__(self, 
                 data_root,
                 crop_size=96,
                 tx_margin=10,
                 split='train',
                 transform=None,
                 vae_type='building',
                 seed=42):
        """
        Args:
            data_root: Root directory of ICASSP2025 dataset
            crop_size: Size of cropped patches (default: 96)
            tx_margin: Minimum distance from Tx to image border (default: 10)
            split: 'train', 'val', or 'test'
            transform: Optional transform to be applied on images
            vae_type: Type of VAE ('building', 'antenna', 'radio')
            seed: Random seed for reproducibility
        """
        self.data_root = Path(data_root)
        self.crop_size = crop_size
        self.tx_margin = tx_margin
        self.split = split
        self.transform = transform
        self.vae_type = vae_type
        self.seed = seed
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Directories
        self.input_dir = self.data_root / "Inputs" / "Task_3_ICASSP_path_loss_results_replaced"
        self.output_dir = self.data_root / "Outputs" / "Task_3_ICASSP"
        self.positions_dir = self.data_root / "Positions"
        
        # Build file list
        self.build_file_list()
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
    
    def build_file_list(self):
        """Build list of input-output pairs"""
        self.file_pairs = []
        
        # Get all input files
        input_files = list(self.input_dir.glob("*.png"))
        input_files.sort()
        
        # Split data based on split ratio
        total_files = len(input_files)
        if self.split == 'train':
            split_files = input_files[:int(0.8 * total_files)]
        elif self.split == 'val':
            split_files = input_files[int(0.8 * total_files):int(0.9 * total_files)]
        else:  # test
            split_files = input_files[int(0.9 * total_files):]
        
        # Find corresponding output files
        for input_file in split_files:
            # Extract filename components
            filename = input_file.stem
            output_file = self.output_dir / f"{filename}.png"
            
            if output_file.exists():
                self.file_pairs.append((input_file, output_file))
        
        print(f"Found {len(self.file_pairs)} file pairs for {self.split} split")
    
    def extract_tx_position(self, filename):
        """Extract Tx position from filename using positions CSV"""
        # Parse filename: B10_Ant1_f1_S0
        parts = filename.stem.split('_')
        building = parts[0]
        antenna = parts[1]
        frequency = parts[2]
        scenario = parts[3]
        
        # Read positions CSV
        pos_file = self.positions_dir / f"Positions_{building}_{antenna}_{frequency}.csv"
        
        if pos_file.exists():
            df = pd.read_csv(pos_file)
            # Find the row for this scenario (using row index as scenario number)
            scenario_idx = int(scenario[1:])
            if scenario_idx < len(df):
                return df.iloc[scenario_idx]['X'], df.iloc[scenario_idx]['Y']
        
        # Default to center if not found
        return 181, 136  # Approximate center of 362x272 image
    
    def get_valid_crop_positions(self, tx_x, tx_y, img_width, img_height):
        """Get valid crop positions that keep Tx away from borders"""
        valid_positions = []
        
        # Calculate valid range for top-left corner
        min_x = max(0, tx_x - self.crop_size + self.tx_margin)
        max_x = min(img_width - self.crop_size, tx_x - self.tx_margin)
        
        min_y = max(0, tx_y - self.crop_size + self.tx_margin)
        max_y = min(img_height - self.crop_size, tx_y - self.tx_margin)
        
        # Generate all valid positions
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                valid_positions.append((x, y))
        
        return valid_positions
    
    def crop_with_tx_awareness(self, image, tx_x, tx_y):
        """Crop image ensuring Tx stays within margins"""
        img_width, img_height = image.size
        
        # Get valid crop positions
        valid_positions = self.get_valid_crop_positions(tx_x, tx_y, img_width, img_height)
        
        if not valid_positions:
            # If no valid positions, find the best possible position that minimizes margin violation
            # This happens when Tx is too close to the image borders
            crop_x = max(0, min(tx_x - self.crop_size // 2, img_width - self.crop_size))
            crop_y = max(0, min(tx_y - self.crop_size // 2, img_height - self.crop_size))
            
            # Adjust crop to maximize margin from Tx
            if tx_x < self.tx_margin:
                crop_x = 0
            elif tx_x > img_width - self.tx_margin:
                crop_x = img_width - self.crop_size
                
            if tx_y < self.tx_margin:
                crop_y = 0
            elif tx_y > img_height - self.tx_margin:
                crop_y = img_height - self.crop_size
        else:
            # Randomly select a valid position
            crop_x, crop_y = random.choice(valid_positions)
        
        # Crop the image
        cropped = image.crop((crop_x, crop_y, crop_x + self.crop_size, crop_y + self.crop_size))
        
        # Adjust Tx position for cropped image
        new_tx_x = tx_x - crop_x
        new_tx_y = tx_y - crop_y
        
        return cropped, new_tx_x, new_tx_y
    
    def extract_channels(self, input_image, output_image):
        """Extract different channel combinations based on VAE type"""
        # Convert to numpy arrays
        input_np = np.array(input_image)
        output_np = np.array(output_image)
        
        if self.vae_type == 'building':
            # VAE₁: Building structure (Reflectance + Transmittance channels)
            # Input image is 3-channel RGB, use first 2 channels
            building_channels = input_np[:, :, :2]  # Reflectance + Transmittance
            return Image.fromarray(building_channels)
        
        elif self.vae_type == 'antenna':
            # VAE₂: Antenna configuration (FSPL channel with radiation patterns)
            # Input image is 3-channel RGB, use third channel
            fspl_channel = input_np[:, :, 2:3]  # FSPL channel
            return Image.fromarray(fspl_channel.squeeze())
        
        elif self.vae_type == 'radio':
            # VAE₃: Radio map output (single channel path loss)
            # Output image is single channel
            return Image.fromarray(output_np)
        
        else:
            raise ValueError(f"Unknown VAE type: {self.vae_type}")
    
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        input_file, output_file = self.file_pairs[idx]
        
        # Load images
        input_image = Image.open(input_file).convert('RGB')
        output_image = Image.open(output_file).convert('L')  # Grayscale
        
        # Extract Tx position
        tx_x, tx_y = self.extract_tx_position(input_file)
        
        # Crop images with Tx awareness
        cropped_input, new_tx_x, new_tx_y = self.crop_with_tx_awareness(input_image, tx_x, tx_y)
        cropped_output, _, _ = self.crop_with_tx_awareness(output_image, tx_x, tx_y)
        
        # Extract appropriate channels based on VAE type
        if self.vae_type == 'radio':
            processed_image = cropped_output
        else:
            processed_image = self.extract_channels(cropped_input, cropped_output)
        
        # Apply transforms
        if self.transform:
            processed_image = self.transform(processed_image)
        
        # Prepare output dictionary
        sample = {
            'image': processed_image,
            'tx_position': torch.tensor([new_tx_x, new_tx_y], dtype=torch.float32),
            'original_size': torch.tensor([input_image.size[0], input_image.size[1]], dtype=torch.float32),
            'crop_position': torch.tensor([tx_x - new_tx_x, tx_y - new_tx_y], dtype=torch.float32),
            'filename': input_file.name
        }
        
        return sample


class ICASSP2025InferenceDataset(Dataset):
    """Dataset for inference with variable-size input conditions"""
    
    def __init__(self, 
                 data_root,
                 transform=None,
                 vae_type='building'):
        """
        Args:
            data_root: Root directory of ICASSP2025 dataset
            transform: Optional transform to be applied on images
            vae_type: Type of VAE ('building', 'antenna', 'radio')
        """
        self.data_root = Path(data_root)
        self.transform = transform
        self.vae_type = vae_type
        
        # Directories
        self.input_dir = self.data_root / "Inputs" / "Task_3_ICASSP_path_loss_results_replaced"
        self.output_dir = self.data_root / "Outputs" / "Task_3_ICASSP"
        self.positions_dir = self.data_root / "Positions"
        
        # Build file list
        self.build_file_list()
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
    
    def build_file_list(self):
        """Build list of input files for inference"""
        self.input_files = []
        
        # Get all input files
        input_files = list(self.input_dir.glob("*.png"))
        input_files.sort()
        
        self.input_files = input_files
        print(f"Found {len(self.input_files)} input files for inference")
    
    def extract_tx_position(self, filename):
        """Extract Tx position from filename using positions CSV"""
        # Parse filename: B10_Ant1_f1_S0
        parts = filename.stem.split('_')
        building = parts[0]
        antenna = parts[1]
        frequency = parts[2]
        scenario = parts[3]
        
        # Read positions CSV
        pos_file = self.positions_dir / f"Positions_{building}_{antenna}_{frequency}.csv"
        
        if pos_file.exists():
            df = pd.read_csv(pos_file)
            # Find the row for this scenario (using row index as scenario number)
            scenario_idx = int(scenario[1:])
            if scenario_idx < len(df):
                return df.iloc[scenario_idx]['X'], df.iloc[scenario_idx]['Y']
        
        # Default to center if not found
        return 181, 136  # Approximate center of 362x272 image
    
    def extract_channels(self, input_image):
        """Extract different channel combinations based on VAE type"""
        # Convert to numpy array
        input_np = np.array(input_image)
        
        if self.vae_type == 'building':
            # VAE₁: Building structure (Reflectance + Transmittance channels)
            building_channels = input_np[:, :, :2]  # First 2 channels
            return Image.fromarray(building_channels)
        
        elif self.vae_type == 'antenna':
            # VAE₂: Antenna configuration (FSPL channel)
            fspl_channel = input_np[:, :, 2:3]  # Third channel
            return Image.fromarray(fspl_channel.squeeze())
        
        else:
            raise ValueError(f"Unknown VAE type: {self.vae_type}")
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        input_file = self.input_files[idx]
        
        # Load input image
        input_image = Image.open(input_file).convert('RGB')
        
        # Extract Tx position
        tx_x, tx_y = self.extract_tx_position(input_file)
        
        # Extract appropriate channels based on VAE type
        processed_image = self.extract_channels(input_image)
        
        # Apply transforms
        if self.transform:
            processed_image = self.transform(processed_image)
        
        # Prepare output dictionary
        sample = {
            'image': processed_image,
            'tx_position': torch.tensor([tx_x, tx_y], dtype=torch.float32),
            'original_size': torch.tensor([input_image.size[0], input_image.size[1]], dtype=torch.float32),
            'filename': input_file.name
        }
        
        return sample


def create_icassp2025_dataloader(data_root, 
                                  crop_size=96, 
                                  tx_margin=10, 
                                  batch_size=16, 
                                  vae_type='building',
                                  split='train',
                                  num_workers=4,
                                  shuffle=True):
    """Create dataloader for ICASSP2025 dataset"""
    
    # Create dataset
    dataset = ICASSP2025Dataset(
        data_root=data_root,
        crop_size=crop_size,
        tx_margin=tx_margin,
        split=split,
        vae_type=vae_type,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def create_icassp2025_inference_dataloader(data_root, 
                                          vae_type='building',
                                          batch_size=1,
                                          num_workers=4):
    """Create inference dataloader for ICASSP2025 dataset"""
    
    # Create dataset
    dataset = ICASSP2025InferenceDataset(
        data_root=data_root,
        vae_type=vae_type,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


# Test function
def test_icassp2025_dataloader():
    """Test the ICASSP2025 dataloader"""
    data_root = "/home/cine/Documents/Github/RadioDiff/datasets/ICASSP2025_Dataset"
    
    print("Testing ICASSP2025 dataloader...")
    
    # Test building VAE dataloader
    print("\nTesting building VAE dataloader:")
    building_loader = create_icassp2025_dataloader(
        data_root=data_root,
        vae_type='building',
        batch_size=2,
        split='train'
    )
    
    for batch in building_loader:
        print(f"Batch image shape: {batch['image'].shape}")
        print(f"Batch tx_position: {batch['tx_position']}")
        print(f"Batch original_size: {batch['original_size']}")
        break
    
    # Test antenna VAE dataloader
    print("\nTesting antenna VAE dataloader:")
    antenna_loader = create_icassp2025_dataloader(
        data_root=data_root,
        vae_type='antenna',
        batch_size=2,
        split='train'
    )
    
    for batch in antenna_loader:
        print(f"Batch image shape: {batch['image'].shape}")
        print(f"Batch tx_position: {batch['tx_position']}")
        break
    
    # Test radio VAE dataloader
    print("\nTesting radio VAE dataloader:")
    radio_loader = create_icassp2025_dataloader(
        data_root=data_root,
        vae_type='radio',
        batch_size=2,
        split='train'
    )
    
    for batch in radio_loader:
        print(f"Batch image shape: {batch['image'].shape}")
        print(f"Batch tx_position: {batch['tx_position']}")
        break
    
    print("\nTesting inference dataloader:")
    inference_loader = create_icassp2025_inference_dataloader(
        data_root=data_root,
        vae_type='building',
        batch_size=1
    )
    
    for batch in inference_loader:
        print(f"Batch image shape: {batch['image'].shape}")
        print(f"Batch original_size: {batch['original_size']}")
        break
    
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    test_icassp2025_dataloader()