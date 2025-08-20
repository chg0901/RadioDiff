#!/usr/bin/env python3
"""
Show detailed dataset folder structure and display paired images
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import random

def show_folder_structure(dataset_path):
    """Show detailed folder structure and statistics"""
    
    dataset_path = Path(dataset_path)
    
    print("📁 RadioMapSeer Dataset - Detailed Folder Structure")
    print("=" * 60)
    
    # Main directories
    main_dirs = ['antenna', 'gain', 'png', 'polygon']
    
    for main_dir in main_dirs:
        dir_path = dataset_path / main_dir
        if dir_path.exists():
            print(f"\n📂 {main_dir.upper()}/")
            
            if main_dir == 'antenna':
                # Count antenna files
                antenna_files = list(dir_path.glob('*.json'))
                print(f"   ├── {len(antenna_files)} JSON files (antenna configurations)")
                if antenna_files:
                    sample_file = antenna_files[0]
                    with open(sample_file, 'r') as f:
                        antenna_data = json.load(f)
                    print(f"   └── Each file contains {len(antenna_data)} antenna positions")
                    
            elif main_dir == 'gain':
                # Gain pattern subdirectories
                gain_subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
                print(f"   ├── {len(gain_subdirs)} gain pattern types:")
                for subdir in gain_subdirs:
                    png_files = list(subdir.glob('*.png'))
                    print(f"   │   ├── {subdir.name}/ ({len(png_files)} images)")
                print(f"   └── Total gain images: {sum(len(list(d.glob('*.png'))) for d in gain_subdirs)}")
                
            elif main_dir == 'png':
                # Radio map subdirectories
                png_subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
                print(f"   ├── {len(png_subdirs)} radio map categories:")
                
                total_images = 0
                for subdir in png_subdirs:
                    if subdir.name.startswith('buildings_') and ('missing' in subdir.name or 'removed' in subdir.name):
                        # These have numbered subdirectories
                        num_subdirs = [d for d in subdir.iterdir() if d.is_dir() and d.name.isdigit()]
                        subdir_images = sum(len(list(d.glob('*.png'))) for d in num_subdirs)
                        print(f"   │   ├── {subdir.name}/ ({len(num_subdirs)} subdirs, {subdir_images} images)")
                    else:
                        png_files = list(subdir.glob('*.png'))
                        print(f"   │   ├── {subdir.name}/ ({len(png_files)} images)")
                        subdir_images = len(png_files)
                    total_images += subdir_images
                print(f"   └── Total radio map images: {total_images}")
                
            elif main_dir == 'polygon':
                # Polygon subdirectories
                polygon_subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
                print(f"   ├── {len(polygon_subdirs)} polygon categories:")
                
                total_files = 0
                for subdir in polygon_subdirs:
                    if subdir.name.startswith('buildings_') and ('difference' in subdir.name or 'removed' in subdir.name):
                        # These have numbered subdirectories
                        num_subdirs = [d for d in subdir.iterdir() if d.is_dir() and d.name.isdigit()]
                        subdir_files = sum(len(list(d.glob('*.json'))) for d in num_subdirs)
                        print(f"   │   ├── {subdir.name}/ ({len(num_subdirs)} subdirs, {subdir_files} files)")
                    else:
                        json_files = list(subdir.glob('*.json'))
                        print(f"   │   ├── {subdir.name}/ ({len(json_files)} files)")
                        subdir_files = len(json_files)
                    total_files += subdir_files
                print(f"   └── Total polygon files: {total_files}")

def display_paired_images(dataset_path, sample_id="17"):
    """Display paired images from different folders for the same sample"""
    
    dataset_path = Path(dataset_path)
    
    print(f"\n🖼️  Paired Images for Sample {sample_id}")
    print("=" * 60)
    
    # Create a comprehensive figure showing all related images
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'RadioMapSeer Dataset - Paired Images for Sample {sample_id}', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    image_count = 0
    
    # 1. Complete buildings radio map
    ax = axes[image_count]
    complete_path = dataset_path / 'png' / 'buildings_complete' / f'{sample_id}.png'
    if complete_path.exists():
        img = Image.open(complete_path)
        ax.imshow(img, cmap='viridis')
        ax.set_title('Complete Buildings\nRadio Map')
    else:
        ax.text(0.5, 0.5, 'Not Found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Complete Buildings\nRadio Map')
    ax.axis('off')
    image_count += 1
    
    # 2. Missing buildings scenarios
    for missing_level in [1, 2]:
        ax = axes[image_count]
        missing_path = dataset_path / 'png' / f'buildings_missing{missing_level}' / f'{sample_id}.png'
        if missing_path.exists():
            img = Image.open(missing_path)
            ax.imshow(img, cmap='viridis')
            ax.set_title(f'Missing Buildings {missing_level}\nRadio Map')
        else:
            ax.text(0.5, 0.5, 'Not Found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Missing Buildings {missing_level}\nRadio Map')
        ax.axis('off')
        image_count += 1
    
    # 3. Cars radio map
    ax = axes[image_count]
    cars_path = dataset_path / 'png' / 'cars' / f'{sample_id}.png'
    if cars_path.exists():
        img = Image.open(cars_path)
        ax.imshow(img, cmap='viridis')
        ax.set_title('Cars Scenario\nRadio Map')
    else:
        ax.text(0.5, 0.5, 'Not Found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Cars Scenario\nRadio Map')
    ax.axis('off')
    image_count += 1
    
    # 4. Roads radio map
    ax = axes[image_count]
    roads_path = dataset_path / 'png' / 'roads' / f'{sample_id}.png'
    if roads_path.exists():
        img = Image.open(roads_path)
        ax.imshow(img, cmap='viridis')
        ax.set_title('Roads Scenario\nRadio Map')
    else:
        ax.text(0.5, 0.5, 'Not Found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Roads Scenario\nRadio Map')
    ax.axis('off')
    image_count += 1
    
    # 5. Antenna positions
    ax = axes[image_count]
    antenna_path = dataset_path / 'antenna' / f'{sample_id}.json'
    if antenna_path.exists():
        with open(antenna_path, 'r') as f:
            antenna_positions = json.load(f)
        x_coords = [pos[0] for pos in antenna_positions]
        y_coords = [pos[1] for pos in antenna_positions]
        ax.scatter(x_coords, y_coords, c='red', s=15, alpha=0.7)
        ax.set_title(f'Antenna Positions\n({len(antenna_positions)} antennas)')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    else:
        ax.text(0.5, 0.5, 'Not Found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Antenna Positions')
    image_count += 1
    
    # 6. Gain patterns (different types)
    gain_types = ['DPM', 'IRT2', 'IRT4', 'carsDPM']
    for gain_type in gain_types:
        ax = axes[image_count]
        gain_path = dataset_path / 'gain' / gain_type / f'{sample_id}.png'
        if gain_path.exists():
            img = Image.open(gain_path)
            ax.imshow(img, cmap='plasma')
            ax.set_title(f'Gain Pattern\n{gain_type}')
        else:
            ax.text(0.5, 0.5, 'Not Found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Gain Pattern\n{gain_type}')
        ax.axis('off')
        image_count += 1
    
    # 7. Building polygons
    ax = axes[image_count]
    polygon_path = dataset_path / 'polygon' / 'buildings_and_cars' / f'{sample_id}.json'
    if polygon_path.exists():
        with open(polygon_path, 'r') as f:
            polygon_data = json.load(f)
        
        # Plot polygons
        for i, vertices in enumerate(polygon_data[:20]):  # Show first 20 objects
            if isinstance(vertices, list) and len(vertices) > 0:
                vertices_array = np.array(vertices)
                vertices_array = np.vstack([vertices_array, vertices_array[0]])
                ax.plot(vertices_array[:, 0], vertices_array[:, 1], 'b-', alpha=0.6)
                ax.fill(vertices_array[:-1, 0], vertices_array[:-1, 1], alpha=0.3)
        
        ax.set_title(f'Building/Car Polygons\n({len(polygon_data)} objects)')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    else:
        ax.text(0.5, 0.5, 'Not Found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Building/Car Polygons')
    image_count += 1
    
    # Hide unused subplots
    for i in range(image_count, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/cine/Documents/Github/RadioDiff/radiomapseer_paired_images.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Paired images saved to radiomapseer_paired_images.png")

def show_multiple_samples(dataset_path, num_samples=3):
    """Show multiple samples with their paired images"""
    
    dataset_path = Path(dataset_path)
    
    # Get available sample IDs
    complete_dir = dataset_path / 'png' / 'buildings_complete'
    if complete_dir.exists():
        sample_files = list(complete_dir.glob('*.png'))
        sample_ids = [f.stem for f in sample_files[:num_samples]]
    else:
        sample_ids = ['17', '50', '100']
    
    print(f"\n🎯 Multiple Sample Comparison (Samples: {', '.join(sample_ids)})")
    print("=" * 60)
    
    fig, axes = plt.subplots(len(sample_ids), 4, figsize=(16, 4*len(sample_ids)))
    if len(sample_ids) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('RadioMapSeer Dataset - Multiple Sample Comparison', fontsize=16, fontweight='bold')
    
    scenarios = [
        ('buildings_complete', 'Complete Buildings'),
        ('buildings_missing1', 'Missing Buildings 1'),
        ('cars', 'Cars Scenario'),
        ('roads', 'Roads Scenario')
    ]
    
    for i, sample_id in enumerate(sample_ids):
        for j, (scenario, title) in enumerate(scenarios):
            ax = axes[i, j]
            img_path = dataset_path / 'png' / scenario / f'{sample_id}.png'
            
            if img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img, cmap='viridis')
                ax.set_title(f'{title}\nSample {sample_id}')
            else:
                ax.text(0.5, 0.5, 'Not Found', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{title}\nSample {sample_id}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/cine/Documents/Github/RadioDiff/radiomapseer_multiple_samples.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Multiple samples comparison saved to radiomapseer_multiple_samples.png")

def main():
    """Main function"""
    dataset_path = "/home/cine/Documents/dataset/RadioMapSeer"
    
    print("🔍 Analyzing RadioMapSeer dataset structure and paired images...")
    
    # Show folder structure
    show_folder_structure(dataset_path)
    
    # Display paired images for sample 17
    display_paired_images(dataset_path, "17")
    
    # Show multiple samples
    show_multiple_samples(dataset_path, num_samples=3)
    
    print("\n🎉 Dataset structure and paired images analysis completed!")

if __name__ == "__main__":
    main()