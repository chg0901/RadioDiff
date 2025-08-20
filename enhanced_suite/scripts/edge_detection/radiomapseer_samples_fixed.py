#!/usr/bin/env python3
"""
Fixed sample visualization script for RadioMapSeer dataset
Displays representative images from different categories
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path

def display_sample_images_fixed(dataset_path):
    """Display sample images from different categories - FIXED VERSION"""
    
    dataset_path = Path(dataset_path)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('RadioMapSeer Dataset - Sample Images', fontsize=16, fontweight='bold')
    
    # 1. Sample radio map
    ax = axes[0, 0]
    sample_file = dataset_path / 'png' / 'buildings_complete' / '0.png'
    if sample_file.exists():
        img = Image.open(sample_file)
        ax.imshow(img, cmap='viridis')
        ax.set_title(f'Radio Map - Complete Buildings\nSample 0')
        ax.axis('off')
        print(f"✓ Found: {sample_file}")
    else:
        ax.text(0.5, 0.5, 'Not Found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Radio Map - Complete Buildings')
        ax.axis('off')
        print(f"✗ Missing: {sample_file}")
    
    # 2. Sample antenna positions
    ax = axes[0, 1]
    antenna_file = dataset_path / 'antenna' / '0.json'
    if antenna_file.exists():
        with open(antenna_file, 'r') as f:
            antenna_positions = json.load(f)
        
        # Plot antenna positions
        x_coords = [pos[0] for pos in antenna_positions]
        y_coords = [pos[1] for pos in antenna_positions]
        ax.scatter(x_coords, y_coords, c='red', s=20, alpha=0.7)
        ax.set_title(f'Antenna Positions\n{len(antenna_positions)} antennas')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        print(f"✓ Found: {antenna_file}")
    else:
        ax.text(0.5, 0.5, 'Not Found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Antenna Positions')
        print(f"✗ Missing: {antenna_file}")
    
    # 3. Sample gain pattern
    ax = axes[0, 2]
    gain_file = dataset_path / 'gain' / 'DPM' / '0_0.png'
    if gain_file.exists():
        img = Image.open(gain_file)
        ax.imshow(img, cmap='plasma')
        ax.set_title(f'Gain Pattern - DPM\nSample 0, Antenna 0')
        ax.axis('off')
        print(f"✓ Found: {gain_file}")
    else:
        ax.text(0.5, 0.5, 'Not Found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Gain Pattern - DPM')
        ax.axis('off')
        print(f"✗ Missing: {gain_file}")
    
    # 4. Sample building polygons
    ax = axes[1, 0]
    polygon_file = dataset_path / 'polygon' / 'buildings_and_cars' / '0.json'
    if polygon_file.exists():
        with open(polygon_file, 'r') as f:
            polygon_data = json.load(f)
        
        # Plot polygons (polygon_data is a list of polygon objects)
        for i, vertices in enumerate(polygon_data[:10]):  # Show first 10 objects
            if isinstance(vertices, list) and len(vertices) > 0:
                vertices_array = np.array(vertices)
                # Close the polygon
                vertices_array = np.vstack([vertices_array, vertices_array[0]])
                ax.plot(vertices_array[:, 0], vertices_array[:, 1], 'b-', alpha=0.6)
                ax.fill(vertices_array[:-1, 0], vertices_array[:-1, 1], alpha=0.3)
        
        ax.set_title(f'Building/Car Polygons\n{len(polygon_data)} objects')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        print(f"✓ Found: {polygon_file}")
    else:
        ax.text(0.5, 0.5, 'Not Found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Building/Car Polygons')
        print(f"✗ Missing: {polygon_file}")
    
    # 5. Sample missing buildings scenario
    ax = axes[1, 1]
    missing_file = dataset_path / 'png' / 'buildings_missing1' / '1' / '0.png'
    if missing_file.exists():
        img = Image.open(missing_file)
        ax.imshow(img, cmap='viridis')
        ax.set_title(f'Radio Map - Missing Buildings\nSample 0, Subdir 1')
        ax.axis('off')
        print(f"✓ Found: {missing_file}")
    else:
        ax.text(0.5, 0.5, 'Not Found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Radio Map - Missing Buildings')
        ax.axis('off')
        print(f"✗ Missing: {missing_file}")
    
    # 6. Sample cars scenario
    ax = axes[1, 2]
    cars_file = dataset_path / 'png' / 'cars' / '0.png'
    if cars_file.exists():
        img = Image.open(cars_file)
        ax.imshow(img, cmap='viridis')
        ax.set_title(f'Radio Map - Cars\nSample 0')
        ax.axis('off')
        print(f"✓ Found: {cars_file}")
    else:
        ax.text(0.5, 0.5, 'Not Found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Radio Map - Cars')
        ax.axis('off')
        print(f"✗ Missing: {cars_file}")
    
    plt.tight_layout()
    plt.savefig('/home/cine/Documents/Github/RadioDiff/radiomapseer_samples_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Sample images saved to radiomapseer_samples_fixed.png")

def main():
    """Main function"""
    dataset_path = "/home/cine/Documents/dataset/RadioMapSeer"
    
    print("🎨 Generating fixed sample visualizations...")
    
    # Display sample images
    display_sample_images_fixed(dataset_path)
    
    print("🎉 Fixed sample visualization completed!")

if __name__ == "__main__":
    main()