#!/usr/bin/env python3
"""
Sample visualization script for RadioMapSeer dataset
Displays representative images from different categories
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import random

def display_sample_images(dataset_path, num_samples=5):
    """Display sample images from different categories"""
    
    dataset_path = Path(dataset_path)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('RadioMapSeer Dataset - Sample Images', fontsize=16, fontweight='bold')
    
    # 1. Sample radio map
    ax = axes[0, 0]
    png_dir = dataset_path / 'png' / 'buildings_complete'
    if png_dir.exists():
        png_files = list(png_dir.glob('*.png'))
        if png_files:
            sample_file = random.choice(png_files)
            img = Image.open(sample_file)
            ax.imshow(img, cmap='viridis')
            ax.set_title(f'Radio Map - Complete Buildings\n{sample_file.name}')
            ax.axis('off')
    
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
    
    # 3. Sample gain pattern
    ax = axes[0, 2]
    gain_dir = dataset_path / 'gain' / 'DPM'
    if gain_dir.exists():
        gain_files = list(gain_dir.glob('*.png'))
        if gain_files:
            sample_file = random.choice(gain_files)
            img = Image.open(sample_file)
            ax.imshow(img, cmap='plasma')
            ax.set_title(f'Gain Pattern - DPM\n{sample_file.name}')
            ax.axis('off')
    
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
    
    # 5. Sample missing buildings scenario
    ax = axes[1, 1]
    missing_dir = dataset_path / 'png' / 'buildings_missing1'
    if missing_dir.exists():
        missing_files = list(missing_dir.glob('*.png'))
        if missing_files:
            sample_file = random.choice(missing_files)
            img = Image.open(sample_file)
            ax.imshow(img, cmap='viridis')
            ax.set_title(f'Radio Map - Missing Buildings\n{sample_file.name}')
            ax.axis('off')
    
    # 6. Sample cars scenario
    ax = axes[1, 2]
    cars_dir = dataset_path / 'png' / 'cars'
    if cars_dir.exists():
        cars_files = list(cars_dir.glob('*.png'))
        if cars_files:
            sample_file = random.choice(cars_files)
            img = Image.open(sample_file)
            ax.imshow(img, cmap='viridis')
            ax.set_title(f'Radio Map - Cars\n{sample_file.name}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/cine/Documents/Github/RadioDiff/radiomapseer_samples.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Sample images saved to radiomapseer_samples.png")

def analyze_image_intensity(dataset_path):
    """Analyze intensity distribution of sample images"""
    
    dataset_path = Path(dataset_path)
    
    # Sample images from different categories
    categories = [
        'png/buildings_complete',
        'png/buildings_missing1',
        'png/cars',
        'gain/DPM'
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('RadioMapSeer - Intensity Distribution Analysis', fontsize=16, fontweight='bold')
    
    for i, category in enumerate(categories):
        ax = axes[i//2, i%2]
        category_path = dataset_path / category
        
        if category_path.exists():
            png_files = list(category_path.glob('*.png'))
            if png_files:
                # Sample 10 images
                sample_files = random.sample(png_files, min(10, len(png_files)))
                intensities = []
                
                for img_file in sample_files:
                    img = Image.open(img_file)
                    img_array = np.array(img)
                    intensities.extend(img_array.flatten())
                
                ax.hist(intensities, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_title(f'{category.replace("/", " / ")}\n({len(sample_files)} samples)')
                ax.set_xlabel('Intensity Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/cine/Documents/Github/RadioDiff/radiomapseer_intensity.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Intensity analysis saved to radiomapseer_intensity.png")

def main():
    """Main function"""
    dataset_path = "/home/cine/Documents/dataset/RadioMapSeer"
    
    print("🎨 Generating sample visualizations...")
    
    # Display sample images
    display_sample_images(dataset_path)
    
    # Analyze intensity distributions
    analyze_image_intensity(dataset_path)
    
    print("🎉 Sample visualization completed!")

if __name__ == "__main__":
    main()