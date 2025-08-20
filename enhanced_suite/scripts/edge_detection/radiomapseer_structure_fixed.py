#!/usr/bin/env python3
"""
Show detailed dataset folder structure and display paired images
Fixed version with correct file path handling
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import random

def display_paired_images_fixed(dataset_path, sample_id="0"):
    """Display paired images from different folders for the same sample - FIXED VERSION"""
    
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
        print(f"✓ Found: {complete_path}")
    else:
        ax.text(0.5, 0.5, 'Not Found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Complete Buildings\nRadio Map')
        print(f"✗ Missing: {complete_path}")
    ax.axis('off')
    image_count += 1
    
    # 2. Missing buildings scenarios (check subdirectories)
    for missing_level in [1, 2]:
        ax = axes[image_count]
        # Try different subdirectories
        found = False
        for subdir in ['1', '2', '3', '4', '5', '6']:
            missing_path = dataset_path / 'png' / f'buildings_missing{missing_level}' / subdir / f'{sample_id}.png'
            if missing_path.exists():
                img = Image.open(missing_path)
                ax.imshow(img, cmap='viridis')
                ax.set_title(f'Missing Buildings {missing_level}\nRadio Map (subdir {subdir})')
                print(f"✓ Found: {missing_path}")
                found = True
                break
        
        if not found:
            ax.text(0.5, 0.5, 'Not Found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Missing Buildings {missing_level}\nRadio Map')
            print(f"✗ Missing: buildings_missing{missing_level} for sample {sample_id}")
        ax.axis('off')
        image_count += 1
    
    # 3. Cars radio map
    ax = axes[image_count]
    cars_path = dataset_path / 'png' / 'cars' / f'{sample_id}.png'
    if cars_path.exists():
        img = Image.open(cars_path)
        ax.imshow(img, cmap='viridis')
        ax.set_title('Cars Scenario\nRadio Map')
        print(f"✓ Found: {cars_path}")
    else:
        ax.text(0.5, 0.5, 'Not Found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Cars Scenario\nRadio Map')
        print(f"✗ Missing: {cars_path}")
    ax.axis('off')
    image_count += 1
    
    # 4. Roads radio map
    ax = axes[image_count]
    roads_path = dataset_path / 'png' / 'roads' / f'{sample_id}.png'
    if roads_path.exists():
        img = Image.open(roads_path)
        ax.imshow(img, cmap='viridis')
        ax.set_title('Roads Scenario\nRadio Map')
        print(f"✓ Found: {roads_path}")
    else:
        ax.text(0.5, 0.5, 'Not Found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Roads Scenario\nRadio Map')
        print(f"✗ Missing: {roads_path}")
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
        print(f"✓ Found: {antenna_path}")
    else:
        ax.text(0.5, 0.5, 'Not Found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Antenna Positions')
        print(f"✗ Missing: {antenna_path}")
    image_count += 1
    
    # 6. Gain patterns (different types) - these use sample_antenna.png format
    gain_types = ['DPM', 'IRT2', 'IRT4', 'carsDPM']
    for gain_type in gain_types:
        ax = axes[image_count]
        # Try different antenna indices for the gain patterns
        found = False
        for antenna_idx in [0, 10, 20, 30, 40]:
            gain_path = dataset_path / 'gain' / gain_type / f'{sample_id}_{antenna_idx}.png'
            if gain_path.exists():
                img = Image.open(gain_path)
                ax.imshow(img, cmap='plasma')
                ax.set_title(f'Gain Pattern\n{gain_type} (antenna {antenna_idx})')
                print(f"✓ Found: {gain_path}")
                found = True
                break
        
        if not found:
            ax.text(0.5, 0.5, 'Not Found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Gain Pattern\n{gain_type}')
            print(f"✗ Missing: gain/{gain_type} for sample {sample_id}")
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
        print(f"✓ Found: {polygon_path}")
    else:
        ax.text(0.5, 0.5, 'Not Found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Building/Car Polygons')
        print(f"✗ Missing: {polygon_path}")
    image_count += 1
    
    # Hide unused subplots
    for i in range(image_count, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/cine/Documents/Github/RadioDiff/radiomapseer_paired_images_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Paired images saved to radiomapseer_paired_images_fixed.png")

def show_multiple_samples_fixed(dataset_path, num_samples=3):
    """Show multiple samples with their paired images - FIXED VERSION"""
    
    dataset_path = Path(dataset_path)
    
    # Get available sample IDs
    complete_dir = dataset_path / 'png' / 'buildings_complete'
    if complete_dir.exists():
        sample_files = list(complete_dir.glob('*.png'))
        # Use the first few samples that actually exist
        sample_ids = [f.stem for f in sample_files[:num_samples]]
        # Make sure we have valid samples
        sample_ids = [sid for sid in sample_ids if sid.isdigit()]
        if len(sample_ids) < num_samples:
            # Add some known samples
            known_samples = ['0', '100', '200']
            sample_ids = known_samples[:num_samples]
    else:
        sample_ids = ['0', '100', '200']
    
    print(f"\n🎯 Multiple Sample Comparison (Samples: {', '.join(sample_ids)})")
    print("=" * 60)
    
    fig, axes = plt.subplots(len(sample_ids), 4, figsize=(16, 4*len(sample_ids)))
    if len(sample_ids) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('RadioMapSeer Dataset - Multiple Sample Comparison', fontsize=16, fontweight='bold')
    
    scenarios = [
        ('buildings_complete', 'Complete Buildings'),
        ('cars', 'Cars Scenario'),
        ('roads', 'Roads Scenario'),
        ('antennas', 'Antenna Positions')
    ]
    
    for i, sample_id in enumerate(sample_ids):
        for j, (scenario, title) in enumerate(scenarios):
            ax = axes[i, j]
            
            if scenario == 'antennas':
                # Handle antenna positions
                antenna_path = dataset_path / 'antenna' / f'{sample_id}.json'
                if antenna_path.exists():
                    with open(antenna_path, 'r') as f:
                        antenna_positions = json.load(f)
                    x_coords = [pos[0] for pos in antenna_positions]
                    y_coords = [pos[1] for pos in antenna_positions]
                    ax.scatter(x_coords, y_coords, c='red', s=10, alpha=0.7)
                    ax.set_title(f'{title}\nSample {sample_id}')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.grid(True, alpha=0.3)
                    ax.set_aspect('equal')
                    print(f"✓ Found antenna positions for sample {sample_id}")
                else:
                    ax.text(0.5, 0.5, 'Not Found', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{title}\nSample {sample_id}')
                    print(f"✗ Missing antenna positions for sample {sample_id}")
            else:
                # Handle image files
                img_path = dataset_path / 'png' / scenario / f'{sample_id}.png'
                if img_path.exists():
                    img = Image.open(img_path)
                    ax.imshow(img, cmap='viridis')
                    ax.set_title(f'{title}\nSample {sample_id}')
                    print(f"✓ Found {scenario} for sample {sample_id}")
                else:
                    ax.text(0.5, 0.5, 'Not Found', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{title}\nSample {sample_id}')
                    print(f"✗ Missing {scenario} for sample {sample_id}")
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/cine/Documents/Github/RadioDiff/radiomapseer_multiple_samples_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Multiple samples comparison saved to radiomapseer_multiple_samples_fixed.png")

def find_available_samples(dataset_path, max_samples=10):
    """Find available samples that exist across different categories"""
    
    dataset_path = Path(dataset_path)
    
    # Get sample IDs from buildings_complete
    complete_dir = dataset_path / 'png' / 'buildings_complete'
    if complete_dir.exists():
        sample_files = list(complete_dir.glob('*.png'))
        sample_ids = [f.stem for f in sample_files if f.stem.isdigit()]
        
        print(f"\n📋 Available Sample IDs (first {min(max_samples, len(sample_ids))}):")
        print("=" * 50)
        
        for i, sample_id in enumerate(sample_ids[:max_samples]):
            print(f"{i+1:2d}. Sample {sample_id}")
            
            # Check what's available for this sample
            checks = []
            
            # Check basic files
            basic_files = [
                ('png/buildings_complete', f'{sample_id}.png'),
                ('png/cars', f'{sample_id}.png'),
                ('png/roads', f'{sample_id}.png'),
                ('antenna', f'{sample_id}.json'),
                ('polygon/buildings_and_cars', f'{sample_id}.json')
            ]
            
            for folder, filename in basic_files:
                path = dataset_path / folder / filename
                if path.exists():
                    checks.append(f"✓ {folder}/{filename}")
                else:
                    checks.append(f"✗ {folder}/{filename}")
            
            # Check missing buildings (first subdirectory)
            for level in [1, 2]:
                for subdir in ['1', '2']:
                    path = dataset_path / 'png' / f'buildings_missing{level}' / subdir / f'{sample_id}.png'
                    if path.exists():
                        checks.append(f"✓ buildings_missing{level}/{subdir}/{sample_id}.png")
                        break
            
            # Check gain patterns
            for gain_type in ['DPM', 'IRT2']:
                path = dataset_path / 'gain' / gain_type / f'{sample_id}_0.png'
                if path.exists():
                    checks.append(f"✓ gain/{gain_type}/{sample_id}_0.png")
            
            print("    " + "\n    ".join(checks))
            print()
            
            if i >= max_samples - 1:
                break

def main():
    """Main function"""
    dataset_path = "/home/cine/Documents/dataset/RadioMapSeer"
    
    print("🔍 Analyzing RadioMapSeer dataset structure and paired images...")
    
    # Find available samples first
    find_available_samples(dataset_path, max_samples=5)
    
    # Display paired images for sample 0 (should exist)
    display_paired_images_fixed(dataset_path, "0")
    
    # Show multiple samples
    show_multiple_samples_fixed(dataset_path, num_samples=3)
    
    print("\n🎉 Dataset structure and paired images analysis completed!")

if __name__ == "__main__":
    main()