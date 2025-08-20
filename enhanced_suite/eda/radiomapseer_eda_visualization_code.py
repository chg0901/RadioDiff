#!/usr/bin/env python3
"""
RadioMapSeer Dataset EDA Visualization Code
This code generates all the figures used in the comprehensive EDA report.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from collections import defaultdict
import cv2

# Set style
plt.style.use('default')
sns.set_palette("husl")

class RadioMapSeerEDA:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.csv_path = self.dataset_path / "dataset.csv"
        self.antenna_path = self.dataset_path / "antenna"
        self.gain_path = self.dataset_path / "gain"
        self.png_path = self.dataset_path / "png"
        self.polygon_path = self.dataset_path / "polygon"
        
        # Load dataset info
        self.df = pd.read_csv(self.csv_path)
        
    def analyze_dataset_structure(self):
        """Analyze and visualize dataset structure"""
        print("Analyzing dataset structure...")
        
        # Count files in each directory
        structure_data = {}
        
        # Main directories
        main_dirs = ['antenna', 'gain', 'png', 'polygon']
        for dir_name in main_dirs:
            dir_path = self.dataset_path / dir_name
            if dir_path.exists():
                if dir_name == 'antenna':
                    structure_data[dir_name] = {'files': len(list(dir_path.glob('*.json'))), 'subdirs': 0}
                elif dir_name == 'gain':
                    subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
                    total_files = sum(len(list(subdir.glob('*.png'))) for subdir in subdirs)
                    structure_data[dir_name] = {'files': total_files, 'subdirs': len(subdirs)}
                elif dir_name == 'png':
                    subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
                    total_files = sum(len(list(subdir.glob('*.png'))) for subdir in subdirs)
                    structure_data[dir_name] = {'files': total_files, 'subdirs': len(subdirs)}
                elif dir_name == 'polygon':
                    subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
                    total_files = sum(len(list(subdir.glob('*.json'))) for subdir in subdirs)
                    structure_data[dir_name] = {'files': total_files, 'subdirs': len(subdirs)}
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. File counts by main directory
        dirs = list(structure_data.keys())
        file_counts = [structure_data[d]['files'] for d in dirs]
        
        bars1 = ax1.bar(dirs, file_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_title('File Count by Main Directory', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Files')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars1, file_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(file_counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Subdirectory structure
        subdir_counts = [structure_data[d]['subdirs'] for d in dirs]
        bars2 = ax2.bar(dirs, subdir_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax2.set_title('Subdirectory Count by Main Directory', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Subdirectories')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars2, subdir_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(subdir_counts)*0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Detailed breakdown by subdirectory type
        gain_subdirs = ['DPM', 'IRT2', 'IRT4', 'carsDPM', 'carsIRT2', 'carsIRT4']
        gain_counts = []
        for subdir in gain_subdirs:
            subdir_path = self.gain_path / subdir
            if subdir_path.exists():
                gain_counts.append(len(list(subdir_path.glob('*.png'))))
            else:
                gain_counts.append(0)
        
        bars3 = ax3.bar(gain_subdirs, gain_counts, color='skyblue')
        ax3.set_title('Gain Pattern Files by Type', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Number of Files')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars3, gain_counts):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(gain_counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Hierarchical directory view
        ax4.text(0.1, 0.9, 'RadioMapSeer/', fontsize=16, fontweight='bold', transform=ax4.transAxes)
        
        y_pos = 0.8
        for dir_name in ['dataset.csv', 'antenna/', 'gain/', 'png/', 'polygon/']:
            ax4.text(0.15, y_pos, dir_name, fontsize=12, transform=ax4.transAxes)
            if dir_name.endswith('/'):
                files_text = f"{structure_data[dir_name[:-1]]['files']:,} files"
                subdirs_text = f"{structure_data[dir_name[:-1]]['subdirs']} subdirs"
                ax4.text(0.45, y_pos, files_text, fontsize=10, color='blue', transform=ax4.transAxes)
                ax4.text(0.65, y_pos, subdirs_text, fontsize=10, color='green', transform=ax4.transAxes)
            y_pos -= 0.08
        
        ax4.set_title('Hierarchical Directory Structure', fontsize=14, fontweight='bold')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig('dataset_structure_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return structure_data
    
    def analyze_antenna_configurations(self):
        """Analyze and visualize antenna configurations"""
        print("Analyzing antenna configurations...")
        
        # Sample antenna configurations for analysis
        sample_files = list(self.antenna_path.glob('*.json'))[:50]  # Analyze first 50
        
        all_positions = []
        antenna_counts = []
        
        for antenna_file in sample_files:
            try:
                with open(antenna_file, 'r') as f:
                    data = json.load(f)
                    positions = [(ant['position']['x'], ant['position']['y']) for ant in data['antennas']]
                    all_positions.extend(positions)
                    antenna_counts.append(len(data['antennas']))
            except Exception as e:
                print(f"Error reading {antenna_file}: {e}")
        
        all_positions = np.array(all_positions)
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Antenna count distribution
        ax1.hist(antenna_counts, bins=20, color='lightblue', edgecolor='black', alpha=0.7)
        ax1.set_title('Antenna Count Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Number of Antennas')
        ax1.set_ylabel('Frequency')
        ax1.axvline(np.mean(antenna_counts), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(antenna_counts):.1f}')
        ax1.legend()
        
        # 2. X-Y position scatter plot
        if len(all_positions) > 0:
            ax2.scatter(all_positions[:, 0], all_positions[:, 1], alpha=0.6, s=20)
            ax2.set_title('Antenna Positions (X-Y)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('X Position')
            ax2.set_ylabel('Y Position')
            ax2.grid(True, alpha=0.3)
        
        # 3. Position distributions
        if len(all_positions) > 0:
            ax3.hist(all_positions[:, 0], bins=30, alpha=0.7, label='X positions', color='blue')
            ax3.hist(all_positions[:, 1], bins=30, alpha=0.7, label='Y positions', color='red')
            ax3.set_title('Position Distribution', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Position Value')
            ax3.set_ylabel('Frequency')
            ax3.legend()
        
        # 4. 2D density heatmap
        if len(all_positions) > 0:
            hist, xedges, yedges = np.histogram2d(all_positions[:, 0], all_positions[:, 1], bins=20)
            im = ax4.imshow(hist.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                           cmap='hot', aspect='auto')
            ax4.set_title('Antenna Position Density Heatmap', fontsize=14, fontweight='bold')
            ax4.set_xlabel('X Position')
            ax4.set_ylabel('Y Position')
            plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        plt.savefig('antenna_configurations_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'total_positions': len(all_positions),
            'mean_antennas': np.mean(antenna_counts),
            'std_antennas': np.std(antenna_counts),
            'position_range': {
                'x': (np.min(all_positions[:, 0]) if len(all_positions) > 0 else 0,
                      np.max(all_positions[:, 0]) if len(all_positions) > 0 else 0),
                'y': (np.min(all_positions[:, 1]) if len(all_positions) > 0 else 0,
                      np.max(all_positions[:, 1]) if len(all_positions) > 0 else 0)
            }
        }
    
    def analyze_polygon_data(self):
        """Analyze and visualize polygon data"""
        print("Analyzing polygon data...")
        
        # Sample polygon files for analysis
        sample_files = list(self.polygon_path.glob('**/*.json'))[:50]  # Analyze first 50
        
        all_polygons = []
        polygon_counts = []
        vertex_counts = []
        
        for polygon_file in sample_files:
            try:
                with open(polygon_file, 'r') as f:
                    data = json.load(f)
                    polygons = data.get('polygons', [])
                    polygon_counts.append(len(polygons))
                    
                    for polygon in polygons:
                        vertices = polygon.get('vertices', [])
                        if vertices:
                            vertex_counts.append(len(vertices))
                            all_polygons.append(vertices)
            except Exception as e:
                print(f"Error reading {polygon_file}: {e}")
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Polygon count distribution
        ax1.hist(polygon_counts, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
        ax1.set_title('Polygon Count Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Number of Polygons')
        ax1.set_ylabel('Frequency')
        ax1.axvline(np.mean(polygon_counts), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(polygon_counts):.1f}')
        ax1.legend()
        
        # 2. Vertex count distribution
        ax2.hist(vertex_counts, bins=30, color='orange', edgecolor='black', alpha=0.7)
        ax2.set_title('Vertex Count Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Number of Vertices per Polygon')
        ax2.set_ylabel('Frequency')
        ax2.axvline(np.mean(vertex_counts), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(vertex_counts):.1f}')
        ax2.legend()
        
        # 3. Sample polygon visualization
        if all_polygons:
            sample_polygon = all_polygons[0]
            if sample_polygon:
                vertices = np.array(sample_polygon)
                ax3.plot(vertices[:, 0], vertices[:, 1], 'bo-', linewidth=2, markersize=6)
                ax3.fill(vertices[:, 0], vertices[:, 1], alpha=0.3, color='lightblue')
                ax3.set_title('Sample Polygon Visualization', fontsize=14, fontweight='bold')
                ax3.set_xlabel('X Coordinate')
                ax3.set_ylabel('Y Coordinate')
                ax3.grid(True, alpha=0.3)
                ax3.set_aspect('equal')
        
        # 4. Polygon statistics summary
        stats_text = f"""
        Polygon Statistics (50 samples):
        
        Total Polygons: {sum(polygon_counts):,}
        Mean Polygons/Sample: {np.mean(polygon_counts):.1f}
        Std Dev: {np.std(polygon_counts):.1f}
        Min-Max: {min(polygon_counts)}-{max(polygon_counts)}
        
        Vertex Statistics:
        Total Vertices: {sum(vertex_counts):,}
        Mean Vertices/Polygon: {np.mean(vertex_counts):.1f}
        Std Dev: {np.std(vertex_counts):.1f}
        Min-Max: {min(vertex_counts)}-{max(vertex_counts)}
        """
        
        ax4.text(0.1, 0.9, stats_text, fontsize=12, transform=ax4.transAxes, 
                verticalalignment='top', fontfamily='monospace')
        ax4.set_title('Polygon Statistics Summary', fontsize=14, fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig('polygon_data_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'total_polygons': sum(polygon_counts),
            'mean_polygons': np.mean(polygon_counts),
            'vertex_stats': {
                'total': sum(vertex_counts),
                'mean': np.mean(vertex_counts),
                'std': np.std(vertex_counts),
                'min': min(vertex_counts),
                'max': max(vertex_counts)
            }
        }
    
    def analyze_image_data(self):
        """Analyze and visualize image data"""
        print("Analyzing image data...")
        
        # Sample images for analysis
        sample_files = list(self.png_path.glob('**/*.png'))[:100]  # Analyze first 100
        
        image_shapes = []
        image_modes = []
        channel_stats = []
        
        for img_file in sample_files:
            try:
                with Image.open(img_file) as img:
                    image_shapes.append(img.size)
                    image_modes.append(img.mode)
                    
                    # Convert to numpy array for channel analysis
                    img_array = np.array(img)
                    if len(img_array.shape) == 2:  # Grayscale
                        channel_stats.append({
                            'channels': 1,
                            'mean': np.mean(img_array),
                            'std': np.std(img_array),
                            'min': np.min(img_array),
                            'max': np.max(img_array)
                        })
                    else:  # RGB or RGBA
                        for i in range(img_array.shape[2]):
                            channel_stats.append({
                                'channels': img_array.shape[2],
                                'mean': np.mean(img_array[:, :, i]),
                                'std': np.std(img_array[:, :, i]),
                                'min': np.min(img_array[:, :, i]),
                                'max': np.max(img_array[:, :, i])
                            })
            except Exception as e:
                print(f"Error reading {img_file}: {e}")
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Image dimensions distribution
        widths, heights = zip(*image_shapes) if image_shapes else ([], [])
        ax1.scatter(widths, heights, alpha=0.6, s=30)
        ax1.set_title('Image Dimensions Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Width (pixels)')
        ax1.set_ylabel('Height (pixels)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Image mode distribution
        mode_counts = pd.Series(image_modes).value_counts()
        bars2 = ax2.bar(mode_counts.index, mode_counts.values, color='lightcoral')
        ax2.set_title('Image Mode Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Image Mode')
        ax2.set_ylabel('Count')
        
        # Add value labels on bars
        for bar, count in zip(bars2, mode_counts.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mode_counts.values)*0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Channel statistics
        if channel_stats:
            means = [stat['mean'] for stat in channel_stats]
            stds = [stat['std'] for stat in channel_stats]
            
            ax3.scatter(means, stds, alpha=0.6, s=30)
            ax3.set_title('Channel Statistics (Mean vs Std)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Mean Intensity')
            ax3.set_ylabel('Standard Deviation')
            ax3.grid(True, alpha=0.3)
        
        # 4. Image statistics summary
        unique_shapes = list(set(image_shapes))
        unique_modes = list(set(image_modes))
        
        stats_text = f"""
        Image Statistics (100 samples):
        
        Dimensions:
        Unique Shapes: {len(unique_shapes)}
        Most Common: {max(set(image_shapes), key=image_shapes.count)}
        
        Modes:
        Unique Modes: {len(unique_modes)}
        Mode Distribution: {dict(pd.Series(image_modes).value_counts())}
        
        Channel Analysis:
        Total Channels Analyzed: {len(channel_stats)}
        Mean Intensity: {np.mean([s['mean'] for s in channel_stats]):.1f}
        Mean Std Dev: {np.mean([s['std'] for s in channel_stats]):.1f}
        """
        
        ax4.text(0.1, 0.9, stats_text, fontsize=12, transform=ax4.transAxes, 
                verticalalignment='top', fontfamily='monospace')
        ax4.set_title('Image Statistics Summary', fontsize=14, fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig('image_data_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'total_images': len(sample_files),
            'unique_shapes': len(unique_shapes),
            'unique_modes': len(unique_modes),
            'most_common_shape': max(set(image_shapes), key=image_shapes.count) if image_shapes else None,
            'channel_stats_summary': {
                'mean_intensity': np.mean([s['mean'] for s in channel_stats]) if channel_stats else 0,
                'mean_std': np.mean([s['std'] for s in channel_stats]) if channel_stats else 0
            }
        }
    
    def create_sample_visualizations(self):
        """Create sample image visualizations"""
        print("Creating sample visualizations...")
        
        # Get sample indices
        sample_indices = [0, 17, 508, 126]
        
        for sample_idx in sample_indices:
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle(f'RadioMapSeer Sample {sample_idx} - Comprehensive Data View', 
                        fontsize=16, fontweight='bold')
            
            # Radio maps
            radio_map_types = ['buildings_complete', 'cars', 'roads', 'antennas']
            for i, map_type in enumerate(radio_map_types):
                img_path = self.png_path / map_type / f"{sample_idx}.png"
                if img_path.exists():
                    img = Image.open(img_path)
                    axes[0, i].imshow(img, cmap='viridis')
                    axes[0, i].set_title(f'Radio Map: {map_type}')
                    axes[0, i].axis('off')
            
            # Gain patterns
            gain_types = ['DPM', 'IRT2', 'IRT4', 'carsDPM']
            for i, gain_type in enumerate(gain_types):
                img_path = self.gain_path / gain_type / f"{sample_idx}_0.png"  # Antenna 0
                if img_path.exists():
                    img = Image.open(img_path)
                    axes[1, i].imshow(img, cmap='plasma')
                    axes[1, i].set_title(f'Gain Pattern: {gain_type}')
                    axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'radiomapseer_sample_{sample_idx}.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def create_paired_visualization(self, sample_idx=0):
        """Create paired image visualization for a specific sample"""
        print(f"Creating paired visualization for sample {sample_idx}...")
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'RadioMapSeer Sample {sample_idx} - Paired Data Visualization', 
                    fontsize=16, fontweight='bold')
        
        # Row 1: Radio maps
        radio_map_types = ['buildings_complete', 'buildings_missing1', 'cars', 'roads']
        for i, map_type in enumerate(radio_map_types):
            img_path = self.png_path / map_type / f"{sample_idx}.png"
            if img_path.exists():
                img = Image.open(img_path)
                axes[0, i].imshow(img, cmap='viridis')
                axes[0, i].set_title(f'Radio Map: {map_type}')
                axes[0, i].axis('off')
        
        # Row 2: Gain patterns
        gain_types = ['DPM', 'IRT2', 'IRT4', 'carsDPM']
        for i, gain_type in enumerate(gain_types):
            img_path = self.gain_path / gain_type / f"{sample_idx}_0.png"
            if img_path.exists():
                img = Image.open(img_path)
                axes[1, i].imshow(img, cmap='plasma')
                axes[1, i].set_title(f'Gain Pattern: {gain_type}')
                axes[1, i].axis('off')
        
        # Row 3: Environmental data
        # Load and visualize polygon data
        polygon_path = self.polygon_path / 'buildings_and_cars' / f"{sample_idx}.json"
        if polygon_path.exists():
            with open(polygon_path, 'r') as f:
                data = json.load(f)
                polygons = data.get('polygons', [])
                
                # Plot polygons
                for j, polygon in enumerate(polygons[:20]):  # Show first 20 polygons
                    vertices = polygon.get('vertices', [])
                    if vertices and len(vertices) > 2:
                        vertices_array = np.array(vertices)
                        axes[2, 0].plot(vertices_array[:, 0], vertices_array[:, 1], 'b-', alpha=0.6)
                        axes[2, 0].fill(vertices_array[:, 0], vertices_array[:, 1], alpha=0.3, color='lightblue')
                
                axes[2, 0].set_title('Building/Car Polygons')
                axes[2, 0].set_xlabel('X Coordinate')
                axes[2, 0].set_ylabel('Y Coordinate')
                axes[2, 0].grid(True, alpha=0.3)
                axes[2, 0].set_aspect('equal')
        
        # Load antenna configuration
        antenna_path = self.antenna_path / f"{sample_idx}.json"
        if antenna_path.exists():
            with open(antenna_path, 'r') as f:
                data = json.load(f)
                antennas = data.get('antennas', [])
                
                # Plot antenna positions
                x_pos = [ant['position']['x'] for ant in antennas]
                y_pos = [ant['position']['y'] for ant in antennas]
                axes[2, 1].scatter(x_pos, y_pos, c='red', s=20, alpha=0.7)
                axes[2, 1].set_title('Antenna Positions')
                axes[2, 1].set_xlabel('X Position')
                axes[2, 1].set_ylabel('Y Position')
                axes[2, 1].grid(True, alpha=0.3)
                axes[2, 1].set_aspect('equal')
        
        # Hide empty subplots
        for i in range(2, 4):
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'radiomapseer_paired_images_{sample_idx}.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to run all EDA visualizations"""
    dataset_path = "/home/cine/Documents/dataset/RadioMapSeer"
    
    print("Starting RadioMapSeer EDA Visualization...")
    print(f"Dataset path: {dataset_path}")
    
    # Initialize EDA object
    eda = RadioMapSeerEDA(dataset_path)
    
    # Run all analyses
    structure_results = eda.analyze_dataset_structure()
    antenna_results = eda.analyze_antenna_configurations()
    polygon_results = eda.analyze_polygon_data()
    image_results = eda.analyze_image_data()
    
    # Create sample visualizations
    eda.create_sample_visualizations()
    eda.create_paired_visualization(sample_idx=0)
    
    # Print summary
    print("\n" + "="*50)
    print("EDA Visualization Complete!")
    print("="*50)
    print(f"Dataset Structure: {structure_results}")
    print(f"Antenna Configurations: {antenna_results}")
    print(f"Polygon Data: {polygon_results}")
    print(f"Image Data: {image_results}")
    print("="*50)

if __name__ == "__main__":
    main()