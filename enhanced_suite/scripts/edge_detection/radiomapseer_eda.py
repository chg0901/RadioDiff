#!/usr/bin/env python3
"""
Comprehensive Exploratory Data Analysis (EDA) for RadioMapSeer Dataset
This script analyzes the RadioMapSeer dataset structure and generates visualizations.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RadioMapSeerEDA:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.results = {}
        
    def analyze_dataset_structure(self):
        """Analyze the overall structure of the dataset"""
        print("🔍 Analyzing dataset structure...")
        
        structure = {
            'total_samples': 0,
            'file_counts': {},
            'directory_structure': {}
        }
        
        # Count files in each directory
        for root, dirs, files in os.walk(self.dataset_path):
            relative_path = os.path.relpath(root, self.dataset_path)
            if relative_path == '.':
                relative_path = 'root'
            
            structure['directory_structure'][relative_path] = {
                'subdirectories': dirs,
                'file_count': len(files),
                'file_types': defaultdict(int)
            }
            
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                structure['directory_structure'][relative_path]['file_types'][ext] += 1
        
        # Read CSV file to get sample information
        csv_path = self.dataset_path / 'dataset.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            structure['total_samples'] = len(df)
            structure['csv_columns'] = list(df.columns)
            structure['csv_sample'] = df.head(5).to_dict()
            
            # Count different types of data
            structure['data_types'] = {
                'maps': df['maps'].notna().sum(),
                'antenna_configs': df['json'].notna().sum(),
                'gain_images': sum([col.startswith('Gain') for col in df.columns])
            }
        
        self.results['structure'] = structure
        return structure
    
    def analyze_antenna_configurations(self):
        """Analyze antenna configuration data"""
        print("📡 Analyzing antenna configurations...")
        
        antenna_dir = self.dataset_path / 'antenna'
        if not antenna_dir.exists():
            return
        
        antenna_stats = {
            'total_configs': 0,
            'antenna_counts': [],
            'position_stats': defaultdict(list),
            'sample_configs': []
        }
        
        json_files = list(antenna_dir.glob('*.json'))
        antenna_stats['total_configs'] = len(json_files)
        
        # Analyze first 100 configurations for patterns
        for i, json_file in enumerate(json_files[:100]):
            try:
                with open(json_file, 'r') as f:
                    config = json.load(f)
                
                antenna_count = len(config)
                antenna_stats['antenna_counts'].append(antenna_count)
                
                # Extract position statistics
                for pos in config:
                    antenna_stats['position_stats']['x'].append(pos[0])
                    antenna_stats['position_stats']['y'].append(pos[1])
                
                if i < 5:  # Save sample configurations
                    antenna_stats['sample_configs'].append({
                        'file': json_file.name,
                        'antenna_count': antenna_count,
                        'positions': config[:3]  # First 3 positions
                    })
                    
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
        
        # Calculate statistics
        if antenna_stats['antenna_counts']:
            antenna_stats['stats'] = {
                'mean_antennas': np.mean(antenna_stats['antenna_counts']),
                'std_antennas': np.std(antenna_stats['antenna_counts']),
                'min_antennas': min(antenna_stats['antenna_counts']),
                'max_antennas': max(antenna_stats['antenna_counts'])
            }
            
            # Position statistics
            for axis in ['x', 'y']:
                if antenna_stats['position_stats'][axis]:
                    antenna_stats['position_stats'][f'{axis}_mean'] = np.mean(antenna_stats['position_stats'][axis])
                    antenna_stats['position_stats'][f'{axis}_std'] = np.std(antenna_stats['position_stats'][axis])
        
        self.results['antenna'] = antenna_stats
        return antenna_stats
    
    def analyze_polygon_data(self):
        """Analyze polygon (building/car) data"""
        print("🏢 Analyzing polygon data...")
        
        polygon_dir = self.dataset_path / 'polygon' / 'buildings_and_cars'
        if not polygon_dir.exists():
            return
        
        polygon_stats = {
            'total_files': 0,
            'sample_data': [],
            'object_counts': [],
            'shape_stats': defaultdict(list)
        }
        
        json_files = list(polygon_dir.glob('*.json'))
        polygon_stats['total_files'] = len(json_files)
        
        # Analyze first 50 files
        for i, json_file in enumerate(json_files[:50]):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, dict):
                    object_count = len(data)
                    polygon_stats['object_counts'].append(object_count)
                    
                    # Analyze shapes
                    for obj_key, obj_data in data.items():
                        if isinstance(obj_data, list) and len(obj_data) > 0:
                            polygon_stats['shape_stats']['vertices'].append(len(obj_data))
                            
                            # Extract coordinates
                            for vertex in obj_data:
                                if len(vertex) >= 2:
                                    polygon_stats['shape_stats']['x_coords'].append(vertex[0])
                                    polygon_stats['shape_stats']['y_coords'].append(vertex[1])
                    
                    if i < 3:  # Save sample data
                        polygon_stats['sample_data'].append({
                            'file': json_file.name,
                            'object_count': object_count,
                            'sample_objects': dict(list(data.items())[:2])
                        })
                        
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
        
        # Calculate statistics
        if polygon_stats['object_counts']:
            polygon_stats['stats'] = {
                'mean_objects': np.mean(polygon_stats['object_counts']),
                'std_objects': np.std(polygon_stats['object_counts']),
                'min_objects': min(polygon_stats['object_counts']),
                'max_objects': max(polygon_stats['object_counts'])
            }
        
        self.results['polygon'] = polygon_stats
        return polygon_stats
    
    def analyze_image_data(self):
        """Analyze image data (radio maps, gain patterns, etc.)"""
        print("🖼️ Analyzing image data...")
        
        image_stats = {
            'directories': {},
            'image_properties': defaultdict(list),
            'sample_images': []
        }
        
        # Analyze different image directories
        image_dirs = ['png', 'gain']
        for dir_name in image_dirs:
            dir_path = self.dataset_path / dir_name
            if dir_path.exists():
                image_stats['directories'][dir_name] = {
                    'total_images': 0,
                    'subdirs': {},
                    'sample_info': []
                }
                
                # Count images in subdirectories
                for subdir in dir_path.iterdir():
                    if subdir.is_dir():
                        png_files = list(subdir.glob('*.png'))
                        image_stats['directories'][dir_name]['subdirs'][subdir.name] = len(png_files)
                        image_stats['directories'][dir_name]['total_images'] += len(png_files)
                        
                        # Analyze sample images
                        if png_files and len(image_stats['directories'][dir_name]['sample_info']) < 5:
                            for img_file in png_files[:3]:
                                try:
                                    img = Image.open(img_file)
                                    img_array = np.array(img)
                                    
                                    image_stats['image_properties']['width'].append(img_array.shape[1])
                                    image_stats['image_properties']['height'].append(img_array.shape[0])
                                    image_stats['image_properties']['channels'].append(img_array.shape[2] if len(img_array.shape) > 2 else 1)
                                    image_stats['image_properties']['dtype'].append(str(img_array.dtype))
                                    
                                    # Basic statistics
                                    if len(img_array.shape) == 2:
                                        image_stats['image_properties']['mean_intensity'].append(np.mean(img_array))
                                        image_stats['image_properties']['std_intensity'].append(np.std(img_array))
                                    
                                    image_stats['directories'][dir_name]['sample_info'].append({
                                        'file': img_file.name,
                                        'shape': img_array.shape,
                                        'dtype': str(img_array.dtype)
                                    })
                                    
                                except Exception as e:
                                    print(f"Error reading image {img_file}: {e}")
        
        # Calculate image statistics
        for prop in ['width', 'height', 'channels']:
            if image_stats['image_properties'][prop]:
                image_stats[f'{prop}_stats'] = {
                    'mean': np.mean(image_stats['image_properties'][prop]),
                    'std': np.std(image_stats['image_properties'][prop]),
                    'min': min(image_stats['image_properties'][prop]),
                    'max': max(image_stats['image_properties'][prop])
                }
        
        self.results['images'] = image_stats
        return image_stats
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        print("📊 Generating visualizations...")
        
        # Create output directory
        viz_dir = self.dataset_path / 'eda_visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Dataset Structure Visualization
        if 'structure' in self.results:
            self._plot_dataset_structure(viz_dir)
        
        # 2. Antenna Configuration Visualizations
        if 'antenna' in self.results:
            self._plot_antenna_configs(viz_dir)
        
        # 3. Polygon Data Visualizations
        if 'polygon' in self.results:
            self._plot_polygon_data(viz_dir)
        
        # 4. Image Data Visualizations
        if 'images' in self.results:
            self._plot_image_data(viz_dir)
        
        print(f"✅ Visualizations saved to {viz_dir}")
        return viz_dir
    
    def _plot_dataset_structure(self, viz_dir):
        """Plot dataset structure overview"""
        structure = self.results['structure']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RadioMapSeer Dataset Structure Analysis', fontsize=16, fontweight='bold')
        
        # 1. Data types distribution
        if 'data_types' in structure:
            data_types = structure['data_types']
            ax = axes[0, 0]
            ax.pie(data_types.values(), labels=data_types.keys(), autopct='%1.1f%%')
            ax.set_title('Data Types Distribution')
        
        # 2. Directory file counts
        if 'directory_structure' in structure:
            dir_counts = {k: v['file_count'] for k, v in structure['directory_structure'].items() if v['file_count'] > 0}
            ax = axes[0, 1]
            if dir_counts:
                ax.bar(range(len(dir_counts)), list(dir_counts.values()))
                ax.set_xticks(range(len(dir_counts)))
                ax.set_xticklabels(list(dir_counts.keys()), rotation=45, ha='right')
                ax.set_title('Files per Directory')
                ax.set_ylabel('Number of Files')
        
        # 3. Sample overview
        ax = axes[1, 0]
        sample_info = [
            f"Total Samples: {structure.get('total_samples', 'N/A')}",
            f"Total PNG Files: {sum(v['file_count'] for v in structure['directory_structure'].values())}",
            f"Total JSON Files: {sum(v['file_types'].get('.json', 0) for v in structure['directory_structure'].values())}",
            f"CSV Columns: {len(structure.get('csv_columns', []))}"
        ]
        ax.text(0.1, 0.5, '\n'.join(sample_info), transform=ax.transAxes, fontsize=12, 
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax.set_title('Dataset Overview')
        ax.axis('off')
        
        # 4. Directory structure tree visualization
        ax = axes[1, 1]
        if 'directory_structure' in structure:
            # Create a hierarchical view of main directories
            main_dirs = {}
            for dir_name, dir_info in structure['directory_structure'].items():
                if dir_name != 'root' and dir_info['file_count'] > 0:
                    # Extract top-level directory
                    top_dir = dir_name.split('/')[0] if '/' in dir_name else dir_name
                    if top_dir not in main_dirs:
                        main_dirs[top_dir] = {'total_files': 0, 'subdirs': []}
                    main_dirs[top_dir]['total_files'] += dir_info['file_count']
                    if '/' in dir_name:
                        main_dirs[top_dir]['subdirs'].append(dir_name)
            
            # Create a bar chart with subdirectory breakdown
            if main_dirs:
                dir_names = list(main_dirs.keys())
                dir_counts = [main_dirs[dir_name]['total_files'] for dir_name in dir_names]
                subdir_counts = [len(main_dirs[dir_name]['subdirs']) for dir_name in dir_names]
                
                # Create stacked bar chart
                y_pos = range(len(dir_names))
                bars = ax.barh(y_pos, dir_counts, alpha=0.7, color='lightblue', edgecolor='black')
                
                # Add value labels on bars
                for i, (bar, count, subcount) in enumerate(zip(bars, dir_counts, subdir_counts)):
                    ax.text(bar.get_width() + max(dir_counts) * 0.01, bar.get_y() + bar.get_height()/2,
                           f'{count}\n({subcount} subs)', va='center', ha='left', fontsize=9)
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(dir_names)
                ax.set_xlabel('Total Files')
                ax.set_title('Directory Structure (Files + Subdirectories)')
                ax.grid(True, alpha=0.3, axis='x')
                
                # Add summary text
                total_files = sum(dir_counts)
                total_subdirs = sum(subdir_counts)
                ax.text(0.02, 0.98, f'Total: {total_files} files\nin {total_subdirs} subdirectories', 
                       transform=ax.transAxes, fontsize=10, va='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'dataset_structure.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_antenna_configs(self, viz_dir):
        """Plot antenna configuration analysis"""
        antenna = self.results['antenna']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Antenna Configuration Analysis', fontsize=16, fontweight='bold')
        
        # 1. Antenna count distribution
        ax = axes[0, 0]
        if antenna['antenna_counts']:
            ax.hist(antenna['antenna_counts'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('Number of Antennas')
            ax.set_ylabel('Frequency')
            ax.set_title('Antenna Count Distribution')
            ax.grid(True, alpha=0.3)
        
        # 2. Position scatter plot
        ax = axes[0, 1]
        if antenna['position_stats']['x'] and antenna['position_stats']['y']:
            ax.scatter(antenna['position_stats']['x'], antenna['position_stats']['y'], 
                      alpha=0.6, s=20, color='red')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title('Antenna Position Distribution')
            ax.grid(True, alpha=0.3)
        
        # 3. Statistics summary
        ax = axes[1, 0]
        if 'stats' in antenna:
            stats_text = []
            for key, value in antenna['stats'].items():
                stats_text.append(f"{key.replace('_', ' ').title()}: {value:.2f}")
            
            ax.text(0.1, 0.5, '\n'.join(stats_text), transform=ax.transAxes, fontsize=12,
                    verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            ax.set_title('Antenna Configuration Statistics')
            ax.axis('off')
        
        # 4. Antenna position density heatmap
        ax = axes[1, 1]
        if antenna['position_stats']['x'] and antenna['position_stats']['y']:
            x_pos = antenna['position_stats']['x']
            y_pos = antenna['position_stats']['y']
            
            # Create 2D histogram for density visualization
            heatmap = ax.hist2d(x_pos, y_pos, bins=20, cmap='YlOrRd', alpha=0.8, 
                               edgecolors='black', linewidths=0.5)
            
            # Add colorbar
            cbar = plt.colorbar(heatmap[3], ax=ax, shrink=0.8)
            cbar.set_label('Antenna Density', rotation=270, labelpad=15)
            
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title('Antenna Position Density Heatmap')
            
            # Add statistics overlay
            if 'position_stats' in antenna:
                stats_text = [
                    f"X: {antenna['position_stats']['x_mean']:.1f} ± {antenna['position_stats']['x_std']:.1f}",
                    f"Y: {antenna['position_stats']['y_mean']:.1f} ± {antenna['position_stats']['y_std']:.1f}",
                    f"Total: {len(x_pos)} antennas"
                ]
                ax.text(0.02, 0.98, '\n'.join(stats_text), transform=ax.transAxes, 
                       fontsize=9, va='top', ha='left',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Set equal aspect ratio for better spatial representation
            ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'antenna_configurations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_polygon_data(self, viz_dir):
        """Plot polygon data analysis"""
        polygon = self.results['polygon']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Polygon (Building/Car) Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Object count distribution
        ax = axes[0, 0]
        if polygon['object_counts']:
            ax.hist(polygon['object_counts'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            ax.set_xlabel('Number of Objects')
            ax.set_ylabel('Frequency')
            ax.set_title('Object Count Distribution')
            ax.grid(True, alpha=0.3)
        
        # 2. Vertex count distribution
        ax = axes[0, 1]
        if polygon['shape_stats']['vertices']:
            ax.hist(polygon['shape_stats']['vertices'], bins=20, alpha=0.7, color='gold', edgecolor='black')
            ax.set_xlabel('Number of Vertices')
            ax.set_ylabel('Frequency')
            ax.set_title('Polygon Vertex Count Distribution')
            ax.grid(True, alpha=0.3)
        
        # 3. Statistics summary
        ax = axes[1, 0]
        if 'stats' in polygon:
            stats_text = []
            for key, value in polygon['stats'].items():
                stats_text.append(f"{key.replace('_', ' ').title()}: {value:.2f}")
            
            ax.text(0.1, 0.5, '\n'.join(stats_text), transform=ax.transAxes, fontsize=12,
                    verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            ax.set_title('Polygon Data Statistics')
            ax.axis('off')
        
        # 4. Coordinate scatter plot
        ax = axes[1, 1]
        if polygon['shape_stats']['x_coords'] and polygon['shape_stats']['y_coords']:
            ax.scatter(polygon['shape_stats']['x_coords'], polygon['shape_stats']['y_coords'], 
                      alpha=0.3, s=10, color='purple')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_title('Polygon Vertex Positions')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'polygon_data.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_image_data(self, viz_dir):
        """Plot image data analysis"""
        images = self.results['images']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Image Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Directory image counts
        ax = axes[0, 0]
        if images['directories']:
            dir_names = []
            dir_counts = []
            for dir_name, dir_info in images['directories'].items():
                for subdir, count in dir_info['subdirs'].items():
                    dir_names.append(f"{dir_name}/{subdir}")
                    dir_counts.append(count)
            
            if dir_names:
                ax.barh(range(len(dir_names)), dir_counts)
                ax.set_yticks(range(len(dir_names)))
                ax.set_yticklabels(dir_names)
                ax.set_xlabel('Number of Images')
                ax.set_title('Images per Directory')
                ax.grid(True, alpha=0.3)
        
        # 2. Image dimensions
        ax = axes[0, 1]
        if images['image_properties']['width'] and images['image_properties']['height']:
            ax.scatter(images['image_properties']['width'], images['image_properties']['height'], 
                      alpha=0.6, s=30, color='green')
            ax.set_xlabel('Width (pixels)')
            ax.set_ylabel('Height (pixels)')
            ax.set_title('Image Dimensions')
            ax.grid(True, alpha=0.3)
        
        # 3. Channel distribution
        ax = axes[1, 0]
        if images['image_properties']['channels']:
            channel_counts = pd.Series(images['image_properties']['channels']).value_counts()
            ax.pie(channel_counts.values, labels=[f"{k} channels" for k in channel_counts.index], 
                   autopct='%1.1f%%')
            ax.set_title('Image Channel Distribution')
        
        # 4. Intensity distribution
        ax = axes[1, 1]
        if images['image_properties']['mean_intensity']:
            ax.hist(images['image_properties']['mean_intensity'], bins=30, alpha=0.7, 
                   color='cyan', edgecolor='black')
            ax.set_xlabel('Mean Intensity')
            ax.set_ylabel('Frequency')
            ax.set_title('Image Intensity Distribution')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'image_data.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self):
        """Generate comprehensive EDA report"""
        print("📝 Generating comprehensive EDA report...")
        
        report_path = self.dataset_path / 'RadioMapSeer_EDA_Report.md'
        
        with open(report_path, 'w') as f:
            f.write("# RadioMapSeer Dataset - Exploratory Data Analysis Report\n\n")
            f.write("## Overview\n")
            f.write("This report provides a comprehensive analysis of the RadioMapSeer dataset, ")
            f.write("which contains radio map data with antenna configurations, building/car polygons, ")
            f.write("and various image representations.\n\n")
            
            # Dataset Structure
            if 'structure' in self.results:
                f.write("## Dataset Structure\n\n")
                structure = self.results['structure']
                f.write(f"- **Total Samples**: {structure.get('total_samples', 'N/A')}\n")
                f.write(f"- **CSV Columns**: {len(structure.get('csv_columns', []))}\n")
                f.write(f"- **Main Directories**: {list(structure.get('directory_structure', {}).keys())}\n\n")
                
                if 'data_types' in structure:
                    f.write("### Data Types Distribution\n")
                    for data_type, count in structure['data_types'].items():
                        f.write(f"- **{data_type}**: {count}\n")
                    f.write("\n")
            
            # Antenna Configurations
            if 'antenna' in self.results:
                f.write("## Antenna Configuration Analysis\n\n")
                antenna = self.results['antenna']
                f.write(f"- **Total Configurations**: {antenna.get('total_configs', 'N/A')}\n")
                
                if 'stats' in antenna:
                    f.write("### Configuration Statistics\n")
                    for key, value in antenna['stats'].items():
                        f.write(f"- **{key.replace('_', ' ').title()}**: {value:.2f}\n")
                    f.write("\n")
            
            # Polygon Data
            if 'polygon' in self.results:
                f.write("## Polygon (Building/Car) Data Analysis\n\n")
                polygon = self.results['polygon']
                f.write(f"- **Total Files**: {polygon.get('total_files', 'N/A')}\n")
                
                if 'stats' in polygon:
                    f.write("### Object Statistics\n")
                    for key, value in polygon['stats'].items():
                        f.write(f"- **{key.replace('_', ' ').title()}**: {value:.2f}\n")
                    f.write("\n")
            
            # Image Data
            if 'images' in self.results:
                f.write("## Image Data Analysis\n\n")
                images = self.results['images']
                
                total_images = sum(dir_info['total_images'] for dir_info in images['directories'].values())
                f.write(f"- **Total Images**: {total_images}\n")
                
                if 'width_stats' in images:
                    f.write("### Image Dimensions\n")
                    f.write(f"- **Mean Width**: {images['width_stats']['mean']:.1f} pixels\n")
                    f.write(f"- **Mean Height**: {images['height_stats']['mean']:.1f} pixels\n")
                    f.write(f"- **Width Range**: {images['width_stats']['min']} - {images['width_stats']['max']} pixels\n")
                    f.write(f"- **Height Range**: {images['height_stats']['min']} - {images['height_stats']['max']} pixels\n\n")
            
            # Conclusions
            f.write("## Key Findings\n\n")
            f.write("1. **Dataset Size**: Large-scale dataset with hundreds of thousands of images\n")
            f.write("2. **Data Diversity**: Contains multiple types of data (radio maps, antenna configs, polygons)\n")
            f.write("3. **Image Characteristics**: Various image dimensions and formats\n")
            f.write("4. **Antenna Configurations**: Diverse antenna placement patterns\n")
            f.write("5. **Environment Data**: Detailed building and car polygon information\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("1. **Data Preprocessing**: Consider standardizing image dimensions\n")
            f.write("2. **Feature Engineering**: Leverage antenna position patterns\n")
            f.write("3. **Model Selection**: Suitable for computer vision and spatial analysis tasks\n")
            f.write("4. **Validation Strategy**: Use diverse samples from different configurations\n")
        
        print(f"✅ Report saved to {report_path}")
        return report_path
    
    def run_full_eda(self):
        """Run complete EDA analysis"""
        print("🚀 Starting comprehensive EDA of RadioMapSeer dataset...")
        
        # Run all analyses
        self.analyze_dataset_structure()
        self.analyze_antenna_configurations()
        self.analyze_polygon_data()
        self.analyze_image_data()
        
        # Generate visualizations
        viz_dir = self.generate_visualizations()
        
        # Generate report
        report_path = self.generate_report()
        
        print("🎉 EDA completed successfully!")
        print(f"📊 Visualizations: {viz_dir}")
        print(f"📝 Report: {report_path}")
        
        return {
            'results': self.results,
            'visualizations': viz_dir,
            'report': report_path
        }

def main():
    """Main function to run EDA"""
    dataset_path = "/home/cine/Documents/dataset/RadioMapSeer"
    
    # Initialize EDA
    eda = RadioMapSeerEDA(dataset_path)
    
    # Run full analysis
    results = eda.run_full_eda()
    
    return results

if __name__ == "__main__":
    main()