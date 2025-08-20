#!/usr/bin/env python3
"""
Improved EDA for RadioMapSeer Dataset with meaningful visualizations
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ImprovedRadioMapSeerEDA:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.results = {}
        
    def analyze_dataset_structure(self):
        """Analyze the overall structure of the dataset"""
        print("🔍 Analyzing dataset structure...")
        
        structure = {
            'total_samples': 0,
            'file_counts': {},
            'directory_structure': {},
            'data_categories': {}
        }
        
        # Read CSV file to get sample information
        csv_path = self.dataset_path / 'dataset.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            structure['total_samples'] = len(df)
            structure['csv_columns'] = list(df.columns)
            structure['csv_sample'] = df.head(3).to_dict()
            
            # Analyze data completeness
            structure['data_completeness'] = {
                'maps': df['maps'].notna().sum(),
                'antenna_configs': df['json'].notna().sum(),
                'gain_columns': sum([col.startswith('Gain') for col in df.columns])
            }
        
        # Analyze directory structure with actual counts
        structure['detailed_structure'] = {}
        
        # Main categories
        main_categories = ['png', 'gain', 'polygon', 'antenna']
        for category in main_categories:
            category_path = self.dataset_path / category
            if category_path.exists():
                structure['detailed_structure'][category] = self._analyze_category(category_path)
        
        self.results['structure'] = structure
        return structure
    
    def _analyze_category(self, category_path):
        """Analyze a specific category directory"""
        category_info = {
            'total_files': 0,
            'subdirectories': {},
            'file_types': Counter()
        }
        
        for root, dirs, files in os.walk(category_path):
            relative_path = os.path.relpath(root, category_path)
            if relative_path == '.':
                relative_path = 'root'
            
            file_count = len(files)
            category_info['total_files'] += file_count
            
            if file_count > 0:
                category_info['subdirectories'][relative_path] = file_count
                
                # Count file types
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    category_info['file_types'][ext] += 1
        
        return category_info
    
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
            'sample_configs': [],
            'spatial_analysis': {}
        }
        
        json_files = list(antenna_dir.glob('*.json'))
        antenna_stats['total_configs'] = len(json_files)
        
        # Analyze all configurations for comprehensive statistics
        for i, json_file in enumerate(json_files):
            try:
                with open(json_file, 'r') as f:
                    config = json.load(f)
                
                antenna_count = len(config)
                antenna_stats['antenna_counts'].append(antenna_count)
                
                # Extract position statistics
                for pos in config:
                    antenna_stats['position_stats']['x'].append(pos[0])
                    antenna_stats['position_stats']['y'].append(pos[1])
                
                # Save sample configurations
                if i < 3:
                    antenna_stats['sample_configs'].append({
                        'file': json_file.name,
                        'antenna_count': antenna_count,
                        'positions': config[:5]  # First 5 positions
                    })
                    
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
        
        # Calculate comprehensive statistics
        if antenna_stats['antenna_counts']:
            antenna_stats['stats'] = {
                'mean_antennas': np.mean(antenna_stats['antenna_counts']),
                'std_antennas': np.std(antenna_stats['antenna_counts']),
                'min_antennas': min(antenna_stats['antenna_counts']),
                'max_antennas': max(antenna_stats['antenna_counts']),
                'median_antennas': np.median(antenna_stats['antenna_counts'])
            }
            
            # Position statistics
            for axis in ['x', 'y']:
                if antenna_stats['position_stats'][axis]:
                    positions = antenna_stats['position_stats'][axis]
                    antenna_stats['position_stats'][f'{axis}_mean'] = np.mean(positions)
                    antenna_stats['position_stats'][f'{axis}_std'] = np.std(positions)
                    antenna_stats['position_stats'][f'{axis}_min'] = min(positions)
                    antenna_stats['position_stats'][f'{axis}_max'] = max(positions)
                    antenna_stats['position_stats'][f'{axis}_median'] = np.median(positions)
        
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
            'shape_stats': defaultdict(list),
            'complexity_analysis': {}
        }
        
        json_files = list(polygon_dir.glob('*.json'))
        polygon_stats['total_files'] = len(json_files)
        
        # Analyze all files for comprehensive statistics
        for i, json_file in enumerate(json_files):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Fixed: data is a list of polygon objects
                if isinstance(data, list):
                    object_count = len(data)
                    polygon_stats['object_counts'].append(object_count)
                    
                    # Analyze shapes - each object is a list of vertices
                    vertex_counts = []
                    for obj_data in data:
                        if isinstance(obj_data, list) and len(obj_data) > 0:
                            vertex_count = len(obj_data)
                            vertex_counts.append(vertex_count)
                            polygon_stats['shape_stats']['vertices'].append(vertex_count)
                            
                            # Extract coordinates
                            for vertex in obj_data:
                                if len(vertex) >= 2:
                                    polygon_stats['shape_stats']['x_coords'].append(vertex[0])
                                    polygon_stats['shape_stats']['y_coords'].append(vertex[1])
                    
                    # Complexity analysis
                    if vertex_counts:
                        polygon_stats['complexity_analysis'][f'sample_{i}'] = {
                            'total_objects': object_count,
                            'mean_vertices': np.mean(vertex_counts),
                            'max_vertices': max(vertex_counts),
                            'min_vertices': min(vertex_counts),
                            'simple_objects': len([v for v in vertex_counts if v <= 6]),
                            'complex_objects': len([v for v in vertex_counts if v > 20])
                        }
                    
                    if i < 3:  # Save sample data
                        polygon_stats['sample_data'].append({
                            'file': json_file.name,
                            'object_count': object_count,
                            'sample_objects': data[:3]  # First 3 objects
                        })
                        
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
        
        # Calculate statistics
        if polygon_stats['object_counts']:
            polygon_stats['stats'] = {
                'mean_objects': np.mean(polygon_stats['object_counts']),
                'std_objects': np.std(polygon_stats['object_counts']),
                'min_objects': min(polygon_stats['object_counts']),
                'max_objects': max(polygon_stats['object_counts']),
                'median_objects': np.median(polygon_stats['object_counts'])
            }
        
        if polygon_stats['shape_stats']['vertices']:
            vertices = polygon_stats['shape_stats']['vertices']
            polygon_stats['vertex_stats'] = {
                'mean_vertices': np.mean(vertices),
                'std_vertices': np.std(vertices),
                'min_vertices': min(vertices),
                'max_vertices': max(vertices),
                'median_vertices': np.median(vertices)
            }
        
        self.results['polygon'] = polygon_stats
        return polygon_stats
    
    def analyze_image_data(self):
        """Analyze image data (radio maps, gain patterns, etc.)"""
        print("🖼️ Analyzing image data...")
        
        image_stats = {
            'categories': {},
            'image_properties': defaultdict(list),
            'sample_analysis': {}
        }
        
        # Analyze different image categories
        categories = ['png', 'gain']
        for category in categories:
            category_path = self.dataset_path / category
            if category_path.exists():
                image_stats['categories'][category] = self._analyze_image_category(category_path)
        
        self.results['images'] = image_stats
        return image_stats
    
    def _analyze_image_category(self, category_path):
        """Analyze images in a specific category"""
        category_info = {
            'total_images': 0,
            'subdirectories': {},
            'image_dimensions': defaultdict(list),
            'sample_images': []
        }
        
        # Sample some images for analysis
        sample_count = 0
        max_samples = 50
        
        for root, dirs, files in os.walk(category_path):
            relative_path = os.path.relpath(root, category_path)
            if relative_path == '.':
                relative_path = 'root'
            
            png_files = [f for f in files if f.endswith('.png')]
            category_info['total_images'] += len(png_files)
            category_info['subdirectories'][relative_path] = len(png_files)
            
            # Analyze sample images
            for img_file in png_files:
                if sample_count < max_samples:
                    try:
                        img_path = Path(root) / img_file
                        img = Image.open(img_path)
                        img_array = np.array(img)
                        
                        category_info['image_dimensions']['width'].append(img_array.shape[1])
                        category_info['image_dimensions']['height'].append(img_array.shape[0])
                        category_info['image_dimensions']['channels'].append(
                            img_array.shape[2] if len(img_array.shape) > 2 else 1
                        )
                        
                        # Basic intensity statistics
                        if len(img_array.shape) == 2:
                            category_info['image_dimensions']['mean_intensity'].append(np.mean(img_array))
                            category_info['image_dimensions']['std_intensity'].append(np.std(img_array))
                        
                        if sample_count < 5:
                            category_info['sample_images'].append({
                                'file': img_file,
                                'path': relative_path,
                                'shape': img_array.shape,
                                'dtype': str(img_array.dtype)
                            })
                        
                        sample_count += 1
                        
                    except Exception as e:
                        print(f"Error reading image {img_file}: {e}")
        
        return category_info
    
    def generate_improved_visualizations(self):
        """Generate improved, meaningful visualizations"""
        print("📊 Generating improved visualizations...")
        
        # Create output directory
        viz_dir = self.dataset_path / 'improved_visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Improved Dataset Structure Visualization
        if 'structure' in self.results:
            self._plot_improved_dataset_structure(viz_dir)
        
        # 2. Improved Antenna Configuration Visualizations
        if 'antenna' in self.results:
            self._plot_improved_antenna_configs(viz_dir)
        
        # 3. Improved Polygon Data Visualizations
        if 'polygon' in self.results:
            self._plot_improved_polygon_data(viz_dir)
        
        # 4. Improved Image Data Visualizations
        if 'images' in self.results:
            self._plot_improved_image_data(viz_dir)
        
        print(f"✅ Improved visualizations saved to {viz_dir}")
        return viz_dir
    
    def _plot_improved_dataset_structure(self, viz_dir):
        """Plot improved dataset structure visualization"""
        structure = self.results['structure']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('RadioMapSeer Dataset Structure - Comprehensive Analysis', fontsize=16, fontweight='bold')
        
        # 1. Data completeness analysis
        ax = axes[0, 0]
        if 'data_completeness' in structure:
            completeness = structure['data_completeness']
            categories = list(completeness.keys())
            values = list(completeness.values())
            
            bars = ax.bar(categories, values, color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.8)
            ax.set_ylabel('Count')
            ax.set_title('Data Completeness Analysis')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                       f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Directory structure breakdown
        ax = axes[0, 1]
        if 'detailed_structure' in structure:
            categories = list(structure['detailed_structure'].keys())
            file_counts = [structure['detailed_structure'][cat]['total_files'] for cat in categories]
            
            # Create horizontal bar chart
            y_pos = range(len(categories))
            bars = ax.barh(y_pos, file_counts, color='lightblue', alpha=0.8, edgecolor='black')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(categories)
            ax.set_xlabel('Number of Files')
            ax.set_title('Files per Data Category')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, count) in enumerate(zip(bars, file_counts)):
                ax.text(bar.get_width() + max(file_counts)*0.01, bar.get_y() + bar.get_height()/2,
                       f'{count:,}', va='center', ha='left', fontweight='bold')
        
        # 3. Sample distribution overview
        ax = axes[1, 0]
        if 'total_samples' in structure:
            total_samples = structure['total_samples']
            
            # Create a comprehensive overview
            overview_data = [
                f'Total Samples: {total_samples:,}',
                f'Total Files: {sum(structure["detailed_structure"][cat]["total_files"] for cat in structure["detailed_structure"].keys()):,}',
                f'CSV Columns: {len(structure.get("csv_columns", []))}',
                f'Main Categories: {len(structure["detailed_structure"])}'
            ]
            
            ax.text(0.1, 0.5, '\n'.join(overview_data), transform=ax.transAxes, 
                   fontsize=14, verticalalignment='center',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
            ax.set_title('Dataset Overview')
            ax.axis('off')
        
        # 4. Detailed subdirectory analysis
        ax = axes[1, 1]
        if 'detailed_structure' in structure:
            # Analyze subdirectory distribution
            subdir_counts = {}
            for category, cat_info in structure['detailed_structure'].items():
                for subdir, count in cat_info['subdirectories'].items():
                    if subdir != 'root':
                        subdir_counts[f"{category}/{subdir}"] = count
            
            # Get top 15 subdirectories by file count
            top_subdirs = sorted(subdir_counts.items(), key=lambda x: x[1], reverse=True)[:15]
            
            if top_subdirs:
                names = [item[0] for item in top_subdirs]
                counts = [item[1] for item in top_subdirs]
                
                # Create horizontal bar chart
                y_pos = range(len(names))
                bars = ax.barh(y_pos, counts, color='lightgreen', alpha=0.8, edgecolor='black')
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(names, fontsize=8)
                ax.set_xlabel('Number of Files')
                ax.set_title('Top 15 Subdirectories by File Count')
                ax.grid(True, alpha=0.3, axis='x')
                
                # Add value labels
                for i, (bar, count) in enumerate(zip(bars, counts)):
                    ax.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                           f'{count:,}', va='center', ha='left', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'dataset_structure_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_improved_antenna_configs(self, viz_dir):
        """Plot improved antenna configuration visualization"""
        antenna = self.results['antenna']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Antenna Configuration Analysis - Detailed Statistics', fontsize=16, fontweight='bold')
        
        # 1. Antenna count distribution
        ax = axes[0, 0]
        if antenna['antenna_counts']:
            ax.hist(antenna['antenna_counts'], bins=20, alpha=0.7, color='skyblue', 
                   edgecolor='black', density=True)
            ax.set_xlabel('Number of Antennas')
            ax.set_ylabel('Density')
            ax.set_title('Antenna Count Distribution')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(antenna['antenna_counts'])
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_val:.1f}')
            ax.legend()
        
        # 2. Spatial distribution heatmap
        ax = axes[0, 1]
        if antenna['position_stats']['x'] and antenna['position_stats']['y']:
            x_pos = antenna['position_stats']['x']
            y_pos = antenna['position_stats']['y']
            
            # Create 2D histogram
            heatmap = ax.hist2d(x_pos, y_pos, bins=25, cmap='YlOrRd', alpha=0.8,
                               edgecolors='black', linewidths=0.5)
            
            # Add colorbar
            cbar = plt.colorbar(heatmap[3], ax=ax, shrink=0.8)
            cbar.set_label('Antenna Density', rotation=270, labelpad=15)
            
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title('Antenna Position Density Heatmap')
            ax.set_aspect('equal')
        
        # 3. Position coordinate analysis
        ax = axes[1, 0]
        if antenna['position_stats']['x'] and antenna['position_stats']['y']:
            x_pos = antenna['position_stats']['x']
            y_pos = antenna['position_stats']['y']
            
            # Create box plots for both coordinates
            positions = [x_pos, y_pos]
            labels = ['X Positions', 'Y Positions']
            
            bp = ax.boxplot(positions, labels=labels, patch_artist=True)
            colors = ['lightblue', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel('Position Value')
            ax.set_title('Position Coordinate Distribution')
            ax.grid(True, alpha=0.3)
        
        # 4. Comprehensive statistics summary
        ax = axes[1, 1]
        if 'stats' in antenna and 'position_stats' in antenna:
            stats_text = []
            
            # Antenna count statistics
            stats_text.append("=== Antenna Count Statistics ===")
            for key, value in antenna['stats'].items():
                stats_text.append(f"{key.replace('_', ' ').title()}: {value:.2f}")
            
            stats_text.append("")
            
            # Position statistics
            stats_text.append("=== Position Statistics ===")
            pos_stats = ['x_mean', 'x_std', 'y_mean', 'y_std']
            for stat in pos_stats:
                if stat in antenna['position_stats']:
                    value = antenna['position_stats'][stat]
                    stats_text.append(f"{stat.replace('_', ' ').title()}: {value:.2f}")
            
            stats_text.append("")
            stats_text.append(f"Total Antennas Analyzed: {len(antenna['position_stats']['x'])}")
            stats_text.append(f"Configuration Files: {antenna['total_configs']}")
            
            ax.text(0.05, 0.95, '\n'.join(stats_text), transform=ax.transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
            ax.set_title('Comprehensive Statistics Summary')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'antenna_configurations_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_improved_polygon_data(self, viz_dir):
        """Plot improved polygon data visualization"""
        polygon = self.results['polygon']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Polygon (Building/Car) Data Analysis - Detailed Insights', fontsize=16, fontweight='bold')
        
        # 1. Object count distribution
        ax = axes[0, 0]
        if polygon['object_counts']:
            ax.hist(polygon['object_counts'], bins=25, alpha=0.7, color='lightcoral', 
                   edgecolor='black', density=True)
            ax.set_xlabel('Number of Objects')
            ax.set_ylabel('Density')
            ax.set_title('Object Count Distribution')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(polygon['object_counts'])
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_val:.1f}')
            ax.legend()
        
        # 2. Vertex count distribution
        ax = axes[0, 1]
        if polygon['shape_stats']['vertices']:
            vertices = polygon['shape_stats']['vertices']
            ax.hist(vertices, bins=30, alpha=0.7, color='gold', 
                   edgecolor='black', density=True)
            ax.set_xlabel('Number of Vertices')
            ax.set_ylabel('Density')
            ax.set_title('Polygon Vertex Count Distribution')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(vertices)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_val:.1f}')
            ax.legend()
        
        # 3. Spatial distribution
        ax = axes[1, 0]
        if polygon['shape_stats']['x_coords'] and polygon['shape_stats']['y_coords']:
            x_coords = polygon['shape_stats']['x_coords']
            y_coords = polygon['shape_stats']['y_coords']
            
            # Create scatter plot with density
            ax.scatter(x_coords, y_coords, alpha=0.3, s=1, color='purple')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_title('Polygon Vertex Spatial Distribution')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        # 4. Comprehensive statistics summary
        ax = axes[1, 1]
        if 'stats' in polygon and 'vertex_stats' in polygon:
            stats_text = []
            
            # Object statistics
            stats_text.append("=== Object Statistics ===")
            for key, value in polygon['stats'].items():
                stats_text.append(f"{key.replace('_', ' ').title()}: {value:.2f}")
            
            stats_text.append("")
            
            # Vertex statistics
            stats_text.append("=== Vertex Statistics ===")
            for key, value in polygon['vertex_stats'].items():
                stats_text.append(f"{key.replace('_', ' ').title()}: {value:.2f}")
            
            stats_text.append("")
            stats_text.append(f"Total Files: {polygon['total_files']}")
            stats_text.append(f"Total Objects: {sum(polygon['object_counts'])}")
            stats_text.append(f"Total Vertices: {len(polygon['shape_stats']['vertices'])}")
            
            ax.text(0.05, 0.95, '\n'.join(stats_text), transform=ax.transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
            ax.set_title('Comprehensive Polygon Statistics')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'polygon_data_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_improved_image_data(self, viz_dir):
        """Plot improved image data visualization"""
        images = self.results['images']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Image Data Analysis - Comprehensive Overview', fontsize=16, fontweight='bold')
        
        # 1. Category comparison
        ax = axes[0, 0]
        if images['categories']:
            categories = list(images['categories'].keys())
            image_counts = [images['categories'][cat]['total_images'] for cat in categories]
            
            bars = ax.bar(categories, image_counts, color=['skyblue', 'lightgreen'], alpha=0.8)
            ax.set_ylabel('Number of Images')
            ax.set_title('Images per Category')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, count in zip(bars, image_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(image_counts)*0.01,
                       f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Image dimensions analysis
        ax = axes[0, 1]
        dimensions_data = []
        for category, cat_info in images['categories'].items():
            if cat_info['image_dimensions']['width']:
                for width, height in zip(cat_info['image_dimensions']['width'], 
                                       cat_info['image_dimensions']['height']):
                    dimensions_data.append((category, width, height))
        
        if dimensions_data:
            df_dims = pd.DataFrame(dimensions_data, columns=['Category', 'Width', 'Height'])
            
            # Create scatter plot
            for category in df_dims['Category'].unique():
                cat_data = df_dims[df_dims['Category'] == category]
                ax.scatter(cat_data['Width'], cat_data['Height'], 
                          alpha=0.6, s=30, label=category)
            
            ax.set_xlabel('Width (pixels)')
            ax.set_ylabel('Height (pixels)')
            ax.set_title('Image Dimensions Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. Channel distribution
        ax = axes[1, 0]
        channel_data = []
        for category, cat_info in images['categories'].items():
            if cat_info['image_dimensions']['channels']:
                for channels in cat_info['image_dimensions']['channels']:
                    channel_data.append((category, channels))
        
        if channel_data:
            df_channels = pd.DataFrame(channel_data, columns=['Category', 'Channels'])
            channel_counts = df_channels['Channels'].value_counts()
            
            ax.pie(channel_counts.values, labels=[f'{k} channels' for k in channel_counts.index], 
                   autopct='%1.1f%%', startangle=90)
            ax.set_title('Image Channel Distribution')
        
        # 4. Subdirectory breakdown
        ax = axes[1, 1]
        subdir_data = []
        for category, cat_info in images['categories'].items():
            for subdir, count in cat_info['subdirectories'].items():
                if subdir != 'root':
                    subdir_data.append((f"{category}/{subdir}", count))
        
        if subdir_data:
            # Get top 10 subdirectories
            subdir_data.sort(key=lambda x: x[1], reverse=True)
            top_subdirs = subdir_data[:10]
            
            names = [item[0] for item in top_subdirs]
            counts = [item[1] for item in top_subdirs]
            
            # Create horizontal bar chart
            y_pos = range(len(names))
            bars = ax.barh(y_pos, counts, color='lightblue', alpha=0.8, edgecolor='black')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=8)
            ax.set_xlabel('Number of Images')
            ax.set_title('Top 10 Image Subdirectories')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                       f'{count:,}', va='center', ha='left', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'image_data_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_full_improved_eda(self):
        """Run complete improved EDA analysis"""
        print("🚀 Starting improved comprehensive EDA of RadioMapSeer dataset...")
        
        # Run all analyses
        self.analyze_dataset_structure()
        self.analyze_antenna_configurations()
        self.analyze_polygon_data()
        self.analyze_image_data()
        
        # Generate improved visualizations
        viz_dir = self.generate_improved_visualizations()
        
        print("🎉 Improved EDA completed successfully!")
        print(f"📊 Visualizations: {viz_dir}")
        
        return {
            'results': self.results,
            'visualizations': viz_dir
        }

def main():
    """Main function to run improved EDA"""
    dataset_path = "/home/cine/Documents/dataset/RadioMapSeer"
    
    # Initialize improved EDA
    eda = ImprovedRadioMapSeerEDA(dataset_path)
    
    # Run full analysis
    results = eda.run_full_improved_eda()
    
    return results

if __name__ == "__main__":
    main()