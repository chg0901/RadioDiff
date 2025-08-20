#!/usr/bin/env python3
"""
Fixed polygon data analysis for RadioMapSeer dataset
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict
import random

def analyze_polygon_data_fixed(dataset_path):
    """Analyze polygon (building/car) data - FIXED VERSION"""
    
    dataset_path = Path(dataset_path)
    
    print("🏢 Analyzing polygon data...")
    
    polygon_dir = dataset_path / 'polygon' / 'buildings_and_cars'
    if not polygon_dir.exists():
        print("Polygon directory not found!")
        return
    
    polygon_stats = {
        'total_files': 0,
        'sample_data': [],
        'object_counts': [],
        'shape_stats': defaultdict(list),
        'stats': {}
    }
    
    json_files = list(polygon_dir.glob('*.json'))
    polygon_stats['total_files'] = len(json_files)
    
    print(f"Found {len(json_files)} polygon files")
    
    # Analyze first 50 files
    for i, json_file in enumerate(json_files[:50]):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Fixed: data is a list of polygon objects, not a dictionary
            if isinstance(data, list):
                object_count = len(data)
                polygon_stats['object_counts'].append(object_count)
                
                # Analyze shapes - each object is a list of vertices
                for obj_data in data:
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
                        'sample_objects': data[:2]  # First 2 objects
                    })
                    
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    
    # Calculate statistics
    if polygon_stats['object_counts']:
        polygon_stats['stats'] = {
            'mean_objects': np.mean(polygon_stats['object_counts']),
            'std_objects': np.std(polygon_stats['object_counts']),
            'min_objects': np.min(polygon_stats['object_counts']),
            'max_objects': np.max(polygon_stats['object_counts'])
        }
    
    if polygon_stats['shape_stats']['vertices']:
        polygon_stats['stats'].update({
            'mean_vertices': np.mean(polygon_stats['shape_stats']['vertices']),
            'std_vertices': np.std(polygon_stats['shape_stats']['vertices']),
            'min_vertices': np.min(polygon_stats['shape_stats']['vertices']),
            'max_vertices': np.max(polygon_stats['shape_stats']['vertices'])
        })
    
    return polygon_stats

def plot_polygon_data_fixed(polygon_stats, output_path):
    """Plot polygon data analysis - FIXED VERSION"""
    
    print("📊 Creating polygon data visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Polygon (Building/Car) Data Analysis', fontsize=16, fontweight='bold')
    
    # 1. Object count distribution
    ax = axes[0, 0]
    if polygon_stats['object_counts']:
        ax.hist(polygon_stats['object_counts'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax.set_xlabel('Number of Objects')
        ax.set_ylabel('Frequency')
        ax.set_title('Object Count Distribution')
        ax.grid(True, alpha=0.3)
        print(f"✓ Object count distribution: {len(polygon_stats['object_counts'])} samples")
    else:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Object Count Distribution')
        print("✗ No object count data")
    
    # 2. Vertex count distribution
    ax = axes[0, 1]
    if polygon_stats['shape_stats']['vertices']:
        ax.hist(polygon_stats['shape_stats']['vertices'], bins=20, alpha=0.7, color='gold', edgecolor='black')
        ax.set_xlabel('Number of Vertices')
        ax.set_ylabel('Frequency')
        ax.set_title('Polygon Vertex Count Distribution')
        ax.grid(True, alpha=0.3)
        print(f"✓ Vertex count distribution: {len(polygon_stats['shape_stats']['vertices'])} polygons")
    else:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Polygon Vertex Count Distribution')
        print("✗ No vertex count data")
    
    # 3. Statistics summary
    ax = axes[1, 0]
    if polygon_stats['stats']:
        stats_text = []
        for key, value in polygon_stats['stats'].items():
            stats_text.append(f"{key.replace('_', ' ').title()}: {value:.2f}")
        
        ax.text(0.1, 0.5, '\n'.join(stats_text), transform=ax.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        ax.set_title('Polygon Data Statistics')
        ax.axis('off')
        print(f"✓ Statistics summary: {len(polygon_stats['stats'])} metrics")
    else:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Polygon Data Statistics')
        ax.axis('off')
        print("✗ No statistics data")
    
    # 4. Coordinate scatter plot
    ax = axes[1, 1]
    if polygon_stats['shape_stats']['x_coords'] and polygon_stats['shape_stats']['y_coords']:
        ax.scatter(polygon_stats['shape_stats']['x_coords'], polygon_stats['shape_stats']['y_coords'], 
                  alpha=0.3, s=10, color='purple')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Polygon Vertex Positions')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        print(f"✓ Coordinate scatter plot: {len(polygon_stats['shape_stats']['x_coords'])} vertices")
    else:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Polygon Vertex Positions')
        print("✗ No coordinate data")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Polygon data analysis saved to {output_path}")

def show_sample_polygon_visualization(dataset_path, sample_id="0"):
    """Show a detailed visualization of sample polygon data"""
    
    dataset_path = Path(dataset_path)
    
    print(f"\n🏗️  Sample Polygon Visualization for Sample {sample_id}")
    print("=" * 60)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Sample Polygon Data - Sample {sample_id}', fontsize=16, fontweight='bold')
    
    # Load polygon data
    polygon_file = dataset_path / 'polygon' / 'buildings_and_cars' / f'{sample_id}.json'
    
    if polygon_file.exists():
        with open(polygon_file, 'r') as f:
            polygon_data = json.load(f)
        
        print(f"✓ Loaded polygon data: {len(polygon_data)} objects")
        
        # 1. All polygons
        ax = axes[0, 0]
        for i, vertices in enumerate(polygon_data):
            if isinstance(vertices, list) and len(vertices) > 0:
                vertices_array = np.array(vertices)
                vertices_array = np.vstack([vertices_array, vertices_array[0]])
                ax.plot(vertices_array[:, 0], vertices_array[:, 1], 'b-', alpha=0.6, linewidth=1)
                ax.fill(vertices_array[:-1, 0], vertices_array[:-1, 1], alpha=0.3)
        
        ax.set_title(f'All Polygons ({len(polygon_data)} objects)')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 2. Large polygons only (> 20 vertices)
        ax = axes[0, 1]
        large_polygons = [p for p in polygon_data if len(p) > 20]
        for i, vertices in enumerate(large_polygons):
            if isinstance(vertices, list) and len(vertices) > 0:
                vertices_array = np.array(vertices)
                vertices_array = np.vstack([vertices_array, vertices_array[0]])
                ax.plot(vertices_array[:, 0], vertices_array[:, 1], 'r-', alpha=0.7, linewidth=1.5)
                ax.fill(vertices_array[:-1, 0], vertices_array[:-1, 1], alpha=0.4)
        
        ax.set_title(f'Large Polygons (>20 vertices) ({len(large_polygons)} objects)')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 3. Small polygons only (<= 20 vertices)
        ax = axes[1, 0]
        small_polygons = [p for p in polygon_data if len(p) <= 20]
        for i, vertices in enumerate(small_polygons):
            if isinstance(vertices, list) and len(vertices) > 0:
                vertices_array = np.array(vertices)
                vertices_array = np.vstack([vertices_array, vertices_array[0]])
                ax.plot(vertices_array[:, 0], vertices_array[:, 1], 'g-', alpha=0.6, linewidth=1)
                ax.fill(vertices_array[:-1, 0], vertices_array[:-1, 1], alpha=0.3)
        
        ax.set_title(f'Small Polygons (≤20 vertices) ({len(small_polygons)} objects)')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 4. Statistics summary
        ax = axes[1, 1]
        vertex_counts = [len(p) for p in polygon_data if isinstance(p, list)]
        
        stats_text = [
            f"Total Objects: {len(polygon_data)}",
            f"Mean Vertices: {np.mean(vertex_counts):.2f}",
            f"Std Vertices: {np.std(vertex_counts):.2f}",
            f"Min Vertices: {np.min(vertex_counts)}",
            f"Max Vertices: {np.max(vertex_counts)}",
            f"Large Polygons: {len(large_polygons)}",
            f"Small Polygons: {len(small_polygons)}"
        ]
        
        ax.text(0.1, 0.5, '\n'.join(stats_text), transform=ax.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax.set_title('Sample Statistics')
        ax.axis('off')
        
    else:
        print(f"✗ Polygon file not found: {polygon_file}")
        for ax in axes.flatten():
            ax.text(0.5, 0.5, 'Not Found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Polygon Data')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/cine/Documents/Github/RadioDiff/radiomapseer_polygon_sample.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Sample polygon visualization saved to radiomapseer_polygon_sample.png")

def main():
    """Main function"""
    dataset_path = "/home/cine/Documents/dataset/RadioMapSeer"
    
    print("🔍 Analyzing RadioMapSeer polygon data...")
    
    # Analyze polygon data
    polygon_stats = analyze_polygon_data_fixed(dataset_path)
    
    # Plot polygon data analysis
    plot_polygon_data_fixed(polygon_stats, '/home/cine/Documents/Github/RadioDiff/polygon_data_fixed.png')
    
    # Show sample polygon visualization
    show_sample_polygon_visualization(dataset_path, "0")
    
    # Print summary
    print("\n📊 Polygon Data Summary:")
    print("=" * 50)
    if polygon_stats['stats']:
        for key, value in polygon_stats['stats'].items():
            print(f"{key.replace('_', ' ').title()}: {value:.2f}")
    
    print(f"\nTotal Files Analyzed: {polygon_stats['total_files']}")
    print(f"Sample Files: {len(polygon_stats['sample_data'])}")
    
    print("\n🎉 Polygon data analysis completed!")

if __name__ == "__main__":
    main()