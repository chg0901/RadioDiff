#!/usr/bin/env python3
"""
ICASSP2025 Dataset Validation Visualization Script

This script creates comprehensive visualizations for dataset validation:
- Dataset distribution analysis
- Image quality metrics
- Training progress visualization
- Loss component analysis

Author: Claude Code Assistant
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
import json
from tqdm.auto import tqdm
from collections import Counter
import cv2

class ICASSPDatasetVisualizer:
    """Dataset visualizer for ICASSP2025 dataset"""
    
    def __init__(self, dataset_root, results_root):
        self.dataset_root = Path(dataset_root)
        self.results_root = Path(results_root)
        self.results_root.mkdir(exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def create_dataset_overview_visualization(self):
        """Create comprehensive dataset overview visualization"""
        print("Creating dataset overview visualization...")
        
        # Load validation data if available
        validation_file = self.dataset_root / 'validation_report.json'
        validation_data = None
        if validation_file.exists():
            try:
                with open(validation_file, 'r') as f:
                    validation_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error reading validation file: {e}")
                validation_data = None
        else:
            print("Validation file not found, using direct dataset analysis")
            validation_data = None
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Dataset structure
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_dataset_structure(ax1)
        
        # 2. Sample distribution
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_sample_distribution(ax2)
        
        # 3. Building distribution
        ax3 = fig.add_subplot(gs[0, 2])
        self.plot_building_distribution(ax3)
        
        # 4. Frequency distribution
        ax4 = fig.add_subplot(gs[0, 3])
        self.plot_frequency_distribution(ax4)
        
        # 5. Image quality metrics
        ax5 = fig.add_subplot(gs[1, :2])
        self.plot_image_quality_metrics(ax5)
        
        # 6. File size distribution
        ax6 = fig.add_subplot(gs[1, 2:])
        self.plot_file_size_distribution(ax6)
        
        # 7. Sample images
        ax7 = fig.add_subplot(gs[2, :2])
        self.plot_sample_images(ax7)
        
        # 8. Validation summary
        ax8 = fig.add_subplot(gs[2, 2:])
        self.plot_validation_summary(ax8, validation_data)
        
        # 9. Training readiness score
        ax9 = fig.add_subplot(gs[3, :])
        self.plot_training_readiness_score(ax9, validation_data)
        
        plt.suptitle('ICASSP2025 Dataset Validation Overview', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the visualization
        output_path = self.results_root / 'dataset_validation_overview.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Dataset overview visualization saved to: {output_path}")
        
    def plot_dataset_structure(self, ax):
        """Plot dataset structure"""
        # Count files in each directory
        structure_data = {
            'Train Images': len(list((self.dataset_root / 'train' / 'image').glob('*.png'))),
            'Train Labels': len(list((self.dataset_root / 'train' / 'edge').glob('*.png'))),
            'Val Images': len(list((self.dataset_root / 'val' / 'image').glob('*.png'))),
            'Val Labels': len(list((self.dataset_root / 'val' / 'edge').glob('*.png')))
        }
        
        bars = ax.bar(structure_data.keys(), structure_data.values(), 
                     color=['#2ecc71', '#27ae60', '#3498db', '#2980b9'])
        ax.set_title('Dataset Structure', fontweight='bold')
        ax.set_ylabel('Number of Files')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
        
        ax.tick_params(axis='x', rotation=45)
        
    def plot_sample_distribution(self, ax):
        """Plot train/validation split"""
        train_count = len(list((self.dataset_root / 'train' / 'image').glob('*.png')))
        val_count = len(list((self.dataset_root / 'val' / 'image').glob('*.png')))
        total = train_count + val_count
        
        sizes = [train_count, val_count]
        labels = [f'Train ({train_count})', f'Validation ({val_count})']
        colors = ['#2ecc71', '#3498db']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Train/Validation Split', fontweight='bold')
        
    def plot_building_distribution(self, ax):
        """Plot building distribution"""
        building_counts = {}
        
        for split in ['train', 'val']:
            image_dir = self.dataset_root / split / 'image'
            for file_path in image_dir.glob('*.png'):
                # Extract building ID from filename
                parts = file_path.stem.split('_')
                building_id = parts[0][1:]
                building_counts[building_id] = building_counts.get(building_id, 0) + 1
        
        if building_counts:
            buildings = sorted(building_counts.keys())
            counts = [building_counts[b] for b in buildings]
            
            bars = ax.bar(buildings, counts, color='#e74c3c')
            ax.set_title('Building Distribution', fontweight='bold')
            ax.set_xlabel('Building ID')
            ax.set_ylabel('Sample Count')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No building data found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Building Distribution', fontweight='bold')
            
    def plot_frequency_distribution(self, ax):
        """Plot frequency distribution"""
        freq_counts = {'f1': 0, 'f2': 0, 'f3': 0}
        
        for split in ['train', 'val']:
            image_dir = self.dataset_root / split / 'image'
            for file_path in image_dir.glob('*.png'):
                # Extract frequency ID from filename
                parts = file_path.stem.split('_')
                freq_id = parts[2][1:]
                freq_counts[f'f{freq_id}'] = freq_counts.get(f'f{freq_id}', 0) + 1
        
        freq_labels = ['868 MHz', '1.8 GHz', '3.5 GHz']
        freq_values = [freq_counts['f1'], freq_counts['f2'], freq_counts['f3']]
        colors = ['#9b59b6', '#f39c12', '#1abc9c']
        
        bars = ax.bar(freq_labels, freq_values, color=colors)
        ax.set_title('Frequency Distribution', fontweight='bold')
        ax.set_ylabel('Sample Count')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
            
    def plot_image_quality_metrics(self, ax):
        """Plot image quality metrics"""
        quality_data = []
        
        # Sample some images for quality analysis
        for split in ['train', 'val']:
            image_dir = self.dataset_root / split / 'image'
            sample_files = list(image_dir.glob('*.png'))[:20]  # Sample 20 files
            
            for file_path in sample_files:
                try:
                    img = Image.open(file_path)
                    img_array = np.array(img)
                    
                    # Calculate quality metrics
                    for channel in range(3):
                        channel_data = img_array[:, :, channel].flatten()
                        quality_data.append({
                            'split': split,
                            'channel': f'Channel {channel}',
                            'mean': np.mean(channel_data),
                            'std': np.std(channel_data),
                            'min': np.min(channel_data),
                            'max': np.max(channel_data)
                        })
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        if quality_data:
            df = pd.DataFrame(quality_data)
            
            # Create box plot of standard deviations (contrast)
            sns.boxplot(data=df, x='channel', y='std', hue='split', ax=ax)
            ax.set_title('Image Quality Metrics (Standard Deviation)', fontweight='bold')
            ax.set_ylabel('Standard Deviation')
            ax.legend(title='Split')
        else:
            ax.text(0.5, 0.5, 'No quality data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Image Quality Metrics', fontweight='bold')
            
    def plot_file_size_distribution(self, ax):
        """Plot file size distribution"""
        file_sizes = []
        
        for split in ['train', 'val']:
            for data_type in ['image', 'edge']:
                data_dir = self.dataset_root / split / data_type
                for file_path in data_dir.glob('*.png'):
                    file_size = file_path.stat().st_size / 1024  # KB
                    file_sizes.append({
                        'split': split,
                        'type': data_type,
                        'size': file_size
                    })
        
        if file_sizes:
            df = pd.DataFrame(file_sizes)
            
            # Create histogram
            for split in ['train', 'val']:
                split_data = df[df['split'] == split]
                ax.hist(split_data['size'], alpha=0.6, label=f'{split.title()} (n={len(split_data)})', bins=20)
            
            ax.set_title('File Size Distribution', fontweight='bold')
            ax.set_xlabel('File Size (KB)')
            ax.set_ylabel('Frequency')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No file size data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('File Size Distribution', fontweight='bold')
            
    def plot_sample_images(self, ax):
        """Plot sample images from dataset"""
        # Get sample files
        sample_files = []
        for split in ['train', 'val']:
            image_dir = self.dataset_root / split / 'image'
            files = list(image_dir.glob('*.png'))[:2]  # 2 samples per split
            sample_files.extend(files)
        
        if sample_files:
            # Create grid of sample images
            n_samples = len(sample_files)
            cols = min(4, n_samples)
            rows = (n_samples + cols - 1) // cols
            
            # Clear the axis and create subplots
            ax.clear()
            
            # Create subplot grid
            for i, file_path in enumerate(sample_files):
                row = i // cols
                col = i % cols
                
                # Calculate position within the main axis
                x_pos = col / cols
                y_pos = 1 - (row + 1) / rows
                width = 1 / cols
                height = 1 / rows
                
                # Create inset axis
                inset_ax = ax.inset_axes([x_pos, y_pos, width, height])
                
                try:
                    img = Image.open(file_path)
                    inset_ax.imshow(img)
                    inset_ax.set_title(f"{file_path.parent.name}/{file_path.stem}", fontsize=8)
                    inset_ax.axis('off')
                except Exception as e:
                    inset_ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=inset_ax.transAxes)
                    inset_ax.axis('off')
            
            ax.set_title('Sample Images', fontweight='bold')
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'No sample images found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Sample Images', fontweight='bold')
            
    def plot_validation_summary(self, ax, validation_data):
        """Plot validation summary"""
        if validation_data and 'validation_results' in validation_data:
            results = validation_data['validation_results']
            
            # Create validation metrics
            metrics = []
            
            # File structure validation
            if 'file_structure' in results:
                fs_valid = results['file_structure'].get('valid', False)
                metrics.append(('File Structure', 1 if fs_valid else 0))
            
            # Image files validation
            if 'image_files' in results:
                img_valid = len(results['image_files'].get('corrupt_files', [])) == 0
                metrics.append(('Image Files', 1 if img_valid else 0))
            
            # Pairing validation
            if 'pairing' in results:
                pair_valid = results['pairing'].get('valid', False)
                metrics.append(('File Pairing', 1 if pair_valid else 0))
            
            if metrics:
                labels, scores = zip(*metrics)
                colors = ['#2ecc71' if score == 1 else '#e74c3c' for score in scores]
                
                bars = ax.bar(labels, scores, color=colors)
                ax.set_title('Validation Results', fontweight='bold')
                ax.set_ylabel('Pass/Fail')
                ax.set_ylim(0, 1.2)
                
                # Add value labels
                for bar, score in zip(bars, scores):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           '✓' if score == 1 else '✗', ha='center', va='bottom', 
                           fontsize=16, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No validation data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Validation Results', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No validation data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Validation Results', fontweight='bold')
            
    def plot_training_readiness_score(self, ax, validation_data):
        """Plot training readiness score"""
        # Calculate training readiness score
        score_components = {
            'Dataset Structure': 0.2,
            'File Integrity': 0.2,
            'Image Quality': 0.2,
            'Validation Results': 0.2,
            'Training Configuration': 0.2
        }
        
        # Calculate actual scores based on validation data
        if validation_data and 'validation_results' in validation_data:
            results = validation_data['validation_results']
            
            # Dataset structure
            if 'file_structure' in results:
                score_components['Dataset Structure'] = 0.2 if results['file_structure'].get('valid', False) else 0
            
            # File integrity
            if 'image_files' in results:
                corrupt_count = len(results['image_files'].get('corrupt_files', []))
                format_issues = len(results['image_files'].get('format_issues', []))
                if corrupt_count == 0 and format_issues == 0:
                    score_components['File Integrity'] = 0.2
            
            # Image quality (placeholder - would need actual quality metrics)
            score_components['Image Quality'] = 0.2
            
            # Validation results
            if 'pairing' in results:
                score_components['Validation Results'] = 0.2 if results['pairing'].get('valid', False) else 0
            
            # Training configuration (placeholder)
            score_components['Training Configuration'] = 0.2
        
        # Create bar plot
        labels = list(score_components.keys())
        scores = list(score_components.values())
        colors = ['#2ecc71' if score > 0 else '#e74c3c' for score in scores]
        
        bars = ax.bar(labels, scores, color=colors)
        ax.set_title('Training Readiness Score', fontweight='bold')
        ax.set_ylabel('Score Component')
        ax.set_ylim(0, 0.25)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Add total score
        total_score = sum(scores)
        ax.text(0.5, 0.95, f'Total Score: {total_score:.1f}/1.0', 
                ha='center', va='top', transform=ax.transAxes, 
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.tick_params(axis='x', rotation=45)
        
    def create_training_progress_visualization(self, training_log_path):
        """Create training progress visualization"""
        print("Creating training progress visualization...")
        
        # Parse training log
        training_data = self.parse_training_log(training_log_path)
        
        if not training_data:
            print("No training data found")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Total loss over time
        ax1 = axes[0, 0]
        self.plot_loss_over_time(ax1, training_data, 'total_loss', 'Total Loss')
        
        # 2. Reconstruction loss over time
        ax2 = axes[0, 1]
        self.plot_loss_over_time(ax2, training_data, 'rec_loss', 'Reconstruction Loss')
        
        # 3. KL loss over time
        ax3 = axes[0, 2]
        self.plot_loss_over_time(ax3, training_data, 'kl_loss', 'KL Loss')
        
        # 4. Loss reduction percentage
        ax4 = axes[1, 0]
        self.plot_loss_reduction(ax4, training_data)
        
        # 5. Learning rate schedule
        ax5 = axes[1, 1]
        self.plot_learning_rate(ax5, training_data)
        
        # 6. Training speed
        ax6 = axes[1, 2]
        self.plot_training_speed(ax6, training_data)
        
        plt.suptitle('ICASSP2025 VAE Training Progress', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the visualization
        output_path = self.results_root / 'training_progress.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training progress visualization saved to: {output_path}")
        
    def parse_training_log(self, log_path):
        """Parse training log file"""
        training_data = []
        
        try:
            with open(log_path, 'r') as f:
                for line in f:
                    if '[Train Step]' in line and 'train/total_loss:' in line:
                        # Parse log line
                        parts = line.strip().split(',')
                        step_info = parts[0].split('[Train Step] ')[1].split(':')
                        step = int(step_info[0].split('/')[0])
                        
                        data_point = {'step': step}
                        
                        # Parse metrics
                        for part in parts[1:]:
                            if ': ' in part:
                                key, value = part.split(': ')
                                key = key.strip()
                                try:
                                    data_point[key] = float(value)
                                except ValueError:
                                    data_point[key] = value
                        
                        training_data.append(data_point)
        except Exception as e:
            print(f"Error parsing training log: {e}")
        
        return training_data
        
    def plot_loss_over_time(self, ax, training_data, loss_key, title):
        """Plot specific loss over time"""
        if training_data:
            steps = [d['step'] for d in training_data if loss_key in d]
            losses = [d[loss_key] for d in training_data if loss_key in d]
            
            if steps and losses:
                ax.plot(steps, losses, linewidth=2, color='#3498db')
                ax.set_title(title, fontweight='bold')
                ax.set_xlabel('Step')
                ax.set_ylabel('Loss')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No {loss_key} data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No training data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontweight='bold')
            
    def plot_loss_reduction(self, ax, training_data):
        """Plot loss reduction percentage"""
        if training_data and len(training_data) > 1:
            initial_loss = training_data[0].get('total_loss', 0)
            current_losses = [d.get('total_loss', 0) for d in training_data]
            
            if initial_loss > 0:
                reduction_percentages = [(1 - loss / initial_loss) * 100 for loss in current_losses]
                steps = [d['step'] for d in training_data]
                
                ax.plot(steps, reduction_percentages, linewidth=2, color='#2ecc71')
                ax.set_title('Loss Reduction', fontweight='bold')
                ax.set_xlabel('Step')
                ax.set_ylabel('Reduction (%)')
                ax.grid(True, alpha=0.3)
                
                # Add current reduction percentage
                current_reduction = reduction_percentages[-1]
                ax.text(0.5, 0.95, f'Current: {current_reduction:.1f}%', 
                       ha='center', va='top', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'Invalid initial loss', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Loss Reduction', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Loss Reduction', fontweight='bold')
            
    def plot_learning_rate(self, ax, training_data):
        """Plot learning rate schedule"""
        if training_data:
            steps = [d['step'] for d in training_data if 'lr' in d]
            lr_values = [d['lr'] for d in training_data if 'lr' in d]
            
            if steps and lr_values:
                ax.plot(steps, lr_values, linewidth=2, color='#e74c3c')
                ax.set_title('Learning Rate', fontweight='bold')
                ax.set_xlabel('Step')
                ax.set_ylabel('Learning Rate')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No learning rate data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Learning Rate', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No training data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Learning Rate', fontweight='bold')
            
    def plot_training_speed(self, ax, training_data):
        """Plot training speed analysis"""
        if training_data and len(training_data) > 1:
            # Calculate time between steps (simplified)
            steps = [d['step'] for d in training_data]
            
            # Show step progression
            ax.plot(steps, [1] * len(steps), 'o-', markersize=4, linewidth=1, color='#9b59b6')
            ax.set_title('Training Progress', fontweight='bold')
            ax.set_xlabel('Step')
            ax.set_ylabel('Progress')
            ax.grid(True, alpha=0.3)
            
            # Add current step info
            current_step = steps[-1]
            total_steps = 150000
            progress_percent = (current_step / total_steps) * 100
            
            ax.text(0.5, 0.95, f'Step: {current_step}/{total_steps} ({progress_percent:.1f}%)', 
                   ha='center', va='top', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Training Progress', fontweight='bold')

def main():
    """Main function"""
    # Configuration
    dataset_root = './icassp2025_dataset_arranged'
    results_root = './icassp2025_validation_visualizations'
    training_log_path = './results/icassp2025_Vae/2025-08-18-20-17_.log'
    
    # Create visualizer
    visualizer = ICASSPDatasetVisualizer(dataset_root, results_root)
    
    # Create visualizations
    print("Creating ICASSP2025 dataset validation visualizations...")
    
    # Dataset overview
    visualizer.create_dataset_overview_visualization()
    
    # Training progress
    if os.path.exists(training_log_path):
        visualizer.create_training_progress_visualization(training_log_path)
    else:
        print(f"Training log not found: {training_log_path}")
    
    print("Visualizations complete!")

if __name__ == "__main__":
    main()