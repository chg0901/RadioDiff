#!/usr/bin/env python3
"""
ICASSP2025 Training Progress Visualization

This script creates training progress visualizations from the training logs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path
from datetime import datetime

class TrainingProgressVisualizer:
    """Training progress visualizer"""
    
    def __init__(self, log_file_path, output_dir):
        self.log_file_path = Path(log_file_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Parse training data
        self.training_data = self.parse_training_log()
        
    def parse_training_log(self):
        """Parse training log file"""
        training_data = []
        
        if not self.log_file_path.exists():
            print(f"Log file not found: {self.log_file_path}")
            return training_data
        
        try:
            with open(self.log_file_path, 'r') as f:
                for line in f:
                    # Parse training step lines
                    if '[Train Step]' in line and 'train/total_loss:' in line:
                        # Extract step number
                        step_match = re.search(r'\[Train Step\] (\d+)/(\d+):', line)
                        if step_match:
                            step = int(step_match.group(1))
                            total_steps = int(step_match.group(2))
                            
                            # Extract all metrics
                            metrics = {'step': step, 'total_steps': total_steps}
                            
                            # Parse all key-value pairs
                            metrics_pattern = r'(\w+)/?(\w*):\s*([\d\.eE-]+)'
                            matches = re.findall(metrics_pattern, line)
                            
                            for match in matches:
                                prefix, suffix, value = match
                                key = f"{prefix}_{suffix}" if suffix else prefix
                                try:
                                    metrics[key] = float(value)
                                except ValueError:
                                    metrics[key] = value
                            
                            training_data.append(metrics)
                            
        except Exception as e:
            print(f"Error parsing training log: {e}")
        
        return training_data
    
    def create_training_progress_dashboard(self):
        """Create comprehensive training progress dashboard"""
        if not self.training_data:
            print("No training data available")
            return
        
        print("Creating training progress dashboard...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(self.training_data)
        
        # 1. Total Loss over time
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_loss_component(ax1, df, 'train_total_loss', 'Total Loss', '#3498db')
        
        # 2. Reconstruction Loss
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_loss_component(ax2, df, 'train_rec_loss', 'Reconstruction Loss', '#2ecc71')
        
        # 3. KL Loss
        ax3 = fig.add_subplot(gs[0, 2])
        self.plot_loss_component(ax3, df, 'train_kl_loss', 'KL Loss', '#e74c3c')
        
        # 4. Loss Reduction
        ax4 = fig.add_subplot(gs[0, 3])
        self.plot_loss_reduction(ax4, df)
        
        # 5. Learning Rate
        ax5 = fig.add_subplot(gs[1, 0])
        self.plot_learning_rate(ax5, df)
        
        # 6. Training Progress
        ax6 = fig.add_subplot(gs[1, 1])
        self.plot_training_progress(ax6, df)
        
        # 7. Loss Components Comparison
        ax7 = fig.add_subplot(gs[1, 2])
        self.plot_loss_components_comparison(ax7, df)
        
        # 8. Training Speed
        ax8 = fig.add_subplot(gs[1, 3])
        self.plot_training_speed(ax8, df)
        
        # 9. Current Status
        ax9 = fig.add_subplot(gs[2, :])
        self.plot_current_status(ax9, df)
        
        plt.suptitle('ICASSP2025 VAE Training Progress Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the dashboard
        output_path = self.output_dir / 'training_progress_dashboard.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training progress dashboard saved to: {output_path}")
        
    def plot_loss_component(self, ax, df, column, title, color):
        """Plot a specific loss component"""
        if column in df.columns:
            ax.plot(df['step'], df[column], linewidth=2, color=color)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            
            # Add current value
            current_value = df[column].iloc[-1]
            ax.text(0.98, 0.95, f'Current: {current_value:.2f}', 
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, f'No {column} data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontweight='bold')
    
    def plot_loss_reduction(self, ax, df):
        """Plot loss reduction percentage"""
        if 'train_total_loss' in df.columns and len(df) > 1:
            initial_loss = df['train_total_loss'].iloc[0]
            current_losses = df['train_total_loss']
            
            if initial_loss > 0:
                reduction_percentages = [(1 - loss / initial_loss) * 100 for loss in current_losses]
                
                ax.plot(df['step'], reduction_percentages, linewidth=2, color='#9b59b6')
                ax.set_title('Loss Reduction', fontweight='bold')
                ax.set_xlabel('Step')
                ax.set_ylabel('Reduction (%)')
                ax.grid(True, alpha=0.3)
                
                # Add current reduction percentage
                current_reduction = reduction_percentages[-1]
                ax.text(0.98, 0.95, f'Current: {current_reduction:.1f}%', 
                       transform=ax.transAxes, ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'Invalid initial loss', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Loss Reduction', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Loss Reduction', fontweight='bold')
    
    def plot_learning_rate(self, ax, df):
        """Plot learning rate schedule"""
        if 'lr' in df.columns:
            ax.plot(df['step'], df['lr'], linewidth=2, color='#f39c12')
            ax.set_title('Learning Rate', fontweight='bold')
            ax.set_xlabel('Step')
            ax.set_ylabel('Learning Rate')
            ax.grid(True, alpha=0.3)
            
            # Add current learning rate
            current_lr = df['lr'].iloc[-1]
            ax.text(0.98, 0.95, f'Current: {current_lr:.2e}', 
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No learning rate data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Learning Rate', fontweight='bold')
    
    def plot_training_progress(self, ax, df):
        """Plot training progress"""
        if len(df) > 0:
            current_step = df['step'].iloc[-1]
            total_steps = df['total_steps'].iloc[0] if 'total_steps' in df.columns else 150000
            progress_percent = (current_step / total_steps) * 100
            
            # Create progress bar
            ax.barh(0, progress_percent, height=0.3, color='#2ecc71')
            ax.barh(0, 100 - progress_percent, height=0.3, left=progress_percent, color='#ecf0f1')
            
            ax.set_xlim(0, 100)
            ax.set_ylim(-0.5, 0.5)
            ax.set_title('Training Progress', fontweight='bold')
            ax.set_xlabel('Progress (%)')
            ax.set_yticks([])
            
            # Add progress text
            ax.text(progress_percent/2, 0, f'{progress_percent:.1f}%', 
                   ha='center', va='center', fontweight='bold')
            ax.text(50, 0.3, f'Step {current_step:,}/{total_steps:,}', 
                   ha='center', va='bottom', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No progress data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Training Progress', fontweight='bold')
    
    def plot_loss_components_comparison(self, ax, df):
        """Plot comparison of different loss components"""
        loss_components = ['train_rec_loss', 'train_kl_loss']
        available_components = [col for col in loss_components if col in df.columns]
        
        if available_components:
            for component in available_components:
                ax.plot(df['step'], df[component], linewidth=2, label=component.replace('train_', '').replace('_', ' ').title())
            
            ax.set_title('Loss Components', fontweight='bold')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No loss components data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Loss Components', fontweight='bold')
    
    def plot_training_speed(self, ax, df):
        """Plot training speed analysis"""
        if len(df) > 1:
            # Calculate steps per minute (simplified)
            steps = df['step'].values
            time_intervals = np.diff(steps)
            
            if len(time_intervals) > 0:
                avg_steps_per_interval = np.mean(time_intervals)
                steps_per_minute = 60 / avg_steps_per_interval if avg_steps_per_interval > 0 else 0
                
                ax.text(0.5, 0.7, f'Avg Speed: {steps_per_minute:.1f} steps/min', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12, fontweight='bold')
                ax.text(0.5, 0.5, f'Interval: {avg_steps_per_interval:.1f} steps', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.text(0.5, 0.3, f'Total Steps: {len(df)}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
            else:
                ax.text(0.5, 0.5, 'Insufficient data for speed analysis', 
                       ha='center', va='center', transform=ax.transAxes)
            
            ax.set_title('Training Speed', fontweight='bold')
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'No training speed data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Training Speed', fontweight='bold')
    
    def plot_current_status(self, ax, df):
        """Plot current training status"""
        if len(df) > 0:
            current_step = df['step'].iloc[-1]
            total_steps = df['total_steps'].iloc[0] if 'total_steps' in df.columns else 150000
            progress_percent = (current_step / total_steps) * 100
            
            # Calculate loss reduction
            if 'train_total_loss' in df.columns:
                initial_loss = df['train_total_loss'].iloc[0]
                current_loss = df['train_total_loss'].iloc[-1]
                loss_reduction = (1 - current_loss / initial_loss) * 100 if initial_loss > 0 else 0
            else:
                loss_reduction = 0
            
            # Create status summary
            lr_text = f"🎮 Learning Rate: {df['lr'].iloc[-1]:.2e}" if 'lr' in df.columns else "🎮 Learning Rate: N/A"
            status_text = (
                f"🚀 ICASSP2025 VAE Training Status\n"
                f"{'='*50}\n"
                f"📊 Current Progress: {current_step:,}/{total_steps:,} steps ({progress_percent:.1f}%)\n"
                f"📈 Loss Reduction: {loss_reduction:.1f}%\n"
                f"🎯 Current Loss: {current_loss:.2f}\n"
                f"⚡ Training Status: {'ACTIVE' if progress_percent < 100 else 'COMPLETED'}\n"
                f"📅 Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"{'='*50}\n"
                f"🔧 Configuration: 64 base channels, 2 latent channels\n"
                f"💾 Memory Usage: ~10GB GPU, 4 samples/batch\n"
                f"{lr_text}"
            )
            
            ax.text(0.05, 0.95, status_text, transform=ax.transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            ax.set_title('Current Training Status', fontweight='bold')
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'No status data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Current Training Status', fontweight='bold')

def main():
    """Main function"""
    # Configuration
    log_file_path = './results/icassp2025_Vae/2025-08-18-20-17_.log'
    output_dir = './icassp2025_training_visualizations'
    
    # Create visualizer
    visualizer = TrainingProgressVisualizer(log_file_path, output_dir)
    
    # Create training progress dashboard
    visualizer.create_training_progress_dashboard()
    
    print("Training progress visualization complete!")

if __name__ == "__main__":
    main()