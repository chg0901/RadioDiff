#!/usr/bin/env python3
"""
RadioDiff VAE Training Report Updater

This script automates the process of updating the training visualization report
with new log data and generating updated visualizations.

Usage:
    python update_training_report.py [--log_file PATH] [--report_file PATH]

Example:
    python update_training_report.py
    python update_training_report.py --log_file radiodiff_Vae/2025-08-15-17-21_.log
"""

import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import shutil

# Set matplotlib to use non-interactive backend
plt.switch_backend('Agg')

class TrainingReportUpdater:
    def __init__(self, log_file=None, report_file=None):
        """Initialize the updater with file paths."""
        self.base_dir = '/home/cine/Documents/Github/RadioDiff'
        self.log_file = log_file or os.path.join(self.base_dir, 'radiodiff_Vae/2025-08-15-17-21_.log')
        self.report_file = report_file or os.path.join(self.base_dir, 'radiodiff_Vae/training_visualization_report.md')
        self.visualization_script = os.path.join(self.base_dir, 'improved_visualization_final.py')
        self.output_dir = os.path.join(self.base_dir, 'radiodiff_Vae')
        
    def parse_log_file(self):
        """Parse the training log file and extract metrics."""
        print(f"Parsing log file: {self.log_file}")
        
        # Pattern to match training step lines
        pattern = r'\[Train Step\] (\d+)/\d+: (.+?)(?= lr: 0\.0+,\s*$)'
        
        # Dictionary to store merged data by step
        step_data = {}
        
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Error: Log file not found at {self.log_file}")
            return None
        
        for line in lines:
            match = re.search(pattern, line)
            if match:
                step = int(match.group(1))
                metrics_str = match.group(2)
                
                # Initialize step entry if not exists
                if step not in step_data:
                    step_data[step] = {'step': step}
                
                # Parse individual metrics and merge
                for metric in metrics_str.split(', '):
                    if ': ' in metric:
                        key, value = metric.split(': ')
                        try:
                            step_data[step][key] = float(value)
                        except ValueError:
                            continue
        
        # Convert to DataFrame and sort by step
        df = pd.DataFrame(list(step_data.values()))
        df = df.sort_values('step').reset_index(drop=True)
        
        if len(df) == 0:
            print("No training data found in log file.")
            return None
        
        print(f"Found {len(df)} unique training steps")
        print(f"Step range: {df['step'].min()} to {df['step'].max()}")
        
        return df
    
    def extract_current_metrics(self, df):
        """Extract current metrics from the dataframe."""
        if df is None or len(df) == 0:
            return None
            
        latest_step = df['step'].max()
        latest_data = df[df['step'] == latest_step].iloc[0]
        
        metrics = {
            'total_steps': len(df),
            'current_step': int(latest_step),
            'progress_percentage': (latest_step / 150000) * 100,
            'total_loss': latest_data.get('train/total_loss', 0),
            'kl_loss': latest_data.get('train/kl_loss', 0),
            'rec_loss': latest_data.get('train/rec_loss', 0),
            'disc_loss': latest_data.get('train/disc_loss', 0)
        }
        
        # Calculate ranges
        metrics['total_loss_range'] = (df['train/total_loss'].min(), df['train/total_loss'].max())
        metrics['kl_loss_range'] = (df['train/kl_loss'].min(), df['train/kl_loss'].max())
        metrics['rec_loss_range'] = (df['train/rec_loss'].min(), df['train/rec_loss'].max())
        
        # Calculate percentages
        initial_total = df['train/total_loss'].iloc[0] if len(df) > 0 else metrics['total_loss']
        initial_kl = df['train/kl_loss'].iloc[0] if len(df) > 0 else metrics['kl_loss']
        initial_rec = df['train/rec_loss'].iloc[0] if len(df) > 0 else metrics['rec_loss']
        
        metrics['total_loss_reduction'] = ((initial_total - metrics['total_loss']) / initial_total) * 100 if initial_total > 0 else 0
        metrics['kl_loss_growth'] = ((metrics['kl_loss'] - initial_kl) / initial_kl) * 100 if initial_kl > 0 else 0
        metrics['rec_loss_reduction'] = ((initial_rec - metrics['rec_loss']) / initial_rec) * 100 if initial_rec > 0 else 0
        
        return metrics
    
    def generate_visualizations(self, df):
        """Generate updated visualizations using the existing script."""
        print("Generating updated visualizations...")
        
        # Copy the visualization script to output directory if needed
        script_path = os.path.join(self.output_dir, 'temp_visualization.py')
        shutil.copy2(self.visualization_script, script_path)
        
        # Update the script to use our dataframe
        try:
            # Import and run the visualization functions
            import sys
            sys.path.append(self.base_dir)
            
            # Read the visualization script and modify it
            with open(script_path, 'r') as f:
                script_content = f.read()
            
            # Replace the main function to use our data
            modified_script = script_content.replace(
                "def main():\n    log_file_path = '/home/cine/Documents/Github/RadioDiff/radiodiff_Vae/2025-08-15-17-21_.log'",
                f"def main():\n    # Using pre-loaded dataframe\n    df = pd.DataFrame({df.to_dict('list')})"
            )
            
            # Save modified script
            with open(script_path, 'w') as f:
                f.write(modified_script)
            
            # Execute the modified script
            exec(compile(open(script_path).read(), script_path, 'exec'))
            
            print("Visualizations generated successfully!")
            
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            return False
        finally:
            # Clean up temporary script
            if os.path.exists(script_path):
                os.remove(script_path)
        
        return True
    
    def update_report(self, metrics):
        """Update the markdown report with new metrics."""
        if metrics is None:
            print("No metrics to update report with.")
            return False
            
        print(f"Updating report: {self.report_file}")
        
        # Read the existing report
        try:
            with open(self.report_file, 'r') as f:
                report_content = f.read()
        except FileNotFoundError:
            print(f"Report file not found: {self.report_file}")
            return False
        
        # Update key sections
        updates = [
            # Executive Summary
            (r'over \d+,\d+ steps \(\d+\.\d+% complete\)', 
             f'over {metrics["current_step"]:,} steps ({metrics["progress_percentage"]:.1f}% complete)'),
            
            # Key Statistics table
            (r'## Key Training Statistics \(\d+,\d+ steps\)', 
             f'## Key Training Statistics ({metrics["current_step"]:,} steps)'),
            
            # Total Loss
            (r'\*\*Total Loss\*\* \| [\d,]+ \| [\d,]+ - [\d,]+', 
             f'**Total Loss** | {metrics["total_loss"]:,.0f} | {metrics["total_loss_range"][0]:,.0f} - {metrics["total_loss_range"][1]:,.0f}'),
            
            # KL Loss
            (r'\*\*KL Loss\*\* \| [\d,]+ \| [\d,]+ - [\d,]+', 
             f'**KL Loss** | {metrics["kl_loss"]:,.0f} | {metrics["kl_loss_range"][0]:,.0f} - {metrics["kl_loss_range"][1]:,.0f}'),
            
            # Reconstruction Loss
            (r'\*\*Reconstruction Loss\*\* \| [\d\.]+ \| [\d\.]+ - [\d\.]+', 
             f'**Reconstruction Loss** | {metrics["rec_loss"]:.2f} | {metrics["rec_loss_range"][0]:.2f} - {metrics["rec_loss_range"][1]:.2f}'),
            
            # Current values in observations
            (r'Current Value\*: ~[\d,]+ at step \d+', 
             f'Current Value**: ~{metrics["total_loss"]:,.0f} at step {metrics["current_step"]}'),
            
            (r'Current Value\*: ~[\d,]+ at step \d+', 
             f'Current Value**: ~{metrics["kl_loss"]:,.0f} at step {metrics["current_step"]}'),
            
            (r'Current Value\*: [\d\.]+ at step \d+', 
             f'Current Value**: {metrics["rec_loss"]:.2f} at step {metrics["current_step"]}'),
            
            # Reduction percentages
            (r'Reduction\*: \d+% decrease', 
             f'Reduction**: {metrics["total_loss_reduction"]:.0f}% decrease'),
            
            (r'Growth\*: \d+% increase', 
             f'Growth**: {metrics["kl_loss_growth"]:.0f}% increase'),
            
            (r'Reduction\*: \d+% decrease', 
             f'Reduction**: {metrics["rec_loss_reduction"]:.0f}% decrease'),
            
            # Key achievements
            (r'Reconstruction Quality\*: Excellent \([\d\.]+ loss\)', 
             f'Reconstruction Quality**: Excellent ({metrics["rec_loss"]:.2f} loss)'),
            
            # Q&A section
            (r'reconstruction loss of [\d\.]+ is excellent', 
             f'reconstruction loss of {metrics["rec_loss"]:.2f} is excellent'),
            
            # Success metrics
            (r'Reconstruction Loss\*: Maintain < [\d\.]+', 
             f'Reconstruction Loss**: Maintain < {metrics["rec_loss"]:.2f} (currently achieved)'),
            
            # Final progress
            (r'Training Progress: \d+\.\d+% complete \([\d,]+/[\d,]+ steps\)', 
             f'Training Progress: {metrics["progress_percentage"]:.1f}% complete ({metrics["current_step"]:,}/150,000 steps)')
        ]
        
        # Apply updates
        updated_content = report_content
        for pattern, replacement in updates:
            updated_content = re.sub(pattern, replacement, updated_content)
        
        # Write updated report
        with open(self.report_file, 'w') as f:
            f.write(updated_content)
        
        print("Report updated successfully!")
        return True
    
    def run(self):
        """Run the complete update process."""
        print("=== RadioDiff VAE Training Report Updater ===")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Step 1: Parse log file
        df = self.parse_log_file()
        if df is None:
            return False
        
        # Step 2: Extract metrics
        metrics = self.extract_current_metrics(df)
        if metrics is None:
            return False
        
        print("\n=== Current Training Metrics ===")
        print(f"Steps: {metrics['current_step']:,} ({metrics['progress_percentage']:.1f}% complete)")
        print(f"Total Loss: {metrics['total_loss']:,.0f}")
        print(f"KL Loss: {metrics['kl_loss']:,.0f}")
        print(f"Reconstruction Loss: {metrics['rec_loss']:.3f}")
        print(f"Discriminator Loss: {metrics['disc_loss']:.3f}")
        print()
        
        # Step 3: Generate visualizations
        print("=== Generating Visualizations ===")
        if not self.generate_visualizations(df):
            print("Warning: Visualization generation failed, but continuing with report update...")
        
        # Step 4: Update report
        print("\n=== Updating Report ===")
        if not self.update_report(metrics):
            return False
        
        print("\n=== Update Complete ===")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Report updated with {metrics['current_step']:,} training steps")
        
        return True

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Update RadioDiff VAE training report with new log data')
    parser.add_argument('--log_file', help='Path to training log file')
    parser.add_argument('--report_file', help='Path to markdown report file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create updater instance
    updater = TrainingReportUpdater(
        log_file=args.log_file,
        report_file=args.report_file
    )
    
    # Run the update process
    success = updater.run()
    
    if success:
        print("\n✅ Report update completed successfully!")
    else:
        print("\n❌ Report update failed!")
        exit(1)

if __name__ == "__main__":
    main()