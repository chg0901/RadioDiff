#!/usr/bin/env python3
"""
Script to parse VAE training logs and extract training data for visualization
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

def parse_vae_training_log(log_file_path):
    """
    Parse VAE training log file and extract training metrics
    
    Args:
        log_file_path (str): Path to the log file
        
    Returns:
        tuple: (config_dict, training_data_df)
    """
    print(f"Parsing log file: {log_file_path}")
    
    # Read the log file
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    # Extract configuration (first two lines)
    config_lines = lines[:2]
    config_dict = {}
    
    for line in config_lines:
        # Extract the dictionary part
        match = re.search(r'\{.*\}', line)
        if match:
            config_str = match.group(0)
            try:
                # Parse the configuration dictionary
                config_data = eval(config_str)
                config_dict.update(config_data)
            except:
                print(f"Warning: Could not parse config line: {line}")
    
    # Extract training data
    training_data = []
    
    # Pattern to match training steps
    step_pattern = r'\[(Train Step)\] (\d+)/(\d+): (.+)'
    
    for line in lines:
        match = re.search(step_pattern, line)
        if match:
            step_type, current_step, total_steps, metrics_str = match.groups()
            
            # Parse metrics
            metrics = {}
            metrics_list = metrics_str.strip().split(', ')
            
            for metric in metrics_list:
                if ': ' in metric:
                    key, value = metric.split(': ')
                    try:
                        # Clean up the value - remove trailing comma and other non-numeric characters
                        value = value.strip().rstrip(',')
                        metrics[key] = float(value)
                    except:
                        metrics[key] = value
            
            # Add step information
            metrics['step'] = int(current_step)
            metrics['total_steps'] = int(total_steps)
            
            # Extract timestamp
            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}', line)
            if timestamp_match:
                metrics['timestamp'] = timestamp_match.group(1)
                metrics['datetime'] = datetime.strptime(metrics['timestamp'], '%Y-%m-%d %H:%M:%S')
            
            training_data.append(metrics)
    
    # Convert to DataFrame
    df = pd.DataFrame(training_data)
    
    print(f"Extracted {len(df)} training steps")
    print(f"Config: Resume from milestone {config_dict.get('trainer', {}).get('resume_milestone', 'N/A')}")
    
    return config_dict, df

def analyze_training_progress(df):
    """
    Analyze training progress and compute statistics
    """
    analysis = {}
    
    # Basic statistics
    analysis['total_steps'] = len(df)
    analysis['start_step'] = df['step'].min()
    analysis['end_step'] = df['step'].max()
    analysis['training_duration'] = (df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600  # hours
    
    # Loss statistics
    loss_cols = ['train/total_loss', 'train/kl_loss', 'train/nll_loss', 'train/rec_loss']
    for col in loss_cols:
        if col in df.columns:
            analysis[f'{col}_mean'] = df[col].mean()
            analysis[f'{col}_std'] = df[col].std()
            analysis[f'{col}_min'] = df[col].min()
            analysis[f'{col}_max'] = df[col].max()
            analysis[f'{col}_final'] = df[col].iloc[-1]
    
    # Learning rate analysis
    if 'lr' in df.columns:
        analysis['lr_final'] = df['lr'].iloc[-1]
        analysis['lr_mean'] = df['lr'].mean()
    
    # Discriminator weight analysis
    if 'train/d_weight' in df.columns:
        analysis['d_weight_final'] = df['train/d_weight'].iloc[-1]
        analysis['d_weight_mean'] = df['train/d_weight'].mean()
    
    return analysis

if __name__ == "__main__":
    # Parse the log file
    log_file = "radiodiff_Vae/2025-08-15-20-41_.log"
    config, df = parse_vae_training_log(log_file)
    
    # Analyze training progress
    analysis = analyze_training_progress(df)
    
    # Print analysis summary
    print("\n=== Training Analysis Summary ===")
    print(f"Total training steps: {analysis['total_steps']}")
    print(f"Step range: {analysis['start_step']} to {analysis['end_step']}")
    print(f"Training duration: {analysis['training_duration']:.2f} hours")
    print(f"Final total loss: {analysis['train/total_loss_final']:.2f}")
    print(f"Final KL loss: {analysis['train/kl_loss_final']:.2f}")
    print(f"Final reconstruction loss: {analysis['train/rec_loss_final']:.6f}")
    print(f"Final learning rate: {analysis.get('lr_final', 'N/A')}")
    print(f"Final discriminator weight: {analysis.get('d_weight_final', 'N/A')}")
    
    # Save parsed data
    df.to_csv('radiodiff_Vae/training_data_parsed.csv', index=False)
    print(f"\nSaved parsed training data to: radiodiff_Vae/training_data_parsed.csv")
    
    # Save analysis
    import json
    # Convert numpy types to JSON serializable
    analysis_json = {k: float(v) if isinstance(v, (np.int64, np.float64)) else v for k, v in analysis.items()}
    with open('radiodiff_Vae/training_analysis.json', 'w') as f:
        json.dump(analysis_json, f, indent=2)
    print(f"Saved training analysis to: radiodiff_Vae/training_analysis.json")