#!/usr/bin/env python3
"""
Test script to verify log parsing for RadioDiff VAE training data
"""

import re
import pandas as pd

def test_parsing():
    log_file = '/home/cine/Documents/Github/RadioDiff/radiodiff_Vae/2025-08-15-20-41_.log'
    
    # Test the current pattern
    pattern = r'\[Train Step\] (\d+)/\d+: (.+?)(?= lr: 0\.0+,\s*$)'
    
    step_data = {}
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    print(f"Total lines in log file: {len(lines)}")
    
    # Test first few lines with the pattern
    for i, line in enumerate(lines[:10]):
        match = re.search(pattern, line)
        if match:
            step = int(match.group(1))
            metrics_str = match.group(2)
            print(f"Line {i+1}: Step {step} - Metrics length: {len(metrics_str)}")
            print(f"  Metrics: {metrics_str[:100]}...")
        else:
            print(f"Line {i+1}: NO MATCH")
            print(f"  Content: {line.strip()}")
    
    # Count total matches
    total_matches = 0
    for line in lines:
        if re.search(pattern, line):
            total_matches += 1
    
    print(f"\nTotal matches found: {total_matches}")
    
    # Test improved pattern
    print("\n=== Testing improved pattern ===")
    improved_pattern = r'\[Train Step\] (\d+)/\d+: (.+?)(?= lr: 0\.0+)'
    
    step_data = {}
    for line in lines:
        match = re.search(improved_pattern, line)
        if match:
            step = int(match.group(1))
            metrics_str = match.group(2)
            
            if step not in step_data:
                step_data[step] = {'step': step}
            
            # Parse individual metrics
            for metric in metrics_str.split(', '):
                if ': ' in metric:
                    key, value = metric.split(': ')
                    try:
                        step_data[step][key] = float(value)
                    except ValueError:
                        continue
    
    print(f"Improved pattern found {len(step_data)} unique steps")
    if step_data:
        first_step = min(step_data.keys())
        last_step = max(step_data.keys())
        print(f"Step range: {first_step} to {last_step}")
        
        # Show sample metrics
        sample_step = list(step_data.keys())[0]
        print(f"Sample metrics from step {sample_step}:")
        for key, value in step_data[sample_step].items():
            if key != 'step':
                print(f"  {key}: {value}")
    
    return step_data

if __name__ == "__main__":
    test_parsing()