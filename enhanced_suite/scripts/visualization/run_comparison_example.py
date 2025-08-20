#!/usr/bin/env python3
"""
Example usage script for RadioDiff image comparison

This script demonstrates how to use the compare_images.py tool with 
the existing RadioDiff model outputs.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_comparison_example():
    """Run an example comparison using existing RadioDiff outputs"""
    
    # Define paths based on the project structure
    config_path = "configs/radio_sample_m.yaml"
    
    # Check different result directories
    possible_gen_dirs = [
        "radiodiff_LDM",      # LDM results
        "radiodiff_LDM2",     # LDM2 results  
        "radio_diff_DPM_Train", # DPM training results
        "results/test"        # Test results
    ]
    
    # Find the first existing directory with generated images
    gen_dir = None
    for directory in possible_gen_dirs:
        if Path(directory).exists():
            # Check if it contains sample images
            sample_files = list(Path(directory).glob("sample-*.png"))
            if sample_files:
                gen_dir = directory
                print(f"Found generated images in: {gen_dir}")
                print(f"Number of sample images: {len(sample_files)}")
                break
    
    if not gen_dir:
        print("Error: No generated images found in expected directories")
        return
    
    # Check for ground truth images
    gt_dir = gen_dir  # GT images are usually in the same directory
    gt_files = list(Path(gt_dir).glob("gt-sample-*.png"))
    
    if not gt_files:
        print(f"Error: No ground truth images found in {gt_dir}")
        return
    
    print(f"Found ground truth images: {len(gt_files)}")
    
    # Create output directory
    output_dir = "./enhanced_suite/archive/comparison_results"
    Path(output_dir).mkdir(exist_ok=True)
    
    # Build the command
    cmd = [
        sys.executable, "compare_images.py",
        "--config", config_path,
        "--gt_dir", gt_dir,
        "--gen_dir", gen_dir,
        "--output_dir", output_dir
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the comparison
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Comparison completed successfully!")
        print("Output:")
        print(result.stdout)
        
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"Error running comparison: {e}")
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
        return False
    
    # List generated files
    print(f"\nGenerated comparison files in {output_dir}:")
    for file in Path(output_dir).glob("*"):
        print(f"  - {file.name}")
    
    return True

def main():
    """Main function"""
    print("RadioDiff Image Comparison Example")
    print("=" * 40)
    
    # Check if compare_images.py exists
    if not Path("compare_images.py").exists():
        print("Error: compare_images.py not found in current directory")
        return
    
    # Run the example
    success = run_comparison_example()
    
    if success:
        print("\n" + "=" * 40)
        print("Example completed successfully!")
        print("You can now:")
        print("1. Check the comparison_results/ directory for detailed analysis")
        print("2. View the generated PNG files for visual comparisons")
        print("3. Examine the CSV files for numerical metrics")
    else:
        print("\n" + "=" * 40)
        print("Example failed. Please check the error messages above.")

if __name__ == "__main__":
    main()