#!/usr/bin/env python3
"""
Quick generation script for RadioDiff VAE training visualizations.
This is a simplified version of the main visualization generator.
"""

from generate_streamlined_visualizations import RadioDiffVisualizer

def main():
    """Main function to generate all streamlined figures"""
    log_file = "radiodiff_Vae/2025-08-15-20-41_.log"
    
    print("RadioDiff VAE Quick Visualization Generator")
    print("=" * 50)
    
    try:
        visualizer = RadioDiffVisualizer(log_file)
        visualizer.generate_all_figures()
        print("\n✅ All streamlined figures generated successfully!")
        print("Check the radiodiff_Vae/ directory for results.")
    except Exception as e:
        print(f"❌ Error generating figures: {e}")
        print("Make sure the log file exists and all dependencies are installed.")

if __name__ == "__main__":
    main()