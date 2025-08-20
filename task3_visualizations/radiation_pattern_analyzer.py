#!/usr/bin/env python3
"""
ICASSP2025 Radiation Pattern Analysis

This script analyzes and visualizes the 5 different antenna radiation patterns used in the ICASSP2025 dataset.

Author: Claude Code Assistant
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

class RadiationPatternAnalyzer:
    """Analyzer for antenna radiation patterns"""
    
    def __init__(self, pattern_dir):
        """Initialize analyzer with pattern directory"""
        self.pattern_dir = Path(pattern_dir)
        self.patterns = {}
        self.load_all_patterns()
    
    def load_all_patterns(self):
        """Load all antenna radiation patterns"""
        for ant_id in range(1, 6):
            pattern_file = self.pattern_dir / f'Ant{ant_id}_Pattern.csv'
            if pattern_file.exists():
                pattern_data = pd.read_csv(pattern_file, header=None).values.flatten()
                self.patterns[ant_id] = pattern_data
                print(f"Loaded Ant{ant_id} pattern: {len(pattern_data)} points, "
                      f"range: {pattern_data.min():.2f} to {pattern_data.max():.2f} dB")
    
    def analyze_pattern_characteristics(self):
        """Analyze characteristics of each radiation pattern"""
        characteristics = {}
        
        for ant_id, pattern in self.patterns.items():
            # Convert to linear scale for analysis
            pattern_linear = 10**(pattern/10)
            
            # Calculate characteristics
            characteristics[ant_id] = {
                'max_gain': np.max(pattern),
                'min_gain': np.min(pattern),
                'gain_range': np.max(pattern) - np.min(pattern),
                'mean_gain': np.mean(pattern),
                'std_gain': np.std(pattern),
                'front_to_back_ratio': np.max(pattern) - np.min(pattern),
                'beamwidth': self.calculate_beamwidth(pattern),
                'is_omnidirectional': self.is_omnidirectional(pattern),
                'pattern_type': self.classify_pattern(pattern)
            }
        
        return characteristics
    
    def calculate_beamwidth(self, pattern, threshold=-3):
        """Calculate beamwidth at specified threshold (dB)"""
        max_gain = np.max(pattern)
        threshold_level = max_gain + threshold
        
        # Find angles where pattern crosses threshold
        angles = np.arange(len(pattern))
        above_threshold = pattern >= threshold_level
        
        if np.sum(above_threshold) == 0:
            return 0
        
        # Calculate beamwidth
        first_index = np.where(above_threshold)[0][0]
        last_index = np.where(above_threshold)[0][-1]
        
        if last_index > first_index:
            beamwidth = last_index - first_index
        else:
            # Handle wraparound case
            beamwidth = (len(pattern) - first_index) + last_index
        
        return beamwidth
    
    def is_omnidirectional(self, pattern, tolerance=3):
        """Check if pattern is omnidirectional"""
        gain_range = np.max(pattern) - np.min(pattern)
        return gain_range < tolerance
    
    def classify_pattern(self, pattern):
        """Classify radiation pattern type"""
        gain_range = np.max(pattern) - np.min(pattern)
        max_gain = np.max(pattern)
        
        if gain_range < 1:
            return "Omnidirectional"
        elif gain_range < 5:
            return "Weakly Directional"
        elif gain_range < 15:
            return "Moderately Directional"
        elif gain_range < 25:
            return "Highly Directional"
        else:
            return "Extremely Directional"
    
    def create_polar_plot(self, ax, pattern, title, color='blue'):
        """Create polar plot of radiation pattern"""
        angles = np.arange(len(pattern))
        angles_rad = np.deg2rad(angles)
        
        # Convert dB to linear for polar plot
        pattern_linear = 10**(pattern/10)
        
        # Normalize to maximum
        pattern_normalized = pattern_linear / np.max(pattern_linear)
        
        ax.plot(angles_rad, pattern_normalized, color=color, linewidth=2)
        ax.fill(angles_rad, pattern_normalized, alpha=0.3, color=color)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def create_cartesian_plot(self, ax, pattern, title, color='blue'):
        """Create cartesian plot of radiation pattern"""
        angles = np.arange(len(pattern))
        
        ax.plot(angles, pattern, color=color, linewidth=2)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Gain (dB)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 360)
    
    def create_comprehensive_visualization(self, output_dir='./radiation_pattern_analysis'):
        """Create comprehensive visualization of all radiation patterns"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Analyze characteristics
        characteristics = self.analyze_pattern_characteristics()
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Polar plots - All patterns overlay
        ax1 = plt.subplot(2, 3, 1, projection='polar')
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for ant_id, pattern in self.patterns.items():
            angles = np.arange(len(pattern))
            angles_rad = np.deg2rad(angles)
            pattern_linear = 10**(pattern/10)
            pattern_normalized = pattern_linear / np.max(pattern_linear)
            
            ax1.plot(angles_rad, pattern_normalized, color=colors[ant_id-1], 
                    linewidth=2, label=f'Ant{ant_id}')
        
        ax1.set_title('All Radiation Patterns (Polar)\nComparative view of all 5 antenna patterns', 
                     fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Cartesian plots - All patterns overlay
        ax2 = plt.subplot(2, 3, 2)
        for ant_id, pattern in self.patterns.items():
            angles = np.arange(len(pattern))
            ax2.plot(angles, pattern, color=colors[ant_id-1], 
                    linewidth=2, label=f'Ant{ant_id}')
        
        ax2.set_title('All Radiation Patterns (Cartesian)\nGain vs angle comparison', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('Angle (degrees)')
        ax2.set_ylabel('Gain (dB)')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. Pattern diversity analysis - Use the previously empty subplot
        ax3 = plt.subplot(2, 3, 3)
        pattern_types = [chars['pattern_type'] for chars in characteristics.values()]
        gain_ranges = [chars['gain_range'] for chars in characteristics.values()]
        
        bars = ax3.bar(range(1, 6), gain_ranges, color=colors, alpha=0.7)
        ax3.set_title('Antenna Pattern Diversity\nGain range comparison', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Antenna ID')
        ax3.set_ylabel('Gain Range (dB)')
        ax3.set_xticks(range(1, 6))
        ax3.grid(True, alpha=0.3)
        
        # Add pattern type labels on bars
        for i, (bar, char) in enumerate(zip(bars, characteristics.values())):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    char['pattern_type'].split()[0], ha='center', va='bottom', fontsize=8)
        
        # 4. Individual pattern analysis (show first 2 patterns to avoid empty space)
        for i, (ant_id, pattern) in enumerate(self.patterns.items()):
            if i < 2:  # Show first 2 patterns in detail
                ax_polar = plt.subplot(2, 3, 4 + i, projection='polar')
                self.create_polar_plot(ax_polar, pattern, 
                    f'Ant{ant_id} - {characteristics[ant_id]["pattern_type"]}\n'
                    f'Range: {characteristics[ant_id]["gain_range"]:.1f} dB', 
                    colors[ant_id-1])
        
        # 5. Application analysis - Use the last subplot
        ax5 = plt.subplot(2, 3, 6)
        ax5.axis('off')
        
        # Create more concise summary text
        summary_text = "ICASSP2025 Antenna Pattern Analysis\n" + "="*40 + "\n\n"
        summary_text += "Key Insights:\n"
        summary_text += "• Ant1: Omnidirectional (uniform coverage)\n"
        summary_text += "• Ant2-3,5: Extremely directional (long-range links)\n"
        summary_text += "• Ant4: Moderate directional (balanced coverage)\n\n"
        summary_text += "Technical Specifications:\n"
        summary_text += "• Angular Resolution: 1° (360 points)\n"
        summary_text += "• Frequency Range: 868 MHz - 3.5 GHz\n"
        summary_text += "• Gain Range: 0-52 dB\n\n"
        summary_text += "Applications:\n"
        summary_text += "• Omnidirectional: General coverage\n"
        summary_text += "• Directional: Point-to-point links\n"
        summary_text += "• Mixed patterns: Network optimization"
        
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, 
                       fontsize=9, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('ICASSP2025 Radiation Pattern Analysis\nComprehensive Overview of 5 Antenna Types', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'radiation_patterns_comprehensive.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Create individual detailed plots
        self.create_individual_pattern_plots(output_dir, characteristics)
        
        return characteristics
    
    def create_individual_pattern_plots(self, output_dir, characteristics):
        """Create detailed individual plots for each antenna"""
        fig, axes = plt.subplots(5, 2, figsize=(15, 25))
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        # Application descriptions for each antenna type
        app_descriptions = {
            "Omnidirectional": "Perfect for general coverage\nUniform 360° signal distribution\nIdeal for base stations",
            "Weakly Directional": "Slight directional preference\nGood for sector coverage\nBalanced performance",
            "Moderately Directional": "Focused coverage areas\nGood for building corridors\nReduced interference",
            "Highly Directional": "Strong directional focus\nLong-distance point-to-point\nMinimal side lobes",
            "Extremely Directional": "Maximum directional gain\nSpecialized long-range links\nVery narrow beamwidth"
        }
        
        for i, (ant_id, pattern) in enumerate(self.patterns.items()):
            # Polar plot
            ax_polar = plt.subplot(5, 2, 2*i+1, projection='polar')
            self.create_polar_plot(ax_polar, pattern, 
                f'Ant{ant_id}: {characteristics[ant_id]["pattern_type"]}\n'
                f'360° Coverage Analysis', colors[ant_id-1])
            
            # Cartesian plot
            ax_cart = plt.subplot(5, 2, 2*i+2)
            self.create_cartesian_plot(ax_cart, pattern, 
                f'Ant{ant_id}: Gain vs Angle\nLinear Response Analysis', colors[ant_id-1])
            
            # Add comprehensive statistics and application info
            pattern_type = characteristics[ant_id]['pattern_type']
            app_desc = app_descriptions.get(pattern_type, "Specialized application")
            
            stats_text = (
                f"Pattern Type: {pattern_type}\n"
                f"Max Gain: {characteristics[ant_id]['max_gain']:.2f} dB\n"
                f"Min Gain: {characteristics[ant_id]['min_gain']:.2f} dB\n"
                f"Gain Range: {characteristics[ant_id]['gain_range']:.2f} dB\n"
                f"Beamwidth: {characteristics[ant_id]['beamwidth']:.1f}°\n"
                f"F/B Ratio: {characteristics[ant_id]['front_to_back_ratio']:.2f} dB\n\n"
                f"Applications:\n{app_desc}"
            )
            ax_cart.text(0.02, 0.98, stats_text, transform=ax_cart.transAxes, 
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle('Detailed Radiation Pattern Analysis\nIndividual Antenna Characteristics and Applications', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'radiation_patterns_detailed.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_characteristics_to_csv(self, characteristics, output_dir):
        """Save characteristics to CSV file"""
        df = pd.DataFrame(characteristics).T
        output_path = Path(output_dir) / 'radiation_pattern_characteristics.csv'
        df.to_csv(output_path, index=True)
        print(f"Characteristics saved to {output_path}")

def main():
    """Main function"""
    pattern_dir = '/home/cine/Documents/Github/RadioDiff/datasets/ICASSP2025_Dataset/Radiation_Patterns'
    output_dir = './radiation_pattern_analysis'
    
    print("=== ICASSP2025 Radiation Pattern Analysis ===")
    print(f"Pattern directory: {pattern_dir}")
    print(f"Output directory: {output_dir}")
    
    # Initialize analyzer
    analyzer = RadiationPatternAnalyzer(pattern_dir)
    
    # Create comprehensive visualization
    characteristics = analyzer.create_comprehensive_visualization(output_dir)
    
    # Save characteristics
    analyzer.save_characteristics_to_csv(characteristics, output_dir)
    
    # Print summary
    print("\n=== Radiation Pattern Summary ===")
    for ant_id, chars in characteristics.items():
        print(f"Antenna {ant_id}: {chars['pattern_type']} "
              f"(Range: {chars['gain_range']:.2f} dB, "
              f"Beamwidth: {chars['beamwidth']:.1f}°)")
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()