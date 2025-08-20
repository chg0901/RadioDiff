#!/usr/bin/env python3
"""
More thorough cleaning of mermaid diagrams to fix syntax issues
"""

import os
import re

def clean_mermaid_code_thorough(mermaid_code):
    """Thoroughly clean mermaid code to fix all syntax issues."""
    
    # Replace problematic characters in node labels
    def clean_node_label(match):
        label = match.group(1)
        # Clean the label content
        label = label.replace('∈', ' in ')
        label = label.replace('×', ' x ')
        label = label.replace('φ', 'phi')
        label = label.replace('θ', 'theta')
        label = label.replace('ε', 'epsilon')
        label = label.replace('α', 'alpha')
        label = label.replace('β', 'beta')
        label = label.replace('𝒩', 'N')
        label = label.replace('√', 'sqrt')
        label = label.replace('‖', '||')
        label = label.replace('∏', 'prod')
        label = label.replace('∑', 'sum')
        label = label.replace('→', '->')
        label = label.replace('≤', '<=')
        label = label.replace('≥', '>=')
        label = label.replace('≠', '!=')
        label = label.replace('∞', 'inf')
        label = label.replace('∂', 'partial')
        label = label.replace('∇', 'nabla')
        label = label.replace('∫', 'int')
        label = label.replace('Δ', 'Delta')
        label = label.replace('π', 'pi')
        label = label.replace('σ', 'sigma')
        label = label.replace('μ', 'mu')
        label = label.replace('λ', 'lambda')
        label = label.replace('ω', 'omega')
        label = label.replace('ψ', 'psi')
        label = label.replace('χ', 'chi')
        label = label.replace('ℝ', 'R')
        label = label.replace('ℕ', 'N')
        label = label.replace('ℤ', 'Z')
        label = label.replace('ℂ', 'C')
        label = label.replace('ℚ', 'Q')
        label = label.replace('𝐈', 'I')
        label = label.replace('𝐓', 'T')
        label = label.replace('𝐄', 'E')
        label = label.replace('𝐍', 'N')
        label = label.replace('𝐂', 'C')
        label = label.replace('𝐊', 'K')
        label = label.replace('𝐋', 'L')
        label = label.replace('𝐌', 'M')
        label = label.replace('𝐎', 'O')
        label = label.replace('𝐏', 'P')
        label = label.replace('𝐑', 'R')
        label = label.replace('𝐒', 'S')
        label = label.replace('𝐓', 'T')
        label = label.replace('𝐔', 'U')
        label = label.replace('𝐕', 'V')
        label = label.replace('𝐖', 'W')
        label = label.replace('𝐗', 'X')
        label = label.replace('𝐘', 'Y')
        label = label.replace('𝐙', 'Z')
        label = label.replace('𝐋', 'L')
        label = label.replace('𝐌', 'M')
        label = label.replace('𝐎', 'O')
        label = label.replace('𝐏', 'P')
        label = label.replace('𝐑', 'R')
        label = label.replace('𝐒', 'S')
        label = label.replace('𝐓', 'T')
        label = label.replace('𝐔', 'U')
        label = label.replace('𝐕', 'V')
        label = label.replace('𝐖', 'W')
        label = label.replace('𝐗', 'X')
        label = label.replace('𝐘', 'Y')
        label = label.replace('𝐙', 'Z')
        
        # Fix mathematical notation
        label = re.sub(r'(\w+)\^\(([^)]+)\)', r'\1^\2', label)  # x^(y) -> x^y
        label = re.sub(r'(\w+)_\(([^)]+)\)', r'\1_\2', label)    # x_(y) -> x_y
        
        return f'[{label}]'
    
    # Clean node labels
    cleaned = re.sub(r'\[([^\]]+)\]', clean_node_label, mermaid_code)
    
    # Fix arrow directions
    cleaned = cleaned.replace('-->', '->')
    
    # Clean style definitions
    cleaned = re.sub(r'style\s+(\w+)\s+fill:([^,\n]+),?([^,\n]*)', lambda m: f'style {m.group(1)} fill:{m.group(2)}', cleaned)
    
    # Remove extra brackets
    cleaned = cleaned.replace(']]', ']')
    
    return cleaned

def clean_all_mermaid_files(input_dir, output_dir):
    """Clean all mermaid files in the input directory."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .mmd files
    mmd_files = [f for f in os.listdir(input_dir) if f.endswith('.mmd')]
    mmd_files.sort()
    
    print(f"Found {len(mmd_files)} mermaid files to clean")
    
    for mmd_file in mmd_files:
        input_path = os.path.join(input_dir, mmd_file)
        output_path = os.path.join(output_dir, mmd_file)
        
        print(f"Cleaning {mmd_file}...")
        
        # Read the original file
        with open(input_path, 'r') as f:
            original_content = f.read()
        
        # Clean the content
        cleaned_content = clean_mermaid_code_thorough(original_content)
        
        # Write the cleaned version
        with open(output_path, 'w') as f:
            f.write(cleaned_content)
        
        print(f"✅ Cleaned: {output_path}")

def main():
    input_dir = 'radiodiff_standardized_report_mermaid'
    output_dir = 'radiodiff_standardized_report_mermaid_final'
    
    print("🧹 RadioDiff Standardized Report Mermaid Cleaner (Thorough)")
    print("=" * 60)
    
    # Clean all mermaid files
    print(f"\nCleaning mermaid files...")
    clean_all_mermaid_files(input_dir, output_dir)
    
    print(f"\n✅ Complete! All diagrams cleaned to {output_dir}/")
    print("📁 Cleaned files:")
    
    # List cleaned files
    for file in sorted(os.listdir(output_dir)):
        if file.endswith('.mmd'):
            print(f"   - {output_dir}/{file}")

if __name__ == "__main__":
    main()