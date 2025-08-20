#!/usr/bin/env python3
"""
Clean mermaid diagrams to fix syntax issues and make them renderable
"""

import os
import re

def clean_mermaid_code(mermaid_code):
    """Clean mermaid code to fix common syntax issues."""
    
    # Replace problematic mathematical symbols
    replacements = {
        'φ': 'phi',
        'θ': 'theta',
        'ε': 'epsilon',
        'α': 'alpha',
        'β': 'beta',
        '𝒩': 'N',
        '√': 'sqrt',
        '‖': '||',
        '∏': 'prod',
        '∑': 'sum',
        '∈': 'in',
        '×': 'x',
        '→': '->',
        '≤': '<=',
        '≥': '>=',
        '≠': '!=',
        '∞': 'inf',
        '∂': 'partial',
        '∇': 'nabla',
        '∫': 'int',
        'Δ': 'Delta',
        'π': 'pi',
        'σ': 'sigma',
        'μ': 'mu',
        'λ': 'lambda',
        'ω': 'omega',
        'ψ': 'psi',
        'χ': 'chi',
        'ℝ': 'R',
        'ℕ': 'N',
        'ℤ': 'Z',
        'ℂ': 'C',
        'ℚ': 'Q',
        '𝐈': 'I',
        '𝐓': 'T',
        '𝐄': 'E',
        '𝐍': 'N',
        '𝐂': 'C',
        '𝐊': 'K',
        '𝐋': 'L',
        '𝐌': 'M',
        '𝐎': 'O',
        '𝐏': 'P',
        '𝐑': 'R',
        '𝐒': 'S',
        '𝐓': 'T',
        '𝐔': 'U',
        '𝐕': 'V',
        '𝐖': 'W',
        '𝐗': 'X',
        '𝐘': 'Y',
        '𝐙': 'Z'
    }
    
    # Apply replacements
    cleaned = mermaid_code
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    
    # Fix bracket issues in node labels
    cleaned = re.sub(r'\[([^\]]*?)\^\(([^)]*?)\)', lambda m: f'[{m.group(1)}^({m.group(2)})]', cleaned)
    cleaned = re.sub(r'\[([^\]]*?)_\(([^)]*?)\)', lambda m: f'[{m.group(1)}_({m.group(2)})]', cleaned)
    cleaned = re.sub(r'\[([^\]]*?)∈\(([^)]*?)\)', lambda m: f'[{m.group(1)} in ({m.group(2)})]', cleaned)
    cleaned = re.sub(r'\[([^\]]*?)×\(([^)]*?)\)', lambda m: f'[{m.group(1)} x ({m.group(2)})]', cleaned)
    
    # Fix mathematical notation in labels
    cleaned = re.sub(r'\[([^\]]*?)ℝ\^\(([^)]*?)\)', lambda m: f'[{m.group(1)} R^({m.group(2)})]', cleaned)
    cleaned = re.sub(r'\[([^\]]*?)√\(([^)]*?)\)', lambda m: f'[{m.group(1)} sqrt({m.group(2)})]', cleaned)
    cleaned = re.sub(r'\[([^\]]*?)𝒩\(([^)]*?)\)', lambda m: f'[{m.group(1)} N({m.group(2)})]', cleaned)
    
    # Fix arrays with commas and brackets
    cleaned = re.sub(r'\[([^\]]*?)\[([^,\]]+),([^,\]]+)\]', lambda m: f'[{m.group(1)}[{m.group(2)},{m.group(3)}]]', cleaned)
    cleaned = re.sub(r'\[([^\]]*?)\[([^,\]]+),([^,\]]+),([^,\]]+)\]', lambda m: f'[{m.group(1)}[{m.group(2)},{m.group(3)},{m.group(4)}]]', cleaned)
    
    # Fix style definitions that might have issues
    cleaned = re.sub(r'style\s+(\w+)\s+fill:([^,\n]+)', lambda m: f'style {m.group(1)} fill:{m.group(2)}', cleaned)
    
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
        cleaned_content = clean_mermaid_code(original_content)
        
        # Write the cleaned version
        with open(output_path, 'w') as f:
            f.write(cleaned_content)
        
        print(f"✅ Cleaned: {output_path}")
        
        # Show a preview of changes
        print("Preview of changes:")
        lines_orig = original_content.split('\n')
        lines_clean = cleaned_content.split('\n')
        for i, (orig, clean) in enumerate(zip(lines_orig[:10], lines_clean[:10])):
            if orig != clean:
                print(f"  Line {i+1}: '{orig}' -> '{clean}'")
        print()

def main():
    input_dir = 'radiodiff_standardized_report_mermaid'
    output_dir = 'radiodiff_standardized_report_mermaid_cleaned'
    
    print("🧹 RadioDiff Standardized Report Mermaid Cleaner")
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