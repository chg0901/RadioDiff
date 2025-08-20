#!/usr/bin/env python3
"""
Final cleanup of mermaid diagrams with specific fixes for RadioDiff
"""

import os
import re

def fix_mermaid_syntax(mermaid_code):
    """Fix specific syntax issues in mermaid code."""
    
    # Fix lines with array notation that breaks parsing
    lines = mermaid_code.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Fix array notation in node labels
        if '[' in line and ']' in line:
            # Extract the label content
            start = line.find('[') + 1
            end = line.rfind(']')
            label_content = line[start:end]
            
            # Fix specific issues in labels
            label_content = label_content.replace(' in R^(', ' in R^')
            label_content = label_content.replace(' x ', ' x ')
            label_content = label_content.replace('->', '->')
            
            # Remove problematic characters that cause parsing errors
            label_content = label_content.replace('‖', '||')
            label_content = label_content.replace('√', 'sqrt')
            label_content = label_content.replace('𝒩', 'N')
            label_content = label_content.replace('√ᾱ', 'sqrt_alpha')
            label_content = label_content.replace('ᾱ', 'alpha_bar')
            label_content = label_content.replace('ε_θ', 'epsilon_theta')
            
            # Reconstruct the line
            prefix = line[:start]
            suffix = line[end+1:]
            fixed_line = prefix + label_content + suffix
            
            # Ensure proper closing of node labels
            if fixed_line.count('[') > fixed_line.count(']'):
                fixed_line += ']'
            
            fixed_lines.append(fixed_line)
        else:
            fixed_lines.append(line)
    
    # Join the lines
    cleaned = '\n'.join(fixed_lines)
    
    # Fix arrow syntax
    cleaned = cleaned.replace('-->', '->')
    
    # Fix style definitions - remove complex styles
    style_pattern = r'style\s+\w+\s+fill:#[0-9a-fA-F]{6}(,stroke:#[0-9a-fA-F]{6})?(,stroke-width:\d+px)?(,font-weight:bold)?'
    cleaned = re.sub(style_pattern, lambda m: m.group(0).split(',')[0] + ']', cleaned)
    
    # Remove any remaining double brackets
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
        cleaned_content = fix_mermaid_syntax(original_content)
        
        # Write the cleaned version
        with open(output_path, 'w') as f:
            f.write(cleaned_content)
        
        print(f"✅ Cleaned: {output_path}")
        
        # Show first few lines for verification
        with open(output_path, 'r') as f:
            preview = f.read()[:500]
        print(f"Preview: {preview}...")

def main():
    input_dir = 'radiodiff_standardized_report_mermaid'
    output_dir = 'radiodiff_standardized_report_mermaid_simple'
    
    print("🧹 RadioDiff Standardized Report Mermaid Cleaner (Simple)")
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