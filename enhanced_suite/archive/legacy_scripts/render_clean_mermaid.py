#!/usr/bin/env python3
"""
Simplified script to render mermaid diagrams for the merged RadioDiff report.
Fixes syntax issues and creates clean diagrams.
"""

import os
import re
import subprocess
from pathlib import Path

def clean_mermaid_code(mermaid_code):
    """Clean mermaid code to fix syntax issues."""
    
    # Remove problematic characters and fix syntax
    cleaned = mermaid_code
    
    # Replace special mathematical symbols that cause parsing issues
    replacements = {
        'ℝ': 'R',
        '∈': 'in',
        '×': 'x',
        '‖': '|',
        'φ': 'phi',
        'θ': 'theta',
        '∇': 'nabla',
        '∑': 'sum',
        '∫': 'integral',
        '𝒩': 'N',
        '√': 'sqrt',
        'α': 'alpha',
        'β': 'beta',
        'σ': 'sigma',
        'μ': 'mu',
        'ε': 'epsilon',
        'λ': 'lambda',
        '‖': '|',
        '[B, 3, 80, 80]': '[Bx3x80x80]',
        '[B, 1, 320, 320]': '[Bx1x320x320]',
        'z~q_φ(z|x)': 'z ~ q_phi(z|x)',
        'q_φ(z|x)': 'q_phi(z|x)',
        'p_θ(x|z)': 'p_theta(x|z)',
        'D_KL(q_φ(z|x)‖p(z))': 'D_KL(q_phi(z|x) || p(z))',
        '√ᾱ_t': 'sqrt(alpha_bar_t)',
        'ᾱ_t': 'alpha_bar_t',
        'x̂': 'x_hat',
        'μ_φ': 'mu_phi',
        'σ_φ': 'sigma_phi',
        'ε_θ': 'epsilon_theta',
        '(1-t/T)^0.96': '(1-t/T)^0.96',
        '320×320': '320x320',
        '80×80': '80x80',
        '8×8': '8x8',
        '4×4': '4x4',
        '2×2': '2x2',
        '1×1': '1x1',
        '16×': '16x',
        '𝐈': 'I',
        '∇θ': 'grad_theta',
        '‖∇θ‖_2': '|grad_theta|_2',
        '[1,2,4,4]': '[1,2,4,4]',
        '[-1, 1]': '[-1, 1]',
        '(0.0001, 0.02, T)': '(0.0001, 0.02, T)',
        '[-147 to -47 dB]': '[-147 to -47 dB]',
        '[-1, 1] range': '[-1, 1] range',
        '0.000001': '1e-6',
        '1e-4': '0.0001',
        '1e-8': '0.00000001',
        '5e-5': '0.00005',
        '5e-6': '0.000005'
    }
    
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    
    # Remove problematic HTML-like tags in node labels
    cleaned = re.sub(r'<br/>', '\n', cleaned)
    cleaned = re.sub(r'<[^>]*>', '', cleaned)
    
    # Fix array notation
    cleaned = re.sub(r'\[([^\]]+)\]', lambda m: '[' + m.group(1).replace(',', ', ') + ']', cleaned)
    
    return cleaned

def extract_and_clean_diagrams(markdown_file):
    """Extract and clean all mermaid diagrams from the markdown file."""
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all mermaid code blocks
    pattern = r'```mermaid\n(.*?)\n```'
    matches = re.findall(pattern, content, re.DOTALL)
    
    diagrams = []
    for i, match in enumerate(matches):
        cleaned_code = clean_mermaid_code(match.strip())
        diagrams.append({
            'index': i + 1,
            'original_code': match.strip(),
            'cleaned_code': cleaned_code,
            'title': f"Figure {i + 1}"
        })
    
    return diagrams

def create_simple_mermaid_config():
    """Create a simple mermaid configuration."""
    config_content = '''{
        "theme": "default",
        "themeVariables": {
            "primaryColor": "#007bff",
            "primaryTextColor": "#ffffff",
            "primaryBorderColor": "#0056b3",
            "lineColor": "#007bff",
            "secondaryColor": "#6c757d",
            "tertiaryColor": "#f8f9fa",
            "background": "#ffffff",
            "fontFamily": "Arial, sans-serif"
        },
        "flowchart": {
            "curve": "basis",
            "padding": 20,
            "nodeSpacing": 50,
            "rankSpacing": 50,
            "htmlLabels": true,
            "useMaxWidth": true
        }
    }'''
    
    with open('/tmp/mermaid-config.json', 'w', encoding='utf-8') as f:
        f.write(config_content)

def render_mermaid_to_image(mermaid_code, output_path, diagram_index):
    """Render mermaid code to image using mmdc."""
    
    # Create temporary mermaid file
    temp_mmd_file = f"/tmp/diagram_{diagram_index}.mmd"
    with open(temp_mmd_file, 'w', encoding='utf-8') as f:
        f.write(mermaid_code)
    
    # Render using mmdc
    cmd = [
        'mmdc',
        '-i', temp_mmd_file,
        '-o', output_path,
        '-t', 'default',  # Use default theme
        '-w', '1200',     # Width
        '-H', '800',      # Height
        '-b', 'white',    # Background
        '-e', 'png',      # Output format
        '-c', '/tmp/mermaid-config.json'  # Config file path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Successfully rendered {output_path}")
        
        # Clean up temp file
        os.remove(temp_mmd_file)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error rendering {output_path}: {e}")
        print(f"Stderr: {e.stderr}")
        # Save the problematic code for debugging
        debug_file = f"/tmp/debug_diagram_{diagram_index}.mmd"
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(mermaid_code)
        print(f"Debug code saved to {debug_file}")
        return False
    except FileNotFoundError:
        print("mmdc command not found. Please install mermaid-cli.")
        return False

def create_output_directories():
    """Create output directories for rendered images."""
    base_dir = Path("radiodiff_standardized_mermaid_vis")
    base_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    subdirs = ["architecture", "training", "loss", "optimization", "pipeline"]
    for subdir in subdirs:
        (base_dir / subdir).mkdir(exist_ok=True)
    
    return base_dir

def main():
    """Main function to render all mermaid diagrams."""
    
    # Input file
    markdown_file = "RADIODIFF_MERGED_STANDARDIZED_REPORT.md"
    
    # Check if file exists
    if not os.path.exists(markdown_file):
        print(f"Error: {markdown_file} not found")
        return
    
    # Create output directories
    output_base = create_output_directories()
    
    # Create simple config
    create_simple_mermaid_config()
    
    # Extract and clean diagrams
    diagrams = extract_and_clean_diagrams(markdown_file)
    print(f"Found {len(diagrams)} mermaid diagrams")
    
    # Directory mapping for different diagram types
    dir_mapping = {
        1: "architecture",   # Complete Model Pipeline
        2: "architecture",   # Detailed Architecture Components
        3: "pipeline",       # Training Data Pipeline
        4: "architecture",   # Conditional Information Integration
        5: "loss",          # Comprehensive Loss Architecture
        6: "loss",          # Multi-Component Loss Breakdown
        7: "training",      # Two-Phase Training Strategy
        8: "optimization",  # Key Configuration Parameters
        9: "optimization",  # Optimization Strategy
        10: "architecture", # VAE Architecture
        11: "architecture", # Conditional U-Net Architecture
        12: "pipeline"      # Training Pipeline Execution
    }
    
    # Render each diagram
    success_count = 0
    for diagram in diagrams:
        diagram_index = diagram['index']
        
        # Determine output directory
        output_dir = dir_mapping.get(diagram_index, "architecture")
        
        # Output path
        output_path = output_base / output_dir / f"figure_{diagram_index:02d}.png"
        
        # Render diagram
        if render_mermaid_to_image(diagram['cleaned_code'], str(output_path), diagram_index):
            success_count += 1
        else:
            print(f"Failed to render diagram {diagram_index}")
    
    print(f"\nSuccessfully rendered {success_count}/{len(diagrams)} diagrams")
    print(f"Output directory: {output_base}")
    
    # Clean up config file
    if os.path.exists('/tmp/mermaid-config.json'):
        os.remove('/tmp/mermaid-config.json')

if __name__ == "__main__":
    main()