#!/usr/bin/env python3
"""
Script to render standardized mermaid diagrams for the merged RadioDiff report.
Creates high-quality images with consistent styling.
"""

import os
import re
import subprocess
from pathlib import Path

def extract_mermaid_diagrams(markdown_file):
    """Extract all mermaid diagrams from the markdown file."""
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all mermaid code blocks
    pattern = r'```mermaid\n(.*?)\n```'
    matches = re.findall(pattern, content, re.DOTALL)
    
    diagrams = []
    for i, match in enumerate(matches):
        diagrams.append({
            'index': i + 1,
            'code': match.strip(),
            'title': f"Figure {i + 1}"
        })
    
    return diagrams

def create_enhanced_mermaid_code(diagram_code, diagram_index):
    """Enhance mermaid code with better styling."""
    
    # Common styling improvements
    enhanced_code = diagram_code
    
    # Add title if not present
    if not enhanced_code.startswith('title'):
        title_map = {
            1: "RadioDiff Complete Model Pipeline",
            2: "Detailed Architecture Components", 
            3: "Training Data Pipeline",
            4: "Conditional Information Integration",
            5: "Comprehensive Loss Architecture",
            6: "Multi-Component Loss Breakdown",
            7: "Two-Phase Training Strategy",
            8: "Key Configuration Parameters",
            9: "Optimization Strategy",
            10: "VAE Architecture (First Stage)",
            11: "Conditional U-Net Architecture",
            12: "Training Pipeline Execution"
        }
        
        title = title_map.get(diagram_index, f"RadioDiff Diagram {diagram_index}")
        
        # For different diagram types, add titles appropriately
        if 'graph TD' in enhanced_code or 'graph LR' in enhanced_code or 'graph TB' in enhanced_code:
            # Add title as a comment at the beginning
            enhanced_code = f"%% {title}\n" + enhanced_code
        elif 'gantt' in enhanced_code:
            # Add title for gantt charts
            enhanced_code = enhanced_code.replace('title ', f'title {title}: ')
        elif 'sequenceDiagram' in enhanced_code:
            # Add title for sequence diagrams
            enhanced_code = f"    Note over Main,Trainer: {title}\n" + enhanced_code
    
    # Improve styling for all diagrams
    style_improvements = [
        "%% Enhanced styling",
        "linkStyle 0 stroke:#007bff,stroke-width:2px,fill:none;",
        "linkStyle 1 stroke:#007bff,stroke-width:2px,fill:none;",
        "linkStyle 2 stroke:#007bff,stroke-width:2px,fill:none;",
        "linkStyle 3 stroke:#007bff,stroke-width:2px,fill:none;",
        "linkStyle 4 stroke:#007bff,stroke-width:2px,fill:none;",
        "classDef default fill:#f8f9fa,stroke:#007bff,stroke-width:2px;",
        "classDef active fill:#e3f2fd,stroke:#1976d2,stroke-width:3px;",
        "classDef crit fill:#ffebee,stroke:#c62828,stroke-width:2px;",
        "classDef milestone fill:#fff3e0,stroke:#f57c00,stroke-width:2px;"
    ]
    
    # Add styling improvements
    for improvement in style_improvements:
        if improvement not in enhanced_code:
            enhanced_code += "\n" + improvement
    
    return enhanced_code

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
    
    # Create config file for better styling
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
            "primaryFontFamily": "Arial, sans-serif",
            "fontSize": "14px"
        },
        "flowchart": {
            "curve": "basis",
            "padding": 20,
            "nodeSpacing": 50,
            "rankSpacing": 50,
            "htmlLabels": true,
            "useMaxWidth": true
        },
        "sequence": {
            "mirrorActors": true,
            "bottomMarginAdj": true,
            "useMaxWidth": true,
            "rightAngles": false,
            "showSequenceNumbers": false
        },
        "gantt": {
            "titleTopMargin": 25,
            "barHeight": 20,
            "barGap": 4,
            "topPadding": 50,
            "leftPadding": 75,
            "gridLineStartPadding": 35,
            "fontSize": 11,
            "sectionFontSize": 11,
            "numberSectionStyles": 4
        }
    }'''
    
    with open('/tmp/mermaid-config.json', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Successfully rendered {output_path}")
        
        # Clean up temp files
        os.remove(temp_mmd_file)
        os.remove('/tmp/mermaid-config.json')
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error rendering {output_path}: {e}")
        print(f"Stderr: {e.stderr}")
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
    
    # Extract diagrams
    diagrams = extract_mermaid_diagrams(markdown_file)
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
        
        # Create enhanced mermaid code
        enhanced_code = create_enhanced_mermaid_code(diagram['code'], diagram_index)
        
        # Output path
        output_path = output_base / output_dir / f"figure_{diagram_index:02d}.png"
        
        # Render diagram
        if render_mermaid_to_image(enhanced_code, str(output_path), diagram_index):
            success_count += 1
        else:
            print(f"Failed to render diagram {diagram_index}")
    
    print(f"\nSuccessfully rendered {success_count}/{len(diagrams)} diagrams")
    print(f"Output directory: {output_base}")
    
    # Create index file
    create_index_file(output_base, diagrams, dir_mapping)

def create_index_file(output_base, diagrams, dir_mapping):
    """Create an index file listing all rendered diagrams."""
    
    index_path = output_base / "index.md"
    
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write("# RadioDiff Standardized Mermaid Diagrams\n\n")
        f.write("This directory contains all mermaid diagrams from the merged report, rendered as high-quality PNG images with standardized styling.\n\n")
        
        # Group by directory
        by_dir = {}
        for diagram in diagrams:
            dir_name = dir_mapping.get(diagram['index'], "architecture")
            if dir_name not in by_dir:
                by_dir[dir_name] = []
            by_dir[dir_name].append(diagram)
        
        # Write each section
        for dir_name, dir_diagrams in by_dir.items():
            f.write(f"## {dir_name.title()}\n\n")
            
            for diagram in dir_diagrams:
                f.write(f"### {diagram['title']}\n")
                f.write(f"![{diagram['title']}]({dir_name}/figure_{diagram['index']:02d}.png)\n\n")
        
        f.write("## Rendering Details\n\n")
        f.write("- **Tool**: mermaid-cli (mmdc)\n")
        f.write("- **Theme**: Default with custom styling\n")
        f.write("- **Dimensions**: 1200x800 pixels\n")
        f.write("- **Format**: PNG with transparent background\n")
        f.write("- **Styling**: Consistent color scheme and typography\n")

if __name__ == "__main__":
    main()