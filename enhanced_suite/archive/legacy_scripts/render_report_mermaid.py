#!/usr/bin/env python3
"""
Script to render mermaid diagrams from RADIODIFF_MERGED_STANDARDIZED_REPORT.md
with 16:9 aspect ratio and proper font sizes, handling LaTeX equations.
"""

import re
import os
import subprocess
import tempfile
from pathlib import Path

def extract_mermaid_diagrams(md_file):
    """Extract all mermaid diagrams from markdown file."""
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all mermaid code blocks
    pattern = r'```mermaid\n(.*?)\n```'
    matches = re.findall(pattern, content, re.DOTALL)
    
    diagrams = []
    for i, match in enumerate(matches):
        diagrams.append({
            'id': f'diagram_{i+1}',
            'code': match.strip(),
            'index': i
        })
    
    return diagrams

def clean_mermaid_code(mermaid_code):
    """Clean mermaid code, handle LaTeX equations and special characters."""
    
    cleaned_code = mermaid_code
    
    # Simple LaTeX replacements (direct string replacements)
    latex_replacements = {
        '\\mathcal{N}': 'N',
        '\\mathbf{I}': 'I',
        '\\bar{\\alpha}_t': 'ᾱ_t',
        '\\sqrt{\\bar{\\alpha}_t}': '√(ᾱ_t)',
        '\\sqrt{1-\\bar{\\alpha}_t}': '√(1-ᾱ_t)',
        '\\epsilon': 'ε',
        '\\theta': 'θ',
        '\\phi': 'φ',
        '\\mu': 'μ',
        '\\sigma': 'σ',
        '\\mathbb{E}': 'E',
        '\\alpha': 'α',
        '\\beta': 'β',
        '\\gamma': 'γ',
        '\\delta': 'δ',
        '\\lambda': 'λ',
        '\\pi': 'π',
        '\\sum': 'Σ',
        '\\prod': 'Π',
        '\\int': '∫',
        '\\partial': '∂',
        '\\nabla': '∇',
        '\\Delta': 'Δ',
        '\\infty': '∞',
        '\\pm': '±',
        '\\leq': '≤',
        '\\geq': '≥',
        '\\neq': '≠',
        '\\approx': '≈',
        '\\sim': '∼',
        '\\cdot': '·',
        '\\circ': '∘',
        '\\perp': '⊥',
        '\\parallel': '∥',
        '\\angle': '∠',
        '\\degree': '°',
        '\\prime': '′',
        '\\Prime': '″',
        '\\leftarrow': '←',
        '\\rightarrow': '→',
        '\\Leftarrow': '⇐',
        '\\Rightarrow': '⇒',
        '\\times': '×',
        '\\in': '∈',
        '\\|': '|',
        '\\^2': '²',
        '\\^3': '³',
    }
    
    # Apply LaTeX replacements
    for pattern, replacement in latex_replacements.items():
        cleaned_code = cleaned_code.replace(pattern, replacement)
    
    # Handle special HTML entities
    html_entities = {
        '&lt;': '<',
        '&gt;': '>',
        '&amp;': '&',
        '&quot;': '"',
        '&apos;': "'",
        '&nbsp;': ' ',
    }
    
    for entity, char in html_entities.items():
        cleaned_code = cleaned_code.replace(entity, char)
    
    # Remove problematic mermaid syntax that might cause issues
    cleaned_code = re.sub(r'style.*?fill:.*?;', '', cleaned_code, flags=re.DOTALL)
    cleaned_code = re.sub(r'style.*?stroke:.*?;', '', cleaned_code, flags=re.DOTALL)
    cleaned_code = re.sub(r'style.*?stroke-width:.*?;', '', cleaned_code, flags=re.DOTALL)
    cleaned_code = re.sub(r'style.*?font-weight:.*?;', '', cleaned_code, flags=re.DOTALL)
    
    return cleaned_code

def render_mermaid_diagram(mermaid_code, output_path, width=1200, height=675):
    """Render a mermaid diagram using mmdc with specified dimensions."""
    
    # Clean the mermaid code
    cleaned_code = clean_mermaid_code(mermaid_code)
    
    # Create temporary mermaid file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as temp_file:
        temp_file.write(cleaned_code)
        temp_mmd_path = temp_file.name
    
    try:
        # Render using mmdc with 16:9 aspect ratio
        cmd = [
            'mmdc',
            '-i', temp_mmd_path,
            '-o', output_path,
            '-w', str(width),
            '-H', str(height),
            '-t', 'default',
            '-b', 'white',
            '-s', '2',
            '-c', 'mermaid_config.json'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error rendering diagram: {result.stderr}")
            return False
        
        return True
    
    except Exception as e:
        print(f"Error rendering diagram: {e}")
        return False
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_mmd_path):
            os.unlink(temp_mmd_path)

def create_mermaid_config():
    """Create mermaid configuration file for consistent styling."""
    config = {
        "theme": "default",
        "themeVariables": {
            "fontFamily": "sans-serif",
            "fontSize": "18px",
            "primaryColor": "#e3f2fd",
            "primaryTextColor": "#000000",
            "primaryBorderColor": "#01579b",
            "lineColor": "#666666",
            "secondaryColor": "#f3e5f5",
            "tertiaryColor": "#fff3e0",
            "background": "white",
            "primaryEdgeColor": "#333333",
            "clusterBkg": "#f5f5f5",
            "clusterBorder": "#666666"
        },
        "flowchart": {
            "htmlLabels": True,
            "curve": "basis",
            "padding": 20,
            "nodeSpacing": 50,
            "rankSpacing": 50
        },
        "sequence": {
            "diagramMarginX": 50,
            "diagramMarginY": 10,
            "actorMargin": 50,
            "width": 150,
            "height": 65,
            "boxMargin": 10,
            "boxTextMargin": 5,
            "noteMargin": 10,
            "messageMargin": 35,
            "mirrorActors": True,
            "bottomMarginAdj": 1,
            "useMaxWidth": True,
            "rightAngles": False,
            "showSequenceNumbers": False
        },
        "gantt": {
            "titleTopMargin": 25,
            "barHeight": 20,
            "barGap": 4,
            "topPadding": 50,
            "leftPadding": 75,
            "gridLineStartPadding": 35,
            "fontSize": 16,
            "sectionFontSize": 16,
            "numberSectionStyles": 4,
            "axisFormat": "%Y-%m-%d"
        }
    }
    
    import json
    with open('mermaid_config.json', 'w') as f:
        json.dump(config, f, indent=2)

def main():
    """Main function to extract and render all mermaid diagrams."""
    
    # Create output directory
    output_dir = Path("radiodiff_rendered_mermaid")
    output_dir.mkdir(exist_ok=True)
    
    # Create mermaid config
    create_mermaid_config()
    
    # Extract diagrams
    md_file = "RADIODIFF_MERGED_STANDARDIZED_REPORT.md"
    diagrams = extract_mermaid_diagrams(md_file)
    
    print(f"Found {len(diagrams)} mermaid diagrams")
    
    # Render each diagram
    successful_renders = []
    
    for diagram in diagrams:
        output_path = output_dir / f"{diagram['id']}.png"
        
        print(f"Rendering {diagram['id']}...")
        
        if render_mermaid_diagram(diagram['code'], str(output_path)):
            successful_renders.append({
                'id': diagram['id'],
                'path': str(output_path),
                'index': diagram['index']
            })
            print(f"✓ Successfully rendered {diagram['id']}")
        else:
            print(f"✗ Failed to render {diagram['id']}")
    
    # Clean up config file
    if os.path.exists('mermaid_config.json'):
        os.unlink('mermaid_config.json')
    
    print(f"\nSuccessfully rendered {len(successful_renders)} out of {len(diagrams)} diagrams")
    
    # Create a mapping file for reference
    mapping_file = output_dir / "diagram_mapping.txt"
    with open(mapping_file, 'w') as f:
        f.write("Mermaid Diagram Mapping\n")
        f.write("=" * 50 + "\n\n")
        for i, render in enumerate(successful_renders):
            f.write(f"Diagram {i+1}: {render['id']}\n")
            f.write(f"File: {render['path']}\n")
            f.write(f"Original Index: {render['index']}\n\n")
    
    return successful_renders

if __name__ == "__main__":
    main()