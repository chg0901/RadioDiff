#!/usr/bin/env python3
"""
Mermaid Renderer for RadioDiff Comprehensive Analysis Report
This script renders Mermaid diagrams and creates a final report with embedded images.
"""

import os
import subprocess
import tempfile
import base64
from pathlib import Path
import re

def render_mermaid_to_base64(mermaid_code, width=1280, height=720):
    """
    Render Mermaid code to base64 encoded image using mmdc
    """
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as mmd_file:
        mmd_file.write(mermaid_code)
        mmd_path = mmd_file.name
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as png_file:
        png_path = png_file.name
    
    try:
        # Render using mmdc
        cmd = [
            'mmdc',
            '-i', mmd_path,
            '-o', png_path,
            '-w', str(width),
            '-H', str(height),
            '-t', 'default',
            '-b', 'white'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Convert to base64
            with open(png_path, 'rb') as f:
                image_data = f.read()
                base64_data = base64.b64encode(image_data).decode('utf-8')
                return f"data:image/png;base64,{base64_data}"
        else:
            print(f"Error rendering Mermaid: {result.stderr}")
            return None
    finally:
        # Clean up temporary files
        try:
            os.unlink(mmd_path)
            os.unlink(png_path)
        except:
            pass
    
    return None

def extract_mermaid_blocks(markdown_content):
    """
    Extract Mermaid code blocks from markdown content
    """
    pattern = r'```mermaid\n(.*?)\n```'
    matches = re.findall(pattern, markdown_content, re.DOTALL)
    return matches

def create_enhanced_report():
    """
    Create enhanced report with rendered Mermaid diagrams
    """
    # Read the mermaid visualization report
    report_path = Path("/home/cine/Documents/Github/RadioDiff/RADIODIFF_MERMAID_VISUALIZATION_REPORT.md")
    
    if not report_path.exists():
        print("Mermaid visualization report not found!")
        return
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract all Mermaid blocks
    mermaid_blocks = extract_mermaid_blocks(content)
    
    print(f"Found {len(mermaid_blocks)} Mermaid diagrams to render")
    
    # Create enhanced content
    enhanced_content = content
    
    # Replace each Mermaid block with rendered image
    for i, mermaid_code in enumerate(mermaid_blocks):
        print(f"Rendering diagram {i+1}/{len(mermaid_blocks)}...")
        
        # Render the diagram
        image_data = render_mermaid_to_base64(mermaid_code)
        
        if image_data:
            # Create image markdown
            diagram_names = ["Training Architecture", "Data Flow", "Training Process"]
            diagram_name = diagram_names[i] if i < len(diagram_names) else f"Diagram {i+1}"
            image_markdown = f"""
![RadioDiff {diagram_name} Diagram]({image_data})

*Figure {i+1}: RadioDiff {diagram_name} - Complete visualization of the training pipeline*
"""
            
            # Replace the mermaid block
            enhanced_content = enhanced_content.replace(
                f'```mermaid\n{mermaid_code}\n```',
                image_markdown
            )
        else:
            print(f"Failed to render diagram {i+1}")
    
    # Add a note about the rendered diagrams
    enhanced_content += """

---

## 📊 Rendered Diagrams Summary

This report includes **{} rendered Mermaid diagrams** showing:

1. **Training Architecture Diagram** - Complete training pipeline showing configuration loading, model initialization (VAE + U-Net), data pipeline, and training process
2. **Data Flow Diagram** - Detailed data pipeline from input through processing to model flow and output generation
3. **Training Process Diagram** - Comprehensive training process including setup, loop, optimization, and monitoring components

All diagrams are rendered with **16:9 aspect ratio** (1280×720) and **18px font size** for optimal readability and include professional styling with high contrast for better visibility.

*Diagrams rendered using Mermaid CLI with custom styling for RadioDiff visualization*
""".format(len(mermaid_blocks))
    
    # Save the enhanced report
    enhanced_path = Path("/home/cine/Documents/Github/RadioDiff/RADIODIFF_MERMAID_REPORT_WITH_IMAGES.md")
    
    with open(enhanced_path, 'w', encoding='utf-8') as f:
        f.write(enhanced_content)
    
    print(f"Enhanced report saved to: {enhanced_path}")
    return enhanced_path

def create_mermaid_reference():
    """
    Create a separate file with all Mermaid diagrams for reference
    """
    report_path = Path("/home/cine/Documents/Github/RadioDiff/RADIODIFF_MERMAID_VISUALIZATION_REPORT.md")
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    mermaid_blocks = extract_mermaid_blocks(content)
    
    reference_content = """# RadioDiff Mermaid Diagrams Reference

This file contains all Mermaid diagrams from the comprehensive analysis report for reference and debugging.

---

"""
    
    for i, mermaid_code in enumerate(mermaid_blocks):
        reference_content += f"## Diagram {i+1}\n\n"
        reference_content += "```mermaid\n"
        reference_content += mermaid_code
        reference_content += "\n```\n\n"
        reference_content += "---\n\n"
    
    reference_path = Path("/home/cine/Documents/Github/RadioDiff/MERMAID_DIAGRAMS_REFERENCE.md")
    
    with open(reference_path, 'w', encoding='utf-8') as f:
        f.write(reference_content)
    
    print(f"Mermaid reference saved to: {reference_path}")
    return reference_path

if __name__ == "__main__":
    print("RadioDiff Mermaid Renderer")
    print("=" * 50)
    
    # Check if mmdc is available
    try:
        subprocess.run(['mmdc', '--version'], capture_output=True, check=True)
        print("✅ Mermaid CLI (mmdc) is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Mermaid CLI (mmdc) not found. Please install:")
        print("   npm install -g @mermaid-js/mermaid-cli")
        exit(1)
    
    # Create enhanced report
    enhanced_report = create_enhanced_report()
    
    # Create mermaid reference
    mermaid_reference = create_mermaid_reference()
    
    print("\n" + "=" * 50)
    print("✅ Rendering complete!")
    print(f"📄 Enhanced report: {enhanced_report}")
    print(f"📋 Mermaid reference: {mermaid_reference}")
    print("=" * 50)