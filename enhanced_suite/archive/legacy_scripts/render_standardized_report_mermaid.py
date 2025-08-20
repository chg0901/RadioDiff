#!/usr/bin/env python3
"""
Render mermaid diagrams from the standardized report
"""

import os
import subprocess
from pathlib import Path

def render_mermaid_diagrams(input_dir, output_dir):
    """Render all mermaid files to PNG images."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .mmd files
    mmd_files = [f for f in os.listdir(input_dir) if f.endswith('.mmd')]
    mmd_files.sort()
    
    print(f"Found {len(mmd_files)} mermaid files to render")
    
    for mmd_file in mmd_files:
        mmd_path = os.path.join(input_dir, mmd_file)
        png_name = mmd_file.replace('.mmd', '.png')
        png_path = os.path.join(output_dir, png_name)
        
        print(f"Rendering {mmd_file} to PNG...")
        
        # Use mermaid-cli to render
        cmd = [
            'mmdc',
            '-i', mmd_path,
            '-o', png_path,
            '-t', 'default',
            '-w', '1400',
            '-H', '1000',
            '-b', 'white'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Successfully rendered: {png_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to render {mmd_file}: {e}")
            print(f"Error output: {e.stderr}")
            
            # Try alternative approach with node
            try:
                print("Trying alternative rendering approach...")
                alt_cmd = [
                    'node', '-e',
                    f'''
                    const {{ render }} = require("mermaid-cli");
                    render("{mmd_path}", "{png_path}", {{
                        theme: "default",
                        width: 1400,
                        height: 1000,
                        backgroundColor: "white"
                    }});
                    '''
                ]
                result = subprocess.run(alt_cmd, capture_output=True, text=True, check=True)
                print(f"✅ Successfully rendered with node: {png_path}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"❌ Alternative rendering also failed for {mmd_file}")

def main():
    input_dir = 'radiodiff_standardized_report_mermaid'
    output_dir = 'radiodiff_standardized_report_images'
    
    print("🎨 RadioDiff Standardized Report Mermaid Renderer")
    print("=" * 60)
    
    # Render to PNG
    print(f"\nRendering mermaid diagrams to PNG images...")
    render_mermaid_diagrams(input_dir, output_dir)
    
    print(f"\n✅ Complete! All diagrams rendered to {output_dir}/")
    print("📁 Generated files:")
    
    # List generated files
    if os.path.exists(output_dir):
        for file in sorted(os.listdir(output_dir)):
            if file.endswith('.png'):
                print(f"   - {output_dir}/{file}")

if __name__ == "__main__":
    main()