#!/usr/bin/env python3
"""
Extract and render mermaid diagrams from README_RadioDiff_VAE.md
"""

import re
import os
import subprocess
from pathlib import Path

def extract_mermaid_diagrams(readme_path):
    """Extract all mermaid diagrams from the README file."""
    
    with open(readme_path, 'r') as f:
        content = f.read()
    
    # Find all mermaid code blocks
    pattern = r'```mermaid\n(.*?)\n```'
    matches = re.findall(pattern, content, re.DOTALL)
    
    diagrams = []
    for i, match in enumerate(matches):
        diagrams.append({
            'index': i + 1,
            'code': match.strip(),
            'filename': f'diagram_{i+1}.mmd'
        })
    
    return diagrams

def create_mermaid_files(diagrams, output_dir):
    """Create individual mermaid files for each diagram."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    for diagram in diagrams:
        filepath = os.path.join(output_dir, diagram['filename'])
        with open(filepath, 'w') as f:
            f.write(diagram['code'])
        print(f"Created: {filepath}")

def render_mermaid_to_png(diagrams, output_dir):
    """Render mermaid files to PNG images."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    for diagram in diagrams:
        mmd_path = os.path.join(output_dir, diagram['filename'])
        png_path = os.path.join(output_dir, f"diagram_{diagram['index']}.png")
        
        print(f"Rendering {diagram['filename']} to PNG...")
        
        # Use mermaid-cli to render
        cmd = [
            'mmdc',
            '-i', mmd_path,
            '-o', png_path,
            '-t', 'default',
            '-w', '1200',
            '-H', '800',
            '-b', 'white'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Successfully rendered: {png_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to render {diagram['filename']}: {e}")
            print(f"Error output: {e.stderr}")

def update_readme_with_images(readme_path, output_dir):
    """Update README to include image references after mermaid blocks."""
    
    with open(readme_path, 'r') as f:
        content = f.read()
    
    # Replace mermaid blocks with image references
    pattern = r'```mermaid\n(.*?)\n```'
    
    def replace_with_image(match):
        # Find which diagram this is
        mermaid_code = match.group(1).strip()
        
        # Count how many diagrams we've seen so far
        diagram_count = len(re.findall(pattern, content[:match.start()], re.DOTALL)) + 1
        
        image_path = f"./{output_dir}/diagram_{diagram_count}.png"
        
        return f"""```mermaid
{mermaid_code}
```

![Diagram {diagram_count}]({image_path})

*Figure {diagram_count}: Mermaid diagram rendered as image*"""
    
    updated_content = re.sub(pattern, replace_with_image, content, flags=re.DOTALL)
    
    # Write updated content
    with open(readme_path, 'w') as f:
        f.write(updated_content)
    
    print("✅ Updated README with image references")

def main():
    readme_path = '/home/cine/Documents/Github/RadioDiff/README_RadioDiff_VAE.md'
    output_dir = 'mermaid_vis'
    
    print("🎨 Mermaid Diagram Renderer")
    print("=" * 50)
    
    # Extract diagrams
    print("1. Extracting mermaid diagrams...")
    diagrams = extract_mermaid_diagrams(readme_path)
    print(f"   Found {len(diagrams)} mermaid diagrams")
    
    # Create mermaid files
    print(f"\n2. Creating mermaid files in {output_dir}/...")
    create_mermaid_files(diagrams, output_dir)
    
    # Render to PNG
    print(f"\n3. Rendering mermaid diagrams to PNG images...")
    render_mermaid_to_png(diagrams, output_dir)
    
    # Update README
    print(f"\n4. Updating README with image references...")
    update_readme_with_images(readme_path, output_dir)
    
    print(f"\n✅ Complete! All diagrams rendered to {output_dir}/")
    print("📁 Generated files:")
    
    # List generated files
    for file in os.listdir(output_dir):
        if file.endswith('.png'):
            print(f"   - {output_dir}/{file}")

if __name__ == "__main__":
    main()