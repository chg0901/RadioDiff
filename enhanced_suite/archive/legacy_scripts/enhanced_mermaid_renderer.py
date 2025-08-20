#!/usr/bin/env python3
"""
Enhanced mermaid renderer with better styling, colors, and shapes
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
            'filename': f'enhanced_diagram_{i+1}.mmd'
        })
    
    return diagrams

def enhance_mermaid_styling(mermaid_code):
    """Enhance mermaid code with better styling."""
    
    # Add enhanced styling to the beginning of the code
    enhanced_code = f"""%%{{
  init: {{
    'theme': 'default',
    'themeVariables': {{
      'primaryColor': '#ff6b6b',
      'primaryTextColor': '#2c3e50',
      'primaryBorderColor': '#3498db',
      'lineColor': '#34495e',
      'secondaryColor': '#f8f9fa',
      'tertiaryColor': '#e9ecef',
      'clusterBkg': '#ffffff',
      'clusterBorder': '#dee2e6',
      'fontSize': '16px',
      'fontFamily': 'Arial, sans-serif'
    }}
  }}
}}%%

{mermaid_code}"""
    
    return enhanced_code

def create_enhanced_mermaid_files(diagrams, output_dir):
    """Create enhanced individual mermaid files for each diagram."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    for diagram in diagrams:
        filepath = os.path.join(output_dir, diagram['filename'])
        enhanced_code = enhance_mermaid_styling(diagram['code'])
        
        with open(filepath, 'w') as f:
            f.write(enhanced_code)
        print(f"Created enhanced: {filepath}")

def render_mermaid_to_enhanced_png(diagrams, output_dir):
    """Render mermaid files to enhanced PNG images with better quality."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    for diagram in diagrams:
        mmd_path = os.path.join(output_dir, diagram['filename'])
        png_path = os.path.join(output_dir, f"enhanced_diagram_{diagram['index']}.png")
        
        print(f"Rendering enhanced {diagram['filename']} to PNG...")
        
        # Use mermaid-cli to render with enhanced settings
        cmd = [
            'mmdc',
            '-i', mmd_path,
            '-o', png_path,
            '-t', 'default',
            '-w', '1400',  # Higher resolution
            '-H', '1000',  # Higher resolution
            '-b', 'white'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Successfully rendered enhanced: {png_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to render {diagram['filename']}: {e}")
            print(f"Error output: {e.stderr}")

def create_enhanced_readme_with_images(readme_path, output_dir):
    """Create enhanced README with better image references."""
    
    with open(readme_path, 'r') as f:
        content = f.read()
    
    # Replace mermaid blocks with enhanced image references
    pattern = r'```mermaid\n(.*?)\n```'
    
    def replace_with_enhanced_image(match):
        # Find which diagram this is
        mermaid_code = match.group(1).strip()
        
        # Count how many diagrams we've seen so far
        diagram_count = len(re.findall(pattern, content[:match.start()], re.DOTALL)) + 1
        
        image_path = f"./{output_dir}/enhanced_diagram_{diagram_count}.png"
        
        return f"""```mermaid
{mermaid_code}
```

<div style="text-align: center; margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 8px; border-left: 4px solid #007bff;">
    <img src="{image_path}" alt="Enhanced Diagram {diagram_count}" style="max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    <p style="margin-top: 10px; font-size: 14px; color: #6c757d; font-style: italic;">
        <strong>Figure {diagram_count}:</strong> Enhanced mermaid diagram with improved styling and resolution
    </p>
</div>"""
    
    updated_content = re.sub(pattern, replace_with_enhanced_image, content, flags=re.DOTALL)
    
    # Write enhanced content
    enhanced_readme_path = readme_path.replace('.md', '_ENHANCED.md')
    with open(enhanced_readme_path, 'w') as f:
        f.write(updated_content)
    
    print(f"✅ Created enhanced README: {enhanced_readme_path}")

def create_html_viewer(output_dir):
    """Create an HTML viewer for all enhanced diagrams."""
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RadioDiff VAE Enhanced Diagrams Viewer</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .diagram-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .diagram-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }}
        .diagram-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
        .diagram-card img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
            margin-bottom: 10px;
        }}
        .diagram-title {{
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        .diagram-description {{
            font-size: 14px;
            color: #6c757d;
            line-height: 1.4;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            background-color: #e9ecef;
            border-radius: 8px;
            color: #6c757d;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎨 RadioDiff VAE Enhanced Diagrams Viewer</h1>
        <p>Interactive viewer for all enhanced mermaid diagrams with improved styling and resolution</p>
    </div>
    
    <div class="diagram-grid">
"""

    # Add diagram cards
    diagram_descriptions = [
        "Loss function architecture overview showing the complete VAE-GAN training pipeline",
        "Reconstruction loss pipeline with L1 and MSE components for input-output comparison",
        "Perceptual loss integration using LPIPS network for feature space comparison",
        "Variational lower bound calculation with learnable variance parameter",
        "KL divergence loss computation for latent space regularization",
        "Discriminator network architecture for adversarial training",
        "Generator adversarial training component for realistic generation",
        "Discriminator training with hinge loss for real/fake classification",
        "Adaptive weight calculation using gradient norm balancing",
        "Phase 1 VAE pre-training with reconstruction and KL losses",
        "Phase 2 VAE-GAN training with full adversarial components",
        "Comprehensive loss metrics logging system",
        "Edge-aware loss components for radio astronomy applications",
        "Radio propagation physics integration with DPM2IRT4 modeling",
        "Two-phase training strategy with progressive learning",
        "Loss weight dynamics and evolution during training"
    ]
    
    for i in range(1, 17):  # Assuming 16 diagrams
        if i <= len(diagram_descriptions):
            html_content += f"""
        <div class="diagram-card">
            <div class="diagram-title">Diagram {i}: {diagram_descriptions[i-1].split(':')[0]}</div>
            <img src="enhanced_diagram_{i}.png" alt="Enhanced Diagram {i}">
            <div class="diagram-description">{diagram_descriptions[i-1]}</div>
        </div>"""
    
    html_content += """
    </div>
    
    <div class="footer">
        <p>Generated with enhanced mermaid rendering • RadioDiff VAE Documentation</p>
    </div>
</body>
</html>"""
    
    html_path = os.path.join(output_dir, 'enhanced_diagrams_viewer.html')
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Created HTML viewer: {html_path}")

def main():
    readme_path = '/home/cine/Documents/Github/RadioDiff/VAE_LOSS_FUNCTIONS_DETAILED_REPORT.md'
    output_dir = 'enhanced_mermaid_vis'
    
    print("🎨 Enhanced Mermaid Diagram Renderer")
    print("=" * 60)
    
    # Extract diagrams
    print("1. Extracting mermaid diagrams...")
    diagrams = extract_mermaid_diagrams(readme_path)
    print(f"   Found {len(diagrams)} mermaid diagrams")
    
    # Create enhanced mermaid files
    print(f"\n2. Creating enhanced mermaid files in {output_dir}/...")
    create_enhanced_mermaid_files(diagrams, output_dir)
    
    # Render to PNG
    print(f"\n3. Rendering enhanced mermaid diagrams to PNG images...")
    render_mermaid_to_enhanced_png(diagrams, output_dir)
    
    # Create enhanced README
    print(f"\n4. Creating enhanced README with better image references...")
    create_enhanced_readme_with_images(readme_path, output_dir)
    
    # Create HTML viewer
    print(f"\n5. Creating interactive HTML viewer...")
    create_html_viewer(output_dir)
    
    print(f"\n✅ Enhanced rendering complete! All diagrams saved to {output_dir}/")
    print("📁 Generated files:")
    
    # List generated files
    png_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    for file in png_files:
        print(f"   - {output_dir}/{file}")
    
    print(f"\n🌐 View enhanced diagrams: file://{os.path.abspath(output_dir)}/enhanced_diagrams_viewer.html")

if __name__ == "__main__":
    main()