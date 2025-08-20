#!/usr/bin/env python3
"""
Replace mermaid text with rendered images in the standardized report
"""

import re
import os

def replace_mermaid_with_images(report_path, images_dir):
    """Replace mermaid blocks with rendered images."""
    
    # Read the original report
    with open(report_path, 'r') as f:
        content = f.read()
    
    # Find all mermaid blocks and replace them
    pattern = r'(\*Note: Mermaid diagram could not be rendered due to formatting issues\. Original code below:\*\n\n)?```mermaid\n(.*?)\n```'
    
    diagram_count = 0
    
    def replace_function(match):
        nonlocal diagram_count
        diagram_count += 1
        
        # Check if we have a rendered image for this diagram
        image_path = f"{images_dir}/diagram_{diagram_count}.png"
        
        if os.path.exists(image_path):
            # Return image reference
            return f"![Figure {diagram_count}: RadioDiff Architecture Diagram]({image_path})\n\n*Figure {diagram_count}: Rendered mermaid diagram*"
        else:
            # Keep original if no image available
            return match.group(0)
    
    # Replace all mermaid blocks
    updated_content = re.sub(pattern, replace_function, content, flags=re.DOTALL)
    
    return updated_content, diagram_count

def create_appendix_with_originals(original_mermaid_dir, rendered_images_dir, output_dir):
    """Create an appendix with original mermaid code and rendered images."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all mermaid files
    mmd_files = [f for f in os.listdir(original_mermaid_dir) if f.endswith('.mmd')]
    mmd_files.sort()
    
    appendix_content = "# Appendix: Original Mermaid Diagrams\n\n"
    appendix_content += "This appendix contains the original mermaid code and rendered images for all diagrams in the report.\n\n"
    
    for mmd_file in mmd_files:
        diagram_num = mmd_file.replace('diagram_', '').replace('.mmd', '')
        
        # Read original mermaid code
        mmd_path = os.path.join(original_mermaid_dir, mmd_file)
        with open(mmd_path, 'r') as f:
            original_code = f.read()
        
        # Check for rendered image
        image_path = f"{rendered_images_dir}/diagram_{diagram_num}.png"
        has_image = os.path.exists(image_path)
        
        appendix_content += f"## Diagram {diagram_num}\n\n"
        
        if has_image:
            appendix_content += f"![Diagram {diagram_num}]({image_path})\n\n"
        
        appendix_content += "### Original Mermaid Code\n\n"
        appendix_content += "```mermaid\n"
        appendix_content += original_code
        appendix_content += "\n```\n\n"
        appendix_content += "---\n\n"
    
    # Write appendix
    appendix_path = os.path.join(output_dir, "appendix_mermaid_diagrams.md")
    with open(appendix_path, 'w') as f:
        f.write(appendix_content)
    
    return appendix_path

def main():
    report_path = 'RADIODIFF_MERGED_STANDARDIZED_REPORT.md'
    images_dir = 'radiodiff_standardized_report_images_final'
    original_mermaid_dir = 'radiodiff_standardized_report_mermaid'
    output_report_path = 'RADIODIFF_MERGED_STANDARDIZED_REPORT_WITH_IMAGES.md'
    appendix_dir = 'radiodiff_standardized_report_appendix'
    
    print("🔄 RadioDiff Standardized Report - Replace Mermaid with Images")
    print("=" * 70)
    
    # Replace mermaid blocks with images
    print(f"\n1. Replacing mermaid blocks with images...")
    updated_content, diagram_count = replace_mermaid_with_images(report_path, images_dir)
    
    print(f"   Processed {diagram_count} mermaid diagrams")
    
    # Write updated report
    with open(output_report_path, 'w') as f:
        f.write(updated_content)
    
    print(f"   ✅ Updated report saved to: {output_report_path}")
    
    # Create appendix with originals
    print(f"\n2. Creating appendix with original mermaid code...")
    appendix_path = create_appendix_with_originals(original_mermaid_dir, images_dir, appendix_dir)
    
    print(f"   ✅ Appendix created: {appendix_path}")
    
    # Update the main report to include appendix reference
    print(f"\n3. Adding appendix reference to main report...")
    
    with open(output_report_path, 'r') as f:
        final_content = f.read()
    
    # Add appendix reference at the end
    appendix_ref = f"\n\n## Appendix\n\nFor the original mermaid code and additional details, see the [appendix file]({appendix_path}).\n"
    final_content += appendix_ref
    
    with open(output_report_path, 'w') as f:
        f.write(final_content)
    
    print(f"   ✅ Final report updated with appendix reference")
    
    print(f"\n🎉 Complete! Report updated with rendered images.")
    print(f"📄 Main report: {output_report_path}")
    print(f"📋 Appendix: {appendix_path}")

if __name__ == "__main__":
    main()
