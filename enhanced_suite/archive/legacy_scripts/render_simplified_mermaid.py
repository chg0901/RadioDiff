#!/usr/bin/env python3
"""
Generate simplified mermaid diagrams for RadioDiff with clearer structure and less detail.
Based on MERMAID_DIAGRAMS_REFERENCE.md template format.
"""

import subprocess
import os
from pathlib import Path

def create_simplified_mermaid_diagrams():
    """Create simplified mermaid diagrams with clearer structure."""
    
    output_dir = Path("mermaid_vis_simplified")
    output_dir.mkdir(exist_ok=True)
    
    # Simplified diagrams following the template format
    diagrams = [
        {
            "name": "system_overview",
            "title": "RadioDiff System Overview",
            "content": """%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#1e40af', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#3b82f6', 'lineColor': '#60a5fa', 'secondaryColor': '#7c3aed', 'tertiaryColor': '#10b981', 'clusterBkg': '#f3f4f6', 'clusterBorder': '#9ca3af', 'fontSize': '18px'}}}%%
graph TD
    A["RadioDiff System"] --> B["Data Input"]
    A --> C["VAE Processing"]
    A --> D["Conditional U-Net"]
    A --> E["Diffusion Process"]
    
    B --> B1["Radio Signals"]
    B --> B2["Building Layout"]
    B --> B3["Environmental Data"]
    
    C --> C1["Encoder: 320×320×1"]
    C --> C2["Latent Space: 3×80×80"]
    C --> C3["Decoder: 320×320×1"]
    
    D --> D1["Swin Transformer"]
    D --> D2["Window Attention"]
    D --> D3["Fourier Features"]
    
    E --> E1["Forward Process"]
    E --> E2["Reverse Process"]
    E --> E3["Sampling"]
    
    classDef primaryBox fill:#1e40af,stroke:#3b82f6,stroke-width:2px,color:#ffffff
    classDef secondaryBox fill:#7c3aed,stroke:#8b5cf6,stroke-width:2px,color:#ffffff
    classDef tertiaryBox fill:#10b981,stroke:#34d399,stroke-width:2px,color:#ffffff
    classDef quaternaryBox fill:#ef4444,stroke:#f87171,stroke-width:2px,color:#ffffff
    classDef quinaryBox fill:#f59e0b,stroke:#fbbf24,stroke-width:2px,color:#ffffff
    
    class A primaryBox
    class B secondaryBox
    class C tertiaryBox
    class D quaternaryBox
    class E quinaryBox
"""
        },
        {
            "name": "vae_architecture",
            "title": "VAE Architecture",
            "content": """%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#7c3aed', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#8b5cf6', 'lineColor': '#a78bfa', 'secondaryColor': '#10b981', 'tertiaryColor': '#f59e0b', 'clusterBkg': '#f3f4f6', 'clusterBorder': '#9ca3af', 'fontSize': '18px'}}}%%
graph TD
    A["VAE Architecture"] --> B["Encoder Path"]
    A --> C["Latent Space"]
    A --> D["Decoder Path"]
    
    B --> B1["Input: 320×320×1"]
    B --> B2["Conv Layers: 128→512ch"]
    B --> B3["Downsampling"]
    B --> B4["μ, σ Parameters"]
    
    C --> C1["Latent: 3×80×80"]
    C --> C2["Reparameterization"]
    C --> C3["KL Divergence"]
    
    D --> D1["Upsampling"]
    D --> D2["Conv Layers: 512→128ch"]
    D --> D3["Output: 320×320×1"]
    D --> D4["Reconstruction"]
    
    classDef primaryBox fill:#7c3aed,stroke:#8b5cf6,stroke-width:2px,color:#ffffff
    classDef secondaryBox fill:#10b981,stroke:#34d399,stroke-width:2px,color:#ffffff
    classDef tertiaryBox fill:#f59e0b,stroke:#fbbf24,stroke-width:2px,color:#ffffff
    classDef quaternaryBox fill:#ef4444,stroke:#f87171,stroke-width:2px,color:#ffffff
    
    class A primaryBox
    class B secondaryBox
    class C tertiaryBox
    class D quaternaryBox
"""
        },
        {
            "name": "unet_structure",
            "title": "Conditional U-Net Structure",
            "content": """%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#10b981', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#34d399', 'lineColor': '#6ee7b7', 'secondaryColor': '#f59e0b', 'tertiaryColor': '#ef4444', 'clusterBkg': '#f3f4f6', 'clusterBorder': '#9ca3af', 'fontSize': '18px'}}}%%
graph TD
    A["U-Net Structure"] --> B["Input Processing"]
    A --> C["Encoder Path"]
    A --> D["Bottleneck"]
    A --> E["Decoder Path"]
    A --> F["Output"]
    
    B --> B1["Latent Input: 3×80×80"]
    B --> B2["Time Embedding"]
    B --> B3["Condition Input"]
    
    C --> C1["Level 1: 128×80×80"]
    C --> C2["Level 2: 256×40×40"]
    C --> C3["Level 3: 512×20×20"]
    C --> C4["Level 4: 512×10×10"]
    
    D --> D1["Swin Transformer"]
    D --> D2["Cross-Attention"]
    D --> D3["Feature Fusion"]
    
    E --> E1["Skip Connections"]
    E --> E2["Upsampling"]
    E --> E3["Feature Refinement"]
    
    F --> F1["Noise Prediction"]
    F --> F2["Output: 3×80×80"]
    
    classDef primaryBox fill:#10b981,stroke:#34d399,stroke-width:2px,color:#ffffff
    classDef secondaryBox fill:#f59e0b,stroke:#fbbf24,stroke-width:2px,color:#ffffff
    classDef tertiaryBox fill:#ef4444,stroke:#f87171,stroke-width:2px,color:#ffffff
    classDef quaternaryBox fill:#7c3aed,stroke:#8b5cf6,stroke-width:2px,color:#ffffff
    classDef quinaryBox fill:#1e40af,stroke:#3b82f6,stroke-width:2px,color:#ffffff
    
    class A primaryBox
    class B secondaryBox
    class C tertiaryBox
    class D quaternaryBox
    class E quinaryBox
    class F quaternaryBox
"""
        },
        {
            "name": "training_workflow",
            "title": "Training Workflow",
            "content": """%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#f59e0b', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#fbbf24', 'lineColor': '#fcd34d', 'secondaryColor': '#ef4444', 'tertiaryColor': '#7c3aed', 'clusterBkg': '#f3f4f6', 'clusterBorder': '#9ca3af', 'fontSize': '18px'}}}%%
graph TD
    A["Training Workflow"] --> B["Setup"]
    A --> C["Training Loop"]
    A --> D["Optimization"]
    A --> E["Evaluation"]
    
    B --> B1["Load Config"]
    B --> B2["Initialize Model"]
    B --> B3["Setup Data"]
    B --> B4["Configure Optimizer"]
    
    C --> C1["Load Batch"]
    C --> C2["Forward Pass"]
    C --> C3["Loss Calculation"]
    C --> C4["Backward Pass"]
    
    D --> D1["Gradient Accumulation"]
    D --> D2["Optimizer Step"]
    D --> D3["EMA Update"]
    D --> D4["LR Schedule"]
    
    E --> E1["Generate Samples"]
    E --> E2["Save Checkpoint"]
    E --> E3["Logging"]
    E --> E4["Metrics"]
    
    classDef primaryBox fill:#f59e0b,stroke:#fbbf24,stroke-width:2px,color:#ffffff
    classDef secondaryBox fill:#ef4444,stroke:#f87171,stroke-width:2px,color:#ffffff
    classDef tertiaryBox fill:#7c3aed,stroke:#8b5cf6,stroke-width:2px,color:#ffffff
    classDef quaternaryBox fill:#10b981,stroke:#34d399,stroke-width:2px,color:#ffffff
    classDef quinaryBox fill:#1e40af,stroke:#3b82f6,stroke-width:2px,color:#ffffff
    
    class A primaryBox
    class B secondaryBox
    class C tertiaryBox
    class D quaternaryBox
    class E quinaryBox
"""
        },
        {
            "name": "loss_components",
            "title": "Loss Components",
            "content": """%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ef4444', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#f87171', 'lineColor': '#fca5a5', 'secondaryColor': '#f59e0b', 'tertiaryColor': '#10b981', 'clusterBkg': '#f3f4f6', 'clusterBorder': '#9ca3af', 'fontSize': '18px'}}}%%
graph TD
    A["Loss Components"] --> B["Reconstruction"]
    A --> C["Regularization"]
    A --> D["Adversarial"]
    A --> E["Perceptual"]
    
    B --> B1["L2 Loss"]
    B --> B2["MSE"]
    B --> B3["Pixel-wise"]
    
    C --> C1["KL Divergence"]
    C --> C2["Latent Space"]
    C --> C3["Prevents Overfitting"]
    
    D --> D1["Discriminator"]
    D --> D2["Real/Fake"]
    D --> D3["Quality Enhancement"]
    
    E --> E1["LPIPS"]
    E --> E2["Feature Similarity"]
    E --> E3["Visual Quality"]
    
    A --> F["Total Loss"]
    F --> F1["Weighted Sum"]
    F --> F2["Multi-objective"]
    F --> F3["Optimization Target"]
    
    classDef primaryBox fill:#ef4444,stroke:#f87171,stroke-width:2px,color:#ffffff
    classDef secondaryBox fill:#f59e0b,stroke:#fbbf24,stroke-width:2px,color:#ffffff
    classDef tertiaryBox fill:#10b981,stroke:#34d399,stroke-width:2px,color:#ffffff
    classDef quaternaryBox fill:#7c3aed,stroke:#8b5cf6,stroke-width:2px,color:#ffffff
    classDef quinaryBox fill:#1e40af,stroke:#3b82f6,stroke-width:2px,color:#ffffff
    
    class A primaryBox
    class B secondaryBox
    class C tertiaryBox
    class D quaternaryBox
    class E quinaryBox
    class F primaryBox
"""
        },
        {
            "name": "optimization",
            "title": "Optimization Strategy",
            "content": """%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#1e40af', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#3b82f6', 'lineColor': '#60a5fa', 'secondaryColor': '#10b981', 'tertiaryColor': '#f59e0b', 'clusterBkg': '#f3f4f6', 'clusterBorder': '#9ca3af', 'fontSize': '18px'}}}%%
graph TD
    A["Optimization"] --> B["Learning Rate"]
    A --> C["Training Strategy"]
    A --> D["Model Updates"]
    A --> E["Stability"]
    
    B --> B1["Initial: 5e-5"]
    B --> B2["Minimum: 5e-6"]
    B --> B3["Cosine Decay"]
    B --> B4["Power: 0.96"]
    
    C --> C1["Batch Size: 16"]
    C --> C2["Accumulation: 8"]
    C --> C3["Effective: 128"]
    C --> C4["Clipping: 1.0"]
    
    D --> D1["AdamW"]
    D --> D2["Weight Decay"]
    D --> D3["EMA Updates"]
    D --> D4["EMA Beta: 0.999"]
    
    E --> E1["FP16: False"]
    E --> E2["Full Precision"]
    E --> E3["Stable Training"]
    E --> E4["No Overflow"]
    
    classDef primaryBox fill:#1e40af,stroke:#3b82f6,stroke-width:2px,color:#ffffff
    classDef secondaryBox fill:#10b981,stroke:#34d399,stroke-width:2px,color:#ffffff
    classDef tertiaryBox fill:#f59e0b,stroke:#fbbf24,stroke-width:2px,color:#ffffff
    classDef quaternaryBox fill:#ef4444,stroke:#f87171,stroke-width:2px,color:#ffffff
    classDef quinaryBox fill:#7c3aed,stroke:#8b5cf6,stroke-width:2px,color:#ffffff
    
    class A primaryBox
    class B secondaryBox
    class C tertiaryBox
    class D quaternaryBox
    class E quinaryBox
"""
        },
        {
            "name": "data_pipeline",
            "title": "Data Pipeline",
            "content": """%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#10b981', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#34d399', 'lineColor': '#6ee7b7', 'secondaryColor': '#7c3aed', 'tertiaryColor': '#f59e0b', 'clusterBkg': '#f3f4f6', 'clusterBorder': '#9ca3af', 'fontSize': '18px'}}}%%
graph TD
    A["Data Pipeline"] --> B["Dataset"]
    A --> C["Processing"]
    A --> D["Batch Assembly"]
    A --> E["Training Input"]
    
    B --> B1["RadioMapSeer"]
    B --> B2["Building Layouts"]
    B --> B3["Transmitter Positions"]
    B --> B4["Radio Measurements"]
    
    C --> C1["Image Processing"]
    C --> C2["320×320×1 Resize"]
    C --> C3["Normalization"]
    C --> C4["Augmentation"]
    
    D --> D1["Batch Creation"]
    D --> D2["16 Samples"]
    D --> D3["Parallel Loading"]
    D --> D4["Device Transfer"]
    
    E --> E1["VAE Input"]
    E --> E2["U-Net Condition"]
    E --> E3["Diffusion Target"]
    E --> E4["Training Labels"]
    
    classDef primaryBox fill:#10b981,stroke:#34d399,stroke-width:2px,color:#ffffff
    classDef secondaryBox fill:#7c3aed,stroke:#8b5cf6,stroke-width:2px,color:#ffffff
    classDef tertiaryBox fill:#f59e0b,stroke:#fbbf24,stroke-width:2px,color:#ffffff
    classDef quaternaryBox fill:#ef4444,stroke:#f87171,stroke-width:2px,color:#ffffff
    classDef quinaryBox fill:#1e40af,stroke:#3b82f6,stroke-width:2px,color:#ffffff
    
    class A primaryBox
    class B secondaryBox
    class C tertiaryBox
    class D quaternaryBox
    class E quinaryBox
"""
        }
    ]
    
    # Render each diagram
    for diagram in diagrams:
        print(f"Rendering simplified diagram: {diagram['name']}")
        
        # Create temporary mermaid file
        mermaid_file = output_dir / f"{diagram['name']}.mmd"
        with open(mermaid_file, 'w', encoding='utf-8') as f:
            f.write(diagram['content'])
        
        # Render using mmdc
        output_file = output_dir / f"{diagram['name']}.png"
        
        try:
            cmd = [
                'mmdc',
                '-i', str(mermaid_file),
                '-o', str(output_file),
                '-t', 'default',
                '-w', '1920',
                '-H', '1080',
                '-s', '3',
                '-b', 'transparent'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✓ Successfully rendered {diagram['name']}")
            else:
                print(f"✗ Failed to render {diagram['name']}: {result.stderr}")
                
        except FileNotFoundError:
            print("❌ mmdc (mermaid-cli) not found. Please install it with:")
            print("npm install -g @mermaid-js/mermaid-cli")
            return False
        except Exception as e:
            print(f"❌ Error rendering {diagram['name']}: {e}")
            return False
    
    # Clean up temporary files
    for diagram in diagrams:
        mermaid_file = output_dir / f"{diagram['name']}.mmd"
        if mermaid_file.exists():
            mermaid_file.unlink()
    
    print(f"\n✓ All simplified diagrams rendered to {output_dir}/")
    return True

if __name__ == "__main__":
    success = create_simplified_mermaid_diagrams()
    if success:
        print("\n🎉 Simplified mermaid diagrams rendered successfully!")
        print("Features:")
        print("  - 🎯 Clear model structure")
        print("  - 📝 Less detail, more focus")
        print("  - 🎨 Consistent styling")
        print("  - 📏 16:9 aspect ratio (1920×1080)")
        print("  - 🔤 18px font size")
    else:
        print("\n❌ Failed to render some diagrams.")