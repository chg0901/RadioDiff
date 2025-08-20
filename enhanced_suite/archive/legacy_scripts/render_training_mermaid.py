#!/usr/bin/env python3
"""
Render mermaid diagrams from the training documentation.
Creates colorful, latex-friendly visualizations with proper aspect ratios.
"""

import os
import subprocess
from pathlib import Path

def render_mermaid_diagrams():
    """Render all mermaid diagrams from the training documentation."""
    
    # Create output directory
    output_dir = Path("training_mermaid_vis")
    output_dir.mkdir(exist_ok=True)
    
    # Define mermaid diagrams to render
    diagrams = [
        {
            "name": "architecture",
            "title": "RadioDiff VAE Architecture",
            "content": """
graph TB
    subgraph "Data Pipeline"
        A[RadioUNet_c Dataset] --> B[DataLoader]
        B --> C[Batch Processing]
        C --> D[image: 320x320x1]
        C --> E[cond: 3x80x80]
        C --> F[ori_mask: optional]
    end
    
    subgraph "First Stage VAE"
        G[AutoencoderKL] --> H[Encoder]
        H --> I[Latent Space: 3x80x80]
        I --> J[Decoder]
        J --> K[Reconstruction]
    end
    
    subgraph "Conditional U-Net"
        L[Unet Model] --> M[Conditioning Network]
        M --> N[Swin Transformer]
        N --> O[Feature Extraction]
        O --> P[Fourier Features]
        P --> Q[Denoising Process]
    end
    
    subgraph "Latent Diffusion"
        R[LatentDiffusion] --> S[Forward Process]
        S --> T[Noise Scheduling]
        T --> U[Reverse Process]
        U --> V[Sampling]
    end
    
    D --> G
    E --> M
    I --> L
    Q --> R
    V --> W[Generated Images]
    
    style A fill:#e1f5fe
    style G fill:#f3e5f5
    style L fill:#e8f5e8
    style R fill:#fff3e0
"""
        },
        {
            "name": "training_flow",
            "title": "Training Data Flow",
            "content": """
graph LR
    subgraph "Training Loop"
        A[Data Loading] --> B[Batch Preparation]
        B --> C[VAE Encoding]
        C --> D[Latent Diffusion]
        D --> E[Loss Calculation]
        E --> F[Backpropagation]
        F --> G[Model Update]
        G --> H[EMA Update]
        H --> I[Sampling & Saving]
    end
    
    subgraph "Loss Components"
        J[L_reconstruction] --> E
        K[L_KL_divergence] --> E
        L[L_adversarial] --> E
        M[L_perceptual] --> E
    end
    
    subgraph "Optimization"
        N[AdamW Optimizer] --> F
        O[Learning Rate Schedule] --> G
        P[Gradient Clipping] --> F
    end
    
    style A fill:#e3f2fd
    style E fill:#ffebee
    style G fill:#e8f5e8
    style I fill:#fff3e0
"""
        },
        {
            "name": "vae_architecture",
            "title": "VAE Architecture Details",
            "content": """
graph TB
    subgraph "VAE Architecture"
        A[Input: 320x320x1] --> B[Encoder]
        B --> C[Downsampling: 128 channels]
        C --> D[Channel Multipliers: 1,2,4]
        D --> E[Residual Blocks: 2 per level]
        E --> F[Latent Space: 3x80x80]
        F --> G[Decoder]
        G --> H[Output: 320x320x1]
    end
    
    subgraph "Loss Configuration"
        I[KL Weight: 0.000001] --> J[Total Loss]
        K[Discriminator Weight: 0.5] --> J
        L[Discriminator Start: 50001] --> J
    end
    
    style A fill:#e1f5fe
    style F fill:#f3e5f5
    style J fill:#ffebee
"""
        },
        {
            "name": "unet_architecture",
            "title": "U-Net Architecture Details",
            "content": """
graph TB
    subgraph "U-Net Structure"
        A[Input: 3x80x80] --> B[Base Dim: 128]
        B --> C[Dim Multipliers: 1,2,4,4]
        C --> D[Conditioning Network]
        D --> E[Swin Transformer]
        E --> F[Window Attention]
        F --> G[Fourier Scale: 16]
        G --> H[Output: 3x80x80]
    end
    
    subgraph "Conditioning"
        I[Cond Input Dim: 3] --> D
        J[Cond Dim: 128] --> D
        K[Cond Dim Multipliers: 2,4] --> D
        L[Window Sizes: 8x8, 4x4, 2x2, 1x1] --> F
    end
    
    style A fill:#e8f5e8
    style E fill:#fff3e0
    style H fill:#e1f5fe
"""
        },
        {
            "name": "learning_rate",
            "title": "Learning Rate Schedule",
            "content": """
graph LR
    subgraph "Learning Rate Schedule"
        A[Initial LR: 5e-5] --> B[Cosine Decay]
        B --> C[Minimum LR: 5e-6]
        C --> D[Decay Power: 0.96]
    end
    
    subgraph "Training Strategy"
        E[Gradient Accumulation] --> F[Effective Batch: 128]
        F --> G[Gradient Clipping: 1.0]
        G --> H[EMA Smoothing: 0.999]
    end
    
    subgraph "Mixed Precision"
        I[FP16: False] --> J[AMP: False]
        J --> K[Full Precision Training]
    end
    
    style A fill:#e3f2fd
    style F fill:#e8f5e8
    style K fill:#fff3e0
"""
        },
        {
            "name": "data_pipeline",
            "title": "Data Pipeline",
            "content": """
graph TB
    subgraph "Data Structure"
        A[Building Maps] --> B[Transmitter Locations]
        B --> C[Radio Measurements]
        C --> D[Environmental Conditions]
    end
    
    subgraph "Data Processing"
        E[Image: 320x320x1] --> F[Normalization]
        F --> G[Condition: 3x80x80]
        G --> H[Mask: Optional]
    end
    
    subgraph "Augmentation"
        I[Random Crops] --> J[Flips]
        J --> K[Rotations]
        K --> L[Color Jitter]
    end
    
    style A fill:#e1f5fe
    style F fill:#e8f5e8
    style I fill:#fff3e0
"""
        },
        {
            "name": "training_process",
            "title": "Training Process",
            "content": """
graph TD
    subgraph "Training Step"
        A[Load Batch] --> B[Move to Device]
        B --> C[VAE Forward Pass]
        C --> D[Diffusion Forward Pass]
        D --> E[Loss Calculation]
        E --> F[Backward Pass]
        F --> G[Gradient Accumulation]
        G --> H[Optimizer Step]
        H --> I[EMA Update]
        I --> J[Logging]
    end
    
    subgraph "Sampling"
        K[Every 200 Steps] --> L[Generate Samples]
        L --> M[Save Images]
        M --> N[Save Checkpoint]
    end
    
    subgraph "Monitoring"
        O[TensorBoard Logging] --> P[Loss Curves]
        P --> Q[Learning Rate]
        Q --> R[Sample Quality]
    end
    
    style A fill:#e3f2fd
    style E fill:#ffebee
    style L fill:#e8f5e8
    style O fill:#fff3e0
"""
        },
        {
            "name": "loss_components",
            "title": "Loss Components",
            "content": """
graph LR
    subgraph "Reconstruction Loss"
        A[L2 Loss] --> B[Pixel-wise MSE]
        B --> C[Latent Space Reconstruction]
    end
    
    subgraph "Regularization"
        D[KL Divergence] --> E[Latent Space Prior]
        E --> F[Prevents Overfitting]
    end
    
    subgraph "Adversarial Training"
        G[Discriminator Loss] --> H[Real/Fake Classification]
        H --> I[Improves Realism]
    end
    
    subgraph "Perceptual Loss"
        J[LPIPS] --> K[Feature Similarity]
        K --> L[Better Visual Quality]
    end
    
    A --> M[Total Loss]
    D --> M
    G --> M
    J --> M
    
    style M fill:#ffebee
"""
        }
    ]
    
    # Render each diagram
    for diagram in diagrams:
        print(f"Rendering {diagram['name']}...")
        
        # Create temporary mermaid file
        mermaid_file = output_dir / f"{diagram['name']}.mmd"
        with open(mermaid_file, 'w') as f:
            f.write(diagram['content'])
        
        # Render using mmdc (mermaid-cli)
        output_file = output_dir / f"{diagram['name']}.png"
        
        try:
            # Use mmdc with proper configuration for 16:9 aspect ratio
            cmd = [
                'mmdc',
                '-i', str(mermaid_file),
                '-o', str(output_file),
                '-t', 'default',  # theme
                '-w', '1920',     # width for 16:9
                '-H', '1080',     # height for 16:9
                '-s', '3',        # scale factor for better quality
                '-b', 'transparent'  # transparent background
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
    
    print(f"\n✓ All diagrams rendered to {output_dir}/")
    return True

if __name__ == "__main__":
    success = render_mermaid_diagrams()
    if success:
        print("\n🎉 Mermaid diagrams rendered successfully!")
        print("You can now use these diagrams in your documentation.")
    else:
        print("\n❌ Failed to render some diagrams.")