#!/usr/bin/env python3
"""
Create simplified mermaid diagrams that will definitely render
"""

import os

def create_simplified_mermaid_diagrams(output_dir):
    """Create simplified mermaid diagrams."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    diagrams = [
        {
            'filename': 'diagram_1.mmd',
            'content': '''graph TB
    subgraph "Input Data Pipeline"
        A[RadioMapSeer Dataset] --> B[RadioUNet_c Loader]
        B --> C[Gradient Accumulation]
        C --> D[Input Tensors]
    end
    
    subgraph "VAE Encoder"
        E[AutoencoderKL] --> F[Encoder]
        F --> G[Latent Space]
        G --> H[Compression]
    end
    
    subgraph "Conditional U-Net"
        I[Conditional U-Net] --> J[Time Embedding]
        J --> K[Condition Integration]
        K --> L[Multi-scale Features]
        L --> M[Adaptive FFT]
        M --> N[Noise Prediction]
    end
    
    subgraph "Diffusion Process"
        O[Forward Diffusion] --> P[Noise Schedule]
        P --> Q[Reverse Process]
        Q --> R[Knowledge-Aware]
    end
    
    subgraph "Training Loop"
        S[L2 Loss] --> T[Backpropagation]
        T --> U[AdamW Optimizer]
        U --> V[LR Schedule]
        V --> W[EMA Update]
    end
    
    D --> E
    G --> I
    N --> S
    R --> S'''
        },
        {
            'filename': 'diagram_2.mmd',
            'content': '''graph LR
    subgraph "Input Specifications"
        A[Image: Radio Map] --> B[Condition: Building Layout]
        B --> C[Mask: Optional]
        C --> D[Latent: Compressed]
    end
    
    subgraph "VAE Architecture"
        E[Encoder] --> F[Downsampling]
        F --> G[Bottleneck]
        G --> H[Decoder]
        H --> I[Upsampling]
        I --> J[Reconstruction]
    end
    
    subgraph "U-Net Architecture"
        K[Input] --> L[Time Embedding]
        L --> M[Conditional Processing]
        M --> N[Multi-scale Features]
        N --> O[Window Attention]
        O --> P[Adaptive FFT]
        P --> Q[Output: Noise]
    end
    
    subgraph "Diffusion Parameters"
        R[Timesteps] --> S[Beta Schedule]
        S --> T[Objective]
        T --> U[Loss]
        U --> V[Scale Factor]
    end
    
    A --> E
    B --> M
    D --> G
    R --> S'''
        },
        {
            'filename': 'diagram_3.mmd',
            'content': '''flowchart TD
    subgraph "Data Loading"
        A[RadioMapSeer Dataset] --> B[RadioUNet_c Loader]
        B --> C[Batch Creation]
        C --> D[Data Augmentation]
    end
    
    subgraph "Input Processing"
        E[Image] --> F[Normalization]
        G[Condition] --> H[Multi-channel Encoding]
        I[Mask] --> J[Binary Processing]
    end
    
    subgraph "Batch Structure"
        K[Batch Dictionary] --> L[image: Pathloss]
        K --> M[cond: Building Layout]
        K --> N[ori_mask: Spatial Mask]
        K --> O[img_name: Metadata]
    end
    
    subgraph "Forward Pass"
        P[VAE Encoder] --> Q[Latent Representation]
        Q --> R[Conditional U-Net]
        R --> S[Loss Computation]
    end
    
    A --> B
    D --> E
    D --> G
    D --> I
    F --> L
    H --> M
    J --> N
    L --> K
    M --> K
    N --> K
    K --> P
    Q --> R'''
        },
        {
            'filename': 'diagram_4.mmd',
            'content': '''graph TB
    subgraph "Reconstruction Loss"
        A[Ground Truth] --> B[VAE Reconstruction]
        B --> C[L1 Loss]
        B --> D[MSE Loss]
        C --> E[Reconstruction Loss]
        D --> E
    end
    
    subgraph "Perceptual Loss"
        F[Input Images] --> G[LPIPS Network]
        H[Reconstructions] --> G
        G --> I[Feature Comparison]
        I --> J[Perceptual Distance]
        J --> K[Weighted Addition]
    end
    
    subgraph "KL Divergence"
        L[Encoder Output] --> M[Posterior]
        N[Prior] --> O[KL Divergence]
        M --> O
        O --> P[Batch Mean]
        P --> Q[Weighted KL]
    end
    
    subgraph "Adversarial Training"
        R[VAE Generator] --> S[Reconstructions]
        S --> T[Discriminator]
        T --> U[Logits]
        U --> V[Generator Loss]
        V --> W[Realistic Generations]
    end'''
        },
        {
            'filename': 'diagram_5.mmd',
            'content': '''graph LR
    subgraph "Model Configuration"
        A[Model Type] --> B[Image Size]
        B --> C[Timesteps]
        C --> D[Sampling Steps]
        D --> E[Objective]
    end
    
    subgraph "VAE Configuration"
        F[Embed Dim] --> G[Z Channels]
        G --> H[Resolution]
        H --> I[Channels]
        I --> J[Multiplier]
    end
    
    subgraph "U-Net Configuration"
        K[Dim] --> L[Channels]
        L --> M[Dim Mults]
        M --> N[Cond Dim]
        N --> O[Window Sizes]
    end
    
    subgraph "Training Configuration"
        P[Batch Size] --> Q[Grad Accum]
        Q --> R[LR]
        R --> S[Steps]
        S --> T[Save Every]
    end
    
    A --> F
    F --> K
    K --> P'''
        },
        {
            'filename': 'diagram_6.mmd',
            'content': '''graph TB
    subgraph "Cosine Decay Schedule"
        A[Initial LR] --> B[Warmup Phase]
        B --> C[Decay Phase]
        C --> D[Minimum LR]
    end
    
    subgraph "Optimization Parameters"
        E[AdamW Optimizer] --> F[Weight Decay]
        F --> G[Gradient Clipping]
        G --> H[Accumulation]
    end
    
    subgraph "Model Regularization"
        I[EMA Update] --> J[Beta]
        J --> K[Start After]
        K --> L[Update Every]
    end
    
    A --> E
    E --> I'''
        },
        {
            'filename': 'diagram_7.mmd',
            'content': '''graph TB
    subgraph "Encoder Path"
        A[Input] --> B[Conv2D]
        B --> C[ResNet Block 1]
        C --> D[ResNet Block 2]
        D --> E[Bottleneck]
    end
    
    subgraph "Decoder Path"
        F[Latent] --> G[ResNet Block 1]
        G --> H[ResNet Block 2]
        H --> I[Conv2D]
    end
    
    subgraph "Regularization"
        J[KL Divergence] --> K[Reconstruction Loss]
        K --> L[Total Loss]
    end
    
    E --> F
    J --> L
    K --> L'''
        }
    ]
    
    for diagram in diagrams:
        filepath = os.path.join(output_dir, diagram['filename'])
        with open(filepath, 'w') as f:
            f.write(diagram['content'])
        print(f"Created: {filepath}")
    
    print(f"Created {len(diagrams)} simplified mermaid diagrams")

def main():
    output_dir = 'radiodiff_standardized_report_mermaid_renderable'
    
    print("🎨 RadioDiff Standardized Report Simplified Mermaid Creator")
    print("=" * 60)
    
    # Create simplified mermaid diagrams
    print(f"\nCreating simplified mermaid diagrams...")
    create_simplified_mermaid_diagrams(output_dir)
    
    print(f"\n✅ Complete! All diagrams created in {output_dir}/")

if __name__ == "__main__":
    main()