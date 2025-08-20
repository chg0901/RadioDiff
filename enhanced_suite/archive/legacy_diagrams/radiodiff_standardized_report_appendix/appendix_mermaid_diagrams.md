# Appendix: Original Mermaid Diagrams

This appendix contains the original mermaid code and rendered images for all diagrams in the report.

## Diagram 1

![Diagram 1](radiodiff_standardized_report_images_final/diagram_1.png)

### Original Mermaid Code

```mermaid
graph TB
    subgraph "📡 Input Data Pipeline"
        A[RadioMapSeer Dataset<br/>320×320 Radio Maps] --> B[RadioUNet_c Loader<br/>batch_size: 66]
        B --> C[Gradient Accumulation<br/>8 steps → Effective: 528]
        C --> D[Input Tensors<br/>image: B×1×320×320<br/>cond: B×3×320×320]
    end
    
    subgraph "🎯 First Stage: VAE Encoder"
        E[AutoencoderKL<br/>embed_dim: 3] --> F[Encoder<br/>ResNet Architecture]
        F --> G[Latent Space z<br/>z~q_φ(z|x)<br/>Shape: [B, 3, 80, 80]]
        G --> H[16× Compression<br/>320×320 → 80×80<br/>Computational Efficiency]
    end
    
    subgraph "🔄 Second Stage: Conditional U-Net"
        I[Conditional U-Net<br/>dim: 128] --> J[Time Embedding<br/>Sinusoidal Encoding]
        J --> K[Condition Integration<br/>Swin Transformer<br/>Window-based Attention]
        K --> L[Multi-scale Features<br/>dim_mults: [1,2,4,4]]
        L --> M[Adaptive FFT Module<br/>Frequency Domain Enhancement]
        M --> N[Noise Prediction<br/>ε_θ(x_t, t, c)]
    end
    
    subgraph "🌊 Diffusion Process"
        O[Forward Diffusion<br/>q(x_t|x_0) = 𝒩(√ᾱ_tx_0, (1-ᾱ_t)𝐈)] --> P[Noise Schedule<br/>β_t: linear 0.0001→0.02]
        P --> Q[Reverse Process<br/>p_θ(x_0|x_t, c)]
        Q --> R[Knowledge-Aware Objective<br/>pred_KC]
    end
    
    subgraph "⚙️ Training Loop"
        S[L2 Loss Computation] --> T[Backpropagation<br/>Gradient Clipping: 1.0]
        T --> U[AdamW Optimizer<br/>lr: 5e-5, wd: 1e-4]
        U --> V[Cosine LR Schedule<br/>lr(t) = max(5e-6, 5e-5×(1-t/T)^0.96)]
        V --> W[EMA Model Update<br/>β: 0.999<br/>after 10,000 steps]
    end
    
    D --> E
    G --> I
    N --> S
    R --> S
    
    style A fill:#E3F2FD
    style E fill:#F3E5F5
    style I fill:#E8F5E8
    style O fill:#FFF3E0
    style S fill:#FCE4EC
```

---

## Diagram 2

![Diagram 2](radiodiff_standardized_report_images_final/diagram_2.png)

### Original Mermaid Code

```mermaid
graph LR
    subgraph "📊 Input Specifications"
        A[Image: 320×320×1<br/>Radio Map Pathloss] --> B[Condition: 320×320×3<br/>Building Layout]
        B --> C[Mask: Optional<br/>Spatial Constraints]
        C --> D[Latent: 80×80×3<br/>Compressed Representation]
    end
    
    subgraph "🏗️ VAE Architecture (First Stage)"
        E[Encoder<br/>ResNet Blocks] --> F[Downsampling<br/>4× Reduction]
        F --> G[Bottleneck<br/>Latent Space]
        G --> H[Decoder<br/>ResNet Blocks]
        H --> I[Upsampling<br/>4× Expansion]
        I --> J[Reconstruction<br/>Original Space]
    end
    
    subgraph "🎛️ Conditional U-Net Architecture"
        K[Input: x_t + t + c] --> L[Time Embedding<br/>Positional Encoding]
        L --> M[Conditional Processing<br/>Swin Transformer]
        M --> N[Multi-scale Features<br/>Hierarchical Extraction]
        N --> O[Window Attention<br/>[8,8], [4,4], [2,2], [1,1]]
        O --> P[Adaptive FFT<br/>Fourier Scale: 16]
        P --> Q[Output: Predicted Noise<br/>ε_θ(x_t, t, c)]
    end
    
    subgraph "📈 Diffusion Parameters"
        R[Timesteps: 1000] --> S[Beta Schedule<br/>Linear: 0.0001→0.02]
        S --> T[Objective: pred_KC<br/>Knowledge-Aware]
        T --> U[Loss: L2<br/>Mean Squared Error]
        U --> V[Scale Factor: 0.3<br/>Latent Space Scaling]
    end
    
    A --> E
    B --> M
    D --> G
    R --> S
    
    style E fill:#E3F2FD
    style K fill:#F1F8E9
    style R fill:#FFF8E1
```

---

## Diagram 3

![Diagram 3](radiodiff_standardized_report_images_final/diagram_3.png)

### Original Mermaid Code

```mermaid
flowchart TD
    subgraph "📂 Data Loading"
        A[RadioMapSeer Dataset<br/>Real-world Radio Maps] --> B[RadioUNet_c Loader<br/>DPM Simulation]
        B --> C[Batch Creation<br/>batch_size: 66]
        C --> D[Data Augmentation<br/>Horizontal Flip]
    end
    
    subgraph "🔧 Input Processing"
        E[Image: 320×320×1<br/>Radio Map] --> F[Normalization<br/>[-1, 1]]
        G[Condition: 320×320×3<br/>Building Info] --> H[Multi-channel Encoding]
        I[Mask: Optional<br/>320×320×1] --> J[Binary Processing]
    end
    
    subgraph "📦 Batch Structure"
        K[Batch Dictionary] --> L[image: B×1×320×320<br/>Pathloss Values]
        K --> M[cond: B×3×320×320<br/>Building Layout]
        K --> N[ori_mask: B×1×320×320<br/>Spatial Mask]
        K --> O[img_name: List[str]<br/>Metadata]
    end
    
    subgraph "🎯 Forward Pass"
        P[VAE Encoder] --> Q[Latent Representation<br/>B×3×80×80]
        Q --> R[Conditional U-Net<br/>Noise Prediction]
        R --> S[Loss Computation<br/>L2 Distance]
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
    Q --> R
    
    style A fill:#E8F5E8
    style K fill:#FFF3E0
    style P fill:#E3F2FD
    style S fill:#FCE4EC
```

---

## Diagram 4

![Diagram 4](radiodiff_standardized_report_images_final/diagram_4.png)

### Original Mermaid Code

```mermaid
graph TB
    subgraph "Reconstruction Loss Pipeline"
        A[Ground Truth<br/>x ∈ ℝ^(320×320×1)] --> B[VAE Reconstruction<br/>x̂ ∈ ℝ^(320×320×1)]
        B --> C[L1 Loss<br/>|x - x̂|]
        B --> D[MSE Loss<br/>||x - x̂||²]
        C --> E[Reconstruction Loss<br/>L_rec = L1 + MSE]
        D --> E
    end
    
    subgraph "Perceptual Loss Integration"
        F[Input Images] --> G[LPIPS Network<br/>Pre-trained VGG]
        H[Reconstructions] --> G
        G --> I[Feature Space<br/>Comparison]
        I --> J[Perceptual Distance<br/>L_perceptual]
        J --> K[Weighted Addition<br/>L_rec + w_p × L_perceptual]
    end
    
    subgraph "KL Divergence Components"
        L[Encoder Output<br/>μ, σ] --> M[Posterior<br/>q(z|x) = N(μ, σ²)]
        N[Prior<br/>p(z) = N(0, I)] --> O[KL Divergence<br/>KL[q||p]]
        M --> O
        O --> P[Batch Mean<br/>L_kl = mean(KL)]
        P --> Q[Weighted KL<br/>w_kl × L_kl]
    end
    
    subgraph "Adversarial Training Components"
        R[VAE Generator] --> S[Reconstructions]
        S --> T[Discriminator]
        T --> U[Logits: Fake]
        U --> V[Generator Loss<br/>L_g = -mean(logits_fake)]
        V --> W[Encourage Realistic<br/>Generations]
    end
    
    style A fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,font-weight:bold
    style B fill:#e3f2fd,stroke:#1565c0,stroke-width:3px,font-weight:bold
    style E fill:#fff3e0,stroke:#e65100,stroke-width:3px,font-weight:bold
    style G fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px,font-weight:bold
    style K fill:#ffecb3,stroke:#ff8f00,stroke-width:3px,font-weight:bold
    style L fill:#bbdefb,stroke:#1565c0,stroke-width:3px,font-weight:bold
    style N fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px,font-weight:bold
    style Q fill:#fff9c4,stroke:#f57f17,stroke-width:3px,font-weight:bold
    style R fill:#e1f5fe,stroke:#0277bd,stroke-width:3px,font-weight:bold
    style V fill:#ffebee,stroke:#c62828,stroke-width:3px,font-weight:bold
```

---

## Diagram 5

![Diagram 5](radiodiff_standardized_report_images_final/diagram_5.png)

### Original Mermaid Code

```mermaid
graph LR
    subgraph "🤖 Model Configuration"
        A[Model Type: const_sde<br/>Constant SDE] --> B[Image Size: [320, 320]]
        B --> C[Timesteps: 1000<br/>Diffusion Steps]
        C --> D[Sampling Steps: 1<br/>Inference Efficiency]
        D --> E[Objective: pred_KC<br/>Knowledge-Aware]
    end
    
    subgraph "🎛️ VAE Configuration"
        F[Embed Dim: 3<br/>Latent Space] --> G[Z Channels: 3<br/>Compression]
        G --> H[Resolution: 320×320<br/>Input Size]
        H --> I[Channels: 128<br/>Feature Dim]
        I --> J[Multiplier: [1,2,4]<br/>Scale Factors]
    end
    
    subgraph "🔧 U-Net Configuration"
        K[Dim: 128<br/>Base Dimension] --> L[Channels: 3<br/>Input Channels]
        L --> M[Dim Mults: [1,2,4,4]<br/>Multi-scale]
        M --> N[Cond Dim: 128<br/>Condition Dim]
        N --> O[Window Sizes<br/>[8,8], [4,4], [2,2], [1,1]]
    end
    
    subgraph "⚙️ Training Configuration"
        P[Batch Size: 66<br/>Per GPU] --> Q[Grad Accum: 8<br/>Effective: 528]
        Q --> R[LR: 5e-5<br/>Learning Rate]
        R --> S[Steps: 50000<br/>Total Training]
        S --> T[Save Every: 200<br/>Checkpoint Freq]
    end
    
    A --> F
    F --> K
    K --> P
    
    style A fill:#E3F2FD
    style F fill:#F3E5F5
    style K fill:#F1F8E9
    style P fill:#FFF8E1
```

---

## Diagram 6

![Diagram 6](radiodiff_standardized_report_images_final/diagram_6.png)

### Original Mermaid Code

```mermaid
graph TB
    subgraph "📈 Cosine Decay Schedule"
        A[Initial LR: 5e-5] --> B[Warmup Phase<br/>Optional]
        B --> C[Decay Phase<br/>(1-t/T)^0.96]
        C --> D[Minimum LR: 5e-6<br/>Floor Value]
    end
    
    subgraph "🔧 Optimization Parameters"
        E[Optimizer: AdamW] --> F[Weight Decay: 1e-4]
        F --> G[Gradient Clipping: 1.0]
        G --> H[Accumulation: 8 steps<br/>Effective Batch: 528]
    end
    
    subgraph "🎯 Model Regularization"
        I[EMA Update] --> J[Beta: 0.999]
        J --> K[Start After: 10,000 steps]
        K --> L[Update Every: 10 steps]
    end
    
    A --> E
    E --> I
    
    style A fill:#E3F2FD
    style E fill:#F1F8E9
    style I fill:#FFF8E1
```

---

## Diagram 7

![Diagram 7](radiodiff_standardized_report_images_final/diagram_7.png)

### Original Mermaid Code

```mermaid
graph TB
    subgraph "📥 Encoder Path"
        A[Input: 320×320×1<br/>Radio Map] --> B[Conv2D: 128×320×320]
        B --> C[ResNet Block 1<br/>Downsample: 160×160]
        C --> D[ResNet Block 2<br/>Downsample: 80×80]
        D --> E[Bottleneck<br/>Latent Space: 80×80×3]
    end
    
    subgraph "📤 Decoder Path"
        F[Latent: 80×80×3] --> G[ResNet Block 1<br/>Upsample: 160×160]
        G --> H[ResNet Block 2<br/>Upsample: 320×320]
        H --> I[Conv2D: 1×320×320<br/>Reconstruction]
    end
    
    subgraph "🔧 Regularization"
        J[KL Divergence<br/>D_KL(q_φ(z|x)‖p(z))] --> K[Reconstruction Loss<br/>‖x-x̂‖²]
        K --> L[Total Loss<br/>ELBO Optimization]
    end
    
    E --> F
    J --> L
    K --> L
    
    style A fill:#E3F2FD
    style F fill:#F3E5F5
    style L fill:#FFF8E1
```

---

