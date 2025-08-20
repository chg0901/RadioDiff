# RadioDiff: Comprehensive Technical Report with Enhanced Visualizations

## 🚀 Executive Summary

This comprehensive technical report merges insights from multiple analysis reports to provide a unified understanding of the RadioDiff conditional latent diffusion model for radio wave propagation prediction. The report includes detailed mathematical foundations, architectural analysis, training methodology, and enhanced mermaid visualizations with proper formatting and LaTeX support.

## 🎯 Quick Start

```bash
accelerate launch train_cond_ldm.py --cfg ./configs/radio_train_m.yaml
```

## 📊 System Architecture Overview

```mermaid
graph TB
    subgraph "📡 Input Data Pipeline"
        A[RadioMapSeer Dataset<br/>320×320 Radio Maps] --> B[RadioUNet_c Loader<br/>batch_size: 32]
        B --> C[Gradient Accumulation<br/>Effective: 32]
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
        T --> U[AdamW Optimizer<br/>lr: 1e-5, wd: 1e-4]
        U --> V[Cosine LR Schedule<br/>lr(t) = max(1e-6, 1e-5×(1-t/T)^0.96)]
        V --> W[EMA Model Update<br/>β: 0.999<br/>after 10,000 steps]
    end
    
    D --> E
    G --> I
    N --> S
    R --> S
    
    style A fill:#E3F2FD,stroke:#01579B,stroke-width:2px
    style E fill:#F3E5F5,stroke:#4A148C,stroke-width:2px
    style I fill:#E8F5E8,stroke:#1B5E20,stroke-width:2px
    style O fill:#FFF3E0,stroke:#E65100,stroke-width:2px
    style S fill:#FCE4EC,stroke:#880E4F,stroke-width:2px
```

## 🏗️ Detailed VAE Architecture

### VAE Encoder Pipeline

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
    
    style A fill:#E3F2FD,stroke:#01579B,stroke-width:2px
    style F fill:#F3E5F5,stroke:#4A148C,stroke-width:2px
    style L fill:#FFF8E1,stroke:#F57F17,stroke-width:2px
```

### Mathematical Foundation

The VAE is trained to maximize the Evidence Lower Bound (ELBO):

$$\text{ELBO} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x) \| p(z))$$

Where:
- $q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \sigma_\phi^2(x)\mathbf{I})$ is the encoder distribution
- $p_\theta(x|z) = \mathcal{N}(x; \mu_\theta(z), \sigma^2\mathbf{I})$ is the decoder distribution  
- $p(z) = \mathcal{N}(0, \mathbf{I})$ is the prior distribution
- $D_{\text{KL}}$ denotes Kullback-Leibler divergence

The reparameterization trick enables gradient-based optimization:

$$z = \mu_\phi(x) + \sigma_\phi(x) \cdot \epsilon, \text{ where } \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

## 🎨 Enhanced Conditional U-Net

### U-Net Architecture Features

```mermaid
graph LR
    subgraph "🎯 Input Processing"
        A[Noisy Latent: x_t<br/>80×80×3] --> B[Time Embedding<br/>Sinusoidal]
        C[Condition: Building<br/>80×80×3] --> D[Condition Encoder<br/>Swin Transformer]
        B --> E[Concatenation<br/>x_t + t + c]
        D --> E
    end
    
    subgraph "🔍 Multi-scale Processing"
        E --> F[Scale 1: 80×80×128<br/>Window: 8×8]
        F --> G[Scale 2: 40×40×256<br/>Window: 4×4]
        G --> H[Scale 3: 20×20×512<br/>Window: 2×2]
        H --> I[Scale 4: 10×10×512<br/>Window: 1×1]
    end
    
    subgraph "🌊 Bottleneck"
        I --> J[Adaptive FFT<br/>Frequency Domain]
        J --> K[Self-Attention<br/>Global Context]
        K --> L[Cross-Attention<br/>Condition Integration]
    end
    
    subgraph "📈 Decoder"
        L --> M[Scale 4: 10×10×512<br/>Upsample]
        M --> N[Scale 3: 20×20×512<br/>Upsample]
        N --> O[Scale 2: 40×40×256<br/>Upsample]
        O --> P[Scale 1: 80×80×128<br/>Final Output]
        P --> Q[Predicted Noise: ε_θ<br/>80×80×3]
    end
    
    style A fill:#E3F2FD,stroke:#01579B,stroke-width:2px
    style F fill:#F1F8E9,stroke:#33691E,stroke-width:2px
    style J fill:#FFF8E1,stroke:#F57F17,stroke-width:2px
    style Q fill:#FCE4EC,stroke:#880E4F,stroke-width:2px
```

### Key Features

- **Base Dimension**: 128 channels with multipliers [1,2,4,4]
- **Conditioning**: Swin Transformer with cross-attention
- **Window Attention**: Multi-scale processing with window sizes [8×8, 4×4, 2×2, 1×1]
- **Fourier Features**: Spatial encoding with scale 16

### Mathematical Foundation

The U-Net predicts noise in the latent space:

$$\epsilon_\theta(z_t, t, c) = \text{UNet}(z_t, t, c)$$

Where:
- $z_t$: Noisy latent at timestep $t$
- $t$: Diffusion timestep
- $c$: Conditional information (building layout, transmitter position)

## 🔄 Training Workflow

### Training Process (50,000 Steps)

```mermaid
sequenceDiagram
    participant Main as main()
    participant Config as radio_train_m.yaml
    participant VAE as AutoencoderKL
    participant UNet as Conditional U-Net
    participant LDM as LatentDiffusion
    participant Data as RadioUNet_c Dataset
    participant Trainer as Trainer Class
    
    Main->>Config: Load YAML Configuration
    Config->>Main: Model & Training Parameters
    
    Main->>VAE: Initialize First Stage
    VAE->>Main: Pre-trained VAE Encoder
    
    Main->>UNet: Initialize Conditional U-Net
    UNet->>Main: Swin Transformer Backbone
    
    Main->>LDM: Create LatentDiffusion Model
    LDM->>Main: Combined VAE + U-Net
    
    Main->>Data: Load RadioMapSeer Dataset
    Data->>Main: DataLoader with Batching
    
    Main->>Trainer: Initialize Training Loop
    Trainer->>Trainer: Setup Optimizer & EMA
    
    loop Training Steps (50,000)
        Trainer->>Data: Get Batch
        Data->>Trainer: image, cond, mask
        
        Trainer->>LDM: Forward Pass
        LDM->>Trainer: Loss & Predictions
        
        Trainer->>Trainer: Backpropagation
        Trainer->>Trainer: Gradient Clipping
        
        Trainer->>Trainer: Optimizer Step
        Trainer->>Trainer: LR Schedule Update
        
        Trainer->>Trainer: EMA Update
        
        alt Every 200 Steps
            Trainer->>Trainer: Save Checkpoint
            Trainer->>Trainer: Generate Samples
        end
    end
    
    Note over Main,Trainer: Training Complete: model-250.pt
```

### Key Training Parameters

- **Learning Rate**: 1e-5 → 1e-6 (cosine decay)
- **Gradient Accumulation**: 1 step
- **EMA Updates**: After 10k steps, every 10 steps
- **Sampling**: Every 200 steps for quality assessment

## 📊 Enhanced Data Pipeline

### RadioMapSeer Dataset Processing

```mermaid
flowchart TD
    subgraph "📂 Data Loading"
        A[RadioMapSeer Dataset<br/>Real-world Radio Maps] --> B[RadioUNet_c Loader<br/>DPM Simulation]
        B --> C[Batch Creation<br/>batch_size: 32]
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
    
    style A fill:#E8F5E8,stroke:#1B5E20,stroke-width:2px
    style K fill:#FFF3E0,stroke:#E65100,stroke-width:2px
    style P fill:#E3F2FD,stroke:#01579B,stroke-width:2px
    style S fill:#FCE4EC,stroke:#880E4F,stroke-width:2px
```

### Data Structure

Each training sample contains:
- **image**: Radio wave propagation map (320×320×1) normalized to [-1, 1]
- **cond**: Building and transmitter information (3×80×80) for conditioning
- **ori_mask**: Optional building mask (320×320×1) for constrained generation
- **name**: File identifier for tracking and debugging

## ⚖️ Enhanced Loss Components Analysis

### Multi-Objective Loss Function

```mermaid
graph TB
    subgraph "🎯 Loss Components"
        A[Reconstruction Loss<br/>L2: ||x-x̂||²] --> E[Total Loss]
        B[KL Divergence<br/>KL(q‖p)] --> E
        C[Adversarial Loss<br/>GAN Training] --> E
        D[Perceptual Loss<br/>LPIPS] --> E
    end
    
    subgraph "⚖️ Loss Weights"
        F[λ_rec = 1.0] --> A
        G[λ_kl = 1e-6] --> B
        H[λ_adv = 0.5] --> C
        I[λ_perceptual = 0.0] --> D
    end
    
    subgraph "📊 Loss Dynamics"
        E --> J[L_total = Σλ_i·L_i]
        J --> K[Training Optimization]
        K --> L[Model Convergence]
    end
    
    style A fill:#E3F2FD,stroke:#01579B,stroke-width:2px
    style B fill:#F3E5F5,stroke:#4A148C,stroke-width:2px
    style C fill:#FFF3E0,stroke:#E65100,stroke-width:2px
    style D fill:#E8F5E8,stroke:#1B5E20,stroke-width:2px
    style J fill:#FCE4EC,stroke:#880E4F,stroke-width:2px
```

### Mathematical Formulation

The total loss is a weighted combination:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{rec}} \cdot \mathcal{L}_{\text{reconstruction}} + \lambda_{\text{kl}} \cdot \mathcal{L}_{\text{kl}} + \lambda_{\text{adv}} \cdot \mathcal{L}_{\text{adversarial}} + \lambda_{\text{perceptual}} \cdot \mathcal{L}_{\text{perceptual}}$$

Where:
- $\lambda_{\text{rec}} = 1.0$ (reconstruction weight)
- $\lambda_{\text{kl}} = 0.000001$ (KL regularization weight)
- $\lambda_{\text{adv}} = 0.5$ (adversarial weight, activated after 50k steps)
- $\lambda_{\text{perceptual}} = 0.0$ (perceptual weight)

Individual loss components:
- **Reconstruction**: $\mathcal{L}_{\text{rec}} = \|x - x_{\text{reconstructed}}\|^2$
- **KL Divergence**: $\mathcal{L}_{\text{kl}} = D_{\text{KL}}(q_\phi(z|x) \| \mathcal{N}(0, \mathbf{I})) = \frac{1}{2}(\mu^2 + \sigma^2 - \log(\sigma^2) - 1)$
- **Adversarial**: $\mathcal{L}_{\text{adv}} = -\log(D(G(z)))$ for generator, hinge loss for discriminator
- **Perceptual**: $\mathcal{L}_{\text{perceptual}} = \text{LPIPS}(x, x_{\text{reconstructed}})$

## 🎛️ Optimization Strategy

### Learning Rate Schedule

```mermaid
graph TB
    subgraph "📈 Cosine Decay Schedule"
        A[Initial LR: 1e-5] --> B[Warmup Phase<br/>Optional]
        B --> C[Decay Phase<br/>(1-t/T)^0.96]
        C --> D[Minimum LR: 1e-6<br/>Floor Value]
    end
    
    subgraph "🔧 Optimization Parameters"
        E[Optimizer: AdamW] --> F[Weight Decay: 1e-4]
        F --> G[Gradient Clipping: 1.0]
        G --> H[Accumulation: 1 step<br/>Effective Batch: 32]
    end
    
    subgraph "🎯 Model Regularization"
        I[EMA Update] --> J[Beta: 0.999]
        J --> K[Start After: 10,000 steps]
        K --> L[Update Every: 10 steps]
    end
    
    A --> E
    E --> I
    
    style A fill:#E3F2FD,stroke:#01579B,stroke-width:2px
    style E fill:#F1F8E9,stroke:#33691E,stroke-width:2px
    style I fill:#FFF8E1,stroke:#F57F17,stroke-width:2px
```

### Learning Rate Schedule

The cosine annealing schedule provides smooth learning rate decay:

$$\text{lr}(t) = \max\left(\text{lr}_{\min}, \text{lr}_{\max} \cdot (1 - t/T)^{0.96}\right)$$

Where:
- $t$ = current step (0 to 50,000)
- $T$ = total steps (50,000)
- $\text{lr}_{\max} = 1 \times 10^{-5}$
- $\text{lr}_{\min} = 1 \times 10^{-6}$

### AdamW Optimizer Configuration

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,
    betas=(0.9, 0.999),  # Momentum terms
    eps=1e-8,            # Numerical stability
    weight_decay=1e-4    # L2 regularization
)
```

## 🧮 Mathematical Foundation

### Diffusion Process Theory

The RadioDiff model implements a conditional latent diffusion process with sophisticated mathematical formulations:

#### Forward Diffusion Process:
$$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)\mathbf{I})$$

where:
- $\bar{\alpha}_t = \prod_{i=1}^{t} (1-\beta_i)$
- $\beta_t$ follows a linear schedule: $\beta_t = \text{linear}(0.0001, 0.02, T)$
- $T = 1000$ timesteps

#### Reverse Process with Conditioning:
$$p_\theta(x_{0:T}|c) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1}|x_t, c)$$

where $c$ represents the conditional information (building layout).

### Knowledge-Aware Objective (pred_KC)

The model uses a knowledge-aware prediction objective that incorporates radio propagation physics:

$$\mathcal{L}_{\text{KC}} = \mathbb{E}_{t,x_0,c,\epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t, c) \|^2 \right]$$

where:
- $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$
- $\epsilon_\theta$ is the noise prediction network with conditioning

### Conditional Generation Framework

Radio map construction is framed as a conditional generative problem:

$$p(x|c) = \int p(x|z,c) p(z|c) dz$$

where:
- $x$ is the radio map (pathloss distribution)
- $c$ is the conditional information (building layout)
- $z$ is the latent representation

## 📈 Configuration Analysis

### Model Configuration (radio_train_m.yaml)

#### Core Parameters
- **Model Type**: `const_sde` - Constant SDE diffusion
- **Model Name**: `cond_unet` - Conditional U-Net architecture
- **Image Size**: `[320, 320]` - Input radio signal dimensions
- **Timesteps**: `1000` - Number of diffusion steps
- **Loss Type**: `l2` - L2 reconstruction loss
- **Objective**: `pred_KC` - Predict key components

#### VAE First Stage
- **Embed Dim**: 3 - Latent space dimensions
- **Channels**: [1, 128, 256, 512] - Progressive feature extraction
- **Resolution**: [320, 320] - Input image size
- **KL Weight**: 0.000001 - Regularization strength
- **Discriminator Weight**: 0.5 - Adversarial training weight

#### U-Net Architecture
- **Base Dim**: 128 - Feature dimension
- **Dim Multipliers**: [1, 2, 4, 4] - Depth scaling
- **Conditioning**: Swin Transformer with window attention
- **Fourier Scale**: 16 - Frequency domain processing
- **Window Sizes**: [8×8, 4×4, 2×2, 1×1] - Multi-scale processing

### Training Configuration

#### Optimization Parameters
- **Batch Size**: 32 - Samples per iteration
- **Learning Rate**: 1e-5 - Initial learning rate
- **Min Learning Rate**: 1e-6 - Final learning rate
- **Training Steps**: 50,000 - Total iterations
- **Gradient Accumulation**: 1 - Effective batch size 32

#### Training Strategy
- **Save Interval**: Every 200 steps
- **EMA Update**: After 10,000 steps, every 10 steps
- **Mixed Precision**: Disabled for stability
- **Resume**: Optional checkpoint loading

## 🎯 Usage Instructions

### Basic Training

```bash
# Start training with default configuration
accelerate launch train_cond_ldm.py --cfg ./configs/radio_train_m.yaml
```

### Advanced Options

```bash
# Specific GPU usage
accelerate launch --gpu_ids=0 train_cond_ldm.py --cfg ./configs/radio_train_m.yaml

# Mixed precision training (if enabled)
accelerate launch --mixed_precision=fp16 train_cond_ldm.py --cfg ./configs/radio_train_m.yaml

# Resume from checkpoint
# Update config: trainer.enable_resume: true, trainer.resume_milestone: <checkpoint_number>
```

### Monitoring Training

```bash
# TensorBoard monitoring
tensorboard --logdir ./radiodiff_LDM

# Check progress
ls -la ./radiodiff_LDM/
tail -f ./radiodiff_LDM/train.log
```

## 📊 Expected Results

### Training Metrics
- **Total Loss**: Should decrease from ~1.0 to ~0.1 over 50k steps
- **Reconstruction Loss**: Primary indicator of model performance
- **KL Loss**: Should remain stable (low values indicate good latent space)
- **Sample Quality**: Visual inspection of generated samples every 200 steps

### Model Outputs
- **Checkpoints**: Saved every 200 steps in `./radiodiff_LDM/`
- **Samples**: Generated images for quality assessment
- **Logs**: TensorBoard logs for monitoring training progress
- **Model Reports**: Parameter statistics and freezing information

## 🔧 Technical Implementation

### Hardware Requirements
- **GPU**: High-end GPU with at least 16GB VRAM for batch size 32
- **Memory**: Sufficient RAM for data loading and processing
- **Storage**: Adequate space for model checkpoints and samples

### Software Dependencies
- **PyTorch**: Deep learning framework
- **Accelerate**: Distributed training support
- **Mermaid-cli**: Diagram rendering
- **TensorBoard**: Training monitoring

## 🎉 Conclusion

This comprehensive technical report provides a unified understanding of the RadioDiff VAE training process with enhanced mermaid visualizations and detailed mathematical formulations. The model combines VAE compression, conditional U-Net denoising, and latent diffusion for high-quality radio wave propagation prediction.

### Key Innovations
- **Efficient Architecture**: VAE compression reduces computational requirements
- **Conditional Generation**: Building-aware radio signal prediction
- **Stable Training**: EMA smoothing and cosine annealing for convergence
- **Multi-scale Processing**: Swin Transformer for efficient spatial modeling
- **Knowledge-Aware Objectives**: Radio propagation physics integration

### Theoretical Contributions
- **Conditional Generative Framework**: First comprehensive approach for radio map construction
- **Latent Space Efficiency**: 16× compression while preserving essential information
- **Multi-objective Optimization**: Balanced loss components for stable training
- **Physics-Informed Diffusion**: Knowledge-aware objectives for domain-specific constraints

This implementation demonstrates state-of-the-art performance in radio wave propagation prediction while maintaining computational efficiency and theoretical rigor.

## 📚 References

- **Latent Diffusion Models**: Rombach et al. (2022) - "High-Resolution Image Synthesis with Latent Diffusion Models"
- **Swin Transformer**: Liu et al. (2021) - "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
- **Variational Autoencoders**: Kingma & Welling (2014) - "Auto-Encoding Variational Bayes"
- **RadioMapSeer Dataset**: Radio wave propagation dataset for building-aware prediction
- **AdamW Optimizer**: Loshchilov & Hutter (2019) - "Decoupled Weight Decay Regularization"
- **RadioDiff Paper**: "RadioDiff: An Effective Generative Diffusion Model for Sampling-Free Dynamic Radio Map Construction"

## 📄 License

This project is for research purposes. Please ensure compliance with the original licenses of the referenced works and datasets.

---

*Generated with comprehensive analysis of RadioDiff codebase and enhanced mermaid visualizations*  
*Configuration: radio_train_m.yaml | Implementation: train_cond_ldm.py*