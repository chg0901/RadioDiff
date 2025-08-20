# RadioDiff VAE Enhanced Training Documentation

## 🚀 Overview

RadioDiff is a state-of-the-art conditional Latent Diffusion Model (LDM) designed for radio wave propagation prediction and generation. This enhanced documentation provides comprehensive insights into the model architecture, training methodology, and mathematical foundations with rich visualizations.

## 🎯 Quick Start

```bash
accelerate launch train_cond_ldm.py --cfg ./configs/radio_train_m.yaml
```

## 📊 System Architecture Overview

![RadioDiff System Overview](./mermaid_vis/radio_diff_overview.png)

The RadioDiff system consists of four main components:
1. **Data Input Layer**: Radio signals, building layouts, and environmental factors
2. **VAE Processing Core**: Encoder-decoder architecture for latent space compression
3. **Conditional U-Net**: Swin Transformer-based denoising network
4. **Diffusion Process**: Forward and reverse diffusion for high-quality generation

## 🏗️ Detailed VAE Architecture

![VAE Architecture Details](./mermaid_vis/vae_detailed_architecture.png)

### VAE Encoder Pipeline
- **Input**: 320×320×1 radio signal maps
- **Processing**: Progressive downsampling with residual blocks
- **Output**: 3×80×80 latent representation with μ and σ parameters

### Key VAE Components
- **Reparameterization**: z = μ + σ*ε, ε ~ N(0,1)
- **KL Divergence**: Regularization with weight 1e-6
- **Multi-scale Processing**: 3-level downsampling with skip connections

## 🎨 Enhanced Conditional U-Net

![Conditional U-Net Architecture](./mermaid_vis/unet_enhanced_architecture.png)

### U-Net Architecture Features
- **Base Dimension**: 128 channels with multipliers [1,2,4,4]
- **Conditioning**: Swin Transformer with cross-attention
- **Window Attention**: Multi-scale processing with window sizes [8×8, 4×4, 2×2, 1×1]
- **Fourier Features**: Spatial encoding with scale 16

### Mathematical Foundation
The U-Net predicts noise in the latent space:
```
ε_θ(z_t, t, c) = UNet(z_t, t, c)
```

Where:
- z_t: Noisy latent at timestep t
- t: Diffusion timestep
- c: Conditional information (building layout, transmitter position)

## 🔄 Enhanced Training Workflow

![Training Workflow](./mermaid_vis/training_workflow_enhanced.png)

### Training Process (50,000 Steps)
1. **Initialization**: Load configuration and pretrained weights
2. **Data Loading**: RadioUNet_c dataset with batch size 16
3. **Forward Pass**: VAE encoding and diffusion processing
4. **Loss Calculation**: Multi-objective optimization
5. **Backpropagation**: Gradient accumulation (effective batch 128)
6. **Model Update**: AdamW optimizer with EMA smoothing

### Key Training Parameters
- **Learning Rate**: 5e-5 → 5e-6 (cosine decay)
- **Gradient Accumulation**: 8 steps
- **EMA Updates**: After 10k steps, every 10 steps
- **Sampling**: Every 200 steps for quality assessment

## 📊 Enhanced Data Pipeline

![Data Pipeline](./mermaid_vis/data_pipeline_enhanced.png)

### RadioMapSeer Dataset Processing
- **Building Layouts**: 2D floor plans with wall structures
- **Transmitter Positions**: TX locations and antenna types
- **Radio Measurements**: Signal strength and path loss data
- **Environmental Factors**: Materials and obstacles

### Data Preprocessing
- **Image Processing**: 320×320×1 resize and normalization
- **Condition Extraction**: 3×80×80 spatial features
- **Augmentation**: Flips, rotations, and color jittering

## ⚖️ Enhanced Loss Components Analysis

![Loss Components](./mermaid_vis/loss_components_enhanced.png)

### Multi-Objective Loss Function
```
L_total = L_reconstruction + λ_kl * L_kl + λ_adv * L_adv + λ_perceptual * L_perceptual
```

### Loss Components Breakdown

#### 1. Reconstruction Loss (Weight: 1.0)
- **L2 Loss**: Pixel-wise MSE between input and output
- **VAE Reconstruction**: Signal fidelity in pixel space
- **Latent Space**: Reconstruction in compressed domain

#### 2. KL Divergence (Weight: 0.000001)
- **Regularization**: Prevents overfitting in latent space
- **Prior Distribution**: N(0, I) standard normal
- **Generalization**: Ensures better latent space quality

#### 3. Adversarial Loss (Weight: 0.5)
- **GAN Training**: Real vs fake classification
- **Quality Enhancement**: Improves realism of generated samples
- **Activation**: After 50k steps for refinement

#### 4. Perceptual Loss (Optional)
- **LPIPS**: Learned perceptual image patch similarity
- **Feature Matching**: Deep feature similarity using VGG/ResNet
- **Visual Quality**: Enhances human perception of results

## 🎛️ Optimization Strategy

![Optimization Strategy](./mermaid_vis/optimization_strategy.png)

### Learning Rate Schedule
- **Initial LR**: 5e-5
- **Minimum LR**: 5e-6
- **Decay Function**: Cosine annealing with power 0.96
- **Schedule**: LR(t) = max(min_lr, (1-t)^0.96 * init_lr)

### AdamW Optimizer Configuration
- **Beta1**: 0.9 (first moment)
- **Beta2**: 0.999 (second moment)
- **Epsilon**: 1e-8 (numerical stability)
- **Weight Decay**: 1e-4 (L2 regularization)

### Model Stabilization
- **Gradient Accumulation**: 8 steps for effective batch size 128
- **Gradient Clipping**: Max norm 1.0 for training stability
- **EMA Updates**: Exponential moving average with β=0.999
- **Mixed Precision**: Disabled for numerical stability

## 🧮 Mathematical Foundation

![Mathematical Foundation](./mermaid_vis/mathematical_foundation.png)

### VAE Formulation

#### Encoder
```
q_φ(z|x) = N(z; μ_φ(x), σ²_φ(x))
```

#### Decoder
```
p_θ(x|z) = N(x; μ_θ(z), σ²_θ(z))
```

#### Evidence Lower Bound (ELBO)
```
L_elbo = E_q[log p_θ(x|z)] - KL(q_φ(z|x) || p(z))
```

#### Reparameterization Trick
```
z = μ_φ(x) + σ_φ(x) * ε, where ε ~ N(0, I)
```

### Diffusion Process

#### Forward Process (Noise Addition)
```
q(z_t|z_{t-1}) = N(z_t; √(1-β_t) * z_{t-1}, β_t * I)
```

#### Reverse Process (Denoising)
```
p_θ(z_{t-1}|z_t) = N(z_{t-1}; μ_θ(z_t, t), Σ_θ(z_t, t))
```

#### Loss Function
```
L_simple = ||ε - ε_θ(z_t, t, c)||²
```

### Conditional Generation

#### Condition Integration
```
c = f_condition(building_layout, transmitter_pos)
```

#### Cross-Attention Mechanism
```
Attention(Q, K, V) where Q = f(z_t), K = V = f(c)
```

#### Conditional U-Net
```
ε_θ(z_t, t, c) = UNet(z_t, t, c)
```

## 📈 Performance Metrics and Evaluation

### Training Monitoring
- **Loss Curves**: Real-time tracking of all loss components
- **Learning Rate**: Cosine decay visualization
- **Sample Quality**: Generated images every 200 steps
- **Convergence**: Brightest point distance calculation

### Quality Assessment
- **Reconstruction Quality**: MSE and perceptual metrics
- **Generation Quality**: Visual inspection and quantitative measures
- **Latent Space**: KL divergence monitoring
- **Stability**: Gradient norm and training stability

## 🎯 Configuration Analysis

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
- **Batch Size**: 16 - Samples per iteration
- **Learning Rate**: 5e-5 - Initial learning rate
- **Min Learning Rate**: 5e-6 - Final learning rate
- **Training Steps**: 50,000 - Total iterations
- **Gradient Accumulation**: 8 - Effective batch size 128

#### Training Strategy
- **Save Interval**: Every 200 steps
- **EMA Update**: After 10,000 steps, every 10 steps
- **Mixed Precision**: Disabled for stability
- **Resume**: Optional checkpoint loading

## 🚀 Advanced Features

### Conditional Generation Capabilities
- **Building Layout Integration**: Spatial conditioning through cross-attention
- **Multi-modal Input**: Building + transmitter + environmental information
- **Flexible Generation**: Support for various input configurations

### Efficient Architecture Design
- **Swin Transformer**: Window-based attention for computational efficiency
- **Multi-scale Processing**: Hierarchical feature extraction
- **Latent Space Compression**: Efficient representation learning

### Robust Training Strategy
- **EMA Smoothing**: Stable model weights
- **Gradient Clipping**: Training stability
- **Cosine Annealing**: Optimal learning rate schedule

## 📊 Results and Outputs

### Generated Outputs
- **Radio Signal Maps**: High-quality generated propagation maps
- **Building-aware Generation**: Context-aware signal prediction
- **Multi-scale Features**: Detailed and coarse-grained information

### Training Artifacts
- **Model Checkpoints**: Saved every 200 steps
- **Sample Images**: Generated samples for quality assessment
- **Training Logs**: Comprehensive training metrics
- **Visualization**: Rich visualizations for analysis

## 🔧 Technical Implementation

### Hardware Requirements
- **GPU**: High-end GPU with sufficient memory
- **Memory**: At least 16GB VRAM for batch size 16
- **Storage**: Adequate space for model checkpoints and samples

### Software Dependencies
- **PyTorch**: Deep learning framework
- **Accelerate**: Distributed training support
- **Mermaid-cli**: Diagram rendering
- **TensorBoard**: Training monitoring

## 🎯 Usage Instructions

### Basic Training
```bash
accelerate launch train_cond_ldm.py --cfg ./configs/radio_train_m.yaml
```

### Advanced Options
```bash
# Specific GPU usage
accelerate launch --gpu_ids=0 train_cond_ldm.py --cfg ./configs/radio_train_m.yaml

# Mixed precision training
accelerate launch --mixed_precision=fp16 train_cond_ldm.py --cfg ./configs/radio_train_m.yaml

# Resume from checkpoint
# Update config: trainer.enable_resume: true, trainer.resume_milestone: <checkpoint_number>
```

### Monitoring Training
```bash
# TensorBoard monitoring
tensorboard --logdir ./radiodiff_results_LDM

# Check progress
ls -la ./radiodiff_results_LDM/
tail -f ./radiodiff_results_LDM/train.log
```

## 🎉 Conclusion

This enhanced documentation provides a comprehensive understanding of the RadioDiff VAE training process, combining theoretical foundations with practical implementation details. The rich visualizations and detailed explanations facilitate better understanding and deployment of the model for radio wave propagation prediction tasks.

The model's conditional generation capabilities, efficient architecture, and robust training strategy make it suitable for various radio signal processing applications, particularly in scenarios where building layout and environmental factors significantly impact signal propagation.

## 📚 References

- **Latent Diffusion Models**: Rombach et al. (2022)
- **High-Resolution Image Synthesis**: Rombach et al. (2022)
- **RadioMapSeer Dataset**: Radio wave propagation dataset
- **Swin Transformer**: Liu et al. (2021)
- **Variational Autoencoders**: Kingma & Welling (2014)

## 📄 License

This project is for research purposes. Please ensure compliance with the original licenses of the referenced works and datasets.