# RadioDiff VAE Model Documentation

## Overview

RadioDiff VAE is a Variational Autoencoder with adversarial training components designed for radio map prediction and generation tasks. This documentation provides a comprehensive understanding of the model architecture, training pipeline, and optimization strategies.

## Quick Start

```bash
accelerate launch train_cond_ldm.py --cfg ./configs/first_radio.yaml
```

## Model Architecture

### Encoder-Decoder Structure
The VAE follows a standard encoder-decoder architecture with the following specifications:

**Encoder Configuration:**
- **Input Resolution**: 320×320×1 (grayscale radio maps)
- **Channels**: 128 base channels with multiplicative factors [1, 2, 4]
- **Downsampling**: 3 levels (320→160→80→40)
- **Latent Space**: 3-dimensional with double z (double_z: True)
- **Residual Blocks**: 2 blocks per resolution level
- **Attention**: No attention layers (attn_resolutions: [])
- **Dropout**: 0.0 (no dropout)

**Decoder Configuration:**
- **Output Channels**: 1 (grayscale radio maps)
- **Upsampling**: Transposed convolutions with skip connections
- **Activation**: Swish nonlinearity
- **Normalization**: GroupNorm (32 groups)

### Loss Function Design

The model employs a sophisticated loss function combining multiple components:

#### 1. Reconstruction Loss
```python
rec_loss = torch.abs(inputs - reconstructions) + F.mse_loss(inputs, reconstructions, reduction="none")
```

**Components:**
- **L1 Loss**: Absolute difference between input and reconstruction
- **MSE Loss**: Mean squared error for pixel-level accuracy
- **Perceptual Loss**: LPIPS (Learned Perceptual Image Patch Similarity) for high-level features

#### 2. KL Divergence
```python
kl_loss = posteriors.kl()  # Standard VAE KL divergence
```

**Weighting Strategy:**
- **KL Weight**: 0.000001 (very small to prevent posterior collapse)
- **Log Variance**: Learnable parameter for loss scaling

#### 3. Adversarial Loss (VAE-GAN)
**Generator Loss:**
```python
g_loss = -torch.mean(logits_fake)  # Negative mean of fake logits
```

**Discriminator Loss:**
```python
# Hinge loss for real samples
real_loss = torch.mean(F.relu(1.0 - logits_real))
# Hinge loss for fake samples  
fake_loss = torch.mean(F.relu(1.0 + logits_fake))
disc_loss = real_loss + fake_loss
```

#### 4. Adaptive Weight Calculation
```python
d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
d_weight = torch.clamp(d_weight, 0.0, 1e4)
```

**Training Phases:**
- **Phase 1 (Steps 0-50,000)**: VAE pre-training only (disc_factor = 0.0)
- **Phase 2 (Steps 50,000-150,000)**: Full adversarial training (disc_factor = 0.5)

## Training Pipeline

### Data Pipeline

#### Data Sources
- **Primary Dataset**: RadioMapSeer dataset
- **Simulation Types**: IRT4, DPM, IRT2
- **Data Format**: Radio maps with building information
- **Resolution**: 320×320 pixels
- **Channels**: Single channel (radio signal strength)

#### Data Preprocessing
```python
# Normalization
transform_BZ = transforms.Normalize(mean=[0.5], std=[0.5])

# Data augmentation (optional)
augment_horizontal_flip: True/False
```

#### Data Loading Configuration
- **Batch Size**: 8 samples per batch
- **Data Splitting**: 
  - Training: 0-600 maps
  - Validation: 501-600 maps  
  - Testing: 600-699 maps
- **Shuffling**: Deterministic shuffle with seed 42

### Training Configuration

#### Optimizer Settings
- **Learning Rate**: 5e-6 (initial)
- **Minimum Learning Rate**: 5e-7
- **Gradient Accumulation**: Every 2 steps
- **Mixed Precision**: Disabled (amp: False, fp16: False)

#### Training Schedule
- **Total Steps**: 150,000
- **Save Frequency**: Every 5,000 steps
- **Logging Frequency**: Every 100 steps
- **Discriminator Start**: Step 50,001

#### Checkpoint Strategy
```python
# Model saving
save_and_sample_every: 5000
results_folder: '/data/chenglaoshi/DiffRadio/radio_Vae_3'
```

### Model Training Strategy

#### Phase 1: VAE Pre-training (0-50,000 steps)
**Focus**: Pure reconstruction capability
- **Loss Components**: NLL + KL only
- **Discriminator Weight**: 0.0 (disabled)
- **Goal**: Learn meaningful latent representations

#### Phase 2: Adversarial Training (50,000-150,000 steps)
**Focus**: Realistic generation with GAN
- **Loss Components**: NLL + KL + Adversarial
- **Discriminator Weight**: 0.0 → 0.5 (gradual introduction)
- **Adaptive Weighting**: Dynamic balance between reconstruction and adversarial losses

## Optimization Guidelines for New Datasets

### 1. Data-Specific Adaptations

#### Input Resolution Considerations
```python
# For different input sizes, adjust:
ddconfig:
  resolution: [new_width, new_height]  # Keep aspect ratio
  ch: base_channels  # Adjust based on complexity
  ch_mult: [1, 2, 4]  # Adjust depth based on resolution
```

#### Channel Configuration
- **Single Channel**: Radio maps, grayscale images
- **Multi-channel**: RGB images, multi-modal data
- **Adjust**: `in_channels` and `out_ch` accordingly

### 2. Hyperparameter Tuning Strategy

#### Learning Rate Optimization
```python
# Recommended ranges for different dataset sizes:
# Small datasets (<10k samples): 1e-5 to 1e-4
# Medium datasets (10k-100k samples): 5e-6 to 5e-5  
# Large datasets (>100k samples): 1e-6 to 1e-5
```

#### Loss Weight Balancing
```python
# KL divergence weight tuning:
kl_weight: 1e-6 to 1e-3  # Start small, increase if needed

# Adversarial weight tuning:
disc_weight: 0.1 to 1.0  # Balance generation quality vs. stability

# Perceptual weight tuning:
perceptual_weight: 0.1 to 1.0  # Higher for perceptual quality
```

### 3. Architecture Modifications

#### Latent Space Dimension
```python
# Adjust based on data complexity:
z_channels: 3  # Current (radio maps)
z_channels: 4-8  # For more complex data
z_channels: 16-32  # For very complex multi-modal data
```

#### Network Depth
```python
# Adjust based on input resolution:
ch_mult: [1, 2, 4]     # 320×320 → 40×40 (current)
ch_mult: [1, 2, 4, 8]   # For higher resolution inputs
ch_mult: [1, 2]         # For lower resolution inputs
```

### 4. Training Strategy Adaptations

#### Phase Duration Adjustment
```python
# Adjust phase lengths based on dataset complexity:
# Simple datasets: Phase 1: 20k steps, Phase 2: 80k steps
# Complex datasets: Phase 1: 100k steps, Phase 2: 200k steps
```

#### Discriminator Training
```python
# Adjust discriminator start based on reconstruction quality:
disc_start: 50001  # Current (50k steps)
disc_start: 20001  # Earlier for faster convergence
disc_start: 100001  # Later for better reconstruction first
```

### 5. Performance Monitoring

#### Key Metrics to Track
1. **Reconstruction Quality**: 
   - PSNR (Peak Signal-to-Noise Ratio)
   - SSIM (Structural Similarity Index)
   - Visual inspection

2. **Generation Quality**:
   - FID (Fréchet Inception Distance)
   - Human evaluation
   - Diversity metrics

3. **Training Stability**:
   - Loss curves (reconstruction, KL, adversarial)
   - Gradient norms
   - Weight distributions

#### Convergence Criteria
```python
# Early stopping conditions:
# - Validation loss plateaus for 10k steps
# - KL divergence becomes too small (<1e-8)
# - Training becomes unstable (loss NaN or inf)
```

## Implementation Details

### Model Initialization
```python
first_stage_model = AutoencoderKL(
    ddconfig=first_stage_cfg.ddconfig,
    lossconfig=first_stage_cfg.lossconfig,
    embed_dim=first_stage_cfg.embed_dim,
    ckpt_path=first_stage_cfg.ckpt_path,
)
```

### Training Loop Structure
```python
# Two-optimizer setup (generator + discriminator)
optimizer_idx = 0  # Generator update
optimizer_idx = 1  # Discriminator update

# Adaptive weight calculation for stability
d_weight = calculate_adaptive_weight(nll_loss, g_loss)
```

### Data Loading Optimization
```python
# Parallel data loading
DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=cpu_count(),
    pin_memory=True
)
```

## Configuration Analysis

### Key Parameters

```yaml
model:
  embed_dim: 3                    # 3D latent space
  lossconfig:
    disc_start: 50001             # Discriminator activation step
    kl_weight: 0.000001          # Very small KL regularization
    disc_weight: 0.5             # Discriminator influence weight
  ddconfig:
    z_channels: 3                # Latent space channels
    resolution: [320, 320]        # Input resolution
    ch_mult: [1, 2, 4]           # Channel multipliers
    num_res_blocks: 2             # Residual blocks per level
```

### Training Configuration

```yaml
trainer:
  train_num_steps: 150000        # Total training steps
  lr: 5e-6                      # Learning rate
  min_lr: 5e-7                   # Minimum learning rate
  batch_size: 8                  # Samples per batch
  gradient_accumulate_every: 2   # Gradient accumulation
  save_and_sample_every: 5000    # Checkpoint frequency
  log_freq: 100                  # Logging frequency
```

## Common Issues and Solutions

### 1. Posterior Collapse
**Symptoms**: KL loss → 0, poor generation quality
**Solutions**:
- Increase KL weight (1e-6 → 1e-4)
- Use stronger encoder
- Reduce latent dimension
- Add skip connections

### 2. Training Instability
**Symptoms**: Loss NaN/inf, mode collapse
**Solutions**:
- Reduce learning rate
- Add gradient clipping
- Use mixed precision training
- Adjust adversarial weight

### 3. Poor Reconstruction
**Symptoms**: Blurry outputs, low PSNR
**Solutions**:
- Extend Phase 1 duration
- Increase perceptual weight
- Add more residual blocks
- Use larger latent space

## Performance Benchmarks

### Current Configuration Performance
- **Input Resolution**: 320×320×1
- **Latent Dimension**: 3
- **Training Time**: ~150k steps
- **Memory Usage**: ~8GB (batch size 8)
- **Expected PSNR**: 25-30 dB (radio maps)

### Scaling Guidelines
- **2x Resolution**: 4x memory, 2x training time
- **2x Batch Size**: 2x memory, 1.5x training time
- **2x Latent Dim**: 1.5x memory, 1.2x training time

## Expected Training Behavior

### Phase 1 (0-50,000 steps)
- ✅ **KL Loss**: Increases gradually (normal VAE behavior)
- ✅ **Reconstruction Loss**: Decreases to low values (0.03-0.04)
- ✅ **Discriminator Loss**: Zero (inactive)
- ✅ **Total Loss**: Steady convergence

### Phase 2 (50,001-150,000 steps)
- 🔄 **KL Loss**: May stabilize or adjust
- 🔄 **Reconstruction Loss**: Should remain low
- 🔄 **Discriminator Loss**: Shows training progress
- 🔄 **Total Loss**: May fluctuate during GAN stabilization

## Results Storage

Training results are saved to:
```
/data/chenglaoshi/DiffRadio/radio_Vae_3/
├── model-5000.pt
├── model-10000.pt
├── model-15000.pt
└── ...
```

## Monitoring and Checkpoints

### Logging Frequency
- **Training Log**: Every 100 steps
- **Model Checkpoints**: Every 5,000 steps
- **Progress Monitoring**: Continuous TensorBoard logging

### Key Metrics to Monitor
1. **Reconstruction Quality**: Should remain high (< 0.1 loss)
2. **KL Divergence**: Should show healthy development
3. **Discriminator Balance**: Stable generator-discriminator dynamics
4. **Training Stability**: Consistent convergence patterns

## Best Practices

1. **Monitor Phase Transition**: Watch step 50,001 for discriminator activation
2. **Check Reconstruction Quality**: Ensure it remains high after GAN activation
3. **Watch for Mode Collapse**: Monitor discriminator training stability
4. **Adjust Learning Rates**: May need fine-tuning during GAN phase
5. **Regular Checkpoints**: Save progress every 5,000 steps

## Conclusion

RadioDiff VAE provides a robust framework for radio map generation and prediction. The key to successful adaptation to new datasets lies in:

1. **Proper data preprocessing and normalization**
2. **Careful hyperparameter tuning** 
3. **Phased training approach**
4. **Continuous monitoring of loss components**
5. **Architecture adjustments based on data complexity**

By following the guidelines in this documentation, practitioners can effectively adapt the RadioDiff VAE model to new datasets and applications while maintaining training stability and generation quality.

## References

- Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes.
- Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised representation learning with deep convolutional generative adversarial networks.
- Zhao, S., Song, J., & Ermon, S. (2017). Learning hierarchical features from generative models.

---

*For detailed implementation questions, refer to the source code and the VAE Report Generation Prompt Template (VAE_REPORT_GENERATION_PROMPT.md).*