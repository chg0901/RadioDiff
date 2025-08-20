# RadioDiff VAE Training Documentation - Summary

## 📋 Completed Tasks

✅ **Configuration Analysis**: Analyzed `radio_train_m.yaml` configuration file
✅ **Training Script Analysis**: Examined `train_cond_ldm.py` data flow and architecture
✅ **Model Architecture**: Analyzed VAE, U-Net, and Latent Diffusion components
✅ **Mermaid Diagrams**: Created comprehensive data flow and training process diagrams
✅ **README Documentation**: Wrote detailed mathematical explanations and usage guide
✅ **Diagram Rendering**: Successfully rendered all 8 mermaid diagrams using mermaid-cli

## 📁 Generated Files

### Documentation Files
- `README_TRAINING.md` - Comprehensive training guide with mermaid diagrams
- `training_mermaid_diagrams.md` - Source mermaid diagrams
- `render_training_mermaid.py` - Python script for rendering diagrams

### Visualizations (16:9 aspect ratio, 1920x1080px)
All diagrams are rendered in `training_mermaid_vis/` directory:
- `architecture.png` - Overall system architecture
- `training_flow.png` - Training data flow
- `vae_architecture.png` - VAE architecture details
- `unet_architecture.png` - U-Net architecture details
- `learning_rate.png` - Learning rate schedule
- `data_pipeline.png` - Data processing pipeline
- `training_process.png` - Training process steps
- `loss_components.png` - Loss function components

## 🔧 Key Configuration Parameters

### Model Configuration
- **Model Type**: `const_sde` (Stochastic Differential Equation)
- **Architecture**: `cond_unet` (Conditional U-Net)
- **Image Size**: 320×320 pixels
- **Timesteps**: 1000 diffusion steps
- **Loss Type**: L2 reconstruction loss
- **Objective**: `pred_KC` (predict key components)

### VAE First Stage
- **Embedding Dim**: 3
- **Latent Space**: 3×80×80
- **Channels**: 128 base, multipliers [1,2,4]
- **KL Weight**: 0.000001
- **Discriminator Weight**: 0.5

### U-Net Architecture
- **Base Dimension**: 128
- **Channel Multipliers**: [1,2,4,4]
- **Conditioning**: Swin Transformer
- **Window Sizes**: [8×8, 4×4, 2×2, 1×1]
- **Fourier Scale**: 16

### Training Configuration
- **Batch Size**: 16 (effective 128 with gradient accumulation)
- **Learning Rate**: 5e-5 to 5e-6 (cosine decay)
- **Training Steps**: 50,000
- **Gradient Accumulation**: 8
- **EMA Update**: After 10,000 steps, every 10 steps
- **Save Interval**: Every 200 steps

## 🧮 Mathematical Foundation

### Diffusion Process
- **Forward Process**: `q(z_t | z_{t-1}) = N(z_t; sqrt(1-beta_t) * z_{t-1}, beta_t * I)`
- **Reverse Process**: `p_theta(z_{t-1} | z_t) = N(z_{t-1}; mu_theta(z_t, t), sigma_t^2 * I)`
- **Total Loss**: `L_total = L_reconstruction + w_kl * L_kl + w_adv * L_adv + w_perceptual * L_perceptual`

### VAE Formulation
- **Encoder**: `q_phi(z|x) = N(z; mu_phi(x), sigma_phi(x)^2 * I)`
- **Decoder**: `p_theta(x|z) = N(x; mu_theta(z), sigma^2 * I)`
- **ELBO**: `E_q[log p_theta(x|z)] - KL(q_phi(z|x) || p(z))`

## 🚀 Training Command

```bash
accelerate launch train_cond_ldm.py --cfg ./configs/radio_train_m.yaml
```

## 📊 Training Process

1. **Data Loading**: RadioUNet_c dataset with building maps and radio measurements
2. **VAE Encoding**: Compress 320×320×1 images to 3×80×80 latent space
3. **Diffusion Training**: Train U-Net to denoise latent representations
4. **Loss Calculation**: Multi-component loss with reconstruction, KL, adversarial, and perceptual terms
5. **Optimization**: AdamW with cosine decay learning rate schedule
6. **EMA Updates**: Exponential moving average for stable weights
7. **Sampling**: Generate samples every 200 steps for quality assessment

## 🎯 Expected Results

- **Model Checkpoints**: Saved every 200 steps
- **Sample Images**: Generated for quality monitoring
- **Loss Curves**: Monitored via TensorBoard
- **Training Logs**: Detailed progress tracking

## 🔍 Monitoring

```bash
# Monitor training progress
tensorboard --logdir ./radiodiff_results_LDM

# Check GPU usage
nvidia-smi

# View generated samples
ls -la ./radiodiff_results_LDM/
```

## ✨ Features

- **Conditional Generation**: Uses building maps and transmitter locations as conditions
- **Efficient Architecture**: Swin Transformer for window-based attention
- **Mixed Precision**: Supports FP16 training (disabled in current config)
- **Gradient Accumulation**: Reduces memory usage
- **EMA Smoothing**: Stable model weights
- **Comprehensive Logging**: TensorBoard integration

## 📚 Usage

The documentation provides:
- Complete mathematical background
- Detailed configuration analysis
- Step-by-step training process
- Troubleshooting guide
- Performance optimization tips
- Visual architecture diagrams

All diagrams are rendered in high-quality 16:9 format suitable for presentations and publications.