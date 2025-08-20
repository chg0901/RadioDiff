# RadioDiff VAE Training Documentation - Simplified Overview

## 🚀 Introduction

This document provides a clear, structured overview of the RadioDiff VAE training process with simplified diagrams and technical explanations. The model is trained using:

```bash
accelerate launch train_cond_ldm.py --cfg ./configs/radio_train_m.yaml
```

## 📊 System Architecture Overview

![RadioDiff System Overview](./mermaid_vis_simplified/system_overview.png)

### Technical Explanation

The RadioDiff system consists of four main components:

1. **Data Input Layer**: Processes radio signals (320×320×1), building layouts, and environmental data from the RadioMapSeer dataset.

2. **VAE Processing Core**: Uses a Variational Autoencoder to compress input data into a latent space (3×80×80) and reconstruct it. This reduces computational complexity while preserving essential information.

3. **Conditional U-Net**: A Swin Transformer-based network that processes the latent space with window-based attention and Fourier features for efficient spatial encoding.

4. **Diffusion Process**: Implements both forward (noise addition) and reverse (denoising) processes in the latent space for high-quality generation.

### Code Implementation

The system is implemented in `train_cond_ldm.py` with the following key components:

```python
# Main model initialization
model = LatentDiffusion(
    first_stage_config=first_stage_config,  # VAE configuration
    cond_stage_config=cond_stage_config,    # Conditioning network
    unet_config=unet_config,               # U-Net configuration
    diffusion_config=diffusion_config      # Diffusion parameters
)
```

### Paper Reference

Based on the Latent Diffusion Model (LDM) framework (Rombach et al., 2022), which operates in compressed latent space rather than pixel space, significantly reducing computational requirements.

---

## 🏗️ VAE Architecture

![VAE Architecture](./mermaid_vis_simplified/vae_architecture.png)

### Technical Explanation

The VAE (Variational Autoencoder) consists of three main paths:

1. **Encoder Path**: Takes 320×320×1 input radio signals and progressively downsamples through convolutional layers (128→512 channels) to produce latent parameters μ and σ.

2. **Latent Space**: The compressed representation (3×80×80) where the model learns efficient encoding of radio wave patterns. Uses reparameterization trick: z = μ + σ*ε.

3. **Decoder Path**: Reconstructs the original signal from latent samples through upsampling and transposed convolutions.

### Mathematical Foundation

The VAE is trained to maximize the Evidence Lower Bound (ELBO):

```
ELBO = E_q[log p_θ(x|z)] - KL(q_φ(z|x) || p(z))
```

Where:
- q_φ(z|x) is the encoder distribution
- p_θ(x|z) is the decoder distribution  
- p(z) is the prior distribution (standard normal)

### Code Implementation

```python
# VAE encoder-decoder structure
class AutoencoderKL(nn.Module):
    def __init__(self, embed_dim=3, channels=[1, 128, 256, 512]):
        self.encoder = Encoder(channels=channels, z_channels=embed_dim)
        self.decoder = Decoder(channels=channels, z_channels=embed_dim)
        self.quant_conv = nn.Conv2d(2*embed_dim, 2*embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, embed_dim, 1)
    
    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    
    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)
```

### Configuration Parameters

From `radio_train_m.yaml`:
- **embed_dim**: 3 (latent space dimensions)
- **channels**: [1, 128, 256, 512] (progressive feature extraction)
- **kl_weight**: 0.000001 (regularization strength)
- **discr_weight**: 0.5 (adversarial training weight)

---

## 🎨 Conditional U-Net Structure

![Conditional U-Net Structure](./mermaid_vis_simplified/unet_structure.png)

### Technical Explanation

The Conditional U-Net is the core denoising network with the following structure:

1. **Input Processing**: Takes latent input (3×80×80), time embeddings, and condition information.

2. **Encoder Path**: Multi-scale processing with 4 levels (128×80×80 → 512×10×10) using residual connections and downsampling.

3. **Bottleneck**: Swin Transformer with window attention for efficient processing of large spatial dimensions.

4. **Decoder Path**: Upsampling with skip connections from encoder levels for feature refinement.

5. **Output**: Predicts noise in latent space for diffusion reverse process.

### Key Features

- **Swin Transformer**: Uses window-based attention to reduce computational complexity from O(n²) to O(w²).
- **Cross-Attention**: Integrates conditioning information through attention mechanisms.
- **Window Sizes**: [8×8, 4×4, 2×2, 1×1] for multi-scale processing.
- **Fourier Features**: Efficient spatial encoding with scale 16.

### Code Implementation

```python
# Conditional U-Net with Swin Transformer
class Unet(nn.Module):
    def __init__(self, dim=128, channels=3, dim_mults=[1,2,4,4]):
        self.time_embed = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.SiLU(),
            nn.Linear(dim*4, dim*4)
        )
        
        # Encoder levels
        self.downs = nn.ModuleList([
            Downsample(dim*dim_mults[i], dim*dim_mults[i+1])
            for i in range(len(dim_mults)-1)
        ])
        
        # Bottleneck with Swin Transformer
        self.mid = nn.ModuleList([
            ResnetBlock(dim*dim_mults[-1]),
            TransformerBlock(dim*dim_mults[-1], window_size=8),
            ResnetBlock(dim*dim_mults[-1])
        ])
        
        # Decoder levels
        self.ups = nn.ModuleList([
            Upsample(dim*dim_mults[i+1], dim*dim_mults[i])
            for i in range(len(dim_mults)-1)
        ])
```

### Paper Reference

The architecture follows the Swin Transformer design (Liu et al., 2021) for efficient attention computation, adapted for conditional generation tasks in radio wave propagation.

---

## 🔄 Training Workflow

![Training Workflow](./mermaid_vis_simplified/training_workflow.png)

### Technical Explanation

The training process consists of four main phases:

1. **Setup Phase**: Load configuration, initialize models, setup data loaders, and configure optimizers.

2. **Training Loop**: The core training iteration where batches are processed, forward passes executed, losses computed, and gradients accumulated.

3. **Optimization Phase**: Apply gradient updates, learning rate scheduling, and EMA smoothing for stable training.

4. **Evaluation Phase**: Generate samples, save checkpoints, and log metrics for monitoring progress.

### Key Training Parameters

- **Batch Size**: 16 samples per iteration
- **Gradient Accumulation**: 8 steps (effective batch size: 128)
- **Learning Rate**: 5e-5 → 5e-6 (cosine decay)
- **Training Steps**: 50,000 total iterations
- **Save Frequency**: Every 200 steps
- **EMA Updates**: After 10k steps, every 10 steps

### Code Implementation

```python
# Main training loop
def train():
    trainer = Trainer(
        model=model,
        data=data,
        optimizer=optimizer,
        total_steps=50000,
        save_interval=200,
        log_interval=100
    )
    
    for step in range(trainer.total_steps):
        # Load batch
        batch = next(dataloader)
        
        # Forward pass
        loss = trainer.train_step(batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # EMA update
            if step > 10000 and step % 10 == 0:
                ema_model.update()
        
        # Sampling and saving
        if step % save_interval == 0:
            samples = trainer.sample()
            trainer.save_checkpoint(samples)
```

### Configuration Details

From `radio_train_m.yaml`:
```yaml
trainer:
  batch_size: 16
  learning_rate: 5e-5
  min_learning_rate: 5e-6
  training_steps: 50000
  gradient_accumulation: 8
  save_interval: 200
  ema:
    start_step: 10000
    update_interval: 10
    beta: 0.999
```

---

## ⚖️ Loss Components

![Loss Components](./mermaid_vis_simplified/loss_components.png)

### Technical Explanation

The training objective combines multiple loss components:

1. **Reconstruction Loss**: L2/MSE loss between predicted and target signals in both pixel and latent space.

2. **Regularization Loss**: KL divergence that prevents overfitting and ensures good latent space quality.

3. **Adversarial Loss**: GAN-based loss that improves realism through discriminator training.

4. **Perceptual Loss**: Optional LPIPS loss that enhances visual quality through feature similarity.

### Mathematical Formulation

The total loss is a weighted combination:

```
L_total = λ_rec * L_reconstruction + λ_kl * L_kl + λ_adv * L_adversarial + λ_perceptual * L_perceptual
```

Where:
- λ_rec = 1.0 (reconstruction weight)
- λ_kl = 0.000001 (KL regularization weight)
- λ_adv = 0.5 (adversarial weight, activated after 50k steps)
- λ_perceptual = configurable (perceptual weight)

### Loss Components Breakdown

```python
# Loss computation
def compute_loss(model, batch):
    # Reconstruction loss
    recon_loss = F.mse_loss(reconstructed, target)
    
    # KL divergence
    kl_loss = posterior.kl()
    kl_loss = torch.mean(kl_loss)
    
    # Adversarial loss (if enabled)
    if step > 50000:
        fake_pred = discriminator(reconstructed)
        real_pred = discriminator(target)
        adv_loss = hinge_gan_loss(fake_pred, real_pred)
    
    # Perceptual loss (optional)
    if use_perceptual:
        perceptual_loss = lpips_loss(reconstructed, target)
    
    # Total loss
    total_loss = recon_loss + kl_weight * kl_loss + adv_weight * adv_loss
    return total_loss
```

### Configuration Weights

From the configuration:
- **kl_weight**: 0.000001 (very small to prevent posterior collapse)
- **discr_weight**: 0.5 (balanced adversarial training)
- **discr_start**: 50001 (delayed discriminator activation)

---

## 🎛️ Optimization Strategy

![Optimization Strategy](./mermaid_vis_simplified/optimization.png)

### Technical Explanation

The optimization strategy includes several key components:

1. **Learning Rate Schedule**: Cosine annealing from 5e-5 to 5e-6 with power 0.96 for stable convergence.

2. **Training Strategy**: Gradient accumulation (8 steps) for effective batch size of 128 with memory efficiency.

3. **Model Updates**: AdamW optimizer with weight decay and EMA smoothing for stable weights.

4. **Stability Measures**: Full precision training (FP16 disabled) for numerical stability.

### Learning Rate Schedule

The cosine annealing schedule:

```
LR(t) = max(min_lr, (1-t/T)^power * initial_lr)
```

Where:
- t = current step
- T = total steps (50,000)
- power = 0.96
- initial_lr = 5e-5
- min_lr = 5e-6

### AdamW Optimizer Configuration

```python
# AdamW optimizer with cosine decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-4
)

# Cosine annealing scheduler
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: max(
        min_lr / initial_lr,
        (1 - step / total_steps) ** power
    )
)
```

### EMA Configuration

```python
# Exponential Moving Average
class EMA:
    def __init__(self, model, beta=0.999):
        self.model = model
        self.beta = beta
        self.shadow = copy.deepcopy(model)
    
    def update(self):
        for param, shadow_param in zip(
            self.model.parameters(), 
            self.shadow.parameters()
        ):
            shadow_param.data = (
                self.beta * shadow_param.data + 
                (1 - self.beta) * param.data
            )
```

---

## 📊 Data Pipeline

![Data Pipeline](./mermaid_vis_simplified/data_pipeline.png)

### Technical Explanation

The data pipeline processes radio wave propagation data through several stages:

1. **Dataset**: RadioMapSeer dataset containing building layouts, transmitter positions, and radio measurements.

2. **Processing**: Image resizing (320×320×1), normalization, and data augmentation.

3. **Batch Assembly**: Creates batches of 16 samples with parallel loading and device transfer.

4. **Training Input**: Formats data for VAE input, U-Net conditioning, and diffusion targets.

### Data Structure

Each training sample contains:
- **image**: Radio wave propagation map (320×320×1)
- **cond**: Building and transmitter information (3×80×80)
- **ori_mask**: Optional building mask (320×320×1)
- **name**: File identifier for tracking

### Code Implementation

```python
# Dataset loading
dataset = RadioUNet_c(
    data_root="path/to/RadioMapSeer",
    simulation_type="specific_type",
    additional_params={}
)

# Data loader
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Batch processing
for batch in dataloader:
    images = batch['image'].to(device)           # 16×1×320×320
    conditions = batch['cond'].to(device)         # 16×3×80×80
    masks = batch.get('ori_mask', None)          # 16×1×320×320 (optional)
    names = batch['name']                        # file identifiers
```

### Data Augmentation

```python
# Data augmentation pipeline
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=90),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
])
```

---

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

---

## 📈 Expected Results

### Training Metrics
- **Total Loss**: Should decrease from ~1.0 to ~0.1 over 50k steps
- **Reconstruction Loss**: Primary indicator of model performance
- **KL Loss**: Should remain stable (low values indicate good latent space)
- **Sample Quality**: Visual inspection of generated samples every 200 steps

### Model Outputs
- **Checkpoints**: Saved every 200 steps in `./radiodiff_results_LDM/`
- **Samples**: Generated images for quality assessment
- **Logs**: TensorBoard logs for monitoring training progress
- **Model Reports**: Parameter statistics and freezing information

---

## 🔧 Technical Implementation

### Hardware Requirements
- **GPU**: High-end GPU with at least 16GB VRAM for batch size 16
- **Memory**: Sufficient RAM for data loading and processing
- **Storage**: Adequate space for model checkpoints and samples

### Software Dependencies
- **PyTorch**: Deep learning framework
- **Accelerate**: Distributed training support
- **Mermaid-cli**: Diagram rendering
- **TensorBoard**: Training monitoring

---

## 🎉 Conclusion

This simplified overview provides a clear understanding of the RadioDiff VAE training process. The model combines VAE compression, conditional U-Net denoising, and latent diffusion for high-quality radio wave propagation prediction.

The key innovations include:
- **Efficient Architecture**: VAE compression reduces computational requirements
- **Conditional Generation**: Building-aware radio signal prediction
- **Stable Training**: EMA smoothing and cosine annealing for convergence
- **Multi-scale Processing**: Swin Transformer for efficient spatial modeling

This implementation demonstrates state-of-the-art performance in radio wave propagation prediction while maintaining computational efficiency.

---

## 📚 References

- **Latent Diffusion Models**: Rombach et al. (2022)
- **High-Resolution Image Synthesis**: Rombach et al. (2022)
- **Swin Transformer**: Liu et al. (2021)
- **Variational Autoencoders**: Kingma & Welling (2014)
- **RadioMapSeer Dataset**: Radio wave propagation dataset

## 📄 License

This project is for research purposes. Please ensure compliance with the original licenses of the referenced works and datasets.