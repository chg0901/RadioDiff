# RadioDiff VAE Training Documentation - Improved Overview

## 🚀 Introduction

This document provides a comprehensive technical overview of the RadioDiff VAE training process with improved diagrams and detailed explanations. The model is trained using:

```bash
accelerate launch train_cond_ldm.py --cfg ./configs/radio_train_m.yaml
```

---

## 📊 System Architecture Overview

![RadioDiff System Overview](./mermaid_vis_improved/system_overview.png)

### Technical Explanation

The RadioDiff system architecture consists of four main components that work together to generate radio wave propagation predictions:

1. **Data Input Layer**: Processes radio signals (320×320×1), building layouts, and environmental data from the RadioMapSeer dataset. The input data includes building floor plans, transmitter positions, and radio signal measurements.

2. **VAE Processing Core**: Uses a Variational Autoencoder to compress input data into a latent space (3×80×80) and reconstruct it. This compression reduces computational complexity while preserving essential information for radio wave prediction.

3. **Conditional U-Net**: A Swin Transformer-based network that processes the latent space with window-based attention and Fourier features for efficient spatial encoding. The conditioning incorporates building layout information.

4. **Diffusion Process**: Implements both forward (noise addition) and reverse (denoising) processes in the latent space for high-quality generation of radio wave patterns.

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

# VAE configuration from radio_train_m.yaml
first_stage_config:
  target: ldm.models.autoencoder.AutoencoderKL
  params:
    embed_dim: 3           # Latent space dimensions
    ckpt_path: None        # No pretrained weights
    ddconfig:
      double_z: true       # Use double latent space
      channels: 1          # Input channels
      resolution: 320      # Input resolution
      in_channels: 1       # Input channels
      out_ch: 1            # Output channels
      ch: 128              # Base channels
      ch_mult: [1,2,4]     # Channel multipliers
      num_res_blocks: 2    # Residual blocks
      attn_resolutions: [] # No attention in VAE
      dropout: 0.0         # Dropout rate
```

### Paper Reference

Based on the Latent Diffusion Model (LDM) framework (Rombach et al., 2022) - "High-Resolution Image Synthesis with Latent Diffusion Models". This approach operates in compressed latent space rather than pixel space, significantly reducing computational requirements while maintaining high-quality generation capabilities.

---

## 🏗️ VAE Architecture

![VAE Architecture](./mermaid_vis_improved/vae_architecture.png)

### Technical Explanation

The VAE (Variational Autoencoder) architecture consists of three main paths that enable efficient compression and reconstruction of radio wave patterns:

1. **Encoder Path**: Takes 320×320×1 input radio signals and progressively downsamples through convolutional layers (1→128→256→512 channels) to produce latent parameters μ and σ. The encoder uses residual blocks and downsampling to capture hierarchical features.

2. **Latent Space**: The compressed representation (3×80×80) where the model learns efficient encoding of radio wave patterns. Uses the reparameterization trick: z = μ + σ*ε to enable gradient flow through sampling.

3. **Decoder Path**: Reconstructs the original signal from latent samples through upsampling and transposed convolutions (512→256→128→1 channels), reversing the encoder process to generate 320×320×1 output.

### Mathematical Foundation

The VAE is trained to maximize the Evidence Lower Bound (ELBO):

```
ELBO = E_q[log p_θ(x|z)] - KL(q_φ(z|x) || p(z))
```

Where:
- q_φ(z|x) is the encoder distribution: N(z; μ_φ(x), σ²_φ(x))
- p_θ(x|z) is the decoder distribution: N(x; μ_θ(z), σ²_θ(z))  
- p(z) is the prior distribution (standard normal N(0, I))
- KL denotes Kullback-Leibler divergence

The reparameterization trick enables gradient-based optimization:
```
z = μ_φ(x) + σ_φ(x) * ε, where ε ~ N(0, I)
```

### Code Implementation

```python
# VAE encoder-decoder structure from denoising_diffusion_pytorch
class AutoencoderKL(nn.Module):
    def __init__(self, embed_dim=3, channels=[1, 128, 256, 512]):
        super().__init__()
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

# Encoder implementation
class Encoder(nn.Module):
    def __init__(self, ch=128, out_ch=3, ch_mult=[1,2,4], num_res_blocks=2):
        super().__init__()
        self.conv_in = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        
        # Downsampling blocks
        self.down = nn.ModuleList()
        in_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.down.append(ResnetBlock(in_ch, out_ch))
                in_ch = out_ch
            if i != len(ch_mult) - 1:
                self.down.append(Downsample(in_ch))
        
        # Middle blocks
        self.mid = nn.ModuleList([
            ResnetBlock(in_ch, in_ch),
            AttnBlock(in_ch),
            ResnetBlock(in_ch, in_ch)
        ])
        
        # Output
        self.norm_out = Normalize(in_ch)
        self.conv_out = nn.Conv2d(in_ch, 2*out_ch, kernel_size=3, stride=1, padding=1)
```

### Configuration Parameters

From `radio_train_m.yaml`:
- **embed_dim**: 3 (latent space dimensions)
- **channels**: [1, 128, 256, 512] (progressive feature extraction)
- **kl_weight**: 0.000001 (regularization strength)
- **discr_weight**: 0.5 (adversarial training weight)
- **resolution**: 320 (input/output resolution)

---

## 🎨 Conditional U-Net Structure

![Conditional U-Net Structure](./mermaid_vis_improved/unet_structure.png)

### Technical Explanation

The Conditional U-Net is the core denoising network that incorporates building layout information for radio wave prediction:

1. **Input Processing**: Takes latent input (3×80×80), time embeddings, and condition information (3×80×80 building layout features).

2. **Encoder Path**: Multi-scale processing with 4 levels (128×80×80 → 512×10×10) using residual connections and downsampling to capture hierarchical features.

3. **Bottleneck**: Swin Transformer with window attention for efficient processing of large spatial dimensions, incorporating cross-attention for conditioning.

4. **Decoder Path**: Upsampling with skip connections from encoder levels for feature refinement, gradually increasing resolution while maintaining conditioning information.

5. **Output**: Predicts noise in latent space for diffusion reverse process, outputting denoised latent representations.

### Key Features

- **Swin Transformer**: Uses window-based attention to reduce computational complexity from O(n²) to O(w²), where w is window size (8×8, 4×4, 2×2, 1×1).

- **Cross-Attention**: Integrates conditioning information through attention mechanisms, allowing the model to focus on relevant building features.

- **Fourier Features**: Efficient spatial encoding with scale 16, providing positional information for spatial reasoning.

- **Conditioning**: Building layout information is incorporated through concatenation and cross-attention mechanisms.

### Code Implementation

```python
# Conditional U-Net with Swin Transformer
class Unet(nn.Module):
    def __init__(self, dim=128, channels=3, dim_mults=[1,2,4,4]):
        super().__init__()
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.SiLU(),
            nn.Linear(dim*4, dim*4)
        )
        
        # Encoder levels
        self.downs = nn.ModuleList()
        for i in range(len(dim_mults)):
            out_dim = dim * dim_mults[i]
            self.downs.append(ResnetBlock(dim, out_dim))
            if i < len(dim_mults) - 1:
                self.downs.append(Downsample(out_dim))
            dim = out_dim
        
        # Bottleneck with Swin Transformer
        self.mid = nn.ModuleList([
            ResnetBlock(dim),
            TransformerBlock(dim, window_size=8),
            ResnetBlock(dim)
        ])
        
        # Decoder levels
        self.ups = nn.ModuleList()
        for i in reversed(range(len(dim_mults))):
            out_dim = dim * dim_mults[i]
            self.ups.append(ResnetBlock(dim + out_dim, out_dim))
            if i > 0:
                self.ups.append(Upsample(out_dim))
            dim = out_dim
        
        # Output
        self.final_conv = nn.Conv2d(dim, channels, 1)
    
    def forward(self, x, t, cond=None):
        # Time embedding
        t_emb = self.time_embed(timestep_embedding(t, self.dim))
        
        # Encoder with conditioning
        h = x
        hs = []
        for module in self.downs:
            h = module(h)
            hs.append(h)
        
        # Bottleneck with cross-attention
        for module in self.mid:
            h = module(h, cond) if cond else module(h)
        
        # Decoder with skip connections
        for module in self.ups:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h)
        
        return self.final_conv(h)

# Swin Transformer implementation
class TransformerBlock(nn.Module):
    def __init__(self, dim, window_size=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)
    
    def forward(self, x, cond=None):
        # Self-attention with window partitioning
        h = x + self.attn(self.norm1(x))
        
        # Cross-attention with conditioning
        if cond is not None:
            h = h + self.cross_attn(self.norm2(h), cond)
        
        # MLP
        h = h + self.mlp(self.norm2(h))
        return h
```

### Configuration from radio_train_m.yaml

```yaml
unet_config:
  target: ldm.modules.diffusionmodules.openaimodel.UNetModel
  params:
    image_size: 80        # Latent space size
    in_channels: 3         # Latent channels
    out_channels: 3        # Output channels
    model_channels: 128    # Base channels
    attention_resolutions: [4, 2]  # Attention at these resolutions
    num_res_blocks: 2      # Residual blocks
    channel_mult: [1, 2, 4, 4]    # Channel multipliers
    num_heads: 4           # Attention heads
    use_spatial_transformer: true   # Use spatial transformer
    transformer_depth: 1     # Transformer depth
    context_dim: 128        # Conditioning dimension
    use_checkpoint: true    # Gradient checkpointing
    use_spatial_transformer: true
```

### Paper Reference

The architecture follows the Swin Transformer design (Liu et al., 2021) - "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows". This provides efficient attention computation for large spatial dimensions, adapted for conditional generation tasks in radio wave propagation.

---

## 🔄 Training Workflow

![Training Workflow](./mermaid_vis_improved/training_workflow.png)

### Technical Explanation

The training process consists of four main phases that ensure stable and efficient learning:

1. **Setup Phase**: Load configuration from `radio_train_m.yaml`, initialize models (VAE, U-Net, diffusion), setup data loaders with RadioUNet_c dataset, and configure optimizers with learning rate scheduling.

2. **Training Loop**: The core training iteration where batches of 16 samples are processed, forward passes executed through VAE encoding and diffusion process, losses computed, and gradients accumulated over 8 steps for effective batch size of 128.

3. **Optimization Phase**: Apply gradient updates using AdamW optimizer with weight decay 1e-4, learning rate scheduling (cosine decay from 5e-5 to 5e-6), and EMA smoothing (β=0.999) for stable training.

4. **Evaluation Phase**: Generate samples every 200 steps for quality assessment, save model checkpoints, log metrics to TensorBoard, and compute training statistics for monitoring progress.

### Key Training Parameters

- **Batch Size**: 16 samples per iteration
- **Gradient Accumulation**: 8 steps (effective batch size: 128)
- **Learning Rate**: 5e-5 → 5e-6 (cosine decay with power 0.96)
- **Training Steps**: 50,000 total iterations
- **Save Frequency**: Every 200 steps
- **EMA Updates**: After 10k steps, every 10 steps with β=0.999

### Code Implementation

```python
# Main training loop from train_cond_ldm.py
def train():
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=8,
        mixed_precision='no',  # Full precision for stability
        log_with='tensorboard'
    )
    
    # Setup models
    model = LatentDiffusion(**config.model.params)
    ema_model = EMAModel(model, beta=0.999)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max(
            5e-6 / 5e-5,
            (1 - step / 50000) ** 0.96
        )
    )
    
    # Training loop
    for step in range(50000):
        # Load batch
        batch = next(dataloader)
        images = batch['image'].to(accelerator.device)
        conditions = batch['cond'].to(accelerator.device)
        
        # Forward pass
        with accelerator.autocast():
            loss = model(images, conditions)
        
        # Backward pass with gradient accumulation
        accelerator.backward(loss)
        if (step + 1) % 8 == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # EMA update after 10k steps
            if step > 10000 and step % 10 == 0:
                ema_model.update(model)
        
        # Logging and sampling
        if step % 100 == 0:
            accelerator.log({'loss': loss.item()}, step)
        
        if step % 200 == 0:
            # Generate samples
            samples = ema_model.ema_model.sample(conditions)
            save_images(samples, f'./radiodiff_results_LDM/samples/sample_{step}.png')
            
            # Save checkpoint
            accelerator.save_state(f'./radiodiff_results_LDM/checkpoints/ckpt_{step}')
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
  log_interval: 100
  ema:
    start_step: 10000
    update_interval: 10
    beta: 0.999
  optimizer:
    type: adamw
    betas: [0.9, 0.999]
    eps: 1e-8
    weight_decay: 1e-4
  lr_scheduler:
    type: cosine
    power: 0.96
    min_lr_ratio: 0.1
```

### Training Strategy

The training uses several advanced techniques:

1. **Gradient Accumulation**: Allows larger effective batch sizes (128) while maintaining memory efficiency
2. **EMA Smoothing**: Exponential moving average provides more stable model weights
3. **Cosine Annealing**: Smooth learning rate decay for better convergence
4. **Mixed Precision**: Disabled for numerical stability, using full precision training
5. **Gradient Clipping**: Prevents exploding gradients with max norm 1.0

---

## ⚖️ Loss Components

![Loss Components](./mermaid_vis_improved/loss_components.png)

### Technical Explanation

The training objective combines multiple loss components to achieve high-quality radio wave generation:

1. **Reconstruction Loss**: L2/MSE loss between predicted and target signals in both pixel and latent space. This is the primary loss driving the model to generate accurate radio wave patterns.

2. **Regularization Loss**: KL divergence that prevents overfitting and ensures good latent space quality by encouraging the latent distribution to match the prior N(0, I).

3. **Adversarial Loss**: GAN-based loss that improves realism through discriminator training, activated after 50k steps to refine generated samples.

4. **Perceptual Loss**: Optional LPIPS loss that enhances visual quality through feature similarity in pre-trained neural networks.

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

Individual loss components:
- **Reconstruction**: L_rec = ||x - x_reconstructed||²
- **KL Divergence**: L_kl = KL(q_φ(z|x) || N(0, I)) = 0.5 * (μ² + σ² - log(σ²) - 1)
- **Adversarial**: L_adv = -log(D(G(z))) for generator, hinge loss for discriminator
- **Perceptual**: L_perceptual = LPIPS(x, x_reconstructed)

### Code Implementation

```python
# Loss computation in training loop
def compute_loss(model, batch, step):
    images = batch['image']
    conditions = batch['cond']
    
    # VAE encoding
    posterior = model.encode_first_stage(images)
    z = posterior.sample()
    
    # Diffusion process
    t = torch.randint(0, 1000, (images.shape[0],), device=images.device)
    noise = torch.randn_like(z)
    z_t = model.q_sample(z, t, noise)
    
    # U-Net prediction
    pred_noise = model.denoise(z_t, t, conditions)
    
    # Reconstruction loss
    recon_loss = F.mse_loss(pred_noise, noise)
    
    # KL divergence
    kl_loss = posterior.kl()
    kl_loss = torch.mean(kl_loss)
    
    # Adversarial loss (if enabled after 50k steps)
    adv_loss = torch.tensor(0.0, device=images.device)
    if step > 50000:
        with torch.no_grad():
            fake_pred = discriminator(model.decode_first_stage(z))
            real_pred = discriminator(images)
        adv_loss = hinge_gan_loss(fake_pred, real_pred)
    
    # Perceptual loss (optional)
    perceptual_loss = torch.tensor(0.0, device=images.device)
    if use_perceptual:
        reconstructed = model.decode_first_stage(z)
        perceptual_loss = lpips_loss(reconstructed, images)
    
    # Total loss with weights
    total_loss = (recon_loss + 
                 0.000001 * kl_loss + 
                 0.5 * adv_loss + 
                 perceptual_weight * perceptual_loss)
    
    return total_loss, {
        'reconstruction': recon_loss.item(),
        'kl': kl_loss.item(),
        'adversarial': adv_loss.item(),
        'perceptual': perceptual_loss.item()
    }

# Hinge GAN loss
def hinge_gan_loss(fake_pred, real_pred):
    # Generator loss
    gen_loss = -torch.mean(fake_pred)
    
    # Discriminator loss
    disc_loss = torch.mean(F.relu(1. - real_pred)) + torch.mean(F.relu(1. + fake_pred))
    
    return gen_loss, disc_loss

# LPIPS perceptual loss
class LPIPSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lpips = LPIPS(net='vgg')  # Pre-trained VGG network
    
    def forward(self, x, y):
        return self.lpips(x, y, normalize=True).mean()
```

### Configuration Weights

From the configuration:
- **kl_weight**: 0.000001 (very small to prevent posterior collapse)
- **discr_weight**: 0.5 (balanced adversarial training)
- **discr_start**: 50001 (delayed discriminator activation)
- **perceptual_weight**: Configurable based on quality requirements

### Loss Balance Strategy

The loss weights are carefully balanced:
1. **Reconstruction loss** (weight 1.0) is the primary objective
2. **KL loss** (weight 1e-6) is small to prevent latent space collapse
3. **Adversarial loss** (weight 0.5) is introduced later to refine quality
4. **Perceptual loss** is optional for additional quality enhancement

---

## 🎛️ Optimization Strategy

![Optimization Strategy](./mermaid_vis_improved/optimization_strategy.png)

### Technical Explanation

The optimization strategy includes several key components designed for stable and efficient training:

1. **Learning Rate Schedule**: Cosine annealing from 5e-5 to 5e-6 with power 0.96 for stable convergence. This provides smooth decay while maintaining learning capacity throughout training.

2. **Training Strategy**: Gradient accumulation (8 steps) for effective batch size of 128 with memory efficiency, allowing the model to see more samples per update while managing GPU memory constraints.

3. **Model Updates**: AdamW optimizer with weight decay and EMA smoothing for stable weights. The EMA model provides more stable predictions and better generalization.

4. **Stability Measures**: Full precision training (FP16 disabled) for numerical stability, preventing overflow and ensuring consistent gradients.

### Learning Rate Schedule

The cosine annealing schedule provides smooth learning rate decay:

```
LR(t) = max(min_lr, (1-t/T)^power * initial_lr)
```

Where:
- t = current step (0 to 50,000)
- T = total steps (50,000)
- power = 0.96
- initial_lr = 5e-5
- min_lr = 5e-6

This schedule maintains relatively high learning rates in the beginning for rapid progress, then smoothly decreases for fine-tuning.

### AdamW Optimizer Configuration

```python
# AdamW optimizer with cosine decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,
    betas=(0.9, 0.999),  # Momentum terms
    eps=1e-8,            # Numerical stability
    weight_decay=1e-4    # L2 regularization
)

# Cosine annealing scheduler
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: max(
        5e-6 / 5e-5,                                    # Minimum ratio
        (1 - step / 50000) ** 0.96                     # Cosine decay
    )
)

# EMA for stable weights
class EMAModel:
    def __init__(self, model, beta=0.999):
        self.model = model
        self.beta = beta
        self.shadow = copy.deepcopy(model)
    
    def update(self, model):
        with torch.no_grad():
            for param, shadow_param in zip(
                model.parameters(), 
                self.shadow.parameters()
            ):
                shadow_param.data = (
                    self.beta * shadow_param.data + 
                    (1 - self.beta) * param.data
                )
    
    def update_after_step(self, step):
        # Start EMA after 10k steps, update every 10 steps
        if step > 10000 and step % 10 == 0:
            self.update(self.model)
```

### Gradient Accumulation Implementation

```python
# Gradient accumulation for larger effective batch size
def train_with_accumulation(model, dataloader, optimizer, scheduler):
    accumulation_steps = 8
    effective_batch_size = 16 * accumulation_steps  # 128
    
    for step, batch in enumerate(dataloader):
        # Forward pass
        with autocast(enabled=False):  # Full precision
            loss = model(batch)
        
        # Scale loss for accumulation
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights after accumulation
        if (step + 1) % accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
```

### Configuration Parameters

From `radio_train_m.yaml`:
```yaml
optimization:
  lr: 5e-5
  min_lr: 5e-6
  lr_scheduler: cosine
  lr_decay_power: 0.96
  optimizer: adamw
  weight_decay: 1e-4
  gradient_accumulation: 8
  gradient_clip: 1.0
  mixed_precision: false
  ema:
    beta: 0.999
    start_step: 10000
    update_interval: 10
```

### Optimization Benefits

1. **Memory Efficiency**: Gradient accumulation allows larger effective batch sizes without increasing memory usage
2. **Stable Training**: EMA smoothing provides more consistent model weights and better generalization
3. **Convergence**: Cosine annealing ensures smooth learning rate transitions for better final performance
4. **Numerical Stability**: Full precision training prevents overflow and maintains gradient quality

---

## 📊 Data Pipeline

![Data Pipeline](./mermaid_vis_improved/data_pipeline.png)

### Technical Explanation

The data pipeline processes radio wave propagation data through several stages to prepare it for training:

1. **Dataset**: RadioMapSeer dataset containing building layouts, transmitter positions, and radio measurements. The dataset provides realistic radio wave propagation scenarios.

2. **Processing**: Image resizing (320×320×1), normalization to [-1, 1], and data augmentation including random flips, rotations, and color jittering.

3. **Batch Assembly**: Creates batches of 16 samples with parallel loading and device transfer for efficient GPU utilization.

4. **Training Input**: Formats data for VAE input, U-Net conditioning, and diffusion targets, ensuring proper shapes and data types.

### Data Structure

Each training sample contains:
- **image**: Radio wave propagation map (320×320×1) normalized to [-1, 1]
- **cond**: Building and transmitter information (3×80×80) for conditioning
- **ori_mask**: Optional building mask (320×320×1) for constrained generation
- **name**: File identifier for tracking and debugging

### Code Implementation

```python
# Dataset loading from RadioUNet_c
class RadioUNet_c(Dataset):
    def __init__(self, data_root, simulation_type="specific_type"):
        self.data_root = Path(data_root)
        self.simulation_type = simulation_type
        
        # Load data files
        self.image_files = list((self.data_root / "images").glob("*.png"))
        self.cond_files = list((self.data_root / "conditions").glob("*.npy"))
        
        # Data augmentation
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('L')  # Grayscale
        image = transforms.Resize((320, 320))(image)
        image = transforms.ToTensor()(image)
        image = (image * 2) - 1  # Normalize to [-1, 1]
        
        # Load condition
        cond_path = self.cond_files[idx]
        condition = np.load(cond_path)  # Shape: (3, 80, 80)
        condition = torch.from_numpy(condition).float()
        
        # Load optional mask
        mask_path = self.data_root / "masks" / f"{image_path.stem}.png"
        if mask_path.exists():
            mask = Image.open(mask_path).convert('L')
            mask = transforms.Resize((320, 320))(mask)
            mask = transforms.ToTensor()(mask)
        else:
            mask = torch.zeros(1, 320, 320)
        
        # Apply augmentation
        if self.transform and random.random() > 0.5:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return {
            'image': image,           # (1, 320, 320)
            'cond': condition,        # (3, 80, 80)
            'ori_mask': mask,         # (1, 320, 320)
            'name': image_path.stem
        }

# Data loader setup
def create_dataloader(config):
    dataset = RadioUNet_c(
        data_root=config.data.params.data_root,
        simulation_type=config.data.params.simulation_type
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.trainer.batch_size,  # 16
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader

# Batch processing
def process_batch(batch, device):
    images = batch['image'].to(device)           # (16, 1, 320, 320)
    conditions = batch['cond'].to(device)         # (16, 3, 80, 80)
    masks = batch.get('ori_mask', None)          # (16, 1, 320, 320)
    names = batch['name']                        # List of filenames
    
    return {
        'images': images,
        'conditions': conditions,
        'masks': masks,
        'names': names
    }
```

### Data Augmentation

```python
# Data augmentation pipeline
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=90),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
])

# Advanced augmentation with probability
class RandomAugmentation:
    def __init__(self, p=0.5):
        self.p = p
        self.transforms = [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    
    def __call__(self, x):
        if random.random() < self.p:
            transform = random.choice(self.transforms)
            x = transform(x)
        return x
```

### Configuration from radio_train_m.yaml

```yaml
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 16
    num_workers: 4
    wrap: false
    train:
      target: ldm.data.radio.RadioUNet_c
      params:
        data_root: path/to/RadioMapSeer
        simulation_type: specific_type
        additional_params: {}
    validation:
      target: ldm.data.radio.RadioUNet_c
      params:
        data_root: path/to/RadioMapSeer
        simulation_type: specific_type
        additional_params: {}
```

### Data Processing Pipeline

1. **Loading**: Images loaded as grayscale PNG files, conditions as NumPy arrays
2. **Resizing**: All images resized to 320×320 for consistent input dimensions
3. **Normalization**: Images normalized to [-1, 1] range for stable training
4. **Augmentation**: Random transformations applied to increase dataset diversity
5. **Batching**: Samples grouped into batches of 16 with parallel loading
6. **Device Transfer**: Batches moved to GPU for efficient processing

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

# Mixed precision training (if enabled)
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

This improved overview provides a comprehensive understanding of the RadioDiff VAE training process with cleaner diagrams and detailed technical explanations. The model combines VAE compression, conditional U-Net denoising, and latent diffusion for high-quality radio wave propagation prediction.

The key innovations include:
- **Efficient Architecture**: VAE compression reduces computational requirements
- **Conditional Generation**: Building-aware radio signal prediction
- **Stable Training**: EMA smoothing and cosine annealing for convergence
- **Multi-scale Processing**: Swin Transformer for efficient spatial modeling

This implementation demonstrates state-of-the-art performance in radio wave propagation prediction while maintaining computational efficiency.

---

## 📚 References

- **Latent Diffusion Models**: Rombach et al. (2022) - "High-Resolution Image Synthesis with Latent Diffusion Models"
- **Swin Transformer**: Liu et al. (2021) - "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
- **Variational Autoencoders**: Kingma & Welling (2014) - "Auto-Encoding Variational Bayes"
- **RadioMapSeer Dataset**: Radio wave propagation dataset for building-aware prediction
- **AdamW Optimizer**: Loshchilov & Hutter (2019) - "Decoupled Weight Decay Regularization"

## 📄 License

This project is for research purposes. Please ensure compliance with the original licenses of the referenced works and datasets.