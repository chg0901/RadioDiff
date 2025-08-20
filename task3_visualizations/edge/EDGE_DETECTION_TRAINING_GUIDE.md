# Edge Detection Training and Inference Guide

This guide provides comprehensive instructions for training and performing edge detection inference using the RadioDiff framework with AdaptEdgeDataset.

## Overview

The edge detection implementation consists of three main stages:
1. **VAE Training**: Train a Variational Autoencoder for edge representation
2. **LDM Training**: Train a Latent Diffusion Model for edge generation
3. **Inference**: Use trained models for edge detection on new images

## Configuration Files

### 1. VAE Training Configuration (`configs_edge/edge_vae_train.yaml`)

This configuration trains a VAE model specifically designed for edge detection:

```yaml
model:
  embed_dim: 3
  ddconfig:
    in_channels: 1  # Edge maps are single channel
    out_ch: 1
    resolution: [320, 320]
    ch: 128
    ch_mult: [1, 2, 4]
    
data:
  name: edge
  batch_size: 8
  data_root: '/path/to/edge/dataset'
  threshold: 0.3
  use_uncertainty: False
  crop_type: 'rand_crop'
```

**Key Features:**
- Single-channel input/output for edge maps
- Configurable edge detection threshold
- Support for uncertainty modeling
- Flexible data augmentation options

### 2. LDM Training Configuration (`configs_edge/edge_ldm_train.yaml`)

This configuration trains a Latent Diffusion Model for edge generation:

```yaml
model:
  model_type: const_sde
  model_name: cond_unet
  sampling_timesteps: 50
  objective: pred_noise
  
data:
  name: edge
  batch_size: 16
  data_root: '/path/to/edge/dataset'
  threshold: 0.3
  crop_type: 'rand_crop'
```

**Key Features:**
- Conditional U-Net architecture
- Sliding window support for large images
- Fast sampling with few timesteps
- Configurable batch size for memory efficiency

### 3. Inference Configuration (`configs_edge/edge_sample.yaml`)

This configuration is used for edge detection inference:

```yaml
sampler:
  sample_type: "slide"
  stride: [240, 240]
  batch_size: 1
  use_ema: True
  
data:
  name: edge
  img_folder: '/path/to/test/images'
```

**Key Features:**
- Sliding window inference for large images
- EMA model support for better quality
- Configurable overlap between windows
- Batch processing support

## Dataset Structure

The AdaptEdgeDataset expects the following directory structure:

```
data_root/
├── image/
│   ├── raw/
│   │   ├── subset1/
│   │   │   ├── img1.jpg
│   │   │   └── img2.png
│   │   └── subset2/
│   │       ├── img3.jpg
│   │       └── img4.png
└── edge/
    └── raw/
        ├── subset1/
        │   ├── img1.jpg
        │   └── img2.png
        └── subset2/
            ├── img3.jpg
            └── img4.png
```

**Requirements:**
- Images should be in RGB format
- Edge maps should be grayscale
- Supported formats: .jpg, .png, .pgm, .ppm
- File extensions are automatically corrected if needed

## Training Scripts

### 1. VAE Training Script (`train_edge_vae.sh`)

```bash
#!/bin/bash
# Basic usage
./train_edge_vae.sh

# Or manually with specific configuration
python train_vae.py --cfg configs_edge/edge_vae_train.yaml
```

**Configuration Updates Needed:**
- Update `data_root` in `configs_edge/edge_vae_train.yaml`
- Adjust `batch_size` based on GPU memory
- Modify `results_folder` as needed

### 2. LDM Training Script (`train_edge_ldm.sh`)

```bash
#!/bin/bash
# Basic usage
./train_edge_ldm.sh

# Or manually with specific configuration
python train_cond_ldm_m.py --cfg configs_edge/edge_ldm_train.yaml
```

**Configuration Updates Needed:**
- Update `data_root` in `configs_edge/edge_ldm_train.yaml`
- Ensure `first_stage.ckpt_path` points to trained VAE
- Adjust learning rate based on convergence

### 3. Inference Script (`infer_edge.sh`)

```bash
#!/bin/bash
# Basic usage with required parameters
./infer_edge.sh \
    --input_dir /path/to/test/images \
    --out_dir /path/to/results \
    --pre_weight /path/to/trained/model.pt

# Or manually with all parameters
python demo.py \
    --cfg configs_edge/edge_sample.yaml \
    --input_dir /path/to/test/images \
    --out_dir /path/to/results \
    --pre_weight /path/to/trained/model.pt \
    --bs 8 \
    --sampling_timesteps 1
```

## Step-by-Step Usage Guide

### Step 1: Prepare Dataset

1. Organize your dataset according to the expected structure
2. Ensure edge maps are properly thresholded
3. Verify file formats and extensions

### Step 2: Train VAE Model

1. Update `configs_edge/edge_vae_train.yaml`:
   ```yaml
   data:
     data_root: '/your/dataset/path'
     threshold: 0.3  # Adjust based on your edge maps
     use_uncertainty: false  # Set to true for uncertain edges
   ```

2. Run VAE training:
   ```bash
   ./train_edge_vae.sh
   ```

3. Monitor training progress in `edge_vae_results/`

### Step 3: Train LDM Model

1. Update `configs_edge/edge_ldm_train.yaml`:
   ```yaml
   data:
     data_root: '/your/dataset/path'
   first_stage:
     ckpt_path: './edge_vae_results/model-30.pt'  # Update with your VAE path
   ```

2. Run LDM training:
   ```bash
   ./train_edge_ldm.sh
   ```

3. Monitor training progress in `edge_ldm_results/`

### Step 4: Perform Inference

1. Update `configs_edge/edge_sample.yaml`:
   ```yaml
   data:
     img_folder: '/your/test/images/path'
   sampler:
     ckpt_path: './edge_ldm_results/model-50.pt'  # Update with your LDM path
   ```

2. Run inference:
   ```bash
   ./infer_edge.sh \
       --input_dir /your/test/images \
       --out_dir /your/results \
       --pre_weight ./edge_ldm_results/model-50.pt
   ```

## Advanced Configuration Options

### Edge Detection Parameters

- **threshold**: Controls edge sensitivity (0.0-1.0)
- **use_uncertainty**: Enables handling of ambiguous edge regions
- **crop_type**: Data augmentation strategy ('rand_crop' or 'rand_resize_crop')

### Training Parameters

- **batch_size**: Adjust based on GPU memory
- **lr**: Learning rate for training
- **train_num_steps**: Total training iterations
- **sampling_timesteps**: Number of denoising steps (fewer = faster but lower quality)

### Inference Parameters

- **sample_type**: Inference strategy ('slide' or 'whole')
- **stride**: Overlap between sliding windows
- **batch_size**: Number of images processed simultaneously
- **use_ema**: Use EMA model for better quality

## Performance Optimization

### GPU Memory Optimization

1. Reduce batch size if encountering OOM errors
2. Use mixed precision training (set `amp: True`)
3. Enable gradient accumulation for large effective batch sizes

### Training Speed Optimization

1. Increase number of workers (`num_workers`)
2. Use smaller image sizes if acceptable
3. Reduce sampling timesteps for faster inference

### Quality Optimization

1. Use EMA models for inference
2. Increase sampling timesteps for better quality
3. Fine-tune threshold parameter for your specific dataset

## Troubleshooting

### Common Issues

**VAE Training Issues:**
- Ensure edge maps are properly normalized
- Check that dataset structure matches expectations
- Verify file permissions and paths

**LDM Training Issues:**
- Confirm VAE checkpoint path is correct
- Check that first stage model is properly trained
- Monitor loss curves for convergence

**Inference Issues:**
- Ensure input images are in correct format
- Verify model checkpoint paths
- Check output directory permissions

### Debug Commands

```bash
# Test dataset loading
python -c "
from denoising_diffusion_pytorch.data import AdaptEdgeDataset
dataset = AdaptEdgeDataset('/path/to/dataset', [320, 320])
print(f'Dataset size: {len(dataset)}')
print(f'Sample: {dataset[0]}')
"

# Test model loading
python -c "
import torch
model = torch.load('/path/to/model.pt', map_location='cpu')
print(f'Model keys: {list(model.keys())}')
"
```

## Results and Evaluation

### Expected Results

1. **VAE Training**: Should converge within 50k-150k steps
2. **LDM Training**: Should show stable loss curves
3. **Inference**: Should produce clean, thresholded edge maps

### Quality Metrics

- Visual inspection of edge maps
- Comparison with ground truth edges
- Processing speed benchmarks
- Memory usage statistics

## Conclusion

This edge detection implementation provides a complete pipeline for training and inference using diffusion models. The AdaptEdgeDataset offers flexible data handling with support for uncertainty modeling and various augmentation strategies. The sliding window inference enables processing of large images while maintaining quality.

For best results, experiment with different threshold values and training parameters to optimize for your specific use case.