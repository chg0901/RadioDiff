# ICASSP2025 Multi-Condition Dataset Documentation

## Overview

The ICASSP2025 Multi-Condition Dataset is a comprehensive dataset designed for training multi-condition Variational Autoencoders (VAEs) and diffusion models for radio map prediction. This dataset provides structured conditional inputs and target outputs for advanced radio propagation modeling.

## Dataset Architecture

### Multi-Condition Structure

The dataset is organized into three main components:

1. **Condition 1**: 2-channel input (reflectance + transmittance maps)
2. **Condition 2**: 1-channel input (distance/FSPL maps)
3. **Target**: 1-channel output (path loss maps)

### Data Flow

```
Input (3-channel RGB) → Multi-Condition Splitting → Training Ready
├── Channel 0: Reflectance → Condition 1 Channel 0
├── Channel 1: Transmittance → Condition 1 Channel 1
└── Channel 2: Distance → Condition 2 Channel 0
```

## Dataset Statistics

### Overall Statistics
- **Total Samples**: 1,250
- **Image Resolution**: 320×320 pixels
- **Spatial Resolution**: 0.25m per pixel
- **Data Format**: PNG (uint8, 0-255 range)
- **Success Rate**: 100% (all samples processed successfully)

### Data Split
- **Training**: 1,000 samples (80%)
- **Validation**: 125 samples (10%)
- **Test**: 125 samples (10%)

### Building Coverage
The dataset includes 25 different buildings:
- Buildings: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
- **All buildings** are represented across all splits

### Metadata Distribution
- **Frequencies**: 1 frequency band
- **Antennas**: 1 antenna configuration
- **Samples per Building**: ~50 samples on average

## Directory Structure

```
icassp2025_multi_condition_vae/
├── train/
│   ├── condition1/     # 2-channel (reflectance + transmittance)
│   ├── condition2/     # 1-channel (distance maps)
│   └── target/         # 1-channel (path loss)
├── val/
│   ├── condition1/     # 2-channel (reflectance + transmittance)
│   ├── condition2/     # 1-channel (distance maps)
│   └── target/         # 1-channel (path loss)
├── test/
│   ├── condition1/     # 2-channel (reflectance + transmittance)
│   ├── condition2/     # 1-channel (distance maps)
│   └── target/         # 1-channel (path loss)
├── dataset_statistics.yaml    # Dataset metadata
└── vae_config.yaml           # VAE configuration
```

## File Naming Convention

Files follow the ICASSP2025 naming convention:
```
B{building_id}_Ant{antenna_id}_f{frequency_id}_S{sample_id}.png
```

Example: `B1_Ant1_f1_S0.png` represents:
- Building ID: 1
- Antenna ID: 1
- Frequency ID: 1
- Sample ID: 0

## Data Characteristics

### Condition 1 (2-channel RGB)
- **Channel 0**: Reflectance map
- **Channel 1**: Transmittance map
- **Value Range**: [0, 0.063] (normalized from 0-255)
- **Purpose**: Material properties affecting radio propagation

### Condition 2 (1-channel Grayscale)
- **Channel 0**: Distance/FSPL map
- **Value Range**: [0, 0.353] (normalized from 0-255)
- **Purpose**: Geometric distance information

### Target (1-channel Grayscale)
- **Channel 0**: Path loss map
- **Value Range**: [0.067, 0.486] (normalized from 0-255)
- **Purpose**: Ground truth radio signal strength

## Data Preprocessing

### Normalization
- All images are normalized to [0, 1] range during preprocessing
- For VAE training, images can be further normalized to [-1, 1]

### Resizing
- Original images are 320×320 pixels
- Configurable resizing support in the data loader

### Augmentation
- Currently disabled for reproducibility
- Can be enabled for training (horizontal flip, vertical flip, rotation)

## Configuration Files

### Dataset Configuration (`vae_config.yaml`)
```yaml
model:
  name: MultiConditionVAE
  embed_dim: 3
  condition_config:
    condition1_channels: 2  # reflectance + transmittance
    condition2_channels: 1  # distance map
    fusion_method: 'concatenation'

data:
  name: multi_condition_icassp2025
  batch_size: 16
  dataset_root: './icassp2025_multi_condition_vae'
  image_size: [320, 320]
```

### Training Configuration
- **Learning Rate**: 5e-6 to 5e-7 (cosine annealing)
- **Batch Size**: 16
- **Training Steps**: 150,000
- **Mixed Precision**: Disabled (for stability)
- **Gradient Accumulation**: Every 2 steps

## Multi-Condition VAE Architecture

### Supported Approaches

1. **Single VAE Approach**
   - Concatenates all conditions (3 channels total)
   - Uses single encoder-decoder architecture
   - Simpler implementation, faster training

2. **Dual VAE Approach**
   - Separate encoders for conditions and target
   - Supports cross-attention fusion
   - More flexible but complex

### Model Configuration
```yaml
architecture:
  vae_approach: 'single'  # or 'dual'
  single_vae:
    total_channels: 4
    use_bottleneck: True
  dual_vae:
    condition_encoder_channels: 3
    target_encoder_channels: 1
    shared_decoder: True
```

## Usage Examples

### Dataset Loading
```python
from denoising_diffusion_pytorch.multi_condition_data import MultiConditionICASSP2025Dataset

# Create dataset
dataset = MultiConditionICASSP2025Dataset(
    dataset_root='./icassp2025_multi_condition_vae',
    split='train',
    image_size=(320, 320),
    normalize=True
)

# Get sample
sample = dataset[0]
print(f'Condition1: {sample["condition1"].shape}')  # [3, 320, 320]
print(f'Condition2: {sample["condition2"].shape}')  # [1, 320, 320]
print(f'Target: {sample["target"].shape}')         # [1, 320, 320]
```

### Training Pipeline
```python
from denoising_diffusion_pytorch.multi_condition_data import MultiConditionDataLoader

# Initialize data loader
data_loader = MultiConditionDataLoader('./configs/icassp2025_multi_condition_vae.yaml')

# Get dataloaders
dataloaders = data_loader.get_dataloaders()

# Training loop
for batch in dataloaders['train']:
    condition1 = batch['condition1']  # [B, 3, 320, 320]
    condition2 = batch['condition2']  # [B, 1, 320, 320]
    target = batch['target']         # [B, 1, 320, 320]
    # Training logic here
```

### VAE Model
```python
from denoising_diffusion_pytorch.multi_condition_data import MultiConditionVAE

# Initialize VAE
vae = MultiConditionVAE('./configs/icassp2025_multi_condition_vae.yaml')

# Forward pass
result = vae(batch)
reconstruction = result['reconstruction']
```

## Dataset Generation

### Preparation Script
```bash
python icassp2025_multi_condition_dataset.py
```

### Custom Configuration
```bash
python icassp2025_multi_condition_dataset.py \
    --input_root /path/to/inputs \
    --output_root /path/to/outputs \
    --config custom_config.yaml
```

## Testing and Validation

### Dataset Validation
```bash
# Test dataset loader
python denoising_diffusion_pytorch/multi_condition_data.py

# Validate dataset structure
python -c "
from denoising_diffusion_pytorch.multi_condition_data import MultiConditionICASSP2025Dataset
dataset = MultiConditionICASSP2025Dataset('./icassp2025_multi_condition_vae')
print(f'Dataset size: {len(dataset)}')
print(f'Sample shape: {dataset[0][\"condition1\"].shape}')
"
```

### Expected Output
```
Testing MultiCondition ICASSP2025 Dataset...
Initialized MultiConditionDataLoader with single VAE approach
Loaded 1000 samples for train split
Created train dataloader with 1000 samples
Loaded 125 samples for val split
Created val dataloader with 125 samples
Loaded 125 samples for test split
Created test dataloader with 125 samples
Dataset test completed!
```

## Performance Considerations

### Memory Usage
- **Single Sample**: ~3MB (all conditions + target)
- **Batch (16)**: ~48MB
- **Full Dataset**: ~3.75GB

### Loading Speed
- **Dataset Initialization**: ~2 seconds
- **Batch Loading**: ~0.1 seconds
- **Full Epoch**: ~8 seconds (with batch size 16)

### Hardware Requirements
- **Minimum**: 8GB RAM, 4GB GPU
- **Recommended**: 16GB RAM, 8GB GPU
- **Optimal**: 32GB RAM, 16GB GPU

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce batch size
   - Enable gradient accumulation
   - Use mixed precision training

2. **Loading Errors**
   - Check dataset paths
   - Verify file permissions
   - Ensure all files exist

3. **Shape Mismatches**
   - Verify image sizes are consistent
   - Check channel configurations
   - Validate model input dimensions

### Debug Commands
```bash
# Check file counts
find icassp2025_multi_condition_vae -name "*.png" | wc -l

# Verify split distribution
for split in train val test; do
    echo "$split: $(find icassp2025_multi_condition_vae/$split/condition1 -name "*.png" | wc -l) files"
done

# Test sample loading
python -c "
dataset = MultiConditionICASSP2025Dataset('./icassp2025_multi_condition_vae')
sample = dataset[0]
print('Sample loaded successfully!')
"
```

## Future Enhancements

### Planned Features
1. **Additional Conditions**: More input modalities (building height, material types)
2. **Higher Resolution**: Support for 640×640 images
3. **Multi-frequency**: Support for multiple frequency bands
4. **Data Augmentation**: Advanced augmentation techniques
5. **Preprocessing**: Additional normalization and filtering options

### Extension Points
1. **Custom Fusion Methods**: Cross-attention, addition, multiplication
2. **Multi-scale Processing**: Hierarchical VAE architectures
3. **Spatio-temporal**: Time-series radio map prediction
4. **Multi-task**: Joint prediction of path loss and other metrics

## References

### Related Work
1. **RadioDiff**: Original RadioDiff paper and implementation
2. **ICASSP2025**: Conference paper introducing the dataset
3. **VAE Architectures**: Variational Autoencoders for radio map prediction

### Citation
```bibtex
@dataset{icassp2025_multi_condition,
  title={ICASSP2025 Multi-Condition Dataset for Radio Map Prediction},
  author={RadioDiff Team},
  year={2025},
  note={Dataset for multi-condition VAE training}
}
```

## Contact

For questions, issues, or contributions:
- **Repository**: [RadioDiff GitHub](https://github.com/your-org/RadioDiff)
- **Issues**: [GitHub Issues](https://github.com/your-org/RadioDiff/issues)
- **Documentation**: [Project Wiki](https://github.com/your-org/RadioDiff/wiki)

---

**Last Updated**: August 2025
**Version**: 1.0
**Maintainers**: RadioDiff Team