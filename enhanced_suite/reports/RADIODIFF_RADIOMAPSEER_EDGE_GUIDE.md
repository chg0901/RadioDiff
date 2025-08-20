# RadioMapSeer Edge Detection Implementation Guide

This guide explains how to use the RadioMapSeer edge detection implementation that works without requiring a specific edge dataset. The implementation uses the existing DPM (Gain) data from the RadioMapSeer dataset to generate edge maps for training and inference.

## Overview

The RadioMapSeer edge detection implementation provides:

1. **Automatic Edge Dataset Generation**: Creates edge datasets from existing DPM images
2. **Multiple Edge Detection Methods**: Canny, Sobel, Laplacian, and Prewitt operators
3. **Synthetic Edge Generation**: Advanced methods for creating synthetic edge maps
4. **Complete Training Pipeline**: VAE → LDM → Inference workflow
5. **Flexible Configuration**: Configurable parameters for different use cases

## Key Files

- `radiomapseer_edge_detection.py`: Main edge detection implementation
- `test_radiomapseer_edges.py`: Testing and visualization script
- `configs_edge/radiomapseer_edge_config_template.yaml`: Configuration template

## Quick Start

### Step 1: Test the Implementation

```bash
# Test all edge detection methods on sample data
python test_radiomapseer_edges.py \
    --data_root /home/cine/Documents/dataset/RadioMapSeer \
    --output_dir ./enhanced_suite/archive/radiomapseer_edge_test_results \
    --test_type all \
    --num_samples 5
```

### Step 2: Generate Edge Dataset

```bash
# Generate edge dataset using Canny edge detection
python radiomapseer_edge_detection.py \
    --data_root /home/cine/Documents/dataset/RadioMapSeer \
    --output_dir ./enhanced_suite/archive/radiomapseer_edge_dataset \
    --method canny \
    --image_size 256 256 \
    --split_ratio 0.8
```

### Step 3: Set Up Configuration

Copy the configuration template and update paths:

```bash
# Copy configuration files
cp configs_edge/radiomapseer_edge_config_template.yaml configs_edge/radiomapseer_edge_vae_train.yaml
cp configs_edge/radiomapseer_edge_config_template.yaml configs_edge/radiomapseer_edge_ldm_train.yaml
cp configs_edge/radiomapseer_edge_config_template.yaml configs_edge/radiomapseer_edge_sample.yaml
```

Update the `data_root` path in each configuration file to point to your generated dataset.

### Step 4: Train Models

```bash
# Train VAE
python train_vae.py --cfg configs_edge/radiomapseer_edge_vae_train.yaml

# Train LDM (update VAE checkpoint path first)
python train_cond_ldm_m.py --cfg configs_edge/radiomapseer_edge_ldm_train.yaml
```

### Step 5: Run Inference

```bash
# Run inference
python demo.py \
    --cfg configs_edge/radiomapseer_edge_sample.yaml \
    --input_dir ./test_images \
    --out_dir ./results \
    --pre_weight ./radiomapseer_edge_ldm_results/model-50.pt
```

## Edge Detection Methods

### 1. Canny Edge Detection
- **Description**: Multi-stage algorithm with optimal edge detection
- **Best for**: Clear, well-defined edges
- **Parameters**: `threshold1`, `threshold2`, `aperture_size`

```bash
python radiomapseer_edge_detection.py \
    --method canny \
    --threshold1 50 \
    --threshold2 150
```

### 2. Sobel Edge Detection
- **Description**: Gradient-based edge detection using Sobel operators
- **Best for**: General-purpose edge detection
- **Parameters**: `ksize`, `scale`, `delta`

```bash
python radiomapseer_edge_detection.py \
    --method sobel \
    --ksize 3
```

### 3. Laplacian Edge Detection
- **Description**: Second-order derivative edge detection
- **Best for**: Fine details and rapid intensity changes
- **Parameters**: `ksize`, `scale`, `delta`

```bash
python radiomapseer_edge_detection.py \
    --method laplacian \
    --ksize 3
```

### 4. Prewitt Edge Detection
- **Description**: Gradient-based edge detection using Prewitt operators
- **Best for**: Edge detection in noisy images
- **Parameters**: `ksize`

```bash
python radiomapseer_edge_detection.py \
    --method prewitt \
    --ksize 3
```

## Synthetic Edge Generation

For advanced use cases, you can generate synthetic edge maps:

### 1. Gradient-Based Synthetic Edges
```bash
python radiomapseer_edge_detection.py \
    --synthetic \
    --synthetic_type gradient
```

### 2. Contour-Based Synthetic Edges
```bash
python radiomapseer_edge_detection.py \
    --synthetic \
    --synthetic_type contour
```

### 3. Ridge-Based Synthetic Edges
```bash
python radiomapseer_edge_detection.py \
    --synthetic \
    --synthetic_type ridge
```

## Configuration Parameters

### Image Parameters
- `image_size`: Target image dimensions (default: [256, 256])
- `split_ratio`: Train/validation split ratio (default: 0.8)

### Edge Detection Parameters
- `method`: Edge detection method
- `threshold1`: Lower threshold for Canny
- `threshold2`: Upper threshold for Canny
- `ksize`: Kernel size for Sobel/Laplacian/Prewitt

### Training Parameters
- `batch_size`: Batch size for training
- `lr`: Learning rate
- `train_num_steps`: Number of training steps

## Dataset Structure

The generated edge dataset follows this structure:

```
enhanced_suite/archive/radiomapseer_edge_dataset/
├── image/
│   └── raw/
│       ├── train/
│       │   ├── train_0.png
│       │   └── train_1.png
│       └── val/
│           ├── val_0.png
│           └── val_1.png
└── edge/
    └── raw/
        ├── train/
        │   ├── train_0.png
        │   └── train_1.png
        └── val/
            ├── val_0.png
            └── val_1.png
```

## Testing and Visualization

### Test Individual Methods
```bash
# Test only Canny edge detection
python test_radiomapseer_edges.py \
    --data_root /home/cine/Documents/dataset/RadioMapSeer \
    --output_dir ./canny_test_results \
    --test_type edge_detection \
    --num_samples 10
```

### Test Synthetic Edges
```bash
# Test synthetic edge generation
python test_radiomapseer_edges.py \
    --data_root /home/cine/Documents/dataset/RadioMapSeer \
    --output_dir ./synthetic_test_results \
    --test_type synthetic \
    --num_samples 5
```

### Create Sample Dataset
```bash
# Create small sample dataset
python test_radiomapseer_edges.py \
    --data_root /home/cine/Documents/dataset/RadioMapSeer \
    --output_dir ./sample_dataset \
    --test_type dataset \
    --num_samples 20
```

## Performance Optimization

### Memory Optimization
- Reduce `batch_size` if encountering OOM errors
- Use smaller `image_size` for faster processing
- Enable mixed precision training with `amp: True`

### Quality Optimization
- Experiment with different edge detection methods
- Adjust edge detection thresholds for your specific data
- Use synthetic edges for better edge continuity

### Speed Optimization
- Increase `num_workers` for faster data loading
- Use smaller training datasets for quick testing
- Reduce `train_num_steps` for faster experiments

## Troubleshooting

### Common Issues

**Dataset Not Found**
```bash
# Check if dataset path is correct
ls /home/cine/Documents/dataset/RadioMapSeer/gain/DPM/
```

**Memory Issues**
```bash
# Reduce batch size and image size
python radiomapseer_edge_detection.py \
    --image_size 128 128 \
    --method canny
```

**Poor Edge Quality**
```bash
# Try different methods or parameters
python radiomapseer_edge_detection.py \
    --method sobel \
    --threshold1 30 \
    --threshold2 100
```

### Debug Commands

```bash
# Test dataset loading
python -c "
from radiomapseer_edge_detection import RadioMapSeerEdgeDataset
dataset = RadioMapSeerEdgeDataset('/home/cine/Documents/dataset/RadioMapSeer')
print(f'Dataset size: {len(dataset.dpm_files)}')
img, edge = dataset.get_sample_pair(0)
print(f'Image shape: {img.shape}, Edge shape: {edge.shape}')
"

# Test edge detection
python -c "
from radiomapseer_edge_detection import EdgeDetector
import cv2
detector = EdgeDetector('canny')
img = cv2.imread('/path/to/image.png', cv2.IMREAD_GRAYSCALE)
edges = detector.detect_edges(img)
print(f'Edges generated: {edges.shape}')
"
```

## Results and Evaluation

### Expected Results
- **VAE Training**: Should converge within 50k-100k steps
- **LDM Training**: Should show stable loss curves
- **Inference**: Should produce clean edge maps

### Quality Metrics
- Visual inspection of edge maps
- Comparison with different edge detection methods
- Processing speed benchmarks

## Advanced Usage

### Custom Edge Detection
```python
from radiomapseer_edge_detection import EdgeDetector

# Create custom edge detector
detector = EdgeDetector('canny', threshold1=30, threshold2=100)
edges = detector.detect_edges(your_image)
```

### Batch Processing
```python
from radiomapseer_edge_detection import RadioMapSeerEdgeDataset

# Process entire dataset
dataset = RadioMapSeerEdgeDataset(data_root, method='sobel')
dataset.create_edge_dataset(output_dir, split_ratio=0.8)
```

### Custom Synthetic Edges
```python
from radiomapseer_edge_detection import SyntheticEdgeGenerator

# Generate synthetic edges
generator = SyntheticEdgeGenerator((256, 256))
edges = generator.generate_synthetic_edges(dpm_image, 'gradient')
```

## Conclusion

The RadioMapSeer edge detection implementation provides a complete pipeline for edge detection without requiring a specific edge dataset. By leveraging the existing DPM data, you can create high-quality edge maps for training and inference.

For best results, experiment with different edge detection methods and parameters to find the optimal configuration for your specific use case.