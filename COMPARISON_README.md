# RadioDiff Image Comparison Tool

This repository contains comprehensive tools for comparing generated radio map images with ground truth images from the RadioDiff project.

## Files

### `compare_images.py`
Main comparison script that calculates various metrics and generates visualizations.

**Features:**
- **Metrics Calculation**: NMSE, RMSE, SSIM, PSNR, MAE, relative error, brightest point distance
- **Statistical Analysis**: Mean, standard deviation, min, max, median for all metrics
- **Visualizations**: Histograms, KDE plots, box plots, sample comparisons
- **Comprehensive Reports**: CSV exports with detailed metrics

### `run_comparison_example.py`
Example script that demonstrates how to use the comparison tool with existing RadioDiff outputs.

## Usage

### Basic Usage

```bash
python compare_images.py \
    --config configs/radio_sample_m.yaml \
    --gt_dir /path/to/ground/truth \
    --gen_dir /path/to/generated/images \
    --output_dir ./enhanced_suite/archive/comparison_results
```

### Using the Example Script

```bash
python run_comparison_example.py
```

This will automatically:
1. Find existing generated images in your RadioDiff directories
2. Locate corresponding ground truth images
3. Run the full comparison analysis
4. Generate comprehensive reports and visualizations

## Configuration

The comparison tool uses the same YAML configuration files as the RadioDiff training scripts. Key parameters:

- `model.image_size`: Image dimensions for comparison
- `model.first_stage`: VAE model configuration
- `model.unet`: UNet model parameters

## Metrics Explained

### Primary Metrics
- **NMSE** (Normalized Mean Squared Error): MSE normalized by target variance
- **RMSE** (Root Mean Squared Error): Square root of average squared differences
- **SSIM** (Structural Similarity Index): Perceptual similarity measure
- **PSNR** (Peak Signal-to-Noise Ratio): Quality measure in decibels
- **MAE** (Mean Absolute Error): Average absolute difference

### Radio-Specific Metrics
- **Relative Error**: Sqrt-normalized relative difference
- **Brightest Point Distance**: Euclidean distance between maximum intensity points
- **Sharpness Ratio**: Comparison of image sharpness using Laplacian variance

## Output Files

### CSV Reports
- `comparison_statistics.csv`: Statistical summary of all metrics
- `detailed_metrics.csv`: Individual metrics for each image pair

### Visualizations
- `metrics_distributions.png`: Histograms of all metrics with mean indicators
- `relative_error_distribution.png`: KDE plot of relative errors
- `metrics_boxplot.png`: Box plot comparison of metric distributions
- `sample_comparisons.png`: Side-by-side visual comparison of sample images

## Example Output

```
RADIO DIFF IMAGE COMPARISON SUMMARY
============================================================
Total images compared: 50
Image size: [320, 320]
------------------------------------------------------------
NMSE                      : Mean=0.023456, Std=0.012345, Min=0.008765, Max=0.067890
RMSE                      : Mean=0.123456, Std=0.034567, Min=0.045678, Max=0.234567
SSIM                      : Mean=0.876543, Std=0.045678, Min=0.765432, Max=0.987654
PSNR                      : Mean=23.456789, Std=2.345678, Min=18.765432, Max=28.901234
MAE                       : Mean=0.098765, Std=0.023456, Min=0.045678, Max=0.167890
============================================================
Results saved to: ./enhanced_suite/archive/comparison_results
============================================================
```

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- Pandas
- scikit-image
- torchmetrics
- torchvision
- PyYAML

## Integration with RadioDiff

The comparison tool integrates seamlessly with existing RadioDiff workflows:

1. **After Training**: Use to evaluate model performance on test sets
2. **During Development**: Compare different model architectures
3. **Quality Assessment**: Monitor generation quality over time
4. **Ablation Studies**: Quantify impact of different components

## Tips

- Ensure ground truth and generated images have consistent naming conventions
- Use the same image size as specified in your training configuration
- The tool automatically handles image preprocessing and normalization
- For large datasets, the progress bar shows comparison progress

## Troubleshooting

### Common Issues

1. **Missing Images**: Ensure `gt-sample-*.png` and `sample-*.png` files exist
2. **Size Mismatch**: Images are automatically resized to match configuration
3. **Memory Issues**: Process smaller batches if needed
4. **CUDA Errors**: Tool automatically falls back to CPU if GPU unavailable

### Error Messages

- "No generated images found": Check your directory paths
- "Image loading failed": Verify image file formats
- "Configuration error": Validate YAML syntax