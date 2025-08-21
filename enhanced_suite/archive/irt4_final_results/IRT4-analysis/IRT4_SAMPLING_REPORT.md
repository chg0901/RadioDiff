
# IRT4 Sampling Analysis Report

## Overview
- **Analysis Date**: 2025-08-20 16:31:19
- **Results Directory**: /home/cine/Documents/Github/RadioDiff/results/IRT4-test
- **Total Images**: 801
- **Unique Samples**: 701

## Performance Metrics
Based on the inference log:
- **NMSE**: 0.022203
- **RMSE**: 0.040338
- **SSIM**: 0.908829
- **PSNR**: 37.868253
- **Average Peak Distance**: 1.326

## Image Statistics

### File Size Analysis
- **Mean File Size**: 17223.66 bytes
- **Std File Size**: 6313.04 bytes
- **Min File Size**: 6728.00 bytes
- **Max File Size**: 45387.00 bytes

### Intensity Analysis
- **Mean Intensity**: 70.0190 ± 19.7457
- **Intensity Range**: [0.0000, 255.0000]

### Variant Comparison
#### Input Images (variant=0)
- Count: 701
- Mean Intensity: 70.1884
- Std Intensity: 47.0036

#### Generated Images (variant=1)
- Count: 100
- Mean Intensity: 68.8317
- Std Intensity: 46.2850

## Model Configuration
- **Model**: IRT4 with conditional U-Net
- **Sampling Timesteps**: 3
- **Image Size**: 320x320
- **Batch Size**: 8
- **FP16**: False
- **Total Parameters**: 332,616,187
- **Trainable Parameters**: 137,101,208

## Sampling Performance
- **Average Sample Time**: 0.3979 seconds
- **Dataloader Length**: 200
- **Total Processing Time**: ~85 seconds

## Quality Assessment
- **NMSE (0.0222)**: Very low, indicating excellent reconstruction quality
- **SSIM (0.9088)**: High structural similarity, good preservation of image features
- **PSNR (37.87)**: Good signal-to-noise ratio
- **Peak Distance (1.33)**: Small average distance between predicted and actual peaks

## Conclusions
1. The IRT4 model demonstrates excellent reconstruction quality with low error metrics
2. High SSIM indicates good structural preservation
3. Fast sampling time (~0.4 seconds per sample) makes it suitable for real-time applications
4. Model maintains consistent performance across the test dataset
5. Generated images show good visual quality and structural fidelity

## Generated Files
- `irt4_sampling_analysis.png`: Comprehensive statistical analysis
- `irt4_sample_comparison.png`: Visual comparison of input-output pairs
- `irt4_sampling_stats.csv`: Detailed statistical data
