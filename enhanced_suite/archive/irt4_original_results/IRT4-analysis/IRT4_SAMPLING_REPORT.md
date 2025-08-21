
# IRT4 Sampling Analysis Report

## Overview
- **Analysis Date**: 2025-08-20 15:35:24
- **Results Directory**: /home/cine/Documents/Github/RadioDiff/results/IRT4-test
- **Total Images**: 200
- **Unique Samples**: 100

## Performance Metrics
Based on the inference log:
- **NMSE**: 0.022203
- **RMSE**: 0.040338
- **SSIM**: 0.908829
- **PSNR**: 37.868253
- **Average Peak Distance**: 1.326

## Image Statistics

### File Size Analysis
- **Mean File Size**: 30867.67 bytes
- **Std File Size**: 6342.44 bytes
- **Min File Size**: 15042.00 bytes
- **Max File Size**: 46815.00 bytes

### Intensity Analysis
- **Mean Intensity**: 68.8176 ± 19.4690
- **Intensity Range**: [0.0000, 255.0000]

### Variant Comparison
#### Input Images (variant=0)
- Count: 100
- Mean Intensity: 68.8035
- Std Intensity: 45.9144

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

## Visual Analysis

### Statistical Analysis Overview
![IRT4 Sampling Statistical Analysis](irt4_sampling_analysis.png)

*Figure 1: Comprehensive statistical analysis of IRT4 sampling results showing file size distribution, intensity statistics, and variant comparisons.*

**Key Observations:**
- **File Size Distribution**: Most generated images have file sizes between 25-35KB, indicating consistent compression and content complexity
- **Intensity Distribution**: The mean intensity shows a normal distribution around 68.8, suggesting good dynamic range utilization
- **Variant Comparison**: Input and generated images show very similar statistical properties, indicating faithful reconstruction
- **Intensity Range**: Full 0-255 range utilization demonstrates good contrast preservation

### Sample Comparison Grid
![IRT4 Sample Comparison](irt4_sample_comparison.png)

*Figure 2: Visual comparison of input-output pairs showing the first 20 samples. Each row shows an input image (left) and its corresponding generated output (right).*

**Quality Assessment:**
- **Structural Fidelity**: Generated images maintain the structural patterns and features of input images
- **Edge Preservation**: Sharp edges and boundaries are well-preserved in the reconstruction
- **Texture Consistency**: Texture patterns and gradients are accurately reproduced
- **Noise Level**: Generated images show appropriate noise levels without excessive smoothing or artifacts

### Performance Metrics Visualization
The quantitative metrics provide strong evidence of the model's reconstruction quality:
- **NMSE (0.0222)**: Very low normalized mean squared error indicates minimal reconstruction deviation
- **SSIM (0.9088)**: High structural similarity index confirms excellent structural preservation
- **PSNR (37.87)**: Good peak signal-to-noise ratio demonstrates clean reconstruction
- **Peak Distance (1.33)**: Small average peak distance confirms accurate feature localization

## Generated Files
- `irt4_sampling_analysis.py`: Analysis script
- `irt4_sampling_analysis.png`: Comprehensive statistical analysis
- `irt4_sample_comparison.png`: Visual comparison of input-output pairs
- `irt4_sampling_stats.csv`: Detailed statistical data
- `IRT4_SAMPLING_REPORT.md`: Comprehensive report with visualizations
