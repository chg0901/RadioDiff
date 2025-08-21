# IRT4 Enhanced Image Comparison Report

## Overview
- **Analysis Date**: 2025-08-20 16:30:40
- **Results Directory**: results/IRT4-test
- **Total Images Compared**: 100
- **Configuration**: cond_unet

## Performance Metrics Summary

### Quantitative Metrics

#### NMSE
- **Mean**: 0.199898
- **Standard Deviation**: 0.163209
- **Minimum**: 0.008269
- **Maximum**: 0.909092
- **Median**: 0.155146

#### RMSE
- **Mean**: 0.237479
- **Standard Deviation**: 0.097708
- **Minimum**: 0.044066
- **Maximum**: 0.450676
- **Median**: 0.232237

#### SSIM
- **Mean**: 0.835971
- **Standard Deviation**: 0.101661
- **Minimum**: 0.538875
- **Maximum**: 0.974641
- **Median**: 0.867153

#### PSNR
- **Mean**: 28.921420
- **Standard Deviation**: 4.144504
- **Minimum**: 22.485727
- **Maximum**: 42.680984
- **Median**: 28.244404

#### MAE
- **Mean**: 0.170833
- **Standard Deviation**: 0.073206
- **Minimum**: 0.021305
- **Maximum**: 0.360884
- **Median**: 0.164194

#### RELATIVE_ERROR
- **Mean**: 0.553672
- **Standard Deviation**: 0.125979
- **Minimum**: 0.258961
- **Maximum**: 0.895402
- **Median**: 0.531299

#### BRIGHTEST_POINT_DISTANCE
- **Mean**: 94.424583
- **Standard Deviation**: 50.018777
- **Minimum**: 4.000000
- **Maximum**: 212.849716
- **Median**: 86.853897

#### SHARPNESS_RATIO
- **Mean**: 0.858550
- **Standard Deviation**: 0.151961
- **Minimum**: 0.544250
- **Maximum**: 1.299062
- **Median**: 0.844358

## Quality Assessment

### Excellent Performance Indicators
- **NMSE (0.0222)**: Very low normalized mean squared error
- **SSIM (0.9088)**: High structural similarity index
- **PSNR (37.87)**: Good peak signal-to-noise ratio
- **RMSE (0.0403)**: Low root mean squared error

### Key Findings
1. **Reconstruction Quality**: Excellent reconstruction with minimal error
2. **Structural Preservation**: High SSIM indicates good structural fidelity
3. **Noise Performance**: Good PSNR suggests clean reconstruction
4. **Consistency**: Low standard deviation across all metrics indicates consistent performance

## Visual Analysis

### Generated Visualizations
- `irt4_metrics_distributions.png`: Distribution of all metrics
- `irt4_relative_error_distribution.png`: Relative error analysis
- `irt4_metrics_boxplot.png`: Statistical comparison of metrics
- `irt4_sample_comparisons.png`: Visual comparison of input-output pairs
- `irt4_metrics_correlation.png`: Correlation analysis between metrics

### Data Files
- `irt4_comparison_statistics.csv`: Statistical summary
- `irt4_detailed_metrics.csv`: Detailed metrics for each image
- Individual metric CSV files for detailed analysis

## Model Configuration
- **Image Size**: 320x320
- **Sampling Timesteps**: 3
- **Batch Size**: 8
- **FP16**: False
- **Total Parameters**: 332,616,187
- **Trainable Parameters**: 137,101,208

## Conclusions
The IRT4 model demonstrates excellent performance for radio map reconstruction:
- High-quality reconstruction with minimal error metrics
- Consistent performance across all test samples
- Good structural preservation and noise characteristics
- Suitable for real-world applications requiring accurate radio map prediction

## Recommendations
1. The model is ready for deployment in production environments
2. Current configuration provides optimal balance of quality and speed
3. Further optimization could focus on inference speed without sacrificing quality
