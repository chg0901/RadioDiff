#!/usr/bin/env python3
"""
ICASSP2025 Dataset Validation Script

This script validates the arranged ICASSP2025 dataset by:
- Checking data integrity and file formats
- Verifying image dimensions and consistency
- Validating train/val split
- Generating dataset statistics
- Creating quality reports

Author: Claude Code Assistant
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm.auto import tqdm
from collections import defaultdict, Counter
import cv2

class ICASSPDatasetValidator:
    """Dataset validator for ICASSP2025 dataset"""
    
    def __init__(self, dataset_root):
        self.dataset_root = Path(dataset_root)
        self.validation_results = {}
        
    def validate_file_structure(self):
        """Validate the file structure"""
        print("Validating file structure...")
        
        required_dirs = [
            'train/image',
            'train/edge',
            'val/image',
            'val/edge'
        ]
        
        structure_valid = True
        missing_dirs = []
        
        for dir_path in required_dirs:
            full_path = self.dataset_root / dir_path
            if not full_path.exists():
                structure_valid = False
                missing_dirs.append(dir_path)
        
        self.validation_results['file_structure'] = {
            'valid': structure_valid,
            'missing_dirs': missing_dirs,
            'required_dirs': required_dirs
        }
        
        return structure_valid
    
    def validate_image_files(self):
        """Validate all image files"""
        print("Validating image files...")
        
        image_stats = {
            'train': {'image': {}, 'edge': {}},
            'val': {'image': {}, 'edge': {}}
        }
        
        corrupt_files = []
        format_issues = []
        
        for split in ['train', 'val']:
            for data_type in ['image', 'edge']:
                data_dir = self.dataset_root / split / data_type
                files = list(data_dir.glob('*.png'))
                
                if not files:
                    print(f"Warning: No files found in {data_dir}")
                    continue
                
                image_stats[split][data_type]['total_files'] = len(files)
                image_stats[split][data_type]['valid_files'] = 0
                image_stats[split][data_type]['corrupt_files'] = 0
                
                dimensions = []
                file_sizes = []
                
                for file_path in tqdm(files, desc=f"Checking {split}/{data_type}"):
                    try:
                        # Load image
                        img = Image.open(file_path)
                        img_array = np.array(img)
                        
                        # Check dimensions
                        if data_type == 'image':
                            # Should be 3-channel
                            if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                                format_issues.append(f"{file_path}: Expected 3-channel, got {img_array.shape}")
                                continue
                        else:
                            # Should be 1-channel
                            if len(img_array.shape) != 2:
                                format_issues.append(f"{file_path}: Expected 1-channel, got {img_array.shape}")
                                continue
                        
                        dimensions.append(img_array.shape[:2])
                        file_sizes.append(file_path.stat().st_size)
                        
                        image_stats[split][data_type]['valid_files'] += 1
                        
                    except Exception as e:
                        corrupt_files.append(f"{file_path}: {str(e)}")
                        image_stats[split][data_type]['corrupt_files'] += 1
                
                # Calculate statistics
                if dimensions:
                    dims_array = np.array(dimensions)
                    image_stats[split][data_type]['dimensions'] = {
                        'unique_shapes': list(set(dimensions)),
                        'most_common': Counter(dimensions).most_common(1)[0][0],
                        'consistency': len(set(dimensions)) == 1
                    }
                    
                    image_stats[split][data_type]['file_sizes'] = {
                        'min': min(file_sizes),
                        'max': max(file_sizes),
                        'mean': np.mean(file_sizes),
                        'std': np.std(file_sizes)
                    }
        
        self.validation_results['image_files'] = {
            'stats': image_stats,
            'corrupt_files': corrupt_files,
            'format_issues': format_issues
        }
        
        return len(corrupt_files) == 0 and len(format_issues) == 0
    
    def validate_pairing_consistency(self):
        """Validate that image and edge files are properly paired"""
        print("Validating pairing consistency...")
        
        pairing_issues = []
        missing_pairs = []
        
        for split in ['train', 'val']:
            image_dir = self.dataset_root / split / 'image'
            edge_dir = self.dataset_root / split / 'edge'
            
            image_files = {f.stem: f for f in image_dir.glob('*.png')}
            edge_files = {f.stem: f for f in edge_dir.glob('*.png')}
            
            # Check for missing pairs
            for name in image_files:
                if name not in edge_files:
                    missing_pairs.append(f"{split}: Missing edge file for {name}")
            
            for name in edge_files:
                if name not in image_files:
                    missing_pairs.append(f"{split}: Missing image file for {name}")
            
            # Check dimension consistency for paired files
            common_names = set(image_files.keys()) & set(edge_files.keys())
            for name in common_names:
                try:
                    img_img = Image.open(image_files[name])
                    edge_img = Image.open(edge_files[name])
                    
                    if img_img.size != edge_img.size:
                        pairing_issues.append(f"{split}: Size mismatch for {name}: "
                                           f"image={img_img.size}, edge={edge_img.size}")
                
                except Exception as e:
                    pairing_issues.append(f"{split}: Error loading {name}: {str(e)}")
        
        self.validation_results['pairing'] = {
            'issues': pairing_issues,
            'missing_pairs': missing_pairs,
            'valid': len(pairing_issues) == 0 and len(missing_pairs) == 0
        }
        
        return len(pairing_issues) == 0 and len(missing_pairs) == 0
    
    def analyze_dataset_distribution(self):
        """Analyze dataset distribution and extract statistics"""
        print("Analyzing dataset distribution...")
        
        # Parse filenames to extract metadata
        metadata = []
        
        for split in ['train', 'val']:
            image_dir = self.dataset_root / split / 'image'
            
            for file_path in image_dir.glob('*.png'):
                try:
                    # Parse filename: B{building_id}_Ant{antenna_id}_f{freq_id}_S{sample_id}
                    parts = file_path.stem.split('_')
                    building_id = int(parts[0][1:])
                    antenna_id = int(parts[1][3:])
                    freq_id = int(parts[2][1:])
                    sample_id = int(parts[3][1:])
                    
                    metadata.append({
                        'split': split,
                        'building_id': building_id,
                        'antenna_id': antenna_id,
                        'freq_id': freq_id,
                        'sample_id': sample_id,
                        'filename': file_path.name
                    })
                
                except Exception as e:
                    print(f"Error parsing filename {file_path.name}: {e}")
        
        if not metadata:
            print("No valid metadata found!")
            return
        
        df = pd.DataFrame(metadata)
        
        # Calculate statistics
        distribution_stats = {
            'total_samples': len(metadata),
            'train_samples': len(df[df['split'] == 'train']),
            'val_samples': len(df[df['split'] == 'val']),
            'unique_buildings': df['building_id'].nunique(),
            'unique_antennas': df['antenna_id'].nunique(),
            'unique_frequencies': df['freq_id'].nunique(),
            'building_distribution': df['building_id'].value_counts().to_dict(),
            'antenna_distribution': df['antenna_id'].value_counts().to_dict(),
            'frequency_distribution': df['freq_id'].value_counts().to_dict(),
            'samples_per_building': df.groupby('building_id').size().describe().to_dict(),
            'split_ratio': len(df[df['split'] == 'train']) / len(df)
        }
        
        self.validation_results['distribution'] = distribution_stats
        
        # Create visualizations
        self.create_distribution_plots(df)
    
    def create_distribution_plots(self, df):
        """Create distribution visualization plots"""
        print("Creating distribution plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Building distribution
        building_counts = df['building_id'].value_counts().sort_index()
        axes[0, 0].bar(building_counts.index, building_counts.values)
        axes[0, 0].set_title('Building Distribution')
        axes[0, 0].set_xlabel('Building ID')
        axes[0, 0].set_ylabel('Count')
        
        # Antenna distribution
        antenna_counts = df['antenna_id'].value_counts().sort_index()
        axes[0, 1].bar(antenna_counts.index, antenna_counts.values)
        axes[0, 1].set_title('Antenna Distribution')
        axes[0, 1].set_xlabel('Antenna ID')
        axes[0, 1].set_ylabel('Count')
        
        # Frequency distribution
        freq_counts = df['freq_id'].value_counts().sort_index()
        axes[1, 0].bar(freq_counts.index, freq_counts.values)
        axes[1, 0].set_title('Frequency Distribution')
        axes[1, 0].set_xlabel('Frequency ID')
        axes[1, 0].set_ylabel('Count')
        
        # Train/Val split
        split_counts = df['split'].value_counts()
        axes[1, 1].pie(split_counts.values, labels=split_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Train/Validation Split')
        
        plt.tight_layout()
        plt.savefig(self.dataset_root / 'dataset_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def validate_image_quality(self):
        """Validate image quality and statistics"""
        print("Validating image quality...")
        
        quality_stats = {
            'train': {'image': {}, 'edge': {}},
            'val': {'image': {}, 'edge': {}}
        }
        
        for split in ['train', 'val']:
            for data_type in ['image', 'edge']:
                data_dir = self.dataset_root / split / data_type
                
                # Sample first 100 images for quality analysis
                files = list(data_dir.glob('*.png'))[:100]
                
                if not files:
                    continue
                
                pixel_values = []
                contrast_values = []
                brightness_values = []
                
                for file_path in tqdm(files, desc=f"Quality check {split}/{data_type}"):
                    try:
                        img = Image.open(file_path)
                        img_array = np.array(img)
                        
                        if data_type == 'image':
                            # For 3-channel images, analyze each channel
                            for channel in range(3):
                                channel_data = img_array[:, :, channel].flatten()
                                pixel_values.extend(channel_data)
                        else:
                            # For 1-channel images
                            pixel_values.extend(img_array.flatten())
                        
                        # Calculate contrast (standard deviation)
                        contrast = np.std(img_array)
                        contrast_values.append(contrast)
                        
                        # Calculate brightness (mean)
                        brightness = np.mean(img_array)
                        brightness_values.append(brightness)
                        
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                
                if pixel_values:
                    quality_stats[split][data_type] = {
                        'pixel_stats': {
                            'min': np.min(pixel_values),
                            'max': np.max(pixel_values),
                            'mean': np.mean(pixel_values),
                            'std': np.std(pixel_values),
                            'median': np.median(pixel_values)
                        },
                        'contrast_stats': {
                            'min': np.min(contrast_values),
                            'max': np.max(contrast_values),
                            'mean': np.mean(contrast_values),
                            'std': np.std(contrast_values)
                        },
                        'brightness_stats': {
                            'min': np.min(brightness_values),
                            'max': np.max(brightness_values),
                            'mean': np.mean(brightness_values),
                            'std': np.std(brightness_values)
                        }
                    }
        
        self.validation_results['quality'] = quality_stats
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print("Generating validation report...")
        
        report = {
            'dataset_path': str(self.dataset_root),
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'validation_results': self.validation_results,
            'summary': {
                'overall_valid': all([
                    self.validation_results.get('file_structure', {}).get('valid', False),
                    self.validation_results.get('image_files', {}).get('valid', True),
                    self.validation_results.get('pairing', {}).get('valid', True)
                ]),
                'total_issues': sum([
                    len(self.validation_results.get('file_structure', {}).get('missing_dirs', [])),
                    len(self.validation_results.get('image_files', {}).get('corrupt_files', [])),
                    len(self.validation_results.get('image_files', {}).get('format_issues', [])),
                    len(self.validation_results.get('pairing', {}).get('issues', [])),
                    len(self.validation_results.get('pairing', {}).get('missing_pairs', []))
                ])
            }
        }
        
        # Save report
        report_path = self.dataset_root / 'validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create text summary
        summary_text = self.create_text_summary(report)
        summary_path = self.dataset_root / 'validation_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        
        print(f"Validation report saved to: {report_path}")
        print(f"Summary saved to: {summary_path}")
        
        return report
    
    def create_text_summary(self, report):
        """Create human-readable text summary"""
        summary = []
        summary.append("ICASSP2025 Dataset Validation Report")
        summary.append("=" * 50)
        summary.append(f"Dataset Path: {report['dataset_path']}")
        summary.append(f"Validation Time: {report['validation_timestamp']}")
        summary.append("")
        
        # Overall status
        overall_valid = report['summary']['overall_valid']
        total_issues = report['summary']['total_issues']
        summary.append(f"Overall Status: {'VALID' if overall_valid else 'ISSUES FOUND'}")
        summary.append(f"Total Issues: {total_issues}")
        summary.append("")
        
        # File structure
        fs_result = report['validation_results']['file_structure']
        summary.append("File Structure:")
        summary.append(f"  Status: {'VALID' if fs_result['valid'] else 'INVALID'}")
        if fs_result['missing_dirs']:
            summary.append(f"  Missing Directories: {', '.join(fs_result['missing_dirs'])}")
        summary.append("")
        
        # Image files
        img_result = report['validation_results']['image_files']
        summary.append("Image Files:")
        summary.append(f"  Status: {'VALID' if len(img_result['corrupt_files']) == 0 and len(img_result['format_issues']) == 0 else 'ISSUES FOUND'}")
        if img_result['corrupt_files']:
            summary.append(f"  Corrupt Files: {len(img_result['corrupt_files'])}")
        if img_result['format_issues']:
            summary.append(f"  Format Issues: {len(img_result['format_issues'])}")
        
        # Sample counts
        stats = img_result['stats']
        for split in ['train', 'val']:
            for data_type in ['image', 'edge']:
                if split in stats and data_type in stats[split]:
                    total = stats[split][data_type].get('total_files', 0)
                    valid = stats[split][data_type].get('valid_files', 0)
                    summary.append(f"  {split}/{data_type}: {valid}/{total} valid files")
        summary.append("")
        
        # Distribution
        if 'distribution' in report['validation_results']:
            dist = report['validation_results']['distribution']
            summary.append("Dataset Distribution:")
            summary.append(f"  Total Samples: {dist['total_samples']}")
            summary.append(f"  Training Samples: {dist['train_samples']}")
            summary.append(f"  Validation Samples: {dist['val_samples']}")
            summary.append(f"  Unique Buildings: {dist['unique_buildings']}")
            summary.append(f"  Unique Antennas: {dist['unique_antennas']}")
            summary.append(f"  Unique Frequencies: {dist['unique_frequencies']}")
            summary.append(f"  Train/Val Ratio: {dist['split_ratio']:.2%}")
            summary.append("")
        
        # Pairing
        pairing_result = report['validation_results']['pairing']
        summary.append("File Pairing:")
        summary.append(f"  Status: {'VALID' if pairing_result['valid'] else 'ISSUES FOUND'}")
        if pairing_result['issues']:
            summary.append(f"  Pairing Issues: {len(pairing_result['issues'])}")
        if pairing_result['missing_pairs']:
            summary.append(f"  Missing Pairs: {len(pairing_result['missing_pairs'])}")
        summary.append("")
        
        # Recommendations
        summary.append("Recommendations:")
        if total_issues > 0:
            summary.append("  - Fix identified issues before training")
        else:
            summary.append("  - Dataset is ready for training")
        summary.append("  - Monitor training stability")
        summary.append("  - Consider data augmentation")
        
        return "\n".join(summary)
    
    def run_full_validation(self):
        """Run complete validation process"""
        print("Starting full validation of ICASSP2025 dataset...")
        print(f"Dataset path: {self.dataset_root}")
        print("")
        
        # Run all validation steps
        steps = [
            ("File Structure", self.validate_file_structure),
            ("Image Files", self.validate_image_files),
            ("Pairing Consistency", self.validate_pairing_consistency),
            ("Dataset Distribution", self.analyze_dataset_distribution),
            ("Image Quality", self.validate_image_quality)
        ]
        
        for step_name, step_func in steps:
            print(f"Running {step_name} validation...")
            try:
                result = step_func()
                print(f"  {step_name}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                print(f"  {step_name}: ERROR - {str(e)}")
            print("")
        
        # Generate final report
        report = self.generate_validation_report()
        
        print("Validation complete!")
        print(f"Overall Status: {'VALID' if report['summary']['overall_valid'] else 'ISSUES FOUND'}")
        print(f"Total Issues: {report['summary']['total_issues']}")
        
        return report

def main():
    """Main function"""
    # Configuration
    dataset_root = './icassp2025_dataset_arranged'
    
    # Create validator
    validator = ICASSPDatasetValidator(dataset_root)
    
    # Run full validation
    report = validator.run_full_validation()
    
    # Print summary
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    
    if report['summary']['overall_valid']:
        print("✅ Dataset is valid and ready for training!")
    else:
        print("❌ Dataset has issues that need to be fixed")
        print(f"   Total issues: {report['summary']['total_issues']}")
    
    print(f"\nDetailed reports saved to: {dataset_root}")
    print("  - validation_report.json (detailed JSON)")
    print("  - validation_summary.txt (human-readable)")
    print("  - dataset_distribution.png (visualization)")

if __name__ == "__main__":
    main()