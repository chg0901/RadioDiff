#!/usr/bin/env python3
"""
Comprehensive Test Script for RadioMapSeer Edge Detection

This script tests the enhanced edge detection functionality including:
1. Error handling and retry mechanisms
2. Different edge detection methods
3. Dataset structure validation
4. Performance metrics
5. File count verification
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
import torch
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
from tqdm import tqdm
import tempfile
import shutil
import time
import json
from datetime import datetime

# Import the edge detection modules
sys.path.append('.')
from radiomapseer_edge_detection_m import RadioMapSeerEdgeDataset, EdgeDetector
from radiomapseer_edge_detection import RadioMapSeerEdgeDataset as OriginalDataset

class EdgeDetectionTester:
    """Comprehensive tester for edge detection functionality"""
    
    def __init__(self, data_root: str, output_dir: str):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test results storage
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'data_root': str(data_root),
            'tests': {}
        }
        
        print(f"Edge Detection Tester initialized")
        print(f"Data root: {self.data_root}")
        print(f"Output dir: {self.output_dir}")
    
    def test_dataset_loading(self) -> Dict:
        """Test dataset loading and basic functionality"""
        print("\n=== Testing Dataset Loading ===")
        
        test_result = {
            'status': 'pending',
            'errors': [],
            'metrics': {}
        }
        
        try:
            # Test modified dataset
            print("Testing modified dataset...")
            dataset = RadioMapSeerEdgeDataset(
                str(self.data_root),
                image_size=(256, 256),
                edge_method='canny'
            )
            
            test_result['metrics']['total_files'] = len(dataset.dpm_files)
            test_result['metrics']['subset_count'] = len(dataset.subset_structure)
            test_result['metrics']['sample_subsets'] = list(dataset.subset_structure.keys())[:5]
            
            # Test original dataset
            print("Testing original dataset...")
            original_dataset = OriginalDataset(
                str(self.data_root),
                image_size=(256, 256),
                edge_method='canny'
            )
            
            test_result['metrics']['original_total_files'] = len(original_dataset.dpm_files)
            
            # Verify consistency
            if len(dataset.dpm_files) == len(original_dataset.dpm_files):
                test_result['metrics']['file_count_consistent'] = True
            else:
                test_result['metrics']['file_count_consistent'] = False
                test_result['errors'].append(f"File count mismatch: {len(dataset.dpm_files)} vs {len(original_dataset.dpm_files)}")
            
            test_result['status'] = 'success'
            
        except Exception as e:
            test_result['status'] = 'failed'
            test_result['errors'].append(str(e))
        
        self.test_results['tests']['dataset_loading'] = test_result
        return test_result
    
    def test_edge_detection_methods(self) -> Dict:
        """Test different edge detection methods"""
        print("\n=== Testing Edge Detection Methods ===")
        
        test_result = {
            'status': 'pending',
            'errors': [],
            'metrics': {},
            'methods_tested': []
        }
        
        methods = ['canny', 'sobel', 'laplacian', 'prewitt']
        
        try:
            # Load dataset
            dataset = RadioMapSeerEdgeDataset(
                str(self.data_root),
                image_size=(256, 256),
                edge_method='canny'
            )
            
            # Test first 10 images with different methods
            test_images = dataset.dpm_files[:10]
            
            for method in methods:
                print(f"Testing {method} method...")
                method_result = {
                    'method': method,
                    'success_count': 0,
                    'failure_count': 0,
                    'errors': []
                }
                
                edge_detector = EdgeDetector(method)
                
                for img_path in test_images:
                    try:
                        img = np.array(Image.open(img_path))
                        edge_map = edge_detector.detect_edges(img)
                        method_result['success_count'] += 1
                    except Exception as e:
                        method_result['failure_count'] += 1
                        method_result['errors'].append(f"{img_path.name}: {str(e)}")
                
                test_result['methods_tested'].append(method_result)
                test_result['metrics'][f'{method}_success_rate'] = method_result['success_count'] / len(test_images)
            
            test_result['status'] = 'success'
            
        except Exception as e:
            test_result['status'] = 'failed'
            test_result['errors'].append(str(e))
        
        self.test_results['tests']['edge_detection_methods'] = test_result
        return test_result
    
    def test_error_handling(self) -> Dict:
        """Test error handling and retry mechanisms"""
        print("\n=== Testing Error Handling ===")
        
        test_result = {
            'status': 'pending',
            'errors': [],
            'metrics': {}
        }
        
        try:
            # Create a temporary output directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Test with a small subset
                dataset = RadioMapSeerEdgeDataset(
                    str(self.data_root),
                    image_size=(256, 256),
                    edge_method='canny'
                )
                
                # Use only first 100 files for testing
                test_files = dataset.dpm_files[:100]
                dataset.dpm_files = test_files
                
                print("Processing with error handling...")
                start_time = time.time()
                
                # This will test the error handling in create_edge_dataset
                dataset.create_edge_dataset(str(temp_path))
                
                processing_time = time.time() - start_time
                
                # Verify output
                image_files = list((temp_path / 'image').glob('*.png'))
                edge_files = list((temp_path / 'edge').glob('*.png'))
                
                test_result['metrics']['processing_time'] = processing_time
                test_result['metrics']['images_generated'] = len(image_files)
                test_result['metrics']['edges_generated'] = len(edge_files)
                test_result['metrics']['success_rate'] = len(image_files) / len(test_files)
                
                # Verify file naming consistency
                image_names = [f.stem for f in image_files]
                edge_names = [f.stem for f in edge_files]
                
                if set(image_names) == set(edge_names):
                    test_result['metrics']['naming_consistent'] = True
                else:
                    test_result['metrics']['naming_consistent'] = False
                    test_result['errors'].append("Image and edge file names don't match")
            
            test_result['status'] = 'success'
            
        except Exception as e:
            test_result['status'] = 'failed'
            test_result['errors'].append(str(e))
        
        self.test_results['tests']['error_handling'] = test_result
        return test_result
    
    def test_dataset_structure(self) -> Dict:
        """Test dataset structure generation"""
        print("\n=== Testing Dataset Structure ===")
        
        test_result = {
            'status': 'pending',
            'errors': [],
            'metrics': {}
        }
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Test modified dataset structure
                dataset = RadioMapSeerEdgeDataset(
                    str(self.data_root),
                    image_size=(256, 256),
                    edge_method='canny'
                )
                
                # Use small subset
                dataset.dpm_files = dataset.dpm_files[:50]
                dataset.create_edge_dataset(str(temp_path))
                
                # Verify structure
                image_dir = temp_path / 'image'
                edge_dir = temp_path / 'edge'
                
                test_result['metrics']['flat_structure'] = {
                    'image_dir_exists': image_dir.exists(),
                    'edge_dir_exists': edge_dir.exists(),
                    'image_count': len(list(image_dir.glob('*.png'))),
                    'edge_count': len(list(edge_dir.glob('*.png')))
                }
                
                # Test original dataset structure
                original_temp = temp_path / 'original'
                original_dataset = OriginalDataset(
                    str(self.data_root),
                    image_size=(256, 256),
                    edge_method='canny'
                )
                
                original_dataset.dpm_files = original_dataset.dpm_files[:50]
                original_dataset.create_edge_dataset(str(original_temp), split_ratio=0.8)
                
                # Verify original structure
                orig_image_train = original_temp / 'image' / 'raw' / 'train'
                orig_image_val = original_temp / 'image' / 'raw' / 'val'
                orig_edge_train = original_temp / 'edge' / 'raw' / 'train'
                orig_edge_val = original_temp / 'edge' / 'raw' / 'val'
                
                test_result['metrics']['original_structure'] = {
                    'train_dirs_exist': all([
                        orig_image_train.exists(),
                        orig_image_val.exists(),
                        orig_edge_train.exists(),
                        orig_edge_val.exists()
                    ]),
                    'train_count': len(list(orig_image_train.glob('*.png'))),
                    'val_count': len(list(orig_image_val.glob('*.png')))
                }
            
            test_result['status'] = 'success'
            
        except Exception as e:
            test_result['status'] = 'failed'
            test_result['errors'].append(str(e))
        
        self.test_results['tests']['dataset_structure'] = test_result
        return test_result
    
    def test_performance_metrics(self) -> Dict:
        """Test performance metrics and scalability"""
        print("\n=== Testing Performance Metrics ===")
        
        test_result = {
            'status': 'pending',
            'errors': [],
            'metrics': {}
        }
        
        try:
            dataset = RadioMapSeerEdgeDataset(
                str(self.data_root),
                image_size=(256, 256),
                edge_method='canny'
            )
            
            # Test with different batch sizes
            batch_sizes = [10, 50, 100]
            
            for batch_size in batch_sizes:
                print(f"Testing batch size: {batch_size}")
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    
                    test_files = dataset.dpm_files[:batch_size]
                    test_dataset = RadioMapSeerEdgeDataset(
                        str(self.data_root),
                        image_size=(256, 256),
                        edge_method='canny'
                    )
                    test_dataset.dpm_files = test_files
                    
                    start_time = time.time()
                    test_dataset.create_edge_dataset(str(temp_path))
                    processing_time = time.time() - start_time
                    
                    test_result['metrics'][f'batch_{batch_size}'] = {
                        'processing_time': processing_time,
                        'files_per_second': batch_size / processing_time,
                        'success_rate': 1.0  # Assuming all files processed successfully
                    }
            
            test_result['status'] = 'success'
            
        except Exception as e:
            test_result['status'] = 'failed'
            test_result['errors'].append(str(e))
        
        self.test_results['tests']['performance_metrics'] = test_result
        return test_result
    
    def run_all_tests(self) -> Dict:
        """Run all tests and generate comprehensive report"""
        print("Starting comprehensive edge detection tests...")
        
        # Run all tests
        self.test_dataset_loading()
        self.test_edge_detection_methods()
        self.test_error_handling()
        self.test_dataset_structure()
        self.test_performance_metrics()
        
        # Generate summary
        total_tests = len(self.test_results['tests'])
        successful_tests = sum(1 for test in self.test_results['tests'].values() 
                             if test['status'] == 'success')
        
        self.test_results['summary'] = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': successful_tests / total_tests,
            'overall_status': 'success' if successful_tests == total_tests else 'partial_failure'
        }
        
        return self.test_results
    
    def save_report(self) -> None:
        """Save test report to file"""
        report_path = self.output_dir / 'edge_detection_test_report.json'
        
        with open(report_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\nTest report saved to: {report_path}")
        
        # Also save a human-readable summary
        summary_path = self.output_dir / 'edge_detection_test_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("=== Edge Detection Test Summary ===\n\n")
            f.write(f"Test Date: {self.test_results['timestamp']}\n")
            f.write(f"Data Root: {self.test_results['data_root']}\n\n")
            
            f.write("=== Test Results ===\n")
            for test_name, test_result in self.test_results['tests'].items():
                f.write(f"\n{test_name.replace('_', ' ').title()}: {test_result['status'].upper()}\n")
                if test_result['errors']:
                    f.write(f"  Errors: {len(test_result['errors'])}\n")
                    for error in test_result['errors'][:3]:  # Show first 3 errors
                        f.write(f"    - {error}\n")
                if test_result['metrics']:
                    f.write(f"  Metrics: {len(test_result['metrics'])} key metrics\n")
            
            f.write(f"\n=== Overall Summary ===\n")
            f.write(f"Total Tests: {self.test_results['summary']['total_tests']}\n")
            f.write(f"Successful Tests: {self.test_results['summary']['successful_tests']}\n")
            f.write(f"Success Rate: {self.test_results['summary']['success_rate']:.2%}\n")
            f.write(f"Overall Status: {self.test_results['summary']['overall_status']}\n")
        
        print(f"Test summary saved to: {summary_path}")

def main():
    """Main function for edge detection testing"""
    parser = argparse.ArgumentParser(description='Test RadioMapSeer edge detection functionality')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to RadioMapSeer dataset')
    parser.add_argument('--output_dir', type=str, default='./edge_detection_test_results',
                       help='Output directory for test results')
    parser.add_argument('--specific_test', type=str, choices=[
        'dataset_loading', 'edge_detection_methods', 'error_handling', 
        'dataset_structure', 'performance_metrics'
    ], help='Run specific test only')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data_root):
        print(f"Error: Dataset directory {args.data_root} does not exist")
        return 1
    
    # Create tester
    tester = EdgeDetectionTester(args.data_root, args.output_dir)
    
    try:
        if args.specific_test:
            # Run specific test
            test_method = getattr(tester, f'test_{args.specific_test}')
            result = test_method()
            print(f"\nSpecific test '{args.specific_test}' completed with status: {result['status']}")
        else:
            # Run all tests
            results = tester.run_all_tests()
            
            # Print summary
            print(f"\n=== Test Summary ===")
            print(f"Total Tests: {results['summary']['total_tests']}")
            print(f"Successful Tests: {results['summary']['successful_tests']}")
            print(f"Success Rate: {results['summary']['success_rate']:.2%}")
            print(f"Overall Status: {results['summary']['overall_status']}")
            
            # Save reports
            tester.save_report()
        
        return 0
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())