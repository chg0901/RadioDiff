#!/bin/bash

# Edge Detection Inference Script
# This script performs edge detection inference using trained models

# Default configuration
CONFIG_FILE="configs_edge/edge_sample.yaml"
PYTHON_SCRIPT="demo.py"
INPUT_DIR=""
OUTPUT_DIR=""
PRETRAINED_WEIGHT=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cfg)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --input_dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --out_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --pre_weight)
            PRETRAINED_WEIGHT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --cfg CONFIG_FILE         Configuration file (default: configs_edge/edge_sample.yaml)"
            echo "  --input_dir INPUT_DIR     Input directory containing test images"
            echo "  --out_dir OUTPUT_DIR      Output directory for results"
            echo "  --pre_weight WEIGHT_PATH  Path to pretrained model weights"
            echo "  --help                    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE not found!"
    exit 1
fi

# Check if demo script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Demo script $PYTHON_SCRIPT not found!"
    exit 1
fi

# Validate required parameters
if [ -z "$INPUT_DIR" ]; then
    echo "Error: --input_dir is required!"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --out_dir is required!"
    exit 1
fi

if [ -z "$PRETRAINED_WEIGHT" ]; then
    echo "Error: --pre_weight is required!"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run edge detection inference
echo "Starting Edge Detection Inference..."
echo "Configuration: $CONFIG_FILE"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Pretrained weights: $PRETRAINED_WEIGHT"

# Set GPU if available
export CUDA_VISIBLE_DEVICES=0

# Run the inference
python "$PYTHON_SCRIPT" \
    --cfg "$CONFIG_FILE" \
    --input_dir "$INPUT_DIR" \
    --out_dir "$OUTPUT_DIR" \
    --pre_weight "$PRETRAINED_WEIGHT" \
    --bs 8 \
    --sampling_timesteps 1

echo "Edge detection inference completed!"
echo "Results saved to: $OUTPUT_DIR/"