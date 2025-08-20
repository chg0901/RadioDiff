#!/bin/bash

# Edge Detection LDM Training Script
# This script trains a Latent Diffusion Model for edge detection using AdaptEdgeDataset

# Configuration
CONFIG_FILE="configs_edge/edge_ldm_train.yaml"
PYTHON_SCRIPT="train_cond_ldm_m.py"

# Check if configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE not found!"
    exit 1
fi

# Check if training script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Training script $PYTHON_SCRIPT not found!"
    exit 1
fi

# Create results directory if it doesn't exist
mkdir -p edge_ldm_results

# Run LDM training
echo "Starting Edge Detection LDM Training..."
echo "Configuration: $CONFIG_FILE"
echo "Script: $PYTHON_SCRIPT"
echo "Results will be saved to: edge_ldm_results/"

# Set GPU if available
export CUDA_VISIBLE_DEVICES=0

# Run the training
python "$PYTHON_SCRIPT" --cfg "$CONFIG_FILE"

echo "LDM training completed!"
echo "Check the edge_ldm_results/ directory for trained models and logs."