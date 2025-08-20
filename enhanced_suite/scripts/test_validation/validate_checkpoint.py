#!/usr/bin/env python3
"""
Checkpoint validation script for RadioDiff training
"""

import os
import torch
import yaml
from pathlib import Path

def validate_checkpoint_config(config_path):
    """Validate checkpoint configuration before training"""
    
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    print("=== Checkpoint Configuration Validation ===")
    
    # Check initial model checkpoint
    model_cfg = cfg.get('model', {})
    ckpt_path = model_cfg.get('ckpt_path')
    
    # Handle commented out or None checkpoint path
    if not ckpt_path or (isinstance(ckpt_path, str) and ckpt_path.strip().startswith('#')):
        ckpt_path = None
        print("ℹ  No initial checkpoint specified")
    elif ckpt_path:
        if os.path.exists(ckpt_path):
            print(f"✓ Initial checkpoint found: {ckpt_path}")
            # Try to load it
            try:
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                print(f"  - Checkpoint keys: {list(checkpoint.keys())}")
                if 'model' in checkpoint:
                    print(f"  - Model parameters: {len(checkpoint['model'])}")
            except Exception as e:
                print(f"  ✗ Error loading checkpoint: {e}")
        else:
            print(f"✗ Initial checkpoint NOT found: {ckpt_path}")
    
    # Check resume checkpoint
    trainer_cfg = cfg.get('trainer', {})
    resume_milestone = trainer_cfg.get('resume_milestone', 0)
    results_folder = trainer_cfg.get('results_folder', './results')
    
    if resume_milestone > 0:
        resume_file = Path(results_folder) / f'model-{resume_milestone}.pt'
        if resume_file.exists():
            print(f"✓ Resume checkpoint found: {resume_file}")
        else:
            print(f"✗ Resume checkpoint NOT found: {resume_file}")
            print("  This will cause training to start from scratch!")
    
    # Check first stage model checkpoint
    first_stage_cfg = model_cfg.get('first_stage', {})
    vae_ckpt_path = first_stage_cfg.get('ckpt_path')
    
    if vae_ckpt_path:
        if os.path.exists(vae_ckpt_path):
            print(f"✓ VAE checkpoint found: {vae_ckpt_path}")
        else:
            print(f"✗ VAE checkpoint NOT found: {vae_ckpt_path}")
    
    print("\n=== Recommendations ===")
    
    if not ckpt_path and resume_milestone > 0 and not resume_file.exists():
        print("⚠  WARNING: No initial checkpoint and resume checkpoint not found.")
        print("   Training will start from scratch, which may cause NaN issues.")
        print("   Consider:")
        print("   1. Setting resume_milestone: 0 for fresh training")
        print("   2. Or providing a valid initial checkpoint path")
        print("   3. Or copying an existing checkpoint to the expected location")
    
    if not ckpt_path and resume_milestone == 0:
        print("ℹ  INFO: Starting fresh training with no initial weights.")
        print("   This is normal for new training runs.")
    
    if ckpt_path and not os.path.exists(ckpt_path):
        print("⚠  WARNING: Initial checkpoint path does not exist.")
        print("   Training will start with random weights.")
    
    if resume_milestone > 0 and not resume_file.exists():
        print(f"⚠  WARNING: Resume milestone {resume_milestone} specified but checkpoint not found.")
        print(f"   Expected: {resume_file}")
        print("   Training will start from step 0 instead.")
    
    return True

if __name__ == "__main__":
    validate_checkpoint_config('./configs/radio_train_m.yaml')