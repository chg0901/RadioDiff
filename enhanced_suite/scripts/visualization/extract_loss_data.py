#!/usr/bin/env python3
"""
Extract loss data from training logs for visualization
"""
import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def extract_loss_data(log_file):
    """Extract loss data from training log file"""
    steps = []
    total_losses = []
    kl_losses = []
    rec_losses = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Match pattern: [Train Step] X/Y: train/total_loss: Z, train/kl_loss: A, train/rec_loss: B
            match = re.search(r'\[Train Step\] (\d+)/\d+.*?train/total_loss: ([\d.]+).*?train/kl_loss: ([\d.]+).*?train/rec_loss: ([\d.]+)', line)
            if match:
                step = int(match.group(1))
                total_loss = float(match.group(2))
                kl_loss = float(match.group(3))
                rec_loss = float(match.group(4))
                
                steps.append(step)
                total_losses.append(total_loss)
                kl_losses.append(kl_loss)
                rec_losses.append(rec_loss)
    
    return {
        'steps': np.array(steps),
        'total_losses': np.array(total_losses),
        'kl_losses': np.array(kl_losses),
        'rec_losses': np.array(rec_losses)
    }

def create_comparison_plot(old_data, new_data, output_file):
    """Create comparison plot of old vs new training"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total Loss
    ax1.plot(old_data['steps'], old_data['total_losses'], 'b-', label='Original Training', alpha=0.7)
    ax1.plot(new_data['steps'], new_data['total_losses'], 'r-', label='Resumed Training', alpha=0.7)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # KL Loss
    ax2.plot(old_data['steps'], old_data['kl_losses'], 'b-', label='Original Training', alpha=0.7)
    ax2.plot(new_data['steps'], new_data['kl_losses'], 'r-', label='Resumed Training', alpha=0.7)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('KL Loss')
    ax2.set_title('KL Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Reconstruction Loss
    ax3.plot(old_data['steps'], old_data['rec_losses'], 'b-', label='Original Training', alpha=0.7)
    ax3.plot(new_data['steps'], new_data['rec_losses'], 'r-', label='Resumed Training', alpha=0.7)
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Reconstruction Loss')
    ax3.set_title('Reconstruction Loss Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Combined zoom view (last 5000 steps before resume + resumed training)
    zoom_start = max(0, len(old_data['steps']) - 50)
    zoom_end = len(new_data['steps'])
    
    ax4.plot(old_data['steps'][zoom_start:], old_data['total_losses'][zoom_start:], 
             'b-', label='Original Training (End)', alpha=0.7, linewidth=2)
    ax4.plot(new_data['steps'][:zoom_end], new_data['total_losses'][:zoom_end], 
             'r-', label='Resumed Training', alpha=0.7, linewidth=2)
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Total Loss')
    ax4.set_title('Resume Transition (Zoomed View)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Extract data from both logs
    old_log = 'radiodiff_Vae/2025-08-15-17-21_.log'
    new_log = 'radiodiff_Vae/2025-08-15-20-41_.log'
    
    print("Extracting data from original training log...")
    old_data = extract_loss_data(old_log)
    print(f"Original training: {len(old_data['steps'])} data points")
    print(f"Step range: {old_data['steps'][0]} to {old_data['steps'][-1]}")
    
    print("\nExtracting data from resumed training log...")
    new_data = extract_loss_data(new_log)
    print(f"Resumed training: {len(new_data['steps'])} data points")
    print(f"Step range: {new_data['steps'][0]} to {new_data['steps'][-1]}")
    
    # Create comparison plot
    print("\nCreating comparison visualization...")
    create_comparison_plot(old_data, new_data, 'radiodiff_Vae/training_loss_comparison.png')
    
    # Print statistics
    print(f"\n=== Training Statistics ===")
    print(f"Original Training Final Step: {old_data['steps'][-1]}")
    print(f"Original Training Final Loss: {old_data['total_losses'][-1]:.2f}")
    print(f"Resumed Training Start Step: {new_data['steps'][0]}")
    print(f"Resumed Training Start Loss: {new_data['total_losses'][0]:.2f}")
    print(f"Resumed Training Current Step: {new_data['steps'][-1]}")
    print(f"Resumed Training Current Loss: {new_data['total_losses'][-1]:.2f}")
    
    # Calculate loss continuity
    expected_resume_step = 35000
    old_step_idx = np.where(old_data['steps'] >= expected_resume_step)[0]
    if len(old_step_idx) > 0:
        old_step_idx = old_step_idx[0]
        old_loss_at_resume = old_data['total_losses'][old_step_idx]
        new_loss_at_resume = new_data['total_losses'][0]
        loss_diff = abs(old_loss_at_resume - new_loss_at_resume)
        print(f"\n=== Resume Continuity Check ===")
        print(f"Expected resume step: {expected_resume_step}")
        print(f"Original training loss at step {expected_resume_step}: {old_loss_at_resume:.2f}")
        print(f"Resumed training loss at step {expected_resume_step}: {new_loss_at_resume:.2f}")
        print(f"Loss difference: {loss_diff:.2f}")
        print(f"Continuity status: {'✅ GOOD' if loss_diff < 1000 else '⚠️ LARGE GAP'}")

if __name__ == "__main__":
    main()