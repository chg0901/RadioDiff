# RadioDiff Training NaN Values Bug Analysis Report

## Issue Summary
When resuming training with `accelerate launch train_cond_ldm.py --cfg ./configs/radio_train_m.yaml`, the training process encounters persistent NaN (Not a Number) values in model predictions (`C_pred` and `noise_pred`), leading to training instability.

## Problem Analysis

### 1. Root Cause Identification

#### A. Model State Incompatibility
- **Issue**: The model is being resumed from checkpoint `radiodiff_LDM/model-61.pt` (line 23 in config)
- **Problem**: The checkpoint was saved with a different model state or configuration than the current setup
- **Evidence**: NaN warnings appear immediately when resuming training at step 12200

#### B. Loss Function Implementation Issues
**Location**: `denoising_diffusion_pytorch/ddm_const_sde.py:863-869`

```python
# Current problematic code:
if torch.isnan(C_pred).any():
    print("Warning: C_pred contains NaN values")
    C_pred = torch.nan_to_num(C_pred, nan=0.0)

if torch.isnan(noise_pred).any():
    print("Warning: noise_pred contains NaN values")
    noise_pred = torch.nan_to_num(noise_pred, nan=0.0)
```

**Issues**:
1. **Inconsistent NaN handling**: NaN values are only detected but not properly handled
2. **Missing gradient clipping**: No gradient clipping in the training loop
3. **Inadequate loss monitoring**: Loss continues to propagate NaN values despite warnings

#### C. Training Configuration Problems
**Configuration Issues**:
- `weighting_loss: True` (line 20) - Complex weighting may cause numerical instability
- `scale_factor: 0.3` (line 15) - May be too aggressive for the resumed model
- `lr: 1e-5` (line 74) - Learning rate might be too high for resumed training
- `fp16: False` (line 84) - Mixed precision disabled, but `amp: True` is enabled

#### D. Model Architecture Mismatch
**Potential Issues**:
- VAE checkpoint path: `radiodiff_Vae/model-29.pt` (line 43)
- LDM checkpoint path: `radiodiff_LDM/model-61.pt` (line 23)
- These checkpoints may have been trained with different configurations

### 2. Specific Code Issues

#### A. Missing Data Validation
**Location**: `train_cond_ldm.py:434-436`
```python
for key in batch.keys():
    if isinstance(batch[key], torch.Tensor):
        batch[key].to(device)  # Missing assignment!
```
**Problem**: `.to(device)` doesn't modify tensors in-place, so data remains on CPU.

#### B. Inadequate Gradient Clipping
**Location**: `train_cond_ldm.py:486`
```python
accelerator.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), 0.5)
```
**Problem**: Gradient clipping threshold (0.5) may be too restrictive.

#### C. Weighting Loss Calculation
**Location**: `ddm_const_sde.py:889-901`
```python
simple_weight1 = 2*torch.exp(1-t)
simple_weight2 = torch.exp(torch.sqrt(t))
```
**Problem**: Exponential weights can explode when `t` approaches 0 or 1.

## Proposed Solutions

### 1. Immediate Fixes

#### A. Fix Data Transfer Issue
```python
# Replace lines 434-436 in train_cond_ldm.py:
for key in batch.keys():
    if isinstance(batch[key], torch.Tensor):
        batch[key] = batch[key].to(device)  # Add assignment
```

#### B. Add Proper NaN Handling
```python
# Add comprehensive NaN checks in ddm_const_sde.py p_losses method:
def p_losses(self, x_start, t, *args, **kwargs):
    # Input validation
    if torch.isnan(x_start).any():
        print("Warning: x_start contains NaN values - skipping batch")
        return torch.tensor(0.0, requires_grad=True), {}
    
    # ... existing code ...
    
    # Model prediction with error handling
    try:
        C_pred, noise_pred = self.model(x_noisy, t, *args, **kwargs)
    except Exception as e:
        print(f"Model prediction failed: {e}")
        return torch.tensor(0.0, requires_grad=True), {}
    
    # Validate predictions
    if torch.isnan(C_pred).any() or torch.isnan(noise_pred).any():
        print("Warning: Model predictions contain NaN - skipping batch")
        return torch.tensor(0.0, requires_grad=True), {}
```

#### C. Improve Loss Weighting
```python
# Add clipping to prevent weight explosion
simple_weight1 = torch.clamp(2*torch.exp(1-t), max=100.0)
simple_weight2 = torch.clamp(torch.exp(torch.sqrt(t)), max=100.0)
```

### 2. Configuration Adjustments

#### A. Learning Rate Adjustment
```yaml
# Reduce learning rate for resumed training
trainer:
  lr: !!float 5e-6  # Reduced from 1e-5
```

#### B. Disable Weighting Loss Initially
```yaml
# Disable complex weighting during resume
model:
  weighting_loss: False  # Disable temporarily
```

#### C. Adjust Scale Factor
```yaml
# Use more conservative scaling
model:
  scale_factor: 0.18215  # Standard VAE scaling
```

### 3. Model State Synchronization

#### A. Checkpoint Compatibility
```python
# Add checkpoint validation in training script:
def validate_checkpoint_compatibility(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check key compatibility
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(checkpoint['model'].keys())
    
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys
    
    if missing_keys or unexpected_keys:
        print(f"Checkpoint compatibility issues:")
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        return False
    
    return True
```

### 4. Enhanced Training Stability

#### A. Gradient Clipping Improvement
```python
# Use adaptive gradient clipping
grad_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(), 
    max_norm=1.0,  # Increased from 0.5
    norm_type=2.0
)
```

#### B. Loss Monitoring
```python
# Add loss monitoring in training loop
if torch.isnan(loss) or loss > 1000.0:
    print(f"Warning: Abnormal loss detected: {loss}")
    optimizer.zero_grad()
    continue
```

## Implementation Priority

1. **High Priority**: Fix data transfer issue (batch tensors not moving to device)
2. **High Priority**: Add comprehensive NaN handling with batch skipping
3. **Medium Priority**: Adjust configuration parameters (LR, weighting, scaling)
4. **Medium Priority**: Implement checkpoint validation
5. **Low Priority**: Enhanced monitoring and logging

## Expected Outcome

After implementing these fixes:
- NaN warnings should be eliminated
- Training should proceed stably
- Loss should decrease normally
- Model convergence should be achievable

## Verification Steps

1. Apply fixes incrementally
2. Monitor training logs for NaN warnings
3. Check loss values for stability
4. Validate model outputs periodically
5. Ensure checkpoint saving works correctly

## Additional Recommendations

1. **Logging**: Enhanced logging for debugging
2. **Validation**: Regular validation checks
3. **Backup**: Save multiple checkpoint versions
4. **Monitoring**: Implement gradient and parameter norm monitoring
5. **Testing**: Test with smaller batch sizes first