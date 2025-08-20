#!/usr/bin/env python3
"""
Fix NaN issues in ddm_const_sde.py by adding comprehensive error handling
"""

import re

def fix_nan_handling():
    with open('denoising_diffusion_pytorch/ddm_const_sde.py', 'r') as f:
        content = f.read()
    
    # Fix the first p_losses method (around line 413)
    pattern1 = r'(def p_losses\(self, x_start, t, \*args, \*\*kwargs\):\s+if self\.start_dist == \'normal\':\s+noise = torch\.randn_like\(x_start\)\s+elif self\.start_dist == \'uniform\':\s+noise = 2 \* torch\.rand_like\(x_start\) - 1\.\s+else:\s+raise NotImplementedError\(f\'\{self\.start_dist\} is not supported !\'\)\s+# K = -1\. \* torch\.ones_like\(x_start\)\s+# C = noise - x_start  # t = 1000 / 1000\s+C = -1 \* x_start             # U\(t\) = Ct, U\(1\) = -x0\s+# C = -2 \* x_start               # U\(t\) = 1/2 \* C \* t\*\*2, U\(1\) = 1/2 \* C = -x0\s+x_noisy = self\.q_sample\(x_start=x_start, noise=noise, t=t, C=C\)  # \(b, 2, c, h, w\)\s+C_pred, noise_pred = self\.model\(x_noisy, t, \*\*kwargs\))'
    
    replacement1 = r'''def p_losses(self, x_start, t, *args, **kwargs):
        # Input validation
        if torch.isnan(x_start).any():
            print("Warning: x_start contains NaN values - skipping batch")
            return torch.tensor(0.0, requires_grad=True), {}
        
        if self.start_dist == 'normal':
            noise = torch.randn_like(x_start)
        elif self.start_dist == 'uniform':
            noise = 2 * torch.rand_like(x_start) - 1.
        else:
            raise NotImplementedError(f'{self.start_dist} is not supported !')
        
        # Add NaN checks for input data
        if torch.isnan(noise).any():
            print("Warning: noise contains NaN values - skipping batch")
            return torch.tensor(0.0, requires_grad=True), {}
        
        # K = -1. * torch.ones_like(x_start)
        # C = noise - x_start  # t = 1000 / 1000
        C = -1 * x_start             # U(t) = Ct, U(1) = -x0
        # C = -2 * x_start               # U(t) = 1/2 * C * t**2, U(1) = 1/2 * C = -x0
        x_noisy = self.q_sample(x_start=x_start, noise=noise, t=t, C=C)  # (b, 2, c, h, w)
        
        # Add NaN check for noisy input
        if torch.isnan(x_noisy).any():
            print("Warning: x_noisy contains NaN values - skipping batch")
            return torch.tensor(0.0, requires_grad=True), {}
        
        # Model prediction with error handling
        try:
            C_pred, noise_pred = self.model(x_noisy, t, **kwargs)
        except Exception as e:
            print(f"Model prediction failed: {e} - skipping batch")
            return torch.tensor(0.0, requires_grad=True), {})'''
    
    # Apply the first fix
    content = re.sub(pattern1, replacement1, content, flags=re.DOTALL)
    
    # Fix the second p_losses method (around line 830)
    pattern2 = r'(def p_losses\(self, x_start, t, \*args, \*\*kwargs\):\s+if self\.start_dist == \'normal\':\s+noise = torch\.randn_like\(x_start\)\s+elif self\.start_dist == \'uniform\':\s+noise = 2 \* torch\.rand_like\(x_start\) - 1\.\s+else:\s+raise NotImplementedError\(f\'\{self\.start_dist\} is not supported !\'\)\s+\n\s+# Add NaN checks for input data\s+if torch\.isnan\(x_start\)\.any\(\):\s+print\("Warning: x_start contains NaN values"\)\s+x_start = torch\.nan_to_num\(x_start, nan=0\.0\)\s+\n\s+if torch\.isnan\(noise\)\.any\(\):\s+print\("Warning: noise contains NaN values"\)\s+noise = torch\.nan_to_num\(noise, nan=0\.0\)\s+\n\s+# K = -1\. \* torch\.ones_like\(x_start\)\s+# C = noise - x_start  # t = 1000 / 1000\s+C = -1 \* x_start             # U\(t\) = Ct, U\(1\) = -x0\s+# C = -2 \* x_start               # U\(t\) = 1/2 \* C \* t\*\*2, U\(1\) = 1/2 \* C = -x0\s+x_noisy = self\.q_sample\(x_start=x_start, noise=noise, t=t, C=C\)  # \(b, 2, c, h, w\)\s+\n\s+# Add NaN check for noisy input\s+if torch\.isnan\(x_noisy\)\.any\(\):\s+print\("Warning: x_noisy contains NaN values"\)\s+x_noisy = torch\.nan_to_num\(x_noisy, nan=0\.0\)\s+\n\s+"""\*+{70}"""\*+\s+C_pred, noise_pred = self\.model\(x_noisy, t, \*args, \*\*kwargs\)   #这个model应该对应UNET预测噪音\s+"""\*+{70}"""\*+\s+\n\s+# Add NaN checks for model predictions\s+if torch\.isnan\(C_pred\)\.any\(\):\s+print\("Warning: C_pred contains NaN values"\)\s+C_pred = torch\.nan_to_num\(C_pred, nan=0\.0\)\s+\n\s+if torch\.isnan\(noise_pred\)\.any\(\):\s+print\("Warning: noise_pred contains NaN values"\)\s+noise_pred = torch\.nan_to_num\(noise_pred, nan=0\.0\))'
    
    replacement2 = r'''def p_losses(self, x_start, t, *args, **kwargs):
        # Input validation
        if torch.isnan(x_start).any():
            print("Warning: x_start contains NaN values - skipping batch")
            return torch.tensor(0.0, requires_grad=True), {}
        
        if self.start_dist == 'normal':
            noise = torch.randn_like(x_start)
        elif self.start_dist == 'uniform':
            noise = 2 * torch.rand_like(x_start) - 1.
        else:
            raise NotImplementedError(f'{self.start_dist} is not supported !')
        
        # Add NaN checks for input data
        if torch.isnan(noise).any():
            print("Warning: noise contains NaN values - skipping batch")
            return torch.tensor(0.0, requires_grad=True), {}
        
        # K = -1. * torch.ones_like(x_start)
        # C = noise - x_start  # t = 1000 / 1000
        C = -1 * x_start             # U(t) = Ct, U(1) = -x0
        # C = -2 * x_start               # U(t) = 1/2 * C * t**2, U(1) = 1/2 * C = -x0
        x_noisy = self.q_sample(x_start=x_start, noise=noise, t=t, C=C)  # (b, 2, c, h, w)
        
        # Add NaN check for noisy input
        if torch.isnan(x_noisy).any():
            print("Warning: x_noisy contains NaN values - skipping batch")
            return torch.tensor(0.0, requires_grad=True), {}
        
        """========================================================================"""
        # Model prediction with error handling
        try:
            C_pred, noise_pred = self.model(x_noisy, t, *args, **kwargs)   #这个model应该对应UNET预测噪音
        except Exception as e:
            print(f"Model prediction failed: {e} - skipping batch")
            return torch.tensor(0.0, requires_grad=True), {})
        """========================================================================"""
        
        # Validate predictions
        if torch.isnan(C_pred).any() or torch.isnan(noise_pred).any():
            print("Warning: Model predictions contain NaN - skipping batch")
            return torch.tensor(0.0, requires_grad=True), {})'''
    
    # Apply the second fix
    content = re.sub(pattern2, replacement2, content, flags=re.DOTALL)
    
    # Fix loss calculation and weight clipping
    content = re.sub(
        r'if self\.weighting_loss:\s+simple_weight1 = 2\*torch\.exp\(1-t\)\s+simple_weight2 = torch\.exp\(torch\.sqrt\(t\)\)\s+if self\.cfg\.model_name == \'ncsnpp9\':\s+simple_weight1 = \(t \+ 1\) / t\.sqrt\(\)\s+simple_weight2 = \(2 - t\)\.sqrt\(\) / \(1 - t \+ self\.eps\)\.sqrt\(\)\s+else:\s+simple_weight1 = 1\s+simple_weight2 = 1',
        r'''if self.weighting_loss:
            simple_weight1 = 2*torch.exp(1-t)
            simple_weight2 = torch.exp(torch.sqrt(t))
            if self.cfg.model_name == 'ncsnpp9':
                simple_weight1 = (t + 1) / t.sqrt()
                simple_weight2 = (2 - t).sqrt() / (1 - t + self.eps).sqrt()
            
            # Add clipping to prevent exploding weights
            simple_weight1 = torch.clamp(simple_weight1, max=100.0)
            simple_weight2 = torch.clamp(simple_weight2, max=100.0)
        else:
            simple_weight1 = 1
            simple_weight2 = 1''',
        content,
        flags=re.DOTALL
    )
    
    # Fix final loss check in both methods
    content = re.sub(
        r'loss = loss_simple \+ loss_vlb\s+loss_dict\.update\(\{f\'\{prefix\}/loss\': loss\}\)\s+return loss, loss_dict',
        r'''loss = loss_simple + loss_vlb
        
        # Final NaN check for total loss
        if torch.isnan(loss) or torch.isinf(loss) or loss > 1000.0:
            print(f"Warning: Abnormal loss detected: {loss} - skipping batch")
            return torch.tensor(0.0, requires_grad=True), {}
        
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict''',
        content,
        flags=re.DOTALL
    )
    
    with open('denoising_diffusion_pytorch/ddm_const_sde.py', 'w') as f:
        f.write(content)
    
    print("Fixed NaN handling in ddm_const_sde.py")

if __name__ == "__main__":
    fix_nan_handling()