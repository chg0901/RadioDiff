# RadioDiff VAE Training Report Update Prompt

## Overview
This prompt outlines the complete process for updating the RadioDiff VAE training visualization report with new log data using automated scripts.

## Prerequisites
- Training log file in `radiodiff_Vae/` directory (format: `YYYY-MM-DD-HH-MM_.log`)
- Existing visualization script: `improved_visualization_final.py`
- Existing report template: `radiodiff_Vae/training_visualization_report.md`

## Automated Solution

### 1. Quick Update (Recommended)
```bash
# Run the automated script
python update_training_report.py

# Or with specific files
python update_training_report.py --log_file radiodiff_Vae/2025-08-15-17-21_.log --report_file radiodiff_Vae/training_visualization_report.md
```

### 2. Manual Step-by-Step Process

#### Step 1: Parse Training Log
```python
import pandas as pd
import re
import matplotlib.pyplot as plt

def parse_log_file(log_file_path):
    pattern = r'\[Train Step\] (\d+)/\d+: (.+?)(?= lr: 0\.0+,\s*$)'
    step_data = {}
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        match = re.search(pattern, line)
        if match:
            step = int(match.group(1))
            metrics_str = match.group(2)
            
            if step not in step_data:
                step_data[step] = {'step': step}
            
            for metric in metrics_str.split(', '):
                if ': ' in metric:
                    key, value = metric.split(': ')
                    try:
                        step_data[step][key] = float(value)
                    except ValueError:
                        continue
    
    df = pd.DataFrame(list(step_data.values()))
    return df.sort_values('step').reset_index(drop=True)

# Usage
df = parse_log_file('radiodiff_Vae/2025-08-15-17-21_.log')
print(f"Found {len(df)} training steps")
```

#### Step 2: Generate Visualizations
```bash
# Use existing visualization script
python improved_visualization_final.py
```

#### Step 3: Extract Key Metrics
```python
def extract_metrics(df):
    latest_step = df['step'].max()
    latest_data = df[df['step'] == latest_step].iloc[0]
    
    metrics = {
        'total_steps': len(df),
        'current_step': int(latest_step),
        'progress_percentage': (latest_step / 150000) * 100,
        'total_loss': latest_data.get('train/total_loss', 0),
        'kl_loss': latest_data.get('train/kl_loss', 0),
        'rec_loss': latest_data.get('train/rec_loss', 0),
        'disc_loss': latest_data.get('train/disc_loss', 0)
    }
    
    # Calculate ranges and percentages
    metrics['total_loss_range'] = (df['train/total_loss'].min(), df['train/total_loss'].max())
    metrics['kl_loss_range'] = (df['train/kl_loss'].min(), df['train/kl_loss'].max())
    metrics['rec_loss_range'] = (df['train/rec_loss'].min(), df['train/rec_loss'].max())
    
    return metrics

metrics = extract_metrics(df)
```

#### Step 4: Update Report Template
```python
def update_report_template(metrics, report_file):
    with open(report_file, 'r') as f:
        content = f.read()
    
    # Update key sections
    replacements = {
        r'over \d+,\d+ steps \(\d+\.\d+% complete\)': f'over {metrics["current_step"]:,} steps ({metrics["progress_percentage"]:.1f}% complete)',
        r'\*\*Total Loss\*\* \| [\d,]+ \| [\d,]+ - [\d,]+': f'**Total Loss** | {metrics["total_loss"]:,.0f} | {metrics["total_loss_range"][0]:,.0f} - {metrics["total_loss_range"][1]:,.0f}',
        r'\*\*KL Loss\*\* \| [\d,]+ \| [\d,]+ - [\d,]+': f'**KL Loss** | {metrics["kl_loss"]:,.0f} | {metrics["kl_loss_range"][0]:,.0f} - {metrics["kl_loss_range"][1]:,.0f}',
        r'\*\*Reconstruction Loss\*\* \| [\d\.]+ \| [\d\.]+ - [\d\.]+': f'**Reconstruction Loss** | {metrics["rec_loss"]:.2f} | {metrics["rec_loss_range"][0]:.2f} - {metrics["rec_loss_range"][1]:.2f}',
    }
    
    for pattern, replacement in replacements.items():
        content = re.sub(pattern, replacement, content)
    
    with open(report_file, 'w') as f:
        f.write(content)
```

## File Structure

```
RadioDiff/
├── update_training_report.py          # Automated script
├── improved_visualization_final.py    # Visualization generator
├── radiodiff_Vae/
│   ├── training_visualization_report.md # Report template
│   ├── 2025-08-15-17-21_.log          # Training log
│   ├── normalized_comparison_improved.png
│   ├── multi_axis_losses_improved.png
│   └── metrics_overview_improved.png
└── requirements.txt                   # Dependencies
```

## Dependencies

Required packages (install with `pip install -r requirements.txt`):
```
pandas>=1.3.0
matplotlib>=3.5.0
numpy>=1.21.0
```

## Usage Examples

### 1. Standard Update
```bash
# Update with latest log file
python update_training_report.py
```

### 2. Custom Files
```bash
# Update with specific log and report files
python update_training_report.py \
    --log_file radiodiff_Vae/2025-08-16-10-30_.log \
    --report_file radiodiff_Vae/custom_report.md
```

### 3. Verbose Output
```bash
# Enable detailed logging
python update_training_report.py --verbose
```

## Expected Output

The script will:
1. Parse the training log file
2. Extract current metrics (losses, steps, progress)
3. Generate updated visualizations
4. Update the markdown report with new data
5. Display summary statistics

## Troubleshooting

### Common Issues:
1. **Log file not found**: Ensure the log file exists in the correct location
2. **Permission errors**: Make sure the script has write permissions
3. **Missing dependencies**: Install required packages with pip
4. **Visualization errors**: Check matplotlib backend settings

### Debug Mode:
```bash
# Enable verbose output for debugging
python update_training_report.py --verbose
```

## Automated Scheduling

To run updates automatically, add to crontab:
```bash
# Update report every 6 hours
0 */6 * * * cd /home/cine/Documents/Github/RadioDiff && python update_training_report.py >> /var/log/radiodiff_report.log 2>&1
```

## Integration with Training Pipeline

Add to training script:
```python
# At the end of training or at checkpoints
from update_training_report import TrainingReportUpdater

updater = TrainingReportUpdater()
updater.run()
```

This complete solution provides both automated and manual methods for keeping the training visualization report current with the latest training data.