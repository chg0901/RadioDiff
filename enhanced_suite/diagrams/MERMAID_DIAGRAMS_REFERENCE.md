# RadioDiff Mermaid Diagrams Reference

This file contains all Mermaid diagrams from the comprehensive analysis report for reference and debugging.

---

## Diagram 1

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#1e40af', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#3b82f6', 'lineColor': '#60a5fa', 'secondaryColor': '#7c3aed', 'tertiaryColor': '#10b981', 'clusterBkg': '#f3f4f6', 'clusterBorder': '#9ca3af', 'fontSize': '18px'}}}%%
graph TD
    A["RadioDiff Training Pipeline"] --> B["Configuration Loading"]
    A --> C["Model Initialization"]
    A --> D["Data Pipeline"]
    A --> E["Training Process"]
    
    B --> B1["Load YAML Config"]
    B1 --> B2["Parse Arguments"]
    
    C --> C1["First Stage: VAE"]
    C1 --> C2["Encoder: 320x320 → 80x80"]
    C2 --> C3["Decoder: 80x80 → 320x320"]
    
    C --> C4["Second Stage: U-Net"]
    C4 --> C5["Dimensions: dim=128, channels=3"]
    C5 --> C6["Conditional Processing"]
    C6 --> C7["Window Attention"]
    
    C --> C8["Latent Diffusion Model"]
    C8 --> C9["Diffusion Parameters"]
    C9 --> C10["Objective: pred_KC"]
    
    D --> D1["Radio Dataset"]
    D1 --> D2["Data Root: RadioMapSeer"]
    D2 --> D3["Batch Size: 66"]
    D3 --> D4["DataLoader Setup"]
    
    E --> E1["Trainer Initialization"]
    E1 --> E2["Training Parameters"]
    E2 --> E3["Gradient Accumulation"]
    E3 --> E4["EMA Setup"]
    
    E --> E5["Training Loop"]
    E5 --> E6["Forward Pass"]
    E6 --> E7["Loss Computation"]
    E7 --> E8["Backward Pass"]
    E8 --> E9["Optimizer Step"]
    E9 --> E10["EMA Update"]
    
    E --> E11["Sampling & Saving"]
    E11 --> E12["Model Sampling"]
    E12 --> E13["Save Samples"]
    E13 --> E14["Save Model"]
    
    classDef primaryBox fill:#1e40af,stroke:#3b82f6,stroke-width:2px,color:#ffffff
    classDef secondaryBox fill:#7c3aed,stroke:#8b5cf6,stroke-width:2px,color:#ffffff
    classDef tertiaryBox fill:#10b981,stroke:#34d399,stroke-width:2px,color:#ffffff
    classDef quaternaryBox fill:#ef4444,stroke:#f87171,stroke-width:2px,color:#ffffff
    classDef quinaryBox fill:#f59e0b,stroke:#fbbf24,stroke-width:2px,color:#ffffff
    
    class A primaryBox
    class B secondaryBox
    class C tertiaryBox
    class D quaternaryBox
    class E quinaryBox
```

---

## Diagram 2

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#1e40af', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#3b82f6', 'lineColor': '#60a5fa', 'secondaryColor': '#7c3aed', 'tertiaryColor': '#10b981', 'clusterBkg': '#f3f4f6', 'clusterBorder': '#9ca3af', 'fontSize': '18px'}}}%%
graph TD
    A["RadioDiff Data Pipeline"] --> B["Data Input"]
    A --> C["Data Processing"]
    A --> D["Model Flow"]
    A --> E["Output Flow"]
    
    B --> B1["Dataset Configuration"]
    B1 --> B2["RadioUNet_c Dataset"]
    B2 --> B3["Dataset Directory"]
    B3 --> B4["Simulation Type"]
    B4 --> B5["Additional Parameters"]
    
    C --> C1["Data Loading"]
    C1 --> C2["Batch Configuration"]
    C2 --> C3["Data Iterator"]
    C3 --> C4["Batch Extraction"]
    
    C --> C5["Batch Structure"]
    C5 --> C6["Image Data"]
    C6 --> C7["Condition Data"]
    C7 --> C8["Optional Mask"]
    C8 --> C9["Image Names"]
    
    D --> D1["VAE Encoding"]
    D1 --> D2["Input: 320x320x1"]
    D2 --> D3["Encoder: VAE Encoder"]
    D3 --> D4["Latent Space"]
    D4 --> D5["VAE Decoder"]
    
    D --> D6["Conditional U-Net"]
    D6 --> D7["Condition Input"]
    D7 --> D8["Condition Processing"]
    D8 --> D9["Multi-Scale Features"]
    D9 --> D10["Window Attention"]
    
    D --> D11["Diffusion Process"]
    D11 --> D12["Forward Diffusion"]
    D12 --> D13["Reverse Diffusion"]
    D13 --> D14["Loss Computation"]
    
    E --> E1["Training Output"]
    E1 --> E2["Total Loss"]
    E2 --> E3["Reconstruction Loss"]
    E3 --> E4["KL Divergence Loss"]
    E4 --> E5["Logging"]
    
    E --> E6["Sampling Output"]
    E6 --> E7["Model Sampling"]
    E7 --> E8["Generated Images"]
    E8 --> E9["Sample Saving"]
    E9 --> E10["Distance Calculation"]
    
    E --> E11["Model Checkpoints"]
    E11 --> E12["Model State"]
    E12 --> E13["EMA Model"]
    E13 --> E14["Checkpoint Files"]
    
    classDef primaryBox fill:#1e40af,stroke:#3b82f6,stroke-width:2px,color:#ffffff
    classDef secondaryBox fill:#7c3aed,stroke:#8b5cf6,stroke-width:2px,color:#ffffff
    classDef tertiaryBox fill:#10b981,stroke:#34d399,stroke-width:2px,color:#ffffff
    classDef quaternaryBox fill:#ef4444,stroke:#f87171,stroke-width:2px,color:#ffffff
    classDef quinaryBox fill:#f59e0b,stroke:#fbbf24,stroke-width:2px,color:#ffffff
    
    class A primaryBox
    class B secondaryBox
    class C tertiaryBox
    class D quaternaryBox
    class E quinaryBox
```

---

## Diagram 3

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#1e40af', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#3b82f6', 'lineColor': '#60a5fa', 'secondaryColor': '#7c3aed', 'tertiaryColor': '#10b981', 'clusterBkg': '#f3f4f6', 'clusterBorder': '#9ca3af', 'fontSize': '18px'}}}%%
graph TD
    A["RadioDiff Training Process"] --> B["Training Setup"]
    A --> C["Training Loop"]
    A --> D["Optimization"]
    A --> E["Monitoring & Evaluation"]
    
    B --> B1["Accelerator Setup"]
    B1 --> B2["GPU Configuration"]
    B2 --> B3["Mixed Precision"]
    B3 --> B4["Distributed Strategy"]
    
    B --> B5["Optimizer Configuration"]
    B5 --> B6["Learning Rate"]
    B6 --> B7["Weight Decay"]
    B7 --> B8["LR Scheduler"]
    B8 --> B9["LR Lambda Function"]
    
    B --> B10["Training Parameters"]
    B10 --> B11["Total Steps"]
    B11 --> B12["Gradient Accumulation"]
    B12 --> B13["Save Frequency"]
    B13 --> B14["EMA Configuration"]
    
    C --> C1["Training Loop Structure"]
    C1 --> C2["Progress Bar"]
    C2 --> C3["Step Range"]
    C3 --> C4["Gradient Accumulation Loop"]
    
    C --> C5["Batch Processing"]
    C5 --> C6["Data Loading"]
    C6 --> C7["Device Transfer"]
    C7 --> C8["Model Hook"]
    C8 --> C9["Forward Pass"]
    
    C --> C10["Loss Computation"]
    C10 --> C11["Loss Scaling"]
    C11 --> C12["Loss Components"]
    C12 --> C13["Accumulated Loss"]
    C13 --> C14["Learning Rate Tracking"]
    
    D --> D1["Backward Pass"]
    D1 --> D2["Accelerator Backward"]
    D2 --> D3["Gradient Clipping"]
    D3 --> D4["Synchronization"]
    
    D --> D5["Optimizer Step"]
    D5 --> D6["Zero Gradients"]
    D6 --> D7["Optimizer Step"]
    D7 --> D8["LR Scheduler Step"]
    D8 --> D9["EMA Update"]
    
    D --> D10["Parameter Management"]
    D10 --> D11["Trainable Parameters"]
    D11 --> D12["Gradient Tracking"]
    D12 --> D13["Parameter Updates"]
    
    E --> E1["Logging System"]
    E1 --> E2["Console Logging"]
    E2 --> E3["File Logging"]
    E3 --> E4["TensorBoard Logging"]
    E4 --> E5["Log Frequency"]
    
    E --> E6["Model Evaluation"]
    E6 --> E7["Sampling Frequency"]
    E7 --> E8["Model Evaluation Mode"]
    E8 --> E9["Conditional Sampling"]
    E9 --> E10["Quality Metrics"]
    
    E --> E11["Checkpoint Management"]
    E11 --> E12["Save Checkpoint"]
    E12 --> E13["Checkpoint Contents"]
    E13 --> E14["Resume Capability"]
    
    E --> E15["Sample Management"]
    E15 --> E16["Sample Saving"]
    E16 --> E17["File Naming"]
    E17 --> E18["Ground Truth Comparison"]
    
    classDef primaryBox fill:#1e40af,stroke:#3b82f6,stroke-width:2px,color:#ffffff
    classDef secondaryBox fill:#7c3aed,stroke:#8b5cf6,stroke-width:2px,color:#ffffff
    classDef tertiaryBox fill:#10b981,stroke:#34d399,stroke-width:2px,color:#ffffff
    classDef quaternaryBox fill:#ef4444,stroke:#f87171,stroke-width:2px,color:#ffffff
    classDef quinaryBox fill:#f59e0b,stroke:#fbbf24,stroke-width:2px,color:#ffffff
    
    class A primaryBox
    class B secondaryBox
    class C tertiaryBox
    class D quaternaryBox
    class E quinaryBox
```

---

