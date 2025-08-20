#!/usr/bin/env python3
"""
Enhanced Mermaid Diagram Renderer for RadioDiff
Renders mermaid diagrams with proper formatting, LaTeX support, and 16:9 aspect ratio
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any

class EnhancedMermaidRenderer:
    def __init__(self, output_dir: str = "./enhanced_mermaid_vis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Mermaid CLI configuration for enhanced rendering
        self.mmdc_config = {
            'width': 1920,    # 16:9 aspect ratio
            'height': 1080,
            'scale': 2,       # High resolution
            'theme': 'default',
            'background': 'transparent',
            'pdfFit': False,
            'fontFamily': 'Arial',
            'fontSize': 14,   # Larger font for better readability
        }
        
        # CSS styling for enhanced appearance
        self.css_style = """
        .nodeLabel {
            font-size: 14px;
            font-weight: bold;
        }
        .edgeLabel {
            font-size: 12px;
        }
        .cluster-label {
            font-size: 16px;
            font-weight: bold;
        }
        .mermaid {
            font-family: Arial, sans-serif;
        }
        """
    
    def create_enhanced_diagrams(self) -> List[str]:
        """Create all enhanced mermaid diagrams for the comprehensive report"""
        
        diagrams = []
        
        # 1. System Architecture Overview
        diagrams.append(self.create_system_architecture())
        
        # 2. VAE Architecture
        diagrams.append(self.create_vae_architecture())
        
        # 3. Conditional U-Net
        diagrams.append(self.create_conditional_unet())
        
        # 4. Training Workflow
        diagrams.append(self.create_training_workflow())
        
        # 5. Data Pipeline
        diagrams.append(self.create_data_pipeline())
        
        # 6. Loss Components
        diagrams.append(self.create_loss_components())
        
        # 7. Optimization Strategy
        diagrams.append(self.create_optimization_strategy())
        
        # 8. Mathematical Foundation
        diagrams.append(self.create_mathematical_foundation())
        
        # 9. Configuration Analysis
        diagrams.append(self.create_configuration_analysis())
        
        return diagrams
    
    def create_system_architecture(self) -> str:
        """Create enhanced system architecture diagram"""
        mermaid_code = """
        graph TB
            subgraph "📡 Input Data Pipeline"
                A[RadioMapSeer Dataset<br/>320×320 Radio Maps] --> B[RadioUNet_c Loader<br/>batch_size: 32]
                B --> C[Effective Batch: 32]
                C --> D[Input Tensors<br/>image: B×1×320×320<br/>cond: B×3×320×320]
            end
            
            subgraph "🎯 First Stage: VAE Encoder"
                E[AutoencoderKL<br/>embed_dim: 3] --> F[Encoder<br/>ResNet Architecture]
                F --> G[Latent Space z<br/>z~q_φ(z|x)<br/>Shape: [B, 3, 80, 80]]
                G --> H[16× Compression<br/>320×320 → 80×80<br/>Computational Efficiency]
            end
            
            subgraph "🔄 Second Stage: Conditional U-Net"
                I[Conditional U-Net<br/>dim: 128] --> J[Time Embedding<br/>Sinusoidal Encoding]
                J --> K[Condition Integration<br/>Swin Transformer<br/>Window-based Attention]
                K --> L[Multi-scale Features<br/>dim_mults: [1,2,4,4]]
                L --> M[Adaptive FFT Module<br/>Frequency Domain Enhancement]
                M --> N[Noise Prediction<br/>ε_θ(x_t, t, c)]
            end
            
            subgraph "🌊 Diffusion Process"
                O[Forward Diffusion<br/>q(x_t|x_0) = 𝒩(√ᾱ_tx_0, (1-ᾱ_t)𝐈)] --> P[Noise Schedule<br/>β_t: linear 0.0001→0.02]
                P --> Q[Reverse Process<br/>p_θ(x_0|x_t, c)]
                Q --> R[Knowledge-Aware Objective<br/>pred_KC]
            end
            
            subgraph "⚙️ Training Loop"
                S[L2 Loss Computation] --> T[Backpropagation<br/>Gradient Clipping: 1.0]
                T --> U[AdamW Optimizer<br/>lr: 1e-5, wd: 1e-4]
                U --> V[Cosine LR Schedule<br/>lr(t) = max(1e-6, 1e-5×(1-t/T)^0.96)]
                V --> W[EMA Model Update<br/>β: 0.999<br/>after 10,000 steps]
            end
            
            D --> E
            G --> I
            N --> S
            R --> S
            
            style A fill:#E3F2FD,stroke:#01579B,stroke-width:2px
            style E fill:#F3E5F5,stroke:#4A148C,stroke-width:2px
            style I fill:#E8F5E8,stroke:#1B5E20,stroke-width:2px
            style O fill:#FFF3E0,stroke:#E65100,stroke-width:2px
            style S fill:#FCE4EC,stroke:#880E4F,stroke-width:2px
        """
        
        return self.render_mermaid(mermaid_code, "system_architecture")
    
    def create_vae_architecture(self) -> str:
        """Create enhanced VAE architecture diagram"""
        mermaid_code = """
        graph TB
            subgraph "📥 Encoder Path"
                A[Input: 320×320×1<br/>Radio Map] --> B[Conv2D: 128×320×320]
                B --> C[ResNet Block 1<br/>Downsample: 160×160]
                C --> D[ResNet Block 2<br/>Downsample: 80×80]
                D --> E[Bottleneck<br/>Latent Space: 80×80×3]
            end
            
            subgraph "📤 Decoder Path"
                F[Latent: 80×80×3] --> G[ResNet Block 1<br/>Upsample: 160×160]
                G --> H[ResNet Block 2<br/>Upsample: 320×320]
                H --> I[Conv2D: 1×320×320<br/>Reconstruction]
            end
            
            subgraph "🔧 Regularization"
                J[KL Divergence<br/>D_KL(q_φ(z|x)‖p(z))] --> K[Reconstruction Loss<br/>‖x-x̂‖²]
                K --> L[Total Loss<br/>ELBO Optimization]
            end
            
            E --> F
            J --> L
            K --> L
            
            style A fill:#E3F2FD,stroke:#01579B,stroke-width:2px
            style F fill:#F3E5F5,stroke:#4A148C,stroke-width:2px
            style L fill:#FFF8E1,stroke:#F57F17,stroke-width:2px
        """
        
        return self.render_mermaid(mermaid_code, "vae_architecture")
    
    def create_conditional_unet(self) -> str:
        """Create enhanced conditional U-Net diagram"""
        mermaid_code = """
        graph LR
            subgraph "🎯 Input Processing"
                A[Noisy Latent: x_t<br/>80×80×3] --> B[Time Embedding<br/>Sinusoidal]
                C[Condition: Building<br/>80×80×3] --> D[Condition Encoder<br/>Swin Transformer]
                B --> E[Concatenation<br/>x_t + t + c]
                D --> E
            end
            
            subgraph "🔍 Multi-scale Processing"
                E --> F[Scale 1: 80×80×128<br/>Window: 8×8]
                F --> G[Scale 2: 40×40×256<br/>Window: 4×4]
                G --> H[Scale 3: 20×20×512<br/>Window: 2×2]
                H --> I[Scale 4: 10×10×512<br/>Window: 1×1]
            end
            
            subgraph "🌊 Bottleneck"
                I --> J[Adaptive FFT<br/>Frequency Domain]
                J --> K[Self-Attention<br/>Global Context]
                K --> L[Cross-Attention<br/>Condition Integration]
            end
            
            subgraph "📈 Decoder"
                L --> M[Scale 4: 10×10×512<br/>Upsample]
                M --> N[Scale 3: 20×20×512<br/>Upsample]
                N --> O[Scale 2: 40×40×256<br/>Upsample]
                O --> P[Scale 1: 80×80×128<br/>Final Output]
                P --> Q[Predicted Noise: ε_θ<br/>80×80×3]
            end
            
            style A fill:#E3F2FD,stroke:#01579B,stroke-width:2px
            style F fill:#F1F8E9,stroke:#33691E,stroke-width:2px
            style J fill:#FFF8E1,stroke:#F57F17,stroke-width:2px
            style Q fill:#FCE4EC,stroke:#880E4F,stroke-width:2px
        """
        
        return self.render_mermaid(mermaid_code, "conditional_unet")
    
    def create_training_workflow(self) -> str:
        """Create enhanced training workflow diagram"""
        mermaid_code = """
        sequenceDiagram
            participant Main as main()
            participant Config as radio_train_m.yaml
            participant VAE as AutoencoderKL
            participant UNet as Conditional U-Net
            participant LDM as LatentDiffusion
            participant Data as RadioUNet_c Dataset
            participant Trainer as Trainer Class
            
            Main->>Config: Load YAML Configuration
            Config->>Main: Model & Training Parameters
            
            Main->>VAE: Initialize First Stage
            VAE->>Main: Pre-trained VAE Encoder
            
            Main->>UNet: Initialize Conditional U-Net
            UNet->>Main: Swin Transformer Backbone
            
            Main->>LDM: Create LatentDiffusion Model
            LDM->>Main: Combined VAE + U-Net
            
            Main->>Data: Load RadioMapSeer Dataset
            Data->>Main: DataLoader with Batching
            
            Main->>Trainer: Initialize Training Loop
            Trainer->>Trainer: Setup Optimizer & EMA
            
            loop Training Steps (50,000)
                Trainer->>Data: Get Batch
                Data->>Trainer: image, cond, mask
                
                Trainer->>LDM: Forward Pass
                LDM->>Trainer: Loss & Predictions
                
                Trainer->>Trainer: Backpropagation
                Trainer->>Trainer: Gradient Clipping
                
                Trainer->>Trainer: Optimizer Step
                Trainer->>Trainer: LR Schedule Update
                
                Trainer->>Trainer: EMA Update
                
                alt Every 200 Steps
                    Trainer->>Trainer: Save Checkpoint
                    Trainer->>Trainer: Generate Samples
                end
            end
            
            Note over Main,Trainer: Training Complete: model-250.pt
        """
        
        return self.render_mermaid(mermaid_code, "training_workflow")
    
    def create_data_pipeline(self) -> str:
        """Create enhanced data pipeline diagram"""
        mermaid_code = """
        flowchart TD
            subgraph "📂 Data Loading"
                A[RadioMapSeer Dataset<br/>Real-world Radio Maps] --> B[RadioUNet_c Loader<br/>DPM Simulation]
                B --> C[Batch Creation<br/>batch_size: 32]
                C --> D[Data Augmentation<br/>Horizontal Flip]
            end
            
            subgraph "🔧 Input Processing"
                E[Image: 320×320×1<br/>Radio Map] --> F[Normalization<br/>[-1, 1]]
                G[Condition: 320×320×3<br/>Building Info] --> H[Multi-channel Encoding]
                I[Mask: Optional<br/>320×320×1] --> J[Binary Processing]
            end
            
            subgraph "📦 Batch Structure"
                K[Batch Dictionary] --> L[image: B×1×320×320<br/>Pathloss Values]
                K --> M[cond: B×3×320×320<br/>Building Layout]
                K --> N[ori_mask: B×1×320×320<br/>Spatial Mask]
                K --> O[img_name: List[str]<br/>Metadata]
            end
            
            subgraph "🎯 Forward Pass"
                P[VAE Encoder] --> Q[Latent Representation<br/>B×3×80×80]
                Q --> R[Conditional U-Net<br/>Noise Prediction]
                R --> S[Loss Computation<br/>L2 Distance]
            end
            
            A --> B
            D --> E
            D --> G
            D --> I
            F --> L
            H --> M
            J --> N
            L --> K
            M --> K
            N --> K
            K --> P
            Q --> R
            
            style A fill:#E8F5E8,stroke:#1B5E20,stroke-width:2px
            style K fill:#FFF3E0,stroke:#E65100,stroke-width:2px
            style P fill:#E3F2FD,stroke:#01579B,stroke-width:2px
            style S fill:#FCE4EC,stroke:#880E4F,stroke-width:2px
        """
        
        return self.render_mermaid(mermaid_code, "data_pipeline")
    
    def create_loss_components(self) -> str:
        """Create enhanced loss components diagram"""
        mermaid_code = """
        graph TB
            subgraph "🎯 Loss Components"
                A[Reconstruction Loss<br/>L2: ||x-x̂||²] --> E[Total Loss]
                B[KL Divergence<br/>KL(q‖p)] --> E
                C[Adversarial Loss<br/>GAN Training] --> E
                D[Perceptual Loss<br/>LPIPS] --> E
            end
            
            subgraph "⚖️ Loss Weights"
                F[λ_rec = 1.0] --> A
                G[λ_kl = 1e-6] --> B
                H[λ_adv = 0.5] --> C
                I[λ_perceptual = 0.0] --> D
            end
            
            subgraph "📊 Loss Dynamics"
                E --> J[L_total = Σλ_i·L_i]
                J --> K[Training Optimization]
                K --> L[Model Convergence]
            end
            
            style A fill:#E3F2FD,stroke:#01579B,stroke-width:2px
            style B fill:#F3E5F5,stroke:#4A148C,stroke-width:2px
            style C fill:#FFF3E0,stroke:#E65100,stroke-width:2px
            style D fill:#E8F5E8,stroke:#1B5E20,stroke-width:2px
            style J fill:#FCE4EC,stroke:#880E4F,stroke-width:2px
        """
        
        return self.render_mermaid(mermaid_code, "loss_components")
    
    def create_optimization_strategy(self) -> str:
        """Create enhanced optimization strategy diagram"""
        mermaid_code = """
        graph TB
            subgraph "📈 Cosine Decay Schedule"
                A[Initial LR: 1e-5] --> B[Warmup Phase<br/>Optional]
                B --> C[Decay Phase<br/>(1-t/T)^0.96]
                C --> D[Minimum LR: 1e-6<br/>Floor Value]
            end
            
            subgraph "🔧 Optimization Parameters"
                E[Optimizer: AdamW] --> F[Weight Decay: 1e-4]
                F --> G[Gradient Clipping: 1.0]
                G --> H[Accumulation: 1 step<br/>Effective Batch: 32]
            end
            
            subgraph "🎯 Model Regularization"
                I[EMA Update] --> J[Beta: 0.999]
                J --> K[Start After: 10,000 steps]
                K --> L[Update Every: 10 steps]
            end
            
            A --> E
            E --> I
            
            style A fill:#E3F2FD,stroke:#01579B,stroke-width:2px
            style E fill:#F1F8E9,stroke:#33691E,stroke-width:2px
            style I fill:#FFF8E1,stroke:#F57F17,stroke-width:2px
        """
        
        return self.render_mermaid(mermaid_code, "optimization_strategy")
    
    def create_mathematical_foundation(self) -> str:
        """Create enhanced mathematical foundation diagram"""
        mermaid_code = """
        graph TB
            subgraph "🧮 Diffusion Process"
                A[Forward Process<br/>q(x_t|x_0) = 𝒩(√ᾱ_tx_0, (1-ᾱ_t)𝐈)] --> B[Noise Schedule<br/>β_t = linear(0.0001, 0.02)]
                B --> C[Reverse Process<br/>p_θ(x_{t-1}|x_t, c)]
            end
            
            subgraph "🎯 Knowledge-Aware Objective"
                D[Objective: pred_KC<br/>𝔼[‖ε - ε_θ(x_t, t, c)‖²]] --> E[Radio Physics<br/>Integration]
            end
            
            subgraph "🏗️ VAE Formulation"
                F[Encoder: q_φ(z|x)] --> G[Decoder: p_θ(x|z)]
                G --> H[ELBO: 𝔼[log p_θ] - KL(q‖p)]
            end
            
            subgraph "🔄 Conditional Generation"
                I[Condition: c] --> J[Latent: z] --> K[Output: x]
                J --> L[p(x|c) = ∫p(x|z,c)p(z|c)dz]
            end
            
            style A fill:#E3F2FD,stroke:#01579B,stroke-width:2px
            style D fill:#F3E5F5,stroke:#4A148C,stroke-width:2px
            style F fill:#E8F5E8,stroke:#1B5E20,stroke-width:2px
            style I fill:#FFF3E0,stroke:#E65100,stroke-width:2px
        """
        
        return self.render_mermaid(mermaid_code, "mathematical_foundation")
    
    def create_configuration_analysis(self) -> str:
        """Create enhanced configuration analysis diagram"""
        mermaid_code = """
        graph LR
            subgraph "🤖 Model Configuration"
                A[Model Type: const_sde] --> B[Image Size: 320×320]
                B --> C[Timesteps: 1000]
                C --> D[Objective: pred_KC]
            end
            
            subgraph "🎛️ VAE Configuration"
                E[Embed Dim: 3] --> F[Channels: [1,128,256,512]]
                F --> G[KL Weight: 1e-6]
                G --> H[Disc Weight: 0.5]
            end
            
            subgraph "🔧 U-Net Configuration"
                I[Base Dim: 128] --> J[Dim Mults: [1,2,4,4]]
                J --> K[Window Sizes: [8×8,4×4,2×2,1×1]]
                K --> L[Fourier Scale: 16]
            end
            
            subgraph "⚙️ Training Configuration"
                M[Batch Size: 32] --> N[LR: 1e-5 → 1e-6]
                N --> O[Steps: 50,000]
                O --> P[Save Every: 200]
            end
            
            A --> E
            E --> I
            I --> M
            
            style A fill:#E3F2FD,stroke:#01579B,stroke-width:2px
            style E fill:#F3E5F5,stroke:#4A148C,stroke-width:2px
            style I fill:#E8F5E8,stroke:#1B5E20,stroke-width:2px
            style M fill:#FFF3E0,stroke:#E65100,stroke-width:2px
        """
        
        return self.render_mermaid(mermaid_code, "configuration_analysis")
    
    def render_mermaid(self, mermaid_code: str, diagram_name: str) -> str:
        """Render a mermaid diagram with enhanced configuration"""
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as mmd_file:
            mmd_file.write(mermaid_code)
            mmd_path = mmd_file.name
        
        # Create temporary CSS file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.css', delete=False) as css_file:
            css_file.write(self.css_style)
            css_path = css_file.name
        
        # Output path
        output_path = self.output_dir / f"{diagram_name}.png"
        
        # Build mmdc command
        cmd = [
            'mmdc',
            '-i', mmd_path,
            '-o', str(output_path),
            '-w', str(self.mmdc_config['width']),
            '-H', str(self.mmdc_config['height']),
            '-s', str(self.mmdc_config['scale']),
            '-t', self.mmdc_config['theme'],
            '-b', self.mmdc_config['background'],
            '--css-file', css_path,
            '--config-file', self.mmdc_config['fontFamily'],
            '--scale', str(self.mmdc_config['fontSize'])
        ]
        
        try:
            # Run mmdc
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if result.returncode == 0:
                print(f"✅ Successfully rendered {diagram_name}")
                return str(output_path)
            else:
                print(f"❌ Error rendering {diagram_name}: {result.stderr}")
                return ""
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Error rendering {diagram_name}: {e.stderr}")
            return ""
        except FileNotFoundError:
            print("❌ mmdc not found. Please install mermaid-cli: npm install -g @mermaid-js/mermaid-cli")
            return ""
        finally:
            # Clean up temporary files
            try:
                os.unlink(mmd_path)
                os.unlink(css_path)
            except:
                pass
    
    def render_all_diagrams(self) -> Dict[str, str]:
        """Render all enhanced diagrams"""
        print("🎨 Rendering enhanced mermaid diagrams...")
        
        diagrams = {}
        created_diagrams = self.create_enhanced_diagrams()
        
        for diagram_path in created_diagrams:
            if diagram_path:
                diagram_name = Path(diagram_path).stem
                diagrams[diagram_name] = diagram_path
        
        print(f"✅ Rendered {len(diagrams)} enhanced diagrams")
        return diagrams

def main():
    """Main function to render all enhanced diagrams"""
    renderer = EnhancedMermaidRenderer()
    diagrams = renderer.render_all_diagrams()
    
    print("\n📊 Enhanced Diagrams Created:")
    for name, path in diagrams.items():
        print(f"  - {name}: {path}")
    
    print(f"\n📁 Output directory: {renderer.output_dir}")
    print("🎉 All enhanced mermaid diagrams rendered successfully!")

if __name__ == "__main__":
    main()
