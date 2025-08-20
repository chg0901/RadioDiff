import yaml
import argparse
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from denoising_diffusion_pytorch.ema import EMA
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from denoising_diffusion_pytorch.utils import *
import torchvision as tv
from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
from denoising_diffusion_pytorch.data import *
from torch.utils.data import DataLoader
from lib import loaders
from multiprocessing import cpu_count
import sys
import os
from pathlib import Path

# Add the datasets directory to the path
sys.path.append('./datasets')
from icassp2025_dataloader import create_icassp2025_dataloader, create_icassp2025_inference_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Training ICASSP2025 VAE configure")
    parser.add_argument("--cfg", help="experiment configure file name", type=str, required=True)
    parser.add_argument("--vae_type", help="VAE type (building, antenna, radio)", type=str, required=True)
    parser.add_argument("--mode", help="train or inference", type=str, default="train")
    parser.add_argument("--checkpoint", help="checkpoint path for inference", type=str, default=None)
    args = parser.parse_args()
    args.cfg = load_conf(args.cfg)
    return args


def load_conf(config_file, conf={}):
    with open(config_file) as f:
        exp_conf = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in exp_conf.items():
            conf[k] = v
    return conf


class VariableSizeVAETrainer:
    """Trainer for VAE with variable-size inference support"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Initialize accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            mixed_precision='fp16' if config['trainer'].get('fp16', False) else 'no',
            kwargs_handlers=[ddp_kwargs]
        )
        
        # Setup model, optimizer, and scheduler
        self.model, self.optimizer = self.setup_training()
        
        # Setup EMA
        self.ema = EMA(
            self.model.model,
            beta=config['trainer'].get('ema_decay', 0.995),
            update_after=config['trainer'].get('ema_update_every', 10)
        )
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.step = 0
        self.best_loss = float('inf')
        
    def setup_training(self):
        """Setup model, optimizer, and learning rate scheduler"""
        # Move model to device
        self.model = self.model.to(self.accelerator.device)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['trainer']['lr'],
            weight_decay=1e-6
        )
        
        # Setup learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['trainer']['train_num_steps'],
            eta_min=self.config['trainer']['min_lr']
        )
        
        # Prepare with accelerator
        self.model, optimizer, scheduler = self.accelerator.prepare(
            self.model, optimizer, scheduler
        )
        
        return self.model, optimizer
    
    def setup_logging(self):
        """Setup tensorboard logging"""
        results_folder = Path(self.config['trainer']['results_folder'])
        results_folder.mkdir(parents=True, exist_ok=True)
        
        if self.accelerator.is_main_process:
            self.writer = SummaryWriter(log_dir=str(results_folder / 'logs'))
            
            # Save configuration
            with open(results_folder / 'config.yaml', 'w') as f:
                yaml.dump(self.config, f)
    
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        
        # Get input images
        images = batch['image']
        
        # Forward pass
        reconstructions, posterior = self.model(images)
        
        # Compute loss
        rec_loss = torch.abs(images.contiguous() - reconstructions.contiguous())
        rec_loss = rec_loss.mean()
        
        # KLD loss
        kld_loss = posterior.kl()
        kld_loss = kld_loss.mean()
        
        # Total loss
        loss = rec_loss + self.config['model']['lossconfig']['kl_weight'] * kld_loss
        
        # Backward pass
        self.accelerator.backward(loss)
        
        # Gradient clipping
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Update EMA
        self.ema.update()
        
        # Step scheduler
        self.scheduler.step()
        
        return {
            'loss': loss.item(),
            'rec_loss': rec_loss.item(),
            'kld_loss': kld_loss.item()
        }
    
    def validate(self):
        """Validation step"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image']
                reconstructions, posterior = self.model(images)
                
                rec_loss = torch.abs(images.contiguous() - reconstructions.contiguous())
                rec_loss = rec_loss.mean()
                
                kld_loss = posterior.kl()
                kld_loss = kld_loss.mean()
                
                loss = rec_loss + self.config['model']['lossconfig']['kl_weight'] * kld_loss
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self):
        """Main training loop"""
        train_cfg = self.config['trainer']
        
        # Create progress bar
        progress_bar = tqdm(
            range(self.step, train_cfg['train_num_steps']),
            desc='Training',
            disable=not self.accelerator.is_main_process
        )
        
        while self.step < train_cfg['train_num_steps']:
            # Training epoch
            for batch in self.train_loader:
                # Training step
                metrics = self.train_step(batch)
                
                # Logging
                if self.step % train_cfg['log_freq'] == 0:
                    if self.accelerator.is_main_process:
                        self.writer.add_scalar('train/loss', metrics['loss'], self.step)
                        self.writer.add_scalar('train/rec_loss', metrics['rec_loss'], self.step)
                        self.writer.add_scalar('train/kld_loss', metrics['kld_loss'], self.step)
                        self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.step)
                        
                        # Print metrics
                        print(f"Step {self.step}: Loss={metrics['loss']:.4f}, Rec={metrics['rec_loss']:.4f}, KLD={metrics['kld_loss']:.4f}")
                
                # Validation
                if self.step % train_cfg['save_and_sample_every'] == 0:
                    val_loss = self.validate()
                    
                    if self.accelerator.is_main_process:
                        self.writer.add_scalar('val/loss', val_loss, self.step)
                        
                        # Save checkpoint
                        if val_loss < self.best_loss:
                            self.best_loss = val_loss
                            self.save_checkpoint('best_model.pth')
                        
                        # Save regular checkpoint
                        self.save_checkpoint(f'model_{self.step}.pth')
                
                self.step += 1
                progress_bar.update(1)
                
                if self.step >= train_cfg['train_num_steps']:
                    break
        
        # Save final checkpoint
        self.save_checkpoint('final_model.pth')
        
        if self.accelerator.is_main_process:
            self.writer.close()
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        if self.accelerator.is_main_process:
            checkpoint = {
                'model': self.accelerator.unwrap_model(self.model).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'step': self.step,
                'config': self.config
            }
            
            checkpoint_path = Path(self.config['trainer']['results_folder']) / filename
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.step = checkpoint['step']
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def inference_variable_size(self, input_images, original_sizes):
        """Inference with variable-size input support"""
        self.model.eval()
        
        with torch.no_grad():
            # Process each image with its original size
            reconstructions = []
            
            for img, orig_size in zip(input_images, original_sizes):
                # Resize model if needed (this would require a more sophisticated implementation)
                # For now, we'll assume the model can handle variable sizes through adaptive pooling
                img = img.unsqueeze(0).to(self.accelerator.device)
                
                # Forward pass
                recon, _ = self.model(img)
                reconstructions.append(recon.squeeze(0).cpu())
            
            return reconstructions


def main(args):
    cfg = args.cfg
    vae_type = args.vae_type
    mode = args.mode
    
    print(f"Starting {mode} for {vae_type} VAE...")
    
    # Create model
    model_cfg = cfg['model']
    model = AutoencoderKL(
        ddconfig=model_cfg['ddconfig'],
        lossconfig=model_cfg['lossconfig'],
        embed_dim=model_cfg['embed_dim'],
        ckpt_path=model_cfg['ckpt_path'],
    )
    
    # Create data loaders
    data_cfg = cfg["data"]
    
    if mode == "train":
        # Training dataloaders
        train_loader = create_icassp2025_dataloader(
            data_root=data_cfg['data_root'],
            crop_size=data_cfg.get('crop_size', 96),
            tx_margin=data_cfg.get('tx_margin', 10),
            batch_size=data_cfg['batch_size'],
            vae_type=vae_type,
            split='train',
            num_workers=4,
            shuffle=True
        )
        
        val_loader = create_icassp2025_dataloader(
            data_root=data_cfg['data_root'],
            crop_size=data_cfg.get('crop_size', 96),
            tx_margin=data_cfg.get('tx_margin', 10),
            batch_size=data_cfg['batch_size'],
            vae_type=vae_type,
            split='val',
            num_workers=4,
            shuffle=False
        )
        
        # Create trainer and start training
        trainer = VariableSizeVAETrainer(model, train_loader, val_loader, cfg)
        trainer.train()
        
    elif mode == "inference":
        # Inference mode
        if args.checkpoint is None:
            raise ValueError("Checkpoint path is required for inference mode")
        
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        
        # Create inference dataloader
        inference_loader = create_icassp2025_inference_dataloader(
            data_root=data_cfg['data_root'],
            vae_type=vae_type,
            batch_size=1,
            num_workers=4
        )
        
        # Create trainer for inference
        trainer = VariableSizeVAETrainer(model, None, None, cfg)
        trainer.load_checkpoint(args.checkpoint)
        
        # Run inference
        print("Running inference...")
        results = []
        
        for batch in tqdm(inference_loader, desc="Inference"):
            images = batch['image']
            original_sizes = batch['original_size']
            filenames = batch['filename']
            
            reconstructions = trainer.inference_variable_size(images, original_sizes)
            
            # Save results
            results_dir = Path(cfg['trainer']['results_folder']) / 'inference_results'
            results_dir.mkdir(parents=True, exist_ok=True)
            
            for i, (recon, filename) in enumerate(zip(reconstructions, filenames)):
                save_path = results_dir / f"recon_{filename}"
                tv.utils.save_image(recon, save_path)
        
        print(f"Inference completed. Results saved to {results_dir}")
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    args = parse_args()
    main(args)