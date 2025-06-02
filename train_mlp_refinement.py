#!/usr/bin/env python3
"""
Training Script for MLP Refinement Network
Trains an MLP to refine HRNetV2 landmark predictions for better accuracy.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import warnings
import time
import glob
from typing import Dict, Optional
import matplotlib.pyplot as plt
from mmengine.registry import init_default_scope

# Import custom modules
from mlp_refinement_network import CephalometricMLPRefinement, create_model
from mlp_refinement_dataset import create_dataloaders, analyze_hrnet_predictions
import cephalometric_dataset_info

warnings.filterwarnings('ignore')

# Apply PyTorch safe loading fix
import functools
_original_torch_load = torch.load
def safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = safe_torch_load


class MLPRefinementTrainer:
    """Trainer for MLP refinement network."""
    
    def __init__(self,
                 model: CephalometricMLPRefinement,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 work_dir: str,
                 device: str = 'cuda:0'):
        """
        Args:
            model: MLP refinement model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            work_dir: Working directory for saving results
            device: Training device
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.work_dir = work_dir
        self.device = device
        
        # Create work directory
        os.makedirs(work_dir, exist_ok=True)
        
        # Training history
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_mre': [],
            'val_mre': [],
            'improvement': []
        }
        
        # Best model tracking
        self.best_val_mre = float('inf')
        self.best_epoch = 0
        
        print(f"Trainer initialized. Work dir: {work_dir}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def compute_mre(self, 
                   predictions: torch.Tensor, 
                   targets: torch.Tensor, 
                   valid_mask: torch.Tensor) -> float:
        """Compute Mean Radial Error for valid landmarks."""
        errors = torch.norm(predictions - targets, dim=-1)  # (B, num_landmarks)
        valid_errors = errors[valid_mask > 0]
        return valid_errors.mean().item() if len(valid_errors) > 0 else 0.0
    
    def train_epoch(self, optimizer: optim.Optimizer, 
                   loss_type: str = 'smooth_l1') -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_refinement_loss = 0.0
        total_initial_loss = 0.0
        total_mre_refined = 0.0
        total_mre_initial = 0.0
        num_batches = 0
        num_samples = 0
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move data to device
            images = batch['image'].to(self.device)              # (B, 3, H, W)
            hrnet_preds = batch['hrnet_predictions'].to(self.device)  # (B, 19, 2)
            targets = batch['ground_truth'].to(self.device)      # (B, 19, 2)
            valid_mask = batch['valid_mask'].to(self.device)     # (B, 19)
            
            # Forward pass
            predictions = self.model(images, hrnet_preds)
            
            # Compute loss
            losses = self.model.compute_loss(predictions, targets, loss_type)
            
            # Backward pass
            optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate metrics
            total_loss += losses['total_loss'].item()
            total_refinement_loss += losses['refinement_loss'].item()
            total_initial_loss += losses['initial_loss'].item()
            
            # Compute MRE for refined and initial predictions
            mre_refined = self.compute_mre(predictions['refined_predictions'], targets, valid_mask)
            mre_initial = self.compute_mre(predictions['initial_predictions'], targets, valid_mask)
            
            total_mre_refined += mre_refined
            total_mre_initial += mre_initial
            
            num_batches += 1
            num_samples += images.size(0)
            
            # Progress logging
            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_dataloader)}: "
                      f"Loss={losses['total_loss'].item():.4f}, "
                      f"MRE_refined={mre_refined:.3f}, "
                      f"MRE_initial={mre_initial:.3f}")
        
        return {
            'loss': total_loss / num_batches,
            'refinement_loss': total_refinement_loss / num_batches,
            'initial_loss': total_initial_loss / num_batches,
            'mre_refined': total_mre_refined / num_batches,
            'mre_initial': total_mre_initial / num_batches,
            'improvement': (total_mre_initial - total_mre_refined) / num_batches
        }
    
    def validate_epoch(self, loss_type: str = 'smooth_l1') -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_refinement_loss = 0.0
        total_initial_loss = 0.0
        total_mre_refined = 0.0
        total_mre_initial = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move data to device
                images = batch['image'].to(self.device)
                hrnet_preds = batch['hrnet_predictions'].to(self.device)
                targets = batch['ground_truth'].to(self.device)
                valid_mask = batch['valid_mask'].to(self.device)
                
                # Forward pass
                predictions = self.model(images, hrnet_preds)
                
                # Compute loss
                losses = self.model.compute_loss(predictions, targets, loss_type)
                
                # Accumulate metrics
                total_loss += losses['total_loss'].item()
                total_refinement_loss += losses['refinement_loss'].item()
                total_initial_loss += losses['initial_loss'].item()
                
                # Compute MRE
                mre_refined = self.compute_mre(predictions['refined_predictions'], targets, valid_mask)
                mre_initial = self.compute_mre(predictions['initial_predictions'], targets, valid_mask)
                
                total_mre_refined += mre_refined
                total_mre_initial += mre_initial
                
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'refinement_loss': total_refinement_loss / num_batches,
            'initial_loss': total_initial_loss / num_batches,
            'mre_refined': total_mre_refined / num_batches,
            'mre_initial': total_mre_initial / num_batches,
            'improvement': (total_mre_initial - total_mre_refined) / num_batches
        }
    
    def save_checkpoint(self, epoch: int, optimizer: optim.Optimizer, 
                       is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_mre': self.best_val_mre,
            'train_history': self.train_history
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.work_dir, f'epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.work_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"  ✓ New best model saved (MRE: {self.best_val_mre:.3f})")
    
    def plot_training_progress(self):
        """Plot training progress."""
        if len(self.train_history['epoch']) < 2:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = self.train_history['epoch']
        
        # Loss plot
        ax1.plot(epochs, self.train_history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.train_history['val_loss'], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MRE plot
        ax2.plot(epochs, self.train_history['train_mre'], 'b-', label='Train MRE')
        ax2.plot(epochs, self.train_history['val_mre'], 'r-', label='Val MRE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MRE (pixels)')
        ax2.set_title('Mean Radial Error')
        ax2.legend()
        ax2.grid(True)
        
        # Improvement plot
        ax3.plot(epochs, self.train_history['improvement'], 'g-', label='MRE Improvement')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Improvement (pixels)')
        ax3.set_title('MRE Improvement (Initial - Refined)')
        ax3.legend()
        ax3.grid(True)
        
        # Learning curve
        if len(epochs) > 5:
            recent_epochs = epochs[-10:]
            recent_val_mre = self.train_history['val_mre'][-10:]
            ax4.plot(recent_epochs, recent_val_mre, 'r-o', label='Recent Val MRE')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Validation MRE (pixels)')
            ax4.set_title('Recent Validation Performance')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.work_dir, 'training_progress.png'), dpi=150)
        plt.close()
    
    def train(self, 
              num_epochs: int = 50,
              learning_rate: float = 1e-3,
              weight_decay: float = 1e-4,
              loss_type: str = 'smooth_l1',
              scheduler_type: str = 'cosine'):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            loss_type: Loss function type ('mse', 'smooth_l1', 'huber')
            scheduler_type: Learning rate scheduler ('cosine', 'step', 'none')
        """
        print(f"\n🚀 Starting MLP Refinement Training")
        print(f"📊 Training Config:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Weight Decay: {weight_decay}")
        print(f"  Loss Type: {loss_type}")
        print(f"  Scheduler: {scheduler_type}")
        
        # Create optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Create scheduler
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs//3, gamma=0.1)
        else:
            scheduler = None
        
        print(f"\n🎯 Starting training loop...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_metrics = self.train_epoch(optimizer, loss_type)
            
            # Validation
            val_metrics = self.validate_epoch(loss_type)
            
            # Update learning rate
            if scheduler:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = learning_rate
            
            # Log metrics
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"MRE: {train_metrics['mre_refined']:.3f} "
                  f"(Δ: {train_metrics['improvement']:+.3f})")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"MRE: {val_metrics['mre_refined']:.3f} "
                  f"(Δ: {val_metrics['improvement']:+.3f})")
            print(f"LR: {current_lr:.6f}")
            
            # Update history
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_metrics['loss'])
            self.train_history['val_loss'].append(val_metrics['loss'])
            self.train_history['train_mre'].append(train_metrics['mre_refined'])
            self.train_history['val_mre'].append(val_metrics['mre_refined'])
            self.train_history['improvement'].append(val_metrics['improvement'])
            
            # Check if best model
            is_best = val_metrics['mre_refined'] < self.best_val_mre
            if is_best:
                self.best_val_mre = val_metrics['mre_refined']
                self.best_epoch = epoch + 1
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, optimizer, is_best)
            
            # Plot progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.plot_training_progress()
        
        total_time = time.time() - start_time
        print(f"\n🎉 Training completed!")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"🏆 Best validation MRE: {self.best_val_mre:.3f} pixels (epoch {self.best_epoch})")
        
        # Final plots
        self.plot_training_progress()
        
        # Save training history
        history_df = pd.DataFrame(self.train_history)
        history_df.to_csv(os.path.join(self.work_dir, 'training_history.csv'), index=False)


def main():
    """Main training function."""
    print("="*80)
    print("MLP REFINEMENT NETWORK TRAINING")
    print("🎯 Goal: Refine HRNetV2 predictions for better landmark accuracy")
    print("="*80)
    
    # Initialize MMPose scope
    init_default_scope('mmpose')
    
    # Import custom modules
    import custom_cephalometric_dataset
    import custom_transforms
    
    # Configuration
    config = {
        'data_file': "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json",
        'hrnet_config': "Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py",
        'hrnet_checkpoint_pattern': "work_dirs/hrnetv2_w18_cephalometric_384x384_adaptive_wing_loss_v4/epoch_54.pth",
        'work_dir': "work_dirs/mlp_refinement_v1",
        'input_size': 384,
        'batch_size': 16,  
        'num_epochs': 50,
        'learning_rate': 1e-3,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'force_cpu_hrnet': False  # Re-enable GPU for HRNetV2 now that image issues are fixed
    }
    
    print(f"📁 Data file: {config['data_file']}")
    print(f"🏗️  Work dir: {config['work_dir']}")
    print(f"🖥️  Device: {config['device']}")
    
    # Check if data file exists
    if not os.path.exists(config['data_file']):
        print(f"❌ Data file not found: {config['data_file']}")
        print("Please check the data file path.")
        return
    
    # Find HRNetV2 checkpoint
    if config['hrnet_checkpoint_pattern'].endswith('.pth'):
        # Direct path to checkpoint
        if os.path.exists(config['hrnet_checkpoint_pattern']):
            hrnet_checkpoint = config['hrnet_checkpoint_pattern']
        else:
            print(f"❌ HRNetV2 checkpoint not found: {config['hrnet_checkpoint_pattern']}")
            return
    else:
        # Pattern-based search
        checkpoints = glob.glob(config['hrnet_checkpoint_pattern'])
        if not checkpoints:
            print("❌ No HRNetV2 checkpoint found!")
            return
        hrnet_checkpoint = max(checkpoints, key=os.path.getctime)
    
    print(f"🔗 HRNetV2 checkpoint: {hrnet_checkpoint}")
    
    # Load data
    print("\n📊 Loading data...")
    main_df = pd.read_json(config['data_file'])
    train_df = main_df[main_df['set'] == 'train'].reset_index(drop=True)
    val_df = main_df[main_df['set'] == 'dev'].reset_index(drop=True)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Create dataloaders
    print("\n🔄 Creating dataloaders and extracting HRNetV2 predictions...")
    train_dataloader, val_dataloader = create_dataloaders(
        train_df, val_df,
        config['hrnet_config'], hrnet_checkpoint,
        input_size=config['input_size'],
        batch_size=config['batch_size'],
        cache_predictions=True,
        num_workers=0,  # Reduced for stability
        force_cpu=config['force_cpu_hrnet']  # Force CPU inference to avoid CUDA errors
    )
    
    # Analyze HRNetV2 baseline
    print("\n📈 Analyzing HRNetV2 baseline performance...")
    analysis_file = os.path.join(config['work_dir'], "hrnet_baseline_analysis.txt")
    os.makedirs(config['work_dir'], exist_ok=True)
    analyze_hrnet_predictions(val_dataloader, analysis_file)
    
    # Create model
    print("\n🧠 Creating MLP refinement model...")
    model_config = {
        'num_landmarks': 19,
        'input_size': config['input_size'],
        'image_feature_dim': 512,
        'landmark_hidden_dims': [256, 128, 64],
        'dropout': 0.3,
        'use_landmark_weights': True
    }
    
    model = create_model(model_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = MLPRefinementTrainer(
        model, train_dataloader, val_dataloader,
        config['work_dir'], config['device']
    )
    
    # Start training
    trainer.train(
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        loss_type='smooth_l1',
        scheduler_type='cosine'
    )
    
    print(f"\n✅ Training completed! Results saved to: {config['work_dir']}")
    print(f"🔍 Next steps:")
    print(f"  1. Run evaluation: python evaluate_mlp_refinement.py")
    print(f"  2. Compare with HRNetV2 baseline results")
    print(f"  3. Visualize improvements on challenging landmarks")


if __name__ == "__main__":
    main() 