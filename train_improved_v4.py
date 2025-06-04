#!/usr/bin/env python3
"""
Improved Training Script for Cephalometric Landmark Detection - V4
MAJOR UPGRADES: 384x384 resolution + Online Hard Keypoint Mining Loss
"""

import os
import torch
import warnings
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.apis import init_model
from mmengine.runner import Runner
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')

# Apply PyTorch safe loading fix
import functools
_original_torch_load = torch.load

def safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = safe_torch_load

def plot_training_progress(work_dir):
    """Plot training progress from log files."""
    try:
        import json
        log_file = os.path.join(work_dir, "vis_data", "scalars.json")
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = [json.loads(line) for line in f]
            
            # Extract metrics
            train_loss = [log['loss'] for log in logs if 'loss' in log and log.get('mode') == 'train']
            val_nme = [log['NME'] for log in logs if 'NME' in log and log.get('mode') == 'val']
            
            if train_loss and val_nme:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                ax1.plot(train_loss)
                ax1.set_title('Training Loss')
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('Loss')
                ax1.grid(True)
                
                ax2.plot(val_nme)
                ax2.set_title('Validation NME')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('NME')
                ax2.grid(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join(work_dir, 'training_progress.png'), dpi=150)
                plt.close()
                print(f"Training progress plot saved to {work_dir}/training_progress.png")
    except Exception as e:
        print(f"Could not plot training progress: {e}")

def main():
    """Main improved training function with resolution and loss upgrades."""
    
    print("="*80)
    print("IMPROVED CEPHALOMETRIC TRAINING - V4")
    print("🚀 MAJOR UPGRADES: 384×384 Resolution + Hard Keypoint Mining")
    print("="*80)
    
    # Initialize MMPose scope
    init_default_scope('mmpose')
    
    # Import custom modules
    try:
        import custom_cephalometric_dataset
        import custom_transforms
        import cephalometric_dataset_info
        import custom_losses
        print("✓ Custom modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import custom modules: {e}")
        return
    
    # Configuration
    config_path = "Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py"
    work_dir = "work_dirs/hrnetv2_w18_cephalometric_384x384_adaptive_wing_loss_v4"  # New work dir for this experiment
    
    print(f"Config: {config_path}")
    print(f"Work Dir: {work_dir}")
    
    # Load config
    try:
        cfg = Config.fromfile(config_path)
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return
    
    # Set work directory
    cfg.work_dir = os.path.abspath(work_dir)
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # Load and prepare data
    data_file_path = "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json"
    print(f"Loading main data file from: {data_file_path}")
    
    try:
        main_df = pd.read_json(data_file_path)
        print(f"Main DataFrame loaded. Shape: {main_df.shape}")

        train_df = main_df[main_df['set'] == 'train'].reset_index(drop=True)
        val_df = main_df[main_df['set'] == 'dev'].reset_index(drop=True)
        test_df = main_df[main_df['set'] == 'test'].reset_index(drop=True)

        print(f"Train DataFrame shape: {train_df.shape}")
        print(f"Validation DataFrame shape: {val_df.shape}")
        print(f"Test DataFrame shape: {test_df.shape}")

        if train_df.empty or val_df.empty:
            print("ERROR: Training or validation DataFrame is empty.")
            return

        # Save DataFrames to temporary JSON files
        temp_train_ann_file = os.path.join(cfg.work_dir, 'temp_train_ann.json')
        temp_val_ann_file = os.path.join(cfg.work_dir, 'temp_val_ann.json')
        temp_test_ann_file = os.path.join(cfg.work_dir, 'temp_test_ann.json')

        train_df.to_json(temp_train_ann_file, orient='records', indent=2)
        val_df.to_json(temp_val_ann_file, orient='records', indent=2)
        if not test_df.empty:
            test_df.to_json(temp_test_ann_file, orient='records', indent=2)

        # Update config
        cfg.train_dataloader.dataset.ann_file = temp_train_ann_file
        cfg.train_dataloader.dataset.data_df = None
        cfg.train_dataloader.dataset.data_root = ''

        cfg.val_dataloader.dataset.ann_file = temp_val_ann_file
        cfg.val_dataloader.dataset.data_df = None
        cfg.val_dataloader.dataset.data_root = ''

        if not test_df.empty:
            cfg.test_dataloader.dataset.ann_file = temp_test_ann_file
        else:
            cfg.test_dataloader.dataset.ann_file = temp_val_ann_file
        cfg.test_dataloader.dataset.data_df = None
        cfg.test_dataloader.dataset.data_root = ''
        
        print("✓ Configuration updated with data files.")

    except Exception as e:
        print(f"ERROR: Failed to load or process data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print major upgrade information
    print("\n" + "="*70)
    print("🚀 MAJOR UPGRADES IN THIS VERSION")
    print("="*70)
    print(f"📐 Resolution Upgrade:")
    print(f"   • Input size: 256×256 → 384×384 (+50% resolution)")
    print(f"   • Heatmap size: 64×64 → 96×96 (+50% heatmap precision)")
    print(f"   • Expected: 10-20% MRE reduction from sub-pixel precision")
    
    print(f"\n🎯 Loss Function Upgrade:")
    print(f"   • Loss: KeypointMSELoss → AdaptiveWingLoss")
    print(f"   • Focus: Robust heatmap regression with adaptive behavior")
    print(f"   • Expected: Better handling of difficult landmarks (Sella/Gonion) and outliers")
    
    print(f"\n📊 Memory Management:")
    print(f"   • Batch size: 32 → 20 (reduced for 384×384)")
    print(f"   • Training will be slightly slower but more precise")
    
    # Print enhanced training parameters
    print("\n" + "="*60)
    print("ENHANCED TRAINING PARAMETERS")
    print("="*60)
    print(f"Optimizer: {cfg.optim_wrapper.optimizer.type}")
    print(f"Learning Rate: {cfg.optim_wrapper.optimizer.lr}")
    print(f"Batch Size: {cfg.train_dataloader.batch_size}")
    print(f"Max Epochs: {cfg.train_cfg.max_epochs}")
    print(f"Val Interval: {cfg.train_cfg.val_interval}")
    
    # Print scheduler info
    print(f"Learning Rate Schedule:")
    for i, scheduler in enumerate(cfg.param_scheduler):
        print(f"  {i+1}. {scheduler['type']}")
        if scheduler['type'] == 'LinearLR':
            print(f"     Warm-up: {scheduler['end']} iterations")
        elif scheduler['type'] == 'CosineAnnealingLR':
            print(f"     T_max: {scheduler['T_max']}, eta_min: {scheduler['eta_min']}")
    
    # Print joint weights for problematic landmarks
    joint_weights = cephalometric_dataset_info.dataset_info['joint_weights']
    print(f"\nLandmark Weights (targeting difficult landmarks):")
    landmark_names = cephalometric_dataset_info.landmark_names_in_order
    for i, (name, weight) in enumerate(zip(landmark_names, joint_weights)):
        if weight > 1.0:
            print(f"  {i:2d}. {name:<20} : {weight}x (enhanced)")
    
    print(f"\nAugmentation:")
    print(f"  • Rotation: ±30°")
    print(f"  • Scale range: 0.7-1.3")
    print(f"  • Horizontal flipping")
    
    # Build runner and start training
    try:
        print("\n" + "="*70)
        print("🚀 STARTING UPGRADED TRAINING")
        print("="*70)
        
        runner = Runner.from_cfg(cfg)
        
        # Enhanced monitoring message
        print("🎯 Training with major upgrades in progress...")
        print("📊 Monitor metrics for improvements:")
        print("🔹 Target Overall MRE: 2.7px → <2.5px")
        print("🔹 Target Sella error: 5.4px → <4.5px")
        print("🔹 Target Gonion error: 4.9px → <4.0px")
        print("🔹 Training: 60 epochs with validation every 2 epochs")
        print("🔹 The model will adapt your existing fine-tuned weights to higher resolution")
        
        print("\n⏱️  Starting training... (will take longer due to 384×384 resolution)")
        
        runner.train()
        
        print("\n🎉 Enhanced training completed successfully!")
        
        # Plot training progress
        plot_training_progress(cfg.work_dir)
        
    except Exception as e:
        print(f"\n💥 Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Final model validation
    print("\n" + "="*70)
    print("🏆 FINAL UPGRADED MODEL VALIDATION")
    print("="*70)
    
    try:
        import glob
        best_checkpoint = os.path.join(cfg.work_dir, "best_NME_epoch_*.pth")
        checkpoints = glob.glob(best_checkpoint)
        
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            print(f"🏅 Best checkpoint: {latest_checkpoint}")
            
            # Evaluation suggestions
            print("\n🎯 Training completed! Next steps:")
            print(f"1. 📊 Run detailed evaluation (update work_dir in script):")
            print(f"   python evaluate_detailed_metrics.py")
            print(f"2. 📈 Compare with V3 results:")
            print(f"   - V3 Overall MRE: 2.706 ± 1.949 pixels")
            print(f"   - V3 Sella error: 5.420 pixels")
            print(f"   - V3 Gonion error: 4.851 pixels")
            print(f"3. 🎨 Visualize improvements:")
            print(f"   python visualize_predictions.py")
            print(f"4. 🔍 Expected improvements:")
            print(f"   - Resolution boost: 10-20% error reduction")
            print(f"   - OHKM loss: Better handling of difficult landmarks")
            print(f"   - Combined: Target <2.5px overall MRE")
            
        else:
            print("⚠️  No best checkpoint found")
            
    except Exception as e:
        print(f"⚠️  Final validation setup failed: {e}")

if __name__ == "__main__":
    main() 