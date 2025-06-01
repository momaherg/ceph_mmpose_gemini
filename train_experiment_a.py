#!/usr/bin/env python3
"""
Training Script for Experiment A: AdaptiveWingOHKMHybridLoss
This experiment combines AdaptiveWingLoss with Online Hard Keypoint Mining
to improve accuracy on difficult landmarks while maintaining overall performance.
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
    """Main training function for Experiment A."""
    
    print("="*80)
    print("🧪 EXPERIMENT A: AdaptiveWingLoss + OHKM Hybrid")
    print("="*80)
    print("📊 Goal: Combine robustness of AdaptiveWing with hard example mining")
    print("🎯 Target: Reduce MRE below 2.3 pixels, especially for Sella/Gonion")
    print("="*80)
    
    # Initialize MMPose scope
    init_default_scope('mmpose')
    
    # Import custom modules
    try:
        import custom_cephalometric_dataset
        import custom_transforms
        import cephalometric_dataset_info
        import custom_losses  # Import our custom losses
        print("✓ Custom modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import custom modules: {e}")
        return
    
    # Configuration
    config_path = "configs/experiment_a_adaptive_wing_ohkm_hybrid.py"
    work_dir = "work_dirs/experiment_a_adaptive_wing_ohkm_hybrid"
    
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
    
    # Update load_from path if we have a checkpoint from V4
    import glob
    v4_checkpoints = glob.glob("work_dirs/hrnetv2_w18_cephalometric_384x384_adaptive_wing_loss_v4/best_NME_epoch_*.pth")
    if v4_checkpoints:
        latest_v4 = max(v4_checkpoints, key=os.path.getctime)
        cfg.load_from = latest_v4
        print(f"📥 Loading from V4 checkpoint: {latest_v4}")
    else:
        print("⚠️  No V4 checkpoint found, starting from scratch")
        cfg.load_from = None
    
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
    
    # Print experiment details
    print("\n" + "="*70)
    print("🔬 EXPERIMENT A: TECHNICAL DETAILS")
    print("="*70)
    print(f"📐 Resolution: 384×384 (maintained from V4)")
    print(f"🎯 Loss Function: AdaptiveWingOHKMHybridLoss")
    print(f"   • Base: AdaptiveWingLoss (robust to outliers)")
    print(f"   • Enhancement: Online Hard Keypoint Mining")
    print(f"   • Top-k: 8 hardest keypoints per sample")
    print(f"   • Hard weight: 2.0x for difficult landmarks")
    print(f"   • Expected benefit: Better focus on Sella/Gonion")
    
    print(f"\n📊 Hypothesis:")
    print(f"   • OHKM will identify and upweight difficult landmarks")
    print(f"   • AdaptiveWing provides stable gradients")
    print(f"   • Combined: Targeted improvement without regression")
    
    print(f"\n🎯 Success Metrics:")
    print(f"   • Overall MRE: <2.3 pixels (from 2.348)")
    print(f"   • Sella error: <4.2 pixels (from 4.674)")
    print(f"   • Gonion error: <3.8 pixels (from 4.281)")
    
    # Print training parameters
    print("\n" + "="*60)
    print("TRAINING PARAMETERS")
    print("="*60)
    print(f"Optimizer: {cfg.optim_wrapper.optimizer.type}")
    print(f"Learning Rate: {cfg.optim_wrapper.optimizer.lr}")
    print(f"Batch Size: {cfg.train_dataloader.batch_size}")
    print(f"Max Epochs: {cfg.train_cfg.max_epochs}")
    print(f"Val Interval: {cfg.train_cfg.val_interval}")
    
    # Build runner and start training
    try:
        print("\n" + "="*70)
        print("🚀 STARTING EXPERIMENT A")
        print("="*70)
        
        runner = Runner.from_cfg(cfg)
        
        print("🔬 Experiment A training in progress...")
        print("📊 Monitor for:")
        print("   • Stable loss convergence")
        print("   • Focus on hard keypoints")
        print("   • Improved accuracy on difficult landmarks")
        
        print("\n⏱️  Starting training...")
        
        runner.train()
        
        print("\n🎉 Experiment A training completed successfully!")
        
        # Plot training progress
        plot_training_progress(cfg.work_dir)
        
    except Exception as e:
        print(f"\n💥 Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Final notes
    print("\n" + "="*70)
    print("📊 EXPERIMENT A COMPLETED")
    print("="*70)
    
    try:
        best_checkpoint = os.path.join(cfg.work_dir, "best_NME_epoch_*.pth")
        checkpoints = glob.glob(best_checkpoint)
        
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            print(f"🏅 Best checkpoint: {latest_checkpoint}")
            
            print("\n🎯 Next steps:")
            print(f"1. 📊 Run detailed evaluation:")
            print(f"   python evaluate_experiment.py --experiment A")
            print(f"2. 📈 Compare with baseline:")
            print(f"   - V4 AdaptiveWing: 2.348 px MRE")
            print(f"   - Target: <2.3 px with OHKM enhancement")
            print(f"3. 🔍 Analyze per-landmark improvements:")
            print(f"   - Check if Sella/Gonion errors reduced")
            print(f"   - Verify no regression on easy landmarks")
            
        else:
            print("⚠️  No best checkpoint found")
            
    except Exception as e:
        print(f"⚠️  Final validation setup failed: {e}")

if __name__ == "__main__":
    main() 