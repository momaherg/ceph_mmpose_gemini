#!/usr/bin/env python3
"""
Quick Wins Training Script for Cephalometric Landmark Detection
Implements three quick improvements:
1. Increased joint weights (3.0x for Sella/Gonion)
2. UDP heatmap refinement for better coordinate accuracy
3. Test-time augmentation ensemble
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
                ax1.set_title('Training Loss (Quick Wins)')
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('Loss')
                ax1.grid(True)
                
                ax2.plot(val_nme)
                ax2.set_title('Validation NME (Quick Wins)')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('NME')
                ax2.grid(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join(work_dir, 'training_progress_quickwins.png'), dpi=150)
                plt.close()
                print(f"Training progress plot saved to {work_dir}/training_progress_quickwins.png")
    except Exception as e:
        print(f"Could not plot training progress: {e}")

def main():
    """Main quick wins training function."""
    
    print("="*80)
    print("QUICK WINS CEPHALOMETRIC TRAINING")
    print("Implementing: UDP + TTA + Enhanced Joint Weights")
    print("="*80)
    
    # Initialize MMPose scope
    init_default_scope('mmpose')
    
    # Import custom modules
    try:
        import custom_cephalometric_dataset
        import custom_transforms
        import cephalometric_dataset_info
        print("✓ Custom modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import custom modules: {e}")
        return
    
    # Configuration
    config_path = "Pretrained_model/hrnetv2_w18_cephalometric_256x256_quickwins.py"
    work_dir = "work_dirs/hrnetv2_w18_cephalometric_quickwins"
    
    print(f"Config: {config_path}")
    print(f"Work Dir: {work_dir}")
    
    # Load config
    try:
        cfg = Config.fromfile(config_path)
        print("✓ Quick wins configuration loaded successfully")
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
    
    # Print quick wins details
    print("\n" + "="*60)
    print("QUICK WINS IMPLEMENTATION DETAILS")
    print("="*60)
    
    print("QUICK WIN 1: Enhanced Joint Weights")
    joint_weights = cephalometric_dataset_info.dataset_info['joint_weights']
    landmark_names = cephalometric_dataset_info.landmark_names_in_order
    print(f"  - Sella weight: {joint_weights[0]}x (previous: 2.0x)")
    print(f"  - Gonion weight: {joint_weights[10]}x (previous: 2.0x)")
    print(f"  - PNS weight: {joint_weights[9]}x (unchanged)")
    
    print("\nQUICK WIN 2: UDP Heatmap Refinement")
    print(f"  - Codec: {cfg.codec.type}")
    print(f"  - UDP enabled: {cfg.codec.get('use_udp', False)}")
    print(f"  - Expected improvement: Better sub-pixel coordinate accuracy")
    
    print("\nQUICK WIN 3: Test-Time Augmentation")
    print(f"  - Flip test: {cfg.model.test_cfg.get('flip_test', False)}")
    print(f"  - Shift heatmap: {cfg.model.test_cfg.get('shift_heatmap', False)}")
    print(f"  - Expected improvement: Ensemble averaging for better predictions")
    
    print(f"\nTraining Parameters:")
    print(f"  - Optimizer: {cfg.optim_wrapper.optimizer.type}")
    print(f"  - Learning Rate: {cfg.optim_wrapper.optimizer.lr}")
    print(f"  - Max Epochs: {cfg.train_cfg.max_epochs}")
    print(f"  - Batch Size: {cfg.train_dataloader.batch_size}")
    
    # Build runner and start training
    try:
        print("\n" + "="*60)
        print("STARTING QUICK WINS TRAINING")
        print("="*60)
        
        runner = Runner.from_cfg(cfg)
        
        print("Expected improvements with quick wins:")
        print("  - Sella error: 5.4px → target <4.5px (3.0x weight)")
        print("  - Gonion error: 4.9px → target <4.0px (3.0x weight)")
        print("  - Overall MRE: 2.7px → target <2.3px (UDP + TTA)")
        print("  - Better coordinate precision with UDP codec")
        print("  - Ensemble averaging with test-time augmentation")
        
        runner.train()
        
        print("\n✓ Quick wins training completed successfully!")
        
        # Plot training progress
        plot_training_progress(cfg.work_dir)
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Final model validation
    print("\n" + "="*60)
    print("QUICK WINS MODEL VALIDATION")
    print("="*60)
    
    try:
        import glob
        best_checkpoint = os.path.join(cfg.work_dir, "best_NME_epoch_*.pth")
        checkpoints = glob.glob(best_checkpoint)
        
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            print(f"Best checkpoint: {latest_checkpoint}")
            
            print("\n✓ Quick wins training completed! Next steps:")
            print(f"1. Run evaluation with test-time augmentation:")
            print(f"   python evaluate_detailed_metrics_tta.py")
            print(f"2. Compare with previous results:")
            print(f"   - Previous Overall MRE: 2.706 ± 1.949 pixels")
            print(f"   - Previous Sella error: 5.420 pixels") 
            print(f"   - Previous Gonion error: 4.851 pixels")
            print(f"   - Target Overall MRE: <2.3 pixels")
            print(f"3. Expected improvements:")
            print(f"   - ~15-20% further MRE reduction")
            print(f"   - Better landmark localization precision")
            print(f"   - More robust predictions via ensemble")
            
        else:
            print("⚠️  No best checkpoint found")
            
    except Exception as e:
        print(f"⚠️  Final validation setup failed: {e}")

if __name__ == "__main__":
    main() 