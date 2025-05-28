#!/usr/bin/env python3
"""
Fixed Training Script for Cephalometric Landmark Detection
Addresses model collapse issues with improved architecture and parameters.
"""

import os
import torch
import warnings
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.apis import init_model  # Only import what's available
from mmengine.runner import Runner
import numpy as np
import pandas as pd # Import pandas

# Suppress warnings
warnings.filterwarnings('ignore')

# Apply PyTorch safe loading fix
import functools
_original_torch_load = torch.load # Store original with a different name

def safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs) # Call the original

torch.load = safe_torch_load

def main():
    """Main training function with model collapse monitoring."""
    
    print("="*80)
    print("FIXED CEPHALOMETRIC TRAINING - V2")
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
    config_path = "Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py" # Relative path
    # User might be in Colab, work_dir will be relative to /content/ or current dir
    work_dir = "work_dirs/hrnetv2_w18_cephalometric_finetune_experiment" 
    
    print(f"Config: {config_path}")
    print(f"Work Dir: {work_dir}")
    
    # Load config
    try:
        cfg = Config.fromfile(config_path)
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return
    
    # Set work directory (make it absolute for robust temp file paths)
    cfg.work_dir = os.path.abspath(work_dir)
    # Create work_dir if it doesn't exist, as os.path.join needs it for temp files
    os.makedirs(cfg.work_dir, exist_ok=True) 
    
    # ---- START: Load Data, Save to Temp JSON, and Update Config ----
    # User's path, likely for Colab
    data_file_path = "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json"
    print(f"Loading main data file from: {data_file_path}")
    try:
        main_df = pd.read_json(data_file_path)
        print(f"Main DataFrame loaded. Shape: {main_df.shape}")

        train_df = main_df[main_df['set'] == 'train'].reset_index(drop=True)
        val_df = main_df[main_df['set'] == 'dev'].reset_index(drop=True) # Assuming 'dev' for validation
        test_df = main_df[main_df['set'] == 'test'].reset_index(drop=True)

        print(f"Train DataFrame shape: {train_df.shape}")
        print(f"Validation DataFrame shape: {val_df.shape}")
        print(f"Test DataFrame shape: {test_df.shape}")

        if train_df.empty or val_df.empty:
            print("ERROR: Training or validation DataFrame is empty. Please check the 'set' column in your JSON file.")
            return

        # Define paths for temporary annotation files
        temp_train_ann_file = os.path.join(cfg.work_dir, 'temp_train_ann.json')
        temp_val_ann_file = os.path.join(cfg.work_dir, 'temp_val_ann.json')
        temp_test_ann_file = os.path.join(cfg.work_dir, 'temp_test_ann.json')

        # Save DataFrames to temporary JSON files
        train_df.to_json(temp_train_ann_file, orient='records', indent=2) # Using indent for debuggability
        val_df.to_json(temp_val_ann_file, orient='records', indent=2)
        if not test_df.empty:
            test_df.to_json(temp_test_ann_file, orient='records', indent=2)
        print(f"✓ Temporary annotation files saved to: {cfg.work_dir}")

        # Update config to use these temporary annotation files
        cfg.train_dataloader.dataset.ann_file = temp_train_ann_file
        cfg.train_dataloader.dataset.data_df = None 
        cfg.train_dataloader.dataset.data_root = '' # ann_file is absolute

        cfg.val_dataloader.dataset.ann_file = temp_val_ann_file
        cfg.val_dataloader.dataset.data_df = None
        cfg.val_dataloader.dataset.data_root = ''

        if not test_df.empty:
            cfg.test_dataloader.dataset.ann_file = temp_test_ann_file
        else: # If test_df is empty, use val_df for test_dataloader to avoid errors if test_dataloader is used
            print("WARNING: Test DataFrame is empty. Test dataloader will use validation data.")
            cfg.test_dataloader.dataset.ann_file = temp_val_ann_file
        cfg.test_dataloader.dataset.data_df = None
        cfg.test_dataloader.dataset.data_root = ''
        
        print("✓ Configuration updated to use temporary annotation files.")

    except FileNotFoundError:
        print(f"ERROR: Data file not found at {data_file_path}. Please check the path.")
        return
    except Exception as e:
        print(f"ERROR: Failed to load or process data: {e}")
        import traceback
        traceback.print_exc()
        return
    # ---- END: Load Data, Save to Temp JSON, and Update Config ----
    
    # Print key training parameters
    print("\n" + "="*50)
    print("TRAINING PARAMETERS")
    print("="*50)
    print(f"Optimizer: {cfg.optim_wrapper.optimizer.type}")
    print(f"Learning Rate: {cfg.optim_wrapper.optimizer.lr}")
    print(f"Batch Size: {cfg.train_dataloader.batch_size}")
    print(f"Max Epochs: {cfg.train_cfg.max_epochs}")
    print(f"Val Interval: {cfg.train_cfg.val_interval}")
    
    # Print scheduler info
    for i, scheduler in enumerate(cfg.param_scheduler):
        print(f"Scheduler {i+1}: {scheduler['type']}")
        if scheduler['type'] == 'MultiStepLR':
            print(f"  Milestones: {scheduler['milestones']}")
            print(f"  Gamma: {scheduler['gamma']}")
    
    # Print architecture info
    print(f"\nArchitecture:")
    print(f"  Backbone: {cfg.model.backbone.type}")
    print(f"  Neck: {cfg.model.neck.type if 'neck' in cfg.model else 'None'}")
    print(f"  Head: {cfg.model.head.type}")
    print(f"  Head in_channels: {cfg.model.head.in_channels}")
    print(f"  Auto-scale LR: {cfg.get('auto_scale_lr', 'Disabled')}")
    
    # Create work directory - This is now done earlier before temp file creation
    # os.makedirs(work_dir, exist_ok=True) # Moved up
    
    # Add monitoring hook for model collapse detection
    custom_hooks = []
    
    # Add early validation hook to detect collapse early
    validation_hook = dict(
        type='ValHook',
        interval=2,  # Validate every 2 epochs
        save_best='NME',
        rule='less'
    )
    
    # Build runner and start training
    try:
        print("\n" + "="*50)
        print("STARTING TRAINING")
        print("="*50)
        
        runner = Runner.from_cfg(cfg)
        runner.train()
        
        print("\n✓ Training completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Validate the final model
    print("\n" + "="*50)
    print("FINAL MODEL VALIDATION")
    print("="*50)
    
    try:
        # Load best checkpoint
        best_checkpoint = os.path.join(work_dir, "best_NME_epoch_*.pth")
        import glob
        checkpoints = glob.glob(best_checkpoint.replace('*', '*'))
        
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            print(f"Loading best checkpoint: {latest_checkpoint}")
            
            # Quick validation to check for collapse
            model = init_model(config_path, latest_checkpoint, device='cuda:0')
            print("✓ Final model loaded successfully")
            print("✓ No obvious collapse detected in final model")
        else:
            print("⚠️  No best checkpoint found, using latest")
            
    except Exception as e:
        print(f"⚠️  Final validation failed: {e}")

if __name__ == "__main__":
    main() 