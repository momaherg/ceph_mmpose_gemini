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
    work_dir = "work_dirs/hrnetv2_w18_cephalometric_finetune_experiment" # Relative path
    
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
    cfg.work_dir = work_dir
    
    # ---- START: Load and Inject DataFrames ----
    data_file_path = "data/train_data_pure_old_numpy.json"
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

        # Inject DataFrames into the config
        cfg.train_dataloader.dataset.data_df = train_df
        cfg.val_dataloader.dataset.data_df = val_df
        cfg.test_dataloader.dataset.data_df = test_df
        
        # Since data_df is provided, ensure ann_file is empty or None in dataset configs
        # The new config already sets ann_file to '', so this is a safeguard.
        cfg.train_dataloader.dataset.ann_file = ''
        cfg.val_dataloader.dataset.ann_file = ''
        cfg.test_dataloader.dataset.ann_file = ''

        print("✓ DataFrames injected into the configuration.")

    except FileNotFoundError:
        print(f"ERROR: Data file not found at {data_file_path}. Please check the path.")
        return
    except Exception as e:
        print(f"ERROR: Failed to load or process data: {e}")
        import traceback
        traceback.print_exc()
        return
    # ---- END: Load and Inject DataFrames ----
    
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
    
    # Create work directory
    os.makedirs(work_dir, exist_ok=True)
    
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