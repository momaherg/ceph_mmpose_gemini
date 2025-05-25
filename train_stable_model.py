#!/usr/bin/env python3
"""
STABLE CEPHALOMETRIC TRAINING SCRIPT

This script uses the optimized configuration to prevent model collapse
and overfitting observed in the previous training runs.

Key Improvements:
- Drastically reduced learning rate (2e-5)
- Smaller batch size (8)
- Increased heatmap sigma (3.0)
- More frequent validation
- Early stopping
- Increased regularization
"""

import os
import os.path as osp
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import init_default_scope

def main():
    """Train the model with stable configuration."""
    
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
    
    print("="*60)
    print("STABLE CEPHALOMETRIC TRAINING")
    print("="*60)
    
    # Configuration
    config_path = 'configs/hrnetv2/hrnetv2_w18_cephalometric_224x224_STABLE.py'
    work_dir = 'work_dirs/hrnetv2_w18_cephalometric_STABLE'
    
    print(f"Config: {config_path}")
    print(f"Work Dir: {work_dir}")
    
    # Check if config exists
    if not osp.exists(config_path):
        print(f"✗ Config file not found: {config_path}")
        return
    
    # Load configuration
    try:
        cfg = Config.fromfile(config_path)
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return
    
    # Set work directory
    cfg.work_dir = work_dir
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    print(f"✓ Work directory: {osp.abspath(cfg.work_dir)}")
    
    # Print key training parameters
    print("\n📋 TRAINING PARAMETERS:")
    print(f"Learning Rate: {cfg.optim_wrapper.optimizer.lr}")
    print(f"Batch Size: {cfg.train_dataloader.batch_size}")
    print(f"Max Epochs: {cfg.train_cfg.max_epochs}")
    print(f"Validation Interval: {cfg.train_cfg.val_interval}")
    print(f"Weight Decay: {cfg.optim_wrapper.optimizer.weight_decay}")
    
    # Find heatmap sigma in pipeline
    for transform in cfg.train_dataloader.dataset.pipeline:
        if transform.get('type') == 'GenerateTarget':
            sigma = transform['encoder']['sigma']
            print(f"Heatmap Sigma: {sigma}")
            break
    
    # Build and start training
    try:
        print("\n🚀 Building runner...")
        runner = Runner.from_cfg(cfg)
        print("✓ Runner built successfully")
        
        print("\n🎯 Starting training...")
        print("Note: Training will stop early if validation performance plateaus")
        print("Expected improvements:")
        print("- More stable learning (no sudden performance drops)")
        print("- Better generalization (validation should track training)")
        print("- No prediction clustering")
        print("-" * 60)
        
        runner.train()
        
        print("✅ Training completed successfully!")
        print(f"Best checkpoint saved in: {cfg.work_dir}")
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n🎉 STABLE TRAINING FINISHED!")
    print("Check the work_dirs for:")
    print("- Training logs")
    print("- Best checkpoint")
    print("- Validation metrics")

if __name__ == "__main__":
    main() 