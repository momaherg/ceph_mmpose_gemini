#!/usr/bin/env python3
"""
Experiment 1: FocalHeatmapLoss + OHKM for Cephalometric Landmark Detection
Combines focal loss for sharper heatmap peaks with online hard keypoint mining
"""

import os
import sys
sys.path.append(os.path.abspath('../..'))  # Add parent directory to path

import torch
import warnings
from mmengine.config import Config
from mmengine.registry import init_default_scope
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

def main():
    """Main training function for FocalHeatmap + OHKM experiment."""
    
    print("="*80)
    print("EXPERIMENT 1: FocalHeatmapLoss + OHKM")
    print("🎯 Combined loss for sharper peaks and hard example mining")
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
    config_path = "experiments/exp1_focal_ohkm/config_focal_ohkm.py"
    work_dir = "work_dirs/exp1_focal_ohkm_384x384"
    
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
    print("🧪 EXPERIMENT DETAILS")
    print("="*70)
    print(f"📐 Resolution: 384×384 → 96×96 heatmaps")
    print(f"🎯 Loss Configuration:")
    print(f"   • FocalHeatmapLoss (70% weight):")
    print(f"     - Alpha: 2 (balancing factor)")
    print(f"     - Gamma: 4 (focusing parameter)")
    print(f"     - Creates sharper, more focused heatmap peaks")
    print(f"   • OHKMMSELoss (30% weight):")
    print(f"     - TopK: 5 (focus on 5 hardest keypoints)")
    print(f"     - Helps with difficult landmarks (Sella, Gonion)")
    print(f"📊 Expected Benefits:")
    print(f"   • Sharper heatmap peaks → better sub-pixel precision")
    print(f"   • Hard example mining → improved difficult landmarks")
    print(f"   • Target: <2.3px overall MRE")
    
    # Build runner and start training
    try:
        print("\n" + "="*70)
        print("🚀 STARTING EXPERIMENT 1: FOCAL + OHKM")
        print("="*70)
        
        runner = Runner.from_cfg(cfg)
        
        print("🎯 Training with combined loss in progress...")
        print("📊 Monitor for:")
        print("   • Faster convergence on difficult landmarks")
        print("   • Better sub-pixel accuracy overall")
        print("   • More stable training with hard example focus")
        
        runner.train()
        
        print("\n🎉 Experiment 1 completed successfully!")
        
        # Save experiment summary
        with open(os.path.join(cfg.work_dir, 'experiment_summary.txt'), 'w') as f:
            f.write("EXPERIMENT 1: FocalHeatmapLoss + OHKM\n")
            f.write("="*50 + "\n")
            f.write("Loss Configuration:\n")
            f.write("- FocalHeatmapLoss (70%): alpha=2, gamma=4\n")
            f.write("- OHKMMSELoss (30%): topk=5\n")
            f.write("Resolution: 384×384\n")
            f.write("Batch Size: 20\n")
            f.write("Max Epochs: 60\n")
            f.write("Learning Rate: 2e-4 with cosine annealing\n")
        
    except Exception as e:
        print(f"\n💥 Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✅ Next: Run evaluate_detailed_metrics.py to analyze results")

if __name__ == "__main__":
    main() 