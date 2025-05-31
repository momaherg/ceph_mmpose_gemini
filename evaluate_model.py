#!/usr/bin/env python3
"""
Evaluation script for the trained cephalometric landmark detection model.
"""

import os
import torch
import warnings
import pandas as pd
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.apis import init_model
from mmengine.runner import Runner
import glob

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
    """Main evaluation function."""
    
    print("="*80)
    print("CEPHALOMETRIC MODEL EVALUATION")
    print("="*80)
    
    # Initialize MMPose scope
    init_default_scope('mmpose')
    
    # Import custom modules
    import custom_cephalometric_dataset
    import custom_transforms
    import cephalometric_dataset_info
    
    # Configuration
    config_path = "Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py"
    work_dir = "work_dirs/hrnetv2_w18_cephalometric_finetune_experiment"
    
    # Load config
    cfg = Config.fromfile(config_path)
    cfg.work_dir = os.path.abspath(work_dir)
    
    # Find the best checkpoint
    checkpoint_pattern = os.path.join(cfg.work_dir, "best_NME_epoch_*.pth")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        print("No best checkpoint found. Looking for latest checkpoint...")
        checkpoint_pattern = os.path.join(cfg.work_dir, "epoch_*.pth")
        checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        print("ERROR: No checkpoints found in", cfg.work_dir)
        return
    
    # Get the latest/best checkpoint
    checkpoint_path = max(checkpoints, key=os.path.getctime)
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Load and prepare test data
    data_file_path = "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json"
    print(f"Loading data from: {data_file_path}")
    
    main_df = pd.read_json(data_file_path)
    test_df = main_df[main_df['set'] == 'test'].reset_index(drop=True)
    print(f"Test DataFrame shape: {test_df.shape}")
    
    if test_df.empty:
        print("WARNING: Test set is empty. Using validation set instead.")
        test_df = main_df[main_df['set'] == 'dev'].reset_index(drop=True)
    
    # Save test data to temporary file
    temp_test_ann_file = os.path.join(cfg.work_dir, 'temp_test_eval_ann.json')
    test_df.to_json(temp_test_ann_file, orient='records', indent=2)
    
    # Update config for test evaluation
    cfg.test_dataloader.dataset.ann_file = temp_test_ann_file
    cfg.test_dataloader.dataset.data_df = None
    cfg.test_dataloader.dataset.data_root = ''
    
    # Set load_from to the checkpoint
    cfg.load_from = checkpoint_path
    
    # Build the runner
    runner = Runner.from_cfg(cfg)
    
    # Run test evaluation
    print("\n" + "="*50)
    print("RUNNING TEST EVALUATION")
    print("="*50)
    
    metrics = runner.test()
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Additional analysis
    if 'NME' in metrics:
        nme_value = metrics['NME']
        print(f"\nNormalized Mean Error (NME): {nme_value:.4f}")
        print(f"This represents the average error normalized by the distance between landmarks {cfg.test_evaluator['keypoint_indices']}")
        print("(Indices 0 and 1 correspond to Sella and Nasion landmarks)")

if __name__ == "__main__":
    main() 