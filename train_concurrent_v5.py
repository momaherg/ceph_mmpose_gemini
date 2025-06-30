#!/usr/bin/env python3
"""
Concurrent MLP Training Script for Cephalometric Landmark Detection - V5
This script trains HRNetV2 with concurrent MLP refinement using custom hooks.
"""

import os
import sys
import torch
import warnings
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Add current directory to path for custom modules
sys.path.insert(0, os.getcwd())

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
    """Main concurrent training function."""
    
    parser = argparse.ArgumentParser(
        description='Concurrent MLP Training Script for Cephalometric Landmark Detection - V5')
    parser.add_argument(
        '--test_split_file',
        type=str,
        default=None,
        help='Path to a text file containing patient IDs for the test set, one ID per line.'
    )
    args = parser.parse_args()
    
    print("="*80)
    print("CONCURRENT MLP TRAINING - V5")
    print("🚀 HRNetV2 + On-the-fly MLP Refinement")
    print("="*80)
    
    # Initialize MMPose scope
    init_default_scope('mmpose')
    
    # Import custom modules (this must be done after init_default_scope)
    try:
        import custom_cephalometric_dataset
        import custom_transforms
        import cephalometric_dataset_info
        # Import the concurrent training hook
        import mlp_concurrent_training_hook
        print("✓ Custom modules imported successfully")
        print("✓ Concurrent MLP training hook imported")
    except ImportError as e:
        print(f"✗ Failed to import custom modules: {e}")
        return
    
    # Configuration
    config_path = "Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py"
    work_dir = "work_dirs/hrnetv2_w18_cephalometric_concurrent_mlp_v5"
    
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

        # Data splitting logic (same as train_improved_v4.py)
        if args.test_split_file:
            print(f"Splitting data using external test set file: {args.test_split_file}")
            with open(args.test_split_file, 'r') as f:
                test_patient_ids = {
                    int(line.strip())
                    for line in f if line.strip()
                }

            if 'patient_id' not in main_df.columns:
                print("ERROR: 'patient_id' column not found in the main DataFrame.")
                return
            
            main_df['patient_id'] = main_df['patient_id'].astype(int)

            test_df = main_df[main_df['patient_id'].isin(test_patient_ids)].reset_index(drop=True)
            remaining_df = main_df[~main_df['patient_id'].isin(test_patient_ids)]

            if len(remaining_df) >= 100:
                val_df = remaining_df.sample(n=100, random_state=42)
                train_df = remaining_df.drop(val_df.index).reset_index(drop=True)
                val_df = val_df.reset_index(drop=True)
            else:
                print(f"WARNING: Only {len(remaining_df)} patients remaining after selecting the test set.")
                print("Splitting the remaining data into 50% validation and 50% training.")
                if len(remaining_df) > 1:
                    val_df = remaining_df.sample(frac=0.5, random_state=42)
                    train_df = remaining_df.drop(val_df.index).reset_index(drop=True)
                    val_df = val_df.reset_index(drop=True)
                else:
                    val_df = remaining_df.reset_index(drop=True)
                    train_df = pd.DataFrame()
        else:
            print("Splitting data using 'set' column from the JSON file.")
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
    
    # Print concurrent training information
    print("\n" + "="*70)
    print("🚀 CONCURRENT MLP TRAINING APPROACH")
    print("="*70)
    print(f"🔄 Training Cycle:")
    print(f"   • Train HRNetV2 for 1 epoch")
    print(f"   • Run inference on training data with current HRNet weights")
    print(f"   • Train MLP models (X & Y) for 100 epochs on current predictions")
    print(f"   • Repeat for all {cfg.train_cfg.max_epochs} epochs")
    
    print(f"\n🧠 MLP Architecture:")
    print(f"   • Input: 19 predicted coordinates")
    print(f"   • Hidden: 500 neurons (ReLU + Dropout)")
    print(f"   • Output: 19 refined coordinates")
    print(f"   • Two separate models: one for X, one for Y coordinates")
    
    print(f"\n⚙️  Training Parameters:")
    print(f"   • HRNet epochs: {cfg.train_cfg.max_epochs}")
    print(f"   • MLP epochs per cycle: 100")
    print(f"   • MLP batch size: 16")
    print(f"   • MLP learning rate: 1e-5")
    print(f"   • MLP weight decay: 1e-4")
    
    print(f"\n🔒 Independence:")
    print(f"   • MLP gradients do NOT propagate back to HRNet")
    print(f"   • MLP parameters initialized once and persist across training")
    print(f"   • MLPs adapt dynamically to evolving HRNet predictions")
    
    # Check if custom_hooks exists in config
    if not hasattr(cfg, 'custom_hooks'):
        print("⚠️  Warning: custom_hooks not found in config. The hook should be automatically active.")
    else:
        print(f"✓ Custom hooks configured: {len(cfg.custom_hooks)} hook(s)")
        for i, hook in enumerate(cfg.custom_hooks):
            print(f"   {i+1}. {hook['type']}")
    
    # Build runner and start training
    try:
        print("\n" + "="*70)
        print("🚀 STARTING CONCURRENT TRAINING")
        print("="*70)
        
        runner = Runner.from_cfg(cfg)
        
        print("🎯 Concurrent training in progress...")
        print("📊 After each HRNet epoch, MLPs will be trained for 100 epochs")
        print("📈 Monitor logs for both HRNet and MLP training progress")
        print("⏱️  This will take significantly longer due to concurrent MLP training")
        
        runner.train()
        
        print("\n🎉 Concurrent training completed successfully!")
        
        # Plot training progress
        plot_training_progress(cfg.work_dir)
        
        # Check for saved MLP models
        mlp_dir = os.path.join(cfg.work_dir, "concurrent_mlp")
        if os.path.exists(mlp_dir):
            mlp_final_path = os.path.join(mlp_dir, "mlp_joint_final.pth")
            mlp_best_path = os.path.join(mlp_dir, "mlp_joint_best.pth")
            summary_path = os.path.join(mlp_dir, "best_model_summary.txt")
            
            if os.path.exists(mlp_final_path):
                print(f"✓ Concurrent joint MLP models saved:")
                print(f"   Final model: {mlp_final_path}")
                
                if os.path.exists(mlp_best_path):
                    print(f"   Best model: {mlp_best_path}")
                    
                    if os.path.exists(summary_path):
                        try:
                            with open(summary_path, 'r') as f:
                                summary = f.read()
                                for line in summary.split('\n'):
                                    if 'Best Epoch:' in line:
                                        epoch = line.split(':')[1].strip()
                                        print(f"   Best epoch: {epoch}")
                                    elif 'Best NME:' in line:
                                        nme = line.split(':')[1].strip()
                                        print(f"   Best NME: {nme}")
                        except:
                            pass
                    
                    print(f"✅ Best MLP model synchronized with best HRNetV2 checkpoint!")
                else:
                    print(f"⚠️  Best MLP model not found (validation may not have run)")
            else:
                print("⚠️  MLP models not found. Check hook execution.")
        else:
            print("⚠️  MLP directory not found. Check hook execution.")
        
    except Exception as e:
        print(f"\n💥 Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Final validation
    print("\n" + "="*70)
    print("🏆 CONCURRENT TRAINING COMPLETED")
    print("="*70)
    
    try:
        import glob
        best_checkpoint = os.path.join(cfg.work_dir, "best_NME_epoch_*.pth")
        checkpoints = glob.glob(best_checkpoint)
        
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            print(f"🏅 Best HRNet checkpoint: {latest_checkpoint}")
            
            print("\n🎯 Training completed! Key benefits expected:")
            print(f"1. 🔄 Dynamic adaptation: MLPs continuously adapt to HRNet evolution")
            print(f"2. 🎯 Overfitting mitigation: MLPs learn to correct HRNet intermediate errors")
            print(f"3. 🧠 Complex relationships: MLPs capture spatial dependencies between landmarks")
            print(f"4. 🚀 Two-stage refinement: HRNet predictions → MLP refinement")
            
            print(f"\n📋 Next steps:")
            print(f"1. 📊 Evaluate both HRNet and MLP models on test set")
            print(f"2. 🔍 Compare concurrent vs. sequential MLP training")
            print(f"3. 📈 Analyze improvement over baseline HRNet")
            print(f"4. 🎨 Visualize dynamic MLP adaptation over epochs")
            
        else:
            print("⚠️  No best checkpoint found")
            
    except Exception as e:
        print(f"⚠️  Final validation setup failed: {e}")

if __name__ == "__main__":
    main() 