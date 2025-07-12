#!/usr/bin/env python3
"""
Ensemble Concurrent MLP Training Script for Cephalometric Landmark Detection - V5
This script trains 3 HRNetV2 models with concurrent MLP refinement using different train/val splits.
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

def create_ensemble_splits(remaining_df, val_size=100, n_splits=3, random_seed=42):
    """Create n different train/validation splits from remaining data."""
    splits = []
    
    print(f"\n🔀 Creating {n_splits} different train/validation splits...")
    print(f"📊 Remaining data size: {len(remaining_df)}")
    print(f"📋 Validation size per split: {val_size}")
    
    # Set different random seeds for each split to ensure diversity
    for i in range(n_splits):
        split_seed = random_seed + i * 1000  # Use different seeds for diversity
        
        if len(remaining_df) >= val_size * 2:  # Ensure enough data for meaningful splits
            # Sample validation set
            val_df = remaining_df.sample(n=val_size, random_state=split_seed)
            train_df = remaining_df.drop(val_df.index).reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
        else:
            print(f"⚠️  Warning: Limited data for split {i+1}. Using 50% for validation.")
            val_df = remaining_df.sample(frac=0.5, random_state=split_seed)
            train_df = remaining_df.drop(val_df.index).reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
        
        splits.append((train_df, val_df))
        print(f"Split {i+1}: Train={len(train_df)}, Val={len(val_df)}")
    
    return splits

def train_single_model(cfg, split_idx, train_df, val_df, test_df, base_work_dir):
    """Train a single model with the given train/val split."""
    
    # Create work directory for this split
    model_work_dir = os.path.join(base_work_dir, f"model_{split_idx}")
    cfg.work_dir = os.path.abspath(model_work_dir)
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🚀 TRAINING MODEL {split_idx} OF 3")
    print(f"{'='*70}")
    print(f"📁 Work directory: {cfg.work_dir}")
    print(f"📊 Train samples: {len(train_df)}")
    print(f"📊 Validation samples: {len(val_df)}")
    
    try:
        # Save DataFrames to temporary JSON files for this split
        temp_train_ann_file = os.path.join(cfg.work_dir, f'temp_train_ann_split_{split_idx}.json')
        temp_val_ann_file = os.path.join(cfg.work_dir, f'temp_val_ann_split_{split_idx}.json')
        temp_test_ann_file = os.path.join(cfg.work_dir, f'temp_test_ann_split_{split_idx}.json')

        train_df.to_json(temp_train_ann_file, orient='records', indent=2)
        val_df.to_json(temp_val_ann_file, orient='records', indent=2)
        if not test_df.empty:
            test_df.to_json(temp_test_ann_file, orient='records', indent=2)

        # Update config for this split
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
        
        print(f"✓ Configuration updated for model {split_idx}")

        # Build runner and start training
        print(f"🎯 Starting concurrent training for model {split_idx}...")
        print("📊 After each HRNet epoch, MLPs will be trained for 100 epochs")
        
        runner = Runner.from_cfg(cfg)
        runner.train()
        
        print(f"\n🎉 Model {split_idx} training completed successfully!")
        
        # Plot training progress for this model
        plot_training_progress(cfg.work_dir)
        
        # Check for saved MLP models
        mlp_dir = os.path.join(cfg.work_dir, "concurrent_mlp")
        if os.path.exists(mlp_dir):
            mlp_joint_path = os.path.join(mlp_dir, "mlp_joint_final.pth")
            if os.path.exists(mlp_joint_path):
                print(f"✓ Model {split_idx} MLP saved: {mlp_joint_path}")
            else:
                print(f"⚠️  Model {split_idx} MLP not found")
        
        return True
        
    except Exception as e:
        print(f"\n💥 Model {split_idx} training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main ensemble training function."""
    
    parser = argparse.ArgumentParser(
        description='Ensemble Concurrent MLP Training Script for Cephalometric Landmark Detection - V5')
    parser.add_argument(
        '--test_split_file',
        type=str,
        default=None,
        help='Path to a text file containing patient IDs for the test set, one ID per line.'
    )
    parser.add_argument(
        '--n_models',
        type=int,
        default=3,
        help='Number of models in the ensemble (default: 3)'
    )
    parser.add_argument(
        '--val_size',
        type=int,
        default=100,
        help='Validation set size for each split (default: 100)'
    )
    args = parser.parse_args()
    
    print("="*80)
    print("ENSEMBLE CONCURRENT MLP TRAINING - V5")
    print("🚀 Training 3 HRNetV2 + MLP models with different train/val splits")
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
        # Import the new model with classification head
        import hrnetv2_with_classification
        # Import ANB classification utilities
        import anb_classification_utils
        # Import classification evaluator
        import classification_evaluator
        print("✓ Custom modules imported successfully")
        print("✓ Concurrent MLP training hook imported")
        print("✓ HRNetV2 with classification model imported")
        print("✓ Classification evaluator imported")
    except ImportError as e:
        print(f"✗ Failed to import custom modules: {e}")
        return
    
    # Configuration
    config_path = "Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py"
    base_work_dir = "work_dirs/hrnetv2_w18_cephalometric_ensemble_concurrent_mlp_v5"
    
    print(f"Config: {config_path}")
    print(f"Base Work Dir: {base_work_dir}")
    
    # Load config
    try:
        cfg = Config.fromfile(config_path)
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return
    
    # Create base work directory
    os.makedirs(base_work_dir, exist_ok=True)
    
    # Load and prepare data
    data_file_path = "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json"
    print(f"Loading main data file from: {data_file_path}")
    
    try:
        main_df = pd.read_json(data_file_path)
        print(f"Main DataFrame loaded. Shape: {main_df.shape}")

        # Data splitting logic - Extract test set first
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

        else:
            print("Splitting data using 'set' column from the JSON file.")
            test_df = main_df[main_df['set'] == 'test'].reset_index(drop=True)
            remaining_df = main_df[main_df['set'] != 'test']  # Use both train and dev for ensemble splits

        print(f"Test DataFrame shape: {test_df.shape}")
        print(f"Remaining DataFrame shape (for ensemble splits): {remaining_df.shape}")

        if remaining_df.empty:
            print("ERROR: No remaining data for ensemble training.")
            return

        # Create ensemble splits from remaining data
        ensemble_splits = create_ensemble_splits(
            remaining_df, 
            val_size=args.val_size, 
            n_splits=args.n_models,
            random_seed=42
        )
        
        print(f"\n✓ Created {len(ensemble_splits)} ensemble splits")

    except Exception as e:
        print(f"ERROR: Failed to load or process data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print ensemble training information
    print("\n" + "="*70)
    print("🎯 ENSEMBLE CONCURRENT MLP TRAINING APPROACH")
    print("="*70)
    print(f"🔢 Number of models: {args.n_models}")
    print(f"📊 Validation size per model: {args.val_size}")
    print(f"🔄 Training approach per model:")
    print(f"   • Train HRNetV2 for 1 epoch")
    print(f"   • Run inference on training data with current HRNet weights")
    print(f"   • Train joint MLP model for 100 epochs on current predictions")
    print(f"   • Repeat for all {cfg.train_cfg.max_epochs} epochs")
    
    print(f"\n🧠 Joint MLP Architecture:")
    print(f"   • Input: 38 predicted coordinates (19 landmarks × 2)")
    print(f"   • Hidden: 500 neurons (ReLU + Dropout)")
    print(f"   • Output: 38 refined coordinates")
    print(f"   • Residual connections for stable training")
    
    print(f"\n⚙️  Training Parameters:")
    print(f"   • HRNet epochs: {cfg.train_cfg.max_epochs}")
    print(f"   • MLP epochs per cycle: 100")
    print(f"   • MLP batch size: 16")
    print(f"   • MLP learning rate: 1e-5")
    print(f"   • MLP weight decay: 1e-4")
    
    # Train each model in the ensemble
    successful_models = 0
    failed_models = []
    
    print(f"\n{'='*70}")
    print(f"🚀 STARTING ENSEMBLE TRAINING ({args.n_models} MODELS)")
    print(f"{'='*70}")
    
    for i, (train_df, val_df) in enumerate(ensemble_splits, 1):
        print(f"\n🎯 Preparing to train model {i}/{args.n_models}")
        
        # Create a fresh copy of the config for each model
        model_cfg = Config.fromfile(config_path)
        
        success = train_single_model(
            cfg=model_cfg,
            split_idx=i,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            base_work_dir=base_work_dir
        )
        
        if success:
            successful_models += 1
            print(f"✅ Model {i} completed successfully!")
        else:
            failed_models.append(i)
            print(f"❌ Model {i} failed!")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"🏆 ENSEMBLE TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"✅ Successful models: {successful_models}/{args.n_models}")
    
    if failed_models:
        print(f"❌ Failed models: {failed_models}")
    
    if successful_models > 0:
        print(f"\n📁 Model directories:")
        for i in range(1, args.n_models + 1):
            if i not in failed_models:
                model_dir = os.path.join(base_work_dir, f"model_{i}")
                print(f"   Model {i}: {model_dir}")
        
        print(f"\n🎯 Ensemble training completed! Key benefits:")
        print(f"1. 🔄 Diversity: {successful_models} models trained on different data splits")
        print(f"2. 🎯 Robustness: Ensemble predictions reduce overfitting")
        print(f"3. 🧠 Dynamic adaptation: Each MLP adapts to its HRNet evolution")
        print(f"4. 🚀 Two-stage refinement: HRNet predictions → MLP refinement")
        
        print(f"\n📋 Next steps:")
        print(f"1. 📊 Evaluate each model individually on test set")
        print(f"2. 🔄 Create ensemble predictions (average/voting)")
        print(f"3. 📈 Compare ensemble vs. individual model performance")
        print(f"4. 🎨 Analyze model diversity and complementarity")
        
    else:
        print("💥 All models failed to train. Check the error logs above.")

if __name__ == "__main__":
    main() 