#!/usr/bin/env python3
"""
Concurrent MLP Training Script for Cephalometric Landmark Detection - V5
This script trains HRNetV2 with optional concurrent MLP refinement using custom hooks.
Use --disable-mlp flag to train only HRNet without MLP refinement.
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
        '--disable-mlp',
        action='store_true',
        help='Disable concurrent MLP training and train only HRNet'
    )
    parser.add_argument(
        '--data-file',
        type=str,
        default="/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json",
        help='Path to the training data JSON file'
    )
    args = parser.parse_args()
    
    print("="*80)
    if args.disable_mlp:
        print("STANDARD HRNET TRAINING - V5")
        print("🎯 HRNetV2 Baseline Training")
    else:
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
        print("✓ Custom modules imported successfully")
        
        # Conditionally import the concurrent training hook
        if not args.disable_mlp:
            import mlp_concurrent_training_hook
            print("✓ Concurrent MLP training hook imported")
        else:
            print("⚠️  MLP training disabled - training HRNet only")
            
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
    data_file_path = args.data_file
    print(f"Loading main data file from: {data_file_path}")
    
    # Check if file exists first
    if not os.path.exists(data_file_path):
        print(f"ERROR: Data file not found at: {data_file_path}")
        
        # Look for alternative data files
        possible_paths = [
            "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json",
            "train_data_pure_old_numpy.json",
            "data/train_data_pure_old_numpy.json",
            "/content/drive/MyDrive/train_data_pure_old_numpy.json"
        ]
        
        print("Checking for alternative data file locations:")
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✓ Found data file at: {path}")
                data_file_path = path
                break
            else:
                print(f"✗ Not found: {path}")
        else:
            print("\nERROR: No data file found in any expected location.")
            print("Please ensure the data file exists and update the path accordingly.")
            return
    
    try:
        # Check file size first
        file_size = os.path.getsize(data_file_path)
        print(f"Data file size: {file_size:,} bytes")
        
        if file_size == 0:
            print("ERROR: Data file is empty")
            return
            
        # Try different JSON reading methods
        try:
            main_df = pd.read_json(data_file_path)
        except ValueError as e:
            print(f"Failed with pd.read_json: {e}")
            print("Trying alternative JSON loading method...")
            
            try:
                import json
                with open(data_file_path, 'r') as f:
                    data = json.load(f)
                main_df = pd.DataFrame(data)
                print("✓ Successfully loaded with json.load()")
            except Exception as e2:
                print(f"Failed with json.load: {e2}")
                print("Trying to read first few lines to diagnose the issue...")
                
                try:
                    with open(data_file_path, 'r') as f:
                        first_lines = [f.readline() for _ in range(5)]
                    print("First 5 lines of the file:")
                    for i, line in enumerate(first_lines, 1):
                        print(f"  {i}: {line.strip()[:100]}...")
                except Exception as e3:
                    print(f"Could not read file: {e3}")
                
                print("\nPlease check if the JSON file is properly formatted.")
                return
        
        print(f"Main DataFrame loaded. Shape: {main_df.shape}")
        
        # Check DataFrame structure
        print(f"DataFrame columns: {list(main_df.columns)[:10]}...")  # Show first 10 columns
        print(f"DataFrame dtypes: {main_df.dtypes.head()}")

        # Random data splitting logic: 200 test, 100 validation, rest training
        print("Splitting data randomly: 200 test samples, 100 validation samples, rest for training")
        
        # Check if we have enough samples
        total_samples = len(main_df)
        required_samples = 300  # 200 test + 100 validation
        
        if total_samples < required_samples:
            print(f"ERROR: Not enough samples for requested split.")
            print(f"Total samples: {total_samples}, Required: {required_samples} (200 test + 100 validation)")
            return
            
        print(f"Total samples available: {total_samples}")
        
        # Shuffle the entire dataframe with fixed random state for reproducibility
        shuffled_df = main_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split randomly: first 200 for test, next 100 for validation, rest for training
        test_df = shuffled_df.iloc[:200].reset_index(drop=True)
        val_df = shuffled_df.iloc[200:300].reset_index(drop=True)
        train_df = shuffled_df.iloc[300:].reset_index(drop=True)
        
        print(f"Random split completed:")
        print(f"  • Test set: {len(test_df)} samples")
        print(f"  • Validation set: {len(val_df)} samples") 
        print(f"  • Training set: {len(train_df)} samples")

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
    
    # Disable custom hooks if MLP training is disabled
    if args.disable_mlp and hasattr(cfg, 'custom_hooks'):
        cfg.custom_hooks = []
        print("✓ Custom hooks disabled (MLP training turned off)")
    
    # Print training information
    print("\n" + "="*70)
    if args.disable_mlp:
        print("🎯 STANDARD HRNET TRAINING")
        print("="*70)
        print(f"🔄 Training Cycle:")
        print(f"   • Train HRNetV2 for {cfg.train_cfg.max_epochs} epochs")
        print(f"   • Standard pose estimation training")
        print(f"   • No MLP refinement")
        
        print(f"\n⚙️  Training Parameters:")
        print(f"   • HRNet epochs: {cfg.train_cfg.max_epochs}")
        print(f"   • Standard MMPose training pipeline")
    else:
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
    if not hasattr(cfg, 'custom_hooks') or not cfg.custom_hooks:
        if args.disable_mlp:
            print("✓ No custom hooks configured (MLP training disabled)")
        else:
            print("⚠️  Warning: custom_hooks not found in config. The hook should be automatically active.")
    else:
        print(f"✓ Custom hooks configured: {len(cfg.custom_hooks)} hook(s)")
        for i, hook in enumerate(cfg.custom_hooks):
            print(f"   {i+1}. {hook['type']}")
    
    # Build runner and start training
    try:
        print("\n" + "="*70)
        if args.disable_mlp:
            print("🚀 STARTING STANDARD HRNET TRAINING")
        else:
            print("🚀 STARTING CONCURRENT TRAINING")
        print("="*70)
        
        runner = Runner.from_cfg(cfg)
        
        if args.disable_mlp:
            print("🎯 Standard HRNet training in progress...")
            print("📈 Monitor logs for HRNet training progress")
            print("⏱️  Standard training speed - no MLP overhead")
        else:
            print("🎯 Concurrent training in progress...")
            print("📊 After each HRNet epoch, MLPs will be trained for 100 epochs")
            print("📈 Monitor logs for both HRNet and MLP training progress")
            print("⏱️  This will take significantly longer due to concurrent MLP training")
        
        runner.train()
        
        if args.disable_mlp:
            print("\n🎉 HRNet training completed successfully!")
        else:
            print("\n🎉 Concurrent training completed successfully!")
        
        # Plot training progress
        plot_training_progress(cfg.work_dir)
        
        # Check for saved MLP models (only if MLP training was enabled)
        if not args.disable_mlp:
            mlp_dir = os.path.join(cfg.work_dir, "concurrent_mlp")
            if os.path.exists(mlp_dir):
                mlp_joint_path = os.path.join(mlp_dir, "mlp_joint_final.pth")
                if os.path.exists(mlp_joint_path):
                    print(f"✓ Concurrent MLP model saved:")
                    print(f"   Joint model: {mlp_joint_path}")
                else:
                    print("⚠️  MLP model not found. Check hook execution.")
            else:
                print("⚠️  MLP directory not found. Check hook execution.")
        else:
            print("✓ Standard HRNet training completed - no MLP models to save")
        
    except Exception as e:
        print(f"\n💥 Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Final validation
    print("\n" + "="*70)
    if args.disable_mlp:
        print("🏆 HRNET TRAINING COMPLETED")
    else:
        print("🏆 CONCURRENT TRAINING COMPLETED")
    print("="*70)
    
    try:
        import glob
        best_checkpoint = os.path.join(cfg.work_dir, "best_NME_epoch_*.pth")
        checkpoints = glob.glob(best_checkpoint)
        
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            print(f"🏅 Best HRNet checkpoint: {latest_checkpoint}")
            
            if args.disable_mlp:
                print("\n🎯 Standard HRNet training completed!")
                print(f"1. 🏗️  Baseline model: Standard HRNetV2 for cephalometric landmarks")
                print(f"2. 📊 Performance: Evaluate on test set for baseline metrics")
                
                print(f"\n📋 Next steps:")
                print(f"1. 📊 Evaluate HRNet model on test set")
                print(f"2. 📈 Compare with other baseline models")
                print(f"3. 🔍 Analyze landmark prediction accuracy")
                print(f"4. 🚀 Consider enabling MLP refinement for potential improvements")
            else:
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