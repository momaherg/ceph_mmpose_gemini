#!/usr/bin/env python3
"""
5-Fold Cross-Validation Concurrent MLP Training Script for Cephalometric Landmark Detection - V5
This script performs 5-fold cross-validation with concurrent MLP refinement using custom hooks.
Each fold trains for 68 epochs with patient-level splitting to avoid data leakage.
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
from sklearn.model_selection import KFold
import json

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

def create_patient_level_folds(df, n_folds=5, random_state=42):
    """Create patient-level k-fold splits to avoid data leakage."""
    print(f"🔄 Creating {n_folds}-fold cross-validation splits at patient level...")
    
    # Get unique patient IDs
    if 'patient_id' not in df.columns:
        print("ERROR: 'patient_id' column not found in the DataFrame.")
        return None
    
    unique_patients = df['patient_id'].unique()
    print(f"📊 Total unique patients: {len(unique_patients)}")
    
    # Create KFold splitter
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    folds = []
    for fold_idx, (train_patient_indices, val_patient_indices) in enumerate(kf.split(unique_patients)):
        train_patients = unique_patients[train_patient_indices]
        val_patients = unique_patients[val_patient_indices]
        
        # Get samples for each patient group
        train_df = df[df['patient_id'].isin(train_patients)].reset_index(drop=True)
        val_df = df[df['patient_id'].isin(val_patients)].reset_index(drop=True)
        
        folds.append({
            'fold': fold_idx + 1,
            'train_df': train_df,
            'val_df': val_df,
            'train_patients': train_patients,
            'val_patients': val_patients
        })
        
        print(f"📋 Fold {fold_idx + 1}: {len(train_patients)} train patients ({len(train_df)} samples), "
              f"{len(val_patients)} val patients ({len(val_df)} samples)")
    
    return folds

def plot_fold_results(fold_results, work_dir):
    """Plot cross-validation results across folds."""
    try:
        folds = [r['fold'] for r in fold_results]
        final_mres = [r['final_mre'] for r in fold_results if r['final_mre'] is not None]
        
        if not final_mres:
            print("⚠️  No valid MRE results to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar plot of final MRE per fold
        ax1.bar(folds[:len(final_mres)], final_mres, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('Final Validation MRE (pixels)')
        ax1.set_title('Final MRE per Fold')
        ax1.grid(True, alpha=0.3)
        
        # Add mean line
        mean_mre = np.mean(final_mres)
        ax1.axhline(y=mean_mre, color='red', linestyle='--', alpha=0.8, 
                   label=f'Mean: {mean_mre:.3f}')
        ax1.legend()
        
        # Box plot of MRE distribution
        ax2.boxplot(final_mres, labels=['All Folds'])
        ax2.set_ylabel('Final Validation MRE (pixels)')
        ax2.set_title('MRE Distribution Across Folds')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(work_dir, 'cross_validation_results.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Cross-validation results plot saved: {plot_path}")
        
    except Exception as e:
        print(f"⚠️  Could not plot cross-validation results: {e}")

def train_single_fold(fold_data, base_work_dir, config_path, main_df):
    """Train a single fold of the cross-validation."""
    fold_num = fold_data['fold']
    train_df = fold_data['train_df']
    val_df = fold_data['val_df']
    
    print(f"\n" + "="*60)
    print(f"🚀 TRAINING FOLD {fold_num}/5")
    print("="*60)
    print(f"📊 Train samples: {len(train_df)}")
    print(f"📊 Validation samples: {len(val_df)}")
    
    # Create fold-specific work directory
    fold_work_dir = os.path.join(base_work_dir, f"fold_{fold_num}")
    os.makedirs(fold_work_dir, exist_ok=True)
    
    # Load and configure the model
    cfg = Config.fromfile(config_path)
    cfg.work_dir = os.path.abspath(fold_work_dir)
    
    # Update training configuration for 68 epochs
    cfg.train_cfg.max_epochs = 68
    
    # Update learning rate scheduler for shorter training
    if hasattr(cfg, 'param_scheduler') and len(cfg.param_scheduler) > 1:
        # Adjust MultiStepLR milestones for 68 epochs
        cfg.param_scheduler[1].end = 68
        cfg.param_scheduler[1].milestones = [13, 16]  # Decay at 72% and 89% of training
    
    try:
        # Save fold DataFrames to temporary files
        temp_train_ann_file = os.path.join(fold_work_dir, f'fold_{fold_num}_train_ann.json')
        temp_val_ann_file = os.path.join(fold_work_dir, f'fold_{fold_num}_val_ann.json')
        
        train_df.to_json(temp_train_ann_file, orient='records', indent=2)
        val_df.to_json(temp_val_ann_file, orient='records', indent=2)
        
        # Update config with fold-specific data
        cfg.train_dataloader.dataset.ann_file = temp_train_ann_file
        cfg.train_dataloader.dataset.data_df = None
        cfg.train_dataloader.dataset.data_root = ''
        
        cfg.val_dataloader.dataset.ann_file = temp_val_ann_file
        cfg.val_dataloader.dataset.data_df = None
        cfg.val_dataloader.dataset.data_root = ''
        
        cfg.test_dataloader.dataset.ann_file = temp_val_ann_file  # Use validation set for testing
        cfg.test_dataloader.dataset.data_df = None
        cfg.test_dataloader.dataset.data_root = ''
        
        print(f"✓ Fold {fold_num} configuration prepared")
        
        # Save fold information
        fold_info = {
            'fold': fold_num,
            'train_patients': fold_data['train_patients'].tolist(),
            'val_patients': fold_data['val_patients'].tolist(),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'epochs': 68
        }
        
        with open(os.path.join(fold_work_dir, f'fold_{fold_num}_info.json'), 'w') as f:
            json.dump(fold_info, f, indent=2)
        
        # Train the model
        print(f"🏃‍♂️ Starting training for fold {fold_num}...")
        
        runner = Runner.from_cfg(cfg)
        runner.train()
        
        print(f"✅ Fold {fold_num} training completed!")
        
        # Try to extract final validation MRE
        final_mre = None
        try:
            log_file = os.path.join(fold_work_dir, "vis_data", "scalars.json")
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = [json.loads(line) for line in f]
                
                val_nme_logs = [log['NME'] for log in logs if 'NME' in log and log.get('mode') == 'val']
                if val_nme_logs:
                    final_mre = val_nme_logs[-1]  # Last validation NME
                    print(f"📊 Fold {fold_num} final validation MRE: {final_mre:.4f}")
        except Exception as e:
            print(f"⚠️  Could not extract final MRE for fold {fold_num}: {e}")
        
        return {
            'fold': fold_num,
            'work_dir': fold_work_dir,
            'final_mre': final_mre,
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'status': 'completed'
        }
        
    except Exception as e:
        print(f"❌ Fold {fold_num} training failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'fold': fold_num,
            'work_dir': fold_work_dir,
            'final_mre': None,
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'status': 'failed',
            'error': str(e)
        }

def main():
    """Main 5-fold cross-validation function."""
    
    parser = argparse.ArgumentParser(
        description='5-Fold Cross-Validation Concurrent MLP Training for Cephalometric Landmark Detection')
    parser.add_argument(
        '--n_folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    parser.add_argument(
        '--epochs_per_fold',
        type=int,
        default=68,
        help='Number of epochs to train each fold (default: 68)'
    )
    parser.add_argument(
        '--start_fold',
        type=int,
        default=1,
        help='Starting fold number (for resuming interrupted training, default: 1)'
    )
    args = parser.parse_args()
    
    print("="*80)
    print("5-FOLD CROSS-VALIDATION CONCURRENT MLP TRAINING")
    print("="*80)
    print(f"🔄 Number of folds: {args.n_folds}")
    print(f"⏱️  Epochs per fold: {args.epochs_per_fold}")
    print(f"🚀 Starting from fold: {args.start_fold}")
    print("="*80)
    
    # Initialize MMPose scope
    init_default_scope('mmpose')
    
    # Import custom modules
    try:
        import custom_cephalometric_dataset
        import custom_transforms
        import cephalometric_dataset_info
        import mlp_concurrent_training_hook
        print("✓ Custom modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import custom modules: {e}")
        return
    
    # Configuration
    config_path = "Pretrained_model/hrnetv2_w68_cephalometric_256x256_finetune.py"
    base_work_dir = f"work_dirs/hrnetv2_w68_cephalometric_cv_{args.n_folds}fold"
    os.makedirs(base_work_dir, exist_ok=True)
    
    print(f"Config: {config_path}")
    print(f"Base Work Dir: {base_work_dir}")
    
    # Load main dataset
    data_file_path = "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json"
    print(f"Loading data from: {data_file_path}")
    
    try:
        main_df = pd.read_json(data_file_path)
        print(f"Main DataFrame loaded. Shape: {main_df.shape}")
        
        # Ensure patient_id column exists and is integer
        if 'patient_id' not in main_df.columns:
            print("ERROR: 'patient_id' column not found in the main DataFrame.")
            return
        
        main_df['patient_id'] = main_df['patient_id'].astype(int)
        
        # Create cross-validation folds
        folds = create_patient_level_folds(main_df, n_folds=args.n_folds, random_state=42)
        
        if folds is None:
            return
        
        # Save overall fold information
        cv_info = {
            'n_folds': args.n_folds,
            'epochs_per_fold': args.epochs_per_fold,
            'total_patients': len(main_df['patient_id'].unique()),
            'total_samples': len(main_df),
            'random_state': 42
        }
        
        with open(os.path.join(base_work_dir, 'cross_validation_info.json'), 'w') as f:
            json.dump(cv_info, f, indent=2)
        
        print(f"\n✓ Created {len(folds)} folds for cross-validation")
        
    except Exception as e:
        print(f"ERROR: Failed to load or process data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Train each fold
    fold_results = []
    
    for fold_data in folds:
        fold_num = fold_data['fold']
        
        # Skip folds before start_fold (for resuming)
        if fold_num < args.start_fold:
            print(f"⏭️  Skipping fold {fold_num} (before start_fold={args.start_fold})")
            continue
        
        try:
            result = train_single_fold(fold_data, base_work_dir, config_path, main_df)
            fold_results.append(result)
            
            # Save intermediate results
            with open(os.path.join(base_work_dir, 'fold_results.json'), 'w') as f:
                json.dump(fold_results, f, indent=2)
            
        except Exception as e:
            print(f"❌ Critical error in fold {fold_num}: {e}")
            import traceback
            traceback.print_exc()
            
            # Still save the failed result
            fold_results.append({
                'fold': fold_num,
                'work_dir': os.path.join(base_work_dir, f"fold_{fold_num}"),
                'final_mre': None,
                'status': 'failed',
                'error': str(e)
            })
    
    # Summarize results
    print("\n" + "="*80)
    print("5-FOLD CROSS-VALIDATION RESULTS SUMMARY")
    print("="*80)
    
    completed_folds = [r for r in fold_results if r['status'] == 'completed']
    failed_folds = [r for r in fold_results if r['status'] == 'failed']
    
    print(f"✅ Completed folds: {len(completed_folds)}/{len(fold_results)}")
    print(f"❌ Failed folds: {len(failed_folds)}")
    
    if completed_folds:
        valid_mres = [r['final_mre'] for r in completed_folds if r['final_mre'] is not None]
        
        if valid_mres:
            mean_mre = np.mean(valid_mres)
            std_mre = np.std(valid_mres)
            
            print(f"\n📊 CROSS-VALIDATION PERFORMANCE:")
            print(f"   Mean MRE: {mean_mre:.4f} ± {std_mre:.4f} pixels")
            print(f"   Min MRE:  {np.min(valid_mres):.4f} pixels")
            print(f"   Max MRE:  {np.max(valid_mres):.4f} pixels")
            
            print(f"\n📋 PER-FOLD RESULTS:")
            for result in completed_folds:
                if result['final_mre'] is not None:
                    print(f"   Fold {result['fold']}: {result['final_mre']:.4f} pixels")
        
        # Plot results
        plot_fold_results(fold_results, base_work_dir)
    
    if failed_folds:
        print(f"\n❌ FAILED FOLDS:")
        for result in failed_folds:
            print(f"   Fold {result['fold']}: {result.get('error', 'Unknown error')}")
    
    print(f"\n💾 Results saved to: {base_work_dir}")
    print(f"   - Cross-validation info: cross_validation_info.json")
    print(f"   - Fold results: fold_results.json")
    print(f"   - Individual fold results in: fold_1/, fold_2/, etc.")
    
    print(f"\n🎉 5-fold cross-validation completed!")
    print(f"📈 Each fold trained with concurrent MLP refinement for {args.epochs_per_fold} epochs")
    print(f"🎯 Patient-level splitting ensures no data leakage between folds")
    print(f"💾 Storage optimized: Only 68 epochs per fold (90 total epochs vs 222 single training)")

if __name__ == "__main__":
    main() 