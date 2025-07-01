#!/usr/bin/env python3
"""
Concurrent MLP Training Script for Cephalometric Landmark Detection - V5 with 5-Fold Cross-Validation
This script trains HRNetV2 with concurrent MLP refinement using 5-fold cross-validation.
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

def plot_training_progress(work_dir, fold_num):
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
                plt.savefig(os.path.join(work_dir, f'training_progress_fold_{fold_num}.png'), dpi=150)
                plt.close()
                print(f"Training progress plot saved to {work_dir}/training_progress_fold_{fold_num}.png")
    except Exception as e:
        print(f"Could not plot training progress: {e}")

def create_cross_validation_splits(df, n_folds=5, random_state=42):
    """Create 5-fold cross-validation splits ensuring patient-level splitting."""
    
    # Get unique patient IDs
    if 'patient_id' not in df.columns:
        print("ERROR: 'patient_id' column not found. Cannot create patient-level splits.")
        return None
    
    unique_patients = df['patient_id'].unique()
    print(f"📊 Creating {n_folds}-fold cross-validation splits for {len(unique_patients)} unique patients")
    
    # Create KFold splitter
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    cv_splits = []
    for fold_idx, (train_patient_idx, val_patient_idx) in enumerate(kfold.split(unique_patients)):
        train_patients = unique_patients[train_patient_idx]
        val_patients = unique_patients[val_patient_idx]
        
        # Split data based on patient IDs
        train_df = df[df['patient_id'].isin(train_patients)].reset_index(drop=True)
        val_df = df[df['patient_id'].isin(val_patients)].reset_index(drop=True)
        
        cv_splits.append({
            'fold': fold_idx + 1,
            'train_df': train_df,
            'val_df': val_df,
            'train_patients': train_patients.tolist(),
            'val_patients': val_patients.tolist()
        })
        
        print(f"  Fold {fold_idx + 1}: {len(train_df)} train samples ({len(train_patients)} patients), "
              f"{len(val_df)} val samples ({len(val_patients)} patients)")
    
    return cv_splits

def train_single_fold(fold_data, base_config_path, base_work_dir, test_df=None):
    """Train a single fold of cross-validation."""
    
    fold_num = fold_data['fold']
    train_df = fold_data['train_df']
    val_df = fold_data['val_df']
    
    print(f"\n{'='*80}")
    print(f"🚀 TRAINING FOLD {fold_num}")
    print(f"{'='*80}")
    print(f"📊 Training samples: {len(train_df)}")
    print(f"📊 Validation samples: {len(val_df)}")
    
    # Create fold-specific work directory
    fold_work_dir = os.path.join(base_work_dir, f"fold_{fold_num}")
    os.makedirs(fold_work_dir, exist_ok=True)
    
    # Load and modify config for this fold
    cfg = Config.fromfile(base_config_path)
    cfg.work_dir = os.path.abspath(fold_work_dir)
    
    # Save fold data to temporary JSON files
    temp_train_ann_file = os.path.join(fold_work_dir, 'fold_train_ann.json')
    temp_val_ann_file = os.path.join(fold_work_dir, 'fold_val_ann.json')
    
    train_df.to_json(temp_train_ann_file, orient='records', indent=2)
    val_df.to_json(temp_val_ann_file, orient='records', indent=2)
    
    # Update config with fold-specific data
    cfg.train_dataloader.dataset.ann_file = temp_train_ann_file
    cfg.train_dataloader.dataset.data_df = None
    cfg.train_dataloader.dataset.data_root = ''
    
    cfg.val_dataloader.dataset.ann_file = temp_val_ann_file
    cfg.val_dataloader.dataset.data_df = None
    cfg.val_dataloader.dataset.data_root = ''
    
    # Use validation set for test if no separate test set
    if test_df is not None and not test_df.empty:
        temp_test_ann_file = os.path.join(fold_work_dir, 'fold_test_ann.json')
        test_df.to_json(temp_test_ann_file, orient='records', indent=2)
        cfg.test_dataloader.dataset.ann_file = temp_test_ann_file
    else:
        cfg.test_dataloader.dataset.ann_file = temp_val_ann_file
    cfg.test_dataloader.dataset.data_df = None
    cfg.test_dataloader.dataset.data_root = ''
    
    # Save fold information
    fold_info = {
        'fold_number': fold_num,
        'train_patients': fold_data['train_patients'],
        'val_patients': fold_data['val_patients'],
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'work_dir': fold_work_dir
    }
    
    fold_info_file = os.path.join(fold_work_dir, 'fold_info.json')
    with open(fold_info_file, 'w') as f:
        json.dump(fold_info, f, indent=2)
    
    try:
        # Build runner and start training
        print(f"\n🎯 Starting training for fold {fold_num}...")
        runner = Runner.from_cfg(cfg)
        runner.train()
        
        print(f"✅ Fold {fold_num} training completed successfully!")
        
        # Plot training progress for this fold
        plot_training_progress(fold_work_dir, fold_num)
        
        return {
            'fold': fold_num,
            'success': True,
            'work_dir': fold_work_dir,
            'error': None
        }
        
    except Exception as e:
        print(f"❌ Fold {fold_num} training failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'fold': fold_num,
            'success': False,
            'work_dir': fold_work_dir,
            'error': str(e)
        }

def aggregate_cv_results(cv_results, base_work_dir):
    """Aggregate results from all cross-validation folds."""
    
    print(f"\n{'='*80}")
    print("📊 CROSS-VALIDATION RESULTS SUMMARY")
    print(f"{'='*80}")
    
    successful_folds = [r for r in cv_results if r['success']]
    failed_folds = [r for r in cv_results if not r['success']]
    
    print(f"✅ Successful folds: {len(successful_folds)}/5")
    print(f"❌ Failed folds: {len(failed_folds)}/5")
    
    if failed_folds:
        print(f"\n❌ Failed folds:")
        for fold_result in failed_folds:
            print(f"  - Fold {fold_result['fold']}: {fold_result['error']}")
    
    if successful_folds:
        print(f"\n✅ Successful folds:")
        for fold_result in successful_folds:
            fold_dir = fold_result['work_dir']
            # Look for best checkpoint in each fold
            import glob
            best_checkpoints = glob.glob(os.path.join(fold_dir, "best_NME_epoch_*.pth"))
            if best_checkpoints:
                best_checkpoint = max(best_checkpoints, key=os.path.getctime)
                epoch_num = os.path.basename(best_checkpoint).split('_epoch_')[1].split('.')[0]
                print(f"  - Fold {fold_result['fold']}: Best epoch {epoch_num}")
            else:
                print(f"  - Fold {fold_result['fold']}: Training completed")
    
    # Save cross-validation summary
    cv_summary = {
        'total_folds': 5,
        'successful_folds': len(successful_folds),
        'failed_folds': len(failed_folds),
        'results': cv_results
    }
    
    summary_file = os.path.join(base_work_dir, 'cross_validation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(cv_summary, f, indent=2)
    
    print(f"\n💾 Cross-validation summary saved to: {summary_file}")
    
    return cv_summary

def main():
    """Main concurrent training function with 5-fold cross-validation."""
    
    parser = argparse.ArgumentParser(
        description='Concurrent MLP Training Script for Cephalometric Landmark Detection - V5 with 5-Fold CV')
    parser.add_argument(
        '--test_split_file',
        type=str,
        default=None,
        help='Path to a text file containing patient IDs for the test set, one ID per line.'
    )
    parser.add_argument(
        '--n_folds',
        type=int,
        default=5,
        help='Number of folds for cross-validation (default: 5)'
    )
    parser.add_argument(
        '--start_fold',
        type=int,
        default=1,
        help='Starting fold number (1-based, default: 1)'
    )
    parser.add_argument(
        '--end_fold',
        type=int,
        default=None,
        help='Ending fold number (1-based, default: all folds)'
    )
    parser.add_argument(
        '--cv_random_state',
        type=int,
        default=42,
        help='Random state for cross-validation splits (default: 42)'
    )
    args = parser.parse_args()
    
    # Set end_fold to n_folds if not specified
    if args.end_fold is None:
        args.end_fold = args.n_folds
    
    print("="*80)
    print("CONCURRENT MLP TRAINING - V5 with 5-FOLD CROSS-VALIDATION")
    print("🚀 HRNetV2 + On-the-fly MLP Refinement + Cross-Validation")
    print("="*80)
    print(f"📊 Cross-validation: {args.n_folds}-fold")
    print(f"🎯 Training folds: {args.start_fold} to {args.end_fold}")
    print(f"🎲 Random state: {args.cv_random_state}")
    
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
    base_work_dir = "work_dirs/hrnetv2_w18_cephalometric_concurrent_mlp_v5_cv"
    
    print(f"Config: {config_path}")
    print(f"Base Work Dir: {base_work_dir}")
    
    # Create base work directory
    os.makedirs(base_work_dir, exist_ok=True)
    
    # Load and prepare data
    data_file_path = "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json"
    print(f"Loading main data file from: {data_file_path}")
    
    try:
        main_df = pd.read_json(data_file_path)
        print(f"Main DataFrame loaded. Shape: {main_df.shape}")

        # Handle test data splitting (same as before)
        test_df = pd.DataFrame()  # Initialize empty test DataFrame
        
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
            remaining_df = main_df[~main_df['patient_id'].isin(test_patient_ids)].reset_index(drop=True)
            
            print(f"Test DataFrame shape: {test_df.shape}")
            print(f"Remaining DataFrame for CV: {remaining_df.shape}")
            
        else:
            print("Using all data for cross-validation (no external test set)")
            remaining_df = main_df.copy()

        if remaining_df.empty:
            print("ERROR: No data available for cross-validation.")
            return

        # Create cross-validation splits
        cv_splits = create_cross_validation_splits(
            remaining_df, 
            n_folds=args.n_folds, 
            random_state=args.cv_random_state
        )
        
        if cv_splits is None:
            return
        
        # Save cross-validation split information
        cv_info = {
            'n_folds': args.n_folds,
            'random_state': args.cv_random_state,
            'total_patients': len(remaining_df['patient_id'].unique()),
            'total_samples': len(remaining_df),
            'test_patients': len(test_df['patient_id'].unique()) if not test_df.empty else 0,
            'test_samples': len(test_df),
            'folds': []
        }
        
        for split in cv_splits:
            cv_info['folds'].append({
                'fold': split['fold'],
                'train_patients': len(split['train_patients']),
                'val_patients': len(split['val_patients']),
                'train_samples': len(split['train_df']),
                'val_samples': len(split['val_df'])
            })
        
        cv_info_file = os.path.join(base_work_dir, 'cross_validation_info.json')
        with open(cv_info_file, 'w') as f:
            json.dump(cv_info, f, indent=2)
        
        print(f"✓ Cross-validation info saved to: {cv_info_file}")

    except Exception as e:
        print(f"ERROR: Failed to load or process data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print cross-validation training information
    print("\n" + "="*70)
    print("🚀 5-FOLD CROSS-VALIDATION TRAINING APPROACH")
    print("="*70)
    print(f"🔄 Training Cycle (per fold):")
    print(f"   • Train HRNetV2 for 1 epoch")
    print(f"   • Run inference on training data with current HRNet weights")
    print(f"   • Train MLP models for 100 epochs on current predictions")
    print(f"   • Apply hard-example oversampling for next epoch")
    print(f"   • Repeat for all 222 epochs")
    
    print(f"\n📊 Cross-Validation Benefits:")
    print(f"   • More robust performance estimates")
    print(f"   • Better utilization of available data")
    print(f"   • Reduced variance in results")
    print(f"   • Patient-level splitting (no data leakage)")
    
    print(f"\n🧠 MLP Architecture (same for all folds):")
    print(f"   • Input: 38 predicted coordinates (joint model)")
    print(f"   • Hidden: 500 neurons (ReLU + Dropout)")
    print(f"   • Output: 38 refined coordinates")
    print(f"   • Hard-example oversampling for both MLP and HRNet")
    
    # Train each fold
    cv_results = []
    
    for fold_data in cv_splits:
        fold_num = fold_data['fold']
        
        # Check if we should train this fold
        if fold_num < args.start_fold or fold_num > args.end_fold:
            print(f"\n⏭️  Skipping fold {fold_num} (outside range {args.start_fold}-{args.end_fold})")
            continue
        
        # Train this fold
        fold_result = train_single_fold(fold_data, config_path, base_work_dir, test_df)
        cv_results.append(fold_result)
        
        # Save intermediate results
        intermediate_summary = {
            'completed_folds': len(cv_results),
            'successful_folds': len([r for r in cv_results if r['success']]),
            'results_so_far': cv_results
        }
        
        intermediate_file = os.path.join(base_work_dir, 'intermediate_cv_results.json')
        with open(intermediate_file, 'w') as f:
            json.dump(intermediate_summary, f, indent=2)
    
    # Aggregate and summarize results
    if cv_results:
        final_summary = aggregate_cv_results(cv_results, base_work_dir)
        
        print(f"\n🎉 Cross-validation training completed!")
        print(f"📈 {final_summary['successful_folds']}/{final_summary['total_folds']} folds completed successfully")
        
        if final_summary['successful_folds'] > 0:
            print(f"\n📋 Next steps:")
            print(f"1. 📊 Evaluate each fold using evaluate_concurrent_mlp.py")
            print(f"2. 🔍 Aggregate results across all folds")
            print(f"3. 📈 Compare cross-validation performance")
            print(f"4. 🎯 Select best fold or ensemble predictions")
            
            print(f"\n💡 Evaluation example:")
            print(f"   python evaluate_concurrent_mlp.py --work_dir={base_work_dir}/fold_1 --checkpoint_type=best")
            print(f"   python evaluate_concurrent_mlp.py --work_dir={base_work_dir}/fold_2 --checkpoint_type=best")
            print(f"   # ... for each fold")
    else:
        print(f"\n⚠️  No folds were trained in the specified range {args.start_fold}-{args.end_fold}")

if __name__ == "__main__":
    main() 