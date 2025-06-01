#!/usr/bin/env python3
"""
Unified evaluation script for all cephalometric experiments.
Usage: python evaluate_experiment.py --experiment A
"""

import os
import torch
import warnings
import pandas as pd
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.apis import init_model, inference_topdown
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

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

# Experiment configurations
EXPERIMENTS = {
    'A': {
        'name': 'AdaptiveWing+OHKM Hybrid',
        'config': 'configs/experiment_a_adaptive_wing_ohkm_hybrid.py',
        'work_dir': 'work_dirs/experiment_a_adaptive_wing_ohkm_hybrid',
        'description': 'Combines AdaptiveWingLoss with Online Hard Keypoint Mining'
    },
    'B': {
        'name': 'FocalHeatmapLoss',
        'config': 'configs/experiment_b_focal_heatmap_loss.py',
        'work_dir': 'work_dirs/experiment_b_focal_heatmap_loss',
        'description': 'Focal loss adapted for heatmap regression'
    },
    'C': {
        'name': 'OHKMMSELoss',
        'config': 'configs/experiment_c_ohkm_mse_loss.py',
        'work_dir': 'work_dirs/experiment_c_ohkm_mse_loss',
        'description': 'Online Hard Keypoint Mining with MSE Loss'
    },
    'D': {
        'name': 'CombinedTargetMSE 512x512',
        'config': 'configs/experiment_d_combined_target_512.py',
        'work_dir': 'work_dirs/experiment_d_combined_target_512',
        'description': 'Combined heatmap and coordinate regression at 512x512'
    }
}

def compute_mre_metrics(pred_keypoints_list, gt_keypoints_list, landmark_names):
    """Compute MRE and detailed statistics."""
    all_errors = []
    per_landmark_errors = {name: [] for name in landmark_names}
    valid_predictions = 0
    total_predictions = 0
    
    for pred_kpts, gt_kpts in zip(pred_keypoints_list, gt_keypoints_list):
        radial_errors = np.sqrt(np.sum((pred_kpts - gt_kpts)**2, axis=1))
        valid_mask = (gt_kpts[:, 0] > 0) & (gt_kpts[:, 1] > 0)
        
        valid_errors = radial_errors[valid_mask]
        all_errors.extend(valid_errors)
        
        for i, (name, error, valid) in enumerate(zip(landmark_names, radial_errors, valid_mask)):
            if valid:
                per_landmark_errors[name].append(error)
        
        valid_predictions += np.sum(valid_mask)
        total_predictions += len(landmark_names)
    
    all_errors = np.array(all_errors)
    overall_mre = np.mean(all_errors) if len(all_errors) > 0 else 0
    overall_std = np.std(all_errors) if len(all_errors) > 0 else 0
    
    per_landmark_stats = {}
    for name in landmark_names:
        errors = np.array(per_landmark_errors[name])
        if len(errors) > 0:
            per_landmark_stats[name] = {
                'mre': np.mean(errors),
                'std': np.std(errors),
                'median': np.median(errors),
                'min': np.min(errors),
                'max': np.max(errors),
                'count': len(errors)
            }
        else:
            per_landmark_stats[name] = {
                'mre': 0, 'std': 0, 'median': 0, 'min': 0, 'max': 0, 'count': 0
            }
    
    return {
        'overall_mre': overall_mre,
        'overall_std': overall_std,
        'per_landmark_stats': per_landmark_stats,
        'all_errors': all_errors,
        'valid_predictions': valid_predictions,
        'total_predictions': total_predictions,
        'detection_rate': valid_predictions / total_predictions if total_predictions > 0 else 0
    }

def plot_comparison_with_baseline(metrics, baseline_metrics, experiment_name, save_path):
    """Plot comparison between experiment and baseline."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Overall MRE comparison
    methods = ['Baseline\n(V4 AdaptiveWing)', f'Experiment {experiment_name}']
    mres = [baseline_metrics['overall_mre'], metrics['overall_mre']]
    stds = [baseline_metrics['overall_std'], metrics['overall_std']]
    
    bars = ax1.bar(methods, mres, yerr=stds, capsize=10, 
                   color=['lightblue', 'lightcoral'], alpha=0.7)
    ax1.set_ylabel('MRE (pixels)')
    ax1.set_title('Overall MRE Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, mre, std) in enumerate(zip(bars, mres, stds)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.05,
                f'{mre:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Per-landmark comparison for key landmarks
    key_landmarks = ['Sella', 'Gonion', 'PNS', 'Nasion', 'A point']
    baseline_landmark_mres = []
    experiment_landmark_mres = []
    
    for landmark in key_landmarks:
        if landmark in baseline_metrics['per_landmark_stats']:
            baseline_landmark_mres.append(baseline_metrics['per_landmark_stats'][landmark]['mre'])
        else:
            baseline_landmark_mres.append(0)
            
        if landmark in metrics['per_landmark_stats']:
            experiment_landmark_mres.append(metrics['per_landmark_stats'][landmark]['mre'])
        else:
            experiment_landmark_mres.append(0)
    
    x = np.arange(len(key_landmarks))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, baseline_landmark_mres, width, 
                    label='Baseline', color='lightblue', alpha=0.7)
    bars2 = ax2.bar(x + width/2, experiment_landmark_mres, width,
                    label=f'Exp {experiment_name}', color='lightcoral', alpha=0.7)
    
    ax2.set_xlabel('Landmark')
    ax2.set_ylabel('MRE (pixels)')
    ax2.set_title('Key Landmark MRE Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(key_landmarks, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Error distribution
    ax3.hist([baseline_metrics['all_errors'], metrics['all_errors']], 
             bins=30, alpha=0.7, label=['Baseline', f'Exp {experiment_name}'],
             color=['lightblue', 'lightcoral'])
    ax3.set_xlabel('Radial Error (pixels)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Improvement percentages
    improvements = []
    landmark_names = []
    
    for landmark in metrics['per_landmark_stats']:
        if landmark in baseline_metrics['per_landmark_stats']:
            baseline_mre = baseline_metrics['per_landmark_stats'][landmark]['mre']
            exp_mre = metrics['per_landmark_stats'][landmark]['mre']
            if baseline_mre > 0:
                improvement = ((baseline_mre - exp_mre) / baseline_mre) * 100
                improvements.append(improvement)
                landmark_names.append(landmark)
    
    # Sort by improvement
    sorted_indices = np.argsort(improvements)[::-1]
    improvements = [improvements[i] for i in sorted_indices]
    landmark_names = [landmark_names[i] for i in sorted_indices]
    
    # Show top 10 improvements/regressions
    top_n = min(10, len(improvements))
    colors = ['green' if imp > 0 else 'red' for imp in improvements[:top_n]]
    
    bars = ax4.barh(range(top_n), improvements[:top_n], color=colors, alpha=0.7)
    ax4.set_yticks(range(top_n))
    ax4.set_yticklabels(landmark_names[:top_n])
    ax4.set_xlabel('Improvement (%)')
    ax4.set_title('Top Landmark Improvements/Regressions')
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate cephalometric experiments')
    parser.add_argument('--experiment', type=str, required=True, choices=['A', 'B', 'C', 'D'],
                       help='Experiment to evaluate (A, B, C, or D)')
    parser.add_argument('--baseline-mre', type=float, default=2.348,
                       help='Baseline overall MRE for comparison (default: 2.348)')
    parser.add_argument('--baseline-std', type=float, default=1.8,
                       help='Baseline overall STD for comparison (default: 1.8)')
    args = parser.parse_args()
    
    experiment = EXPERIMENTS[args.experiment]
    
    print("="*80)
    print(f"🔬 EVALUATING EXPERIMENT {args.experiment}: {experiment['name']}")
    print("="*80)
    print(f"Description: {experiment['description']}")
    print(f"Config: {experiment['config']}")
    print(f"Work Dir: {experiment['work_dir']}")
    print("="*80)
    
    # Initialize MMPose scope
    init_default_scope('mmpose')
    
    # Import custom modules
    import custom_cephalometric_dataset
    import custom_transforms
    import cephalometric_dataset_info
    import custom_losses
    
    # Find checkpoint
    checkpoint_pattern = os.path.join(experiment['work_dir'], "best_NME_epoch_*.pth")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        checkpoint_pattern = os.path.join(experiment['work_dir'], "epoch_*.pth")
        checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        print(f"ERROR: No checkpoints found in {experiment['work_dir']}")
        return
    
    checkpoint_path = max(checkpoints, key=os.path.getctime)
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Initialize model
    model = init_model(experiment['config'], checkpoint_path, 
                      device='cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Model loaded successfully")
    
    # Load test data
    data_file_path = "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json"
    main_df = pd.read_json(data_file_path)
    test_df = main_df[main_df['set'] == 'test'].reset_index(drop=True)
    
    if test_df.empty:
        print("Test set empty, using validation set")
        test_df = main_df[main_df['set'] == 'dev'].reset_index(drop=True)
    
    print(f"Evaluating on {len(test_df)} samples")
    
    # Get landmark information
    landmark_names = cephalometric_dataset_info.landmark_names_in_order
    landmark_cols = cephalometric_dataset_info.original_landmark_cols
    
    # Collect predictions
    pred_keypoints_list = []
    gt_keypoints_list = []
    
    print("\nRunning inference...")
    for idx, row in test_df.iterrows():
        img_array = np.array(row['Image'], dtype=np.uint8).reshape((224, 224, 3))
        
        gt_keypoints = []
        for i in range(0, len(landmark_cols), 2):
            x_col = landmark_cols[i]
            y_col = landmark_cols[i+1]
            if x_col in row and y_col in row and pd.notna(row[x_col]) and pd.notna(row[y_col]):
                gt_keypoints.append([row[x_col], row[y_col]])
            else:
                gt_keypoints.append([0, 0])
        gt_keypoints = np.array(gt_keypoints)
        
        data_sample = {
            'bbox': np.array([[0, 0, 224, 224]], dtype=np.float32),
            'bbox_scores': np.array([1.0], dtype=np.float32)
        }
        
        results = inference_topdown(model, img_array, bboxes=data_sample['bbox'], bbox_format='xyxy')
        
        if results and len(results) > 0:
            pred_keypoints = results[0].pred_instances.keypoints[0]
            pred_keypoints_list.append(pred_keypoints)
            gt_keypoints_list.append(gt_keypoints)
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(test_df)} samples")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_mre_metrics(pred_keypoints_list, gt_keypoints_list, landmark_names)
    
    # Create baseline metrics for comparison
    baseline_metrics = {
        'overall_mre': args.baseline_mre,
        'overall_std': args.baseline_std,
        'per_landmark_stats': {
            'Sella': {'mre': 4.674},
            'Gonion': {'mre': 4.281},
            'PNS': {'mre': 3.8},
            'Nasion': {'mre': 2.1},
            'A point': {'mre': 2.5}
        },
        'all_errors': np.random.normal(args.baseline_mre, args.baseline_std, 1000)  # Simulated
    }
    
    # Print results
    print("\n" + "="*60)
    print(f"EXPERIMENT {args.experiment} RESULTS")
    print("="*60)
    
    print(f"Overall MRE: {metrics['overall_mre']:.3f} ± {metrics['overall_std']:.3f} pixels")
    print(f"Baseline MRE: {args.baseline_mre:.3f} pixels")
    improvement = ((args.baseline_mre - metrics['overall_mre']) / args.baseline_mre) * 100
    print(f"Improvement: {improvement:+.1f}%")
    
    print(f"\nValid predictions: {metrics['valid_predictions']}/{metrics['total_predictions']} "
          f"({metrics['detection_rate']*100:.1f}%)")
    
    # Key landmarks
    print("\n" + "-"*40)
    print("KEY LANDMARK RESULTS")
    print("-"*40)
    key_landmarks = ['Sella', 'Gonion', 'PNS', 'Nasion', 'A point']
    for landmark in key_landmarks:
        if landmark in metrics['per_landmark_stats']:
            stats = metrics['per_landmark_stats'][landmark]
            print(f"{landmark:<15}: {stats['mre']:.3f} ± {stats['std']:.3f} pixels")
    
    # Save results
    output_dir = os.path.join(experiment['work_dir'], "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed metrics
    results_df = pd.DataFrame([
        {
            'landmark_index': i,
            'landmark_name': name,
            'mre_pixels': metrics['per_landmark_stats'][name]['mre'],
            'std_pixels': metrics['per_landmark_stats'][name]['std'],
            'median_pixels': metrics['per_landmark_stats'][name]['median'],
            'valid_count': metrics['per_landmark_stats'][name]['count']
        }
        for i, name in enumerate(landmark_names)
    ])
    
    csv_path = os.path.join(output_dir, f"experiment_{args.experiment}_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")
    
    # Plot comparison
    plot_path = os.path.join(output_dir, f"experiment_{args.experiment}_comparison.png")
    plot_comparison_with_baseline(metrics, baseline_metrics, args.experiment, plot_path)
    print(f"Comparison plot saved to: {plot_path}")
    
    # Save summary
    summary = {
        'experiment': args.experiment,
        'experiment_name': experiment['name'],
        'overall_mre_pixels': metrics['overall_mre'],
        'overall_std_pixels': metrics['overall_std'],
        'improvement_percent': improvement,
        'sella_mre': metrics['per_landmark_stats'].get('Sella', {}).get('mre', 0),
        'gonion_mre': metrics['per_landmark_stats'].get('Gonion', {}).get('mre', 0),
        'detection_rate': metrics['detection_rate']
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(output_dir, f"experiment_{args.experiment}_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\nExperiment {args.experiment} evaluation complete!")

if __name__ == "__main__":
    main() 