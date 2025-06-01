#!/usr/bin/env python3
"""
Detailed evaluation script for cephalometric landmark detection model.
Computes MRE in pixels, per-landmark statistics, and other metrics.
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

def compute_mre_metrics(pred_keypoints_list, gt_keypoints_list, landmark_names):
    """
    Compute MRE (Mean Radial Error) and detailed statistics.
    
    Args:
        pred_keypoints_list: List of predicted keypoints arrays (N_samples, N_landmarks, 2)
        gt_keypoints_list: List of ground truth keypoints arrays (N_samples, N_landmarks, 2)
        landmark_names: List of landmark names
    
    Returns:
        Dictionary with detailed metrics
    """
    all_errors = []
    per_landmark_errors = {name: [] for name in landmark_names}
    valid_predictions = 0
    total_predictions = 0
    
    for pred_kpts, gt_kpts in zip(pred_keypoints_list, gt_keypoints_list):
        # Compute radial errors for each landmark
        radial_errors = np.sqrt(np.sum((pred_kpts - gt_kpts)**2, axis=1))
        
        # Check which landmarks are valid (ground truth is not [0,0])
        valid_mask = (gt_kpts[:, 0] > 0) & (gt_kpts[:, 1] > 0)
        
        # Collect errors for valid landmarks
        valid_errors = radial_errors[valid_mask]
        all_errors.extend(valid_errors)
        
        # Per-landmark statistics
        for i, (name, error, valid) in enumerate(zip(landmark_names, radial_errors, valid_mask)):
            if valid:
                per_landmark_errors[name].append(error)
        
        valid_predictions += np.sum(valid_mask)
        total_predictions += len(landmark_names)
    
    # Compute overall metrics
    all_errors = np.array(all_errors)
    overall_mre = np.mean(all_errors) if len(all_errors) > 0 else 0
    overall_std = np.std(all_errors) if len(all_errors) > 0 else 0
    
    # Compute per-landmark metrics
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

def plot_error_distribution(all_errors, landmark_stats, save_path=None):
    """Plot error distribution and per-landmark statistics."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Overall error histogram
    ax1.hist(all_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Radial Error (pixels)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Overall Error Distribution\nMRE: {np.mean(all_errors):.2f} ± {np.std(all_errors):.2f} pixels')
    ax1.grid(True, alpha=0.3)
    
    # Box plot of overall errors
    ax2.boxplot(all_errors, patch_artist=True, 
                boxprops=dict(facecolor='lightcoral', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('Radial Error (pixels)')
    ax2.set_title('Overall Error Box Plot')
    ax2.grid(True, alpha=0.3)
    
    # Per-landmark MRE bar plot
    landmark_names = list(landmark_stats.keys())
    mres = [landmark_stats[name]['mre'] for name in landmark_names]
    stds = [landmark_stats[name]['std'] for name in landmark_names]
    
    bars = ax3.bar(range(len(landmark_names)), mres, yerr=stds, 
                   alpha=0.7, color='lightgreen', capsize=5)
    ax3.set_xlabel('Landmark Index')
    ax3.set_ylabel('MRE (pixels)')
    ax3.set_title('Per-Landmark MRE')
    ax3.set_xticks(range(len(landmark_names)))
    ax3.set_xticklabels([f'{i}' for i in range(len(landmark_names))], rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Per-landmark error counts
    counts = [landmark_stats[name]['count'] for name in landmark_names]
    ax4.bar(range(len(landmark_names)), counts, alpha=0.7, color='orange')
    ax4.set_xlabel('Landmark Index')
    ax4.set_ylabel('Number of Valid Predictions')
    ax4.set_title('Per-Landmark Detection Count')
    ax4.set_xticks(range(len(landmark_names)))
    ax4.set_xticklabels([f'{i}' for i in range(len(landmark_names))], rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Error analysis plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    """Main evaluation function."""
    
    print("="*80)
    print("DETAILED CEPHALOMETRIC MODEL EVALUATION")
    print("="*80)
    
    # Initialize MMPose scope
    init_default_scope('mmpose')
    
    # Import custom modules
    import custom_cephalometric_dataset
    import custom_transforms
    import cephalometric_dataset_info
    
    # Configuration
    config_path = "Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py"
    work_dir = "work_dirs/hrnetv2_w18_cephalometric_384x384_adaptive_wing_loss_v4"  # New work dir for this experiment
    
    # Find the best checkpoint
    checkpoint_pattern = os.path.join(work_dir, "best_NME_epoch_*.pth")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        checkpoint_pattern = os.path.join(work_dir, "epoch_*.pth")
        checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        print("ERROR: No checkpoints found")
        return
    
    checkpoint_path = max(checkpoints, key=os.path.getctime)
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Initialize model
    model = init_model(config_path, checkpoint_path, device='cuda:0' if torch.cuda.is_available() else 'cpu')
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
    
    # Collect predictions and ground truth
    pred_keypoints_list = []
    gt_keypoints_list = []
    sample_info = []
    
    print("\nRunning inference on test samples...")
    for idx, row in test_df.iterrows():
        # Get image
        img_array = np.array(row['Image'], dtype=np.uint8).reshape((224, 224, 3))
        
        # Get ground truth keypoints
        gt_keypoints = []
        for i in range(0, len(landmark_cols), 2):
            x_col = landmark_cols[i]
            y_col = landmark_cols[i+1]
            if x_col in row and y_col in row and pd.notna(row[x_col]) and pd.notna(row[y_col]):
                gt_keypoints.append([row[x_col], row[y_col]])
            else:
                gt_keypoints.append([0, 0])
        gt_keypoints = np.array(gt_keypoints)
        
        # Prepare data for inference
        data_sample = {
            'bbox': np.array([[0, 0, 224, 224]], dtype=np.float32),
            'bbox_scores': np.array([1.0], dtype=np.float32)
        }
        
        # Run inference
        results = inference_topdown(model, img_array, bboxes=data_sample['bbox'], bbox_format='xyxy')
        
        if results and len(results) > 0:
            pred_keypoints = results[0].pred_instances.keypoints[0]  # Shape: (19, 2)
            
            pred_keypoints_list.append(pred_keypoints)
            gt_keypoints_list.append(gt_keypoints)
            sample_info.append({
                'patient_id': row.get('patient_id', f'sample_{idx}'),
                'index': idx
            })
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(test_df)} samples")
    
    # Compute detailed metrics
    print("\nComputing MRE and detailed statistics...")
    metrics = compute_mre_metrics(pred_keypoints_list, gt_keypoints_list, landmark_names)
    
    # Print results
    print("\n" + "="*60)
    print("DETAILED EVALUATION RESULTS")
    print("="*60)
    
    print(f"Overall MRE: {metrics['overall_mre']:.3f} ± {metrics['overall_std']:.3f} pixels")
    print(f"Valid predictions: {metrics['valid_predictions']}/{metrics['total_predictions']} "
          f"({metrics['detection_rate']*100:.1f}%)")
    
    if len(metrics['all_errors']) > 0:
        print(f"Median error: {np.median(metrics['all_errors']):.3f} pixels")
        print(f"90th percentile: {np.percentile(metrics['all_errors'], 90):.3f} pixels")
        print(f"95th percentile: {np.percentile(metrics['all_errors'], 95):.3f} pixels")
    
    # Per-landmark statistics
    print("\n" + "="*60)
    print("PER-LANDMARK STATISTICS")
    print("="*60)
    print(f"{'Index':<5} {'Landmark':<20} {'MRE':<8} {'Std':<8} {'Median':<8} {'Count':<6}")
    print("-" * 60)
    
    for i, name in enumerate(landmark_names):
        stats = metrics['per_landmark_stats'][name]
        print(f"{i:<5} {name:<20} {stats['mre']:<8.3f} {stats['std']:<8.3f} "
              f"{stats['median']:<8.3f} {stats['count']:<6}")
    
    # Save detailed results
    output_dir = os.path.join(work_dir, "detailed_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to CSV
    results_df = pd.DataFrame([
        {
            'landmark_index': i,
            'landmark_name': name,
            'mre_pixels': metrics['per_landmark_stats'][name]['mre'],
            'std_pixels': metrics['per_landmark_stats'][name]['std'],
            'median_pixels': metrics['per_landmark_stats'][name]['median'],
            'min_pixels': metrics['per_landmark_stats'][name]['min'],
            'max_pixels': metrics['per_landmark_stats'][name]['max'],
            'valid_count': metrics['per_landmark_stats'][name]['count']
        }
        for i, name in enumerate(landmark_names)
    ])
    
    csv_path = os.path.join(output_dir, "per_landmark_mre_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")
    
    # Plot error analysis
    plot_path = os.path.join(output_dir, "error_analysis.png")
    plot_error_distribution(metrics['all_errors'], metrics['per_landmark_stats'], plot_path)
    
    # Save overall summary
    summary = {
        'overall_mre_pixels': metrics['overall_mre'],
        'overall_std_pixels': metrics['overall_std'],
        'valid_predictions': metrics['valid_predictions'],
        'total_predictions': metrics['total_predictions'],
        'detection_rate': metrics['detection_rate'],
        'median_error_pixels': np.median(metrics['all_errors']) if len(metrics['all_errors']) > 0 else 0,
        'p90_error_pixels': np.percentile(metrics['all_errors'], 90) if len(metrics['all_errors']) > 0 else 0,
        'p95_error_pixels': np.percentile(metrics['all_errors'], 95) if len(metrics['all_errors']) > 0 else 0
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(output_dir, "overall_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print(f"Overall summary saved to: {summary_path}")
    print(f"Evaluation complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 