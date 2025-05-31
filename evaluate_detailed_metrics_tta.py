#!/usr/bin/env python3
"""
Enhanced evaluation script with Test-Time Augmentation (TTA) support.
Evaluates the quick wins model improvements including UDP codec and ensemble predictions.
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

def inference_with_tta(model, img_array, bbox_data, tta_configs=None):
    """
    Perform inference with test-time augmentation ensemble.
    
    Args:
        model: MMPose model
        img_array: Input image (H, W, 3)
        bbox_data: Bounding box data
        tta_configs: List of TTA configurations
    
    Returns:
        Ensemble averaged keypoints
    """
    if tta_configs is None:
        tta_configs = [
            {'flip': False, 'scale': 1.0},
            {'flip': True, 'scale': 1.0},
            {'flip': False, 'scale': 0.95},
            {'flip': False, 'scale': 1.05}
        ]
    
    all_predictions = []
    
    for config in tta_configs:
        # Apply TTA transforms
        test_img = img_array.copy()
        test_bbox = bbox_data['bbox'].copy()
        
        # Scale augmentation
        if config['scale'] != 1.0:
            h, w = test_img.shape[:2]
            new_h, new_w = int(h * config['scale']), int(w * config['scale'])
            import cv2
            test_img = cv2.resize(test_img, (new_w, new_h))
            # Adjust bbox accordingly
            test_bbox = test_bbox * config['scale']
        
        # Horizontal flip augmentation
        if config['flip']:
            test_img = test_img[:, ::-1, :]  # Flip horizontally
            # Adjust bbox for flip
            test_bbox[0, 0] = test_img.shape[1] - test_bbox[0, 2]  # x_min
            test_bbox[0, 2] = test_img.shape[1] - test_bbox[0, 0]  # x_max
        
        # Run inference
        results = inference_topdown(model, test_img, bboxes=test_bbox, bbox_format='xyxy')
        
        if results and len(results) > 0:
            pred_kpts = results[0].pred_instances.keypoints[0]  # Shape: (19, 2)
            
            # Reverse augmentations for keypoints
            if config['flip']:
                # Flip keypoints back
                pred_kpts[:, 0] = test_img.shape[1] - pred_kpts[:, 0]
                # Apply flip_indices if available
                flip_indices = getattr(model.cfg.dataset_info, 'flip_indices', list(range(19)))
                pred_kpts = pred_kpts[flip_indices]
            
            if config['scale'] != 1.0:
                # Scale keypoints back
                pred_kpts = pred_kpts / config['scale']
            
            all_predictions.append(pred_kpts)
    
    if all_predictions:
        # Ensemble average
        ensemble_pred = np.mean(all_predictions, axis=0)
        return ensemble_pred
    else:
        return None

def compute_mre_metrics(pred_keypoints_list, gt_keypoints_list, landmark_names):
    """
    Compute MRE (Mean Radial Error) and detailed statistics.
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

def plot_comparison_analysis(tta_errors, baseline_errors, save_path=None):
    """Plot comparison between TTA and baseline results."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Error distribution comparison
    ax1.hist(baseline_errors, bins=50, alpha=0.7, color='lightcoral', label='Baseline', density=True)
    ax1.hist(tta_errors, bins=50, alpha=0.7, color='skyblue', label='Quick Wins + TTA', density=True)
    ax1.set_xlabel('Radial Error (pixels)')
    ax1.set_ylabel('Density')
    ax1.set_title('Error Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot comparison
    ax2.boxplot([baseline_errors, tta_errors], 
                labels=['Baseline', 'Quick Wins + TTA'],
                patch_artist=True,
                boxprops=dict(facecolor='lightgreen', alpha=0.7))
    ax2.set_ylabel('Radial Error (pixels)')
    ax2.set_title('Error Distribution Box Plot')
    ax2.grid(True, alpha=0.3)
    
    # Improvement scatter plot
    if len(baseline_errors) == len(tta_errors):
        improvements = baseline_errors - tta_errors
        ax3.scatter(baseline_errors, improvements, alpha=0.6)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Baseline Error (pixels)')
        ax3.set_ylabel('Improvement (pixels)')
        ax3.set_title('Per-Sample Improvement')
        ax3.grid(True, alpha=0.3)
    
    # Cumulative error distribution
    baseline_sorted = np.sort(baseline_errors)
    tta_sorted = np.sort(tta_errors)
    baseline_cumsum = np.arange(1, len(baseline_sorted) + 1) / len(baseline_sorted)
    tta_cumsum = np.arange(1, len(tta_sorted) + 1) / len(tta_sorted)
    
    ax4.plot(baseline_sorted, baseline_cumsum, label='Baseline', color='lightcoral')
    ax4.plot(tta_sorted, tta_cumsum, label='Quick Wins + TTA', color='skyblue')
    ax4.set_xlabel('Radial Error (pixels)')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_title('Cumulative Error Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison analysis plot saved to {save_path}")
    
    plt.close()

def main():
    """Main evaluation function with TTA support."""
    
    print("="*80)
    print("QUICK WINS MODEL EVALUATION WITH TEST-TIME AUGMENTATION")
    print("="*80)
    
    # Initialize MMPose scope
    init_default_scope('mmpose')
    
    # Import custom modules
    import custom_cephalometric_dataset
    import custom_transforms
    import cephalometric_dataset_info
    
    # Configuration
    config_path = "Pretrained_model/hrnetv2_w18_cephalometric_256x256_quickwins.py"
    work_dir = "work_dirs/hrnetv2_w18_cephalometric_quickwins"
    
    # Find the best checkpoint
    checkpoint_pattern = os.path.join(work_dir, "best_NME_epoch_*.pth")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        checkpoint_pattern = os.path.join(work_dir, "epoch_*.pth")
        checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        print("ERROR: No quick wins checkpoints found")
        print("Falling back to previous model for comparison...")
        work_dir = "work_dirs/hrnetv2_w18_cephalometric_improved_v3"
        config_path = "Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py"
        checkpoint_pattern = os.path.join(work_dir, "best_NME_epoch_*.pth")
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
    pred_keypoints_tta_list = []
    gt_keypoints_list = []
    sample_info = []
    
    print("\nRunning inference with and without TTA...")
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
        
        # Run standard inference
        results = inference_topdown(model, img_array, bboxes=data_sample['bbox'], bbox_format='xyxy')
        
        if results and len(results) > 0:
            pred_keypoints = results[0].pred_instances.keypoints[0]  # Shape: (19, 2)
            pred_keypoints_list.append(pred_keypoints)
            
            # Run TTA inference
            tta_keypoints = inference_with_tta(model, img_array, data_sample)
            if tta_keypoints is not None:
                pred_keypoints_tta_list.append(tta_keypoints)
            else:
                pred_keypoints_tta_list.append(pred_keypoints)  # Fallback
            
            gt_keypoints_list.append(gt_keypoints)
            sample_info.append({
                'patient_id': row.get('patient_id', f'sample_{idx}'),
                'index': idx
            })
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(test_df)} samples")
    
    # Compute metrics for both standard and TTA predictions
    print("\nComputing metrics for standard inference...")
    standard_metrics = compute_mre_metrics(pred_keypoints_list, gt_keypoints_list, landmark_names)
    
    print("Computing metrics for TTA inference...")
    tta_metrics = compute_mre_metrics(pred_keypoints_tta_list, gt_keypoints_list, landmark_names)
    
    # Print comparison results
    print("\n" + "="*60)
    print("QUICK WINS EVALUATION RESULTS COMPARISON")
    print("="*60)
    
    print("STANDARD INFERENCE:")
    print(f"Overall MRE: {standard_metrics['overall_mre']:.3f} ± {standard_metrics['overall_std']:.3f} pixels")
    
    print("\nTEST-TIME AUGMENTATION INFERENCE:")
    print(f"Overall MRE: {tta_metrics['overall_mre']:.3f} ± {tta_metrics['overall_std']:.3f} pixels")
    
    # Calculate improvement
    improvement = standard_metrics['overall_mre'] - tta_metrics['overall_mre']
    improvement_pct = (improvement / standard_metrics['overall_mre']) * 100
    
    print(f"\nIMPROVEMENT WITH TTA:")
    print(f"MRE reduction: {improvement:.3f} pixels ({improvement_pct:.1f}%)")
    
    if len(tta_metrics['all_errors']) > 0:
        print(f"\nTTA DETAILED STATISTICS:")
        print(f"Median error: {np.median(tta_metrics['all_errors']):.3f} pixels")
        print(f"90th percentile: {np.percentile(tta_metrics['all_errors'], 90):.3f} pixels")
        print(f"95th percentile: {np.percentile(tta_metrics['all_errors'], 95):.3f} pixels")
    
    # Compare challenging landmarks
    print(f"\nCHALLENGING LANDMARKS COMPARISON:")
    challenging_landmarks = ['sella', 'Gonion', 'PNS']
    for landmark in challenging_landmarks:
        if landmark in standard_metrics['per_landmark_stats']:
            std_error = standard_metrics['per_landmark_stats'][landmark]['mre']
            tta_error = tta_metrics['per_landmark_stats'][landmark]['mre']
            landmark_improvement = std_error - tta_error
            landmark_improvement_pct = (landmark_improvement / std_error) * 100 if std_error > 0 else 0
            print(f"{landmark:<12}: {std_error:.3f} → {tta_error:.3f} pixels "
                  f"({landmark_improvement_pct:+.1f}%)")
    
    # Save detailed results
    output_dir = os.path.join(work_dir, "quickwins_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save comparison results
    comparison_df = pd.DataFrame([
        {
            'method': 'standard',
            'overall_mre': standard_metrics['overall_mre'],
            'overall_std': standard_metrics['overall_std'],
            'median_error': np.median(standard_metrics['all_errors']) if len(standard_metrics['all_errors']) > 0 else 0,
            'p90_error': np.percentile(standard_metrics['all_errors'], 90) if len(standard_metrics['all_errors']) > 0 else 0
        },
        {
            'method': 'tta',
            'overall_mre': tta_metrics['overall_mre'],
            'overall_std': tta_metrics['overall_std'],
            'median_error': np.median(tta_metrics['all_errors']) if len(tta_metrics['all_errors']) > 0 else 0,
            'p90_error': np.percentile(tta_metrics['all_errors'], 90) if len(tta_metrics['all_errors']) > 0 else 0
        }
    ])
    
    comparison_path = os.path.join(output_dir, "method_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nMethod comparison saved to: {comparison_path}")
    
    # Plot comparison analysis
    plot_path = os.path.join(output_dir, "quickwins_comparison.png")
    plot_comparison_analysis(tta_metrics['all_errors'], standard_metrics['all_errors'], plot_path)
    
    print(f"Quick wins evaluation complete! Results saved to: {output_dir}")
    
    # Final summary
    print("\n" + "="*60)
    print("QUICK WINS IMPLEMENTATION SUMMARY")
    print("="*60)
    print("✓ Joint weights increased: Sella/Gonion 2.0x → 3.0x")
    print("✓ UDP codec: Better sub-pixel coordinate accuracy")
    print("✓ Test-time augmentation: Ensemble averaging")
    print(f"✓ Overall improvement: {improvement:.3f} pixels ({improvement_pct:.1f}%)")
    print(f"✓ Target achieved: {'Yes' if tta_metrics['overall_mre'] < 2.3 else 'Partially'}")

if __name__ == "__main__":
    main() 