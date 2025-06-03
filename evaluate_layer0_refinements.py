#!/usr/bin/env python3
"""
Layer 0 Refinements Evaluation Script
Implements DARK, flip-test, and median filtering with incremental MRE reporting
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
from scipy.spatial import procrustes
from scipy.stats import mode
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

def compute_mre_simple(pred_keypoints_list, gt_keypoints_list):
    """Compute simple MRE statistics for quick comparison."""
    all_errors = []
    
    for pred_kpts, gt_kpts in zip(pred_keypoints_list, gt_keypoints_list):
        # Compute radial errors
        radial_errors = np.sqrt(np.sum((pred_kpts - gt_kpts)**2, axis=1))
        
        # Check valid landmarks
        valid_mask = (gt_kpts[:, 0] > 0) & (gt_kpts[:, 1] > 0)
        
        # Collect valid errors
        valid_errors = radial_errors[valid_mask]
        all_errors.extend(valid_errors)
    
    all_errors = np.array(all_errors)
    return {
        'mre': np.mean(all_errors),
        'std': np.std(all_errors),
        'median': np.median(all_errors),
        'p90': np.percentile(all_errors, 90),
        'p95': np.percentile(all_errors, 95)
    }

def apply_procrustes_alignment(source_landmarks, target_landmarks):
    """Apply Procrustes alignment to align source to target landmarks."""
    # Filter valid landmarks present in both
    valid_mask = ((source_landmarks[:, 0] > 0) & (source_landmarks[:, 1] > 0) & 
                  (target_landmarks[:, 0] > 0) & (target_landmarks[:, 1] > 0))
    
    if np.sum(valid_mask) < 3:  # Need at least 3 points for alignment
        return source_landmarks.copy()
    
    source_valid = source_landmarks[valid_mask]
    target_valid = target_landmarks[valid_mask]
    
    # Apply Procrustes
    _, aligned_source, _ = procrustes(target_valid, source_valid)
    
    # Apply transformation to all landmarks
    result = source_landmarks.copy()
    result[valid_mask] = aligned_source
    
    return result

def apply_median_filtering(pred_keypoints, train_keypoints_list, train_df, 
                          problematic_indices=[0, 10, 9], k=5):
    """
    Apply median filtering for problematic landmarks using k-NN in shape space.
    
    Args:
        pred_keypoints: Predicted keypoints for test sample (19, 2)
        train_keypoints_list: List of training keypoints arrays
        train_df: Training dataframe
        problematic_indices: Indices of landmarks to apply filtering (Sella=0, Gonion=10, PNS=9)
        k: Number of nearest neighbors
    """
    refined_keypoints = pred_keypoints.copy()
    
    # Compute distances to all training samples after alignment
    distances = []
    aligned_train_keypoints = []
    
    for train_kpts in train_keypoints_list:
        # Align training sample to test sample
        aligned_train = apply_procrustes_alignment(train_kpts, pred_keypoints)
        aligned_train_keypoints.append(aligned_train)
        
        # Compute distance (only on valid landmarks)
        valid_mask = ((pred_keypoints[:, 0] > 0) & (pred_keypoints[:, 1] > 0) & 
                      (aligned_train[:, 0] > 0) & (aligned_train[:, 1] > 0))
        
        if np.sum(valid_mask) > 0:
            dist = np.sqrt(np.sum((pred_keypoints[valid_mask] - aligned_train[valid_mask])**2))
            distances.append(dist)
        else:
            distances.append(np.inf)
    
    # Find k nearest neighbors
    distances = np.array(distances)
    k_nearest_indices = np.argsort(distances)[:k]
    
    # Apply median filtering for problematic landmarks
    for landmark_idx in problematic_indices:
        neighbor_coords = []
        
        for train_idx in k_nearest_indices:
            train_kpts = aligned_train_keypoints[train_idx]
            if train_kpts[landmark_idx, 0] > 0 and train_kpts[landmark_idx, 1] > 0:
                neighbor_coords.append(train_kpts[landmark_idx])
        
        if len(neighbor_coords) >= 3:  # Need at least 3 neighbors
            neighbor_coords = np.array(neighbor_coords)
            median_coord = np.median(neighbor_coords, axis=0)
            refined_keypoints[landmark_idx] = median_coord
    
    return refined_keypoints

def inference_with_flip_test(model, img_array, bbox):
    """Run inference with flip test and average the results."""
    # Original inference
    results_orig = inference_topdown(model, img_array, bboxes=bbox, bbox_format='xyxy')
    
    # Flip image horizontally
    img_flipped = np.fliplr(img_array).copy()
    
    # Run inference on flipped image
    results_flip = inference_topdown(model, img_flipped, bboxes=bbox, bbox_format='xyxy')
    
    if results_orig and results_flip and len(results_orig) > 0 and len(results_flip) > 0:
        # Get keypoints
        kpts_orig = results_orig[0].pred_instances.keypoints[0]  # (19, 2)
        kpts_flip = results_flip[0].pred_instances.keypoints[0]  # (19, 2)
        
        # Unflip the flipped predictions
        kpts_flip_unflipped = kpts_flip.copy()
        kpts_flip_unflipped[:, 0] = img_array.shape[1] - kpts_flip[:, 0]  # Mirror x-coordinates
        
        # Handle flip indices if they exist (some landmarks might swap during flip)
        # For now, we assume symmetric landmarks are handled by the model
        
        # Average the predictions
        kpts_averaged = (kpts_orig + kpts_flip_unflipped) / 2.0
        
        # Update the result
        results_orig[0].pred_instances.keypoints[0] = kpts_averaged
        
    return results_orig

def main():
    """Main evaluation function with Layer 0 refinements."""
    
    print("="*80)
    print("LAYER 0 REFINEMENTS - INCREMENTAL EVALUATION")
    print("="*80)
    
    # Initialize MMPose scope
    init_default_scope('mmpose')
    
    # Import custom modules
    import custom_cephalometric_dataset
    import custom_transforms
    import cephalometric_dataset_info
    
    # Configuration
    config_path = "Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py"
    work_dir = "work_dirs/hrnetv2_w18_cephalometric_384x384_adaptive_wing_loss_v4"
    
    # Load config
    cfg = Config.fromfile(config_path)
    
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
    
    # Load test data
    data_file_path = "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json"
    main_df = pd.read_json(data_file_path)
    test_df = main_df[main_df['set'] == 'test'].reset_index(drop=True)
    train_df = main_df[main_df['set'] == 'train'].reset_index(drop=True)
    
    if test_df.empty:
        print("Test set empty, using validation set")
        test_df = main_df[main_df['set'] == 'dev'].reset_index(drop=True)
    
    print(f"Evaluating on {len(test_df)} test samples")
    print(f"Using {len(train_df)} training samples for median filtering")
    
    # Get landmark information
    landmark_names = cephalometric_dataset_info.landmark_names_in_order
    landmark_cols = cephalometric_dataset_info.original_landmark_cols
    
    # Prepare training keypoints for median filtering
    print("\nPreparing training data for median filtering...")
    train_keypoints_list = []
    for _, row in train_df.iterrows():
        gt_keypoints = []
        for i in range(0, len(landmark_cols), 2):
            x_col = landmark_cols[i]
            y_col = landmark_cols[i+1]
            if x_col in row and y_col in row and pd.notna(row[x_col]) and pd.notna(row[y_col]):
                gt_keypoints.append([row[x_col], row[y_col]])
            else:
                gt_keypoints.append([0, 0])
        train_keypoints_list.append(np.array(gt_keypoints))
    
    # Store results for each refinement level
    results_baseline = {'pred': [], 'gt': []}
    results_dark = {'pred': [], 'gt': []}
    results_flip = {'pred': [], 'gt': []}
    results_median = {'pred': [], 'gt': []}
    
    # =======================
    # STEP 1: BASELINE (No refinements)
    # =======================
    print("\n" + "="*60)
    print("STEP 1: BASELINE EVALUATION (No refinements)")
    print("="*60)
    
    # Initialize model WITHOUT DARK
    model_baseline = init_model(config_path, checkpoint_path, 
                               device='cuda:0' if torch.cuda.is_available() else 'cpu')
    
    for idx, row in test_df.iterrows():
        # Get image
        img_array = np.array(row['Image'], dtype=np.uint8).reshape((224, 224, 3))
        
        # Get ground truth
        gt_keypoints = []
        for i in range(0, len(landmark_cols), 2):
            x_col = landmark_cols[i]
            y_col = landmark_cols[i+1]
            if x_col in row and y_col in row and pd.notna(row[x_col]) and pd.notna(row[y_col]):
                gt_keypoints.append([row[x_col], row[y_col]])
            else:
                gt_keypoints.append([0, 0])
        gt_keypoints = np.array(gt_keypoints)
        
        # Prepare bbox
        bbox = np.array([[0, 0, 224, 224]], dtype=np.float32)
        
        # Run inference
        results = inference_topdown(model_baseline, img_array, bboxes=bbox, bbox_format='xyxy')
        
        if results and len(results) > 0:
            pred_keypoints = results[0].pred_instances.keypoints[0]
            results_baseline['pred'].append(pred_keypoints)
            results_baseline['gt'].append(gt_keypoints)
        
        if (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1}/{len(test_df)} samples")
    
    # Compute baseline MRE
    baseline_metrics = compute_mre_simple(results_baseline['pred'], results_baseline['gt'])
    print(f"\nBASELINE MRE: {baseline_metrics['mre']:.3f} ± {baseline_metrics['std']:.3f} pixels")
    print(f"Median: {baseline_metrics['median']:.3f}, P90: {baseline_metrics['p90']:.3f}, P95: {baseline_metrics['p95']:.3f}")
    
    # =======================
    # STEP 2: DARK DECODING
    # =======================
    print("\n" + "="*60)
    print("STEP 2: WITH DARK DECODING")
    print("="*60)
    
    # Update config to enable DARK
    cfg_dark = cfg.copy()
    cfg_dark.model.test_cfg = dict(
        flip_test=False,
        shift_heatmap=True,
        use_dark=True  # Enable DARK decoding
    )
    
    # Initialize model with DARK
    model_dark = init_model(cfg_dark, checkpoint_path,
                           device='cuda:0' if torch.cuda.is_available() else 'cpu')
    
    for idx, row in test_df.iterrows():
        img_array = np.array(row['Image'], dtype=np.uint8).reshape((224, 224, 3))
        gt_keypoints = results_baseline['gt'][idx]  # Reuse GT from baseline
        
        bbox = np.array([[0, 0, 224, 224]], dtype=np.float32)
        results = inference_topdown(model_dark, img_array, bboxes=bbox, bbox_format='xyxy')
        
        if results and len(results) > 0:
            pred_keypoints = results[0].pred_instances.keypoints[0]
            results_dark['pred'].append(pred_keypoints)
            results_dark['gt'].append(gt_keypoints)
        
        if (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1}/{len(test_df)} samples")
    
    # Compute DARK MRE
    dark_metrics = compute_mre_simple(results_dark['pred'], results_dark['gt'])
    print(f"\nDARK MRE: {dark_metrics['mre']:.3f} ± {dark_metrics['std']:.3f} pixels")
    print(f"Median: {dark_metrics['median']:.3f}, P90: {dark_metrics['p90']:.3f}, P95: {dark_metrics['p95']:.3f}")
    improvement_dark = (baseline_metrics['mre'] - dark_metrics['mre']) / baseline_metrics['mre'] * 100
    print(f"Improvement over baseline: {improvement_dark:.1f}%")
    
    # =======================
    # STEP 3: DARK + FLIP TEST
    # =======================
    print("\n" + "="*60)
    print("STEP 3: WITH DARK + FLIP TEST")
    print("="*60)
    
    for idx, row in test_df.iterrows():
        img_array = np.array(row['Image'], dtype=np.uint8).reshape((224, 224, 3))
        gt_keypoints = results_baseline['gt'][idx]
        
        bbox = np.array([[0, 0, 224, 224]], dtype=np.float32)
        results = inference_with_flip_test(model_dark, img_array, bbox)
        
        if results and len(results) > 0:
            pred_keypoints = results[0].pred_instances.keypoints[0]
            results_flip['pred'].append(pred_keypoints)
            results_flip['gt'].append(gt_keypoints)
        
        if (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1}/{len(test_df)} samples")
    
    # Compute flip test MRE
    flip_metrics = compute_mre_simple(results_flip['pred'], results_flip['gt'])
    print(f"\nDARK + FLIP TEST MRE: {flip_metrics['mre']:.3f} ± {flip_metrics['std']:.3f} pixels")
    print(f"Median: {flip_metrics['median']:.3f}, P90: {flip_metrics['p90']:.3f}, P95: {flip_metrics['p95']:.3f}")
    improvement_flip = (baseline_metrics['mre'] - flip_metrics['mre']) / baseline_metrics['mre'] * 100
    print(f"Improvement over baseline: {improvement_flip:.1f}%")
    improvement_over_dark = (dark_metrics['mre'] - flip_metrics['mre']) / dark_metrics['mre'] * 100
    print(f"Improvement over DARK alone: {improvement_over_dark:.1f}%")
    
    # =======================
    # STEP 4: DARK + FLIP + MEDIAN FILTERING
    # =======================
    print("\n" + "="*60)
    print("STEP 4: WITH DARK + FLIP TEST + MEDIAN FILTERING")
    print("Applying to: Sella (idx=0), Gonion (idx=10), PNS (idx=9)")
    print("="*60)
    
    problematic_indices = [0, 10, 9]  # Sella, Gonion, PNS
    
    for idx in range(len(results_flip['pred'])):
        pred_keypoints = results_flip['pred'][idx].copy()
        
        # Apply median filtering
        refined_keypoints = apply_median_filtering(
            pred_keypoints, 
            train_keypoints_list, 
            train_df, 
            problematic_indices=problematic_indices,
            k=5
        )
        
        results_median['pred'].append(refined_keypoints)
        results_median['gt'].append(results_flip['gt'][idx])
        
        if (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1}/{len(results_flip['pred'])} samples")
    
    # Compute final MRE with median filtering
    median_metrics = compute_mre_simple(results_median['pred'], results_median['gt'])
    print(f"\nFINAL MRE (DARK + FLIP + MEDIAN): {median_metrics['mre']:.3f} ± {median_metrics['std']:.3f} pixels")
    print(f"Median: {median_metrics['median']:.3f}, P90: {median_metrics['p90']:.3f}, P95: {median_metrics['p95']:.3f}")
    improvement_final = (baseline_metrics['mre'] - median_metrics['mre']) / baseline_metrics['mre'] * 100
    print(f"Total improvement over baseline: {improvement_final:.1f}%")
    
    # =======================
    # SUMMARY
    # =======================
    print("\n" + "="*80)
    print("LAYER 0 REFINEMENTS SUMMARY")
    print("="*80)
    print(f"{'Method':<30} {'MRE':<10} {'Improvement':<15}")
    print("-" * 55)
    print(f"{'Baseline':<30} {baseline_metrics['mre']:.3f} ± {baseline_metrics['std']:.3f}")
    print(f"{'+ DARK':<30} {dark_metrics['mre']:.3f} ± {dark_metrics['std']:.3f}  {improvement_dark:.1f}%")
    print(f"{'+ DARK + Flip Test':<30} {flip_metrics['mre']:.3f} ± {flip_metrics['std']:.3f}  {improvement_flip:.1f}%")
    print(f"{'+ DARK + Flip + Median':<30} {median_metrics['mre']:.3f} ± {median_metrics['std']:.3f}  {improvement_final:.1f}%")
    
    # Save results
    output_dir = os.path.join(work_dir, "layer0_refinements")
    os.makedirs(output_dir, exist_ok=True)
    
    summary_df = pd.DataFrame([
        {'method': 'Baseline', 'mre': baseline_metrics['mre'], 'std': baseline_metrics['std'], 
         'median': baseline_metrics['median'], 'p90': baseline_metrics['p90'], 'p95': baseline_metrics['p95']},
        {'method': 'DARK', 'mre': dark_metrics['mre'], 'std': dark_metrics['std'],
         'median': dark_metrics['median'], 'p90': dark_metrics['p90'], 'p95': dark_metrics['p95']},
        {'method': 'DARK+Flip', 'mre': flip_metrics['mre'], 'std': flip_metrics['std'],
         'median': flip_metrics['median'], 'p90': flip_metrics['p90'], 'p95': flip_metrics['p95']},
        {'method': 'DARK+Flip+Median', 'mre': median_metrics['mre'], 'std': median_metrics['std'],
         'median': median_metrics['median'], 'p90': median_metrics['p90'], 'p95': median_metrics['p95']}
    ])
    
    summary_path = os.path.join(output_dir, "refinement_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nResults saved to: {summary_path}")
    
    # Plot improvement graph
    plt.figure(figsize=(10, 6))
    methods = ['Baseline', 'DARK', 'DARK+Flip', 'DARK+Flip+Median']
    mres = [baseline_metrics['mre'], dark_metrics['mre'], flip_metrics['mre'], median_metrics['mre']]
    stds = [baseline_metrics['std'], dark_metrics['std'], flip_metrics['std'], median_metrics['std']]
    
    plt.errorbar(methods, mres, yerr=stds, marker='o', markersize=10, capsize=5, linewidth=2)
    plt.ylabel('Mean Radial Error (pixels)')
    plt.title('Layer 0 Refinements - Progressive Improvement')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=15)
    
    # Add improvement percentages
    for i in range(1, len(mres)):
        improvement = (mres[0] - mres[i]) / mres[0] * 100
        plt.text(i, mres[i] + stds[i] + 0.02, f'-{improvement:.1f}%', 
                ha='center', va='bottom', fontsize=10, color='green')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "refinement_progress.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")
    
    print("\n✅ Layer 0 refinements complete!")
    print(f"🎯 Final MRE: {median_metrics['mre']:.3f} pixels (from {baseline_metrics['mre']:.3f})")
    print(f"📈 Total improvement: {improvement_final:.1f}%")

if __name__ == "__main__":
    main() 