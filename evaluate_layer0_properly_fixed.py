#!/usr/bin/env python3
"""
Layer 0 Refinements Evaluation Script - PROPERLY FIXED VERSION
Fixes all identified issues with config loading, image handling, and DARK implementation
"""

import os
import torch
import warnings
import pandas as pd
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.apis import init_model, inference_topdown
from mmengine.runner import load_checkpoint
import glob
from scipy.spatial import procrustes
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
    
    # Need at least 6 points for stable alignment
    if np.sum(valid_mask) < 6:
        return source_landmarks.copy()
    
    source_valid = source_landmarks[valid_mask]
    target_valid = target_landmarks[valid_mask]
    
    try:
        # Apply Procrustes
        _, aligned_source, _ = procrustes(target_valid, source_valid)
        
        # Apply transformation to all landmarks
        result = source_landmarks.copy()
        result[valid_mask] = aligned_source
        
        return result
    except:
        # If Procrustes fails, return original
        return source_landmarks.copy()

def apply_median_filtering(pred_keypoints, train_keypoints_list, 
                          problematic_indices=[0, 10, 9], k=5):
    """
    Apply median filtering for problematic landmarks using k-NN in shape space.
    
    Args:
        pred_keypoints: Predicted keypoints for test sample (19, 2)
        train_keypoints_list: List of training keypoints arrays (pre-filtered for valid points)
        problematic_indices: Indices of landmarks to apply filtering (Sella=0, Gonion=10, PNS=9)
        k: Number of nearest neighbors
    """
    refined_keypoints = pred_keypoints.copy()
    
    # Compute distances to all training samples after alignment
    distances = []
    aligned_train_keypoints = []
    
    for train_kpts in train_keypoints_list:
        # Check if this training sample has enough valid points
        valid_train_mask = (train_kpts[:, 0] > 0) & (train_kpts[:, 1] > 0)
        valid_pred_mask = (pred_keypoints[:, 0] > 0) & (pred_keypoints[:, 1] > 0)
        common_valid = valid_train_mask & valid_pred_mask
        
        if np.sum(common_valid) < 6:  # Need at least 6 common valid points
            distances.append(np.inf)
            aligned_train_keypoints.append(train_kpts.copy())
            continue
        
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
    valid_distances = distances[distances < np.inf]
    
    if len(valid_distances) < k:
        print(f"Warning: Only {len(valid_distances)} valid neighbors found, using all")
        k_nearest_indices = np.where(distances < np.inf)[0]
    else:
        k_nearest_indices = np.argsort(distances)[:k]
    
    # Apply median filtering for problematic landmarks
    for landmark_idx in problematic_indices:
        neighbor_coords = []
        
        for train_idx in k_nearest_indices:
            if train_idx < len(aligned_train_keypoints):
                train_kpts = aligned_train_keypoints[train_idx]
                if train_kpts[landmark_idx, 0] > 0 and train_kpts[landmark_idx, 1] > 0:
                    neighbor_coords.append(train_kpts[landmark_idx])
        
        if len(neighbor_coords) >= 3:  # Need at least 3 neighbors
            neighbor_coords = np.array(neighbor_coords)
            median_coord = np.median(neighbor_coords, axis=0)
            refined_keypoints[landmark_idx] = median_coord
    
    return refined_keypoints

def main():
    """Main evaluation function with Layer 0 refinements - PROPERLY FIXED."""
    
    print("="*80)
    print("LAYER 0 REFINEMENTS - PROPERLY FIXED VERSION")
    print("="*80)
    
    # Initialize MMPose scope
    init_default_scope('mmpose')
    
    # Import custom modules
    import custom_cephalometric_dataset
    import custom_transforms
    import cephalometric_dataset_info
    
    work_dir = "work_dirs/hrnetv2_w18_cephalometric_384x384_adaptive_wing_loss_v4"
    
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
    
    # Prepare training keypoints for median filtering (filter for quality)
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
        
        gt_keypoints = np.array(gt_keypoints)
        # Only keep training samples with at least 15 valid landmarks
        valid_count = np.sum((gt_keypoints[:, 0] > 0) & (gt_keypoints[:, 1] > 0))
        if valid_count >= 15:
            train_keypoints_list.append(gt_keypoints)
    
    print(f"Filtered training set: {len(train_keypoints_list)} samples with ≥15 valid landmarks")
    
    # Store results for each refinement level
    results_baseline = {'pred': [], 'gt': []}
    results_dark = {'pred': [], 'gt': []}
    results_flip = {'pred': [], 'gt': []}
    results_median = {'pred': [], 'gt': []}
    
    # =======================
    # STEP 1: BASELINE (Using checkpoint's original config)
    # =======================
    print("\n" + "="*60)
    print("STEP 1: BASELINE EVALUATION (Original checkpoint config)")
    print("="*60)
    
    # Load model using checkpoint's internal config (preserves original test_cfg)
    model_baseline = init_model(None, checkpoint_path, 
                               device='cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Print the config being used
    print(f"Baseline test_cfg: {model_baseline.cfg.model.get('test_cfg', 'None')}")
    
    for idx, row in test_df.iterrows():
        # Get image (224x224) - DON'T resize, let inference_topdown handle it
        img_array = np.array(row['Image'], dtype=np.uint8).reshape((224, 224, 3))
        
        # Get ground truth (in 224x224 space)
        gt_keypoints = []
        for i in range(0, len(landmark_cols), 2):
            x_col = landmark_cols[i]
            y_col = landmark_cols[i+1]
            if x_col in row and y_col in row and pd.notna(row[x_col]) and pd.notna(row[y_col]):
                gt_keypoints.append([row[x_col], row[y_col]])
            else:
                gt_keypoints.append([0, 0])
        gt_keypoints = np.array(gt_keypoints)
        
        # Prepare bbox for 224x224 (inference_topdown will resize internally)
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
    
    # Load model and enable DARK
    model_dark = init_model(None, checkpoint_path,
                           device='cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Properly enable DARK while preserving other settings
    if hasattr(model_dark.cfg.model, 'test_cfg') and model_dark.cfg.model.test_cfg is not None:
        model_dark.cfg.model.test_cfg.use_dark = True
        model_dark.cfg.model.test_cfg.flip_test = False
    else:
        model_dark.cfg.model.test_cfg = dict(
            flip_test=False,
            shift_heatmap=True,
            use_dark=True
        )
    
    print(f"DARK test_cfg: {model_dark.cfg.model.test_cfg}")
    
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
    
    # Load model and enable DARK + flip test
    model_flip = init_model(None, checkpoint_path,
                           device='cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if hasattr(model_flip.cfg.model, 'test_cfg') and model_flip.cfg.model.test_cfg is not None:
        model_flip.cfg.model.test_cfg.use_dark = True
        model_flip.cfg.model.test_cfg.flip_test = True
    else:
        model_flip.cfg.model.test_cfg = dict(
            flip_test=True,
            shift_heatmap=True,
            use_dark=True
        )
    
    print(f"DARK+Flip test_cfg: {model_flip.cfg.model.test_cfg}")
    
    for idx, row in test_df.iterrows():
        img_array = np.array(row['Image'], dtype=np.uint8).reshape((224, 224, 3))
        gt_keypoints = results_baseline['gt'][idx]
        
        bbox = np.array([[0, 0, 224, 224]], dtype=np.float32)
        results = inference_topdown(model_flip, img_array, bboxes=bbox, bbox_format='xyxy')
        
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
    print("LAYER 0 REFINEMENTS SUMMARY (PROPERLY FIXED)")
    print("="*80)
    print(f"{'Method':<30} {'MRE':<12} {'Improvement':<15}")
    print("-" * 60)
    print(f"{'Baseline (original cfg)':<30} {baseline_metrics['mre']:.3f} ± {baseline_metrics['std']:.3f}")
    print(f"{'+ DARK':<30} {dark_metrics['mre']:.3f} ± {dark_metrics['std']:.3f}  {improvement_dark:+.1f}%")
    print(f"{'+ DARK + Flip Test':<30} {flip_metrics['mre']:.3f} ± {flip_metrics['std']:.3f}  {improvement_flip:+.1f}%")
    print(f"{'+ DARK + Flip + Median':<30} {median_metrics['mre']:.3f} ± {median_metrics['std']:.3f}  {improvement_final:+.1f}%")
    
    # Save results
    output_dir = os.path.join(work_dir, "layer0_refinements_properly_fixed")
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
    plt.figure(figsize=(12, 6))
    methods = ['Baseline', 'DARK', 'DARK+Flip', 'DARK+Flip+Median']
    mres = [baseline_metrics['mre'], dark_metrics['mre'], flip_metrics['mre'], median_metrics['mre']]
    stds = [baseline_metrics['std'], dark_metrics['std'], flip_metrics['std'], median_metrics['std']]
    
    plt.errorbar(methods, mres, yerr=stds, marker='o', markersize=10, capsize=5, linewidth=2)
    plt.ylabel('Mean Radial Error (pixels)')
    plt.title('Layer 0 Refinements - Progressive Improvement (Properly Fixed)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=15)
    
    # Add improvement percentages
    for i in range(1, len(mres)):
        improvement = (mres[0] - mres[i]) / mres[0] * 100
        color = 'green' if improvement > 0 else 'red'
        plt.text(i, mres[i] + stds[i] + 0.05, f'{improvement:+.1f}%', 
                ha='center', va='bottom', fontsize=10, color=color, weight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "refinement_progress.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")
    
    print("\n✅ Layer 0 refinements complete!")
    print(f"🎯 Final MRE: {median_metrics['mre']:.3f} pixels (from {baseline_metrics['mre']:.3f})")
    print(f"📈 Total improvement: {improvement_final:+.1f}%")
    
    if improvement_final > 5:
        print("🎉 Great! The refinements are working as expected.")
    elif improvement_final > 2:
        print("✓ Good improvements, consider trying Layer 1 refinements next.")
    else:
        print("⚠️  Small improvements. Check if model and data are compatible.")

if __name__ == "__main__":
    main() 