#!/usr/bin/env python3
"""
Test-Time Augmentation (TTA) Evaluation Script
Implements medium and heavy augmentation levels for improved predictions
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
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import rotate as scipy_rotate

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

def rotate_image_and_keypoints(image, keypoints, angle, center=None):
    """Rotate image and keypoints around center."""
    h, w = image.shape[:2]
    if center is None:
        center = (w/2, h/2)
    
    # Rotate image
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    
    # Rotate keypoints
    cos_a = np.cos(np.radians(-angle))  # Negative for inverse rotation
    sin_a = np.sin(np.radians(-angle))
    
    rotated_keypoints = keypoints.copy()
    for i in range(len(keypoints)):
        if keypoints[i, 0] > 0 and keypoints[i, 1] > 0:  # Valid keypoint
            # Translate to origin
            x = keypoints[i, 0] - center[0]
            y = keypoints[i, 1] - center[1]
            
            # Rotate
            new_x = x * cos_a - y * sin_a
            new_y = x * sin_a + y * cos_a
            
            # Translate back
            rotated_keypoints[i, 0] = new_x + center[0]
            rotated_keypoints[i, 1] = new_y + center[1]
    
    return rotated_image, rotated_keypoints, M

def scale_image_and_keypoints(image, keypoints, scale_factor):
    """Scale image and keypoints."""
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    # Scale image
    scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad or crop to original size
    if scale_factor > 1:
        # Crop center
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        scaled_image = scaled_image[start_h:start_h+h, start_w:start_w+w]
        
        # Adjust keypoints
        scaled_keypoints = keypoints.copy()
        scaled_keypoints[:, 0] = keypoints[:, 0] * scale_factor - start_w
        scaled_keypoints[:, 1] = keypoints[:, 1] * scale_factor - start_h
    else:
        # Pad
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        scaled_image = cv2.copyMakeBorder(scaled_image, pad_h, h-new_h-pad_h, 
                                         pad_w, w-new_w-pad_w, cv2.BORDER_CONSTANT)
        
        # Adjust keypoints
        scaled_keypoints = keypoints.copy()
        scaled_keypoints[:, 0] = keypoints[:, 0] * scale_factor + pad_w
        scaled_keypoints[:, 1] = keypoints[:, 1] * scale_factor + pad_h
    
    return scaled_image, scaled_keypoints

def apply_tta_inference(model, img_array, bbox, augmentation_level='medium'):
    """
    Apply test-time augmentation and average predictions.
    
    Args:
        model: The pose model
        img_array: Input image array
        bbox: Bounding box
        augmentation_level: 'medium' or 'heavy'
    
    Returns:
        Averaged keypoints after TTA
    """
    h, w = img_array.shape[:2]
    all_predictions = []
    
    if augmentation_level == 'medium':
        # Medium augmentation settings
        scales = [0.95, 1.0, 1.05]
        rotations = [-10, 0, 10]
        use_flip = True
    else:  # heavy
        # Heavy augmentation settings
        scales = [0.9, 0.95, 1.0, 1.05, 1.1]
        rotations = [-15, -10, -5, 0, 5, 10, 15]
        use_flip = True
    
    # Apply augmentations
    for scale in scales:
        for rotation in rotations:
            # Scale transformation
            if scale != 1.0:
                scaled_img, _ = scale_image_and_keypoints(img_array, np.zeros((19, 2)), scale)
            else:
                scaled_img = img_array.copy()
            
            # Rotation transformation
            if rotation != 0:
                rotated_img, _, M_rot = rotate_image_and_keypoints(scaled_img, np.zeros((19, 2)), rotation)
            else:
                rotated_img = scaled_img.copy()
                M_rot = None
            
            # Run inference
            results = inference_topdown(model, rotated_img, bboxes=bbox, bbox_format='xyxy')
            
            if results and len(results) > 0:
                pred_kpts = results[0].pred_instances.keypoints[0].copy()
                
                # Inverse transform predictions
                # Inverse rotation
                if rotation != 0:
                    M_inv = cv2.getRotationMatrix2D((w/2, h/2), -rotation, 1.0)
                    for i in range(len(pred_kpts)):
                        if pred_kpts[i, 0] > 0 and pred_kpts[i, 1] > 0:
                            pt = np.array([pred_kpts[i, 0], pred_kpts[i, 1], 1])
                            pred_kpts[i, :2] = M_inv.dot(pt)[:2]
                
                # Inverse scale
                if scale != 1.0:
                    if scale > 1:
                        start_w = (int(w * scale) - w) // 2
                        start_h = (int(h * scale) - h) // 2
                        pred_kpts[:, 0] = (pred_kpts[:, 0] + start_w) / scale
                        pred_kpts[:, 1] = (pred_kpts[:, 1] + start_h) / scale
                    else:
                        pad_w = (w - int(w * scale)) // 2
                        pad_h = (h - int(h * scale)) // 2
                        pred_kpts[:, 0] = (pred_kpts[:, 0] - pad_w) / scale
                        pred_kpts[:, 1] = (pred_kpts[:, 1] - pad_h) / scale
                
                all_predictions.append(pred_kpts)
            
            # Add flipped version
            if use_flip:
                flipped_img = np.fliplr(rotated_img).copy()
                results_flip = inference_topdown(model, flipped_img, bboxes=bbox, bbox_format='xyxy')
                
                if results_flip and len(results_flip) > 0:
                    pred_kpts_flip = results_flip[0].pred_instances.keypoints[0].copy()
                    
                    # Unflip
                    pred_kpts_flip[:, 0] = w - pred_kpts_flip[:, 0]
                    
                    # Inverse rotation
                    if rotation != 0:
                        M_inv = cv2.getRotationMatrix2D((w/2, h/2), -rotation, 1.0)
                        for i in range(len(pred_kpts_flip)):
                            if pred_kpts_flip[i, 0] > 0 and pred_kpts_flip[i, 1] > 0:
                                pt = np.array([pred_kpts_flip[i, 0], pred_kpts_flip[i, 1], 1])
                                pred_kpts_flip[i, :2] = M_inv.dot(pt)[:2]
                    
                    # Inverse scale
                    if scale != 1.0:
                        if scale > 1:
                            start_w = (int(w * scale) - w) // 2
                            start_h = (int(h * scale) - h) // 2
                            pred_kpts_flip[:, 0] = (pred_kpts_flip[:, 0] + start_w) / scale
                            pred_kpts_flip[:, 1] = (pred_kpts_flip[:, 1] + start_h) / scale
                        else:
                            pad_w = (w - int(w * scale)) // 2
                            pad_h = (h - int(h * scale)) // 2
                            pred_kpts_flip[:, 0] = (pred_kpts_flip[:, 0] - pad_w) / scale
                            pred_kpts_flip[:, 1] = (pred_kpts_flip[:, 1] - pad_h) / scale
                    
                    all_predictions.append(pred_kpts_flip)
    
    # Average all predictions
    if all_predictions:
        averaged_keypoints = np.mean(all_predictions, axis=0)
        return averaged_keypoints
    else:
        return None

def main():
    """Main evaluation function with TTA refinements."""
    
    print("="*80)
    print("TEST-TIME AUGMENTATION (TTA) EVALUATION")
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
    
    if test_df.empty:
        print("Test set empty, using validation set")
        test_df = main_df[main_df['set'] == 'dev'].reset_index(drop=True)
    
    print(f"Evaluating on {len(test_df)} test samples")
    
    # Get landmark information
    landmark_names = cephalometric_dataset_info.landmark_names_in_order
    landmark_cols = cephalometric_dataset_info.original_landmark_cols
    
    # Store results for each refinement level
    results_baseline = {'pred': [], 'gt': []}
    results_flip = {'pred': [], 'gt': []}
    results_medium_tta = {'pred': [], 'gt': []}
    results_heavy_tta = {'pred': [], 'gt': []}
    
    # Initialize model
    model = init_model(config_path, checkpoint_path, 
                      device='cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # =======================
    # STEP 1: BASELINE (No augmentation)
    # =======================
    print("\n" + "="*60)
    print("STEP 1: BASELINE EVALUATION (No augmentation)")
    print("="*60)
    
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
        results = inference_topdown(model, img_array, bboxes=bbox, bbox_format='xyxy')
        
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
    # STEP 2: SIMPLE FLIP TEST
    # =======================
    print("\n" + "="*60)
    print("STEP 2: SIMPLE FLIP TEST")
    print("="*60)
    
    for idx, row in test_df.iterrows():
        img_array = np.array(row['Image'], dtype=np.uint8).reshape((224, 224, 3))
        gt_keypoints = results_baseline['gt'][idx]
        
        bbox = np.array([[0, 0, 224, 224]], dtype=np.float32)
        
        # Original inference
        results_orig = inference_topdown(model, img_array, bboxes=bbox, bbox_format='xyxy')
        
        # Flipped inference
        img_flipped = np.fliplr(img_array).copy()
        results_flip_img = inference_topdown(model, img_flipped, bboxes=bbox, bbox_format='xyxy')
        
        if results_orig and results_flip_img and len(results_orig) > 0 and len(results_flip_img) > 0:
            kpts_orig = results_orig[0].pred_instances.keypoints[0]
            kpts_flip = results_flip_img[0].pred_instances.keypoints[0].copy()
            
            # Unflip
            kpts_flip[:, 0] = 224 - kpts_flip[:, 0]
            
            # Average
            kpts_averaged = (kpts_orig + kpts_flip) / 2.0
            
            results_flip['pred'].append(kpts_averaged)
            results_flip['gt'].append(gt_keypoints)
        
        if (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1}/{len(test_df)} samples")
    
    # Compute flip test MRE
    flip_metrics = compute_mre_simple(results_flip['pred'], results_flip['gt'])
    print(f"\nFLIP TEST MRE: {flip_metrics['mre']:.3f} ± {flip_metrics['std']:.3f} pixels")
    print(f"Median: {flip_metrics['median']:.3f}, P90: {flip_metrics['p90']:.3f}, P95: {flip_metrics['p95']:.3f}")
    improvement_flip = (baseline_metrics['mre'] - flip_metrics['mre']) / baseline_metrics['mre'] * 100
    print(f"Improvement over baseline: {improvement_flip:.1f}%")
    
    # =======================
    # STEP 3: MEDIUM TTA
    # =======================
    print("\n" + "="*60)
    print("STEP 3: MEDIUM TEST-TIME AUGMENTATION")
    print("Scales: [0.95, 1.0, 1.05], Rotations: [-10°, 0°, +10°], Flip: Yes")
    print("Total augmentations: 3 × 3 × 2 = 18 predictions per image")
    print("="*60)
    
    for idx, row in test_df.iterrows():
        img_array = np.array(row['Image'], dtype=np.uint8).reshape((224, 224, 3))
        gt_keypoints = results_baseline['gt'][idx]
        
        bbox = np.array([[0, 0, 224, 224]], dtype=np.float32)
        
        # Apply medium TTA
        pred_keypoints = apply_tta_inference(model, img_array, bbox, augmentation_level='medium')
        
        if pred_keypoints is not None:
            results_medium_tta['pred'].append(pred_keypoints)
            results_medium_tta['gt'].append(gt_keypoints)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(test_df)} samples")
    
    # Compute medium TTA MRE
    medium_metrics = compute_mre_simple(results_medium_tta['pred'], results_medium_tta['gt'])
    print(f"\nMEDIUM TTA MRE: {medium_metrics['mre']:.3f} ± {medium_metrics['std']:.3f} pixels")
    print(f"Median: {medium_metrics['median']:.3f}, P90: {medium_metrics['p90']:.3f}, P95: {medium_metrics['p95']:.3f}")
    improvement_medium = (baseline_metrics['mre'] - medium_metrics['mre']) / baseline_metrics['mre'] * 100
    print(f"Improvement over baseline: {improvement_medium:.1f}%")
    
    # =======================
    # STEP 4: HEAVY TTA
    # =======================
    print("\n" + "="*60)
    print("STEP 4: HEAVY TEST-TIME AUGMENTATION")
    print("Scales: [0.9, 0.95, 1.0, 1.05, 1.1], Rotations: [-15°, -10°, -5°, 0°, 5°, 10°, 15°], Flip: Yes")
    print("Total augmentations: 5 × 7 × 2 = 70 predictions per image")
    print("="*60)
    
    for idx, row in test_df.iterrows():
        img_array = np.array(row['Image'], dtype=np.uint8).reshape((224, 224, 3))
        gt_keypoints = results_baseline['gt'][idx]
        
        bbox = np.array([[0, 0, 224, 224]], dtype=np.float32)
        
        # Apply heavy TTA
        pred_keypoints = apply_tta_inference(model, img_array, bbox, augmentation_level='heavy')
        
        if pred_keypoints is not None:
            results_heavy_tta['pred'].append(pred_keypoints)
            results_heavy_tta['gt'].append(gt_keypoints)
        
        if (idx + 1) % 5 == 0:
            print(f"Processed {idx + 1}/{len(test_df)} samples")
    
    # Compute heavy TTA MRE
    heavy_metrics = compute_mre_simple(results_heavy_tta['pred'], results_heavy_tta['gt'])
    print(f"\nHEAVY TTA MRE: {heavy_metrics['mre']:.3f} ± {heavy_metrics['std']:.3f} pixels")
    print(f"Median: {heavy_metrics['median']:.3f}, P90: {heavy_metrics['p90']:.3f}, P95: {heavy_metrics['p95']:.3f}")
    improvement_heavy = (baseline_metrics['mre'] - heavy_metrics['mre']) / baseline_metrics['mre'] * 100
    print(f"Improvement over baseline: {improvement_heavy:.1f}%")
    
    # =======================
    # SUMMARY
    # =======================
    print("\n" + "="*80)
    print("TEST-TIME AUGMENTATION SUMMARY")
    print("="*80)
    print(f"{'Method':<25} {'MRE':<10} {'Improvement':<15} {'Inference Time':<20}")
    print("-" * 75)
    print(f"{'Baseline':<25} {baseline_metrics['mre']:.3f} ± {baseline_metrics['std']:.3f}  {'--':<15} {'1x':<20}")
    print(f"{'Flip Test':<25} {flip_metrics['mre']:.3f} ± {flip_metrics['std']:.3f}  {improvement_flip:.1f}%{'':>10} {'2x':<20}")
    print(f"{'Medium TTA':<25} {medium_metrics['mre']:.3f} ± {medium_metrics['std']:.3f}  {improvement_medium:.1f}%{'':>10} {'18x':<20}")
    print(f"{'Heavy TTA':<25} {heavy_metrics['mre']:.3f} ± {heavy_metrics['std']:.3f}  {improvement_heavy:.1f}%{'':>10} {'70x':<20}")
    
    # Save results
    output_dir = os.path.join(work_dir, "tta_refinements")
    os.makedirs(output_dir, exist_ok=True)
    
    summary_df = pd.DataFrame([
        {'method': 'Baseline', 'mre': baseline_metrics['mre'], 'std': baseline_metrics['std'], 
         'median': baseline_metrics['median'], 'p90': baseline_metrics['p90'], 'p95': baseline_metrics['p95'],
         'inference_multiplier': 1},
        {'method': 'Flip Test', 'mre': flip_metrics['mre'], 'std': flip_metrics['std'],
         'median': flip_metrics['median'], 'p90': flip_metrics['p90'], 'p95': flip_metrics['p95'],
         'inference_multiplier': 2},
        {'method': 'Medium TTA', 'mre': medium_metrics['mre'], 'std': medium_metrics['std'],
         'median': medium_metrics['median'], 'p90': medium_metrics['p90'], 'p95': medium_metrics['p95'],
         'inference_multiplier': 18},
        {'method': 'Heavy TTA', 'mre': heavy_metrics['mre'], 'std': heavy_metrics['std'],
         'median': heavy_metrics['median'], 'p90': heavy_metrics['p90'], 'p95': heavy_metrics['p95'],
         'inference_multiplier': 70}
    ])
    
    summary_path = os.path.join(output_dir, "tta_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nResults saved to: {summary_path}")
    
    # Plot improvement graph
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # MRE improvement plot
    methods = ['Baseline', 'Flip Test', 'Medium TTA', 'Heavy TTA']
    mres = [baseline_metrics['mre'], flip_metrics['mre'], medium_metrics['mre'], heavy_metrics['mre']]
    stds = [baseline_metrics['std'], flip_metrics['std'], medium_metrics['std'], heavy_metrics['std']]
    
    ax1.errorbar(methods, mres, yerr=stds, marker='o', markersize=10, capsize=5, linewidth=2)
    ax1.set_ylabel('Mean Radial Error (pixels)')
    ax1.set_title('Test-Time Augmentation - Progressive Improvement')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels(methods, rotation=15)
    
    # Add improvement percentages
    for i in range(1, len(mres)):
        improvement = (mres[0] - mres[i]) / mres[0] * 100
        ax1.text(i, mres[i] + stds[i] + 0.02, f'-{improvement:.1f}%', 
                ha='center', va='bottom', fontsize=10, color='green')
    
    # Accuracy vs Inference Time trade-off
    inference_times = [1, 2, 18, 70]
    improvements = [0, improvement_flip, improvement_medium, improvement_heavy]
    
    ax2.scatter(inference_times, improvements, s=100, alpha=0.7)
    for i, method in enumerate(methods):
        ax2.annotate(method, (inference_times[i], improvements[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    ax2.set_xlabel('Inference Time Multiplier')
    ax2.set_ylabel('Improvement over Baseline (%)')
    ax2.set_title('Accuracy vs Inference Time Trade-off')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "tta_analysis.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")
    
    print("\n✅ Test-time augmentation evaluation complete!")
    print(f"🎯 Best MRE: {heavy_metrics['mre']:.3f} pixels (Heavy TTA)")
    print(f"⚡ Fast option: {flip_metrics['mre']:.3f} pixels (Flip Test, only 2x slower)")
    print(f"📈 Total improvement: {improvement_heavy:.1f}% (but 70x slower)")

if __name__ == "__main__":
    main() 