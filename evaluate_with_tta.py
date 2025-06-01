#!/usr/bin/env python3
"""
Heavy Test-Time Augmentation (TTA) evaluation script for cephalometric landmark detection.
Applies multiple augmentations during inference and averages results for improved accuracy.
"""

import os
import torch
import warnings
import pandas as pd
import numpy as np
import argparse
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.apis import init_model
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from typing import List, Tuple, Dict
import copy

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

class TTATransforms:
    """Test-time augmentation transforms for cephalometric landmark detection."""
    
    def __init__(self, input_size=(384, 384)):
        self.input_size = input_size
        
    def get_tta_transforms(self):
        """Get list of TTA transform configurations."""
        transforms = []
        
        # 1. Original image (no augmentation)
        transforms.append({
            'name': 'original',
            'flip': False,
            'scale': 1.0,
            'rotation': 0,
            'brightness': 1.0,
            'contrast': 1.0,
            'crop_shift': (0, 0)
        })
        
        # 2. Horizontal flip
        transforms.append({
            'name': 'flip_h',
            'flip': True,
            'scale': 1.0,
            'rotation': 0,
            'brightness': 1.0,
            'contrast': 1.0,
            'crop_shift': (0, 0)
        })
        
        # 3. Multiple scales
        for scale in [0.9, 1.1, 1.2]:
            transforms.append({
                'name': f'scale_{scale}',
                'flip': False,
                'scale': scale,
                'rotation': 0,
                'brightness': 1.0,
                'contrast': 1.0,
                'crop_shift': (0, 0)
            })
            
        # 4. Small rotations
        for rotation in [-5, 5, -10, 10]:
            transforms.append({
                'name': f'rot_{rotation}',
                'flip': False,
                'scale': 1.0,
                'rotation': rotation,
                'brightness': 1.0,
                'contrast': 1.0,
                'crop_shift': (0, 0)
            })
            
        # 5. Brightness variations
        for brightness in [0.9, 1.1]:
            transforms.append({
                'name': f'bright_{brightness}',
                'flip': False,
                'scale': 1.0,
                'rotation': 0,
                'brightness': brightness,
                'contrast': 1.0,
                'crop_shift': (0, 0)
            })
            
        # 6. Contrast variations
        for contrast in [0.9, 1.1]:
            transforms.append({
                'name': f'contrast_{contrast}',
                'flip': False,
                'scale': 1.0,
                'rotation': 0,
                'brightness': 1.0,
                'contrast': contrast,
                'crop_shift': (0, 0)
            })
            
        # 7. Small crops/shifts
        shift_pixels = 5
        for dx, dy in [(-shift_pixels, 0), (shift_pixels, 0), (0, -shift_pixels), (0, shift_pixels)]:
            transforms.append({
                'name': f'shift_{dx}_{dy}',
                'flip': False,
                'scale': 1.0,
                'rotation': 0,
                'brightness': 1.0,
                'contrast': 1.0,
                'crop_shift': (dx, dy)
            })
            
        # 8. Combined augmentations (most promising combinations)
        transforms.extend([
            {
                'name': 'flip_scale_0.9',
                'flip': True,
                'scale': 0.9,
                'rotation': 0,
                'brightness': 1.0,
                'contrast': 1.0,
                'crop_shift': (0, 0)
            },
            {
                'name': 'flip_scale_1.1',
                'flip': True,
                'scale': 1.1,
                'rotation': 0,
                'brightness': 1.0,
                'contrast': 1.0,
                'crop_shift': (0, 0)
            },
            {
                'name': 'scale_1.1_rot_5',
                'flip': False,
                'scale': 1.1,
                'rotation': 5,
                'brightness': 1.0,
                'contrast': 1.0,
                'crop_shift': (0, 0)
            }
        ])
        
        return transforms
    
    def apply_transform(self, image: np.ndarray, transform_config: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a single TTA transform to an image.
        
        Args:
            image: Input image (H, W, 3)
            transform_config: Transform configuration
            
        Returns:
            transformed_image: Augmented image
            inverse_transform_matrix: Matrix to transform predictions back
        """
        h, w = image.shape[:2]
        transformed_image = image.copy()
        
        # Create transformation matrix for keypoint inverse transformation
        transform_matrix = np.eye(3)
        
        # 1. Brightness and contrast adjustments
        if transform_config['brightness'] != 1.0 or transform_config['contrast'] != 1.0:
            transformed_image = transformed_image.astype(np.float32)
            transformed_image = transformed_image * transform_config['contrast'] + \
                               (transform_config['brightness'] - 1.0) * 127.5
            transformed_image = np.clip(transformed_image, 0, 255).astype(np.uint8)
        
        # 2. Create affine transformation matrix
        center = (w / 2, h / 2)
        
        # Scale and rotation
        scale = transform_config['scale']
        angle = transform_config['rotation']
        M = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Add crop shift
        M[0, 2] += transform_config['crop_shift'][0]
        M[1, 2] += transform_config['crop_shift'][1]
        
        # Apply affine transformation
        transformed_image = cv2.warpAffine(transformed_image, M, (w, h), 
                                          flags=cv2.INTER_LINEAR, 
                                          borderMode=cv2.BORDER_REFLECT_101)
        
        # 3. Horizontal flip
        if transform_config['flip']:
            transformed_image = cv2.flip(transformed_image, 1)
            flip_matrix = np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1]])
            transform_matrix = flip_matrix @ transform_matrix
        
        # Store transformation info for inverse
        if scale != 1.0 or angle != 0 or transform_config['crop_shift'] != (0, 0):
            # Create 3x3 transformation matrix
            M_3x3 = np.vstack([M, [0, 0, 1]])
            transform_matrix = M_3x3 @ transform_matrix
        
        return transformed_image, transform_matrix
    
    def inverse_transform_keypoints(self, keypoints: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """
        Apply inverse transformation to keypoints.
        
        Args:
            keypoints: Predicted keypoints (N, 2)
            transform_matrix: 3x3 transformation matrix
            
        Returns:
            transformed_keypoints: Keypoints in original coordinate system
        """
        if keypoints.shape[0] == 0:
            return keypoints
            
        # Convert to homogeneous coordinates
        ones = np.ones((keypoints.shape[0], 1))
        keypoints_homo = np.hstack([keypoints, ones])
        
        # Apply inverse transformation
        try:
            inv_matrix = np.linalg.inv(transform_matrix)
            transformed_keypoints_homo = (inv_matrix @ keypoints_homo.T).T
            transformed_keypoints = transformed_keypoints_homo[:, :2]
        except np.linalg.LinAlgError:
            # If matrix is singular, return original keypoints
            transformed_keypoints = keypoints
            
        return transformed_keypoints

def inference_with_tta(model, image: np.ndarray, tta_transforms: TTATransforms, 
                      flip_indices: List[int] = None) -> np.ndarray:
    """
    Perform inference with test-time augmentation.
    
    Args:
        model: MMPose model
        image: Input image (H, W, 3)
        tta_transforms: TTA transform handler
        flip_indices: Indices for flipping keypoints (for symmetric landmarks)
        
    Returns:
        averaged_keypoints: TTA-averaged keypoints (N, 2)
    """
    from mmpose.apis import inference_topdown
    
    transforms_list = tta_transforms.get_tta_transforms()
    all_predictions = []
    
    bbox = np.array([[0, 0, image.shape[1], image.shape[0]]], dtype=np.float32)
    
    for transform_config in transforms_list:
        # Apply transformation
        transformed_image, transform_matrix = tta_transforms.apply_transform(image, transform_config)
        
        # Run inference
        try:
            results = inference_topdown(model, transformed_image, bboxes=bbox, bbox_format='xyxy')
            
            if results and len(results) > 0:
                pred_keypoints = results[0].pred_instances.keypoints[0]  # Shape: (N, 2)
                
                # Handle horizontal flip for symmetric landmarks
                if transform_config['flip'] and flip_indices is not None:
                    pred_keypoints = pred_keypoints[flip_indices]
                
                # Apply inverse transformation
                original_keypoints = tta_transforms.inverse_transform_keypoints(pred_keypoints, transform_matrix)
                all_predictions.append(original_keypoints)
            
        except Exception as e:
            print(f"Warning: TTA transform '{transform_config['name']}' failed: {e}")
            continue
    
    if len(all_predictions) == 0:
        print("Warning: All TTA transforms failed")
        return np.zeros((19, 2))  # Return zeros if all fail
    
    # Average all predictions
    averaged_keypoints = np.mean(all_predictions, axis=0)
    
    print(f"TTA: Successfully averaged {len(all_predictions)}/{len(transforms_list)} predictions")
    return averaged_keypoints

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

def plot_error_distribution(all_errors, landmark_stats, save_path=None, title_suffix=""):
    """Plot error distribution and per-landmark statistics."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Overall error histogram
    ax1.hist(all_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Radial Error (pixels)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Overall Error Distribution {title_suffix}\nMRE: {np.mean(all_errors):.2f} ± {np.std(all_errors):.2f} pixels')
    ax1.grid(True, alpha=0.3)
    
    # Box plot of overall errors
    ax2.boxplot(all_errors, patch_artist=True, 
                boxprops=dict(facecolor='lightcoral', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('Radial Error (pixels)')
    ax2.set_title(f'Overall Error Box Plot {title_suffix}')
    ax2.grid(True, alpha=0.3)
    
    # Per-landmark MRE bar plot
    landmark_names = list(landmark_stats.keys())
    mres = [landmark_stats[name]['mre'] for name in landmark_names]
    stds = [landmark_stats[name]['std'] for name in landmark_names]
    
    bars = ax3.bar(range(len(landmark_names)), mres, yerr=stds, 
                   alpha=0.7, color='lightgreen', capsize=5)
    ax3.set_xlabel('Landmark Index')
    ax3.set_ylabel('MRE (pixels)')
    ax3.set_title(f'Per-Landmark MRE {title_suffix}')
    ax3.set_xticks(range(len(landmark_names)))
    ax3.set_xticklabels([f'{i}' for i in range(len(landmark_names))], rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Per-landmark error counts
    counts = [landmark_stats[name]['count'] for name in landmark_names]
    ax4.bar(range(len(landmark_names)), counts, alpha=0.7, color='orange')
    ax4.set_xlabel('Landmark Index')
    ax4.set_ylabel('Number of Valid Predictions')
    ax4.set_title(f'Per-Landmark Detection Count {title_suffix}')
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
    """Main evaluation function with TTA."""
    
    parser = argparse.ArgumentParser(description='Evaluate cephalometric model with Test-Time Augmentation')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--config', type=str, 
                       default="Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py",
                       help='Path to config file')
    parser.add_argument('--work_dir', type=str, 
                       default="work_dirs/hrnetv2_w18_cephalometric_384x384_adaptive_wing_loss_v4",
                       help='Work directory to search for checkpoints if --checkpoint not specified')
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("HEAVY TEST-TIME AUGMENTATION EVALUATION")
    print("="*80)
    
    # Initialize MMPose scope
    init_default_scope('mmpose')
    
    # Import custom modules
    import custom_cephalometric_dataset
    import custom_transforms
    import cephalometric_dataset_info
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint file not found: {checkpoint_path}")
            return
    else:
        # Find the best checkpoint in work_dir
        checkpoint_pattern = os.path.join(args.work_dir, "best_NME_epoch_*.pth")
        checkpoints = glob.glob(checkpoint_pattern)
        
        if not checkpoints:
            checkpoint_pattern = os.path.join(args.work_dir, "epoch_*.pth")
            checkpoints = glob.glob(checkpoint_pattern)
        
        if not checkpoints:
            print(f"ERROR: No checkpoints found in {args.work_dir}")
            return
        
        checkpoint_path = max(checkpoints, key=os.path.getctime)
    
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Using config: {args.config}")
    
    # Initialize model
    model = init_model(args.config, checkpoint_path, device='cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Model loaded successfully")
    
    # Load test data
    data_file_path = "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json"
    main_df = pd.read_json(data_file_path)
    test_df = main_df[main_df['set'] == 'test'].reset_index(drop=True)
    
    if test_df.empty:
        print("Test set empty, using validation set")
        test_df = main_df[main_df['set'] == 'dev'].reset_index(drop=True)
    
    print(f"Evaluating on {len(test_df)} samples with TTA")
    
    # Get landmark information
    landmark_names = cephalometric_dataset_info.landmark_names_in_order
    landmark_cols = cephalometric_dataset_info.original_landmark_cols
    
    # Initialize TTA transforms
    tta_transforms = TTATransforms(input_size=(384, 384))
    
    # Define flip indices for symmetric landmarks (if applicable)
    # For cephalometric landmarks, you may need to define which landmarks are symmetric
    # This is optional and depends on your specific landmark annotation
    flip_indices = None  # Set to appropriate indices if you have symmetric landmarks
    
    # Collect predictions and ground truth
    pred_keypoints_list = []
    gt_keypoints_list = []
    sample_info = []
    
    print("\nRunning inference with TTA on test samples...")
    print(f"Number of TTA transforms: {len(tta_transforms.get_tta_transforms())}")
    
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
        
        # Run inference with TTA
        try:
            pred_keypoints = inference_with_tta(model, img_array, tta_transforms, flip_indices)
            
            pred_keypoints_list.append(pred_keypoints)
            gt_keypoints_list.append(gt_keypoints)
            sample_info.append({
                'patient_id': row.get('patient_id', f'sample_{idx}'),
                'index': idx
            })
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
        
        if (idx + 1) % 5 == 0:
            print(f"Processed {idx + 1}/{len(test_df)} samples")
    
    # Compute detailed metrics
    print("\nComputing MRE and detailed statistics...")
    metrics = compute_mre_metrics(pred_keypoints_list, gt_keypoints_list, landmark_names)
    
    # Print results
    print("\n" + "="*60)
    print("TTA EVALUATION RESULTS")
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
    print("PER-LANDMARK STATISTICS (with TTA)")
    print("="*60)
    print(f"{'Index':<5} {'Landmark':<20} {'MRE':<8} {'Std':<8} {'Median':<8} {'Count':<6}")
    print("-" * 60)
    
    for i, name in enumerate(landmark_names):
        stats = metrics['per_landmark_stats'][name]
        print(f"{i:<5} {name:<20} {stats['mre']:<8.3f} {stats['std']:<8.3f} "
              f"{stats['median']:<8.3f} {stats['count']:<6}")
    
    # Save detailed results
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.work_dir, "tta_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to CSV
    results_df = pd.DataFrame([
        {
            'landmark_index': i,
            'landmark_name': name,
            'mre_pixels_tta': metrics['per_landmark_stats'][name]['mre'],
            'std_pixels_tta': metrics['per_landmark_stats'][name]['std'],
            'median_pixels_tta': metrics['per_landmark_stats'][name]['median'],
            'min_pixels_tta': metrics['per_landmark_stats'][name]['min'],
            'max_pixels_tta': metrics['per_landmark_stats'][name]['max'],
            'valid_count_tta': metrics['per_landmark_stats'][name]['count']
        }
        for i, name in enumerate(landmark_names)
    ])
    
    csv_path = os.path.join(output_dir, "per_landmark_mre_results_tta.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nDetailed TTA results saved to: {csv_path}")
    
    # Plot error analysis
    plot_path = os.path.join(output_dir, "error_analysis_tta.png")
    plot_error_distribution(metrics['all_errors'], metrics['per_landmark_stats'], 
                          plot_path, title_suffix="(with TTA)")
    
    # Save overall summary
    summary = {
        'overall_mre_pixels_tta': metrics['overall_mre'],
        'overall_std_pixels_tta': metrics['overall_std'],
        'valid_predictions_tta': metrics['valid_predictions'],
        'total_predictions_tta': metrics['total_predictions'],
        'detection_rate_tta': metrics['detection_rate'],
        'median_error_pixels_tta': np.median(metrics['all_errors']) if len(metrics['all_errors']) > 0 else 0,
        'p90_error_pixels_tta': np.percentile(metrics['all_errors'], 90) if len(metrics['all_errors']) > 0 else 0,
        'p95_error_pixels_tta': np.percentile(metrics['all_errors'], 95) if len(metrics['all_errors']) > 0 else 0,
        'checkpoint_path': checkpoint_path,
        'num_tta_transforms': len(tta_transforms.get_tta_transforms())
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(output_dir, "overall_summary_tta.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print(f"Overall TTA summary saved to: {summary_path}")
    print(f"TTA evaluation complete! Results saved to: {output_dir}")
    print(f"Used {len(tta_transforms.get_tta_transforms())} TTA transforms")

if __name__ == "__main__":
    main() 