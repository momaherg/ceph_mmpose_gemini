#!/usr/bin/env python3
"""
Evaluation Script for HRNetV2 with Native Skeletal Classification
=================================================================
This script evaluates models that predict both landmarks and skeletal classification.

It evaluates:
1. Keypoint detection performance (MRE, per-landmark metrics)
2. Native classification performance (accuracy, confusion matrix, per-class metrics)
3. Post-hoc classification from predicted landmarks (for comparison)
4. Agreement between native and post-hoc classification

Usage:
    python evaluate_classification_model.py --work_dir work_dirs/hrnetv2_w18_cephalometric_ensemble_concurrent_mlp_v5
"""

import os
import sys
import torch
import warnings
import pandas as pd
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.apis import init_model, inference_topdown
import glob
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Tuple, Optional
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


def compute_keypoint_metrics(pred_coords: np.ndarray, gt_coords: np.ndarray, 
                           landmark_names: List[str]) -> Tuple[Dict, Dict]:
    """Compute comprehensive keypoint evaluation metrics."""
    # Compute radial errors
    radial_errors = np.sqrt(np.sum((pred_coords - gt_coords)**2, axis=2))
    
    # Overall metrics
    valid_mask = (gt_coords[:, :, 0] > 0) & (gt_coords[:, :, 1] > 0)
    valid_errors = radial_errors[valid_mask]
    
    overall_metrics = {
        'mre': np.mean(valid_errors),
        'std': np.std(valid_errors),
        'median': np.median(valid_errors),
        'p90': np.percentile(valid_errors, 90),
        'p95': np.percentile(valid_errors, 95),
        'max': np.max(valid_errors),
        'count': len(valid_errors)
    }
    
    # Per-landmark metrics
    per_landmark_metrics = {}
    for i, name in enumerate(landmark_names):
        landmark_errors = radial_errors[:, i]
        landmark_valid = valid_mask[:, i]
        
        if np.any(landmark_valid):
            valid_landmark_errors = landmark_errors[landmark_valid]
            per_landmark_metrics[name] = {
                'mre': np.mean(valid_landmark_errors),
                'std': np.std(valid_landmark_errors),
                'median': np.median(valid_landmark_errors),
                'count': len(valid_landmark_errors)
            }
        else:
            per_landmark_metrics[name] = {'mre': 0, 'std': 0, 'median': 0, 'count': 0}
    
    return overall_metrics, per_landmark_metrics


def compute_classification_metrics(pred_classes: np.ndarray, gt_classes: np.ndarray,
                                 class_names: List[str]) -> Dict:
    """Compute classification metrics including confusion matrix and per-class metrics."""
    # Filter out invalid samples (-1 values)
    valid_mask = (gt_classes >= 0) & (pred_classes >= 0)
    pred_valid = pred_classes[valid_mask]
    gt_valid = gt_classes[valid_mask]
    
    if len(pred_valid) == 0:
        return {
            'accuracy': 0.0,
            'confusion_matrix': np.zeros((len(class_names), len(class_names))),
            'classification_report': {},
            'per_class_accuracy': {name: 0.0 for name in class_names},
            'n_samples': 0
        }
    
    # Overall accuracy
    accuracy = np.mean(pred_valid == gt_valid)
    
    # Confusion matrix
    cm = confusion_matrix(gt_valid, pred_valid, labels=list(range(len(class_names))))
    
    # Classification report
    report = classification_report(gt_valid, pred_valid, 
                                 labels=list(range(len(class_names))),
                                 target_names=class_names,
                                 output_dict=True,
                                 zero_division=0)
    
    # Per-class accuracy
    per_class_accuracy = {}
    for i, class_name in enumerate(class_names):
        class_mask = gt_valid == i
        if np.any(class_mask):
            class_acc = np.mean(pred_valid[class_mask] == i)
            per_class_accuracy[class_name] = class_acc
        else:
            per_class_accuracy[class_name] = 0.0
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'per_class_accuracy': per_class_accuracy,
        'n_samples': len(pred_valid)
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         title: str, save_path: str):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Frequency'})
    
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_classification_comparison(native_cm: np.ndarray, posthoc_cm: np.ndarray,
                                 class_names: List[str], save_path: str):
    """Plot side-by-side comparison of native and post-hoc classification."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Native classification
    cm1_normalized = native_cm.astype('float') / native_cm.sum(axis=1)[:, np.newaxis]
    cm1_normalized = np.nan_to_num(cm1_normalized)
    
    sns.heatmap(cm1_normalized, annot=native_cm, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Frequency'}, ax=ax1)
    ax1.set_title('Native Classification (from Neural Network)')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Post-hoc classification
    cm2_normalized = posthoc_cm.astype('float') / posthoc_cm.sum(axis=1)[:, np.newaxis]
    cm2_normalized = np.nan_to_num(cm2_normalized)
    
    sns.heatmap(cm2_normalized, annot=posthoc_cm, fmt='d', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Frequency'}, ax=ax2)
    ax2.set_title('Post-hoc Classification (from Predicted Landmarks)')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main evaluation function."""
    
    parser = argparse.ArgumentParser(
        description='Evaluate HRNetV2 with Native Skeletal Classification')
    parser.add_argument(
        '--work_dir',
        type=str,
        default='work_dirs/hrnetv2_w18_cephalometric_ensemble_concurrent_mlp_v5',
        help='Work directory containing the trained models'
    )
    parser.add_argument(
        '--model_idx',
        type=int,
        default=1,
        help='Model index to evaluate (for ensemble training)'
    )
    parser.add_argument(
        '--test_split_file',
        type=str,
        default=None,
        help='Path to a text file containing patient IDs for the test set'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=None,
        help='Specific epoch to evaluate (default: best or latest)'
    )
    parser.add_argument(
        '--data_file',
        type=str,
        default='/content/drive/MyDrive/Lala\'s Masters/train_data_pure_old_numpy.json',
        help='Path to the data JSON file'
    )
    args = parser.parse_args()
    
    print("="*80)
    print("EVALUATION: HRNetV2 WITH NATIVE SKELETAL CLASSIFICATION")
    print("="*80)
    
    # Initialize MMPose scope
    init_default_scope('mmpose')
    
    # Import custom modules
    try:
        import custom_cephalometric_dataset
        import custom_transforms
        import cephalometric_dataset_info
        import anb_classification_utils
        import hrnetv2_with_classification_simple
        import classification_evaluator
        print("✓ Custom modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import custom modules: {e}")
        return
    
    # Configuration
    config_path = "Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py"
    
    # For ensemble training, look in model-specific directory
    if "ensemble" in args.work_dir:
        model_work_dir = os.path.join(args.work_dir, f"model_{args.model_idx}")
    else:
        model_work_dir = args.work_dir
    
    # Find checkpoint
    if args.epoch is not None:
        checkpoint_path = os.path.join(model_work_dir, f"epoch_{args.epoch}.pth")
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint not found: {checkpoint_path}")
            return
    else:
        # Find best checkpoint
        checkpoint_pattern = os.path.join(model_work_dir, "best_*.pth")
        checkpoints = glob.glob(checkpoint_pattern)
        
        if not checkpoints:
            # Try latest checkpoint
            checkpoint_pattern = os.path.join(model_work_dir, "epoch_*.pth")
            checkpoints = glob.glob(checkpoint_pattern)
        
        if not checkpoints:
            print(f"ERROR: No checkpoints found in {model_work_dir}")
            return
        
        checkpoint_path = max(checkpoints, key=os.path.getctime)
    
    print(f"✓ Using checkpoint: {checkpoint_path}")
    
    # Load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    model = init_model(config_path, checkpoint_path, device=device)
    print("✓ Model loaded successfully")
    
    # Check if model has classification head
    has_classification = hasattr(model, 'head') and hasattr(model.head, 'classification_head')
    if has_classification:
        print("✓ Model has native classification head")
    else:
        print("⚠️  Model does not have classification head - will only compute post-hoc classification")
    
    # Load test data
    data_file_path = args.data_file
    if not os.path.exists(data_file_path):
        print(f"ERROR: Data file not found: {data_file_path}")
        print("Please specify the correct path using --data_file argument")
        return
    
    print(f"Loading data from: {data_file_path}")
    try:
        main_df = pd.read_json(data_file_path)
        print(f"✓ Loaded {len(main_df)} total samples")
    except Exception as e:
        print(f"ERROR: Failed to load data file: {e}")
        return
    
    # Split test data
    if args.test_split_file:
        print(f"Loading test set from: {args.test_split_file}")
        with open(args.test_split_file, 'r') as f:
            test_patient_ids = {int(line.strip()) for line in f if line.strip()}
        
        main_df['patient_id'] = main_df['patient_id'].astype(int)
        test_df = main_df[main_df['patient_id'].isin(test_patient_ids)].reset_index(drop=True)
    else:
        print("Using 'set' column for test set selection")
        test_df = main_df[main_df['set'] == 'test'].reset_index(drop=True)
        if test_df.empty:
            test_df = main_df[main_df['set'] == 'dev'].reset_index(drop=True)
    
    if test_df.empty:
        print("ERROR: No test samples found")
        return
    
    print(f"✓ Evaluating on {len(test_df)} test samples")
    
    # Get landmark information
    landmark_names = cephalometric_dataset_info.landmark_names_in_order
    landmark_cols = cephalometric_dataset_info.original_landmark_cols
    
    # Storage for results
    all_pred_keypoints = []
    all_gt_keypoints = []
    all_native_classes = []
    all_posthoc_classes = []
    all_gt_classes = []
    
    class_names = ['Class I', 'Class II', 'Class III']
    
    print(f"\n🔄 Running evaluation on test set...")
    from tqdm import tqdm
    
    # Debug counters
    skipped_invalid_gt = 0
    skipped_no_gt_class = 0
    failed_inference = 0
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        try:
            # Get image
            if 'Image' not in row:
                print(f"Warning: No 'Image' column in row {idx}")
                failed_inference += 1
                continue
                
            img_array = np.array(row['Image'], dtype=np.uint8).reshape((224, 224, 3))
            
            # Get ground truth keypoints
            gt_keypoints = []
            valid_gt = True
            for i in range(0, len(landmark_cols), 2):
                x_col = landmark_cols[i]
                y_col = landmark_cols[i+1]
                if x_col in row and y_col in row and pd.notna(row[x_col]) and pd.notna(row[y_col]):
                    gt_keypoints.append([row[x_col], row[y_col]])
                else:
                    gt_keypoints.append([0, 0])
                    valid_gt = False
            
            if not valid_gt:
                skipped_invalid_gt += 1
                continue
                
            gt_keypoints = np.array(gt_keypoints)
            
            # Get ground truth classification
            if 'class' in row and pd.notna(row['class']):
                # Map class names to indices
                class_mapping = {'Class I': 0, 'Class II': 1, 'Class III': 2}
                gt_class = class_mapping.get(row['class'], -1)
            else:
                # Compute from ground truth landmarks
                try:
                    anb_angle = anb_classification_utils.calculate_anb_angle(gt_keypoints)
                    if anb_angle is not None and not np.isnan(anb_angle):
                        gt_class = anb_classification_utils.classify_from_anb_angle(anb_angle)
                        if isinstance(gt_class, np.ndarray):
                            gt_class = gt_class.item()
                    else:
                        gt_class = -1
                except Exception:
                    gt_class = -1
            
            if gt_class == -1:
                skipped_no_gt_class += 1
                continue
            
            # Run inference
            bbox = np.array([[0, 0, 224, 224]], dtype=np.float32)
            results = inference_topdown(model, img_array, bboxes=bbox, bbox_format='xyxy')
            
            if results and len(results) > 0:
                # Get predicted keypoints
                pred_keypoints = results[0].pred_instances.keypoints[0]
                if isinstance(pred_keypoints, torch.Tensor):
                    pred_keypoints = pred_keypoints.cpu().numpy()
                
                # Get native classification if available
                native_class = -1
                if has_classification:
                    # Check if classification scores are in the results
                    if hasattr(results[0].pred_instances, 'pred_classification'):
                        # Direct access to classification predictions
                        native_class_probs = results[0].pred_instances.pred_classification
                        if isinstance(native_class_probs, torch.Tensor):
                            native_class_probs = native_class_probs.cpu().numpy()
                        native_class = np.argmax(native_class_probs)
                    elif hasattr(results[0].pred_instances, 'classification_scores'):
                        # Alternative field name
                        native_class_probs = results[0].pred_instances.classification_scores
                        if isinstance(native_class_probs, torch.Tensor):
                            native_class_probs = native_class_probs.cpu().numpy()
                        native_class = np.argmax(native_class_probs)
                    else:
                        # If not in results, try direct model forward pass for debugging
                        try:
                            # Prepare input for model
                            from mmengine.structures import InstanceData, PixelData
                            from mmpose.structures import PoseDataSample
                            
                            # Normalize and prepare image
                            img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1) / 255.0
                            img_tensor = img_tensor.unsqueeze(0).to(device)
                            
                            # Create data sample
                            data_sample = PoseDataSample()
                            data_sample.gt_instances = InstanceData()
                            data_sample.set_metainfo({
                                'img_shape': (224, 224),
                                'ori_shape': (224, 224),
                                'input_size': (224, 224),
                                'input_center': np.array([112., 112.]),
                                'input_scale': np.array([224., 224.])
                            })
                            
                            # Get model predictions
                            with torch.no_grad():
                                # Extract features
                                if hasattr(model, 'extract_feat'):
                                    feats = model.extract_feat(img_tensor)
                                else:
                                    feats = model.backbone(img_tensor)
                                    if hasattr(model, 'neck') and model.neck is not None:
                                        feats = model.neck(feats)
                                
                                # Get head predictions
                                if hasattr(model.head, 'forward'):
                                    head_outputs = model.head.forward(feats, [data_sample])
                                    
                                    # Extract classification from head outputs
                                    if isinstance(head_outputs, tuple) and len(head_outputs) > 1:
                                        # head_outputs might be (heatmaps, classification_logits)
                                        _, classification_logits = head_outputs
                                        if classification_logits is not None:
                                            native_class = torch.argmax(classification_logits, dim=1).item()
                        except Exception as e:
                            print(f"Could not get native classification for sample {idx}: {e}")
                
                # Compute post-hoc classification from predicted landmarks
                try:
                    anb_angle_pred = anb_classification_utils.calculate_anb_angle(pred_keypoints)
                    if anb_angle_pred is not None and not np.isnan(anb_angle_pred):
                        posthoc_class = anb_classification_utils.classify_from_anb_angle(anb_angle_pred)
                        if isinstance(posthoc_class, np.ndarray):
                            posthoc_class = posthoc_class.item()
                    else:
                        posthoc_class = -1
                except Exception:
                    posthoc_class = -1
                
                # Store results
                all_pred_keypoints.append(pred_keypoints)
                all_gt_keypoints.append(gt_keypoints)
                all_native_classes.append(native_class)
                all_posthoc_classes.append(posthoc_class)
                all_gt_classes.append(gt_class)
            
        except Exception as e:
            failed_inference += 1
            if failed_inference <= 5:  # Only print first 5 errors
                print(f"Failed to process sample {idx}: {e}")
            continue
    
    # Print debug summary
    print(f"\n📊 Processing Summary:")
    print(f"  Total samples: {len(test_df)}")
    print(f"  Skipped (invalid GT keypoints): {skipped_invalid_gt}")
    print(f"  Skipped (no GT classification): {skipped_no_gt_class}")
    print(f"  Failed inference: {failed_inference}")
    print(f"  Successfully processed: {len(all_pred_keypoints)}")
    
    if len(all_pred_keypoints) == 0:
        print("\nERROR: No valid predictions generated")
        print("\nPossible causes:")
        print("1. All samples have invalid ground truth keypoints")
        print("2. ANB angle calculation failing for all samples")
        print("3. No 'class' column in data and ANB calculation issues")
        print("\nPlease check:")
        print("- Data file format and landmark columns")
        print("- Whether 'class' column exists in the data")
        print("- Landmark coordinates are valid (not all zeros)")
        return
    
    print(f"\n✓ Successfully evaluated {len(all_pred_keypoints)} samples")
    
    # Convert to numpy arrays
    pred_coords = np.array(all_pred_keypoints)
    gt_coords = np.array(all_gt_keypoints)
    native_classes = np.array(all_native_classes)
    posthoc_classes = np.array(all_posthoc_classes)
    gt_classes = np.array(all_gt_classes)
    
    # Compute keypoint metrics
    print("\n📊 Computing keypoint detection metrics...")
    keypoint_overall, keypoint_per_landmark = compute_keypoint_metrics(
        pred_coords, gt_coords, landmark_names
    )
    
    # Compute classification metrics
    print("\n📊 Computing classification metrics...")
    
    # Post-hoc classification metrics
    posthoc_metrics = compute_classification_metrics(
        posthoc_classes, gt_classes, class_names
    )
    
    # Native classification metrics (if available)
    if has_classification and np.any(native_classes >= 0):
        native_metrics = compute_classification_metrics(
            native_classes, gt_classes, class_names
        )
    else:
        native_metrics = None
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    # Keypoint detection results
    print("\n🎯 KEYPOINT DETECTION PERFORMANCE:")
    print(f"{'Metric':<20} {'Value':<15}")
    print("-" * 35)
    print(f"{'MRE (pixels)':<20} {keypoint_overall['mre']:<15.3f}")
    print(f"{'Std Dev':<20} {keypoint_overall['std']:<15.3f}")
    print(f"{'Median':<20} {keypoint_overall['median']:<15.3f}")
    print(f"{'90th Percentile':<20} {keypoint_overall['p90']:<15.3f}")
    print(f"{'95th Percentile':<20} {keypoint_overall['p95']:<15.3f}")
    
    # Top 5 best and worst landmarks
    sorted_landmarks = sorted(keypoint_per_landmark.items(), 
                            key=lambda x: x[1]['mre'])
    
    print("\n🏆 Top 5 Best Detected Landmarks:")
    for i, (name, metrics) in enumerate(sorted_landmarks[:5]):
        print(f"  {i+1}. {name:<20} MRE: {metrics['mre']:.3f} ± {metrics['std']:.3f}")
    
    print("\n⚠️  Top 5 Worst Detected Landmarks:")
    for i, (name, metrics) in enumerate(sorted_landmarks[-5:]):
        print(f"  {i+1}. {name:<20} MRE: {metrics['mre']:.3f} ± {metrics['std']:.3f}")
    
    # Classification results
    print("\n\n🏷️  SKELETAL CLASSIFICATION PERFORMANCE:")
    print("="*60)
    
    # Post-hoc classification
    print("\n📊 Post-hoc Classification (from predicted landmarks):")
    print(f"{'Metric':<25} {'Value':<15}")
    print("-" * 40)
    print(f"{'Overall Accuracy':<25} {posthoc_metrics['accuracy']*100:<15.1f}%")
    print(f"{'Samples Evaluated':<25} {posthoc_metrics['n_samples']:<15d}")
    
    print("\nPer-Class Accuracy:")
    for class_name, acc in posthoc_metrics['per_class_accuracy'].items():
        print(f"  {class_name:<20} {acc*100:.1f}%")
    
    # Native classification (if available)
    if native_metrics:
        print("\n📊 Native Classification (from neural network):")
        print(f"{'Metric':<25} {'Value':<15}")
        print("-" * 40)
        print(f"{'Overall Accuracy':<25} {native_metrics['accuracy']*100:<15.1f}%")
        print(f"{'Samples Evaluated':<25} {native_metrics['n_samples']:<15d}")
        
        print("\nPer-Class Accuracy:")
        for class_name, acc in native_metrics['per_class_accuracy'].items():
            print(f"  {class_name:<20} {acc*100:.1f}%")
        
        # Compare native vs post-hoc
        print("\n🔄 Native vs Post-hoc Comparison:")
        accuracy_diff = native_metrics['accuracy'] - posthoc_metrics['accuracy']
        print(f"Accuracy Difference: {accuracy_diff*100:+.1f}%")
        
        # Agreement between methods
        valid_mask = (native_classes >= 0) & (posthoc_classes >= 0)
        if np.any(valid_mask):
            agreement = np.mean(native_classes[valid_mask] == posthoc_classes[valid_mask])
            print(f"Agreement Rate: {agreement*100:.1f}%")
    
    # Save results
    output_dir = os.path.join(model_work_dir, "classification_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_summary = {
        'checkpoint': os.path.basename(checkpoint_path),
        'n_samples': len(all_pred_keypoints),
        'keypoint_metrics': {
            'overall': keypoint_overall,
            'per_landmark': keypoint_per_landmark
        },
        'classification_metrics': {
            'posthoc': posthoc_metrics,
            'native': native_metrics if native_metrics else None
        }
    }
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    # Save confusion matrices
    plot_confusion_matrix(
        posthoc_metrics['confusion_matrix'],
        class_names,
        'Post-hoc Classification Confusion Matrix',
        os.path.join(output_dir, 'confusion_matrix_posthoc.png')
    )
    
    if native_metrics:
        plot_confusion_matrix(
            native_metrics['confusion_matrix'],
            class_names,
            'Native Classification Confusion Matrix',
            os.path.join(output_dir, 'confusion_matrix_native.png')
        )
        
        # Side-by-side comparison
        plot_classification_comparison(
            native_metrics['confusion_matrix'],
            posthoc_metrics['confusion_matrix'],
            class_names,
            os.path.join(output_dir, 'classification_comparison.png')
        )
    
    # Print classification reports
    print("\n📋 DETAILED CLASSIFICATION REPORT:")
    print("\nPost-hoc Classification:")
    print("-" * 60)
    if 'classification_report' in posthoc_metrics:
        for class_name in class_names:
            if class_name in posthoc_metrics['classification_report']:
                report = posthoc_metrics['classification_report'][class_name]
                print(f"{class_name}:")
                print(f"  Precision: {report['precision']:.3f}")
                print(f"  Recall: {report['recall']:.3f}")
                print(f"  F1-Score: {report['f1-score']:.3f}")
                print(f"  Support: {report['support']}")
    
    if native_metrics and 'classification_report' in native_metrics:
        print("\nNative Classification:")
        print("-" * 60)
        for class_name in class_names:
            if class_name in native_metrics['classification_report']:
                report = native_metrics['classification_report'][class_name]
                print(f"{class_name}:")
                print(f"  Precision: {report['precision']:.3f}")
                print(f"  Recall: {report['recall']:.3f}")
                print(f"  F1-Score: {report['f1-score']:.3f}")
                print(f"  Support: {report['support']}")
    
    print(f"\n💾 Results saved to: {output_dir}")
    print("  - evaluation_results.json")
    print("  - confusion_matrix_*.png")
    if native_metrics:
        print("  - classification_comparison.png")
    
    print("\n🎉 Evaluation completed successfully!")
    
    # Summary insights
    print("\n💡 INSIGHTS:")
    if native_metrics and native_metrics['accuracy'] > posthoc_metrics['accuracy']:
        print("✅ Native classification outperforms post-hoc classification")
        print("   → The model learns better features for classification than just landmark positions")
    elif native_metrics:
        print("⚠️  Post-hoc classification performs similarly or better than native")
        print("   → The classification head might need more training or better features")
    
    if keypoint_overall['mre'] < 5.0:
        print("✅ Excellent keypoint detection performance (MRE < 5 pixels)")
    elif keypoint_overall['mre'] < 10.0:
        print("✅ Good keypoint detection performance (MRE < 10 pixels)")
    else:
        print("⚠️  Keypoint detection needs improvement (MRE > 10 pixels)")


if __name__ == "__main__":
    main() 