#!/usr/bin/env python3
"""
Cross-Validation MLP Performance Evaluation Script
This script evaluates the performance of concurrent MLP training across cross-validation folds.
It can evaluate specific folds, all folds, or provide summary statistics across folds.
"""

import os
import sys
import torch
import torch.nn as nn
import warnings
import pandas as pd
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.apis import init_model, inference_topdown
import glob
import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
import json
from tqdm import tqdm

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

class JointMLPRefinementModel(nn.Module):
    """Joint MLP model for landmark coordinate refinement - same as in hook."""
    def __init__(self, input_dim=38, hidden_dim=500, output_dim=38):
        super(JointMLPRefinementModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Main network with residual connection
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Residual projection (if dimensions don't match)
        self.residual_proj = None
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Main forward pass
        out = self.net(x)
        
        # Add residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(x)
        else:
            residual = x
            
        return out + 0.1 * residual  # Small residual weight to start

def apply_joint_mlp_refinement(predictions, mlp_joint, scaler_input, scaler_target, device):
    """Apply joint MLP refinement to predictions."""
    try:
        # Flatten predictions to 38-D vector [x1, y1, x2, y2, ..., x19, y19]
        pred_flat = predictions.flatten().reshape(1, -1)
        
        # Normalize input predictions
        pred_scaled = scaler_input.transform(pred_flat)
        
        # Convert to tensor
        pred_tensor = torch.FloatTensor(pred_scaled).to(device)
        
        # Apply joint MLP refinement
        with torch.no_grad():
            refined_scaled = mlp_joint(pred_tensor).cpu().numpy()
        
        # Denormalize outputs
        refined_flat = scaler_target.inverse_transform(refined_scaled).flatten()
        
        # Reshape back to [19, 2] format
        refined_coords = refined_flat.reshape(19, 2)
        
        return refined_coords
        
    except Exception as e:
        print(f"Joint MLP refinement failed: {e}")
        return predictions

def compute_metrics(pred_coords, gt_coords, landmark_names):
    """Compute comprehensive evaluation metrics."""
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
        'max': np.max(valid_errors)
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

def find_available_folds(cv_work_dir):
    """Find all available fold directories."""
    if not os.path.exists(cv_work_dir):
        return []
    
    fold_dirs = [d for d in os.listdir(cv_work_dir) if d.startswith('fold_') and os.path.isdir(os.path.join(cv_work_dir, d))]
    fold_numbers = []
    
    for fold_dir in fold_dirs:
        try:
            fold_num = int(fold_dir.split('_')[1])
            fold_numbers.append(fold_num)
        except:
            continue
    
    return sorted(fold_numbers)

def evaluate_single_fold(fold_num, cv_work_dir, config_path, checkpoint_type, epoch, main_df):
    """Evaluate a single fold."""
    print(f"\n📊 Evaluating Fold {fold_num}")
    print("-" * 50)
    
    fold_work_dir = os.path.join(cv_work_dir, f"fold_{fold_num}")
    
    # Load fold information
    fold_info_path = os.path.join(fold_work_dir, f"fold_{fold_num}_info.json")
    val_patients = None
    
    if os.path.exists(fold_info_path):
        with open(fold_info_path, 'r') as f:
            fold_info = json.load(f)
            val_patients = fold_info.get('val_patients', None)
            print(f"✓ Loaded fold info: {fold_info['train_samples']} train, {fold_info['val_samples']} val samples")
    
    # Find checkpoint
    hrnet_checkpoint = None
    
    if checkpoint_type == 'best':
        pattern = os.path.join(fold_work_dir, "best_NME_epoch_*.pth")
        checkpoints = glob.glob(pattern)
        if checkpoints:
            hrnet_checkpoint = max(checkpoints, key=os.path.getctime)
    elif checkpoint_type == 'latest':
        latest_path = os.path.join(fold_work_dir, "latest.pth")
        if os.path.exists(latest_path):
            hrnet_checkpoint = latest_path
        else:
            pattern = os.path.join(fold_work_dir, "epoch_*.pth")
            checkpoints = glob.glob(pattern)
            if checkpoints:
                hrnet_checkpoint = max(checkpoints, key=lambda x: int(x.split('epoch_')[1].split('.')[0]))
    elif checkpoint_type == 'epoch':
        epoch_path = os.path.join(fold_work_dir, f"epoch_{epoch}.pth")
        if os.path.exists(epoch_path):
            hrnet_checkpoint = epoch_path
    
    if hrnet_checkpoint is None:
        print(f"❌ No {checkpoint_type} checkpoint found for fold {fold_num}")
        return None
    
    checkpoint_name = os.path.basename(hrnet_checkpoint)
    print(f"✓ Using checkpoint: {checkpoint_name}")
    
    # Find MLP model
    mlp_dir = os.path.join(fold_work_dir, "concurrent_mlp")
    mlp_path = None
    
    # Try epoch matching first
    if "epoch_" in checkpoint_name:
        try:
            hrnet_epoch = int(checkpoint_name.split("epoch_")[1].split(".")[0])
            epoch_mlp_path = os.path.join(mlp_dir, f"mlp_joint_epoch_{hrnet_epoch}.pth")
            if os.path.exists(epoch_mlp_path):
                mlp_path = epoch_mlp_path
                print(f"✓ Found matching epoch MLP: mlp_joint_epoch_{hrnet_epoch}.pth")
        except:
            pass
    
    # Fallback to latest
    if mlp_path is None:
        latest_mlp = os.path.join(mlp_dir, "mlp_joint_latest.pth")
        if os.path.exists(latest_mlp):
            mlp_path = latest_mlp
            print(f"✓ Using latest MLP model")
    
    if mlp_path is None:
        print(f"❌ No MLP model found for fold {fold_num}")
        return None
    
    # Load scalers
    scaler_input_path = os.path.join(mlp_dir, "scaler_joint_input.pkl")
    scaler_target_path = os.path.join(mlp_dir, "scaler_joint_target.pkl")
    
    if not (os.path.exists(scaler_input_path) and os.path.exists(scaler_target_path)):
        print(f"❌ Scalers not found for fold {fold_num}")
        return None
    
    try:
        # Load models
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        hrnet_model = init_model(config_path, hrnet_checkpoint, device=device)
        
        mlp_joint = JointMLPRefinementModel().to(device)
        mlp_joint.load_state_dict(torch.load(mlp_path, map_location=device))
        mlp_joint.eval()
        
        scaler_input = joblib.load(scaler_input_path)
        scaler_target = joblib.load(scaler_target_path)
        
        print("✓ Models and scalers loaded")
        
        # Get validation data for this fold
        if val_patients is not None:
            val_df = main_df[main_df['patient_id'].isin(val_patients)].reset_index(drop=True)
        else:
            # Fallback: load from fold's validation file
            val_ann_file = os.path.join(fold_work_dir, f"fold_{fold_num}_val_ann.json")
            if os.path.exists(val_ann_file):
                val_df = pd.read_json(val_ann_file)
            else:
                print(f"❌ Cannot find validation data for fold {fold_num}")
                return None
        
        print(f"✓ Evaluating on {len(val_df)} validation samples")
        
        # Get landmark information
        import cephalometric_dataset_info
        landmark_names = cephalometric_dataset_info.landmark_names_in_order
        landmark_cols = cephalometric_dataset_info.original_landmark_cols
        
        # Run evaluation
        hrnet_predictions = []
        mlp_predictions = []
        ground_truths = []
        
        for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc=f"Fold {fold_num}", leave=False):
            try:
                # Get image and ground truth
                img_array = np.array(row['Image'], dtype=np.uint8).reshape((224, 224, 3))
                
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
                    continue
                    
                gt_keypoints = np.array(gt_keypoints)
                
                # Run HRNetV2 inference
                bbox = np.array([[0, 0, 224, 224]], dtype=np.float32)
                results = inference_topdown(hrnet_model, img_array, bboxes=bbox, bbox_format='xyxy')
                
                if results and len(results) > 0:
                    pred_keypoints = results[0].pred_instances.keypoints[0]
                    if isinstance(pred_keypoints, torch.Tensor):
                        pred_keypoints = pred_keypoints.cpu().numpy()
                else:
                    continue

                if pred_keypoints is None or pred_keypoints.shape[0] != 19:
                    continue
                
                # Apply joint MLP refinement
                refined_keypoints = apply_joint_mlp_refinement(
                    pred_keypoints, mlp_joint, scaler_input, scaler_target, device
                )
                
                # Store results
                hrnet_predictions.append(pred_keypoints)
                mlp_predictions.append(refined_keypoints)
                ground_truths.append(gt_keypoints)
                
            except Exception as e:
                continue
        
        if len(hrnet_predictions) == 0:
            print(f"❌ No valid predictions for fold {fold_num}")
            return None
        
        # Compute metrics
        hrnet_coords = np.array(hrnet_predictions)
        mlp_coords = np.array(mlp_predictions)
        gt_coords = np.array(ground_truths)
        
        hrnet_overall, hrnet_per_landmark = compute_metrics(hrnet_coords, gt_coords, landmark_names)
        mlp_overall, mlp_per_landmark = compute_metrics(mlp_coords, gt_coords, landmark_names)
        
        improvement = (hrnet_overall['mre'] - mlp_overall['mre']) / hrnet_overall['mre'] * 100
        
        print(f"📊 Results: HRNet MRE={hrnet_overall['mre']:.3f}, MLP MRE={mlp_overall['mre']:.3f}, Improvement={improvement:.1f}%")
        
        return {
            'fold': fold_num,
            'checkpoint': checkpoint_name,
            'samples': len(hrnet_predictions),
            'hrnet_mre': hrnet_overall['mre'],
            'mlp_mre': mlp_overall['mre'],
            'improvement': improvement,
            'hrnet_overall': hrnet_overall,
            'mlp_overall': mlp_overall,
            'hrnet_per_landmark': hrnet_per_landmark,
            'mlp_per_landmark': mlp_per_landmark
        }
        
    except Exception as e:
        print(f"❌ Error evaluating fold {fold_num}: {e}")
        return None

def main():
    """Main cross-validation evaluation function."""
    
    parser = argparse.ArgumentParser(
        description='Cross-Validation MLP Performance Evaluation')
    parser.add_argument(
        '--cv_work_dir',
        type=str,
        default='work_dirs/hrnetv2_w18_cephalometric_cv_5fold',
        help='Cross-validation work directory containing fold subdirectories'
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=None,
        help='Specific fold to evaluate (1-5). If not specified, evaluates all available folds.'
    )
    parser.add_argument(
        '--checkpoint_type',
        type=str,
        choices=['best', 'latest', 'epoch'],
        default='latest',
        help='Type of checkpoint to evaluate'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=None,
        help='Specific epoch number (only used with checkpoint_type=epoch)'
    )
    args = parser.parse_args()
    
    print("="*80)
    print("CROSS-VALIDATION MLP PERFORMANCE EVALUATION")
    print("="*80)
    print(f"📁 CV Work Dir: {args.cv_work_dir}")
    print(f"📋 Checkpoint type: {args.checkpoint_type}")
    if args.fold:
        print(f"🎯 Target fold: {args.fold}")
    else:
        print(f"🔄 Evaluating all available folds")
    if args.checkpoint_type == 'epoch':
        print(f"📅 Target epoch: {args.epoch}")
    print()
    
    # Initialize MMPose scope
    init_default_scope('mmpose')
    
    # Import custom modules
    try:
        import custom_cephalometric_dataset
        import custom_transforms
        import cephalometric_dataset_info
        print("✓ Custom modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import custom modules: {e}")
        return
    
    # Load main dataset
    data_file_path = "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json"
    try:
        main_df = pd.read_json(data_file_path)
        main_df['patient_id'] = main_df['patient_id'].astype(int)
        print(f"✓ Loaded main dataset: {len(main_df)} samples")
    except Exception as e:
        print(f"✗ Failed to load main dataset: {e}")
        return
    
    # Find available folds
    available_folds = find_available_folds(args.cv_work_dir)
    if not available_folds:
        print(f"❌ No fold directories found in {args.cv_work_dir}")
        return
    
    print(f"✓ Available folds: {available_folds}")
    
    # Determine which folds to evaluate
    if args.fold is not None:
        if args.fold not in available_folds:
            print(f"❌ Fold {args.fold} not found in available folds")
            return
        folds_to_evaluate = [args.fold]
    else:
        folds_to_evaluate = available_folds
    
    print(f"🎯 Evaluating folds: {folds_to_evaluate}")
    
    # Configuration
    config_path = "Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py"
    
    # Evaluate folds
    fold_results = []
    
    for fold_num in folds_to_evaluate:
        result = evaluate_single_fold(fold_num, args.cv_work_dir, config_path, 
                                     args.checkpoint_type, args.epoch, main_df)
        if result:
            fold_results.append(result)
    
    if not fold_results:
        print("❌ No successful evaluations")
        return
    
    # Summary statistics
    print("\n" + "="*80)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("="*80)
    
    hrnet_mres = [r['hrnet_mre'] for r in fold_results]
    mlp_mres = [r['mlp_mre'] for r in fold_results]
    improvements = [r['improvement'] for r in fold_results]
    
    print(f"📊 PERFORMANCE ACROSS {len(fold_results)} FOLDS:")
    print(f"{'Fold':<6} {'HRNet MRE':<12} {'MLP MRE':<12} {'Improvement':<12} {'Samples':<8}")
    print("-" * 60)
    
    for result in fold_results:
        print(f"{result['fold']:<6} {result['hrnet_mre']:<12.3f} {result['mlp_mre']:<12.3f} "
              f"{result['improvement']:<12.1f}% {result['samples']:<8}")
    
    print("-" * 60)
    print(f"{'Mean':<6} {np.mean(hrnet_mres):<12.3f} {np.mean(mlp_mres):<12.3f} {np.mean(improvements):<12.1f}%")
    print(f"{'Std':<6} {np.std(hrnet_mres):<12.3f} {np.std(mlp_mres):<12.3f} {np.std(improvements):<12.1f}%")
    
    # Save results
    output_dir = os.path.join(args.cv_work_dir, "cv_evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    
    results_summary = {
        'checkpoint_type': args.checkpoint_type,
        'epoch': args.epoch,
        'evaluated_folds': folds_to_evaluate,
        'fold_results': fold_results,
        'summary': {
            'mean_hrnet_mre': np.mean(hrnet_mres),
            'std_hrnet_mre': np.std(hrnet_mres),
            'mean_mlp_mre': np.mean(mlp_mres),
            'std_mlp_mre': np.std(mlp_mres),
            'mean_improvement': np.mean(improvements),
            'std_improvement': np.std(improvements)
        }
    }
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"cv_results_{args.checkpoint_type}_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print(f"🎉 Cross-validation evaluation completed!")
    print(f"📈 Overall MLP improvement: {np.mean(improvements):.1f}% ± {np.std(improvements):.1f}%")

if __name__ == "__main__":
    main() 