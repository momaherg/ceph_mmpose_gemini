#!/usr/bin/env python3
"""
Concurrent Joint MLP Performance Evaluation Script - IMPROVED VERSION
This script evaluates the performance improvement from joint MLP residual refinement
trained concurrently with HRNetV2.

IMPROVEMENTS:
• Support for residual learning models (predicting corrections)
• Single shared scaler for consistent normalization
• Updated model file detection for residual models
• Proper residual application at inference time
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
    """Joint MLP model for landmark coordinate residual prediction - same as in hook."""
    def __init__(self, input_dim=38, hidden_dim=500, output_dim=38):
        super(JointMLPRefinementModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Simplified network for residual learning
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def apply_joint_mlp_residual_refinement(predictions, mlp_joint, shared_scaler, device):
    """Apply joint MLP residual refinement to predictions."""
    try:
        # Flatten predictions to 38-D vector [x1, y1, x2, y2, ..., x19, y19]
        pred_flat = predictions.flatten().reshape(1, -1)
        
        # Normalize input predictions using shared scaler
        pred_scaled = shared_scaler.transform(pred_flat)
        
        # Convert to tensor
        pred_tensor = torch.FloatTensor(pred_scaled).to(device)
        
        # Apply joint MLP to get residual corrections
        with torch.no_grad():
            residual_scaled = mlp_joint(pred_tensor).cpu().numpy()
        
        # Denormalize residual using the same scaler
        # Since residual_scaled = scaler.transform(pred + residual) - scaler.transform(pred)
        # We need: residual = inverse_transform(scaler.transform(pred) + residual_scaled) - pred
        pred_plus_residual_scaled = pred_scaled + residual_scaled
        pred_plus_residual = shared_scaler.inverse_transform(pred_plus_residual_scaled).flatten()
        
        # Reshape back to [19, 2] format
        refined_coords = pred_plus_residual.reshape(19, 2)
        
        return refined_coords
        
    except Exception as e:
        print(f"Joint MLP residual refinement failed: {e}")
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

def main():
    """Main evaluation function."""
    
    parser = argparse.ArgumentParser(
        description='Evaluate Concurrent Joint MLP Residual Refinement Performance')
    parser.add_argument(
        '--test_split_file',
        type=str,
        default=None,
        help='Path to a text file containing patient IDs for the test set, one ID per line.'
    )
    parser.add_argument(
        '--work_dir',
        type=str,
        default='work_dirs/hrnetv2_w18_cephalometric_concurrent_mlp_v5',
        help='Work directory containing the trained models'
    )
    args = parser.parse_args()
    
    print("="*80)
    print("CONCURRENT JOINT MLP RESIDUAL REFINEMENT EVALUATION")
    print("="*80)
    
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
    
    # Configuration
    config_path = "Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py"
    
    # Load config
    cfg = Config.fromfile(config_path)
    
    # Find the best HRNetV2 checkpoint
    hrnet_checkpoint_pattern = os.path.join(args.work_dir, "best_NME_epoch_*.pth")
    hrnet_checkpoints = glob.glob(hrnet_checkpoint_pattern)
    
    if not hrnet_checkpoints:
        hrnet_checkpoint_pattern = os.path.join(args.work_dir, "epoch_*.pth")
        hrnet_checkpoints = glob.glob(hrnet_checkpoint_pattern)
    
    if not hrnet_checkpoints:
        print("ERROR: No HRNetV2 checkpoints found")
        return
    
    hrnet_checkpoint = max(hrnet_checkpoints, key=os.path.getctime)
    print(f"✓ Using HRNetV2 checkpoint: {hrnet_checkpoint}")
    
    # Check for residual joint MLP models
    mlp_dir = os.path.join(args.work_dir, "concurrent_mlp")
    mlp_joint_path = os.path.join(mlp_dir, "mlp_joint_residual_final.pth")
    
    # Check for final models first, then latest, then epoch-specific
    if os.path.exists(mlp_joint_path):
        print(f"✓ Found final residual joint MLP model: {mlp_joint_path}")
        model_type = "final"
    else:
        # Try latest model
        mlp_joint_latest = os.path.join(mlp_dir, "mlp_joint_residual_latest.pth")
        
        if os.path.exists(mlp_joint_latest):
            mlp_joint_path = mlp_joint_latest
            print(f"✓ Found latest residual joint MLP model: {mlp_joint_path}")
            model_type = "latest"
        else:
            # Try to find epoch-specific models
            epoch_models = glob.glob(os.path.join(mlp_dir, "mlp_joint_residual_epoch_*.pth"))
            if epoch_models:
                # Get the latest epoch model
                latest_joint_model = max(epoch_models, key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
                epoch_num = latest_joint_model.split('_epoch_')[1].split('.')[0]
                
                mlp_joint_path = latest_joint_model
                print(f"✓ Found epoch {epoch_num} residual joint MLP model: {mlp_joint_path}")
                model_type = f"epoch_{epoch_num}"
            else:
                print("ERROR: Residual joint MLP model not found.")
                print(f"Searched in: {mlp_dir}")
                print("Available files:")
                if os.path.exists(mlp_dir):
                    for file in os.listdir(mlp_dir):
                        print(f"  - {file}")
                else:
                    print("  MLP directory does not exist")
                print("\nTip: Make sure concurrent residual training is running and has completed at least one epoch.")
                return
    
    # Load models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    # Load HRNetV2 model
    hrnet_model = init_model(config_path, hrnet_checkpoint, device=device)
    print("✓ HRNetV2 model loaded")
    
    # Load joint residual MLP model
    mlp_joint = JointMLPRefinementModel().to(device)
    mlp_joint.load_state_dict(torch.load(mlp_joint_path, map_location=device))
    mlp_joint.eval()
    print("✓ Joint residual MLP model loaded")
    
    # Load test data
    data_file_path = "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json"
    main_df = pd.read_json(data_file_path)
    
    # Split test data (same logic as training scripts)
    if args.test_split_file:
        print(f"Loading test set from external file: {args.test_split_file}")
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
    
    # Load saved shared normalization scaler
    print("Loading saved shared normalization scaler...")
    scaler_dir = os.path.join(args.work_dir, "concurrent_mlp")
    
    shared_scaler_path = os.path.join(scaler_dir, "scaler_shared.pkl")
    
    # Check if scaler exists
    if not os.path.exists(shared_scaler_path):
        print(f"ERROR: Missing shared scaler file: {shared_scaler_path}")
        print("This indicates that concurrent residual MLP training hasn't run yet or scaler wasn't saved.")
        print("Please run concurrent residual training first.")
        return
    
    # Load scaler
    try:
        shared_scaler = joblib.load(shared_scaler_path)
        print("✓ Shared normalization scaler loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load shared scaler: {e}")
        return
    
    # Evaluation on test set
    print(f"\n🔄 Running evaluation on {len(test_df)} test samples...")
    
    # Storage for results
    hrnet_predictions = []
    mlp_predictions = []
    ground_truths = []
    
    from tqdm import tqdm
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
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
            
            # Get dynamic bounding box based on actual image dimensions
            h, w = img_array.shape[:2]
            bbox = np.array([[0, 0, w, h]], dtype=np.float32)
            
            # Run HRNetV2 inference using the standard API
            results = inference_topdown(hrnet_model, img_array, bboxes=bbox, bbox_format='xyxy')
            
            if results and len(results) > 0:
                pred_keypoints = results[0].pred_instances.keypoints[0]
                if isinstance(pred_keypoints, torch.Tensor):
                    pred_keypoints = pred_keypoints.cpu().numpy()
            else:
                continue

            if pred_keypoints is None or pred_keypoints.shape[0] != 19:
                continue
            
            # Apply joint MLP residual refinement
            refined_keypoints = apply_joint_mlp_residual_refinement(
                pred_keypoints, mlp_joint, shared_scaler, device
            )
            
            # Store results
            hrnet_predictions.append(pred_keypoints)
            mlp_predictions.append(refined_keypoints)
            ground_truths.append(gt_keypoints)
            
        except Exception as e:
            print(f"Failed to process sample {idx}: {e}")
            continue
    
    if len(hrnet_predictions) == 0:
        print("ERROR: No valid predictions generated")
        return
    
    print(f"✓ Successfully evaluated {len(hrnet_predictions)} samples")
    
    # Convert to numpy arrays
    hrnet_coords = np.array(hrnet_predictions)
    mlp_coords = np.array(mlp_predictions)
    gt_coords = np.array(ground_truths)
    
    # Compute metrics
    print("\n📊 Computing metrics...")
    
    hrnet_overall, hrnet_per_landmark = compute_metrics(hrnet_coords, gt_coords, landmark_names)
    mlp_overall, mlp_per_landmark = compute_metrics(mlp_coords, gt_coords, landmark_names)
    
    # Print results
    print("\n" + "="*80)
    print("JOINT MLP RESIDUAL EVALUATION RESULTS")
    print("="*80)
    print(f"📊 Evaluated using {model_type} residual joint MLP model")
    print(f"📈 HRNetV2 checkpoint: {os.path.basename(hrnet_checkpoint)}")
    print(f"🔧 Using residual learning approach (predicting corrections)")
    
    print(f"\n🏷️  OVERALL PERFORMANCE:")
    print(f"{'Metric':<15} {'HRNetV2':<15} {'Residual MLP':<15} {'Improvement':<15}")
    print("-" * 65)
    
    improvement_mre = (hrnet_overall['mre'] - mlp_overall['mre']) / hrnet_overall['mre'] * 100
    improvement_std = (hrnet_overall['std'] - mlp_overall['std']) / hrnet_overall['std'] * 100
    improvement_median = (hrnet_overall['median'] - mlp_overall['median']) / hrnet_overall['median'] * 100
    
    print(f"{'MRE':<15} {hrnet_overall['mre']:<15.3f} {mlp_overall['mre']:<15.3f} {improvement_mre:<15.1f}%")
    print(f"{'Std Dev':<15} {hrnet_overall['std']:<15.3f} {mlp_overall['std']:<15.3f} {improvement_std:<15.1f}%")
    print(f"{'Median':<15} {hrnet_overall['median']:<15.3f} {mlp_overall['median']:<15.3f} {improvement_median:<15.1f}%")
    print(f"{'P90':<15} {hrnet_overall['p90']:<15.3f} {mlp_overall['p90']:<15.3f}")
    print(f"{'P95':<15} {hrnet_overall['p95']:<15.3f} {mlp_overall['p95']:<15.3f}")
    
    # Per-landmark comparison for problematic landmarks
    print(f"\n🎯 PROBLEMATIC LANDMARKS COMPARISON:")
    problematic_landmarks = ['sella', 'Gonion', 'PNS', 'A_point', 'B_point']
    
    print(f"{'Landmark':<20} {'HRNetV2 MRE':<15} {'Residual MLP':<15} {'Improvement':<15}")
    print("-" * 70)
    
    for landmark in problematic_landmarks:
        if landmark in hrnet_per_landmark and landmark in mlp_per_landmark:
            hrnet_err = hrnet_per_landmark[landmark]['mre']
            mlp_err = mlp_per_landmark[landmark]['mre']
            if hrnet_err > 0:
                improvement = (hrnet_err - mlp_err) / hrnet_err * 100
                print(f"{landmark:<20} {hrnet_err:<15.3f} {mlp_err:<15.3f} {improvement:<15.1f}%")
    
    # Save results
    output_dir = os.path.join(args.work_dir, "residual_mlp_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_summary = {
        'hrnet_overall': hrnet_overall,
        'mlp_overall': mlp_overall,
        'improvement_mre': improvement_mre,
        'improvement_std': improvement_std,
        'improvement_median': improvement_median,
        'total_samples': len(hrnet_predictions),
        'model_type': model_type,
        'approach': 'residual_learning'
    }
    
    # Save per-landmark comparison
    per_landmark_comparison = []
    for landmark in landmark_names:
        if landmark in hrnet_per_landmark and landmark in mlp_per_landmark:
            hrnet_err = hrnet_per_landmark[landmark]['mre']
            mlp_err = mlp_per_landmark[landmark]['mre']
            improvement = (hrnet_err - mlp_err) / hrnet_err * 100 if hrnet_err > 0 else 0
            
            per_landmark_comparison.append({
                'landmark': landmark,
                'hrnet_mre': hrnet_err,
                'mlp_mre': mlp_err,
                'improvement_percent': improvement,
                'hrnet_std': hrnet_per_landmark[landmark]['std'],
                'mlp_std': mlp_per_landmark[landmark]['std']
            })
    
    comparison_df = pd.DataFrame(per_landmark_comparison)
    comparison_df.to_csv(os.path.join(output_dir, "per_landmark_comparison_residual.csv"), index=False)
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Overall comparison
    methods = ['HRNetV2', 'Residual MLP']
    mres = [hrnet_overall['mre'], mlp_overall['mre']]
    stds = [hrnet_overall['std'], mlp_overall['std']]
    
    ax1.bar(methods, mres, yerr=stds, capsize=5, alpha=0.7, color=['skyblue', 'lightgreen'])
    ax1.set_ylabel('Mean Radial Error (pixels)')
    ax1.set_title('Overall Performance Comparison (Residual Learning)')
    ax1.grid(True, alpha=0.3)
    
    # Add improvement percentage
    ax1.text(1, mlp_overall['mre'] + mlp_overall['std'] + 0.1, 
             f'{improvement_mre:.1f}% improvement', ha='center', va='bottom', 
             fontsize=10, color='green' if improvement_mre > 0 else 'red', fontweight='bold')
    
    # Per-landmark improvements
    landmarks_subset = comparison_df.head(10)  # Top 10 landmarks
    ax2.barh(landmarks_subset['landmark'], landmarks_subset['improvement_percent'], 
             color=['green' if x > 0 else 'red' for x in landmarks_subset['improvement_percent']])
    ax2.set_xlabel('Improvement (%)')
    ax2.set_title('Per-Landmark Improvement (Residual)')
    ax2.grid(True, alpha=0.3)
    
    # Error distribution comparison
    hrnet_errors = np.sqrt(np.sum((hrnet_coords - gt_coords)**2, axis=2)).flatten()
    mlp_errors = np.sqrt(np.sum((mlp_coords - gt_coords)**2, axis=2)).flatten()
    
    # Remove invalid landmarks (where gt is [0,0])
    valid_mask = (gt_coords.reshape(-1, 2)[:, 0] > 0) & (gt_coords.reshape(-1, 2)[:, 1] > 0)
    hrnet_errors = hrnet_errors[valid_mask]
    mlp_errors = mlp_errors[valid_mask]
    
    ax3.hist([hrnet_errors, mlp_errors], bins=50, alpha=0.7, 
             label=['HRNetV2', 'Residual MLP'], color=['skyblue', 'lightgreen'])
    ax3.set_xlabel('Radial Error (pixels)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Scatter plot: HRNet vs MLP errors
    sample_indices = np.random.choice(len(hrnet_errors), min(1000, len(hrnet_errors)), replace=False)
    ax4.scatter(hrnet_errors[sample_indices], mlp_errors[sample_indices], 
                alpha=0.5, color='darkgreen')
    
    # Add diagonal line
    max_error = max(np.max(hrnet_errors), np.max(mlp_errors))
    ax4.plot([0, max_error], [0, max_error], 'r--', alpha=0.8, label='No improvement line')
    ax4.set_xlabel('HRNetV2 Error (pixels)')
    ax4.set_ylabel('Residual MLP Error (pixels)')
    ax4.set_title('Error Correlation (Residual Learning)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "residual_mlp_evaluation_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n💾 Results saved to: {output_dir}")
    print(f"   - Per-landmark comparison: per_landmark_comparison_residual.csv")
    print(f"   - Visualization: residual_mlp_evaluation_results.png")
    
    print(f"\n🎉 Residual MLP evaluation completed!")
    print(f"📈 Overall improvement: {improvement_mre:.1f}% reduction in MRE")
    print(f"🎯 Residual learning approach: predicts corrections instead of absolute coordinates")
    print(f"🔧 Evaluated using: {model_type} residual joint MLP model")
    
    if model_type == "latest":
        print("💡 Note: Training is likely still in progress. Final results may differ.")
    elif "epoch_" in model_type:
        print("💡 Note: Using intermediate checkpoint. Final results may differ.")

if __name__ == "__main__":
    main() 