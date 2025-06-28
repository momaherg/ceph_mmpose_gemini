#!/usr/bin/env python3
"""
Concurrent Joint MLP Performance Evaluation Script
This script evaluates the performance improvement from joint 38-D MLP refinement
trained concurrently with HRNetV2.
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
    """
    Joint MLP model for landmark coordinate refinement.
    Input: 38 predicted coordinates (19 landmarks × 2 coordinates)
    Hidden: 512 neurons with residual connection
    Output: 38 refined coordinates
    """
    def __init__(self, input_dim=38, hidden_dim=512, output_dim=38):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Main network with residual connection
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Residual connection (input -> output)
        self.residual = nn.Linear(input_dim, output_dim)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Main path
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        # Residual connection
        residual = self.residual(x)
        
        # Combine main path and residual
        out = out + residual
        
        return out

def apply_joint_mlp_refinement(predictions, joint_mlp, landmark_scalers_input, landmark_scalers_target, device):
    """Apply joint MLP refinement to predictions using landmark-wise scalers."""
    try:
        # predictions shape: [19, 2]
        predictions = predictions.reshape(19, 2)
        
        # Apply landmark-wise normalization
        preds_normalized = np.zeros_like(predictions)
        for landmark_idx in range(19):
            pred_coords = predictions[landmark_idx:landmark_idx+1, :]  # [1, 2]
            pred_coords_norm = landmark_scalers_input[landmark_idx].transform(pred_coords)
            preds_normalized[landmark_idx, :] = pred_coords_norm.flatten()
        
        # Flatten to 38-D vector
        preds_flat = preds_normalized.flatten()  # [38]
        
        # Convert to tensor and apply MLP
        preds_tensor = torch.FloatTensor(preds_flat).unsqueeze(0).to(device)  # [1, 38]
        
        with torch.no_grad():
            refined_flat = joint_mlp(preds_tensor).cpu().numpy().flatten()  # [38]
        
        # Reshape back to [19, 2]
        refined_normalized = refined_flat.reshape(19, 2)
        
        # Apply inverse landmark-wise normalization
        refined_coords = np.zeros_like(refined_normalized)
        for landmark_idx in range(19):
            refined_coords_norm = refined_normalized[landmark_idx:landmark_idx+1, :]  # [1, 2]
            refined_coords_denorm = landmark_scalers_target[landmark_idx].inverse_transform(refined_coords_norm)
            refined_coords[landmark_idx, :] = refined_coords_denorm.flatten()
        
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

def main():
    """Main evaluation function."""
    
    parser = argparse.ArgumentParser(
        description='Evaluate Concurrent Joint MLP Refinement Performance')
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
    print("CONCURRENT JOINT MLP REFINEMENT EVALUATION")
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
    
    # Check for joint MLP models
    mlp_dir = os.path.join(args.work_dir, "concurrent_mlp")
    joint_mlp_path = os.path.join(mlp_dir, "joint_mlp_final.pth")
    
    # Check for final model first, then latest, then epoch-specific
    if os.path.exists(joint_mlp_path):
        print(f"✓ Found final joint MLP model: {joint_mlp_path}")
        model_type = "final"
    else:
        # Try latest model
        joint_mlp_latest = os.path.join(mlp_dir, "joint_mlp_latest.pth")
        
        if os.path.exists(joint_mlp_latest):
            joint_mlp_path = joint_mlp_latest
            print(f"✓ Found latest joint MLP model: {joint_mlp_path}")
            model_type = "latest"
        else:
            # Try to find epoch-specific models
            epoch_models = glob.glob(os.path.join(mlp_dir, "joint_mlp_epoch_*.pth"))
            if epoch_models:
                # Get the latest epoch model
                latest_model = max(epoch_models, key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
                epoch_num = latest_model.split('_epoch_')[1].split('.')[0]
                
                joint_mlp_path = latest_model
                print(f"✓ Found epoch {epoch_num} joint MLP model: {joint_mlp_path}")
                model_type = f"epoch_{epoch_num}"
            else:
                print("ERROR: No joint MLP models found.")
                print(f"Searched in: {mlp_dir}")
                print("Available files:")
                if os.path.exists(mlp_dir):
                    for file in os.listdir(mlp_dir):
                        print(f"  - {file}")
                else:
                    print("  MLP directory does not exist")
                print("\nTip: Make sure concurrent training is running and has completed at least one epoch.")
                return
    
    # Load models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    # Load HRNetV2 model
    hrnet_model = init_model(config_path, hrnet_checkpoint, device=device)
    print("✓ HRNetV2 model loaded")
    
    # Load joint MLP model
    joint_mlp = JointMLPRefinementModel().to(device)
    joint_mlp.load_state_dict(torch.load(joint_mlp_path, map_location=device))
    joint_mlp.eval()
    print("✓ Joint MLP model loaded")
    
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
    
    # Load saved landmark-wise scalers
    print("Loading saved landmark-wise scalers...")
    scaler_dir = os.path.join(args.work_dir, "concurrent_mlp")
    
    landmark_scalers_input_path = os.path.join(scaler_dir, "landmark_scalers_input.pkl")
    landmark_scalers_target_path = os.path.join(scaler_dir, "landmark_scalers_target.pkl")
    
    # Check if scalers exist
    scaler_files = [landmark_scalers_input_path, landmark_scalers_target_path]
    missing_scalers = [f for f in scaler_files if not os.path.exists(f)]
    
    if missing_scalers:
        print(f"ERROR: Missing scaler files: {missing_scalers}")
        print("This indicates that concurrent joint MLP training hasn't run yet or scalers weren't saved.")
        print("Please run concurrent training first.")
        return
    
    # Load scalers
    try:
        landmark_scalers_input = joblib.load(landmark_scalers_input_path)
        landmark_scalers_target = joblib.load(landmark_scalers_target_path)
        print("✓ Landmark-wise scalers loaded successfully")
        print(f"  Input scalers: {len(landmark_scalers_input)} landmarks")
        print(f"  Target scalers: {len(landmark_scalers_target)} landmarks")
    except Exception as e:
        print(f"ERROR: Failed to load scalers: {e}")
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
            
            # Run HRNetV2 inference using the standard API
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
                pred_keypoints, joint_mlp, landmark_scalers_input, landmark_scalers_target, device
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
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"📊 Evaluated using {model_type} joint MLP model")
    print(f"📈 HRNetV2 checkpoint: {os.path.basename(hrnet_checkpoint)}")
    
    print(f"\n🏷️  OVERALL PERFORMANCE:")
    print(f"{'Metric':<15} {'HRNetV2':<15} {'Joint MLP':<15} {'Improvement':<15}")
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
    
    print(f"{'Landmark':<20} {'HRNetV2 MRE':<15} {'Joint MLP MRE':<15} {'Improvement':<15}")
    print("-" * 70)
    
    for landmark in problematic_landmarks:
        if landmark in hrnet_per_landmark and landmark in mlp_per_landmark:
            hrnet_err = hrnet_per_landmark[landmark]['mre']
            mlp_err = mlp_per_landmark[landmark]['mre']
            if hrnet_err > 0:
                improvement = (hrnet_err - mlp_err) / hrnet_err * 100
                print(f"{landmark:<20} {hrnet_err:<15.3f} {mlp_err:<15.3f} {improvement:<15.1f}%")
    
    # Save results
    output_dir = os.path.join(args.work_dir, "joint_mlp_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_summary = {
        'hrnet_overall': hrnet_overall,
        'mlp_overall': mlp_overall,
        'improvement_mre': improvement_mre,
        'improvement_std': improvement_std,
        'improvement_median': improvement_median,
        'total_samples': len(hrnet_predictions),
        'model_type': model_type
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
    comparison_df.to_csv(os.path.join(output_dir, "per_landmark_comparison.csv"), index=False)
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Overall comparison
    methods = ['HRNetV2', 'Joint MLP']
    mres = [hrnet_overall['mre'], mlp_overall['mre']]
    stds = [hrnet_overall['std'], mlp_overall['std']]
    
    ax1.bar(methods, mres, yerr=stds, capsize=5, alpha=0.7, color=['skyblue', 'lightcoral'])
    ax1.set_ylabel('Mean Radial Error (pixels)')
    ax1.set_title('Overall Performance Comparison')
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
    ax2.set_title('Per-Landmark Improvement')
    ax2.grid(True, alpha=0.3)
    
    # Error distribution comparison
    hrnet_errors = np.sqrt(np.sum((hrnet_coords - gt_coords)**2, axis=2)).flatten()
    mlp_errors = np.sqrt(np.sum((mlp_coords - gt_coords)**2, axis=2)).flatten()
    
    # Remove invalid landmarks (where gt is [0,0])
    valid_mask = (gt_coords.reshape(-1, 2)[:, 0] > 0) & (gt_coords.reshape(-1, 2)[:, 1] > 0)
    hrnet_errors = hrnet_errors[valid_mask]
    mlp_errors = mlp_errors[valid_mask]
    
    ax3.hist([hrnet_errors, mlp_errors], bins=50, alpha=0.7, 
             label=['HRNetV2', 'Joint MLP'], color=['skyblue', 'lightcoral'])
    ax3.set_xlabel('Radial Error (pixels)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Scatter plot: HRNet vs MLP errors
    sample_indices = np.random.choice(len(hrnet_errors), min(1000, len(hrnet_errors)), replace=False)
    ax4.scatter(hrnet_errors[sample_indices], mlp_errors[sample_indices], 
                alpha=0.5, color='purple')
    
    # Add diagonal line
    max_error = max(np.max(hrnet_errors), np.max(mlp_errors))
    ax4.plot([0, max_error], [0, max_error], 'r--', alpha=0.8, label='No improvement line')
    ax4.set_xlabel('HRNetV2 Error (pixels)')
    ax4.set_ylabel('Joint MLP Error (pixels)')
    ax4.set_title('Error Correlation (Sample)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "joint_mlp_evaluation_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n💾 Results saved to: {output_dir}")
    print(f"   - Per-landmark comparison: per_landmark_comparison.csv")
    print(f"   - Visualization: joint_mlp_evaluation_results.png")
    
    print(f"\n🎉 Evaluation completed!")
    print(f"📈 Overall improvement: {improvement_mre:.1f}% reduction in MRE")
    print(f"🎯 Joint 38-D MLP with landmark-wise scalers and cosine LR")
    print(f"🔧 Evaluated using: {model_type} joint MLP model")
    
    if model_type == "latest":
        print("💡 Note: Training is likely still in progress. Final results may differ.")
    elif "epoch_" in model_type:
        print("💡 Note: Using intermediate checkpoint. Final results may differ.")

if __name__ == "__main__":
    main() 