#!/usr/bin/env python3
"""
Landmark-Specific Improvement Visualization
This script visualizes how curriculum learning and hard-example oversampling
improve performance on specific landmarks, especially problematic ones.
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
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, List, Tuple

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
    """Joint MLP model for landmark coordinate refinement."""
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

def evaluate_landmark_specific_performance(work_dir: str, test_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Evaluate landmark-specific performance for different model epochs."""
    
    # Initialize MMPose scope
    init_default_scope('mmpose')
    
    # Import custom modules
    import custom_cephalometric_dataset
    import custom_transforms
    import cephalometric_dataset_info
    
    # Configuration
    config_path = "Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py"
    
    # Load config
    cfg = Config.fromfile(config_path)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Get landmark information
    landmark_names = cephalometric_dataset_info.landmark_names_in_order
    landmark_cols = cephalometric_dataset_info.original_landmark_cols
    
    # Find available MLP models (different epochs)
    mlp_dir = os.path.join(work_dir, "concurrent_mlp")
    epoch_models = glob.glob(os.path.join(mlp_dir, "joint_mlp_epoch_*.pth"))
    
    if not epoch_models:
        print("❌ No epoch-specific MLP models found")
        return {}
    
    # Sort by epoch number
    epoch_models.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
    
    # Load scalers
    landmark_scalers_input_path = os.path.join(mlp_dir, "landmark_scalers_input.pkl")
    landmark_scalers_target_path = os.path.join(mlp_dir, "landmark_scalers_target.pkl")
    
    if not os.path.exists(landmark_scalers_input_path) or not os.path.exists(landmark_scalers_target_path):
        print("❌ Landmark scalers not found")
        return {}
    
    landmark_scalers_input = joblib.load(landmark_scalers_input_path)
    landmark_scalers_target = joblib.load(landmark_scalers_target_path)
    
    # Find HRNet checkpoint
    hrnet_checkpoint_pattern = os.path.join(work_dir, "best_NME_epoch_*.pth")
    hrnet_checkpoints = glob.glob(hrnet_checkpoint_pattern)
    
    if not hrnet_checkpoints:
        hrnet_checkpoint_pattern = os.path.join(work_dir, "epoch_*.pth")
        hrnet_checkpoints = glob.glob(hrnet_checkpoint_pattern)
    
    if not hrnet_checkpoints:
        print("❌ No HRNet checkpoints found")
        return {}
    
    hrnet_checkpoint = max(hrnet_checkpoints, key=os.path.getctime)
    hrnet_model = init_model(config_path, hrnet_checkpoint, device=device)
    
    results = {}
    
    # Evaluate for multiple epochs (sample every few epochs to avoid too much computation)
    selected_models = epoch_models[::max(1, len(epoch_models)//5)]  # Sample up to 5 epochs
    
    for model_path in selected_models:
        epoch_num = int(model_path.split('_epoch_')[1].split('.')[0])
        print(f"🔄 Evaluating epoch {epoch_num}...")
        
        # Load MLP model
        joint_mlp = JointMLPRefinementModel().to(device)
        joint_mlp.load_state_dict(torch.load(model_path, map_location=device))
        joint_mlp.eval()
        
        # Evaluate on test set
        landmark_errors = {name: [] for name in landmark_names}
        
        from tqdm import tqdm
        
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Epoch {epoch_num}", leave=False):
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
                results_inference = inference_topdown(hrnet_model, img_array, bboxes=bbox, bbox_format='xyxy')
                
                if results_inference and len(results_inference) > 0:
                    pred_keypoints = results_inference[0].pred_instances.keypoints[0]
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
                
                # Compute per-landmark errors
                for i, landmark_name in enumerate(landmark_names):
                    if gt_keypoints[i, 0] > 0 and gt_keypoints[i, 1] > 0:  # Valid landmark
                        error = np.sqrt(np.sum((refined_keypoints[i] - gt_keypoints[i])**2))
                        landmark_errors[landmark_name].append(error)
                
            except Exception as e:
                continue
        
        # Compute statistics for this epoch
        epoch_stats = {}
        for landmark_name in landmark_names:
            if landmark_errors[landmark_name]:
                errors = np.array(landmark_errors[landmark_name])
                epoch_stats[landmark_name] = {
                    'mre': np.mean(errors),
                    'std': np.std(errors),
                    'median': np.median(errors),
                    'count': len(errors)
                }
            else:
                epoch_stats[landmark_name] = {'mre': 0, 'std': 0, 'median': 0, 'count': 0}
        
        results[f'epoch_{epoch_num}'] = epoch_stats
    
    return results

def plot_landmark_improvements(results: Dict[str, Dict[str, float]], output_dir: str, landmark_names: List[str]):
    """Plot landmark-specific improvements over training epochs."""
    
    if not results:
        print("❌ No results to plot")
        return
    
    # Extract epoch numbers and sort
    epochs = sorted([int(k.split('_')[1]) for k in results.keys()])
    
    # Focus on problematic landmarks
    problematic_landmarks = ['sella', 'Gonion', 'PNS', 'A_point', 'B_point']
    available_problematic = [name for name in problematic_landmarks if name in landmark_names]
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Curriculum Learning: Landmark-Specific Improvements', fontsize=16, fontweight='bold')
    
    # 1. Problematic landmarks improvement over epochs
    ax = axes[0, 0]
    colors = plt.cm.Set1(np.linspace(0, 1, len(available_problematic)))
    
    for i, landmark in enumerate(available_problematic):
        mres = []
        for epoch in epochs:
            epoch_key = f'epoch_{epoch}'
            if epoch_key in results and landmark in results[epoch_key]:
                mres.append(results[epoch_key][landmark]['mre'])
            else:
                mres.append(np.nan)
        
        ax.plot(epochs, mres, 'o-', color=colors[i], label=landmark, linewidth=2, markersize=6)
    
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Mean Radial Error (pixels)')
    ax.set_title('Problematic Landmarks Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Overall improvement distribution
    ax = axes[0, 1]
    
    if len(epochs) >= 2:
        first_epoch = f'epoch_{epochs[0]}'
        last_epoch = f'epoch_{epochs[-1]}'
        
        improvements = []
        landmark_labels = []
        
        for landmark in landmark_names:
            if (landmark in results[first_epoch] and landmark in results[last_epoch] and
                results[first_epoch][landmark]['count'] > 0 and results[last_epoch][landmark]['count'] > 0):
                
                initial_error = results[first_epoch][landmark]['mre']
                final_error = results[last_epoch][landmark]['mre']
                
                if initial_error > 0:
                    improvement = (initial_error - final_error) / initial_error * 100
                    improvements.append(improvement)
                    landmark_labels.append(landmark)
        
        if improvements:
            colors = ['green' if x > 0 else 'red' for x in improvements]
            bars = ax.barh(landmark_labels, improvements, color=colors, alpha=0.7)
            ax.set_xlabel('Improvement (%)')
            ax.set_title(f'Overall Improvement\n(Epoch {epochs[0]} → {epochs[-1]})')
            ax.grid(True, alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            
            # Add value labels
            for bar, improvement in zip(bars, improvements):
                width = bar.get_width()
                ax.text(width + (1 if width >= 0 else -1), bar.get_y() + bar.get_height()/2,
                       f'{improvement:.1f}%', ha='left' if width >= 0 else 'right', va='center')
    else:
        ax.text(0.5, 0.5, 'Insufficient epochs for comparison', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Overall Improvement (Not Available)')
    
    # 3. Error distribution comparison (first vs last epoch)
    ax = axes[0, 2]
    
    if len(epochs) >= 2:
        first_epoch = f'epoch_{epochs[0]}'
        last_epoch = f'epoch_{epochs[-1]}'
        
        first_errors = []
        last_errors = []
        
        for landmark in landmark_names:
            if (landmark in results[first_epoch] and landmark in results[last_epoch] and
                results[first_epoch][landmark]['count'] > 0 and results[last_epoch][landmark]['count'] > 0):
                first_errors.append(results[first_epoch][landmark]['mre'])
                last_errors.append(results[last_epoch][landmark]['mre'])
        
        if first_errors and last_errors:
            ax.hist([first_errors, last_errors], bins=15, alpha=0.7, 
                   label=[f'Epoch {epochs[0]}', f'Epoch {epochs[-1]}'],
                   color=['lightcoral', 'lightgreen'])
            ax.set_xlabel('Mean Radial Error (pixels)')
            ax.set_ylabel('Number of Landmarks')
            ax.set_title('Error Distribution Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Insufficient epochs for comparison', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Error Distribution (Not Available)')
    
    # 4. Landmark ranking by improvement
    ax = axes[1, 0]
    
    if len(epochs) >= 2:
        first_epoch = f'epoch_{epochs[0]}'
        last_epoch = f'epoch_{epochs[-1]}'
        
        landmark_improvements = []
        
        for landmark in landmark_names:
            if (landmark in results[first_epoch] and landmark in results[last_epoch] and
                results[first_epoch][landmark]['count'] > 0 and results[last_epoch][landmark]['count'] > 0):
                
                initial_error = results[first_epoch][landmark]['mre']
                final_error = results[last_epoch][landmark]['mre']
                
                if initial_error > 0:
                    improvement = (initial_error - final_error) / initial_error * 100
                    landmark_improvements.append((landmark, improvement))
        
        if landmark_improvements:
            # Sort by improvement (best first)
            landmark_improvements.sort(key=lambda x: x[1], reverse=True)
            
            # Take top 10 for visualization
            top_improvements = landmark_improvements[:10]
            landmarks_top = [x[0] for x in top_improvements]
            improvements_top = [x[1] for x in top_improvements]
            
            colors = ['green' if x > 0 else 'red' for x in improvements_top]
            ax.barh(landmarks_top, improvements_top, color=colors, alpha=0.7)
            ax.set_xlabel('Improvement (%)')
            ax.set_title('Top 10 Landmark Improvements')
            ax.grid(True, alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    else:
        ax.text(0.5, 0.5, 'Insufficient epochs for ranking', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Landmark Ranking (Not Available)')
    
    # 5. Curriculum learning phases
    ax = axes[1, 1]
    
    # Assume curriculum starts at epoch 5 (from config)
    curriculum_start = 5
    
    if len(epochs) > 1:
        # Calculate average error across all landmarks for each epoch
        avg_errors = []
        for epoch in epochs:
            epoch_key = f'epoch_{epoch}'
            epoch_errors = []
            for landmark in landmark_names:
                if landmark in results[epoch_key] and results[epoch_key][landmark]['count'] > 0:
                    epoch_errors.append(results[epoch_key][landmark]['mre'])
            
            if epoch_errors:
                avg_errors.append(np.mean(epoch_errors))
            else:
                avg_errors.append(np.nan)
        
        # Plot with curriculum phase distinction
        pre_curriculum_epochs = [e for e in epochs if e < curriculum_start]
        post_curriculum_epochs = [e for e in epochs if e >= curriculum_start]
        
        if pre_curriculum_epochs:
            pre_errors = [avg_errors[epochs.index(e)] for e in pre_curriculum_epochs]
            ax.plot(pre_curriculum_epochs, pre_errors, 'o-', color='red', 
                   label='Pre-Curriculum', linewidth=2, markersize=6)
        
        if post_curriculum_epochs:
            post_errors = [avg_errors[epochs.index(e)] for e in post_curriculum_epochs]
            ax.plot(post_curriculum_epochs, post_errors, 'o-', color='green', 
                   label='Post-Curriculum', linewidth=2, markersize=6)
        
        ax.axvline(x=curriculum_start, color='blue', linestyle='--', alpha=0.7, 
                  label=f'Curriculum Start (Epoch {curriculum_start})')
        
        ax.set_xlabel('Training Epoch')
        ax.set_ylabel('Average MRE (pixels)')
        ax.set_title('Curriculum Learning Phases')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Curriculum Phases (Not Available)')
    
    # 6. Consistency improvement (std deviation)
    ax = axes[1, 2]
    
    if len(epochs) >= 2:
        first_epoch = f'epoch_{epochs[0]}'
        last_epoch = f'epoch_{epochs[-1]}'
        
        consistency_improvements = []
        consistency_landmarks = []
        
        for landmark in available_problematic:  # Focus on problematic landmarks
            if (landmark in results[first_epoch] and landmark in results[last_epoch] and
                results[first_epoch][landmark]['count'] > 0 and results[last_epoch][landmark]['count'] > 0):
                
                initial_std = results[first_epoch][landmark]['std']
                final_std = results[last_epoch][landmark]['std']
                
                if initial_std > 0:
                    consistency_improvement = (initial_std - final_std) / initial_std * 100
                    consistency_improvements.append(consistency_improvement)
                    consistency_landmarks.append(landmark)
        
        if consistency_improvements:
            colors = ['green' if x > 0 else 'red' for x in consistency_improvements]
            ax.barh(consistency_landmarks, consistency_improvements, color=colors, alpha=0.7)
            ax.set_xlabel('Consistency Improvement (%)')
            ax.set_title('Prediction Consistency Improvement\n(Std Deviation Reduction)')
            ax.grid(True, alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    else:
        ax.text(0.5, 0.5, 'Insufficient epochs for comparison', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Consistency Improvement (Not Available)')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "landmark_improvements_curriculum.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path

def main():
    """Main visualization function."""
    
    parser = argparse.ArgumentParser(description='Visualize Landmark-Specific Curriculum Learning Improvements')
    parser.add_argument(
        '--work_dir',
        type=str,
        default='work_dirs/hrnetv2_w18_cephalometric_concurrent_mlp_v5',
        help='Work directory containing concurrent MLP training results'
    )
    parser.add_argument(
        '--test_split_file',
        type=str,
        default=None,
        help='Path to a text file containing patient IDs for the test set, one ID per line.'
    )
    args = parser.parse_args()
    
    print("="*80)
    print("LANDMARK-SPECIFIC CURRICULUM LEARNING ANALYSIS")
    print("="*80)
    
    # Load test data
    data_file_path = "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json"
    main_df = pd.read_json(data_file_path)
    
    # Split test data (same logic as evaluation scripts)
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
        print("❌ No test samples found")
        return
    
    # Limit test set size for faster analysis
    if len(test_df) > 100:
        test_df = test_df.sample(n=100, random_state=42).reset_index(drop=True)
        print(f"⚠️  Limited test set to 100 samples for faster analysis")
    
    print(f"✓ Using {len(test_df)} test samples")
    
    # Evaluate landmark-specific performance
    print("🔄 Evaluating landmark-specific performance across epochs...")
    results = evaluate_landmark_specific_performance(args.work_dir, test_df)
    
    if not results:
        print("❌ No evaluation results obtained")
        return
    
    print(f"✓ Evaluated {len(results)} epochs")
    
    # Get landmark names
    try:
        import cephalometric_dataset_info
        landmark_names = cephalometric_dataset_info.landmark_names_in_order
    except ImportError:
        print("❌ Could not import landmark names")
        return
    
    # Create output directory
    output_dir = os.path.join(args.work_dir, "landmark_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations
    print("📈 Generating landmark improvement visualizations...")
    plot_path = plot_landmark_improvements(results, output_dir, landmark_names)
    print(f"✓ Visualizations saved to: {plot_path}")
    
    # Save detailed results
    results_path = os.path.join(output_dir, "landmark_results.json")
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Detailed results saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("LANDMARK ANALYSIS SUMMARY")
    print("="*80)
    
    if len(results) >= 2:
        epochs = sorted([int(k.split('_')[1]) for k in results.keys()])
        first_epoch = f'epoch_{epochs[0]}'
        last_epoch = f'epoch_{epochs[-1]}'
        
        # Calculate improvements for problematic landmarks
        problematic_landmarks = ['sella', 'Gonion', 'PNS', 'A_point', 'B_point']
        
        print(f"📊 **Epoch Range**: {epochs[0]} → {epochs[-1]}")
        print(f"🎯 **Problematic Landmarks Analysis**:")
        
        for landmark in problematic_landmarks:
            if (landmark in results[first_epoch] and landmark in results[last_epoch] and
                results[first_epoch][landmark]['count'] > 0 and results[last_epoch][landmark]['count'] > 0):
                
                initial_error = results[first_epoch][landmark]['mre']
                final_error = results[last_epoch][landmark]['mre']
                improvement = (initial_error - final_error) / initial_error * 100
                
                print(f"   • {landmark}: {initial_error:.2f} → {final_error:.2f} px ({improvement:+.1f}%)")
        
        # Overall statistics
        all_improvements = []
        for landmark in landmark_names:
            if (landmark in results[first_epoch] and landmark in results[last_epoch] and
                results[first_epoch][landmark]['count'] > 0 and results[last_epoch][landmark]['count'] > 0):
                
                initial_error = results[first_epoch][landmark]['mre']
                final_error = results[last_epoch][landmark]['mre']
                
                if initial_error > 0:
                    improvement = (initial_error - final_error) / initial_error * 100
                    all_improvements.append(improvement)
        
        if all_improvements:
            avg_improvement = np.mean(all_improvements)
            improved_landmarks = sum(1 for x in all_improvements if x > 0)
            total_landmarks = len(all_improvements)
            
            print(f"\n📈 **Overall Statistics**:")
            print(f"   • Average improvement: {avg_improvement:.1f}%")
            print(f"   • Landmarks improved: {improved_landmarks}/{total_landmarks} ({improved_landmarks/total_landmarks*100:.1f}%)")
            print(f"   • Best improvement: {max(all_improvements):.1f}%")
            print(f"   • Worst change: {min(all_improvements):.1f}%")
    
    print(f"\n💾 **Results saved to**: {output_dir}")
    print(f"   - Visualizations: landmark_improvements_curriculum.png")
    print(f"   - Detailed data: landmark_results.json")
    
    print("\n🎉 Landmark analysis completed!")

if __name__ == "__main__":
    main() 