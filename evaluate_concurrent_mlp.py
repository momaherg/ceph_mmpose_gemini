#!/usr/bin/env python3
"""
Concurrent Joint MLP Performance Evaluation Script
This script evaluates the performance improvement from joint MLP refinement
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
import cv2
from matplotlib.patches import Circle

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

def calculate_angle(p1, p2, p3):
    """
    Calculate angle at point p2 formed by points p1-p2-p3.
    Args:
        p1, p2, p3: numpy arrays of shape (2,) representing (x, y) coordinates
    Returns:
        angle in degrees
    """
    # Create vectors
    v1 = p1 - p2  # Vector from p2 to p1
    v2 = p3 - p2  # Vector from p2 to p3
    
    # Calculate angle using dot product
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    # Clamp to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Convert to degrees
    angle = np.arccos(cos_angle) * 180.0 / np.pi
    
    return angle

def classify_anb_angle(anb_angle):
    """
    Classify ANB angle according to orthodontic standards.
    
    Args:
        anb_angle: ANB angle in degrees
    Returns:
        Classification string: 'Class I', 'Class II', or 'Class III'
    """
    if np.isnan(anb_angle):
        return 'Invalid'
    elif anb_angle >= 2.0 and anb_angle <= 4.0:
        return 'Class I'  # Balanced jaw relationship
    elif anb_angle > 4.0:
        return 'Class II'  # Protruding upper jaw
    else:  # anb_angle < 2.0
        return 'Class III'  # Protruding lower jaw

def classify_snb_angle(snb_angle):
    """
    Classify SNB angle according to orthodontic standards for mandibular position.
    
    Args:
        snb_angle: SNB angle in degrees
    Returns:
        Classification string: 'Normal', 'Retrognathic', or 'Prognathic'
    """
    if np.isnan(snb_angle):
        return 'Invalid'
    elif snb_angle >= 74.6 and snb_angle <= 78.7:
        return 'Normal'  # Normal mandibular position
    elif snb_angle < 74.6:
        return 'Retrognathic'  # Lower jaw positioned further back
    else:  # snb_angle > 78.7
        return 'Prognathic'  # Lower jaw positioned further forward

def calculate_snb_angle(landmarks):
    """
    Calculate SNB angle from cephalometric landmarks.
    SNB = angle at Nasion formed by Sella-Nasion-B point
    
    Args:
        landmarks: numpy array of shape (19, 2) with landmark coordinates
    Returns:
        SNB angle in degrees
    """
    # Get landmark indices
    sella_idx = 0    # sella
    nasion_idx = 1   # nasion  
    b_point_idx = 3  # B_point
    
    # Get landmark coordinates
    sella = landmarks[sella_idx]
    nasion = landmarks[nasion_idx]
    b_point = landmarks[b_point_idx]
    
    # Check if landmarks are valid (not [0,0])
    if (np.array_equal(sella, [0, 0]) or np.array_equal(nasion, [0, 0]) or 
        np.array_equal(b_point, [0, 0])):
        return np.nan
    
    # Calculate SNB angle (Sella-Nasion-B point)
    snb_angle = calculate_angle(sella, nasion, b_point)
    
    return snb_angle

def calculate_anb_angle(landmarks):
    """
    Calculate ANB angle from cephalometric landmarks.
    ANB = SNA - SNB
    
    Args:
        landmarks: numpy array of shape (19, 2) with landmark coordinates
    Returns:
        ANB angle in degrees
    """
    # Get landmark indices
    sella_idx = 0    # sella
    nasion_idx = 1   # nasion  
    a_point_idx = 2  # A_point
    b_point_idx = 3  # B_point
    
    # Get landmark coordinates
    sella = landmarks[sella_idx]
    nasion = landmarks[nasion_idx]
    a_point = landmarks[a_point_idx]
    b_point = landmarks[b_point_idx]
    
    # Check if landmarks are valid (not [0,0])
    if (np.array_equal(sella, [0, 0]) or np.array_equal(nasion, [0, 0]) or 
        np.array_equal(a_point, [0, 0]) or np.array_equal(b_point, [0, 0])):
        return np.nan
    
    # Calculate SNA angle (Sella-Nasion-A point)
    sna_angle = calculate_angle(sella, nasion, a_point)
    
    # Calculate SNB angle (Sella-Nasion-B point)  
    snb_angle = calculate_angle(sella, nasion, b_point)
    
    # ANB = SNA - SNB
    anb_angle = sna_angle - snb_angle
    
    return anb_angle

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

def plot_landmarks_on_image(image, pred_coords, gt_coords, landmark_names, title, save_path):
    """
    Plot landmarks on image showing both predictions and ground truth.
    
    Args:
        image: numpy array of shape (224, 224, 3)
        pred_coords: predicted coordinates, shape (19, 2)
        gt_coords: ground truth coordinates, shape (19, 2)
        landmark_names: list of landmark names
        title: plot title
        save_path: path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Display image
    ax.imshow(image)
    
    # Plot landmarks
    for i, (pred, gt, name) in enumerate(zip(pred_coords, gt_coords, landmark_names)):
        # Skip invalid landmarks
        if gt[0] <= 0 or gt[1] <= 0:
            continue
            
        # Plot ground truth (green circles)
        circle_gt = Circle((gt[0], gt[1]), radius=3, color='green', alpha=0.7)
        ax.add_patch(circle_gt)
        
        # Plot predictions (red crosses)
        ax.plot(pred[0], pred[1], 'rx', markersize=8, markeredgewidth=2)
        
        # Add landmark labels
        ax.annotate(f'{i}', (gt[0], gt[1]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=8, color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    # Calculate overall error
    valid_mask = (gt_coords[:, 0] > 0) & (gt_coords[:, 1] > 0)
    if np.any(valid_mask):
        errors = np.sqrt(np.sum((pred_coords[valid_mask] - gt_coords[valid_mask])**2, axis=1))
        mean_error = np.mean(errors)
        ax.set_title(f'{title}\nMean Error: {mean_error:.2f} pixels', fontsize=14)
    else:
        ax.set_title(title, fontsize=14)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
               markersize=8, label='Ground Truth', alpha=0.7),
        Line2D([0], [0], marker='x', color='red', markersize=8, 
               markeredgewidth=2, label='Prediction', linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_xlim(0, 224)
    ax.set_ylim(224, 0)  # Flip y-axis for image coordinates
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

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
    
    # Priority order: best model (matching best HRNet), final, latest, then epoch-specific
    mlp_joint_best = os.path.join(mlp_dir, "mlp_joint_best.pth")
    mlp_joint_final = os.path.join(mlp_dir, "mlp_joint_final.pth")
    mlp_joint_latest = os.path.join(mlp_dir, "mlp_joint_latest.pth")
    
    # Check for best model first (corresponds to best HRNetV2 checkpoint)
    if os.path.exists(mlp_joint_best):
        mlp_joint_path = mlp_joint_best
        print(f"✓ Found best joint MLP model: {mlp_joint_path}")
        model_type = "best"
        
        # Try to read best model summary for additional info
        summary_path = os.path.join(mlp_dir, "best_model_summary.txt")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r') as f:
                    summary_content = f.read()
                    # Extract epoch and NME from summary
                    for line in summary_content.split('\n'):
                        if 'Best Epoch:' in line:
                            best_epoch = line.split(':')[1].strip()
                            print(f"✓ Best model from epoch: {best_epoch}")
                        elif 'Best NME:' in line:
                            best_nme = line.split(':')[1].strip()
                            print(f"✓ Best validation NME: {best_nme}")
            except Exception as e:
                print(f"⚠️  Could not read best model summary: {e}")
    
    elif os.path.exists(mlp_joint_final):
        mlp_joint_path = mlp_joint_final
        print(f"✓ Found final joint MLP model: {mlp_joint_path}")
        model_type = "final"
        print("ℹ️  Using final model (best model not available)")
    
    elif os.path.exists(mlp_joint_latest):
        mlp_joint_path = mlp_joint_latest
        print(f"✓ Found latest joint MLP model: {mlp_joint_path}")
        model_type = "latest"
        print("ℹ️  Using latest model (final model not available)")
    
    else:
        # Try to find epoch-specific models
        epoch_models = glob.glob(os.path.join(mlp_dir, "mlp_joint_epoch_*.pth"))
        best_epoch_models = glob.glob(os.path.join(mlp_dir, "mlp_joint_best_NME_epoch_*.pth"))
        
        if best_epoch_models:
            # Prefer best epoch models
            latest_best_model = max(best_epoch_models, key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
            epoch_num = latest_best_model.split('_epoch_')[1].split('.')[0]
            
            mlp_joint_path = latest_best_model
            print(f"✓ Found best epoch {epoch_num} joint MLP model: {mlp_joint_path}")
            model_type = f"best_epoch_{epoch_num}"
            
        elif epoch_models:
            # Fall back to regular epoch models
            latest_joint_model = max(epoch_models, key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
            epoch_num = latest_joint_model.split('_epoch_')[1].split('.')[0]
            
            mlp_joint_path = latest_joint_model
            print(f"✓ Found epoch {epoch_num} joint MLP model: {mlp_joint_path}")
            model_type = f"epoch_{epoch_num}"
            
        else:
            print("ERROR: Joint MLP model not found.")
            print(f"Searched in: {mlp_dir}")
            print("Available files:")
            if os.path.exists(mlp_dir):
                for file in os.listdir(mlp_dir):
                    print(f"  - {file}")
            else:
                print("  MLP directory does not exist")
            print("\nTip: Make sure concurrent joint training is running and has completed at least one epoch.")
            return
    
    # Load models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    # Load HRNetV2 model
    hrnet_model = init_model(config_path, hrnet_checkpoint, device=device)
    print("✓ HRNetV2 model loaded")
    
    # Load joint MLP model
    mlp_joint = JointMLPRefinementModel().to(device)
    mlp_joint.load_state_dict(torch.load(mlp_joint_path, map_location=device))
    mlp_joint.eval()
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
    
    # Load saved joint normalization scalers
    print("Loading saved joint normalization scalers...")
    scaler_dir = os.path.join(args.work_dir, "concurrent_mlp")
    
    scaler_input_path = os.path.join(scaler_dir, "scaler_joint_input.pkl")
    scaler_target_path = os.path.join(scaler_dir, "scaler_joint_target.pkl")
    
    # Check if scalers exist
    scaler_files = [scaler_input_path, scaler_target_path]
    missing_scalers = [f for f in scaler_files if not os.path.exists(f)]
    
    if missing_scalers:
        print(f"ERROR: Missing joint scaler files: {missing_scalers}")
        print("This indicates that concurrent joint MLP training hasn't run yet or scalers weren't saved.")
        print("Please run concurrent joint training first.")
        return
    
    # Load scalers
    try:
        scaler_input = joblib.load(scaler_input_path)
        scaler_target = joblib.load(scaler_target_path)
        print("✓ Joint normalization scalers loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load joint scalers: {e}")
        return
    
    # Evaluation on test set
    print(f"\n🔄 Running evaluation on {len(test_df)} test samples...")
    
    # Storage for results
    hrnet_predictions = []
    mlp_predictions = []
    ground_truths = []
    test_images = []
    test_indices = []
    
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
                pred_keypoints, mlp_joint, scaler_input, scaler_target, device
            )
            
            # Store results
            hrnet_predictions.append(pred_keypoints)
            mlp_predictions.append(refined_keypoints)
            ground_truths.append(gt_keypoints)
            test_images.append(img_array)
            test_indices.append(idx)
            
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
    
    # Calculate ANB and SNB angles for all samples
    print("\n📐 Computing ANB and SNB angles...")
    hrnet_anb_angles = []
    mlp_anb_angles = []
    gt_anb_angles = []
    hrnet_snb_angles = []
    mlp_snb_angles = []
    gt_snb_angles = []
    
    for i in range(len(ground_truths)):
        # ANB angles
        gt_anb = calculate_anb_angle(gt_coords[i])
        hrnet_anb = calculate_anb_angle(hrnet_coords[i])
        mlp_anb = calculate_anb_angle(mlp_coords[i])
        
        # SNB angles
        gt_snb = calculate_snb_angle(gt_coords[i])
        hrnet_snb = calculate_snb_angle(hrnet_coords[i])
        mlp_snb = calculate_snb_angle(mlp_coords[i])
        
        # Store ANB angles if valid
        if not (np.isnan(gt_anb) or np.isnan(hrnet_anb) or np.isnan(mlp_anb)):
            gt_anb_angles.append(gt_anb)
            hrnet_anb_angles.append(hrnet_anb)
            mlp_anb_angles.append(mlp_anb)
        
        # Store SNB angles if valid
        if not (np.isnan(gt_snb) or np.isnan(hrnet_snb) or np.isnan(mlp_snb)):
            gt_snb_angles.append(gt_snb)
            hrnet_snb_angles.append(hrnet_snb)
            mlp_snb_angles.append(mlp_snb)
    
    gt_anb_angles = np.array(gt_anb_angles)
    hrnet_anb_angles = np.array(hrnet_anb_angles)
    mlp_anb_angles = np.array(mlp_anb_angles)
    gt_snb_angles = np.array(gt_snb_angles)
    hrnet_snb_angles = np.array(hrnet_snb_angles)
    mlp_snb_angles = np.array(mlp_snb_angles)
    
    # Compute ANB angle errors
    hrnet_anb_errors = np.abs(hrnet_anb_angles - gt_anb_angles)
    mlp_anb_errors = np.abs(mlp_anb_angles - gt_anb_angles)
    
    # Compute SNB angle errors
    hrnet_snb_errors = np.abs(hrnet_snb_angles - gt_snb_angles)
    mlp_snb_errors = np.abs(mlp_snb_angles - gt_snb_angles)
    
    # Compute ANB angle classifications
    print("📊 Computing ANB angle classifications...")
    gt_classifications = [classify_anb_angle(angle) for angle in gt_anb_angles]
    hrnet_classifications = [classify_anb_angle(angle) for angle in hrnet_anb_angles]
    mlp_classifications = [classify_anb_angle(angle) for angle in mlp_anb_angles]
    
    # Calculate classification accuracy
    hrnet_correct = sum(1 for gt, pred in zip(gt_classifications, hrnet_classifications) if gt == pred)
    mlp_correct = sum(1 for gt, pred in zip(gt_classifications, mlp_classifications) if gt == pred)
    
    total_valid = len(gt_classifications)
    hrnet_accuracy = (hrnet_correct / total_valid) * 100 if total_valid > 0 else 0
    mlp_accuracy = (mlp_correct / total_valid) * 100 if total_valid > 0 else 0
    
    # Calculate per-class metrics
    from collections import Counter
    gt_class_counts = Counter(gt_classifications)
    hrnet_class_counts = Counter(hrnet_classifications)
    mlp_class_counts = Counter(mlp_classifications)
    
    # Calculate confusion matrices
    classes = ['Class I', 'Class II', 'Class III']
    hrnet_confusion = np.zeros((3, 3), dtype=int)
    mlp_confusion = np.zeros((3, 3), dtype=int)
    
    class_to_idx = {'Class I': 0, 'Class II': 1, 'Class III': 2}
    
    for gt, hrnet_pred, mlp_pred in zip(gt_classifications, hrnet_classifications, mlp_classifications):
        if gt in class_to_idx and hrnet_pred in class_to_idx:
            hrnet_confusion[class_to_idx[gt], class_to_idx[hrnet_pred]] += 1
        if gt in class_to_idx and mlp_pred in class_to_idx:
            mlp_confusion[class_to_idx[gt], class_to_idx[mlp_pred]] += 1
    
    # Compute SNB angle classifications
    gt_snb_classifications = []
    hrnet_snb_classifications = []
    mlp_snb_classifications = []
    
    if len(gt_snb_angles) > 0:
        gt_snb_classifications = [classify_snb_angle(angle) for angle in gt_snb_angles]
        hrnet_snb_classifications = [classify_snb_angle(angle) for angle in hrnet_snb_angles]
        mlp_snb_classifications = [classify_snb_angle(angle) for angle in mlp_snb_angles]
        
        # Calculate SNB classification accuracy
        hrnet_snb_correct = sum(1 for gt, pred in zip(gt_snb_classifications, hrnet_snb_classifications) if gt == pred)
        mlp_snb_correct = sum(1 for gt, pred in zip(gt_snb_classifications, mlp_snb_classifications) if gt == pred)
        
        total_snb_valid = len(gt_snb_classifications)
        hrnet_snb_accuracy = (hrnet_snb_correct / total_snb_valid) * 100 if total_snb_valid > 0 else 0
        mlp_snb_accuracy = (mlp_snb_correct / total_snb_valid) * 100 if total_snb_valid > 0 else 0
        
        # Calculate per-class metrics for SNB
        gt_snb_class_counts = Counter(gt_snb_classifications)
        hrnet_snb_class_counts = Counter(hrnet_snb_classifications)
        mlp_snb_class_counts = Counter(mlp_snb_classifications)
        
        # Calculate SNB confusion matrices
        snb_classes = ['Normal', 'Retrognathic', 'Prognathic']
        hrnet_snb_confusion = np.zeros((3, 3), dtype=int)
        mlp_snb_confusion = np.zeros((3, 3), dtype=int)
        
        snb_class_to_idx = {'Normal': 0, 'Retrognathic': 1, 'Prognathic': 2}
        
        for gt, hrnet_pred, mlp_pred in zip(gt_snb_classifications, hrnet_snb_classifications, mlp_snb_classifications):
            if gt in snb_class_to_idx and hrnet_pred in snb_class_to_idx:
                hrnet_snb_confusion[snb_class_to_idx[gt], snb_class_to_idx[hrnet_pred]] += 1
            if gt in snb_class_to_idx and mlp_pred in snb_class_to_idx:
                mlp_snb_confusion[snb_class_to_idx[gt], snb_class_to_idx[mlp_pred]] += 1
    
    # Find best and worst examples based on overall MRE
    sample_errors = []
    for i in range(len(hrnet_predictions)):
        # Calculate per-sample MRE for HRNet
        valid_mask = (gt_coords[i, :, 0] > 0) & (gt_coords[i, :, 1] > 0)
        if np.any(valid_mask):
            errors = np.sqrt(np.sum((hrnet_coords[i, valid_mask] - gt_coords[i, valid_mask])**2, axis=1))
            sample_mre = np.mean(errors)
            sample_errors.append((sample_mre, i))
    
    # Sort by error
    sample_errors.sort(key=lambda x: x[0])
    
    # Get best and worst examples (top 3 and bottom 3)
    best_examples = sample_errors[:3]  # Lowest errors
    worst_examples = sample_errors[-3:]  # Highest errors
    
    # Print results
    print("\n" + "="*80)
    print("JOINT MLP EVALUATION RESULTS")
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
    
    # ANB angle results
    if len(gt_anb_angles) > 0:
        print(f"\n📐 ANB ANGLE ANALYSIS:")
        print(f"{'Metric':<20} {'HRNetV2':<15} {'Joint MLP':<15} {'Improvement':<15}")
        print("-" * 70)
        
        hrnet_anb_mae = np.mean(hrnet_anb_errors)
        mlp_anb_mae = np.mean(mlp_anb_errors)
        anb_improvement = (hrnet_anb_mae - mlp_anb_mae) / hrnet_anb_mae * 100
        
        hrnet_anb_std = np.std(hrnet_anb_errors)
        mlp_anb_std = np.std(mlp_anb_errors)
        
        print(f"{'Mean Abs Error (°)':<20} {hrnet_anb_mae:<15.2f} {mlp_anb_mae:<15.2f} {anb_improvement:<15.1f}%")
        print(f"{'Std Dev (°)':<20} {hrnet_anb_std:<15.2f} {mlp_anb_std:<15.2f}")
        print(f"{'Max Error (°)':<20} {np.max(hrnet_anb_errors):<15.2f} {np.max(mlp_anb_errors):<15.2f}")
        print(f"{'Samples Analyzed':<20} {len(gt_anb_angles)}")
        
        # ANB Classification Results
        print(f"\n🏷️  ANB CLASSIFICATION ACCURACY:")
        print(f"{'Method':<15} {'Accuracy':<15} {'Correct/Total':<15}")
        print("-" * 50)
        print(f"{'HRNetV2':<15} {hrnet_accuracy:<15.1f}% {hrnet_correct}/{total_valid}")
        print(f"{'Joint MLP':<15} {mlp_accuracy:<15.1f}% {mlp_correct}/{total_valid}")
        
        accuracy_improvement = mlp_accuracy - hrnet_accuracy
        print(f"{'Improvement':<15} {accuracy_improvement:<15.1f}% {'(+' + str(mlp_correct - hrnet_correct) + ' samples)'}")
        
        # Class distribution
        print(f"\n📋 GROUND TRUTH CLASS DISTRIBUTION:")
        for class_name in classes:
            count = gt_class_counts.get(class_name, 0)
            percentage = (count / total_valid) * 100 if total_valid > 0 else 0
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        
        # Confusion matrix summary
        print(f"\n🔍 CLASSIFICATION ERRORS SUMMARY:")
        print("HRNetV2 Confusion Matrix:")
        print("     Pred:  I  II III")
        for i, true_class in enumerate(classes):
            row_str = f"GT {true_class[:3]:>3}: "
            for j in range(3):
                row_str += f"{hrnet_confusion[i,j]:>3}"
            print(row_str)
        
        print("\nJoint MLP Confusion Matrix:")
        print("     Pred:  I  II III")
        for i, true_class in enumerate(classes):
            row_str = f"GT {true_class[:3]:>3}: "
            for j in range(3):
                row_str += f"{mlp_confusion[i,j]:>3}"
            print(row_str)
    else:
        print(f"\n⚠️  No valid ANB angles could be computed (missing landmarks)")
    
    # SNB angle results
    if len(gt_snb_angles) > 0:
        print(f"\n📐 SNB ANGLE ANALYSIS:")
        print(f"{'Metric':<20} {'HRNetV2':<15} {'Joint MLP':<15} {'Improvement':<15}")
        print("-" * 70)
        
        hrnet_snb_mae = np.mean(hrnet_snb_errors)
        mlp_snb_mae = np.mean(mlp_snb_errors)
        snb_improvement = (hrnet_snb_mae - mlp_snb_mae) / hrnet_snb_mae * 100
        
        hrnet_snb_std = np.std(hrnet_snb_errors)
        mlp_snb_std = np.std(mlp_snb_errors)
        
        print(f"{'Mean Abs Error (°)':<20} {hrnet_snb_mae:<15.2f} {mlp_snb_mae:<15.2f} {snb_improvement:<15.1f}%")
        print(f"{'Std Dev (°)':<20} {hrnet_snb_std:<15.2f} {mlp_snb_std:<15.2f}")
        print(f"{'Max Error (°)':<20} {np.max(hrnet_snb_errors):<15.2f} {np.max(mlp_snb_errors):<15.2f}")
        print(f"{'Samples Analyzed':<20} {len(gt_snb_angles)}")
        
        # SNB Classification Results
        print(f"\n🏷️  SNB CLASSIFICATION ACCURACY:")
        print(f"{'Method':<15} {'Accuracy':<15} {'Correct/Total':<15}")
        print("-" * 50)
        print(f"{'HRNetV2':<15} {hrnet_snb_accuracy:<15.1f}% {hrnet_snb_correct}/{total_snb_valid}")
        print(f"{'Joint MLP':<15} {mlp_snb_accuracy:<15.1f}% {mlp_snb_correct}/{total_snb_valid}")
        
        snb_accuracy_improvement = mlp_snb_accuracy - hrnet_snb_accuracy
        print(f"{'Improvement':<15} {snb_accuracy_improvement:<15.1f}% {'(+' + str(mlp_snb_correct - hrnet_snb_correct) + ' samples)'}")
        
        # SNB Class distribution
        print(f"\n📋 GROUND TRUTH SNB CLASS DISTRIBUTION:")
        for class_name in snb_classes:
            count = gt_snb_class_counts.get(class_name, 0)
            percentage = (count / total_snb_valid) * 100 if total_snb_valid > 0 else 0
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        
        # SNB Confusion matrix summary
        print(f"\n🔍 SNB CLASSIFICATION ERRORS SUMMARY:")
        print("HRNetV2 SNB Confusion Matrix:")
        print("     Pred: Nor Ret Pro")
        for i, true_class in enumerate(snb_classes):
            row_str = f"GT {true_class[:3]:>3}: "
            for j in range(3):
                row_str += f"{hrnet_snb_confusion[i,j]:>3}"
            print(row_str)
        
        print("\nJoint MLP SNB Confusion Matrix:")
        print("     Pred: Nor Ret Pro")
        for i, true_class in enumerate(snb_classes):
            row_str = f"GT {true_class[:3]:>3}: "
            for j in range(3):
                row_str += f"{mlp_snb_confusion[i,j]:>3}"
            print(row_str)
    else:
        print(f"\n⚠️  No valid SNB angles could be computed (missing landmarks)")
    
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
    
    # Plot best and worst examples
    print(f"\n🖼️  Saving best and worst prediction examples...")
    examples_dir = os.path.join(output_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)
    
    # Plot best examples (HRNet vs MLP)
    for rank, (error, sample_idx) in enumerate(best_examples):
        # HRNet prediction
        hrnet_plot_path = os.path.join(examples_dir, f"best_{rank+1}_hrnet.png")
        plot_landmarks_on_image(
            test_images[sample_idx], 
            hrnet_coords[sample_idx], 
            gt_coords[sample_idx], 
            landmark_names,
            f"Best Example #{rank+1} - HRNetV2",
            hrnet_plot_path
        )
        
        # MLP prediction
        mlp_plot_path = os.path.join(examples_dir, f"best_{rank+1}_mlp.png")
        plot_landmarks_on_image(
            test_images[sample_idx], 
            mlp_coords[sample_idx], 
            gt_coords[sample_idx], 
            landmark_names,
            f"Best Example #{rank+1} - Joint MLP",
            mlp_plot_path
        )
    
    # Plot worst examples (HRNet vs MLP)
    for rank, (error, sample_idx) in enumerate(worst_examples):
        # HRNet prediction
        hrnet_plot_path = os.path.join(examples_dir, f"worst_{rank+1}_hrnet.png")
        plot_landmarks_on_image(
            test_images[sample_idx], 
            hrnet_coords[sample_idx], 
            gt_coords[sample_idx], 
            landmark_names,
            f"Worst Example #{rank+1} - HRNetV2",
            hrnet_plot_path
        )
        
        # MLP prediction
        mlp_plot_path = os.path.join(examples_dir, f"worst_{rank+1}_mlp.png")
        plot_landmarks_on_image(
            test_images[sample_idx], 
            mlp_coords[sample_idx], 
            gt_coords[sample_idx], 
            landmark_names,
            f"Worst Example #{rank+1} - Joint MLP",
            mlp_plot_path
        )
    
    # Save detailed results
    results_summary = {
        'hrnet_overall': hrnet_overall,
        'mlp_overall': mlp_overall,
        'improvement_mre': improvement_mre,
        'improvement_std': improvement_std,
        'improvement_median': improvement_median,
        'total_samples': len(hrnet_predictions),
        'model_type': model_type,
        'anb_analysis': {
            'hrnet_anb_mae': hrnet_anb_mae if len(gt_anb_angles) > 0 else None,
            'mlp_anb_mae': mlp_anb_mae if len(gt_anb_angles) > 0 else None,
            'anb_improvement': anb_improvement if len(gt_anb_angles) > 0 else None,
            'anb_samples': len(gt_anb_angles),
            'classification_accuracy': {
                'hrnet_accuracy': hrnet_accuracy if len(gt_anb_angles) > 0 else None,
                'mlp_accuracy': mlp_accuracy if len(gt_anb_angles) > 0 else None,
                'accuracy_improvement': accuracy_improvement if len(gt_anb_angles) > 0 else None,
                'hrnet_correct': hrnet_correct if len(gt_anb_angles) > 0 else None,
                'mlp_correct': mlp_correct if len(gt_anb_angles) > 0 else None,
                'total_samples': total_valid if len(gt_anb_angles) > 0 else None
            },
            'class_distribution': dict(gt_class_counts) if len(gt_anb_angles) > 0 else None,
            'confusion_matrices': {
                'hrnet': hrnet_confusion.tolist() if len(gt_anb_angles) > 0 else None,
                'mlp': mlp_confusion.tolist() if len(gt_anb_angles) > 0 else None
            }
        },
        'snb_analysis': {
            'hrnet_snb_mae': hrnet_snb_mae if len(gt_snb_angles) > 0 else None,
            'mlp_snb_mae': mlp_snb_mae if len(gt_snb_angles) > 0 else None,
            'snb_improvement': snb_improvement if len(gt_snb_angles) > 0 else None,
            'snb_samples': len(gt_snb_angles),
            'snb_classification_accuracy': {
                'hrnet_accuracy': hrnet_snb_accuracy if len(gt_snb_angles) > 0 else None,
                'mlp_accuracy': mlp_snb_accuracy if len(gt_snb_angles) > 0 else None,
                'accuracy_improvement': snb_accuracy_improvement if len(gt_snb_angles) > 0 else None,
                'hrnet_correct': hrnet_snb_correct if len(gt_snb_angles) > 0 else None,
                'mlp_correct': mlp_snb_correct if len(gt_snb_angles) > 0 else None,
                'total_samples': total_snb_valid if len(gt_snb_angles) > 0 else None
            },
            'snb_class_distribution': dict(gt_snb_class_counts) if len(gt_snb_angles) > 0 else None,
            'snb_confusion_matrices': {
                'hrnet': hrnet_snb_confusion.tolist() if len(gt_snb_angles) > 0 else None,
                'mlp': mlp_snb_confusion.tolist() if len(gt_snb_angles) > 0 else None
            }
        }
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
    fig = plt.figure(figsize=(24, 18))
    
    # Create 3x3 subplot layout
    ax1 = plt.subplot(3, 3, 1)
    ax2 = plt.subplot(3, 3, 2)
    ax3 = plt.subplot(3, 3, 3)
    ax4 = plt.subplot(3, 3, 4)
    ax5 = plt.subplot(3, 3, 5)
    ax6 = plt.subplot(3, 3, 6)
    ax7 = plt.subplot(3, 3, 7)
    ax8 = plt.subplot(3, 3, 8)
    ax9 = plt.subplot(3, 3, 9)
    
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
    
    # ANB angle comparison
    if len(gt_anb_angles) > 0:
        methods_anb = ['HRNetV2', 'Joint MLP']
        anb_maes = [hrnet_anb_mae, mlp_anb_mae]
        anb_stds = [hrnet_anb_std, mlp_anb_std]
        
        ax5.bar(methods_anb, anb_maes, yerr=anb_stds, capsize=5, alpha=0.7, 
                color=['skyblue', 'lightcoral'])
        ax5.set_ylabel('Mean Absolute Error (degrees)')
        ax5.set_title('ANB Angle Error Comparison')
        ax5.grid(True, alpha=0.3)
        
        # Add improvement percentage
        ax5.text(1, mlp_anb_mae + mlp_anb_std + 0.1, 
                 f'{anb_improvement:.1f}% improvement', ha='center', va='bottom', 
                 fontsize=10, color='green' if anb_improvement > 0 else 'red', fontweight='bold')
        
        # ANB angle scatter plot
        ax6.scatter(hrnet_anb_errors, mlp_anb_errors, alpha=0.6, color='purple')
        
        # Add diagonal line
        max_anb_error = max(np.max(hrnet_anb_errors), np.max(mlp_anb_errors))
        ax6.plot([0, max_anb_error], [0, max_anb_error], 'r--', alpha=0.8, 
                 label='No improvement line')
        ax6.set_xlabel('HRNetV2 ANB Error (degrees)')
        ax6.set_ylabel('Joint MLP ANB Error (degrees)')
        ax6.set_title('ANB Angle Error Correlation')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No valid ANB angles\ncould be computed', 
                 ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title('ANB Angle Analysis')
        
        ax6.text(0.5, 0.5, 'ANB angle analysis\nrequires valid\nSella, Nasion, A, B points', 
                 ha='center', va='center', transform=ax6.transAxes, fontsize=10)
        ax6.set_title('ANB Requirements')
    
    # SNB angle comparison
    if len(gt_snb_angles) > 0:
        methods_snb = ['HRNetV2', 'Joint MLP']
        snb_maes = [hrnet_snb_mae, mlp_snb_mae]
        snb_stds = [hrnet_snb_std, mlp_snb_std]
        
        ax7.bar(methods_snb, snb_maes, yerr=snb_stds, capsize=5, alpha=0.7, 
                color=['skyblue', 'lightcoral'])
        ax7.set_ylabel('Mean Absolute Error (degrees)')
        ax7.set_title('SNB Angle Error Comparison')
        ax7.grid(True, alpha=0.3)
        
        # Add improvement percentage
        ax7.text(1, mlp_snb_mae + mlp_snb_std + 0.1, 
                 f'{snb_improvement:.1f}% improvement', ha='center', va='bottom', 
                 fontsize=10, color='green' if snb_improvement > 0 else 'red', fontweight='bold')
        
        # SNB angle scatter plot
        ax8.scatter(hrnet_snb_errors, mlp_snb_errors, alpha=0.6, color='orange')
        
        # Add diagonal line
        max_snb_error = max(np.max(hrnet_snb_errors), np.max(mlp_snb_errors))
        ax8.plot([0, max_snb_error], [0, max_snb_error], 'r--', alpha=0.8, 
                 label='No improvement line')
        ax8.set_xlabel('HRNetV2 SNB Error (degrees)')
        ax8.set_ylabel('Joint MLP SNB Error (degrees)')
        ax8.set_title('SNB Angle Error Correlation')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # Classification accuracy comparison
        if len(gt_anb_angles) > 0:
            classification_methods = ['ANB\nHRNet', 'ANB\nMLP', 'SNB\nHRNet', 'SNB\nMLP']
            classification_accuracies = [hrnet_accuracy, mlp_accuracy, hrnet_snb_accuracy, mlp_snb_accuracy]
            colors = ['lightblue', 'lightcoral', 'lightgreen', 'orange']
            
            bars = ax9.bar(classification_methods, classification_accuracies, color=colors, alpha=0.7)
            ax9.set_ylabel('Classification Accuracy (%)')
            ax9.set_title('ANB vs SNB Classification Accuracy')
            ax9.grid(True, alpha=0.3)
            
            # Add percentage labels on bars
            for bar, acc in zip(bars, classification_accuracies):
                height = bar.get_height()
                ax9.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
        else:
            ax9.text(0.5, 0.5, 'SNB classification\naccuracy only', 
                     ha='center', va='center', transform=ax9.transAxes, fontsize=12)
            ax9.set_title('SNB Classification Only')
    else:
        ax7.text(0.5, 0.5, 'No valid SNB angles\ncould be computed', 
                 ha='center', va='center', transform=ax7.transAxes, fontsize=12)
        ax7.set_title('SNB Angle Analysis')
        
        ax8.text(0.5, 0.5, 'SNB angle analysis\nrequires valid\nSella, Nasion, B points', 
                 ha='center', va='center', transform=ax8.transAxes, fontsize=10)
        ax8.set_title('SNB Requirements')
        
        ax9.text(0.5, 0.5, 'No classification\nanalysis available', 
                 ha='center', va='center', transform=ax9.transAxes, fontsize=10)
        ax9.set_title('Classification Summary')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "joint_mlp_evaluation_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n💾 Results saved to: {output_dir}")
    print(f"   - Per-landmark comparison: per_landmark_comparison.csv")
    print(f"   - Comprehensive visualization: joint_mlp_evaluation_results.png")
    print(f"   - Best/worst examples: examples/ directory")
    print(f"     • Best 3 examples: best_1_hrnet.png, best_1_mlp.png, etc.")
    print(f"     • Worst 3 examples: worst_1_hrnet.png, worst_1_mlp.png, etc.")
    
    print(f"\n🎉 Joint MLP evaluation completed!")
    print(f"📈 Overall improvement: {improvement_mre:.1f}% reduction in MRE")
    print(f"🎯 Joint model captures cross-correlations between landmarks")
    print(f"🔧 Evaluated using: {model_type} joint MLP model")
    
    if model_type == "best":
        print(f"✅ Using best MLP model synchronized with best HRNetV2 checkpoint")
    elif "best_epoch" in model_type:
        print(f"✅ Using best epoch MLP model synchronized with HRNetV2 performance")
    
    if len(gt_anb_angles) > 0:
        print(f"📐 ANB angle improvement: {anb_improvement:.1f}% reduction in MAE")
        print(f"   ({len(gt_anb_angles)} samples with valid ANB calculations)")
        print(f"🏷️  ANB classification improvement: {accuracy_improvement:.1f}% accuracy gain")
    else:
        print(f"⚠️  ANB angle analysis not available (missing required landmarks)")
    
    if len(gt_snb_angles) > 0:
        print(f"📐 SNB angle improvement: {snb_improvement:.1f}% reduction in MAE")
        print(f"   ({len(gt_snb_angles)} samples with valid SNB calculations)")
        print(f"🏷️  SNB classification improvement: {snb_accuracy_improvement:.1f}% accuracy gain")
    else:
        print(f"⚠️  SNB angle analysis not available (missing required landmarks)")
    
    print(f"\n🖼️  Visual analysis available:")
    print(f"   - Compare HRNet vs MLP predictions on best/worst cases")
    print(f"   - Green circles = Ground truth landmarks")
    print(f"   - Red crosses = Model predictions")
    print(f"   - Numbers indicate landmark indices (see cephalometric_dataset_info.py)")
    
    if model_type == "best":
        print("\n💡 Note: Using best MLP model synchronized with best HRNetV2 checkpoint.")
        print("     This provides the most reliable evaluation of concurrent training benefits.")
    elif "best_epoch" in model_type:
        print("\n💡 Note: Using best epoch MLP model. Results reflect peak performance.")
    elif model_type == "final":
        print("\n💡 Note: Using final MLP model from last training epoch.")
        print("     Best model may have been from an earlier epoch.")
    elif model_type == "latest":
        print("\n💡 Note: Training is likely still in progress. Final results may differ.")
    elif "epoch_" in model_type:
        print("\n💡 Note: Using intermediate checkpoint. Final results may differ.")

if __name__ == "__main__":
    main() 