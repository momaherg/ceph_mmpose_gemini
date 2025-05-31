#!/usr/bin/env python3
"""
Visualization script for cephalometric landmark predictions.
"""

import os
import torch
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.apis import init_model, inference_topdown
import glob

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

def visualize_landmarks(image, pred_keypoints, gt_keypoints, landmark_names, save_path=None):
    """Visualize predicted and ground truth landmarks on an image."""
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # Plot ground truth landmarks in green
    for i, (x, y) in enumerate(gt_keypoints):
        if x > 0 and y > 0:  # Valid landmark
            plt.scatter(x, y, color='green', s=50, marker='o', alpha=0.8)
            plt.text(x+2, y+2, str(i), color='green', fontsize=8)
    
    # Plot predicted landmarks in red
    for i, (x, y) in enumerate(pred_keypoints):
        if x > 0 and y > 0:  # Valid landmark
            plt.scatter(x, y, color='red', s=30, marker='x', alpha=0.8)
            
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Ground Truth'),
        Patch(facecolor='red', label='Predictions')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title('Cephalometric Landmark Detection')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    """Main visualization function."""
    
    print("="*80)
    print("CEPHALOMETRIC PREDICTIONS VISUALIZATION")
    print("="*80)
    
    # Initialize MMPose scope
    init_default_scope('mmpose')
    
    # Import custom modules
    import custom_cephalometric_dataset
    import custom_transforms
    import cephalometric_dataset_info
    
    # Configuration
    config_path = "Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py"
    work_dir = "work_dirs/hrnetv2_w18_cephalometric_finetune_experiment"
    
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
    
    # Initialize model
    model = init_model(config_path, checkpoint_path, device='cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Model loaded successfully")
    
    # Load test data
    data_file_path = "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json"
    main_df = pd.read_json(data_file_path)
    test_df = main_df[main_df['set'] == 'test'].reset_index(drop=True)
    
    if test_df.empty:
        print("Test set empty, using validation set")
        test_df = main_df[main_df['set'] == 'dev'].reset_index(drop=True)
    
    # Get landmark names
    landmark_names = cephalometric_dataset_info.landmark_names_in_order
    landmark_cols = cephalometric_dataset_info.original_landmark_cols
    
    # Create output directory
    output_dir = os.path.join(work_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize predictions for first 5 test images
    num_samples = min(5, len(test_df))
    print(f"\nVisualizing {num_samples} samples...")
    
    for idx in range(num_samples):
        row = test_df.iloc[idx]
        
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
        
        # Prepare data for inference
        data_sample = {
            'bbox': np.array([[0, 0, 224, 224]], dtype=np.float32),
            'bbox_scores': np.array([1.0], dtype=np.float32)
        }
        
        # Run inference
        results = inference_topdown(model, img_array, bboxes=data_sample['bbox'], bbox_format='xyxy')
        
        # Extract predictions
        if results and len(results) > 0:
            pred_keypoints = results[0].pred_instances.keypoints[0]  # Shape: (19, 2)
            
            # Visualize
            save_path = os.path.join(output_dir, f"prediction_{idx+1}.png")
            visualize_landmarks(img_array, pred_keypoints, gt_keypoints, landmark_names, save_path)
            
            # Calculate per-landmark errors
            errors = np.sqrt(np.sum((pred_keypoints - gt_keypoints)**2, axis=1))
            valid_mask = (gt_keypoints[:, 0] > 0) & (gt_keypoints[:, 1] > 0)
            
            print(f"\nSample {idx+1} - Patient ID: {row.get('patient_id', 'Unknown')}")
            print("Per-landmark errors (pixels):")
            for i, (name, error, valid) in enumerate(zip(landmark_names, errors, valid_mask)):
                if valid:
                    print(f"  {name}: {error:.2f}")
    
    print(f"\nVisualizations saved to: {output_dir}")

if __name__ == "__main__":
    main() 