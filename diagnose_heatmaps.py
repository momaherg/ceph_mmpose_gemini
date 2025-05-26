import numpy as np
import pandas as pd
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.apis import init_model
from mmpose.models.losses import KeypointMSELoss
from mmpose.codecs import MSRAHeatmap # Or your specific codec
import os
import os.path as osp
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Apply PyTorch safe loading fix (if needed, already in other scripts)

def diagnose_heatmap_and_loss(config_path: str,
                              checkpoint_path: str, # Can be an early epoch checkpoint
                              data_root: str,
                              ann_file: str = 'train_data_pure_old_numpy.json',
                              sample_id_to_debug: Optional[str] = None):
    """
    Diagnose heatmap generation and loss calculation for a specific sample.
    """
    
    print("="*80)
    print("HEATMAP & LOSS DIAGNOSIS")
    print("="*80)
    
    init_default_scope('mmpose')
    
    try:
        import custom_cephalometric_dataset
        import custom_transforms
        import cephalometric_dataset_info
        from cephalometric_dataset_info import landmark_names_in_order, original_landmark_cols
        print("✓ Custom modules imported successfully")
    except ImportError as e:
        print(f"✗ Custom modules import failed: {e}")
        return

    cfg = Config.fromfile(config_path)
    model = init_model(config_path, checkpoint_path, device='cpu') # Use CPU for easier debugging
    model.eval() 
    print(f"✓ Model loaded from: {checkpoint_path}")

    # Load the specific sample from the dataset
    df = pd.read_json(osp.join(data_root, ann_file))
    if sample_id_to_debug:
        sample_row = df[df['patient_id'] == sample_id_to_debug].iloc[0]
    else:
        sample_row = df[df['set'] == 'train'].iloc[0] # Default to first training sample
        sample_id_to_debug = sample_row['patient_id']
    print(f"✓ Debugging Sample ID: {sample_id_to_debug}")

    # 1. PREPARE DATA FOR A SINGLE SAMPLE (SIMULATE DATALOADER OUTPUT)
    print("\n1. DATA PREPARATION & PIPELINE")
    print("-" * 50)
    
    # Get raw image and keypoints
    img_array = sample_row['Image']
    raw_img_np = np.array(img_array, dtype=np.uint8).reshape((224, 224, 3))
    
    raw_keypoints = np.zeros((1, 19, 2), dtype=np.float32) # Shape (N, K, 2)
    raw_keypoints_visible = np.ones((1, 19), dtype=np.int32) * 2 # Shape (N, K)
    for i, kp_name in enumerate(landmark_names_in_order):
        x_col, y_col = original_landmark_cols[i*2], original_landmark_cols[i*2+1]
        if x_col in sample_row and y_col in sample_row and pd.notna(sample_row[x_col]) and pd.notna(sample_row[y_col]):
            raw_keypoints[0, i, 0], raw_keypoints[0, i, 1] = sample_row[x_col], sample_row[y_col]
        else:
            raw_keypoints[0, i, :] = 0
            raw_keypoints_visible[0, i] = 0
            
    # Manually create the data_info dict that goes into the pipeline
    data_info = {
        'img': raw_img_np,
        'img_path': str(sample_row.get('patient_id')),
        'img_id': str(sample_row.get('patient_id')),
        'bbox': np.array([[0, 0, 224, 224]], dtype=np.float32), # (N, 4)
        'keypoints': raw_keypoints,
        'keypoints_visible': raw_keypoints_visible,
        'id': str(sample_row.get('patient_id')),
        'ori_shape': (224, 224),
        'img_shape': (224, 224),
        'patient_text_id': sample_row.get('patient', ''),
        'set': sample_row.get('set', 'train'),
        'class': sample_row.get('class', None)
    }
    
    # Apply the training pipeline transformations
    train_pipeline_cfg = cfg.train_dataloader.dataset.pipeline
    from mmengine.dataset.base_dataset import Compose # MMPose uses this for pipelines
    pipeline = Compose(train_pipeline_cfg)
    processed_data = pipeline(data_info)
    
    inputs_tensor = processed_data['inputs'].unsqueeze(0) # Add batch dim for model
    data_sample = processed_data['data_samples']
    target_heatmaps = data_sample.heatmaps
    target_weights = data_sample.keypoint_weights
    
    print(f"  Processed Inputs Shape: {inputs_tensor.shape}")
    print(f"  Target Heatmaps Shape: {target_heatmaps.shape}")
    print(f"  Target Weights Shape: {target_weights.shape}")
    print(f"  Target Heatmaps Min/Max: {target_heatmaps.min():.3f}/{target_heatmaps.max():.3f}")

    # Visualize one channel of target heatmap
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(raw_img_np)
    plt.title(f"Raw Image (ID: {sample_id_to_debug})")
    selected_kp_idx = 0 # e.g., Sella
    for k_idx_plot in range(min(3, target_heatmaps.shape[0])): # Plot first 3 valid heatmaps
        if target_weights[k_idx_plot] > 0:
             selected_kp_idx = k_idx_plot
             break
    plt.subplot(1, 2, 2)
    plt.imshow(target_heatmaps[selected_kp_idx].numpy(), cmap='viridis')
    plt.title(f"Target Heatmap for '{landmark_names_in_order[selected_kp_idx]}' (Channel {selected_kp_idx})")
    plt.colorbar()
    plt.savefig(f"debug_target_heatmap_sample_{sample_id_to_debug}.png")
    print(f"✓ Saved target heatmap visualization to debug_target_heatmap_sample_{sample_id_to_debug}.png")
    plt.close()

    # 2. FORWARD PASS & PREDICTED HEATMAPS
    print("\n2. MODEL FORWARD PASS")
    print("-" * 50)
    
    # Get predicted heatmaps from the model's head
    # This requires going through backbone and neck first
    features = model.backbone(inputs_tensor)
    if hasattr(model, 'neck') and model.neck is not None:
        features = model.neck(features)
    predicted_heatmaps = model.head.forward(features) # Get raw heatmap output from head
    
    print(f"  Predicted Heatmaps Shape: {predicted_heatmaps.shape}")
    print(f"  Predicted Heatmaps Min/Max: {predicted_heatmaps.min():.3f}/{predicted_heatmaps.max():.3f}")

    # Visualize one channel of predicted heatmap
    plt.figure(figsize=(5, 5))
    plt.imshow(predicted_heatmaps[0, selected_kp_idx].detach().numpy(), cmap='viridis') # batch 0, channel for selected_kp_idx
    plt.title(f"Predicted Heatmap for '{landmark_names_in_order[selected_kp_idx]}'")
    plt.colorbar()
    plt.savefig(f"debug_predicted_heatmap_sample_{sample_id_to_debug}.png")
    print(f"✓ Saved predicted heatmap visualization to debug_predicted_heatmap_sample_{sample_id_to_debug}.png")
    plt.close()

    # 3. LOSS CALCULATION
    print("\n3. LOSS CALCULATION")
    print("-" * 50)
    
    loss_func_cfg = cfg.model.head.loss
    loss_func = KeypointMSELoss(use_target_weight=loss_func_cfg.get('use_target_weight', True))
    
    # Loss function expects (N, K, H, W) for pred and target
    loss = loss_func(predicted_heatmaps, target_heatmaps.unsqueeze(0), target_weights.unsqueeze(0))
    print(f"  Calculated Loss: {loss.item():.6f}")

    if loss.item() > 1.0: # Arbitrary high threshold
        print("  ⚠️  WARNING: Loss is very high for this sample!")
    elif loss.item() < 1e-5:
        print("  ⚠️  WARNING: Loss is very low (close to zero), model might not be learning or target is empty.")

    print("="*80)
    print("HEATMAP & LOSS DIAGNOSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    config_path = "/content/ceph_mmpose_gemini/configs/hrnetv2/hrnetv2_w18_cephalometric_224x224_FIXED_V2.py"
    # Use an early epoch checkpoint to see initial learning state
    checkpoint_path = "/content/ceph_mmpose_gemini/work_dirs/hrnetv2_w18_cephalometric_experiment_FIXED_V2/epoch_2.pth" 
    data_root = "/content/drive/MyDrive/Lala's Masters/"
    
    # You can specify a patient_id to debug a specific problematic sample
    diagnose_heatmap_and_loss(config_path, checkpoint_path, data_root, sample_id_to_debug='1019') # Example patient ID 