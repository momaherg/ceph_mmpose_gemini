import os
import numpy as np
import pandas as pd
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.registry import init_default_scope
from mmpose.structures import merge_data_samples, split_instances
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Import your custom modules
from custom_cephalometric_dataset import CustomCephalometricDataset
import custom_transforms
import cephalometric_dataset_info

def calculate_mre(pred_coords, gt_coords):
    """
    Calculate Mean Radial Error between predicted and ground truth landmarks.
    
    Args:
        pred_coords (np.ndarray): Predicted coordinates of shape (N, K, 2) or (K, 2)
        gt_coords (np.ndarray): Ground truth coordinates of shape (N, K, 2) or (K, 2)
        
    Returns:
        float: Mean Radial Error in pixels
        np.ndarray: Per-landmark MRE values
    """
    # Ensure arrays have the same shape
    if pred_coords.ndim == 2 and gt_coords.ndim == 2:
        # Single instance case (K, 2)
        radial_errors = np.sqrt(np.sum((pred_coords - gt_coords) ** 2, axis=1))
    elif pred_coords.ndim == 3 and gt_coords.ndim == 3:
        # Batch case (N, K, 2)
        radial_errors = np.sqrt(np.sum((pred_coords - gt_coords) ** 2, axis=2))
        # Average across instances
        radial_errors = np.mean(radial_errors, axis=0)
    else:
        raise ValueError(f"Incompatible shapes: pred_coords {pred_coords.shape}, gt_coords {gt_coords.shape}")
    
    # Per-landmark MRE
    per_landmark_mre = radial_errors
    
    # Overall MRE
    mre = np.mean(radial_errors)
    
    return mre, per_landmark_mre

def visualize_predictions(image, gt_landmarks, pred_landmarks, landmark_names, output_path=None):
    """
    Visualize the ground truth and predicted landmarks on the image.
    
    Args:
        image (np.ndarray): Image to visualize on
        gt_landmarks (np.ndarray): Ground truth landmarks of shape (K, 2)
        pred_landmarks (np.ndarray): Predicted landmarks of shape (K, 2)
        landmark_names (list): List of landmark names
        output_path (str, optional): Path to save the visualization
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # Plot ground truth landmarks
    for i, (x, y) in enumerate(gt_landmarks):
        plt.plot(x, y, 'go', markersize=6)
        plt.gca().add_patch(Circle((x, y), radius=3, color='green', fill=False, linewidth=1.5))
        plt.text(x+5, y+5, landmark_names[i], color='green', fontsize=8)
    
    # Plot predicted landmarks
    for i, (x, y) in enumerate(pred_landmarks):
        plt.plot(x, y, 'ro', markersize=6)
        plt.gca().add_patch(Circle((x, y), radius=3, color='red', fill=False, linewidth=1.5))
    
    plt.title('Landmark Predictions\nGreen: Ground Truth, Red: Prediction')
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        plt.close()
    else:
        plt.show()

def evaluate_checkpoint(
    config_path,
    checkpoint_path,
    test_data_path,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    visualize=False,
    visualize_dir='visualization',
    num_samples_to_visualize=5
):
    """
    Evaluate a trained MMPose model on a test dataset using Mean Radial Error (MRE).
    
    Args:
        config_path (str): Path to the model config file
        checkpoint_path (str): Path to the model checkpoint
        test_data_path (str): Path to the test data JSON file
        device (str): Device to run inference on ('cuda' or 'cpu')
        visualize (bool): Whether to visualize predictions
        visualize_dir (str): Directory to save visualizations
        num_samples_to_visualize (int): Number of samples to visualize
        
    Returns:
        dict: Dictionary containing evaluation results
    """
    # Initialize scope for MMPose
    init_default_scope('mmpose')
    
    # Create visualization directory if needed
    if visualize and not os.path.exists(visualize_dir):
        os.makedirs(visualize_dir)
    
    # Load config
    cfg = Config.fromfile(config_path)
    
    # Update config for inference
    cfg.model.test_cfg = dict(
        flip_test=False,
        shift_heatmap=False,
    )
    
    # Build the model
    from mmpose.registry import MODELS
    model = MODELS.build(cfg.model)
    
    # Load checkpoint
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model = model.to(device)
    model.eval()
    
    # Prepare a minimal inference pipeline
    from mmengine.dataset import Compose
    test_pipeline = Compose([
        dict(type='LoadImageNumpy'),
        dict(type='GetBBoxCenterScale'),
        dict(type='TopdownAffine', input_size=cfg.model.head.decoder.input_size),
        dict(type='PackPoseInputs'),
    ])
    
    # Load test dataset
    print(f"Loading test data from {test_data_path}")
    try:
        test_df = pd.read_json(test_data_path)
        print(f"Test data loaded. Shape: {test_df.shape}")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return {"error": str(e)}
    
    # Create dataset
    from cephalometric_dataset_info import landmark_names_in_order
    test_dataset = CustomCephalometricDataset(
        data_df=test_df,
        pipeline=test_pipeline,
        test_mode=True
    )
    
    print(f"Test dataset prepared with {len(test_dataset)} samples")
    
    # Perform inference
    all_gt_coords = []
    all_pred_coords = []
    
    print("Running inference...")
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset))):
            # Get data
            data = test_dataset[idx]
            data_samples = data['data_samples'].to(device)
            inputs = torch.stack([data['inputs']]).to(device)
            
            # Run model
            predictions = model.test_step((inputs, [data_samples]))
            
            # Get predicted keypoints
            pred_instances = predictions[0].pred_instances
            pred_coords = pred_instances.keypoints.cpu().numpy()
            
            # Get ground truth keypoints
            gt_coords = data_samples.gt_instances.keypoints.cpu().numpy()
            
            all_pred_coords.append(pred_coords)
            all_gt_coords.append(gt_coords)
            
            # Visualize if needed
            if visualize and idx < num_samples_to_visualize:
                # Get the original image (without transforms)
                raw_idx = idx % len(test_dataset)
                patient_id = test_dataset.data_list[raw_idx]['id']
                
                # Get original image from dataset
                original_img = test_dataset.data_list[raw_idx]['img']
                
                # Reshape to remove batch dimension if needed
                gt_vis = gt_coords[0] if gt_coords.shape[0] == 1 else gt_coords
                pred_vis = pred_coords[0] if pred_coords.shape[0] == 1 else pred_coords
                
                output_path = os.path.join(visualize_dir, f'sample_{raw_idx}_{patient_id}.png')
                visualize_predictions(
                    original_img, 
                    gt_vis, 
                    pred_vis, 
                    landmark_names_in_order,
                    output_path
                )
    
    # Calculate MRE
    all_gt_coords = np.concatenate(all_gt_coords, axis=0)
    all_pred_coords = np.concatenate(all_pred_coords, axis=0)
    
    # Reshape if necessary
    if all_gt_coords.shape[0] == 1:
        all_gt_coords = all_gt_coords[0]
        all_pred_coords = all_pred_coords[0]
    
    overall_mre, per_landmark_mre = calculate_mre(all_pred_coords, all_gt_coords)
    
    # Create results dictionary
    results = {
        "overall_mre": float(overall_mre),
        "per_landmark_mre": per_landmark_mre.tolist() if isinstance(per_landmark_mre, np.ndarray) else per_landmark_mre,
        "num_samples": len(test_dataset),
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "test_data_path": test_data_path
    }
    
    # Print results
    print(f"\nEvaluation Results:")
    print(f"Overall MRE: {overall_mre:.4f} pixels")
    print("\nPer-landmark MRE:")
    for i, (name, error) in enumerate(zip(landmark_names_in_order, per_landmark_mre)):
        print(f"{name}: {error:.4f} pixels")
    
    # Plot per-landmark MRE
    plt.figure(figsize=(12, 6))
    plt.bar(landmark_names_in_order, per_landmark_mre)
    plt.xticks(rotation=90)
    plt.title('Mean Radial Error per Landmark')
    plt.ylabel('MRE (pixels)')
    plt.tight_layout()
    
    if visualize:
        plt.savefig(os.path.join(visualize_dir, 'per_landmark_mre.png'))
        plt.close()
    
    return results

# Example usage
if __name__ == "__main__":
    # Example parameters (modify as needed for your specific case)
    CONFIG_PATH = "configs/hrnetv2/hrnetv2_w18_cephalometric_224x224.py"
    CHECKPOINT_PATH = "work_dirs/hrnetv2_w18_cephalometric_experiment/epoch_60.pth"  # Change to your best checkpoint
    TEST_DATA_PATH = "/content/drive/MyDrive/Lala's Masters/test_data_pure_old_numpy.json"  # Path to your test data
    
    # Run evaluation
    results = evaluate_checkpoint(
        config_path=CONFIG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        test_data_path=TEST_DATA_PATH,
        visualize=True,
        num_samples_to_visualize=5
    )
    
    # You can save the results to a file if needed
    import json
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4) 