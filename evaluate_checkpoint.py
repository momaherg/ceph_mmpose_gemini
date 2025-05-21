#!/usr/bin/env python
import os
import os.path as osp
import argparse
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint
from mmengine.registry import init_default_scope, MODELS
from mmengine.config.config import ConfigDict
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import your custom modules to ensure they are registered
init_default_scope('mmpose')
import custom_cephalometric_dataset  # Registers CustomCephalometricDataset
import custom_transforms              # Registers LoadImageNumpy
import cephalometric_dataset_info    # Makes dataset_info available for the config file

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a checkpoint using MRE metric')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--out-dir', default='evaluation_results', help='directory to save the results')
    parser.add_argument('--show', action='store_true', help='show the visualization results')
    parser.add_argument('--show-dir', help='directory where visualized results will be saved')
    parser.add_argument('--test-json', default=None, help='Path to test data JSON file to override the one in config')
    args = parser.parse_args()
    return args

def mean_radial_error(pred_keypoints, gt_keypoints, keypoint_visibility=None):
    """
    Calculate the Mean Radial Error (MRE) between predicted and ground truth keypoints.
    
    Args:
        pred_keypoints (np.ndarray): Predicted keypoints with shape (N, K, 2)
        gt_keypoints (np.ndarray): Ground-truth keypoints with shape (N, K, 2)
        keypoint_visibility (np.ndarray, optional): Visibility flag for keypoints with 
                                                    shape (N, K)
    
    Returns:
        tuple: (MRE per keypoint, overall MRE)
    """
    # Convert to numpy if tensors
    if isinstance(pred_keypoints, torch.Tensor):
        pred_keypoints = pred_keypoints.detach().cpu().numpy()
    if isinstance(gt_keypoints, torch.Tensor):
        gt_keypoints = gt_keypoints.detach().cpu().numpy()
    if keypoint_visibility is not None and isinstance(keypoint_visibility, torch.Tensor):
        keypoint_visibility = keypoint_visibility.detach().cpu().numpy()
    
    N, K, _ = pred_keypoints.shape  # N: batch size, K: number of keypoints
    
    # Calculate Euclidean distance for each keypoint
    distances = np.sqrt(np.sum((pred_keypoints - gt_keypoints) ** 2, axis=2))  # Shape: (N, K)
    
    # Apply visibility masks if available
    if keypoint_visibility is not None:
        # Consider only visible keypoints (visibility > 0)
        mask = keypoint_visibility > 0
        distances = np.where(mask, distances, np.nan)
    
    # Compute MRE per keypoint (across all samples)
    mre_per_keypoint = np.nanmean(distances, axis=0)  # Shape: (K,)
    
    # Compute overall MRE
    overall_mre = np.nanmean(distances)
    
    return mre_per_keypoint, overall_mre

def visualize_results(image, gt_keypoints, pred_keypoints, keypoint_visibility=None, 
                      keypoint_names=None, save_path=None):
    """
    Visualize the ground truth and predicted keypoints on the image.
    
    Args:
        image (np.ndarray): Original image with shape (H, W, 3)
        gt_keypoints (np.ndarray): Ground-truth keypoints with shape (K, 2)
        pred_keypoints (np.ndarray): Predicted keypoints with shape (K, 2)
        keypoint_visibility (np.ndarray, optional): Visibility flag for keypoints with shape (K,)
        keypoint_names (list, optional): List of keypoint names
        save_path (str, optional): Path to save the visualization
    """
    plt.figure(figsize=(12, 10))
    plt.imshow(image)
    
    K = gt_keypoints.shape[0]
    
    # Default visibility - all keypoints visible
    if keypoint_visibility is None:
        keypoint_visibility = np.ones(K)
    
    # Plot ground truth keypoints (green)
    for i in range(K):
        if keypoint_visibility[i] > 0:
            plt.plot(gt_keypoints[i, 0], gt_keypoints[i, 1], 'go', markersize=5)
            if keypoint_names is not None:
                plt.text(gt_keypoints[i, 0] + 3, gt_keypoints[i, 1] + 3, keypoint_names[i], 
                         color='green', fontsize=8)
    
    # Plot predicted keypoints (red)
    for i in range(K):
        if keypoint_visibility[i] > 0:
            plt.plot(pred_keypoints[i, 0], pred_keypoints[i, 1], 'ro', markersize=5)
            if keypoint_names is not None:
                plt.text(pred_keypoints[i, 0] - 3, pred_keypoints[i, 1] - 3, keypoint_names[i], 
                         color='red', fontsize=8)
    
    plt.title('Green: Ground Truth, Red: Prediction')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

def evaluate_checkpoint(config_path, checkpoint_path, test_json=None, out_dir='evaluation_results', 
                        show=False, show_dir=None):
    """
    Evaluate a checkpoint using MRE metric.
    
    Args:
        config_path (str): Path to the config file
        checkpoint_path (str): Path to the checkpoint file
        test_json (str, optional): Path to test data JSON to override the one in config
        out_dir (str): Directory to save the results
        show (bool): Whether to show the visualization results
        show_dir (str, optional): Directory where visualized results will be saved
    
    Returns:
        tuple: (MRE per keypoint, overall MRE)
    """
    # Load config
    cfg = Config.fromfile(config_path)
    
    # Override test_json if provided
    if test_json:
        if 'train_dataloader' in cfg and 'dataset' in cfg.train_dataloader:
            # Update test_json in config
            cfg.train_dataloader.dataset.ann_file = test_json
        
    # Create out_dir and show_dir if needed
    os.makedirs(out_dir, exist_ok=True)
    if show_dir:
        os.makedirs(show_dir, exist_ok=True)
    
    # Build the model
    model = MODELS.build(cfg.model)
    
    # Add ConfigDict to safe globals to avoid PyTorch 2.6+ security errors
    try:
        import torch.serialization
        # First option: Add ConfigDict to safe globals (preferred for security)
        torch.serialization.add_safe_globals([ConfigDict])
        # Load checkpoint
        load_checkpoint(model, checkpoint_path, map_location='cpu')
    except (ImportError, AttributeError):
        # Fallback for older PyTorch versions or if add_safe_globals doesn't exist
        print("Warning: Using torch.load with weights_only=False for compatibility.")
        from mmengine.runner.checkpoint import _load_checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        _load_checkpoint(checkpoint, model, None, strict=True)
    
    model.eval()
    
    # Convert to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Prepare dataset for testing
    from mmengine.dataset import Compose
    
    # Define a simplified test pipeline for inference (no augmentations)
    test_pipeline = [
        dict(type='LoadImageNumpy'),
        dict(type='GetBBoxCenterScale'),
        dict(type='TopdownAffine', input_size=cfg.model.head.decoder.input_size),
        dict(type='PackPoseInputs')
    ]
    
    # Build dataset
    dataset_cfg = cfg.train_dataloader.dataset.copy()
    dataset_cfg.pipeline = test_pipeline
    
    from custom_cephalometric_dataset import CustomCephalometricDataset
    
    # Use the test_json file path if provided, otherwise use the one in config
    if test_json:
        dataset_cfg.ann_file = test_json
    
    dataset = CustomCephalometricDataset(**dataset_cfg)
    
    # Collect all results
    all_pred_keypoints = []
    all_gt_keypoints = []
    all_keypoint_visibility = []
    all_images = []
    
    # Track time
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    inference_times = []
    
    # Keypoint names from dataset metainfo for visualization
    keypoint_names = [info['name'] for _, info in dataset.metainfo['keypoint_info'].items()]
    
    # Process each image
    print(f"Running inference on {len(dataset)} images...")
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        img = data['inputs'].unsqueeze(0).to(device)  # Add batch dimension
        data_samples = data['data_samples'].to(device)
        
        # Record inference time
        start_time.record()
        with torch.no_grad():
            outputs = model.predict(img, data_samples)
        end_time.record()
        torch.cuda.synchronize()
        inference_times.append(start_time.elapsed_time(end_time) / 1000)  # Convert to seconds
        
        # Extract predictions and ground truth
        if isinstance(outputs, list):
            output = outputs[0]
        else:
            output = outputs
        
        pred_instance = output.pred_instances
        gt_instance = data_samples.gt_instances
        
        # Store original image
        orig_img = data_samples.metainfo.get('ori_img', None)
        if orig_img is None:
            # Try to retrieve from dataset if not in data_samples
            orig_img = dataset._load_data_list()[i]['img']
        
        # Convert keypoints back to original image space if needed
        pred_keypoints = pred_instance.keypoints
        gt_keypoints = gt_instance.keypoints
        keypoint_visibility = gt_instance.keypoints_visible
        
        all_pred_keypoints.append(pred_keypoints)
        all_gt_keypoints.append(gt_keypoints)
        all_keypoint_visibility.append(keypoint_visibility)
        all_images.append(orig_img)
        
        # Visualize if requested
        if show or show_dir:
            # Convert to numpy
            pred_kpts_np = pred_keypoints.cpu().numpy()[0]  # Remove batch dim
            gt_kpts_np = gt_keypoints.cpu().numpy()[0]
            vis_np = keypoint_visibility.cpu().numpy()[0]
            img_np = orig_img.cpu().numpy() if isinstance(orig_img, torch.Tensor) else orig_img
            
            if show_dir:
                save_path = osp.join(show_dir, f'vis_{i:04d}.png')
            else:
                save_path = None
                
            visualize_results(img_np, gt_kpts_np, pred_kpts_np, vis_np, 
                              keypoint_names, save_path)
    
    # Stack all results
    all_pred_keypoints = torch.cat(all_pred_keypoints, dim=0).cpu().numpy()
    all_gt_keypoints = torch.cat(all_gt_keypoints, dim=0).cpu().numpy()
    all_keypoint_visibility = torch.cat(all_keypoint_visibility, dim=0).cpu().numpy()
    
    # Calculate MRE
    mre_per_keypoint, overall_mre = mean_radial_error(
        all_pred_keypoints, all_gt_keypoints, all_keypoint_visibility)
    
    # Calculate average inference time
    avg_time = np.mean(inference_times)
    fps = 1.0 / avg_time
    
    # Print results
    print(f"\nMRE Results:")
    print(f"Overall MRE: {overall_mre:.4f} pixels")
    print(f"Average Inference Time: {avg_time:.4f} seconds")
    print(f"FPS: {fps:.2f}")
    
    print("\nMRE per Keypoint:")
    for i, name in enumerate(keypoint_names):
        print(f"{name}: {mre_per_keypoint[i]:.4f} pixels")
    
    # Save results to file
    result_file = osp.join(out_dir, 'mre_results.txt')
    with open(result_file, 'w') as f:
        f.write(f"Overall MRE: {overall_mre:.4f} pixels\n")
        f.write(f"Average Inference Time: {avg_time:.4f} seconds\n")
        f.write(f"FPS: {fps:.2f}\n\n")
        f.write("MRE per Keypoint:\n")
        for i, name in enumerate(keypoint_names):
            f.write(f"{name}: {mre_per_keypoint[i]:.4f} pixels\n")
    
    print(f"Results saved to {result_file}")
    
    # Plot MRE per keypoint
    plt.figure(figsize=(12, 8))
    plt.bar(keypoint_names, mre_per_keypoint)
    plt.xticks(rotation=90)
    plt.ylabel('MRE (pixels)')
    plt.title('Mean Radial Error per Keypoint')
    plt.tight_layout()
    plt.savefig(osp.join(out_dir, 'mre_per_keypoint.png'))
    
    return mre_per_keypoint, overall_mre

if __name__ == '__main__':
    args = parse_args()
    evaluate_checkpoint(
        args.config, 
        args.checkpoint, 
        test_json=args.test_json,
        out_dir=args.out_dir, 
        show=args.show, 
        show_dir=args.show_dir
    ) 