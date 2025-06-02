#!/usr/bin/env python3
"""
Dataset for training MLP refinement network.
This dataset loads HRNetV2 predictions and ground truth landmarks for training the refinement network.
"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import warnings
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.apis import init_model, inference_topdown
import cephalometric_dataset_info

warnings.filterwarnings('ignore')

# Apply PyTorch safe loading fix
import functools
_original_torch_load = torch.load
def safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = safe_torch_load


class HRNetPredictionExtractor:
    """Extract predictions from trained HRNetV2 model."""
    
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda:0'):
        self.device = device
        self.original_device = device  # Remember the original device preference
        try:
            self.model = init_model(config_path, checkpoint_path, device=device)
            print(f"HRNetV2 model loaded from {checkpoint_path} on {device}")
        except Exception as e:
            print(f"Error loading HRNetV2 model on {device}: {e}")
            # Try CPU if original device fails
            if device != 'cpu':
                print("Trying CPU device...")
                self.device = 'cpu'
                self.model = init_model(config_path, checkpoint_path, device='cpu')
                print(f"HRNetV2 model loaded from {checkpoint_path} on CPU")
            else:
                raise e
    
    def extract_predictions(self, image: np.ndarray) -> np.ndarray:
        """
        Extract landmark predictions from HRNetV2.
        
        Args:
            image: Input image (H, W, 3) as numpy array
            
        Returns:
            Predicted landmarks (19, 2) as numpy array
        """
        try:
            # Resize image to manageable size for HRNetV2 inference
            # HRNetV2 was trained on 384x384, so resize to that
            from PIL import Image as PILImage
            
            original_h, original_w = image.shape[:2]
            target_size = 384
            
            # Resize image using PIL for better quality
            pil_image = PILImage.fromarray(image)
            resized_image = pil_image.resize((target_size, target_size))
            resized_image_np = np.array(resized_image)
            
            # Prepare data for inference with proper format
            bbox = np.array([0, 0, target_size, target_size], dtype=np.float32)
            
            # Try GPU inference first
            try:
                # Run inference on resized image with simplified input
                results = inference_topdown(
                    self.model, 
                    resized_image_np, 
                    bboxes=bbox.reshape(1, -1),  # Ensure proper shape (1, 4)
                    bbox_format='xyxy'
                )
                
                if results and len(results) > 0 and hasattr(results[0], 'pred_instances'):
                    if hasattr(results[0].pred_instances, 'keypoints') and len(results[0].pred_instances.keypoints) > 0:
                        predictions = results[0].pred_instances.keypoints[0]  # Shape: (19, 2)
                        predictions_np = predictions.cpu().numpy() if hasattr(predictions, 'cpu') else predictions
                        
                        # Scale predictions back to original image size
                        scale_x = original_w / target_size
                        scale_y = original_h / target_size
                        predictions_np[:, 0] *= scale_x
                        predictions_np[:, 1] *= scale_y
                        
                        return predictions_np
                    else:
                        raise RuntimeError("No keypoints found in predictions")
                else:
                    raise RuntimeError("No predictions returned from model")
                    
            except Exception as gpu_error:
                # Check if this is a CUDA-related error
                error_str = str(gpu_error).lower()
                is_cuda_error = any(keyword in error_str for keyword in ['cuda', 'gpu', 'device', 'out of memory'])
                
                if is_cuda_error and self.device != 'cpu':
                    print(f"CUDA error detected, switching to CPU: {gpu_error}")
                    # Move model to CPU permanently for this session
                    self.model = self.model.cpu()
                    self.device = 'cpu'
                    
                    # Clear GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Retry inference on CPU
                    results = inference_topdown(
                        self.model, 
                        resized_image_np, 
                        bboxes=bbox.reshape(1, -1),  # Ensure proper shape (1, 4)
                        bbox_format='xyxy'
                    )
                    
                    if results and len(results) > 0 and hasattr(results[0], 'pred_instances'):
                        if hasattr(results[0].pred_instances, 'keypoints') and len(results[0].pred_instances.keypoints) > 0:
                            predictions = results[0].pred_instances.keypoints[0]  # Shape: (19, 2)
                            predictions_np = predictions.cpu().numpy() if hasattr(predictions, 'cpu') else predictions
                            
                            # Scale predictions back to original image size
                            scale_x = original_w / target_size
                            scale_y = original_h / target_size
                            predictions_np[:, 0] *= scale_x
                            predictions_np[:, 1] *= scale_y
                            
                            return predictions_np
                        else:
                            raise RuntimeError("No keypoints found in CPU predictions")
                    else:
                        raise RuntimeError("No predictions returned from CPU model")
                else:
                    # Non-CUDA error, re-raise
                    raise gpu_error
                    
            # Attempt to move model back to GPU if it was previously switched to CPU
            if self.device == 'cpu' and self.original_device != 'cpu' and torch.cuda.is_available():
                try:
                    self.model = self.model.to(self.original_device)
                    self.device = self.original_device
                    # Warm-up dummy forward to confirm the model works on GPU again
                    with torch.no_grad():
                        _ = self.model.forward_dummy(torch.randn(1, 3, 64, 64).to(self.original_device)) if hasattr(self.model, 'forward_dummy') else None
                    print("Successfully moved HRNetV2 model back to GPU")
                except Exception as move_err:
                    # If moving back fails, keep using CPU
                    print(f"Failed to move model back to GPU: {move_err}. Continuing on CPU.")
                    
        except Exception as e:
            print(f"Complete inference failure: {e}")
            # Return zero predictions if all inference attempts fail
            return np.zeros((19, 2), dtype=np.float32)


class MLPRefinementDataset(Dataset):
    """
    Dataset for training MLP refinement network.
    
    This dataset:
    1. Loads images and ground truth landmarks
    2. Extracts HRNetV2 predictions for each image
    3. Returns (image, hrnet_predictions, ground_truth) triplets
    """
    
    def __init__(self,
                 data_df: pd.DataFrame,
                 hrnet_config_path: str,
                 hrnet_checkpoint_path: str,
                 input_size: int = 384,
                 cache_predictions: bool = True,
                 device: str = 'cuda:0',
                 force_cpu: bool = False):
        """
        Args:
            data_df: DataFrame with image data and landmarks
            hrnet_config_path: Path to HRNetV2 config file
            hrnet_checkpoint_path: Path to HRNetV2 checkpoint
            input_size: Input image size for MLP network
            cache_predictions: Whether to cache HRNetV2 predictions
            device: Device for HRNetV2 inference
            force_cpu: Force CPU inference to avoid CUDA issues
        """
        self.data_df = data_df.reset_index(drop=True)
        self.input_size = input_size
        self.cache_predictions = cache_predictions
        
        # Override device if force_cpu is True
        if force_cpu:
            device = 'cpu'
            print("Forcing CPU inference to avoid CUDA issues")
        
        # HRNetV2 prediction extractor
        self.hrnet_extractor = HRNetPredictionExtractor(
            hrnet_config_path, hrnet_checkpoint_path, device
        )
        
        # Landmark information
        self.landmark_names = cephalometric_dataset_info.landmark_names_in_order
        self.landmark_cols = cephalometric_dataset_info.original_landmark_cols
        
        # Cache for predictions
        self.prediction_cache = {}
        
        # Pre-extract predictions if caching is enabled
        if cache_predictions:
            print("Pre-extracting HRNetV2 predictions for faster training...")
            self._extract_all_predictions()
    
    def _extract_all_predictions(self):
        """Pre-extract all HRNetV2 predictions and cache them."""
        total_samples = len(self.data_df)
        successful_extractions = 0
        failed_extractions = 0
        
        print(f"Starting prediction extraction using HRNetV2 on {self.hrnet_extractor.device}")
        
        for idx in range(total_samples):
            if idx % 25 == 0:  # More frequent progress updates
                device_info = f"[{self.hrnet_extractor.device.upper()}]"
                print(f"Extracting predictions: {idx}/{total_samples} {device_info} "
                      f"(Success: {successful_extractions}, Failed: {failed_extractions})")
            
            try:
                # Get image
                image = self._get_image(idx)
                
                # Debug: Print image info for first few samples
                if idx < 3:
                    print(f"  Sample {idx}: Image shape {image.shape}, dtype {image.dtype}")
                
                # Extract prediction
                prediction = self.hrnet_extractor.extract_predictions(image)
                
                # Check if prediction is valid (not all zeros)
                if np.all(prediction == 0):
                    print(f"  Warning: Sample {idx} returned zero predictions")
                    failed_extractions += 1
                else:
                    successful_extractions += 1
                
                # Cache prediction
                self.prediction_cache[idx] = prediction
                
                # Memory management: clear GPU cache periodically
                if torch.cuda.is_available() and idx % 100 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                # Cache zero predictions for failed samples
                self.prediction_cache[idx] = np.zeros((19, 2), dtype=np.float32)
                failed_extractions += 1
        
        final_device = self.hrnet_extractor.device
        print(f"Prediction extraction completed on {final_device.upper()}:")
        print(f"  Total samples: {total_samples}")
        print(f"  Successful: {successful_extractions}")
        print(f"  Failed: {failed_extractions}")
        print(f"  Success rate: {successful_extractions/total_samples*100:.1f}%")
        print(f"  Cached {len(self.prediction_cache)} predictions")
    
    def _get_image(self, idx: int) -> np.ndarray:
        """Get image at index."""
        row = self.data_df.iloc[idx]
        
        # Get image from the row (assuming it's stored as 'Image')
        if 'Image' in row:
            image = np.array(row['Image'], dtype=np.uint8)
            
            # Check if image is flattened
            if len(image.shape) == 1:
                # Try different common image sizes
                common_sizes = [224, 256, 384, 512, 640]
                image_reshaped = None
                
                for size in common_sizes:
                    expected_length = size * size * 3
                    if len(image) == expected_length:
                        try:
                            image_reshaped = image.reshape((size, size, 3))
                            break
                        except ValueError:
                            continue
                
                if image_reshaped is None:
                    # If no common size works, try to find the largest square that fits
                    max_pixels = len(image) // 3
                    size = int(np.sqrt(max_pixels))
                    
                    # Find the largest valid size
                    while size > 0:
                        if size * size * 3 <= len(image):
                            try:
                                image_reshaped = image[:size * size * 3].reshape((size, size, 3))
                                print(f"Sample {idx}: Reshaped to {size}x{size} from flattened array of length {len(image)}")
                                break
                            except ValueError:
                                size -= 1
                        else:
                            size -= 1
                
                if image_reshaped is None:
                    # Last resort: create a default image
                    print(f"Warning: Could not reshape image {idx} with length {len(image)}, using default 224x224")
                    image_reshaped = np.zeros((224, 224, 3), dtype=np.uint8)
                
                image = image_reshaped
                
            elif len(image.shape) == 2:
                # If image is 2D, assume it's grayscale and convert to RGB
                image = np.stack([image, image, image], axis=-1)
        else:
            raise ValueError(f"No 'Image' column found in data")
        
        return image
    
    def _get_ground_truth(self, idx: int) -> np.ndarray:
        """Get ground truth landmarks at index."""
        row = self.data_df.iloc[idx]
        
        # Extract ground truth keypoints
        gt_keypoints = []
        for i in range(0, len(self.landmark_cols), 2):
            x_col = self.landmark_cols[i]
            y_col = self.landmark_cols[i+1]
            
            if x_col in row and y_col in row and pd.notna(row[x_col]) and pd.notna(row[y_col]):
                gt_keypoints.append([row[x_col], row[y_col]])
            else:
                gt_keypoints.append([0, 0])  # Invalid landmark
        
        return np.array(gt_keypoints, dtype=np.float32)
    
    def _resize_coordinates(self, coordinates: np.ndarray, original_size: int) -> np.ndarray:
        """Resize coordinates from original image size to target input size."""
        if original_size == self.input_size:
            return coordinates
        
        scale_factor = self.input_size / original_size
        return coordinates * scale_factor
    
    def __len__(self) -> int:
        return len(self.data_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get training sample.
        
        Returns:
            Dictionary containing:
                - 'image': Resized image tensor (3, input_size, input_size)
                - 'hrnet_predictions': HRNetV2 predictions (19, 2)
                - 'ground_truth': Ground truth landmarks (19, 2)
                - 'valid_mask': Mask for valid landmarks (19,)
        """
        # Get image
        image = self._get_image(idx)
        original_size = image.shape[0]  # Assuming square images
        
        # Resize image
        if original_size != self.input_size:
            from PIL import Image as PILImage
            pil_image = PILImage.fromarray(image)
            image = np.array(pil_image.resize((self.input_size, self.input_size)))
        
        # Convert image to tensor and normalize
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Get HRNetV2 predictions
        if self.cache_predictions and idx in self.prediction_cache:
            hrnet_predictions = self.prediction_cache[idx]
        else:
            hrnet_predictions = self.hrnet_extractor.extract_predictions(image)
        
        # Resize predictions to match target input size
        hrnet_predictions = self._resize_coordinates(hrnet_predictions, original_size)
        
        # Get ground truth
        ground_truth = self._get_ground_truth(idx)
        ground_truth = self._resize_coordinates(ground_truth, original_size)
        
        # Create valid mask (landmarks with non-zero coordinates)
        valid_mask = (ground_truth[:, 0] > 0) & (ground_truth[:, 1] > 0)
        
        return {
            'image': image_tensor,
            'hrnet_predictions': torch.from_numpy(hrnet_predictions).float(),
            'ground_truth': torch.from_numpy(ground_truth).float(),
            'valid_mask': torch.from_numpy(valid_mask.astype(np.float32)),
            'idx': idx
        }


def create_dataloaders(train_df: pd.DataFrame,
                      val_df: pd.DataFrame,
                      hrnet_config_path: str,
                      hrnet_checkpoint_path: str,
                      input_size: int = 384,
                      batch_size: int = 16,
                      num_workers: int = 0,
                      cache_predictions: bool = True,
                      force_cpu: bool = False) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for MLP refinement training.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        hrnet_config_path: Path to HRNetV2 config
        hrnet_checkpoint_path: Path to HRNetV2 checkpoint
        input_size: Input image size
        batch_size: Batch size
        num_workers: Number of data loading workers
        cache_predictions: Whether to cache HRNetV2 predictions
        force_cpu: Force CPU inference to avoid CUDA issues
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create datasets
    train_dataset = MLPRefinementDataset(
        train_df, hrnet_config_path, hrnet_checkpoint_path,
        input_size=input_size, cache_predictions=cache_predictions, force_cpu=force_cpu
    )
    
    val_dataset = MLPRefinementDataset(
        val_df, hrnet_config_path, hrnet_checkpoint_path,
        input_size=input_size, cache_predictions=cache_predictions, force_cpu=force_cpu
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_dataloader, val_dataloader


def analyze_hrnet_predictions(dataloader: DataLoader, 
                             output_file: str = "hrnet_prediction_analysis.txt"):
    """
    Analyze HRNetV2 predictions to understand the improvement potential.
    
    Args:
        dataloader: DataLoader with HRNetV2 predictions
        output_file: Output file for analysis results
    """
    all_errors = []
    per_landmark_errors = {name: [] for name in cephalometric_dataset_info.landmark_names_in_order}
    
    print("Analyzing HRNetV2 predictions...")
    
    for batch_idx, batch in enumerate(dataloader):
        hrnet_preds = batch['hrnet_predictions']  # (B, 19, 2)
        ground_truth = batch['ground_truth']      # (B, 19, 2)
        valid_mask = batch['valid_mask']          # (B, 19)
        
        # Compute errors
        errors = torch.norm(hrnet_preds - ground_truth, dim=-1)  # (B, 19)
        
        # Collect valid errors
        for b in range(errors.size(0)):
            for l in range(errors.size(1)):
                if valid_mask[b, l] > 0:
                    error = errors[b, l].item()
                    all_errors.append(error)
                    landmark_name = cephalometric_dataset_info.landmark_names_in_order[l]
                    per_landmark_errors[landmark_name].append(error)
        
        if batch_idx % 10 == 0:
            print(f"Processed {batch_idx}/{len(dataloader)} batches")
    
    # Compute statistics
    all_errors = np.array(all_errors)
    overall_mre = np.mean(all_errors)
    overall_std = np.std(all_errors)
    
    # Write analysis results
    with open(output_file, 'w') as f:
        f.write("HRNetV2 Prediction Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Overall MRE: {overall_mre:.3f} ± {overall_std:.3f} pixels\n")
        f.write(f"Median error: {np.median(all_errors):.3f} pixels\n")
        f.write(f"90th percentile: {np.percentile(all_errors, 90):.3f} pixels\n")
        f.write(f"95th percentile: {np.percentile(all_errors, 95):.3f} pixels\n\n")
        
        f.write("Per-Landmark Statistics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Index':<5} {'Landmark':<20} {'MRE':<8} {'Std':<8} {'Count':<6}\n")
        f.write("-" * 60 + "\n")
        
        for i, name in enumerate(cephalometric_dataset_info.landmark_names_in_order):
            errors = np.array(per_landmark_errors[name])
            if len(errors) > 0:
                mre = np.mean(errors)
                std = np.std(errors)
                count = len(errors)
                f.write(f"{i:<5} {name:<20} {mre:<8.3f} {std:<8.3f} {count:<6}\n")
    
    print(f"Analysis results saved to {output_file}")
    print(f"Overall MRE: {overall_mre:.3f} ± {overall_std:.3f} pixels")


if __name__ == "__main__":
    # Test the dataset
    print("Testing MLP Refinement Dataset...")
    
    # Initialize MMPose scope
    init_default_scope('mmpose')
    
    # Import custom modules
    import custom_cephalometric_dataset
    import custom_transforms
    
    # Load test data
    data_file_path = "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json"
    main_df = pd.read_json(data_file_path)
    
    # Use a small subset for testing
    test_df = main_df[main_df['set'] == 'dev'].head(10).reset_index(drop=True)
    
    # HRNetV2 paths
    hrnet_config = "Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py"
    hrnet_checkpoint = "work_dirs/hrnetv2_w18_cephalometric_384x384_adaptive_wing_loss_v4/best_NME_epoch_*.pth"
    
    # Find actual checkpoint
    import glob
    checkpoints = glob.glob(hrnet_checkpoint)
    if checkpoints:
        hrnet_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"Using checkpoint: {hrnet_checkpoint}")
        
        # Create dataset
        dataset = MLPRefinementDataset(
            test_df, hrnet_config, hrnet_checkpoint,
            input_size=384, cache_predictions=True
        )
        
        # Test a sample
        sample = dataset[0]
        print(f"Sample shapes:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {value}")
        
        print("✓ Dataset test completed successfully!")
    else:
        print("No HRNetV2 checkpoint found for testing") 