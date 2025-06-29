#!/usr/bin/env python3
"""
Concurrent MLP Training Hook for MMEngine / MMPose - Residual Learning Version
-----------------------------------------------------------------------------
This hook trains a joint MLP refinement model **concurrently** with HRNetV2 training.  
After every HRNet training epoch, the hook:

1.  Runs inference on the entire *training* dataloader using the *current*
    HRNetV2 weights to obtain predicted landmark coordinates.
2.  Creates an in-memory dataset of (predicted → residual) coordinate pairs where
    residuals = ground_truth - predictions (corrections to apply).
3.  Trains a joint MLP for a fixed number of epochs (default: 100) to predict these residuals.
4.  Implements hard-example oversampling for samples with high landmark errors.
5.  Updates normalization scalers incrementally to adapt to evolving HRNet predictions.

Important design decisions:
•   **Joint 38-D residual model** – Single MLP that predicts residual corrections for all 
    38 coordinates (19 x,y pairs), allowing cross-correlation learning between landmarks.
•   **Residual learning** – Instead of predicting absolute coordinates, the MLP learns to 
    predict corrections: target = gt - pred, refined = pred + residual_correction.
•   **Incremental scaler updates** – Scalers use partial_fit to adapt to evolving HRNet 
    output distributions instead of being frozen after epoch 0.
•   **Hard-example oversampling** – Samples with any landmark MRE > threshold get
    duplicated in the training batch to focus learning on difficult cases.
•   **One-time model initialisation** – MLP weights and optimisers are created
    once in `before_run` and *persist* across the whole HRNet training.
•   **No gradient leakage** – MLP training is completely detached from the
    HRNetV2 computation graph (`torch.no_grad()`), so gradients do **not**
    propagate back.
•   **CPU/GPU awareness** – Trains on GPU if available, else CPU.

Benefits of residual learning:
•   **Smaller regression range** – Few pixels instead of 0-384, making optimization easier.
•   **Smoother loss landscape** – Residuals are centered around 0 with small magnitude.
•   **Focused error correction** – MLP learns to correct specific HRNet error patterns.
•   **Avoids identity mapping** – No need to learn f(x) ≈ x + small_correction.

To enable this hook, add to your config:

```
custom_hooks = [
    dict(
        type='ConcurrentMLPTrainingHook',
        mlp_epochs=100,
        mlp_batch_size=16,
        mlp_lr=1e-4,  # Increased LR for residual learning
        mlp_weight_decay=1e-4,
        hard_example_threshold=5.0,  # MRE threshold for oversampling
        log_interval=20,
        enable_hrnet_oversampling=True,
        max_oversample_weight=5.0
    )
]
```

Make sure this file is importable (e.g. by placing it in PYTHONPATH or the
workspace root).
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.logging import MMLogger
from mmengine.runner import Runner

from mmpose.apis import inference_topdown
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# -----------------------------------------------------------------------------
#  Joint MLP architecture for 38-D coordinate prediction
# -----------------------------------------------------------------------------

class JointMLPRefinementModel(nn.Module):
    """Joint MLP model for landmark coordinate residual prediction.
    
    Input: 38 predicted coordinates (19 landmarks × 2 coordinates)
    Hidden: 500 neurons with residual connection
    Output: 38 coordinate residuals (corrections to apply to input predictions)
    
    This allows the network to learn cross-correlations between X and Y axes
    and between different landmarks while focusing on error correction.
    """

    def __init__(self, input_dim: int = 38, hidden_dim: int = 500, output_dim: int = 38):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Main network for residual prediction
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights for small residual outputs (important for residual learning)
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # Small initialization
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # Predict residuals (corrections to apply)
        residuals = self.net(x)
        return residuals


# -----------------------------------------------------------------------------
#  Hook implementation
# -----------------------------------------------------------------------------

@HOOKS.register_module()
class ConcurrentMLPTrainingHook(Hook):
    """MMEngine hook that performs concurrent joint MLP refinement training."""

    priority = 'LOW'  # Run after default hooks

    def __init__(
        self,
        mlp_epochs: int = 100,
        mlp_batch_size: int = 16,
        mlp_lr: float = 1e-4,  # Increased default LR for residual learning
        mlp_weight_decay: float = 1e-4,
        hard_example_threshold: float = 5.0,  # MRE threshold in pixels
        log_interval: int = 50,
        enable_hrnet_oversampling: bool = False,
        max_oversample_weight: float = 5.0,
    ) -> None:
        self.mlp_epochs = mlp_epochs
        self.mlp_batch_size = mlp_batch_size
        self.mlp_lr = mlp_lr
        self.mlp_weight_decay = mlp_weight_decay
        self.hard_example_threshold = hard_example_threshold
        self.log_interval = log_interval
        self.enable_hrnet_oversampling = enable_hrnet_oversampling
        self.max_oversample_weight = max_oversample_weight

        # These will be initialised in before_run
        self.mlp_joint: JointMLPRefinementModel | None = None
        self.opt_joint: optim.Optimizer | None = None
        self.scheduler_joint: optim.lr_scheduler.LRScheduler | None = None # Cosine scheduler for MLP
        self.criterion = nn.SmoothL1Loss()  # Switched to SmoothL1Loss for robust residual training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Normalization scalers - updated incrementally with partial_fit
        self.scaler_input: StandardScaler | None = None
        self.scaler_residual: StandardScaler | None = None
        
        # To store sample weights for HRNet oversampling in the next epoch
        self.hrnet_next_epoch_sample_weights: Optional[torch.Tensor] = None

    # ---------------------------------------------------------------------
    # MMEngine lifecycle methods
    # ---------------------------------------------------------------------

    def before_run(self, runner: Runner):
        logger: MMLogger = runner.logger
        logger.info('[ConcurrentMLPTrainingHook] Initialising joint 38-D MLP residual prediction model …')

        self.mlp_joint = JointMLPRefinementModel().to(self.device)
        self.opt_joint = optim.Adam(self.mlp_joint.parameters(), lr=self.mlp_lr, weight_decay=self.mlp_weight_decay)
        self.scheduler_joint = optim.lr_scheduler.CosineAnnealingLR(self.opt_joint, T_max=self.mlp_epochs, eta_min=1e-7)
        
        # Note: Scalers will be initialized incrementally with partial_fit during first epoch
        
        logger.info(f'[ConcurrentMLPTrainingHook] Joint residual MLP initialized with {sum(p.numel() for p in self.mlp_joint.parameters()):,} parameters')
        logger.info(f'[ConcurrentMLPTrainingHook] Hard-example threshold: {self.hard_example_threshold} pixels')
        logger.info(f'[ConcurrentMLPTrainingHook] Residual learning: MLP predicts corrections (gt - pred) instead of absolute coordinates')

        if self.enable_hrnet_oversampling:
            logger.info('[ConcurrentMLPTrainingHook] HRNet hard-example oversampling is ENABLED.')
            logger.info(f'[ConcurrentMLPTrainingHook] HRNet dynamic oversample weight capped at: {self.max_oversample_weight}')
        else:
            logger.info('[ConcurrentMLPTrainingHook] HRNet hard-example oversampling is DISABLED.')

    def before_train_epoch(self, runner: Runner):
        """Before each HRNet epoch, recreate the dataloader to oversample hard examples."""
        if not self.enable_hrnet_oversampling:
            return

        logger: MMLogger = runner.logger
        
        if self.hrnet_next_epoch_sample_weights is None:
            logger.info('[ConcurrentMLPTrainingHook] No sample weights from previous epoch. Training on original dataset.')
            return

        logger.info(f'[ConcurrentMLPTrainingHook] Recreating HRNet dataloader with dynamic sample weights from the previous epoch.')

        original_loader = runner.train_dataloader
        dataset = original_loader.dataset
        
        # The weights are pre-computed from the previous epoch's `after_train_epoch`
        weights = self.hrnet_next_epoch_sample_weights
        
        # Create a new weighted random sampler. replacement=True is important for oversampling.
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(dataset), replacement=True
        )
        
        # Re-create the DataLoader using the new sampler, preserving other settings.
        # This is a robust way to handle this without digging too deep into MMEngine's private APIs.
        new_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=original_loader.batch_size,
            sampler=sampler,
            num_workers=original_loader.num_workers,
            collate_fn=original_loader.collate_fn,
            pin_memory=original_loader.pin_memory,
            drop_last=getattr(original_loader, 'drop_last', False),
            timeout=getattr(original_loader, 'timeout', 0),
            worker_init_fn=getattr(original_loader, 'worker_init_fn', None),
            persistent_workers=getattr(original_loader, 'persistent_workers', False),
        )
        
        # The dataloader is held by the train_loop in MMEngine's runner
        runner.train_loop.dataloader = new_loader
        
        logger.info('[ConcurrentMLPTrainingHook] HRNet dataloader replaced with a WeightedRandomSampler for hard-example oversampling.')

    def after_train_epoch(self, runner: Runner):
        """After each HRNetV2 epoch, train joint MLP on-the-fly using current predictions."""
        logger: MMLogger = runner.logger
        assert self.mlp_joint is not None

        # -----------------------------------------------------------------
        # Step 1: Generate predictions on training data (GPU-optimized)
        # -----------------------------------------------------------------
        
        # Get the actual model, handling potential wrapping
        model = runner.model
        if hasattr(model, 'module'):
            # Handle DDP or other wrapped models
            actual_model = model.module
        else:
            actual_model = model
            
        # Ensure model has required attributes for inference
        if not hasattr(actual_model, 'cfg') and hasattr(runner, 'cfg'):
            actual_model.cfg = runner.cfg
            
        # Set up dataset_meta if missing (required for inference_topdown)
        if not hasattr(actual_model, 'dataset_meta'):
            try:
                import cephalometric_dataset_info
                dataset_meta = {
                    'dataset_name': 'cephalometric',
                    'joint_weights': cephalometric_dataset_info.dataset_info.get('joint_weights', [1.0] * 19),
                    'sigmas': cephalometric_dataset_info.dataset_info.get('sigmas', [0.035] * 19),
                    'flip_indices': cephalometric_dataset_info.dataset_info.get('flip_indices', list(range(19))),
                    'keypoint_info': cephalometric_dataset_info.dataset_info.get('keypoint_info', {}),
                    'skeleton_info': cephalometric_dataset_info.dataset_info.get('skeleton_info', []),
                    'keypoint_name2id': {f'keypoint_{i}': i for i in range(19)},
                    'keypoint_id2name': {i: f'keypoint_{i}' for i in range(19)},
                }
                actual_model.dataset_meta = dataset_meta
                logger.info('[ConcurrentMLPTrainingHook] Set dataset_meta for inference compatibility')
            except Exception as e:
                logger.warning(f'[ConcurrentMLPTrainingHook] Could not set dataset_meta: {e}')
        
        model.eval()
        all_preds: List[np.ndarray] = []
        all_gts: List[np.ndarray] = []
        all_errors: List[np.ndarray] = []  # For hard-example detection

        train_dataset = runner.train_dataloader.dataset
        logger.info('[ConcurrentMLPTrainingHook] Generating predictions for joint MLP using fast batched inference...')

        def tensor_to_numpy(data):
            """Safely convert tensor to numpy, handling both tensor and numpy inputs."""
            if isinstance(data, torch.Tensor):
                return data.cpu().numpy()
            elif isinstance(data, np.ndarray):
                return data
            else:
                return np.array(data)

        # FAST BATCHED INFERENCE: Use the training dataloader directly
        try:
            from tqdm import tqdm
            
            processed_count = 0
            
            # Use the actual training dataloader for batched processing
            train_dataloader = runner.train_dataloader
            
            logger.info(f'[ConcurrentMLPTrainingHook] Processing {len(train_dataloader)} batches with batch size {train_dataloader.batch_size}')
            
            with torch.no_grad():
                for batch_idx, data_batch in enumerate(tqdm(train_dataloader, desc="Batched GPU Inference", disable=runner.rank != 0)):
                    try:
                        # Extract batch data
                        if 'inputs' in data_batch:
                            # Inputs are already processed and on the correct device
                            inputs = data_batch['inputs']
                            if not isinstance(inputs, torch.Tensor):
                                inputs = torch.stack([inp for inp in inputs])
                            
                            # Ensure proper dtype and normalization
                            if inputs.dtype == torch.uint8:
                                # Convert from uint8 [0, 255] to float32 [0, 1] and then normalize
                                inputs = inputs.float() / 255.0
                                
                                # Apply ImageNet normalization (same as training pipeline)
                                # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
                                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(inputs.device)
                                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(inputs.device)
                                inputs = (inputs - mean) / std
                            
                            # Move to device if needed
                            if inputs.device != self.device:
                                inputs = inputs.to(self.device, non_blocking=True)
                        else:
                            continue
                        
                        # Extract ground truth data
                        if 'data_samples' in data_batch:
                            data_samples = data_batch['data_samples']
                            
                            batch_gt_keypoints = []
                            batch_valid = []
                            
                            for sample in data_samples:
                                if hasattr(sample, 'gt_instances') and hasattr(sample.gt_instances, 'keypoints'):
                                    gt_kpts = tensor_to_numpy(sample.gt_instances.keypoints[0])
                                    if gt_kpts.shape[0] == 19:
                                        batch_gt_keypoints.append(gt_kpts)
                                        batch_valid.append(True)
                                    else:
                                        batch_valid.append(False)
                                else:
                                    batch_valid.append(False)
                            
                            if not any(batch_valid):
                                continue
                        else:
                            continue
                        
                        # Run batched inference through the model
                        # Use the model's forward method directly for speed
                        if hasattr(model, 'predict'):
                            # Use predict method if available (MMPose 1.x)
                            results = model.predict(inputs, data_batch['data_samples'])
                        else:
                            # Fallback to direct forward pass
                            results = model(inputs, data_batch['data_samples'], mode='predict')
                        
                        # Process batch results
                        for i, (result, gt_kpts, is_valid) in enumerate(zip(results, batch_gt_keypoints, batch_valid)):
                            if not is_valid:
                                continue
                                
                            try:
                                # Extract predicted keypoints
                                if hasattr(result, 'pred_instances') and hasattr(result.pred_instances, 'keypoints'):
                                    pred_kpts = tensor_to_numpy(result.pred_instances.keypoints[0])
                                else:
                                    continue
                                
                                if pred_kpts.shape[0] != 19:
                                    continue
                                
                                # Flatten coordinates to 38-D vectors
                                pred_flat = pred_kpts.flatten()
                                gt_flat = gt_kpts.flatten()
                                
                                # Calculate per-landmark radial errors
                                landmark_errors = np.sqrt(np.sum((pred_kpts - gt_kpts)**2, axis=1))
                                
                                # Store data
                                all_preds.append(pred_flat)
                                all_gts.append(gt_flat)
                                all_errors.append(landmark_errors)
                                
                                processed_count += 1
                                
                            except Exception as e:
                                logger.warning(f'[ConcurrentMLPTrainingHook] Failed to process batch sample {i}: {e}')
                                continue
                        
                    except Exception as e:
                        logger.warning(f'[ConcurrentMLPTrainingHook] Failed to process batch {batch_idx}: {e}')
                        continue
            
            logger.info(f'[ConcurrentMLPTrainingHook] Successfully processed {processed_count} samples using fast batched inference')

        except Exception as e:
            logger.error(f'[ConcurrentMLPTrainingHook] Critical error during batched inference: {e}')
            
            # Fallback to the old slow method if batched inference fails
            logger.warning('[ConcurrentMLPTrainingHook] Falling back to slow individual inference...')
            
            # Access training data more robustly
            try:
                from tqdm import tqdm
                
                # Method 1: Try to access the raw annotation file
                if hasattr(train_dataset, 'ann_file') and train_dataset.ann_file:
                    logger.info(f'[ConcurrentMLPTrainingHook] Loading data from annotation file: {train_dataset.ann_file}')
                    import pandas as pd
                    
                    try:
                        df = pd.read_json(train_dataset.ann_file)
                        logger.info(f'[ConcurrentMLPTrainingHook] Loaded {len(df)} samples from annotation file')
                        
                        # Import landmark info
                        import cephalometric_dataset_info
                        landmark_names = cephalometric_dataset_info.landmark_names_in_order
                        landmark_cols = cephalometric_dataset_info.original_landmark_cols
                        
                        processed_count = 0
                        
                        for idx, row in tqdm(df.iterrows(), total=len(df), disable=runner.rank != 0, desc="GPU Inference from File"):
                            try:
                                # Get image from row
                                img_array = np.array(row['Image'], dtype=np.uint8).reshape((224, 224, 3))
                                
                                # Get ground truth keypoints
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
                                
                                # Skip samples with invalid ground truth
                                if not valid_gt:
                                    continue
                                    
                                gt_keypoints = np.array(gt_keypoints)
                                
                                # Run inference with the standard mmpose API
                                bbox = np.array([[0, 0, 224, 224]], dtype=np.float32)
                                results = inference_topdown(model, img_array, bboxes=bbox, bbox_format='xyxy')
                                
                                if results and len(results) > 0:
                                    pred_keypoints = results[0].pred_instances.keypoints[0]
                                    pred_keypoints = tensor_to_numpy(pred_keypoints)
                                else:
                                    continue
                                
                                if pred_keypoints is None or pred_keypoints.shape[0] != 19:
                                    continue
                                
                                # Flatten coordinates to 38-D vectors
                                pred_flat = pred_keypoints.flatten()  # [x1, y1, x2, y2, ..., x19, y19]
                                gt_flat = gt_keypoints.flatten()
                                
                                # Calculate per-landmark radial errors for hard-example detection
                                landmark_errors = np.sqrt(np.sum((pred_keypoints - gt_keypoints)**2, axis=1))
                                
                                # Store data
                                all_preds.append(pred_flat)
                                all_gts.append(gt_flat)
                                all_errors.append(landmark_errors)
                                
                                processed_count += 1
                                
                            except Exception as e:
                                logger.warning(f'[ConcurrentMLPTrainingHook] Failed to process sample {idx}: {e}')
                                continue
                        
                        logger.info(f'[ConcurrentMLPTrainingHook] Successfully processed {processed_count} samples from file')
                        
                    except Exception as e:
                        logger.warning(f'[ConcurrentMLPTrainingHook] Failed to load from annotation file: {e}')
                        df = None
                
                # Method 2: Fallback to dataset iteration if file method fails
                if not all_preds:  # No predictions from file method
                    logger.info('[ConcurrentMLPTrainingHook] Using dataset iteration method')
                    
                    processed_count = 0
                    
                    for idx in tqdm(range(len(train_dataset)), disable=runner.rank != 0, desc="Dataset Inference"):
                        try:
                            data_sample = train_dataset[idx]
                            
                            # Extract image - handle different possible formats
                            if 'inputs' in data_sample:
                                img = data_sample['inputs']
                                if isinstance(img, torch.Tensor):
                                    # Convert from (C, H, W) tensor to (H, W, C) numpy
                                    img_np = tensor_to_numpy(img.permute(1, 2, 0) * 255).astype(np.uint8)
                                else:
                                    img_np = np.array(img, dtype=np.uint8)
                            else:
                                continue
                            
                            # Extract ground truth keypoints with robust handling
                            if 'data_samples' in data_sample and hasattr(data_sample['data_samples'], 'gt_instances'):
                                gt_instances = data_sample['data_samples'].gt_instances
                                if hasattr(gt_instances, 'keypoints') and len(gt_instances.keypoints) > 0:
                                    gt_kpts = tensor_to_numpy(gt_instances.keypoints[0])
                                else:
                                    continue
                            else:
                                continue
                            
                            # Run inference with the standard mmpose API
                            bbox = np.array([[0, 0, 224, 224]], dtype=np.float32)
                            results = inference_topdown(model, img_np, bboxes=bbox, bbox_format='xyxy')
                            
                            if results and len(results) > 0:
                                pred_kpts = results[0].pred_instances.keypoints[0]
                                pred_kpts = tensor_to_numpy(pred_kpts)
                            else:
                                continue
                            
                            if pred_kpts is None or pred_kpts.shape[0] != 19:
                                continue
                            
                            # Flatten coordinates to 38-D vectors
                            pred_flat = pred_kpts.flatten()
                            gt_flat = gt_kpts.flatten()
                            
                            # Calculate per-landmark radial errors
                            landmark_errors = np.sqrt(np.sum((pred_kpts - gt_kpts)**2, axis=1))
                            
                            # Store data
                            all_preds.append(pred_flat)
                            all_gts.append(gt_flat)
                            all_errors.append(landmark_errors)
                            
                            processed_count += 1
                            
                        except Exception as e:
                            # Only log every 100th error to avoid spam
                            if idx % 100 == 0:
                                logger.warning(f'[ConcurrentMLPTrainingHook] Failed to process sample {idx}: {e}')
                            continue
                    
                    logger.info(f'[ConcurrentMLPTrainingHook] Successfully processed {processed_count} samples via dataset iteration')

            except Exception as e:
                logger.error(f'[ConcurrentMLPTrainingHook] Critical error during fallback processing: {e}')
                return

        if not all_preds:
            logger.warning('[ConcurrentMLPTrainingHook] No predictions generated; skipping MLP update.')
            return

        all_preds = np.stack(all_preds)  # [N, 38]
        all_gts = np.stack(all_gts)      # [N, 38]
        all_errors = np.stack(all_errors)  # [N, 19]
        
        logger.info(f'[ConcurrentMLPTrainingHook] Generated predictions for {len(all_preds)} samples')

        # -----------------------------------------------------------------
        # Step 1.5: Compute hard-example weights
        # -----------------------------------------------------------------
        # Determine sample weights based on maximum landmark error per sample
        max_errors_per_sample = np.max(all_errors, axis=1)  # [N,] - worst landmark per sample
        
        # Calculate dynamic weights: 1.0 for easy examples, >1.0 for hard examples
        # The weight is proportional to how much the error exceeds the threshold.
        sample_weights = np.maximum(1.0, max_errors_per_sample / self.hard_example_threshold)
        
        # Cap the weights to prevent extreme outliers from dominating
        sample_weights = np.minimum(sample_weights, self.max_oversample_weight)
        
        num_hard_examples = np.sum(max_errors_per_sample > self.hard_example_threshold)
        logger.info(f'[ConcurrentMLPTrainingHook] Hard examples (>{self.hard_example_threshold}px): {num_hard_examples}/{len(all_preds)} ({num_hard_examples/len(all_preds)*100:.1f}%)')
        
        # Store sample weights for HRNet oversampling in the *next* epoch
        if self.enable_hrnet_oversampling:
            self.hrnet_next_epoch_sample_weights = torch.from_numpy(sample_weights).double()
            logger.info(f'[ConcurrentMLPTrainingHook] Stored {len(self.hrnet_next_epoch_sample_weights)} dynamic sample weights for next HRNet epoch.')

        if num_hard_examples > 0:
            hard_example_errors = max_errors_per_sample[max_errors_per_sample > self.hard_example_threshold]
            logger.info(f'[ConcurrentMLPTrainingHook] Hard example errors: min={np.min(hard_example_errors):.2f}, max={np.max(hard_example_errors):.2f}, mean={np.mean(hard_example_errors):.2f}')
            
            hard_example_weights = sample_weights[max_errors_per_sample > self.hard_example_threshold]
            logger.info(f'[ConcurrentMLPTrainingHook] Corresponding sample weights: min={np.min(hard_example_weights):.2f}, max={np.max(hard_example_weights):.2f}, mean={np.mean(hard_example_weights):.2f}')

        # -----------------------------------------------------------------
        # Step 2: Calculate residuals and update scalers incrementally
        # -----------------------------------------------------------------
        # Calculate residuals: target = gt - pred (what corrections to apply)
        all_residuals = all_gts - all_preds  # [N, 38] - small corrections centered around 0
        
        logger.info(f'[ConcurrentMLPTrainingHook] Residual statistics: '
                   f'mean={np.mean(np.abs(all_residuals)):.3f}px, '
                   f'std={np.std(all_residuals):.3f}px, '
                   f'max={np.max(np.abs(all_residuals)):.3f}px')

        # Initialize or update normalization scalers incrementally
        if self.scaler_input is None:
            logger.info('[ConcurrentMLPTrainingHook] Initializing normalization scalers for 38-D data...')
            
            # Initialize scalers with first batch of data
            self.scaler_input = StandardScaler()
            self.scaler_residual = StandardScaler()
            self.scaler_input.partial_fit(all_preds)
            self.scaler_residual.partial_fit(all_residuals)
            
            logger.info('[ConcurrentMLPTrainingHook] Normalization scalers initialized with partial_fit')
            
            # Save scalers for evaluation
            save_dir = os.path.join(runner.work_dir, 'concurrent_mlp')
            os.makedirs(save_dir, exist_ok=True)
            
            joblib.dump(self.scaler_input, os.path.join(save_dir, 'scaler_joint_input.pkl'))
            joblib.dump(self.scaler_residual, os.path.join(save_dir, 'scaler_joint_residual.pkl'))
            
            logger.info(f'[ConcurrentMLPTrainingHook] Joint scalers saved to {save_dir}')
        else:
            # Update scalers incrementally with new epoch data (adapts to evolving HRNet)
            logger.info('[ConcurrentMLPTrainingHook] Updating scalers with partial_fit for evolving HRNet distribution...')
            self.scaler_input.partial_fit(all_preds)
            self.scaler_residual.partial_fit(all_residuals)
            
            # Save updated scalers
            save_dir = os.path.join(runner.work_dir, 'concurrent_mlp')
            joblib.dump(self.scaler_input, os.path.join(save_dir, 'scaler_joint_input.pkl'))
            joblib.dump(self.scaler_residual, os.path.join(save_dir, 'scaler_joint_residual.pkl'))

        # Normalize data using the updated scalers
        preds_scaled = self.scaler_input.transform(all_preds)
        residuals_scaled = self.scaler_residual.transform(all_residuals)

        # -----------------------------------------------------------------
        # Step 3: Train joint MLP for residual prediction (GPU-optimized)
        # -----------------------------------------------------------------
        logger.info('[ConcurrentMLPTrainingHook] Training joint 38-D MLP for residual prediction on GPU…')
        
        # Calculate initial MRE in pixels before MLP refinement for meaningful logging
        initial_mre = self._calculate_mre_pixels(all_preds, all_gts)

        # Build dataset and dataloader with weighted sampling for the MLP
        ds_joint = data.TensorDataset(torch.from_numpy(preds_scaled).float(), torch.from_numpy(residuals_scaled).float())
        sampler_joint = data.WeightedRandomSampler(sample_weights, num_samples=len(ds_joint), replacement=True)
        dl_joint = data.DataLoader(ds_joint, batch_size=self.mlp_batch_size, sampler=sampler_joint, pin_memory=True, num_workers=4)

        def _train_joint(model: JointMLPRefinementModel, optimiser: optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler, loader: data.DataLoader, initial_mre: float):
            model.train()
            total_loss = 0.0
            for ep in range(self.mlp_epochs):
                epoch_loss = 0.0
                for preds_batch, residuals_batch in loader:
                    preds_batch = preds_batch.to(self.device, non_blocking=True)
                    residuals_batch = residuals_batch.to(self.device, non_blocking=True)

                    optimiser.zero_grad()
                    predicted_residuals = model(preds_batch)
                    loss = self.criterion(predicted_residuals, residuals_batch)
                    loss.backward()
                    optimiser.step()

                    epoch_loss += loss.item()
                
                scheduler.step() # Update learning rate
                
                total_loss = epoch_loss / len(loader)
                if (ep + 1) % 20 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    logger.info(f'[ConcurrentMLPTrainingHook] Joint-MLP epoch {ep+1}/{self.mlp_epochs} | Loss: {total_loss:.6f} | LR: {current_lr:.2e} | Initial MRE: {initial_mre:.3f}px')

        _train_joint(self.mlp_joint, self.opt_joint, self.scheduler_joint, dl_joint, initial_mre)

        logger.info('[ConcurrentMLPTrainingHook] Finished joint MLP residual prediction update for this HRNet epoch.')
        
        # Save MLP models after each epoch
        save_dir = os.path.join(runner.work_dir, 'concurrent_mlp')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save current epoch models
        current_epoch = runner.epoch + 1  # runner.epoch is 0-indexed
        mlp_joint_epoch_path = os.path.join(save_dir, f'mlp_joint_epoch_{current_epoch}.pth')
        
        torch.save(self.mlp_joint.state_dict(), mlp_joint_epoch_path)
        
        # Also save as "latest" for easy access
        mlp_joint_latest_path = os.path.join(save_dir, 'mlp_joint_latest.pth')
        torch.save(self.mlp_joint.state_dict(), mlp_joint_latest_path)
        
        logger.info(f'[ConcurrentMLPTrainingHook] Joint MLP model saved for epoch {current_epoch}')
        logger.info(f'[ConcurrentMLPTrainingHook] Latest model: {mlp_joint_latest_path}')

    # ---------------------------------------------------------------------
    # Optional: save MLP weights at end of run
    # ---------------------------------------------------------------------
    def after_run(self, runner: Runner):
        logger: MMLogger = runner.logger
        if self.mlp_joint is None:
            return
        save_dir = os.path.join(runner.work_dir, 'concurrent_mlp')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.mlp_joint.state_dict(), os.path.join(save_dir, 'mlp_joint_final.pth'))
        logger.info(f'[ConcurrentMLPTrainingHook] Saved final joint MLP weights to {save_dir}')

    def _calculate_mre_pixels(self, preds: np.ndarray, gts: np.ndarray) -> float:
        """Calculate Mean Radial Error (MRE) in pixels."""
        # Reshape from [N, 38] to [N, 19, 2] for coordinate pairs
        preds_reshaped = preds.reshape(-1, 19, 2)  # [N, 19, 2]
        gts_reshaped = gts.reshape(-1, 19, 2)      # [N, 19, 2]
        
        # Calculate radial errors per landmark: sqrt((x_pred - x_gt)^2 + (y_pred - y_gt)^2)
        radial_errors = np.sqrt(np.sum((preds_reshaped - gts_reshaped)**2, axis=2))  # [N, 19]
        
        # Return mean radial error across all samples and landmarks
        return np.mean(radial_errors) 