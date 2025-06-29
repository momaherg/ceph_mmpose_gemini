#!/usr/bin/env python3
"""
Concurrent MLP Training Hook for MMEngine / MMPose
-------------------------------------------------
This hook trains a joint MLP refinement model **concurrently** with HRNetV2 training.  
After every HRNet training epoch, the hook:

1.  Runs inference on the entire *training* dataloader using the *current*
    HRNetV2 weights to obtain predicted landmark coordinates.
2.  Creates an in-memory dataset of (predicted → ground-truth) coordinate pairs.
3.  Trains a joint MLP for a fixed number of epochs (default: 100).
4.  Implements hard-example oversampling for samples with high landmark errors.

Important design decisions:
•   **Joint 38-D model** – Single MLP that predicts all 38 coordinates (19 x,y pairs)
    allowing the network to learn cross-correlations between X and Y axes.
•   **Hard-example oversampling** – Samples with any landmark MRE > threshold get
    duplicated in the training batch to focus learning on difficult cases.
•   **One-time initialisation** – MLP weights, optimisers and scalers are created
    once in `before_run` and *persist* across the whole HRNet training.
•   **No gradient leakage** – MLP training is completely detached from the
    HRNetV2 computation graph (`torch.no_grad()`), so gradients do **not**
    propagate back.
•   **CPU/GPU awareness** – Trains on GPU if available, else CPU.

To enable this hook, add to your config:

```
custom_hooks = [
    dict(
        type='ConcurrentMLPTrainingHook',
        mlp_epochs=100,
        mlp_batch_size=16,
        mlp_lr=1e-5,
        mlp_weight_decay=1e-4,
        hard_example_threshold=5.0,  # MRE threshold for oversampling
        log_interval=20,
        inference_batch_size=80
    )
]
```

Make sure this file is importable (e.g. by placing it in PYTHONPATH or the
workspace root).
"""

from __future__ import annotations

import os
from typing import List

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
    """Joint MLP model for landmark coordinate refinement.
    
    Input: 38 predicted coordinates (19 landmarks × 2 coordinates)
    Hidden: 500 neurons with residual connection
    Output: 38 refined coordinates
    
    This allows the network to learn cross-correlations between X and Y axes
    and between different landmarks.
    """

    def __init__(self, input_dim: int = 38, hidden_dim: int = 500, output_dim: int = 38):
        super().__init__()
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


class _MLPDataset(data.Dataset):
    """In-memory dataset with hard-example oversampling."""

    def __init__(self, preds: np.ndarray, gts: np.ndarray, sample_weights: np.ndarray = None):
        # preds/gts shape: [N, 38] (flattened coordinates)
        assert preds.shape == gts.shape
        self.preds = torch.from_numpy(preds).float()
        self.gts = torch.from_numpy(gts).float()
        
        # Create weighted sampling indices for hard examples
        if sample_weights is not None:
            # Oversample hard examples by duplicating their indices
            self.indices = []
            for i, weight in enumerate(sample_weights):
                # Add base sample
                self.indices.append(i)
                # Add extra copies for hard examples (weight > 1)
                extra_copies = int(weight) - 1
                for _ in range(extra_copies):
                    self.indices.append(i)
            self.indices = np.array(self.indices)
        else:
            self.indices = np.arange(len(self.preds))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.preds[actual_idx], self.gts[actual_idx]


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
        mlp_lr: float = 1e-5,
        mlp_weight_decay: float = 1e-4,
        hard_example_threshold: float = 5.0,  # MRE threshold in pixels
        log_interval: int = 50,
        inference_batch_size: int = 80,
    ) -> None:
        self.mlp_epochs = mlp_epochs
        self.mlp_batch_size = mlp_batch_size
        self.mlp_lr = mlp_lr
        self.mlp_weight_decay = mlp_weight_decay
        self.hard_example_threshold = hard_example_threshold
        self.log_interval = log_interval
        self.inference_batch_size = inference_batch_size

        # These will be initialised in before_run
        self.mlp_joint: JointMLPRefinementModel | None = None
        self.opt_joint: optim.Optimizer | None = None
        self.criterion = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Normalization scalers - initialized once and reused
        self.scaler_input: StandardScaler | None = None
        self.scaler_target: StandardScaler | None = None
        self.scalers_initialized = False

    # ---------------------------------------------------------------------
    # MMEngine lifecycle methods
    # ---------------------------------------------------------------------

    def before_run(self, runner: Runner):
        logger: MMLogger = runner.logger
        logger.info('[ConcurrentMLPTrainingHook] Initialising joint 38-D MLP model …')

        self.mlp_joint = JointMLPRefinementModel().to(self.device)
        self.opt_joint = optim.Adam(self.mlp_joint.parameters(), lr=self.mlp_lr, weight_decay=self.mlp_weight_decay)
        
        # Initialize scalers for 38-D input/output
        self.scaler_input = StandardScaler()
        self.scaler_target = StandardScaler()
        
        logger.info(f'[ConcurrentMLPTrainingHook] Joint MLP initialized with {sum(p.numel() for p in self.mlp_joint.parameters()):,} parameters')
        logger.info(f'[ConcurrentMLPTrainingHook] Hard-example threshold: {self.hard_example_threshold} pixels')

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
        logger.info('[ConcurrentMLPTrainingHook] Generating predictions for joint MLP on GPU...')

        def tensor_to_numpy(data):
            """Safely convert tensor to numpy, handling both tensor and numpy inputs."""
            if isinstance(data, torch.Tensor):
                return data.cpu().numpy()
            elif isinstance(data, np.ndarray):
                return data
            else:
                return np.array(data)

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
                    img_batch = []
                    gt_batch = []
                    
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
                            img_batch.append(img_array)
                            gt_batch.append(gt_keypoints)
                            
                            # Process batch when full or at the end
                            if len(img_batch) >= self.inference_batch_size or idx == len(df) - 1:
                                if not img_batch: continue
                                
                                # Run batch inference - process each image individually for compatibility
                                for i, (img_array, gt_keypoints) in enumerate(zip(img_batch, gt_batch)):
                                    try:
                                        bbox = np.array([[0, 0, 224, 224]], dtype=np.float32)
                                        results = inference_topdown(model, img_array, bboxes=bbox, bbox_format='xyxy')
                                        
                                        if results and len(results) > 0:
                                            pred_keypoints = tensor_to_numpy(results[0].pred_instances.keypoints[0])
                                            
                                            if pred_keypoints is None or pred_keypoints.shape[0] != 19:
                                                continue
                                            
                                            # Flatten coordinates to 38-D vectors
                                            pred_flat = pred_keypoints.flatten()
                                            gt_flat = gt_keypoints.flatten()
                                            
                                            # Calculate per-landmark radial errors
                                            landmark_errors = np.sqrt(np.sum((pred_keypoints - gt_keypoints)**2, axis=1))
                                            
                                            # Store data
                                            all_preds.append(pred_flat)
                                            all_gts.append(gt_flat)
                                            all_errors.append(landmark_errors)
                                    
                                    except Exception as e:
                                        logger.warning(f'[ConcurrentMLPTrainingHook] Failed to process batch item {i}: {e}')
                                        continue
                                
                                # Clear batches for next iteration
                                img_batch.clear()
                                gt_batch.clear()

                        except Exception as e:
                            logger.warning(f'[ConcurrentMLPTrainingHook] Failed to process sample {idx}: {e}')
                            img_batch.clear()
                            gt_batch.clear()
                            continue
                    
                    processed_count = len(all_preds)
                    logger.info(f'[ConcurrentMLPTrainingHook] Successfully processed {processed_count} samples from file')
                    
                except Exception as e:
                    logger.warning(f'[ConcurrentMLPTrainingHook] Failed to load from annotation file: {e}')
                    df = None
            
            # Method 2: Fallback to dataset iteration if file method fails
            if not all_preds:  # No predictions from file method
                logger.info('[ConcurrentMLPTrainingHook] Using dataset iteration method')
                
                processed_count = 0
                img_batch = []
                gt_batch = []
                
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
                            
                        img_batch.append(img_np)
                        gt_batch.append(gt_kpts)
                        
                        # Process batch when full or at the end
                        if len(img_batch) >= self.inference_batch_size or idx == len(train_dataset) - 1:
                            if not img_batch: continue
                        
                            # Run inference - process each image individually for compatibility
                            for i, (img_np, gt_kpts) in enumerate(zip(img_batch, gt_batch)):
                                try:
                                    bbox = np.array([[0, 0, 224, 224]], dtype=np.float32)
                                    results = inference_topdown(model, img_np, bboxes=bbox, bbox_format='xyxy')
                                    
                                    if results and len(results) > 0:
                                        pred_kpts = tensor_to_numpy(results[0].pred_instances.keypoints[0])
                                        
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
                                
                                except Exception as e:
                                    logger.warning(f'[ConcurrentMLPTrainingHook] Failed to process dataset batch item {i}: {e}')
                                    continue

                            # Clear batches for next iteration
                            img_batch.clear()
                            gt_batch.clear()
                        
                    except Exception as e:
                        # Only log every 100th error to avoid spam
                        if idx % 100 == 0:
                            logger.warning(f'[ConcurrentMLPTrainingHook] Failed to process sample {idx}: {e}')
                        img_batch.clear()
                        gt_batch.clear()
                        continue
                
                processed_count = len(all_preds)
                logger.info(f'[ConcurrentMLPTrainingHook] Successfully processed {processed_count} samples via dataset iteration')

        except Exception as e:
            logger.error(f'[ConcurrentMLPTrainingHook] Critical error during data processing: {e}')
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
        hard_examples = max_errors_per_sample > self.hard_example_threshold
        
        # Create sample weights: 1.0 for normal, 2.0 for hard examples
        sample_weights = np.ones(len(all_preds))
        sample_weights[hard_examples] = 2.0
        
        num_hard_examples = np.sum(hard_examples)
        logger.info(f'[ConcurrentMLPTrainingHook] Hard examples (>{self.hard_example_threshold}px): {num_hard_examples}/{len(all_preds)} ({num_hard_examples/len(all_preds)*100:.1f}%)')
        
        if num_hard_examples > 0:
            logger.info(f'[ConcurrentMLPTrainingHook] Hard example errors: min={np.min(max_errors_per_sample[hard_examples]):.2f}, max={np.max(max_errors_per_sample[hard_examples]):.2f}, mean={np.mean(max_errors_per_sample[hard_examples]):.2f}')

        # -----------------------------------------------------------------
        # Step 2: Initialize normalization scalers (only once)
        # -----------------------------------------------------------------
        if not self.scalers_initialized:
            logger.info('[ConcurrentMLPTrainingHook] Initializing normalization scalers for 38-D data...')
            
            # Fit scalers on the first batch of data
            self.scaler_input.fit(all_preds)
            self.scaler_target.fit(all_gts)
            
            self.scalers_initialized = True
            logger.info('[ConcurrentMLPTrainingHook] Normalization scalers initialized')
            
            # Save scalers for evaluation
            save_dir = os.path.join(runner.work_dir, 'concurrent_mlp')
            os.makedirs(save_dir, exist_ok=True)
            
            joblib.dump(self.scaler_input, os.path.join(save_dir, 'scaler_joint_input.pkl'))
            joblib.dump(self.scaler_target, os.path.join(save_dir, 'scaler_joint_target.pkl'))
            
            logger.info(f'[ConcurrentMLPTrainingHook] Joint scalers saved to {save_dir}')

        # Normalize data using the consistent scalers
        preds_scaled = self.scaler_input.transform(all_preds)
        gts_scaled = self.scaler_target.transform(all_gts)

        # -----------------------------------------------------------------
        # Step 3: Train joint MLP for fixed number of epochs (GPU-optimized)
        # -----------------------------------------------------------------
        logger.info('[ConcurrentMLPTrainingHook] Training joint 38-D MLP on GPU…')
        
        # Calculate initial loss before refinement for logging
        initial_loss = self.criterion(
            torch.from_numpy(preds_scaled).float(), 
            torch.from_numpy(gts_scaled).float()
        ).item()

        # Build dataset with hard-example oversampling
        ds_joint = _MLPDataset(preds_scaled, gts_scaled, sample_weights)
        dl_joint = data.DataLoader(ds_joint, batch_size=self.mlp_batch_size, shuffle=True, pin_memory=True)

        def _train_joint(model: JointMLPRefinementModel, optimiser: optim.Optimizer, loader: data.DataLoader, initial_loss: float):
            model.train()
            total_loss = 0.0
            for ep in range(self.mlp_epochs):
                epoch_loss = 0.0
                for preds_batch, gts_batch in loader:
                    preds_batch = preds_batch.to(self.device, non_blocking=True)
                    gts_batch = gts_batch.to(self.device, non_blocking=True)

                    optimiser.zero_grad()
                    outputs = model(preds_batch)
                    loss = self.criterion(outputs, gts_batch)
                    loss.backward()
                    optimiser.step()

                    epoch_loss += loss.item()
                
                total_loss = epoch_loss / len(loader)
                if (ep + 1) % 20 == 0:
                    improvement = ((initial_loss - total_loss) / initial_loss * 100) if initial_loss > 0 else 0
                    logger.info(f'[ConcurrentMLPTrainingHook] Joint-MLP epoch {ep+1}/{self.mlp_epochs} | MLP loss: {total_loss:.6f}, Initial loss: {initial_loss:.6f} ({improvement:+.1f}%)')

        _train_joint(self.mlp_joint, self.opt_joint, dl_joint, initial_loss)

        logger.info('[ConcurrentMLPTrainingHook] Finished joint MLP update for this HRNet epoch.')
        
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