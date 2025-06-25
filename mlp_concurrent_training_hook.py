#!/usr/bin/env python3
"""
Concurrent MLP Training Hook for MMEngine / MMPose
-------------------------------------------------
This hook trains two MLP refinement models (one for X, one for Y) **concurrently**
with HRNetV2 training.  After every HRNet training epoch, the hook:

1.  Runs inference on the entire *training* dataloader using the *current*
    HRNetV2 weights to obtain predicted landmark coordinates.
2.  Creates an in-memory dataset of (predicted → ground-truth) coordinate pairs.
3.  Trains each MLP for a fixed number of epochs (default: 100).

Important design decisions:
•   **One-time initialisation** – MLP weights, optimisers and scalers are created
    once in `before_run` and *persist* across the whole HRNet training.
•   **No gradient leakage** – MLP training is completely detached from the
    HRNetV2 computation graph (`torch.no_grad()`), so gradients do **not**
    propagate back.
•   **CPU/GPU awareness** – Trains on GPU if available, else CPU.
•   **Lightweight aggregation** – Keeps everything in RAM; suitable for ≈1.5k
    images.

To enable this hook, add to your config:

```
custom_hooks = [
    dict(
        type='ConcurrentMLPTrainingHook',
        mlp_epochs=100,
        mlp_batch_size=16,
        mlp_lr=1e-5,
        mlp_weight_decay=1e-4,
        log_interval=20  # optional
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

# -----------------------------------------------------------------------------
#  MLP architecture (identical to the one used in train_mlp_refinement.py)
# -----------------------------------------------------------------------------

class MLPRefinementModel(nn.Module):
    """Simple 19→500→19 fully connected network with ReLU + dropout."""

    def __init__(self, input_dim: int = 19, hidden_dim: int = 500, output_dim: int = 19):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class _MLPDataset(data.Dataset):
    """In-memory dataset of predicted → ground-truth coordinates."""

    def __init__(self, preds: np.ndarray, gts: np.ndarray):
        # preds/gts shape: [N, 19]
        assert preds.shape == gts.shape
        self.preds = torch.from_numpy(preds).float()
        self.gts = torch.from_numpy(gts).float()

    def __len__(self):
        return self.preds.shape[0]

    def __getitem__(self, idx):
        return self.preds[idx], self.gts[idx]


# -----------------------------------------------------------------------------
#  Hook implementation
# -----------------------------------------------------------------------------

@HOOKS.register_module()
class ConcurrentMLPTrainingHook(Hook):
    """MMEngine hook that performs concurrent MLP refinement training."""

    priority = 'LOW'  # Run after default hooks

    def __init__(
        self,
        mlp_epochs: int = 100,
        mlp_batch_size: int = 16,
        mlp_lr: float = 1e-5,
        mlp_weight_decay: float = 1e-4,
        log_interval: int = 50,
    ) -> None:
        self.mlp_epochs = mlp_epochs
        self.mlp_batch_size = mlp_batch_size
        self.mlp_lr = mlp_lr
        self.mlp_weight_decay = mlp_weight_decay
        self.log_interval = log_interval

        # These will be initialised in before_run
        self.mlp_x: MLPRefinementModel | None = None
        self.mlp_y: MLPRefinementModel | None = None
        self.opt_x: optim.Optimizer | None = None
        self.opt_y: optim.Optimizer | None = None
        self.criterion = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------------------------------------------------------------------
    # MMEngine lifecycle methods
    # ---------------------------------------------------------------------

    def before_run(self, runner: Runner):
        logger: MMLogger = runner.logger
        logger.info('[ConcurrentMLPTrainingHook] Initialising MLP models …')

        self.mlp_x = MLPRefinementModel().to(self.device)
        self.mlp_y = MLPRefinementModel().to(self.device)
        self.opt_x = optim.Adam(self.mlp_x.parameters(), lr=self.mlp_lr, weight_decay=self.mlp_weight_decay)
        self.opt_y = optim.Adam(self.mlp_y.parameters(), lr=self.mlp_lr, weight_decay=self.mlp_weight_decay)

    def after_train_epoch(self, runner: Runner):
        """After each HRNetV2 epoch, train MLP on-the-fly using current predictions."""
        logger: MMLogger = runner.logger
        assert self.mlp_x is not None and self.mlp_y is not None

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
        preds_x: List[np.ndarray] = []
        preds_y: List[np.ndarray] = []
        gts_x:   List[np.ndarray] = []
        gts_y:   List[np.ndarray] = []

        train_dataset = runner.train_dataloader.dataset
        logger.info('[ConcurrentMLPTrainingHook] Generating predictions for MLP on GPU...')

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
                            
                            # Store coordinates
                            preds_x.append(pred_keypoints[:, 0])
                            preds_y.append(pred_keypoints[:, 1])
                            gts_x.append(gt_keypoints[:, 0])
                            gts_y.append(gt_keypoints[:, 1])
                            
                            processed_count += 1
                            
                        except Exception as e:
                            logger.warning(f'[ConcurrentMLPTrainingHook] Failed to process sample {idx}: {e}')
                            continue
                    
                    logger.info(f'[ConcurrentMLPTrainingHook] Successfully processed {processed_count} samples from file')
                    
                except Exception as e:
                    logger.warning(f'[ConcurrentMLPTrainingHook] Failed to load from annotation file: {e}')
                    df = None
            
            # Method 2: Fallback to dataset iteration if file method fails
            if not preds_x:  # No predictions from file method
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
                        
                        # Store coordinates
                        preds_x.append(pred_kpts[:, 0])
                        preds_y.append(pred_kpts[:, 1])
                        gts_x.append(gt_kpts[:, 0])
                        gts_y.append(gt_kpts[:, 1])
                        
                        processed_count += 1
                        
                    except Exception as e:
                        # Only log every 100th error to avoid spam
                        if idx % 100 == 0:
                            logger.warning(f'[ConcurrentMLPTrainingHook] Failed to process sample {idx}: {e}')
                        continue
                
                logger.info(f'[ConcurrentMLPTrainingHook] Successfully processed {processed_count} samples via dataset iteration')

        except Exception as e:
            logger.error(f'[ConcurrentMLPTrainingHook] Critical error during data processing: {e}')
            return

        if not preds_x:
            logger.warning('[ConcurrentMLPTrainingHook] No predictions generated; skipping MLP update.')
            return

        preds_x = np.stack(preds_x)
        preds_y = np.stack(preds_y)
        gts_x = np.stack(gts_x)
        gts_y = np.stack(gts_y)
        
        logger.info(f'[ConcurrentMLPTrainingHook] Generated predictions for {len(preds_x)} samples')

        # -----------------------------------------------------------------
        # Step 2: Train MLPs for fixed number of epochs (GPU-optimized)
        # -----------------------------------------------------------------
        logger.info('[ConcurrentMLPTrainingHook] Training MLPs on GPU…')

        # Build datasets and loaders (on-the-fly)
        ds_x = _MLPDataset(preds_x, gts_x)
        ds_y = _MLPDataset(preds_y, gts_y)
        dl_x = data.DataLoader(ds_x, batch_size=self.mlp_batch_size, shuffle=True, pin_memory=True)
        dl_y = data.DataLoader(ds_y, batch_size=self.mlp_batch_size, shuffle=True, pin_memory=True)

        def _train_one(model: MLPRefinementModel, optimiser: optim.Optimizer, loader: data.DataLoader, name: str):
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
                    logger.info(f'[ConcurrentMLPTrainingHook] {name} epoch {ep+1}/{self.mlp_epochs} loss: {total_loss:.6f}')

        _train_one(self.mlp_x, self.opt_x, dl_x, 'MLP-X')
        _train_one(self.mlp_y, self.opt_y, dl_y, 'MLP-Y')

        logger.info('[ConcurrentMLPTrainingHook] Finished MLP update for this HRNet epoch.')
        
        # Save MLP models after each epoch
        save_dir = os.path.join(runner.work_dir, 'concurrent_mlp')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save current epoch models
        current_epoch = runner.epoch + 1  # runner.epoch is 0-indexed
        mlp_x_epoch_path = os.path.join(save_dir, f'mlp_x_epoch_{current_epoch}.pth')
        mlp_y_epoch_path = os.path.join(save_dir, f'mlp_y_epoch_{current_epoch}.pth')
        
        torch.save(self.mlp_x.state_dict(), mlp_x_epoch_path)
        torch.save(self.mlp_y.state_dict(), mlp_y_epoch_path)
        
        # Also save as "latest" for easy access
        mlp_x_latest_path = os.path.join(save_dir, 'mlp_x_latest.pth')
        mlp_y_latest_path = os.path.join(save_dir, 'mlp_y_latest.pth')
        
        torch.save(self.mlp_x.state_dict(), mlp_x_latest_path)
        torch.save(self.mlp_y.state_dict(), mlp_y_latest_path)
        
        logger.info(f'[ConcurrentMLPTrainingHook] MLP models saved for epoch {current_epoch}')
        logger.info(f'[ConcurrentMLPTrainingHook] Latest models: {mlp_x_latest_path}, {mlp_y_latest_path}')

    # ---------------------------------------------------------------------
    # Optional: save MLP weights at end of run
    # ---------------------------------------------------------------------
    def after_run(self, runner: Runner):
        logger: MMLogger = runner.logger
        if self.mlp_x is None or self.mlp_y is None:
            return
        save_dir = os.path.join(runner.work_dir, 'concurrent_mlp')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.mlp_x.state_dict(), os.path.join(save_dir, 'mlp_x_final.pth'))
        torch.save(self.mlp_y.state_dict(), os.path.join(save_dir, 'mlp_y_final.pth'))
        logger.info(f'[ConcurrentMLPTrainingHook] Saved final MLP weights to {save_dir}') 