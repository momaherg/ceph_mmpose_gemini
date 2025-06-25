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

        # Online buffers to accumulate batch-wise predictions/GT during the epoch
        self._buf_preds_x: List[np.ndarray] = []
        self._buf_preds_y: List[np.ndarray] = []
        self._buf_gts_x:   List[np.ndarray] = []
        self._buf_gts_y:   List[np.ndarray] = []

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

        # Reset buffers at the very start
        self._reset_epoch_buffers()

    def after_train_epoch(self, runner: Runner):
        """After each HRNetV2 epoch, train MLP on-the-fly using current predictions."""
        logger: MMLogger = runner.logger
        assert self.mlp_x is not None and self.mlp_y is not None

        # -----------------------------------------------------------------
        # NEW STEP 1: Use accumulated predictions from this epoch
        # -----------------------------------------------------------------

        if getattr(runner, 'rank', 0) != 0:
            # Only rank 0 trains the refinement models
            return

        if not self._buf_preds_x:
            logger.warning('[ConcurrentMLPTrainingHook] No accumulated predictions this epoch; skipping MLP update.')
            return

        preds_x = np.stack(self._buf_preds_x)
        preds_y = np.stack(self._buf_preds_y)
        gts_x   = np.stack(self._buf_gts_x)
        gts_y   = np.stack(self._buf_gts_y)

        logger.info(f'[ConcurrentMLPTrainingHook] Collected {len(preds_x)} samples during this epoch for MLP training')

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

        # -----------------------------------------------------------------
        # Reset buffers for next epoch
        # -----------------------------------------------------------------
        self._reset_epoch_buffers()

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

    # ---------------------------------------------------------------------
    # Epoch lifecycle helpers
    # ---------------------------------------------------------------------

    def before_train_epoch(self, runner: Runner):
        """Called by MMEngine at the very start of every epoch."""
        self._reset_epoch_buffers()

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------

    def _reset_epoch_buffers(self):
        """Clear temporary buffers that store per-sample predictions/GT."""
        self._buf_preds_x.clear(); self._buf_preds_y.clear()
        self._buf_gts_x.clear();   self._buf_gts_y.clear()

    # ---------------------------------------------------------------------
    # Online accumulation – executed every training iteration
    # ---------------------------------------------------------------------

    def after_train_iter(self, runner: Runner, batch_idx: int, data_batch=None, outputs=None):
        """Collect predictions and GT from the current mini-batch.

        We run a *second* forward pass in `predict` mode to obtain decoded
        keypoints.  This adds one extra forward per mini-batch but completely
        removes the expensive epoch-end full-dataset sweep, leading to a net
        speed-up because the images are already resident in GPU memory and we
        avoid data-loader overhead.
        """
        # Safety checks – only rank 0 accumulates to avoid duplicate data in DDP
        if getattr(runner, 'rank', 0) != 0:
            return

        if self.mlp_x is None or self.mlp_y is None:
            # Not initialised yet
            return

        if data_batch is None:
            return

        try:
            batch_inputs = data_batch.get('inputs', None)
            batch_data_samples = data_batch.get('data_samples', None)

            if batch_inputs is None or batch_data_samples is None:
                return

            model = runner.model
            # Handle potential DDP wrapping
            if hasattr(model, 'module'):
                model_to_use = model.module
            else:
                model_to_use = model

            # Switch to eval for deterministic predictions
            was_training = model_to_use.training
            model_to_use.eval()

            with torch.no_grad():
                pred_results = model_to_use(batch_inputs, batch_data_samples, mode='predict')

            # Restore original mode
            if was_training:
                model_to_use.train()

            # Iterate over batch
            for res, gt_sample in zip(pred_results, batch_data_samples):
                # Predictions
                if (not hasattr(res, 'pred_instances') or
                        not hasattr(res.pred_instances, 'keypoints')):
                    continue

                pred_kpts = res.pred_instances.keypoints  # shape (num_kpts, 2)
                if isinstance(pred_kpts, torch.Tensor):
                    pred_kpts = pred_kpts.detach().cpu().numpy()

                # Ground truth
                if (not hasattr(gt_sample, 'gt_instances') or
                        not hasattr(gt_sample.gt_instances, 'keypoints')):
                    continue

                gt_kpts = gt_sample.gt_instances.keypoints
                if isinstance(gt_kpts, torch.Tensor):
                    gt_kpts = gt_kpts.detach().cpu().numpy()

                if pred_kpts.shape[0] != 19 or gt_kpts.shape[0] != 19:
                    # Unexpected size – skip
                    continue

                self._buf_preds_x.append(pred_kpts[:, 0])
                self._buf_preds_y.append(pred_kpts[:, 1])
                self._buf_gts_x.append(gt_kpts[:, 0])
                self._buf_gts_y.append(gt_kpts[:, 1])

        except Exception as e:
            runner.logger.warning(f'[ConcurrentMLPTrainingHook] Failed to accumulate batch {batch_idx}: {e}') 