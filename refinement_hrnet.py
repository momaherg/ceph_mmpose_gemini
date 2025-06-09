from typing import Dict, Optional

import torch
from torch import Tensor
from torchvision.ops import roi_align
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS
from mmpose.models.pose_estimators import TopdownPoseEstimator
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.structures.bbox import get_warp_matrix
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptSampleList, PixelDataList, SampleList)
from mmengine.structures import InstanceData

def _get_heatmaps_max_coords(heatmaps: Tensor) -> Tensor:
    """Get coordinates of heatmap maximums."""
    batch_size, num_joints, h, w = heatmaps.shape
    flat_heatmaps = heatmaps.view(batch_size, num_joints, -1)
    _, amax = torch.max(flat_heatmaps, 2)
    
    y_coords = (amax / w).view(batch_size, num_joints, 1).float()
    x_coords = (amax % w).view(batch_size, num_joints, 1).float()
    
    return torch.cat((x_coords, y_coords), dim=2)


class OffsetRegressionHead(nn.Module):
    """A lightweight regression head that predicts (dx, dy) offsets from a
    spatial feature patch. It performs global average pooling followed by two
    fully connected layers.
    """
    def __init__(self, in_channels: int = 18, hidden_dim: int = 128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # (B, C, 1, 1)
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, 2)  # dx, dy

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.pool(x).flatten(1)  # (B, C)
        x = self.relu(self.fc1(x))   # (B, hidden_dim)
        offsets = self.fc2(x)        # (B, 2)
        return offsets

    def loss(self, pred_offsets: Tensor, tgt_offsets: Tensor, weight: Tensor) -> Tensor:
        # weight shape (B, 1) or (B,) -> broadcast to 2 dims
        if weight.dim() == 1:
            weight = weight.unsqueeze(1)
        loss = F.mse_loss(pred_offsets * weight, tgt_offsets * weight, reduction='mean')
        return loss

@MODELS.register_module()
class RefinementHRNet(TopdownPoseEstimator):
    """
    A two-stage HRNet model for landmark detection.
    Stage 1: A standard HRNet predicts coarse heatmaps.
    Stage 2: A refinement head takes feature patches around the coarse
             predictions and predicts a fine-tuned offset.
    """
    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 refine_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        
        super().__init__(backbone, neck, head, train_cfg, test_cfg,
                         data_preprocessor, init_cfg)

        if refine_head:
            self.refine_head = MODELS.build(refine_head)
        else:
            # Fallback to a simple offset regression head
            # For HRNetV2-w18 with concatenated features: 18+36+72+144=270 channels
            self.refine_head = OffsetRegressionHead(in_channels=270)
        
        # refinement stage config
        self.patch_size = (32, 32) # Must match decoder input_size in config

    def forward(self,
                inputs: torch.Tensor,
                data_samples: SampleList,
                mode: str = 'tensor'):
        
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            # This logic is inherited from the parent class, which calls _forward
            return self._forward(inputs, data_samples)
        else:
            raise ValueError(f'Invalid mode "{mode}"')

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        features = self.extract_feat(inputs)
        
        # --- Stage 1 Loss: Calculated by the standard heatmap head ---
        base_head_loss = self.head.loss(features, data_samples)

        # --- Stage 2: Refinement Head Loss ---
        with torch.no_grad():
            # Get coarse heatmap predictions from the base head
            base_heatmaps = self.head.forward(features)
            # Find the peak coordinates from these coarse heatmaps
            coarse_coords = _get_heatmaps_max_coords(base_heatmaps)
            
        # Extract feature patches using RoI Align
        feature_map = features[0] # Highest-resolution features
        
        # Normalize coarse coordinates to [0, 1] for RoI Align
        heatmap_size = base_heatmaps.shape[2:]
        coarse_coords_normalized = coarse_coords / torch.tensor(
            [heatmap_size[1], heatmap_size[0]],
            device=coarse_coords.device, dtype=torch.float32)

        box_centers = coarse_coords_normalized
        patch_size_norm_w = self.patch_size[1] / feature_map.shape[3]
        patch_size_norm_h = self.patch_size[0] / feature_map.shape[2]
        
        # Create bounding boxes for RoI Align
        boxes = torch.cat([
            box_centers[..., 0:1] - patch_size_norm_w / 2,
            box_centers[..., 1:2] - patch_size_norm_h / 2,
            box_centers[..., 0:1] + patch_size_norm_w / 2,
            box_centers[..., 1:2] + patch_size_norm_h / 2
        ], dim=-1)

        batch_size, num_kpts, _ = boxes.shape
        box_indices = torch.arange(
            batch_size, device=boxes.device).view(-1, 1).repeat(1, num_kpts)
        
        # Extract patches
        patches = roi_align(
            feature_map,
            boxes=torch.cat([box_indices.view(-1, 1), boxes.view(-1, 4)], dim=1),
            output_size=self.patch_size,
        )

        # --- Prepare targets for the refinement loss ---
        # Manually transform GT keypoints from image space to heatmap space
        # using the same affine transformation logic as the data pipeline.
        batch_gt_coords_hm = []
        batch_keypoint_weights = []
        for data_sample in data_samples:
            # Get transform parameters from metainfo
            center = data_sample.metainfo['input_center']
            scale = data_sample.metainfo['input_scale']
            rot = data_sample.metainfo.get('bbox_rotation', 0.0)

            # Get the ground-truth keypoints (in original image space)
            # This array is (num_kpts, 2) containing x, y coordinates
            gt_kpts_img = data_sample.gt_instances.keypoints[0]
            # Keypoint weights (visibility) are stored in a separate array
            keypoint_weights = data_sample.gt_instances.keypoints_visible[0]

            # Get the 2x3 affine transformation matrix
            trans = get_warp_matrix(
                center=center,
                scale=scale,
                rot=rot,
                output_size=heatmap_size,
                inv=False)

            # Apply the transform to GT keypoints to get them in heatmap space
            gt_kpts_xy = gt_kpts_img[:, :2]
            gt_kpts_homogeneous = np.concatenate(
                (gt_kpts_xy, np.ones((gt_kpts_xy.shape[0], 1))), axis=1)
            gt_coords_hm = (trans @ gt_kpts_homogeneous.T).T

            batch_gt_coords_hm.append(
                torch.from_numpy(gt_coords_hm).float().to(coarse_coords.device))
            
            batch_keypoint_weights.append(
                torch.from_numpy(keypoint_weights).float().to(coarse_coords.device))

        gt_coords = torch.stack(batch_gt_coords_hm)
        keypoint_weights = torch.stack(batch_keypoint_weights)
        
        # The target is the offset from the coarse prediction to the ground truth
        refine_targets = gt_coords - coarse_coords
        
        # Reshape targets and weights for the loss function
        refine_targets_flat = refine_targets.view(batch_size * num_kpts, 2)
        
        keypoint_weights_flat = keypoint_weights.view(batch_size * num_kpts, 1)

        # --- Run refinement head and calculate loss ---
        predicted_offsets = self.refine_head(patches)
        refine_head_loss = self.refine_head.loss(
            predicted_offsets, refine_targets_flat, keypoint_weights_flat)

        # Combine losses from both stages
        losses = dict()
        losses.update(base_head_loss)
        losses['loss_refine'] = refine_head_loss
        
        return losses

    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        """
        Predict results from a batch of inputs and data samples.
        This method is simplified and does not include TTA for the refinement stage.
        """
        # Stage 1: Get coarse predictions from the base head
        features = self.extract_feat(inputs)
        base_heatmaps = self.head.forward(features)
        
        # Test-Time Augmentation for the base heatmaps
        if self.test_cfg.get('flip_test', False):
             flipped_features = self.extract_feat(torch.flip(inputs, [3]))
             flipped_heatmaps = self.head.forward(flipped_features)
             flipped_heatmaps = flip_heatmaps(
                flipped_heatmaps,
                flip_mode=self.test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=data_samples[0].metainfo['flip_indices'],
                shift_heatmap=self.test_cfg.get('shift_heatmap', False))
             base_heatmaps = (base_heatmaps + flipped_heatmaps) * 0.5
        
        # Get coarse coordinates in heatmap space
        coarse_coords_hm = _get_heatmaps_max_coords(base_heatmaps)
        
        # Stage 2: Refine predictions
        # Normalize coarse coordinates for RoI Align
        heatmap_size = base_heatmaps.shape[2:]
        coarse_coords_normalized = coarse_coords_hm / torch.tensor(
            [heatmap_size[1], heatmap_size[0]],
            device=coarse_coords_hm.device, dtype=torch.float32)
            
        # Extract feature patches
        feature_map = features[0]
        box_centers = coarse_coords_normalized
        patch_size_norm_w = self.patch_size[1] / feature_map.shape[3]
        patch_size_norm_h = self.patch_size[0] / feature_map.shape[2]
        boxes = torch.cat([
            box_centers[..., 0:1] - patch_size_norm_w / 2,
            box_centers[..., 1:2] - patch_size_norm_h / 2,
            box_centers[..., 0:1] + patch_size_norm_w / 2,
            box_centers[..., 1:2] + patch_size_norm_h / 2
        ], dim=-1)
        batch_size, num_kpts, _ = boxes.shape
        box_indices = torch.arange(
            batch_size, device=boxes.device).view(-1, 1).repeat(1, num_kpts)
        
        patches = roi_align(
            feature_map,
            boxes=torch.cat([box_indices.view(-1, 1), boxes.view(-1, 4)], dim=1),
            output_size=self.patch_size
        )

        # Predict offsets from the patches
        predicted_offsets_hm = self.refine_head(patches)
        predicted_offsets_hm = predicted_offsets_hm.view(batch_size, num_kpts, 2)
        
        # Add the predicted offset to the coarse coordinates
        final_coords_hm = coarse_coords_hm + predicted_offsets_hm

        # Manually transform refined coordinates from heatmap space to image space
        batch_size, num_kpts, _ = final_coords_hm.shape
        heatmap_size = base_heatmaps.shape[2:]

        for i, data_sample in enumerate(data_samples):
            # Get transform parameters from metainfo
            center = data_sample.metainfo['input_center']
            scale = data_sample.metainfo['input_scale']
            rot = data_sample.metainfo.get('bbox_rotation', 0.0)

            # Get inverse affine transform matrix
            trans = get_warp_matrix(
                center=center,
                scale=scale,
                rot=rot,
                output_size=heatmap_size,
                inv=True)

            # Apply inverse transform to get coordinates in original image space
            coords_hm_i = final_coords_hm[i].cpu().numpy()
            coords_homo = np.concatenate(
                (coords_hm_i, np.ones((num_kpts, 1))), axis=1)
            coords_img = (trans @ coords_homo.T).T[:, :2]

            # Get scores from the max value in the coarse heatmaps
            scores = torch.amax(
                base_heatmaps[i].view(num_kpts, -1), dim=1).cpu().numpy()

            # Pack results into a new InstanceData object, adding the instance dimension
            pred_instances = InstanceData()
            pred_instances.keypoints = np.expand_dims(coords_img, axis=0) # Shape: (1, K, 2)
            pred_instances.keypoint_scores = np.expand_dims(scores, axis=0) # Shape: (1, K)
            
            # Set the pred_instances attribute of the data_sample
            data_sample.pred_instances = pred_instances

        return data_samples 