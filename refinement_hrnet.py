from typing import Dict, Optional

import torch
from torch import Tensor
from torchvision.ops import roi_align

from mmpose.registry import MODELS
from mmpose.models.pose_estimators.topdown import TopDownPoseEstimator
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.structures.bbox import get_udp_warp_matrix, get_warp_matrix
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptSampleList, PixelDataList, SampleList)

def _get_heatmaps_max_coords(heatmaps: Tensor) -> Tensor:
    """Get coordinates of heatmap maximums."""
    batch_size, num_joints, h, w = heatmaps.shape
    flat_heatmaps = heatmaps.view(batch_size, num_joints, -1)
    _, amax = torch.max(flat_heatmaps, 2)
    
    y_coords = (amax / w).view(batch_size, num_joints, 1).float()
    x_coords = (amax % w).view(batch_size, num_joints, 1).float()
    
    return torch.cat((x_coords, y_coords), dim=2)


@MODELS.register_module()
class RefinementHRNet(TopDownPoseEstimator):
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
        
        # refinement stage config
        self.patch_size = (32, 32)
        self.patch_offset = 0.5

    def forward(self,
                inputs: torch.Tensor,
                data_samples: SampleList,
                mode: str = 'tensor'):
        
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs)
        else:
            raise ValueError(f'Invalid mode "{mode}"')

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        # Stage 1: Base head loss
        features = self.extract_feat(inputs)
        base_head_loss = self.head.loss(features, data_samples)
        
        # Stage 2: Refinement head loss
        with torch.no_grad():
            # Get coarse coordinates from base heatmaps
            base_heatmaps = self.head.forward(features)
            
            # Flip heatmaps during training for TTA-like consistency
            if self.test_cfg.get('flip_test', False):
                flipped_features = self.extract_feat(torch.flip(inputs, [3]))
                flipped_heatmaps = self.head.forward(flipped_features)
                flipped_heatmaps = flip_heatmaps(
                    flipped_heatmaps,
                    flip_mode=self.test_cfg.get('flip_mode', 'heatmap'),
                    flip_indices=self.test_cfg['flip_indices'],
                    shift_heatmap=self.test_cfg.get('shift_heatmap', False))
                base_heatmaps = (base_heatmaps + flipped_heatmaps) * 0.5

            coarse_coords = _get_heatmaps_max_coords(base_heatmaps)
            
            # Normalize coarse coordinates to range [0, 1] for roi_align
            heatmap_size = base_heatmaps.shape[2:]
            coarse_coords_normalized = coarse_coords / torch.tensor(
                [heatmap_size[1], heatmap_size[0]], device=coarse_coords.device)

        # Get feature patches for refinement
        # The boxes for roi_align are (batch_idx, x1, y1, x2, y2)
        box_centers = coarse_coords_normalized
        
        # Create bounding boxes around the coarse predictions
        patch_size_norm_w = self.patch_size[1] / features[0].shape[3]
        patch_size_norm_h = self.patch_size[0] / features[0].shape[2]
        
        boxes = torch.cat([
            box_centers[:, :, 0] - patch_size_norm_w / 2,
            box_centers[:, :, 1] - patch_size_norm_h / 2,
            box_centers[:, :, 0] + patch_size_norm_w / 2,
            box_centers[:, :, 1] + patch_size_norm_h / 2
        ], dim=-1)

        # Assign each box to its corresponding image in the batch
        batch_size, num_kpts, _ = boxes.shape
        box_indices = torch.arange(batch_size, device=boxes.device).view(-1, 1).repeat(1, num_kpts)
        
        # Use roi_align to get feature patches
        patches = roi_align(
            features[0],
            boxes=torch.cat([box_indices.view(-1, 1), boxes.view(-1, 4)], dim=1),
            output_size=self.patch_size
        )

        # Reshape patches for the refine_head
        patches = patches.view(batch_size * num_kpts, -1, self.patch_size[0], self.patch_size[1])

        # Get refinement targets
        gt_coords = data_samples[0].gt_instances.keypoints
        gt_coords = gt_coords.view(batch_size, num_kpts, -1)
        
        # The refinement target is the offset from the coarse prediction
        # Note: coordinates need to be in the same space (pixels on heatmap)
        refine_targets = (gt_coords[:, :, :2] - coarse_coords)
        
        # Flatten targets for loss calculation
        refine_targets = refine_targets.view(batch_size * num_kpts, -1)
        
        # Run refinement head and calculate loss
        predicted_offsets = self.refine_head(patches)
        refine_head_loss = self.refine_head.loss(predicted_offsets, refine_targets)

        # Combine losses
        losses = dict()
        losses.update({'loss_base_head': base_head_loss['loss_kpt']})
        losses.update({'loss_refine_head': refine_head_loss})
        
        return losses

    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples."""
        # Stage 1: Base head prediction
        features = self.extract_feat(inputs)
        base_heatmaps = self.head.forward(features)
        
        # TTA: Flip test
        if self.test_cfg.get('flip_test', False):
             flipped_features = self.extract_feat(torch.flip(inputs, [3]))
             flipped_heatmaps = self.head.forward(flipped_features)
             flipped_heatmaps = flip_heatmaps(
                flipped_heatmaps,
                flip_mode=self.test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=self.test_cfg['flip_indices'],
                shift_heatmap=self.test_cfg.get('shift_heatmap', False))
             base_heatmaps = (base_heatmaps + flipped_heatmaps) * 0.5
        
        coarse_coords = _get_heatmaps_max_coords(base_heatmaps)
        
        # Normalize coarse coordinates for roi_align
        heatmap_size = base_heatmaps.shape[2:]
        coarse_coords_normalized = coarse_coords / torch.tensor(
            [heatmap_size[1], heatmap_size[0]], device=coarse_coords.device)
            
        # Get feature patches
        box_centers = coarse_coords_normalized
        patch_size_norm_w = self.patch_size[1] / features[0].shape[3]
        patch_size_norm_h = self.patch_size[0] / features[0].shape[2]
        boxes = torch.cat([
            box_centers[:, :, 0] - patch_size_norm_w / 2,
            box_centers[:, :, 1] - patch_size_norm_h / 2,
            box_centers[:, :, 0] + patch_size_norm_w / 2,
            box_centers[:, :, 1] + patch_size_norm_h / 2
        ], dim=-1)

        batch_size, num_kpts, _ = boxes.shape
        box_indices = torch.arange(batch_size, device=boxes.device).view(-1, 1).repeat(1, num_kpts)
        
        patches = roi_align(
            features[0],
            boxes=torch.cat([box_indices.view(-1, 1), boxes.view(-1, 4)], dim=1),
            output_size=self.patch_size
        )
        patches = patches.view(batch_size * num_kpts, -1, self.patch_size[0], self.patch_size[1])

        # Stage 2: Refinement
        predicted_offsets = self.refine_head.forward(patches)
        predicted_offsets = predicted_offsets.view(batch_size, num_kpts, -1)
        
        # Final coordinates
        final_coords = coarse_coords + predicted_offsets

        # Pack predictions into data samples
        # Note: This is a simplified packing logic. Depending on evaluation,
        # it might need more fields like scores.
        for i, data_sample in enumerate(data_samples):
            data_sample.pred_instances.keypoints = final_coords[i].cpu().numpy()

        return data_samples 