import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmpose.registry import MODELS # Using MODELS from mmpose.registry
import warnings # Import warnings module

@MODELS.register_module()
class CombinedHeatmapLoss(BaseModule):
    def __init__(self, 
                 adaptive_wing_cfg,
                 ohkm_cfg,
                 ohkm_target_indices: list[int] | None = None, # Indices for OHKM to target
                 ohkm_loss_weight=0.3,
                 init_cfg=None,
                 **kwargs): # Add **kwargs to catch unexpected arguments
        super().__init__(init_cfg=init_cfg)

        # Handle unexpected arguments from base configurations gracefully
        if 'use_target_weight' in kwargs:
            # This argument is typically for the loss function itself, not the container
            # It should be within adaptive_wing_cfg or ohkm_cfg if needed by those losses
            warnings.warn(
                "'use_target_weight' was passed to CombinedHeatmapLoss directly. "
                "This is likely from a base config and is not used by CombinedHeatmapLoss itself. "
                "Ensure it is correctly placed within 'adaptive_wing_cfg' or 'ohkm_cfg' if intended for those specific losses."
            )
        
        if 'loss_weight' in kwargs:
            # This top-level loss_weight is not used by CombinedHeatmapLoss.
            # The effective weighting is handled by ohkm_loss_weight and any loss_weight within the sub-configs.
            warnings.warn(
                f"A top-level 'loss_weight' ({kwargs['loss_weight']}) was passed to CombinedHeatmapLoss. "
                "This is likely from a base config and is NOT used by CombinedHeatmapLoss for overall scaling. "
                "The 'ohkm_loss_weight' parameter controls the OHKM component's contribution, "
                "and individual loss_weights within 'adaptive_wing_cfg' or 'ohkm_cfg' apply to those components."
            )
        
        self.adaptive_wing_loss = MODELS.build(adaptive_wing_cfg)
        self.ohkm_loss = MODELS.build(ohkm_cfg)
        self.ohkm_target_indices = ohkm_target_indices
        self.ohkm_loss_weight = ohkm_loss_weight
        # The adaptive_wing_loss is assumed to have an effective weight of 1.0 
        # relative to the weighted ohkm_loss. Its own internal 'loss_weight' 
        # in its config will also apply if specified.

    def forward(self, pred_heatmaps, gt_heatmaps, keypoint_weights):
        """
        Args:
            pred_heatmaps (Tensor): Predicted heatmaps.
            gt_heatmaps (Tensor): Ground truth heatmaps.
            keypoint_weights (Tensor): Keypoint weights.

        Returns:
            torch.Tensor: The combined loss value.
        """
        loss_aw = self.adaptive_wing_loss(pred_heatmaps, gt_heatmaps, keypoint_weights)

        keypoint_weights_for_ohkm = keypoint_weights.clone()
        if self.ohkm_target_indices is not None and keypoint_weights_for_ohkm.ndim == 2:
            # Create a mask for all keypoints, initially False (don't zero out)
            # Mask shape will be (num_keypoints,)
            num_keypoints = keypoint_weights_for_ohkm.size(1)
            indices_to_zero_out_mask = torch.ones(num_keypoints, dtype=torch.bool, device=keypoint_weights_for_ohkm.device)
            if self.ohkm_target_indices: # Ensure list is not empty
                indices_to_zero_out_mask[self.ohkm_target_indices] = False # These should NOT be zeroed out
            
            # Apply mask: zero out weights for keypoints not in ohkm_target_indices
            # The mask is True for columns to zero out
            keypoint_weights_for_ohkm[:, indices_to_zero_out_mask] = 0.0
        elif self.ohkm_target_indices is not None:
             warnings.warn(
                f"ohkm_target_indices is configured but keypoint_weights has an unexpected dimension: {keypoint_weights_for_ohkm.ndim}. "
                "Expected 2D (batch_size, num_keypoints). Skipping OHKM target filtering for this batch."
            )

        loss_ohkm = self.ohkm_loss(pred_heatmaps, gt_heatmaps, keypoint_weights_for_ohkm)
        
        # Ensure the individual losses are scalar tensors before combining.
        # Typically, loss functions in mmpose return a scalar tensor.
        
        total_loss = loss_aw + self.ohkm_loss_weight * loss_ohkm
        return total_loss 