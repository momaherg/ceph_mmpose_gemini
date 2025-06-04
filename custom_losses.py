import torch.nn as nn
from mmengine.model import BaseModule
from mmpose.registry import MODELS # Using MODELS from mmpose.registry
import warnings # Import warnings module

@MODELS.register_module()
class CombinedHeatmapLoss(BaseModule):
    def __init__(self, 
                 adaptive_wing_cfg, 
                 ohkm_cfg, 
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
        loss_ohkm = self.ohkm_loss(pred_heatmaps, gt_heatmaps, keypoint_weights)
        
        # Ensure the individual losses are scalar tensors before combining.
        # Typically, loss functions in mmpose return a scalar tensor.
        
        total_loss = loss_aw + self.ohkm_loss_weight * loss_ohkm
        return total_loss 