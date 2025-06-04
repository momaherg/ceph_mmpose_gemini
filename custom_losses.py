import torch.nn as nn
from mmengine.model import BaseModule
from mmpose.registry import MODELS # Using MODELS from mmpose.registry

@MODELS.register_module()
class CombinedHeatmapLoss(BaseModule):
    def __init__(self, 
                 adaptive_wing_cfg, 
                 ohkm_cfg, 
                 ohkm_loss_weight=0.3,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
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