#!/usr/bin/env python3
"""
Custom loss functions for cephalometric landmark detection.
Includes hybrid losses combining different strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpose.registry import MODELS
from mmpose.models.losses.heatmap_loss import AdaptiveWingLoss


@MODELS.register_module()
class AdaptiveWingOHKMHybridLoss(nn.Module):
    """
    Hybrid loss combining AdaptiveWingLoss with Online Hard Keypoint Mining.
    
    This loss function:
    1. Computes AdaptiveWingLoss for all keypoints
    2. Identifies the top-k hardest keypoints based on their loss values
    3. Applies additional weight to hard examples
    
    Args:
        topk (int): Number of hard keypoints to mine per sample. Default: 8
        ohkm_weight (float): Additional weight for hard keypoints. Default: 1.5
        alpha (float): AdaptiveWingLoss alpha parameter. Default: 2.1
        omega (float): AdaptiveWingLoss omega parameter. Default: 24.0
        epsilon (float): AdaptiveWingLoss epsilon parameter. Default: 1.0
        theta (float): AdaptiveWingLoss theta parameter. Default: 0.5
        use_target_weight (bool): Whether to use target weights. Default: True
        loss_weight (float): Overall loss weight. Default: 1.0
    """
    
    def __init__(self,
                 topk=8,
                 ohkm_weight=1.5,
                 alpha=2.1,
                 omega=24.0,
                 epsilon=1.0,
                 theta=0.5,
                 use_target_weight=True,
                 loss_weight=1.0):
        super().__init__()
        
        self.topk = topk
        self.ohkm_weight = ohkm_weight
        self.loss_weight = loss_weight
        self.use_target_weight = use_target_weight
        
        # Initialize base AdaptiveWingLoss
        self.adaptive_wing = AdaptiveWingLoss(
            alpha=alpha,
            omega=omega,
            epsilon=epsilon,
            theta=theta,
            use_target_weight=False,  # We'll handle weights ourselves
            loss_weight=1.0  # We'll apply final weight ourselves
        )
    
    def forward(self, output, target, target_weight=None):
        """
        Forward pass of the hybrid loss.
        
        Args:
            output (torch.Tensor): Predicted heatmaps of shape (B, K, H, W)
            target (torch.Tensor): Target heatmaps of shape (B, K, H, W)
            target_weight (torch.Tensor): Weights of shape (B, K, 1)
        
        Returns:
            torch.Tensor: Computed loss value
        """
        batch_size, num_keypoints, height, width = output.shape
        
        # Reshape for per-keypoint loss computation
        output_flat = output.reshape(batch_size, num_keypoints, -1)
        target_flat = target.reshape(batch_size, num_keypoints, -1)
        
        # Compute per-keypoint AdaptiveWingLoss
        per_keypoint_losses = []
        
        for k in range(num_keypoints):
            kpt_output = output[:, k:k+1, :, :]  # (B, 1, H, W)
            kpt_target = target[:, k:k+1, :, :]  # (B, 1, H, W)
            
            # Compute loss for this keypoint
            kpt_loss = self.adaptive_wing(kpt_output, kpt_target)
            per_keypoint_losses.append(kpt_loss)
        
        # Stack losses: (B, K)
        per_keypoint_losses = torch.stack(per_keypoint_losses, dim=1).squeeze(-1)
        
        # Apply target weights if provided
        if self.use_target_weight and target_weight is not None:
            target_weight = target_weight.squeeze(-1)  # (B, K)
            per_keypoint_losses = per_keypoint_losses * target_weight
        
        # Online Hard Keypoint Mining
        # Find top-k hardest keypoints per sample
        topk_losses, topk_indices = torch.topk(
            per_keypoint_losses, 
            min(self.topk, num_keypoints), 
            dim=1
        )
        
        # Create weight mask for hard keypoints
        ohkm_weights = torch.ones_like(per_keypoint_losses)
        
        # Apply additional weight to hard keypoints
        for b in range(batch_size):
            ohkm_weights[b, topk_indices[b]] = self.ohkm_weight
        
        # Apply OHKM weights
        weighted_losses = per_keypoint_losses * ohkm_weights
        
        # Compute final loss
        if self.use_target_weight and target_weight is not None:
            # Only average over valid keypoints
            valid_mask = target_weight > 0
            if valid_mask.sum() > 0:
                loss = weighted_losses[valid_mask].mean()
            else:
                loss = weighted_losses.mean()
        else:
            loss = weighted_losses.mean()
        
        return loss * self.loss_weight


@MODELS.register_module()
class FocalHeatmapLoss(nn.Module):
    """
    Focal loss adapted for heatmap regression.
    Focuses on hard-to-predict pixels in the heatmap.
    
    Args:
        alpha (float): Weighting factor for positive examples. Default: 0.25
        gamma (float): Focusing parameter. Default: 2.0
        use_target_weight (bool): Whether to use target weights. Default: True
        loss_weight (float): Overall loss weight. Default: 1.0
    """
    
    def __init__(self,
                 alpha=0.25,
                 gamma=2.0,
                 use_target_weight=True,
                 loss_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
    
    def forward(self, output, target, target_weight=None):
        """
        Forward pass of focal heatmap loss.
        
        Args:
            output (torch.Tensor): Predicted heatmaps (B, K, H, W)
            target (torch.Tensor): Target heatmaps (B, K, H, W)
            target_weight (torch.Tensor): Weights (B, K, 1)
        
        Returns:
            torch.Tensor: Computed loss
        """
        # MSE as base loss
        diff = (output - target) ** 2
        
        # Focal weighting based on prediction error
        # Higher weight for pixels with larger errors
        focal_weight = (1 + diff) ** self.gamma
        
        # Apply focal weighting
        focal_loss = focal_weight * diff
        
        # Apply alpha weighting for positive locations (where target > 0.1)
        positive_mask = target > 0.1
        alpha_weight = torch.where(positive_mask, 
                                  torch.tensor(self.alpha), 
                                  torch.tensor(1 - self.alpha))
        focal_loss = focal_loss * alpha_weight
        
        # Apply target weight if provided
        if self.use_target_weight and target_weight is not None:
            # Expand target_weight to match heatmap dimensions
            target_weight = target_weight.unsqueeze(-1)  # (B, K, 1, 1)
            focal_loss = focal_loss * target_weight
            
            # Average only over valid keypoints
            valid_mask = target_weight > 0
            if valid_mask.sum() > 0:
                loss = focal_loss[valid_mask.expand_as(focal_loss)].mean()
            else:
                loss = focal_loss.mean()
        else:
            loss = focal_loss.mean()
        
        return loss * self.loss_weight


@MODELS.register_module()
class CombinedTargetMSELoss(nn.Module):
    """
    Combined loss using both heatmap and coordinate regression.
    This helps with direct coordinate prediction alongside heatmap learning.
    
    Args:
        heatmap_weight (float): Weight for heatmap loss. Default: 1.0
        coord_weight (float): Weight for coordinate regression. Default: 0.5
        use_target_weight (bool): Whether to use target weights. Default: True
        loss_weight (float): Overall loss weight. Default: 1.0
    """
    
    def __init__(self,
                 heatmap_weight=1.0,
                 coord_weight=0.5,
                 use_target_weight=True,
                 loss_weight=1.0):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.coord_weight = coord_weight
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
    
    def forward(self, output, target, target_weight=None, keypoint_coords=None):
        """
        Forward pass combining heatmap and coordinate losses.
        
        Args:
            output (torch.Tensor): Predicted heatmaps (B, K, H, W)
            target (torch.Tensor): Target heatmaps (B, K, H, W)
            target_weight (torch.Tensor): Weights (B, K, 1)
            keypoint_coords (torch.Tensor): Ground truth coordinates (B, K, 2)
        
        Returns:
            torch.Tensor: Combined loss
        """
        # Heatmap MSE loss
        heatmap_loss = F.mse_loss(output, target, reduction='none')
        
        # Apply target weight if provided
        if self.use_target_weight and target_weight is not None:
            target_weight_expanded = target_weight.unsqueeze(-1)  # (B, K, 1, 1)
            heatmap_loss = heatmap_loss * target_weight_expanded
            
            valid_mask = target_weight_expanded > 0
            if valid_mask.sum() > 0:
                heatmap_loss = heatmap_loss[valid_mask.expand_as(heatmap_loss)].mean()
            else:
                heatmap_loss = heatmap_loss.mean()
        else:
            heatmap_loss = heatmap_loss.mean()
        
        # Coordinate regression loss (if coordinates provided)
        coord_loss = 0
        if keypoint_coords is not None and self.coord_weight > 0:
            # Extract predicted coordinates from heatmaps
            batch_size, num_keypoints, h, w = output.shape
            
            # Get predicted coordinates via argmax
            output_flat = output.view(batch_size, num_keypoints, -1)
            max_indices = output_flat.argmax(dim=-1)  # (B, K)
            
            pred_y = max_indices // w
            pred_x = max_indices % w
            pred_coords = torch.stack([pred_x, pred_y], dim=-1).float()  # (B, K, 2)
            
            # Normalize to [0, 1] range
            pred_coords[..., 0] /= (w - 1)
            pred_coords[..., 1] /= (h - 1)
            
            # Normalize ground truth coords
            gt_coords_norm = keypoint_coords.clone()
            gt_coords_norm[..., 0] /= (w - 1)
            gt_coords_norm[..., 1] /= (h - 1)
            
            # Compute coordinate loss
            coord_loss = F.mse_loss(pred_coords, gt_coords_norm, reduction='none')
            
            if self.use_target_weight and target_weight is not None:
                coord_loss = coord_loss * target_weight
                valid_mask = target_weight > 0
                if valid_mask.sum() > 0:
                    coord_loss = coord_loss[valid_mask.expand_as(coord_loss)].mean()
                else:
                    coord_loss = coord_loss.mean()
            else:
                coord_loss = coord_loss.mean()
        
        # Combine losses
        total_loss = (self.heatmap_weight * heatmap_loss + 
                     self.coord_weight * coord_loss)
        
        return total_loss * self.loss_weight


# Register all custom losses
print("✓ Custom loss functions registered:")
print("  - AdaptiveWingOHKMHybridLoss")
print("  - FocalHeatmapLoss")
print("  - CombinedTargetMSELoss") 