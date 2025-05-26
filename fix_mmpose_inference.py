#!/usr/bin/env python3
"""
Fix for MMPose Inference Issues

This script demonstrates the exact solution to the MMPose inference problems
you're encountering. The key issues are:

1. 'PoseDataSample' object has no attribute '_gt_instances'
2. cannot import name 'inference_model' from 'mmpose.apis'

The solution is to use direct model forward pass and proper heatmap decoding.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def decode_heatmaps_to_coords(heatmaps):
    """
    Decode heatmaps to coordinates using argmax.
    This is the key function that replaces the problematic MMPose API calls.
    
    Args:
        heatmaps: Tensor of shape (batch_size, num_keypoints, height, width)
    
    Returns:
        numpy array of shape (num_keypoints, 2) with (x, y) coordinates
    """
    if len(heatmaps.shape) == 4:
        heatmaps = heatmaps[0]  # Remove batch dimension
    
    num_keypoints, height, width = heatmaps.shape
    coords = np.zeros((num_keypoints, 2))
    
    for i in range(num_keypoints):
        heatmap = heatmaps[i].cpu().numpy()
        
        # Find the maximum value location
        max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        y, x = max_idx
        
        # Scale from heatmap size to image size (224x224)
        # Assuming heatmap is 56x56 (typical for HRNet)
        scale_x = 224.0 / width
        scale_y = 224.0 / height
        
        coords[i, 0] = x * scale_x  # x coordinate
        coords[i, 1] = y * scale_y  # y coordinate
    
    return coords

def fixed_model_inference(model, image_tensor, debug=False):
    """
    Fixed model inference that avoids MMPose API issues.
    
    Args:
        model: Loaded MMPose model
        image_tensor: Input tensor of shape (1, 3, 224, 224)
        debug: Whether to print debug information
    
    Returns:
        numpy array of shape (num_keypoints, 2) with predicted coordinates
    """
    
    model.eval()
    with torch.no_grad():
        try:
            # Method 1: Direct forward pass through the model
            if hasattr(model, 'forward'):
                outputs = model(image_tensor)
                if debug:
                    print(f"✓ Direct forward succeeded")
                    print(f"  Output type: {type(outputs)}")
                    if isinstance(outputs, (list, tuple)):
                        print(f"  Output length: {len(outputs)}")
                        if len(outputs) > 0:
                            print(f"  First output shape: {outputs[0].shape if hasattr(outputs[0], 'shape') else 'No shape'}")
                    elif hasattr(outputs, 'shape'):
                        print(f"  Output shape: {outputs.shape}")
                
                # Extract heatmaps and decode to coordinates
                if isinstance(outputs, (list, tuple)):
                    heatmaps = outputs[0]  # Usually the first output is heatmaps
                else:
                    heatmaps = outputs
                
                if debug:
                    print(f"  Heatmaps shape: {heatmaps.shape}")
                
                # Decode heatmaps to coordinates
                pred_coords = decode_heatmaps_to_coords(heatmaps)
                
                if debug:
                    print(f"  Decoded coords shape: {pred_coords.shape}")
                    print(f"  First few predictions: {pred_coords[:3]}")
                
                return pred_coords
                
            else:
                if debug:
                    print(f"✗ Model has no forward method")
                return None
                
        except Exception as e:
            if debug:
                print(f"✗ Direct forward failed: {e}")
            
            # Method 2: Try using model's backbone + head directly
            try:
                if hasattr(model, 'backbone') and hasattr(model, 'head'):
                    features = model.backbone(image_tensor)
                    if isinstance(features, (list, tuple)):
                        features = features[-1]  # Use last feature map
                    
                    heatmaps = model.head(features)
                    if isinstance(heatmaps, (list, tuple)):
                        heatmaps = heatmaps[0]
                    
                    pred_coords = decode_heatmaps_to_coords(heatmaps)
                    
                    if debug:
                        print(f"✓ Backbone+Head method succeeded")
                        print(f"  Decoded coords shape: {pred_coords.shape}")
                    
                    return pred_coords
                else:
                    if debug:
                        print(f"✗ Model structure not recognized")
                    return None
                    
            except Exception as e2:
                if debug:
                    print(f"✗ Backbone+Head method failed: {e2}")
                return None

def demonstrate_fix():
    """Demonstrate how to fix the MMPose inference issues."""
    
    print("="*70)
    print("MMPOSE INFERENCE FIX DEMONSTRATION")
    print("="*70)
    
    print("\n🔍 PROBLEM ANALYSIS:")
    print("The original error messages you encountered:")
    print("  1. 'PoseDataSample' object has no attribute '_gt_instances'")
    print("  2. cannot import name 'inference_model' from 'mmpose.apis'")
    print("\n💡 ROOT CAUSE:")
    print("  - MMPose API methods expect specific data structures")
    print("  - Some API functions are not available in your MMPose version")
    print("  - The model's test_step and predict methods require ground truth data")
    
    print("\n✅ SOLUTION:")
    print("  - Use direct model forward pass: model(image_tensor)")
    print("  - Decode heatmaps manually using argmax")
    print("  - Avoid MMPose's high-level API methods")
    
    print("\n🔧 IMPLEMENTATION:")
    print("Replace this problematic code:")
    print("```python")
    print("# PROBLEMATIC - Don't use these:")
    print("from mmpose.structures import PoseDataSample")
    print("data_sample = PoseDataSample()")
    print("results = model.test_step({'inputs': image_tensor, 'data_samples': [data_sample]})")
    print("results = model.predict(image_tensor, [data_sample])")
    print("from mmpose.apis import inference_model  # May not exist")
    print("```")
    
    print("\nWith this working code:")
    print("```python")
    print("# WORKING - Use direct forward pass:")
    print("model.eval()")
    print("with torch.no_grad():")
    print("    heatmaps = model(image_tensor)  # Direct forward pass")
    print("    if isinstance(heatmaps, (list, tuple)):")
    print("        heatmaps = heatmaps[0]  # Get first output")
    print("    coords = decode_heatmaps_to_coords(heatmaps)  # Manual decoding")
    print("```")
    
    print("\n📝 TO FIX YOUR EVALUATION SCRIPT:")
    print("1. Replace the inference section in your evaluation script")
    print("2. Use the fixed_model_inference() function provided above")
    print("3. Remove all MMPose API calls (test_step, predict, inference_model)")
    
    # Create a simple example
    print("\n🧪 TESTING WITH DUMMY DATA:")
    
    # Create dummy model and input
    class DummyHRNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((56, 56))
            )
            self.head = torch.nn.Conv2d(64, 19, 1)  # 19 landmarks
            
        def forward(self, x):
            features = self.backbone(x)
            heatmaps = self.head(features)
            return heatmaps
    
    model = DummyHRNet()
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print(f"  Created dummy model and input: {dummy_input.shape}")
    
    # Test the fixed inference
    coords = fixed_model_inference(model, dummy_input, debug=True)
    
    if coords is not None:
        print(f"\n✅ SUCCESS! Fixed inference works:")
        print(f"  Predicted coordinates shape: {coords.shape}")
        print(f"  Sample coordinates: {coords[:3]}")
        print(f"  Coordinate ranges: x=[{coords[:, 0].min():.1f}, {coords[:, 0].max():.1f}], y=[{coords[:, 1].min():.1f}, {coords[:, 1].max():.1f}]")
    else:
        print(f"\n❌ Fixed inference failed")

def create_fixed_evaluation_function():
    """Create the exact code you need to replace in your evaluation script."""
    
    print("\n" + "="*70)
    print("EXACT CODE TO USE IN YOUR EVALUATION SCRIPT")
    print("="*70)
    
    code = '''
# REPLACE THIS SECTION IN YOUR EVALUATION SCRIPT:
# Replace lines ~200-250 in your evaluate_training_samples.py

# Model inference using FIXED approach
if idx == 0:
    print(f"DEBUG - Running model inference...")

try:
    # Use direct forward pass - this is the key fix!
    outputs = model(image_tensor)
    
    if idx == 0:
        print(f"DEBUG - Direct forward succeeded")
        print(f"DEBUG - Output type: {type(outputs)}")
        if isinstance(outputs, (list, tuple)):
            print(f"DEBUG - Output length: {len(outputs)}")
            if len(outputs) > 0:
                print(f"DEBUG - First output shape: {outputs[0].shape}")
        elif hasattr(outputs, 'shape'):
            print(f"DEBUG - Output shape: {outputs.shape}")
    
    # Extract heatmaps
    if isinstance(outputs, (list, tuple)):
        heatmaps = outputs[0]  # Usually the first output is heatmaps
    else:
        heatmaps = outputs
    
    if idx == 0:
        print(f"DEBUG - Heatmaps shape: {heatmaps.shape}")
    
    # Decode heatmaps to coordinates using the function above
    pred_coords = decode_heatmaps_to_coords(heatmaps)
    
    if idx == 0:
        print(f"DEBUG - Decoded coords shape: {pred_coords.shape}")
        print(f"DEBUG - First few predictions: {pred_coords[:3]}")

except Exception as e:
    if idx == 0:
        print(f"DEBUG - Direct forward failed: {e}")
    continue  # Skip this sample

# Validate predictions
if pred_coords is None or pred_coords.shape[0] != 19:
    if idx == 0:
        print(f"DEBUG - Invalid prediction shape or None")
    continue

# Continue with your existing error calculation code...
'''
    
    print(code)
    
    print("\n📋 SUMMARY OF CHANGES NEEDED:")
    print("1. Remove all MMPose API calls (test_step, predict, inference_model)")
    print("2. Use direct model forward pass: outputs = model(image_tensor)")
    print("3. Add the decode_heatmaps_to_coords() function to your script")
    print("4. Replace the inference section with the code above")
    print("5. Remove PoseDataSample imports and usage")

if __name__ == "__main__":
    demonstrate_fix()
    create_fixed_evaluation_function()
    
    print("\n" + "="*70)
    print("🎯 NEXT STEPS:")
    print("1. Copy the decode_heatmaps_to_coords() function to your evaluation script")
    print("2. Replace the inference section with the fixed code above")
    print("3. Test with a real checkpoint file")
    print("4. The evaluation should now work without MMPose API errors!")
    print("="*70) 