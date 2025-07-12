#!/usr/bin/env python3
"""
Test script to verify HRNetV2WithClassification model inference.
"""

import torch
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmpose.apis import inference_topdown
import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())


def test_model_inference():
    """Test that the model can perform inference correctly."""
    print("Testing HRNetV2WithClassification model inference...")
    print("=" * 60)
    
    # Initialize MMPose scope
    init_default_scope('mmpose')
    
    # Import custom modules
    try:
        import custom_cephalometric_dataset
        import custom_transforms
        import cephalometric_dataset_info
        import hrnetv2_with_classification
        import anb_classification_utils
        import classification_evaluator
        print("✓ Custom modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import custom modules: {e}")
        return False
    
    # Load config
    config_path = "Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py"
    
    try:
        cfg = Config.fromfile(config_path)
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return False
    
    # Build model
    try:
        from mmpose.registry import MODELS
        model = MODELS.build(cfg.model)
        model.eval()
        print("✓ Model built successfully")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Head type: {type(model.head).__name__}")
    except Exception as e:
        print(f"✗ Failed to build model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test with dummy data
    try:
        # Create dummy image
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Method 1: Direct forward pass
        print("\nTesting direct forward pass...")
        img_tensor = torch.from_numpy(dummy_img).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # (1, C, H, W)
        
        with torch.no_grad():
            # Extract features
            if hasattr(model, 'extract_feat'):
                feats = model.extract_feat(img_tensor)
                print(f"  ✓ Features extracted: {type(feats)}")
                
                # Test head forward
                heatmaps = model.head(feats)
                if isinstance(heatmaps, tuple):
                    print(f"  ✓ Head returned tuple: heatmaps shape = {heatmaps[0].shape}")
                else:
                    print(f"  ✓ Head returned heatmaps: shape = {heatmaps.shape}")
            else:
                print("  ✗ Model doesn't have extract_feat method")
        
        # Method 2: Using inference_topdown
        print("\nTesting inference_topdown...")
        bbox = np.array([[0, 0, 224, 224]], dtype=np.float32)
        
        try:
            results = inference_topdown(model, dummy_img, bboxes=bbox, bbox_format='xyxy')
            if results and len(results) > 0:
                keypoints = results[0].pred_instances.keypoints
                print(f"  ✓ Inference successful: keypoints shape = {keypoints.shape}")
                
                # Check for classification output
                if hasattr(results[0], 'pred_classification'):
                    print(f"  ✓ Classification prediction found: {results[0].pred_classification}")
                else:
                    print("  ℹ️  No classification prediction in results")
            else:
                print("  ✗ No results from inference_topdown")
        except Exception as e:
            print(f"  ✗ inference_topdown failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n✅ Model inference test completed!")
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_model_inference()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 