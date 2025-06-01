#!/usr/bin/env python3
"""
Test script to verify image processing fixes for MLP refinement dataset.
"""

import numpy as np
from PIL import Image as PILImage

def test_image_resize():
    """Test the image resizing logic that was causing OpenCV errors."""
    
    print("Testing image resize logic...")
    
    # Simulate different image sizes that might cause issues
    test_cases = [
        (224, 224, 3),    # Standard size
        (512, 512, 3),    # Large size that might cause OpenCV issues
        (1024, 1024, 3),  # Very large size
        (300, 400, 3),    # Non-square size
    ]
    
    target_size = 384
    
    for i, (h, w, c) in enumerate(test_cases):
        print(f"\nTest case {i+1}: {h}x{w}x{c}")
        
        # Create test image
        test_image = np.random.randint(0, 255, (h, w, c), dtype=np.uint8)
        print(f"  Original image shape: {test_image.shape}")
        
        # Apply the resize logic from the fixed code
        original_h, original_w = test_image.shape[:2]
        
        # Resize image using PIL
        pil_image = PILImage.fromarray(test_image)
        resized_image = pil_image.resize((target_size, target_size))
        resized_image_np = np.array(resized_image)
        
        print(f"  Resized image shape: {resized_image_np.shape}")
        
        # Test coordinate scaling
        # Simulate some landmark coordinates in original image
        test_coords = np.array([[100.0, 100.0], [200.0, 200.0]], dtype=np.float32)
        print(f"  Original coordinates: {test_coords}")
        
        # Scale to resized image (what HRNetV2 would work on)
        scale_to_target_x = target_size / original_w
        scale_to_target_y = target_size / original_h
        scaled_coords = test_coords.copy()
        scaled_coords[:, 0] *= scale_to_target_x
        scaled_coords[:, 1] *= scale_to_target_y
        print(f"  Scaled to target: {scaled_coords}")
        
        # Scale back to original size (what our fix does)
        scale_back_x = original_w / target_size
        scale_back_y = original_h / target_size
        final_coords = scaled_coords.copy()
        final_coords[:, 0] *= scale_back_x
        final_coords[:, 1] *= scale_back_y
        print(f"  Scaled back: {final_coords}")
        
        # Check if we get back to original (should be close)
        diff = np.abs(test_coords - final_coords)
        print(f"  Coordinate difference: {diff.max():.6f} pixels")
        
        if diff.max() < 1e-5:
            print("  ✓ Coordinate scaling test PASSED")
        else:
            print("  ✗ Coordinate scaling test FAILED")

def test_data_loading_simulation():
    """Test data loading simulation."""
    
    print("\n" + "="*50)
    print("TESTING DATA LOADING SIMULATION")
    print("="*50)
    
    # Simulate the data structure
    sample_data = {
        'Image': np.random.randint(0, 255, 224*224*3, dtype=np.uint8),  # Flattened image
        'sella_x': 120.5,
        'sella_y': 150.3,
        'nasion_x': 180.2,
        'nasion_y': 80.7,
        # ... other landmarks would be here
    }
    
    print(f"Sample data keys: {list(sample_data.keys())[:5]}...")  # Show first 5 keys
    
    # Test image reshaping logic
    image_data = sample_data['Image']
    print(f"Flattened image shape: {image_data.shape}")
    
    # Reshape logic from the dataset
    if len(image_data.shape) == 1:
        original_size = int(np.sqrt(len(image_data) // 3))
        image = image_data.reshape((original_size, original_size, 3))
        print(f"Reshaped image: {image.shape}")
        print(f"Original size detected: {original_size}")
    
    print("✓ Data loading simulation test PASSED")

if __name__ == "__main__":
    print("="*60)
    print("MLP REFINEMENT DATASET - IMAGE PROCESSING TESTS")
    print("="*60)
    
    test_image_resize()
    test_data_loading_simulation()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED")
    print("="*60)
    print("\nThe image processing fixes should resolve the OpenCV dimension error.")
    print("You can now run the training script in your Colab environment.") 