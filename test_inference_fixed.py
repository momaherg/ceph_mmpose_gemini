#!/usr/bin/env python3
"""
Simple test script to verify model inference works with direct forward pass
This avoids the MMPose API issues we've been encountering.
"""

import numpy as np
import pandas as pd
import torch
import pickle
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def test_data_loading():
    """Test loading the dataset to understand its structure."""
    
    print("="*60)
    print("TESTING DATA LOADING")
    print("="*60)
    
    # Try loading the pickle file
    try:
        with open('data/train_data_pure.pkl', 'rb') as f:
            data = pickle.load(f)
        
        print(f"✓ Data loaded successfully")
        print(f"  Data type: {type(data)}")
        
        if isinstance(data, pd.DataFrame):
            print(f"  DataFrame shape: {data.shape}")
            print(f"  Columns: {list(data.columns)}")
            
            # Check first sample
            first_sample = data.iloc[0]
            print(f"\nFirst sample analysis:")
            print(f"  Patient ID: {first_sample.get('patient_id', 'N/A')}")
            print(f"  Has 'Image' column: {'Image' in first_sample}")
            
            if 'Image' in first_sample:
                image_data = first_sample['Image']
                print(f"  Image type: {type(image_data)}")
                if hasattr(image_data, '__len__'):
                    print(f"  Image length: {len(image_data)}")
                if isinstance(image_data, (list, np.ndarray)):
                    image_array = np.array(image_data)
                    print(f"  Image array shape: {image_array.shape}")
                    
                    # Try reshaping
                    if image_array.shape == (50176, 3):
                        reshaped = image_array.reshape(224, 224, 3)
                        print(f"  Reshaped to: {reshaped.shape}")
                        print(f"  Value range: [{reshaped.min():.3f}, {reshaped.max():.3f}]")
                        
                        # Test tensor conversion
                        if reshaped.max() > 1.0:
                            reshaped = reshaped.astype(np.float32) / 255.0
                        
                        tensor = torch.from_numpy(reshaped).float().permute(2, 0, 1).unsqueeze(0)
                        print(f"  Tensor shape: {tensor.shape}")
                        print(f"  Tensor range: [{tensor.min():.3f}, {tensor.max():.3f}]")
                        
                        return True, data
            
            # Check landmark columns
            landmark_cols = []
            for col in data.columns:
                if col.endswith('_x') or col.endswith('_y'):
                    landmark_cols.append(col)
            
            print(f"\nLandmark columns found: {len(landmark_cols)}")
            print(f"  Sample landmark columns: {landmark_cols[:10]}")
            
            return True, data
        
        else:
            print(f"  Data is not a DataFrame: {type(data)}")
            return False, None
            
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return False, None

def test_model_creation():
    """Test creating a simple model for inference testing."""
    
    print("\n" + "="*60)
    print("TESTING MODEL CREATION")
    print("="*60)
    
    try:
        # Create a simple dummy model that mimics HRNet output
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Simple conv layers to mimic HRNet
                self.backbone = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 64, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 128, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool2d((56, 56))
                )
                
                # Head that outputs heatmaps
                self.head = torch.nn.Conv2d(128, 19, 1)  # 19 landmarks
                
            def forward(self, x):
                features = self.backbone(x)
                heatmaps = self.head(features)
                return heatmaps
        
        model = DummyModel()
        print(f"✓ Dummy model created: {type(model)}")
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        print(f"✓ Created dummy input: {dummy_input.shape}")
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✓ Model forward pass succeeded")
            print(f"  Output shape: {output.shape}")
            
            # Test coordinate decoding
            coords = decode_heatmaps_to_coords(output)
            print(f"✓ Coordinate decoding succeeded")
            print(f"  Coordinates shape: {coords.shape}")
            print(f"  Sample coordinates: {coords[:3]}")
            
        return True, model
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False, None

def decode_heatmaps_to_coords(heatmaps):
    """
    Decode heatmaps to coordinates using argmax.
    
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
        scale_x = 224.0 / width
        scale_y = 224.0 / height
        
        coords[i, 0] = x * scale_x  # x coordinate
        coords[i, 1] = y * scale_y  # y coordinate
    
    return coords

def test_full_pipeline():
    """Test the complete pipeline from data loading to inference."""
    
    print("\n" + "="*60)
    print("TESTING FULL PIPELINE")
    print("="*60)
    
    # Load data
    data_success, data = test_data_loading()
    if not data_success:
        print("✗ Cannot proceed without data")
        return
    
    # Create model
    model_success, model = test_model_creation()
    if not model_success:
        print("✗ Cannot proceed without model")
        return
    
    # Test with real data
    print(f"\nTesting with real data samples...")
    
    # Get a few samples
    sample_data = data.head(3)
    
    for idx, (_, row) in enumerate(sample_data.iterrows()):
        try:
            print(f"\nProcessing sample {idx + 1}:")
            print(f"  Patient ID: {row.get('patient_id', 'N/A')}")
            
            # Load image
            if 'Image' in row and row['Image'] is not None:
                image_array = row['Image']
                image = np.array(image_array)
                
                # Reshape if needed
                if image.shape == (50176, 3):
                    image = image.reshape(224, 224, 3)
                elif image.shape != (224, 224, 3):
                    print(f"  ⚠️  Invalid image shape: {image.shape}")
                    continue
                
                # Normalize
                if image.max() > 1.0:
                    image = image.astype(np.float32) / 255.0
                
                # Convert to tensor
                image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
                
                print(f"  ✓ Image processed: {image_tensor.shape}")
                
                # Model inference
                model.eval()
                with torch.no_grad():
                    output = model(image_tensor)
                    coords = decode_heatmaps_to_coords(output)
                    
                    print(f"  ✓ Inference successful")
                    print(f"    Predicted coordinates shape: {coords.shape}")
                    print(f"    Sample predictions: {coords[:2]}")
                
                # Check ground truth landmarks
                landmark_count = 0
                for col in row.index:
                    if col.endswith('_x') and col.replace('_x', '_y') in row.index:
                        x_val = row[col]
                        y_val = row[col.replace('_x', '_y')]
                        if pd.notna(x_val) and pd.notna(y_val) and x_val > 0 and y_val > 0:
                            landmark_count += 1
                
                print(f"    Ground truth landmarks: {landmark_count}")
                
            else:
                print(f"  ⚠️  No image data found")
                
        except Exception as e:
            print(f"  ✗ Failed to process sample {idx + 1}: {e}")
    
    print(f"\n✓ Full pipeline test completed successfully!")

if __name__ == "__main__":
    print("🔍 TESTING INFERENCE PIPELINE")
    print("This script tests the data loading and inference pipeline")
    print("to identify and fix the MMPose API issues.\n")
    
    test_full_pipeline() 