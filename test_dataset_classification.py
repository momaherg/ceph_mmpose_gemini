#!/usr/bin/env python3
"""
Test script to verify dataset ground truth classification generation.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

def test_dataset_classification():
    """Test that the dataset properly generates ground truth classifications."""
    print("Testing dataset classification generation...")
    print("=" * 60)
    
    # Create a minimal test dataset
    test_data = {
        'patient_id': [1, 2, 3],
        'Image': [
            (np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)).tolist(),
            (np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)).tolist(),
            (np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)).tolist()
        ],
        # Add landmark coordinates for Class I, II, III examples
        'sella_x': [50, 50, 50],
        'sella_y': [50, 50, 50],
        'nasion_x': [100, 100, 100],
        'nasion_y': [100, 100, 100],
        'A point_x': [120, 120, 110],  # Different A-point positions
        'A point_y': [120, 100, 120],
        'B point_x': [118, 110, 120],  # Different B-point positions for different classes
        'B point_y': [122, 120, 100],
        # Add other landmarks (with zeros for simplicity)
        **{f'{landmark}_{coord}': [0, 0, 0] for landmark in [
            'upper 1 tip', 'upper 1 apex', 'lower 1 tip', 'lower 1 apex',
            'ANS', 'PNS', 'Gonion ', 'Menton', 'ST Nasion', 'Tip of the nose',
            'Subnasal', 'Upper lip', 'Lower lip', 'ST Pogonion', 'gnathion'
        ] for coord in ['x', 'y']},
        'set': ['train', 'train', 'train']
    }
    
    # Create DataFrame
    df = pd.DataFrame(test_data)
    
    print(f"Created test dataset with {len(df)} samples")
    print("Landmark coordinates:")
    for i, row in df.iterrows():
        print(f"  Sample {i+1}:")
        print(f"    Nasion: ({row['nasion_x']}, {row['nasion_y']})")
        print(f"    A-point: ({row['A point_x']}, {row['A point_y']})")
        print(f"    B-point: ({row['B point_x']}, {row['B point_y']})")
    
    # Test ANB calculation directly
    try:
        from anb_classification_utils import calculate_anb_angle, classify_from_anb_angle, get_class_name
        
        print("\nTesting ANB calculation...")
        for i, row in df.iterrows():
            # Create landmarks array
            landmarks = np.zeros((1, 19, 2))
            landmarks[0, 1] = [row['nasion_x'], row['nasion_y']]  # Nasion
            landmarks[0, 2] = [row['A point_x'], row['A point_y']]  # A-point  
            landmarks[0, 3] = [row['B point_x'], row['B point_y']]  # B-point
            
            anb_angle = calculate_anb_angle(landmarks)
            classification = classify_from_anb_angle(anb_angle)
            class_name = get_class_name(classification.item())
            
            print(f"  Sample {i+1}: ANB = {anb_angle[0]:.2f}°, Class = {class_name}")
            
    except Exception as e:
        print(f"Error in ANB calculation: {e}")
        return False
    
    # Test dataset creation
    try:
        from custom_cephalometric_dataset import CustomCephalometricDataset
        
        print("\nTesting dataset creation...")
        dataset = CustomCephalometricDataset(
            data_df=df,
            pipeline=[],  # No pipeline for testing
            data_mode='topdown'
        )
        
        print(f"Dataset created with {len(dataset)} samples")
        
        # Check first few samples
        for i in range(min(3, len(dataset))):
            data_info = dataset.get_data_info(i)
            print(f"  Sample {i+1}:")
            print(f"    Has gt_classification: {'gt_classification' in data_info}")
            if 'gt_classification' in data_info:
                gt_class = data_info['gt_classification']
                if gt_class is not None:
                    class_name = get_class_name(gt_class)
                    print(f"    GT Classification: {class_name} (label: {gt_class})")
                else:
                    print(f"    GT Classification: None")
            else:
                print(f"    No gt_classification field found")
        
        print("\n✅ Dataset classification test passed!")
        return True
        
    except Exception as e:
        print(f"Error testing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dataset_classification()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 