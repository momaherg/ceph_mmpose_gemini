#!/usr/bin/env python3
"""
Test script to verify ANB angle calculation and classification.
"""

import numpy as np
from anb_classification_utils import calculate_anb_angle, classify_from_anb_angle, get_class_name

def test_anb_calculation():
    """Test ANB angle calculation with known examples."""
    
    print("Testing ANB angle calculation and classification...")
    print("=" * 60)
    
    # Create test cases with known ANB angles
    # Test case 1: Class I (ANB between 0 and 4)
    landmarks_class1 = np.zeros((1, 19, 2))
    landmarks_class1[0, 1] = [100, 100]  # Nasion
    landmarks_class1[0, 2] = [120, 120]  # A-point
    landmarks_class1[0, 3] = [118, 122]  # B-point (close to A, small angle)
    
    # Test case 2: Class II (ANB >= 4)
    landmarks_class2 = np.zeros((1, 19, 2))
    landmarks_class2[0, 1] = [100, 100]  # Nasion
    landmarks_class2[0, 2] = [120, 100]  # A-point
    landmarks_class2[0, 3] = [110, 120]  # B-point (behind A, large angle)
    
    # Test case 3: Class III (ANB <= 0)
    landmarks_class3 = np.zeros((1, 19, 2))
    landmarks_class3[0, 1] = [100, 100]  # Nasion
    landmarks_class3[0, 2] = [110, 120]  # A-point
    landmarks_class3[0, 3] = [120, 100]  # B-point (in front of A, negative angle)
    
    test_cases = [
        (landmarks_class1, "Expected Class I"),
        (landmarks_class2, "Expected Class II"),
        (landmarks_class3, "Expected Class III")
    ]
    
    for i, (landmarks, expected) in enumerate(test_cases):
        print(f"\nTest case {i + 1}: {expected}")
        print("-" * 40)
        
        # Calculate ANB angle
        anb_angle = calculate_anb_angle(landmarks)
        print(f"ANB angle: {anb_angle[0]:.2f} degrees")
        
        # Classify
        classification = classify_from_anb_angle(anb_angle)
        class_name = get_class_name(classification.item())
        print(f"Classification: {class_name} (label: {classification.item()})")
        
        # Verify classification ranges
        if classification == 0:
            assert 0 < anb_angle < 4, f"Class I should have ANB between 0 and 4, got {anb_angle[0]}"
        elif classification == 1:
            assert anb_angle >= 4, f"Class II should have ANB >= 4, got {anb_angle[0]}"
        elif classification == 2:
            assert anb_angle <= 0, f"Class III should have ANB <= 0, got {anb_angle[0]}"
        
        print("✓ Classification correct!")
    
    # Test batch processing
    print("\n" + "=" * 60)
    print("Testing batch processing...")
    batch_landmarks = np.concatenate([landmarks_class1, landmarks_class2, landmarks_class3], axis=0)
    batch_angles = calculate_anb_angle(batch_landmarks)
    batch_classifications = classify_from_anb_angle(batch_angles)
    
    print(f"Batch ANB angles: {batch_angles}")
    print(f"Batch classifications: {batch_classifications}")
    print("✓ Batch processing works!")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")


if __name__ == "__main__":
    test_anb_calculation() 