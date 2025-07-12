#!/usr/bin/env python3
"""
Debug script to investigate data structure and ANB angle calculation issues.
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse

# Add current directory to path for custom modules
sys.path.insert(0, os.getcwd())

def main():
    parser = argparse.ArgumentParser(description='Debug data and classification issues')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the data JSON file')
    args = parser.parse_args()
    
    if not os.path.exists(args.data_file):
        print(f"ERROR: Data file not found: {args.data_file}")
        return
    
    print("🔍 Debugging Data and Classification Issues")
    print("="*80)
    
    # Load data
    try:
        df = pd.read_json(args.data_file)
        print(f"✓ Loaded {len(df)} total samples")
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        return
    
    # Check columns
    print(f"\n📋 Data columns ({len(df.columns)}):")
    print(list(df.columns)[:20])  # Show first 20 columns
    
    # Check for 'class' column
    if 'class' in df.columns:
        print(f"\n✅ Found 'class' column")
        print(f"Class distribution:")
        print(df['class'].value_counts())
        print(f"Missing values: {df['class'].isna().sum()}")
    else:
        print(f"\n❌ No 'class' column found")
    
    # Check for 'set' column
    if 'set' in df.columns:
        print(f"\n📊 Set distribution:")
        print(df['set'].value_counts())
    
    # Import modules for ANB calculation
    try:
        import cephalometric_dataset_info
        import anb_classification_utils
        
        landmark_cols = cephalometric_dataset_info.original_landmark_cols
        print(f"\n🎯 Checking landmark columns...")
        
        # Check if landmark columns exist
        missing_cols = []
        for col in landmark_cols:
            if col not in df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            print(f"❌ Missing landmark columns: {missing_cols[:10]}...")  # Show first 10
        else:
            print(f"✅ All landmark columns present")
        
        # Test ANB calculation on test samples
        test_df = df[df['set'] == 'test'] if 'set' in df.columns else df.head(10)
        print(f"\n🧪 Testing ANB calculation on {len(test_df)} samples...")
        
        successful_anb = 0
        failed_anb = 0
        anb_values = []
        
        for idx, row in test_df.iterrows():
            try:
                # Extract landmarks
                gt_keypoints = []
                valid_gt = True
                
                for i in range(0, len(landmark_cols), 2):
                    x_col = landmark_cols[i]
                    y_col = landmark_cols[i+1]
                    
                    if x_col in row and y_col in row and pd.notna(row[x_col]) and pd.notna(row[y_col]):
                        x_val = row[x_col]
                        y_val = row[y_col]
                        gt_keypoints.append([x_val, y_val])
                    else:
                        gt_keypoints.append([0, 0])
                        valid_gt = False
                
                if not valid_gt:
                    failed_anb += 1
                    if failed_anb <= 3:
                        print(f"  Sample {idx}: Invalid landmarks")
                    continue
                
                gt_keypoints = np.array(gt_keypoints)
                
                # Calculate ANB angle
                anb_angle = anb_classification_utils.calculate_anb_angle(gt_keypoints)
                
                if anb_angle is not None and not np.isnan(anb_angle):
                    anb_values.append(anb_angle)
                    classification = anb_classification_utils.classify_from_anb_angle(anb_angle)
                    if isinstance(classification, np.ndarray):
                        classification = classification.item()
                    
                    successful_anb += 1
                    if successful_anb <= 3:
                        print(f"  Sample {idx}: ANB={anb_angle:.2f}°, Class={classification}")
                else:
                    failed_anb += 1
                    if failed_anb <= 3:
                        print(f"  Sample {idx}: ANB calculation returned None/NaN")
                        
                        # Debug specific landmarks
                        a_point = gt_keypoints[2]
                        nasion = gt_keypoints[1]
                        b_point = gt_keypoints[3]
                        print(f"    A-point: {a_point}")
                        print(f"    Nasion: {nasion}")
                        print(f"    B-point: {b_point}")
                        
            except Exception as e:
                failed_anb += 1
                if failed_anb <= 3:
                    print(f"  Sample {idx}: Exception: {e}")
        
        print(f"\n📊 ANB Calculation Summary:")
        print(f"  Successful: {successful_anb}/{len(test_df)}")
        print(f"  Failed: {failed_anb}/{len(test_df)}")
        
        if anb_values:
            print(f"\n📈 ANB Angle Statistics:")
            print(f"  Min: {np.min(anb_values):.2f}°")
            print(f"  Max: {np.max(anb_values):.2f}°")
            print(f"  Mean: {np.mean(anb_values):.2f}°")
            print(f"  Std: {np.std(anb_values):.2f}°")
            
            # Show classification distribution
            classifications = [anb_classification_utils.classify_from_anb_angle(a) for a in anb_values]
            class_counts = pd.Series(classifications).value_counts()
            print(f"\n🏷️  Classification Distribution (from ANB):")
            for cls, count in class_counts.items():
                class_name = ["Class I", "Class II", "Class III"][cls]
                print(f"  {class_name}: {count}")
        
    except ImportError as e:
        print(f"\n❌ Failed to import required modules: {e}")
    
    # Show sample data structure
    print(f"\n📄 Sample data structure (first row):")
    if len(df) > 0:
        first_row = df.iloc[0]
        print(f"  Number of fields: {len(first_row)}")
        
        # Check for Image field
        if 'Image' in first_row:
            img_data = first_row['Image']
            if isinstance(img_data, (list, np.ndarray)):
                print(f"  Image shape: {np.array(img_data).shape}")
            else:
                print(f"  Image type: {type(img_data)}")
        
        # Show first few landmark values
        print(f"\n  Sample landmark values:")
        for col in ['sella_x', 'sella_y', 'nasion_x', 'nasion_y', 'A point_x', 'A point_y']:
            if col in first_row:
                print(f"    {col}: {first_row[col]}")


if __name__ == "__main__":
    main() 