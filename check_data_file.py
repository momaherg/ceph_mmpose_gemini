#!/usr/bin/env python3
"""
Data File Diagnostic Script
This script helps locate and examine the cephalometric data file.
"""

import os
import json
import pandas as pd

def check_data_file():
    """Check for data file existence and structure."""
    
    print("="*60)
    print("DATA FILE DIAGNOSTIC")
    print("="*60)
    
    # Possible data file locations
    possible_paths = [
        "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json",
        "/content/drive/MyDrive/train_data_pure_old_numpy.json",
        "train_data_pure_old_numpy.json",
        "data/train_data_pure_old_numpy.json",
        "../train_data_pure_old_numpy.json",
        "data/new_test.txt"  # Alternative text file
    ]
    
    print("🔍 Searching for data files...")
    found_files = []
    
    for path in possible_paths:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"✓ Found: {path} ({size:,} bytes)")
            found_files.append((path, size))
        else:
            print(f"✗ Not found: {path}")
    
    if not found_files:
        print("\n❌ No data files found!")
        print("\n💡 Possible solutions:")
        print("1. Check if Google Drive is mounted (if using Colab)")
        print("2. Verify the file path is correct")
        print("3. Upload the data file to the current directory")
        return None
    
    # Examine the largest file (most likely the main data file)
    main_file_path, main_file_size = max(found_files, key=lambda x: x[1])
    print(f"\n📊 Examining largest file: {main_file_path}")
    
    try:
        # Try to peek at the file structure
        if main_file_path.endswith('.json'):
            print("📋 JSON file detected - checking structure...")
            
            # Try to read just a small portion
            with open(main_file_path, 'r') as f:
                first_chars = f.read(1000)  # Read first 1000 characters
            
            print(f"📝 First 200 characters:")
            print(first_chars[:200])
            
            # Try to load the JSON
            try:
                df = pd.read_json(main_file_path)
                print(f"\n✅ Successfully loaded with pandas!")
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {list(df.columns)[:10]}...")
                print(f"   Sample data types: {df.dtypes.head()}")
                
                # Check for required columns
                required_cols = ['Image', 'set']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"⚠️  Missing columns: {missing_cols}")
                else:
                    print("✅ Required columns present")
                
                return main_file_path
                
            except Exception as e:
                print(f"❌ Failed to load with pandas: {e}")
                
                # Try alternative method
                try:
                    with open(main_file_path, 'r') as f:
                        data = json.load(f)
                    
                    df = pd.DataFrame(data)
                    print(f"✅ Successfully loaded with json.load()!")
                    print(f"   Shape: {df.shape}")
                    return main_file_path
                    
                except Exception as e2:
                    print(f"❌ Failed with json.load(): {e2}")
                    
        else:
            print("📋 Non-JSON file detected")
            with open(main_file_path, 'r') as f:
                first_lines = [f.readline().strip() for _ in range(5)]
            
            print("📝 First 5 lines:")
            for i, line in enumerate(first_lines, 1):
                print(f"   {i}: {line[:100]}...")
    
    except Exception as e:
        print(f"❌ Error examining file: {e}")
    
    print(f"\n💡 To use this file in training, update the data_file_path in train_concurrent_v5.py:")
    print(f"   data_file_path = \"{main_file_path}\"")
    
    return main_file_path

if __name__ == "__main__":
    check_data_file() 