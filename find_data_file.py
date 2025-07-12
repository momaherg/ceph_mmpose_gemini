#!/usr/bin/env python3
"""
Helper script to find the correct data file path for evaluation.
"""

import os
import glob
import json

def find_json_files():
    """Find potential data files."""
    print("🔍 Looking for potential data JSON files...")
    print("="*60)
    
    # Common patterns for data files
    patterns = [
        "**/*train*.json",
        "**/*data*.json",
        "**/*cephalometric*.json",
        "*.json"
    ]
    
    found_files = set()
    
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        found_files.update(files)
    
    # Filter out common non-data files
    exclude_patterns = ['.venv/', '__pycache__/', 'node_modules/', '.git/', 'package']
    data_files = []
    
    for f in found_files:
        if not any(exc in f for exc in exclude_patterns):
            data_files.append(f)
    
    print(f"Found {len(data_files)} potential data files:\n")
    
    for i, f in enumerate(sorted(data_files)):
        size = os.path.getsize(f) / (1024 * 1024)  # Size in MB
        print(f"{i+1}. {f} ({size:.2f} MB)")
        
        # Try to peek into the file
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                if isinstance(data, list) and len(data) > 0:
                    print(f"   - Contains {len(data)} records")
                    if isinstance(data[0], dict):
                        keys = list(data[0].keys())[:5]
                        print(f"   - Sample keys: {keys}")
                        if 'Image' in data[0] and 'sella_x' in data[0]:
                            print(f"   ✅ This looks like the cephalometric dataset!")
        except:
            pass
        print()
    
    print("\n💡 Usage tip:")
    print("python evaluate_classification_model.py --data_file <path_to_json>")
    
    # Check common Google Colab paths
    print("\n📍 Common data locations:")
    colab_paths = [
        "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json",
        "train_data_pure_old_numpy.json",
        "../train_data_pure_old_numpy.json",
        "../../train_data_pure_old_numpy.json",
        os.path.expanduser("~/Desktop/train_data_pure_old_numpy.json"),
        os.path.expanduser("~/Downloads/train_data_pure_old_numpy.json"),
        os.path.expanduser("~/Documents/train_data_pure_old_numpy.json")
    ]
    
    for path in colab_paths:
        if os.path.exists(path):
            print(f"✅ Found: {path}")
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"   Size: {size:.2f} MB")
        else:
            print(f"❌ Not found: {path}")


if __name__ == "__main__":
    find_json_files() 