#!/usr/bin/env python3
"""
Quick Start Script for Cephalometric Training
Checks data file and provides ready-to-run commands.
"""

import os
import sys

def find_data_file():
    """Find the data file and return its path."""
    possible_paths = [
        "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json",
        "/content/drive/MyDrive/train_data_pure_old_numpy.json", 
        "train_data_pure_old_numpy.json",
        "data/train_data_pure_old_numpy.json",
        "../train_data_pure_old_numpy.json"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def main():
    print("="*70)
    print("🚀 CEPHALOMETRIC TRAINING QUICK START")
    print("="*70)
    
    # Check for data file
    data_file = find_data_file()
    
    if data_file:
        print(f"✅ Data file found: {data_file}")
        file_size = os.path.getsize(data_file)
        print(f"   Size: {file_size:,} bytes")
        
        print(f"\n🎯 READY TO TRAIN! Use these commands:")
        print(f"\n1️⃣  Standard HRNet training (baseline):")
        print(f"   python train_concurrent_v5.py --disable-mlp --data-file \"{data_file}\"")
        
        print(f"\n2️⃣  Concurrent MLP training (enhanced):")
        print(f"   python train_concurrent_v5.py --data-file \"{data_file}\"")
        
        print(f"\n📊 EVALUATION commands:")
        print(f"\n3️⃣  Evaluate HRNet only:")
        print(f"   python evaluate_concurrent_mlp.py --hrnet-only --data-file \"{data_file}\"")
        
        print(f"\n4️⃣  Evaluate with MLP refinement:")
        print(f"   python evaluate_concurrent_mlp.py --data-file \"{data_file}\"")
        
        print(f"\n💡 TIPS:")
        print(f"   • Start with HRNet-only training to establish baseline")
        print(f"   • Then try concurrent MLP training for potential improvements")
        print(f"   • Use same --data-file path for both training and evaluation")
        print(f"   • Training uses random split: 200 test, 100 validation, rest training")
        
    else:
        print("❌ No data file found!")
        print(f"\n🔍 Run this to diagnose the issue:")
        print(f"   python check_data_file.py")
        
        print(f"\n💡 Expected file locations:")
        possible_paths = [
            "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json",
            "train_data_pure_old_numpy.json", 
            "data/train_data_pure_old_numpy.json"
        ]
        for path in possible_paths:
            print(f"   • {path}")
        
        print(f"\n🔧 If your file is elsewhere, use:")
        print(f"   python train_concurrent_v5.py --data-file \"/path/to/your/data.json\"")

if __name__ == "__main__":
    main() 