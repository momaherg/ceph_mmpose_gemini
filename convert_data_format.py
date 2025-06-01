#!/usr/bin/env python3
"""
Convert train_data_pure.pkl to train_data_pure_old_numpy.json format
for compatibility with evaluation scripts.
"""

import pandas as pd
import pickle
import json
import os

def convert_pickle_to_json():
    """Convert the pickle file to JSON format."""
    
    # Input and output paths
    pickle_path = "data/train_data_pure.pkl"
    json_output_paths = [
        "data/train_data_pure_old_numpy.json",
        "train_data_pure_old_numpy.json"  # Also create in current directory
    ]
    
    if not os.path.exists(pickle_path):
        print(f"ERROR: Input file not found: {pickle_path}")
        print("Please ensure you have the pickle file in the data/ directory")
        return False
    
    try:
        # Load pickle file
        print(f"Loading pickle file: {pickle_path}")
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        # Convert to DataFrame if it's not already
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            print(f"ERROR: Unexpected data type: {type(data)}")
            return False
        
        print(f"Loaded DataFrame with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check if required columns exist
        required_cols = ['set']  # At minimum we need the 'set' column for train/test/dev split
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"WARNING: Missing required columns: {missing_cols}")
            # If no 'set' column, create a simple split
            if 'set' not in df.columns:
                print("Creating train/dev/test split...")
                n = len(df)
                df['set'] = 'train'
                df.iloc[int(0.7*n):int(0.85*n), df.columns.get_loc('set')] = 'dev'
                df.iloc[int(0.85*n):, df.columns.get_loc('set')] = 'test'
        
        # Save to JSON files
        for json_path in json_output_paths:
            print(f"Saving to: {json_path}")
            # Ensure directory exists
            os.makedirs(os.path.dirname(json_path) if os.path.dirname(json_path) else '.', exist_ok=True)
            
            # Save as JSON
            df.to_json(json_path, orient='records', indent=2)
            print(f"✅ Successfully saved to {json_path}")
        
        # Print summary
        if 'set' in df.columns:
            print("\nData split summary:")
            print(df['set'].value_counts())
        
        return True
        
    except Exception as e:
        print(f"ERROR converting file: {e}")
        return False

def main():
    print("="*60)
    print("DATA FORMAT CONVERTER")
    print("="*60)
    print("Converting train_data_pure.pkl to JSON format...")
    
    success = convert_pickle_to_json()
    
    if success:
        print("\n✅ Conversion completed successfully!")
        print("You can now run the evaluation scripts.")
    else:
        print("\n❌ Conversion failed!")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main() 