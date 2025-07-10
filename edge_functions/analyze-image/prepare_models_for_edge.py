#!/usr/bin/env python3
"""
Prepare ONNX models and scalers for Supabase Edge Function deployment.

This script:
1. Checks ONNX model compatibility
2. Converts pickle scalers to JSON format
3. Creates a test inference to verify models work
4. Provides upload commands for Supabase
"""

import os
import json
import numpy as np
import argparse
from pathlib import Path

def check_onnx_model(model_path: str):
    """Check if ONNX model is valid and get info."""
    try:
        import onnx
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        
        # Get input/output info
        print(f"\n✅ Model: {os.path.basename(model_path)}")
        print("Inputs:")
        for input in model.graph.input:
            shape = [d.dim_value for d in input.type.tensor_type.shape.dim]
            print(f"  - {input.name}: {shape}")
        
        print("Outputs:")
        for output in model.graph.output:
            shape = [d.dim_value for d in output.type.tensor_type.shape.dim]
            print(f"  - {output.name}: {shape}")
            
        return True
    except Exception as e:
        print(f"❌ Error checking {model_path}: {e}")
        return False

def convert_scalers_to_json(scalers_pkl_path: str, output_json_path: str):
    """Convert pickle scalers to JSON format."""
    try:
        import joblib
        
        # Load pickle scalers
        scalers = joblib.load(scalers_pkl_path)
        
        # Convert to JSON-serializable format
        scalers_json = {}
        
        if 'input' in scalers:
            scalers_json['input'] = {
                'mean': scalers['input'].mean_.tolist(),
                'std': np.sqrt(scalers['input'].var_).tolist()
            }
        
        if 'target' in scalers:
            scalers_json['target'] = {
                'mean': scalers['target'].mean_.tolist(),
                'std': np.sqrt(scalers['target'].var_).tolist()
            }
        
        # Save as JSON
        with open(output_json_path, 'w') as f:
            json.dump(scalers_json, f, indent=2)
        
        print(f"\n✅ Scalers converted to: {output_json_path}")
        print(f"  - Input dim: {len(scalers_json.get('input', {}).get('mean', []))}")
        print(f"  - Target dim: {len(scalers_json.get('target', {}).get('mean', []))}")
        
        return True
    except Exception as e:
        print(f"❌ Error converting scalers: {e}")
        return False

def test_models(hrnet_path: str, mlp_path: str, scalers_json_path: str = None):
    """Test ONNX models with dummy input."""
    try:
        import onnxruntime as ort
        
        print("\n🧪 Testing models...")
        
        # Test HRNet
        hrnet_session = ort.InferenceSession(hrnet_path)
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        hrnet_output = hrnet_session.run(None, {'input': dummy_input})
        heatmaps = hrnet_output[0]
        print(f"✅ HRNet output shape: {heatmaps.shape}")
        
        # Decode dummy keypoints (simplified)
        keypoints = np.random.randn(1, 38).astype(np.float32)
        
        # Test MLP
        mlp_session = ort.InferenceSession(mlp_path)
        
        # Apply normalization if scalers available
        if scalers_json_path and os.path.exists(scalers_json_path):
            with open(scalers_json_path, 'r') as f:
                scalers = json.load(f)
            
            if 'input' in scalers:
                mean = np.array(scalers['input']['mean'])
                std = np.array(scalers['input']['std'])
                keypoints = (keypoints - mean) / std
        
        mlp_output = mlp_session.run(None, {'coordinates': keypoints})
        refined = mlp_output[0]
        print(f"✅ MLP output shape: {refined.shape}")
        
        print("\n✅ Models are compatible!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing models: {e}")
        return False

def generate_upload_commands(model_dir: str, bucket_name: str = "models"):
    """Generate Supabase upload commands."""
    print("\n📤 Supabase Upload Commands:")
    print("--------------------------------")
    
    files = ['hrnet_model.onnx', 'mlp_model.onnx', 'mlp_scalers.json']
    urls = {}
    
    for file in files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            print(f"\n# Upload {file}")
            print(f"supabase storage upload {bucket_name}/{file} {file_path}")
            
            # Generate expected URL
            url = f"https://YOUR_PROJECT.supabase.co/storage/v1/object/public/{bucket_name}/{file}"
            urls[file] = url
    
    print("\n\n📝 Environment Variables:")
    print("--------------------------------")
    print(f'supabase secrets set HRNET_MODEL_URL="{urls.get("hrnet_model.onnx", "")}"')
    print(f'supabase secrets set MLP_MODEL_URL="{urls.get("mlp_model.onnx", "")}"')
    if 'mlp_scalers.json' in urls:
        print(f'supabase secrets set MLP_SCALERS_URL="{urls.get("mlp_scalers.json", "")}"')
    
    print("\n\n🚀 Deploy Command:")
    print("--------------------------------")
    print("supabase functions deploy analyze-image")

def main():
    parser = argparse.ArgumentParser(description='Prepare models for edge function deployment')
    parser.add_argument('--onnx_dir', type=str, default='onnx_models',
                        help='Directory containing ONNX models')
    parser.add_argument('--scalers_pkl', type=str, default='deployment_package/mlp_refinement.pth',
                        help='Path to pickle scalers (in PyTorch checkpoint)')
    parser.add_argument('--output_dir', type=str, default='edge_models',
                        help='Output directory for prepared models')
    parser.add_argument('--bucket_name', type=str, default='models',
                        help='Supabase storage bucket name')
    parser.add_argument('--skip_test', action='store_true',
                        help='Skip model testing')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 Preparing Models for Edge Function Deployment")
    print("=" * 50)
    
    # Check ONNX models
    hrnet_path = os.path.join(args.onnx_dir, 'hrnet_model.onnx')
    mlp_path = os.path.join(args.onnx_dir, 'mlp_model.onnx')
    
    if not os.path.exists(hrnet_path):
        print(f"❌ HRNet model not found: {hrnet_path}")
        return
    
    if not os.path.exists(mlp_path):
        print(f"❌ MLP model not found: {mlp_path}")
        return
    
    # Check models
    print("\n📊 Checking ONNX Models...")
    check_onnx_model(hrnet_path)
    check_onnx_model(mlp_path)
    
    # Copy models to output directory
    import shutil
    shutil.copy2(hrnet_path, os.path.join(args.output_dir, 'hrnet_model.onnx'))
    shutil.copy2(mlp_path, os.path.join(args.output_dir, 'mlp_model.onnx'))
    
    # Convert scalers if available
    scalers_json_path = None
    if os.path.exists(args.scalers_pkl):
        print("\n🔄 Converting Scalers...")
        
        # Check if it's a PyTorch checkpoint
        if args.scalers_pkl.endswith('.pth'):
            try:
                import torch
                checkpoint = torch.load(args.scalers_pkl, map_location='cpu')
                
                if 'scalers' in checkpoint and checkpoint['scalers']:
                    # Save scalers temporarily
                    import joblib
                    temp_pkl = 'temp_scalers.pkl'
                    joblib.dump(checkpoint['scalers'], temp_pkl)
                    
                    scalers_json_path = os.path.join(args.output_dir, 'mlp_scalers.json')
                    convert_scalers_to_json(temp_pkl, scalers_json_path)
                    
                    os.remove(temp_pkl)
                else:
                    print("⚠️  No scalers found in checkpoint")
            except Exception as e:
                print(f"⚠️  Could not extract scalers from checkpoint: {e}")
        else:
            scalers_json_path = os.path.join(args.output_dir, 'mlp_scalers.json')
            convert_scalers_to_json(args.scalers_pkl, scalers_json_path)
    
    # Test models
    if not args.skip_test:
        test_models(
            os.path.join(args.output_dir, 'hrnet_model.onnx'),
            os.path.join(args.output_dir, 'mlp_model.onnx'),
            scalers_json_path
        )
    
    # Generate upload commands
    generate_upload_commands(args.output_dir, args.bucket_name)
    
    print(f"\n\n✅ Models prepared in: {args.output_dir}")
    print("\nNext steps:")
    print("1. Review the models in the output directory")
    print("2. Run the Supabase upload commands above")
    print("3. Set the environment variables")
    print("4. Deploy the edge function")

if __name__ == "__main__":
    main() 