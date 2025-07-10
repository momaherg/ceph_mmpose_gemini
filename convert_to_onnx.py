#!/usr/bin/env python3
"""
Convert HRNetV2 + MLP models to ONNX format for CPU deployment
This script converts both the HRNet and MLP models to ONNX format while preserving accuracy.
"""

import os
import sys
import torch
import torch.nn as nn
import warnings
import numpy as np
import onnx
import onnxruntime as ort
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.apis import init_model
import argparse
import json

# Add current directory to path
sys.path.insert(0, os.getcwd())

# Suppress warnings
warnings.filterwarnings('ignore')

# Apply PyTorch safe loading fix
import functools
_original_torch_load = torch.load

def safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = safe_torch_load


def convert_hrnet_to_onnx(model, input_size, output_path, verify=True):
    """Convert HRNetV2 model to ONNX format."""
    print(f"\n🔄 Converting HRNet to ONNX...")
    print(f"📏 Input size: {input_size}")
    
    # Create dummy input
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, input_size[0], input_size[1])
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,  # Use opset 11 for better compatibility
        do_constant_folding=True,
        input_names=['input'],
        output_names=['heatmaps'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'heatmaps': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"✅ HRNet ONNX model saved to: {output_path}")
    
    # Verify the model if requested
    if verify:
        print(f"\n🔍 Verifying HRNet ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        # Test inference
        ort_session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # Compare with PyTorch output
        with torch.no_grad():
            torch_output = model(dummy_input)
            if isinstance(torch_output, dict):
                torch_output = torch_output['heatmaps']
            torch_output = torch_output.numpy()
        
        # Check similarity
        max_diff = np.max(np.abs(torch_output - ort_outputs[0]))
        print(f"✅ Maximum difference between PyTorch and ONNX: {max_diff:.6f}")
        
        if max_diff > 1e-3:
            print(f"⚠️  Warning: Large difference detected. Model may need recalibration.")
        else:
            print(f"✅ ONNX model verified successfully!")
    
    return True


def convert_mlp_to_onnx(mlp_path, output_path, verify=True):
    """Convert MLP refinement model to ONNX format."""
    print(f"\n🔄 Converting MLP to ONNX...")
    
    # Import MLP model class
    from mlp_concurrent_training_hook import JointMLPRefinementModel
    
    # Load MLP checkpoint
    checkpoint = torch.load(mlp_path, map_location='cpu')
    
    # Initialize model
    model_config = checkpoint.get('model_config', {
        'input_dim': 38,
        'hidden_dim': 500,
        'output_dim': 38
    })
    
    mlp_model = JointMLPRefinementModel(**model_config)
    mlp_model.load_state_dict(checkpoint['model_state_dict'])
    mlp_model.eval()
    
    # Create dummy input (38 coordinates)
    dummy_input = torch.randn(1, 38)
    
    # Export to ONNX
    torch.onnx.export(
        mlp_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['coordinates'],
        output_names=['refined_coordinates'],
        dynamic_axes={
            'coordinates': {0: 'batch_size'},
            'refined_coordinates': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"✅ MLP ONNX model saved to: {output_path}")
    
    # Verify the model if requested
    if verify:
        print(f"\n🔍 Verifying MLP ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        # Test inference
        ort_session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # Compare with PyTorch output
        with torch.no_grad():
            torch_output = mlp_model(dummy_input).numpy()
        
        # Check similarity
        max_diff = np.max(np.abs(torch_output - ort_outputs[0]))
        print(f"✅ Maximum difference between PyTorch and ONNX: {max_diff:.6f}")
        
        if max_diff > 1e-3:
            print(f"⚠️  Warning: Large difference detected. Model may need recalibration.")
        else:
            print(f"✅ ONNX model verified successfully!")
    
    # Save scalers if available
    if 'scalers' in checkpoint and checkpoint['scalers']:
        import joblib
        scalers_path = output_path.replace('.onnx', '_scalers.pkl')
        joblib.dump(checkpoint['scalers'], scalers_path)
        print(f"✅ Normalization scalers saved to: {scalers_path}")
    
    return True


def create_onnx_inference_config(deployment_dir, onnx_dir):
    """Create configuration file for ONNX inference."""
    
    # Load deployment info
    with open(os.path.join(deployment_dir, 'deployment_info.json'), 'r') as f:
        deployment_info = json.load(f)
    
    # Create ONNX inference config
    onnx_config = {
        'model_type': deployment_info['model_type'],
        'input_size': deployment_info['input_size'],
        'num_keypoints': deployment_info['num_keypoints'],
        'has_mlp_refinement': deployment_info['has_mlp_refinement'],
        'onnx_models': {
            'hrnet': 'hrnet_model.onnx',
            'mlp': 'mlp_model.onnx' if deployment_info['has_mlp_refinement'] else None
        },
        'preprocessing': {
            'mean': [123.675, 116.28, 103.53],  # ImageNet mean
            'std': [58.395, 57.12, 57.375],     # ImageNet std
            'bgr_to_rgb': True
        },
        'postprocessing': {
            'heatmap_size': (96, 96),  # From config
            'use_udp': True,  # Ultra-decoding for sub-pixel accuracy
            'flip_test': False  # Disable for speed on CPU
        }
    }
    
    config_path = os.path.join(onnx_dir, 'onnx_config.json')
    with open(config_path, 'w') as f:
        json.dump(onnx_config, f, indent=2)
    
    print(f"✅ ONNX inference config saved to: {config_path}")


def convert_to_onnx(args):
    """Main conversion function."""
    
    deployment_dir = args.deployment_dir
    onnx_dir = args.output_dir
    os.makedirs(onnx_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🚀 CONVERTING MODELS TO ONNX")
    print(f"{'='*70}")
    print(f"📁 Deployment directory: {deployment_dir}")
    print(f"📂 ONNX output directory: {onnx_dir}")
    
    # Check deployment package exists
    required_files = ['hrnet_published.pth', 'config.py', 'deployment_info.json']
    for file in required_files:
        if not os.path.exists(os.path.join(deployment_dir, file)):
            print(f"❌ ERROR: Required file not found: {file}")
            print(f"Please run prepare_model_for_deployment.py first")
            return
    
    # Load deployment info
    with open(os.path.join(deployment_dir, 'deployment_info.json'), 'r') as f:
        deployment_info = json.load(f)
    
    # Initialize MMPose
    init_default_scope('mmpose')
    
    # Import custom modules
    try:
        import custom_cephalometric_dataset
        import custom_transforms
        import cephalometric_dataset_info
        import mlp_concurrent_training_hook
        print("✓ Custom modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import custom modules: {e}")
        return
    
    # 1. Convert HRNet model
    print(f"\n📊 Converting HRNetV2 model...")
    
    # Load model
    config_path = os.path.join(deployment_dir, 'config.py')
    checkpoint_path = os.path.join(deployment_dir, 'hrnet_published.pth')
    
    model = init_model(config_path, checkpoint_path, device='cpu')
    model.eval()
    
    # Convert to ONNX
    hrnet_onnx_path = os.path.join(onnx_dir, 'hrnet_model.onnx')
    input_size = tuple(deployment_info['input_size'])
    
    success = convert_hrnet_to_onnx(model, input_size, hrnet_onnx_path, verify=args.verify)
    
    if not success:
        print(f"❌ Failed to convert HRNet model")
        return
    
    # 2. Convert MLP model if available
    mlp_path = os.path.join(deployment_dir, 'mlp_refinement.pth')
    if os.path.exists(mlp_path) and deployment_info['has_mlp_refinement']:
        print(f"\n📊 Converting MLP refinement model...")
        
        mlp_onnx_path = os.path.join(onnx_dir, 'mlp_model.onnx')
        success = convert_mlp_to_onnx(mlp_path, mlp_onnx_path, verify=args.verify)
        
        if not success:
            print(f"❌ Failed to convert MLP model")
            return
    else:
        print(f"\n⚠️  No MLP refinement model found, skipping MLP conversion")
    
    # 3. Create ONNX inference configuration
    create_onnx_inference_config(deployment_dir, onnx_dir)
    
    # 4. Create optimization script for further CPU optimization
    opt_script_path = os.path.join(onnx_dir, 'optimize_for_cpu.py')
    with open(opt_script_path, 'w') as f:
        f.write('''#!/usr/bin/env python3
"""
Optional: Further optimize ONNX models for CPU inference
This can reduce model size and improve inference speed.
"""

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Optimize HRNet model
hrnet_model = onnx.load('hrnet_model.onnx')
quantized_model = quantize_dynamic(
    'hrnet_model.onnx',
    'hrnet_model_quantized.onnx',
    weight_type=QuantType.QUInt8
)
print("✅ Quantized HRNet model saved")

# Optimize MLP model if exists
import os
if os.path.exists('mlp_model.onnx'):
    mlp_model = onnx.load('mlp_model.onnx')
    quantized_mlp = quantize_dynamic(
        'mlp_model.onnx',
        'mlp_model_quantized.onnx',
        weight_type=QuantType.QUInt8
    )
    print("✅ Quantized MLP model saved")

print("\\n📊 Quantization can reduce model size by ~75% with minimal accuracy loss")
print("📋 Use quantized models if inference speed is critical")
''')
    os.chmod(opt_script_path, 0o755)
    print(f"✅ CPU optimization script saved to: {opt_script_path}")
    
    print(f"\n{'='*70}")
    print(f"✅ ONNX CONVERSION COMPLETED")
    print(f"{'='*70}")
    print(f"📁 ONNX models directory: {onnx_dir}")
    print(f"📄 Contents:")
    print(f"   - hrnet_model.onnx: HRNet model for CPU")
    if deployment_info['has_mlp_refinement']:
        print(f"   - mlp_model.onnx: MLP refinement model")
        if os.path.exists(os.path.join(onnx_dir, 'mlp_model_scalers.pkl')):
            print(f"   - mlp_model_scalers.pkl: Normalization scalers")
    print(f"   - onnx_config.json: Inference configuration")
    print(f"   - optimize_for_cpu.py: Optional quantization script")
    
    print(f"\n📋 Next steps:")
    print(f"1. (Optional) Run optimize_for_cpu.py for quantized models")
    print(f"2. Copy ONNX directory to your CPU server")
    print(f"3. Use inference_cpu.py for predictions")
    print(f"\n💡 Tip: Quantized models are ~4x smaller and ~2x faster with <1% accuracy loss")


def main():
    parser = argparse.ArgumentParser(description='Convert models to ONNX format')
    parser.add_argument('--deployment_dir', type=str, default='deployment_package',
                        help='Directory containing deployment package (default: deployment_package)')
    parser.add_argument('--output_dir', type=str, default='onnx_models',
                        help='Output directory for ONNX models (default: onnx_models)')
    parser.add_argument('--verify', action='store_true', default=True,
                        help='Verify ONNX models after conversion (default: True)')
    parser.add_argument('--no-verify', dest='verify', action='store_false',
                        help='Skip ONNX model verification')
    
    args = parser.parse_args()
    
    # Check if onnxruntime is installed
    try:
        import onnxruntime
        print(f"✓ ONNX Runtime version: {onnxruntime.__version__}")
    except ImportError:
        print("❌ ERROR: onnxruntime not installed")
        print("Install with: pip install onnxruntime")
        return
    
    convert_to_onnx(args)


if __name__ == "__main__":
    main() 