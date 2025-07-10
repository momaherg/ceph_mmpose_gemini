#!/usr/bin/env python3
"""
Script to prepare HRNetV2 + MLP ensemble model for deployment
This script:
1. Loads the HRNet checkpoint from model 2, epoch 99
2. Loads the corresponding MLP model
3. Publishes/simplifies the model by removing training-specific components
4. Saves a deployment-ready checkpoint
"""

import os
import sys
import torch
import warnings
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.apis import init_model
import argparse

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


def publish_model(model, checkpoint_path, output_path):
    """
    Publish/simplify the model for deployment.
    Removes training-specific components and optimizer states.
    """
    print(f"\n📦 Publishing model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract only model state dict (remove optimizer, scheduler, etc.)
    published_checkpoint = {
        'state_dict': checkpoint.get('state_dict', checkpoint),
        'meta': {
            'epoch': checkpoint.get('meta', {}).get('epoch', 99),
            'iter': checkpoint.get('meta', {}).get('iter', 0),
            'mmpose_version': checkpoint.get('meta', {}).get('mmpose_version', 'unknown'),
            'config': checkpoint.get('meta', {}).get('config', ''),
        }
    }
    
    # Remove any training-specific keys from state_dict
    state_dict = published_checkpoint['state_dict']
    keys_to_remove = []
    for key in state_dict.keys():
        if 'num_batches_tracked' in key:  # Remove batch norm tracking stats
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del state_dict[key]
    
    # Save published model
    torch.save(published_checkpoint, output_path)
    
    # Calculate size reduction
    original_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
    published_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    reduction = (1 - published_size/original_size) * 100
    
    print(f"✅ Published model saved to: {output_path}")
    print(f"📊 Size reduction: {original_size:.2f}MB → {published_size:.2f}MB ({reduction:.1f}% smaller)")
    
    return published_checkpoint


def load_mlp_model(mlp_path):
    """Load the concurrent MLP model."""
    print(f"\n🧠 Loading MLP model from: {mlp_path}")
    
    if not os.path.exists(mlp_path):
        print(f"⚠️  MLP model not found at: {mlp_path}")
        return None
    
    # Import the MLP model class
    from mlp_concurrent_training_hook import JointMLPRefinementModel
    
    # Load MLP checkpoint
    mlp_checkpoint = torch.load(mlp_path, map_location='cpu')
    
    # Initialize MLP model
    mlp_model = JointMLPRefinementModel(input_dim=38, hidden_dim=500, output_dim=38)
    
    # Load weights
    if 'model_state_dict' in mlp_checkpoint:
        mlp_model.load_state_dict(mlp_checkpoint['model_state_dict'])
    else:
        mlp_model.load_state_dict(mlp_checkpoint)
    
    mlp_model.eval()
    
    print(f"✅ MLP model loaded successfully")
    print(f"📊 MLP parameters: {sum(p.numel() for p in mlp_model.parameters()):,}")
    
    return mlp_model, mlp_checkpoint


def prepare_deployment_package(args):
    """Prepare complete deployment package with HRNet + MLP."""
    
    # Paths
    ensemble_dir = "work_dirs/hrnetv2_w18_cephalometric_ensemble_concurrent_mlp_v5"
    model_dir = os.path.join(ensemble_dir, f"model_{args.model_idx}")
    hrnet_checkpoint_path = os.path.join(model_dir, f"epoch_{args.epoch}.pth")
    mlp_dir = os.path.join(model_dir, "concurrent_mlp")
    mlp_checkpoint_path = os.path.join(mlp_dir, f"mlp_joint_epoch_{args.epoch}.pth")
    
    # Alternative MLP path if epoch-specific not found
    mlp_final_path = os.path.join(mlp_dir, "mlp_joint_final.pth")
    
    # Output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🚀 PREPARING DEPLOYMENT PACKAGE")
    print(f"{'='*70}")
    print(f"📁 Ensemble directory: {ensemble_dir}")
    print(f"🎯 Model index: {args.model_idx}")
    print(f"📍 Epoch: {args.epoch}")
    print(f"📂 Output directory: {output_dir}")
    
    # Check if files exist
    if not os.path.exists(hrnet_checkpoint_path):
        print(f"\n❌ ERROR: HRNet checkpoint not found at: {hrnet_checkpoint_path}")
        print(f"Available checkpoints in {model_dir}:")
        if os.path.exists(model_dir):
            for f in sorted(os.listdir(model_dir)):
                if f.endswith('.pth'):
                    print(f"  - {f}")
        return
    
    # 1. Publish HRNet model
    hrnet_output_path = os.path.join(output_dir, "hrnet_published.pth")
    published_checkpoint = publish_model(hrnet_checkpoint_path, hrnet_checkpoint_path, hrnet_output_path)
    
    # 2. Load and save MLP model
    mlp_model = None
    mlp_checkpoint = None
    
    # Try epoch-specific MLP first
    if os.path.exists(mlp_checkpoint_path):
        mlp_model, mlp_checkpoint = load_mlp_model(mlp_checkpoint_path)
    elif os.path.exists(mlp_final_path):
        print(f"⚠️  Epoch-specific MLP not found, using final MLP")
        mlp_model, mlp_checkpoint = load_mlp_model(mlp_final_path)
    else:
        print(f"⚠️  No MLP model found for model {args.model_idx}")
    
    if mlp_model is not None:
        # Save MLP model for deployment
        mlp_output_path = os.path.join(output_dir, "mlp_refinement.pth")
        mlp_deployment = {
            'model_state_dict': mlp_model.state_dict(),
            'model_config': {
                'input_dim': 38,
                'hidden_dim': 500,
                'output_dim': 38
            },
            'scalers': mlp_checkpoint.get('scalers', {})  # Include normalization scalers if available
        }
        torch.save(mlp_deployment, mlp_output_path)
        print(f"\n✅ MLP model saved to: {mlp_output_path}")
    
    # 3. Create self-contained configuration file
    config_path = "Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py"
    if os.path.exists(config_path):
        try:
            # Load the config to resolve all inheritance
            cfg = Config.fromfile(config_path)
            
            # Create a self-contained config file
            config_output_path = os.path.join(output_dir, "config.py")
            
            # Write the flattened config
            with open(config_output_path, 'w') as f:
                f.write('''#!/usr/bin/env python3
"""
Self-contained deployment configuration for cephalometric landmark detection.
This config has been flattened from the original training config to remove dependencies.
"""

# Core configuration converted from training config
''')
                # Convert the config to a dict and write it as Python code
                config_dict = cfg.to_dict()
                
                # Write the config as a Python dict
                f.write(f"# Flattened configuration\n")
                f.write(f"_config_dict = {repr(config_dict)}\n\n")
                
                # Add utility functions to rebuild the config
                f.write('''
# Rebuild config from dict
from mmengine.config import Config

# Create config object from the flattened dict
_cfg = Config(_config_dict)

# Make all config attributes available at module level
for key, value in _config_dict.items():
    globals()[key] = value

# Ensure the config object is available
config = _cfg
''')
            
            print(f"✅ Self-contained configuration created at: {config_output_path}")
            
        except Exception as e:
            print(f"⚠️  Failed to create self-contained config: {e}")
            print(f"📋 Copying original config file and base dependencies...")
            
            # Fallback: copy both the main config and base config
            import shutil
            config_output_path = os.path.join(output_dir, "config.py")
            shutil.copy2(config_path, config_output_path)
            
            # Also copy the base config
            base_config_path = "Pretrained_model/td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py"
            if os.path.exists(base_config_path):
                base_output_path = os.path.join(output_dir, "td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py")
                shutil.copy2(base_config_path, base_output_path)
                print(f"✅ Base configuration copied to: {base_output_path}")
            
            print(f"✅ Configuration files copied to: {config_output_path}")
    else:
        print(f"⚠️  Configuration file not found: {config_path}")
    
    # 4. Create deployment info
    deployment_info = {
        'model_type': 'hrnetv2_w18_concurrent_mlp',
        'ensemble_model_idx': args.model_idx,
        'epoch': args.epoch,
        'input_size': (384, 384),  # From config
        'num_keypoints': 19,
        'has_mlp_refinement': mlp_model is not None,
        'source_paths': {
            'hrnet_checkpoint': hrnet_checkpoint_path,
            'mlp_checkpoint': mlp_checkpoint_path if os.path.exists(mlp_checkpoint_path) else mlp_final_path,
            'config': config_path
        }
    }
    
    import json
    info_path = os.path.join(output_dir, "deployment_info.json")
    with open(info_path, 'w') as f:
        json.dump(deployment_info, f, indent=2)
    print(f"✅ Deployment info saved to: {info_path}")
    
    print(f"\n{'='*70}")
    print(f"📦 DEPLOYMENT PACKAGE READY")
    print(f"{'='*70}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Contents:")
    print(f"   - hrnet_published.pth: Published HRNet model")
    if mlp_model is not None:
        print(f"   - mlp_refinement.pth: MLP refinement model")
    print(f"   - config.py: Model configuration")
    print(f"   - deployment_info.json: Deployment metadata")
    
    print(f"\n📋 Next steps:")
    print(f"1. Run convert_to_onnx.py to convert models to ONNX format")
    print(f"2. Deploy ONNX models to your CPU server")
    print(f"3. Use inference_cpu.py for predictions")


def main():
    parser = argparse.ArgumentParser(description='Prepare model for deployment')
    parser.add_argument('--model_idx', type=int, default=2,
                        help='Model index from ensemble (default: 2)')
    parser.add_argument('--epoch', type=int, default=99,
                        help='Epoch checkpoint to use (default: 99)')
    parser.add_argument('--output_dir', type=str, default='deployment_package',
                        help='Output directory for deployment files (default: deployment_package)')
    
    args = parser.parse_args()
    
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
    
    # Prepare deployment package
    prepare_deployment_package(args)


if __name__ == "__main__":
    main() 