#!/usr/bin/env python3
"""
Quick fix script to resolve missing base configuration dependencies
Run this if you encounter FileNotFoundError when converting to ONNX
"""

import os
import shutil

def fix_config_dependencies():
    """Copy missing base configuration files to deployment package."""
    
    print("🔧 Fixing configuration dependencies...")
    
    deployment_dir = "deployment_package"
    
    # Check if deployment directory exists
    if not os.path.exists(deployment_dir):
        print(f"❌ Deployment directory not found: {deployment_dir}")
        print("Please run prepare_model_for_deployment.py first")
        return False
    
    # Copy the base config file
    base_config_source = "Pretrained_model/td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py"
    base_config_dest = os.path.join(deployment_dir, "td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py")
    
    if os.path.exists(base_config_source):
        if not os.path.exists(base_config_dest):
            shutil.copy2(base_config_source, base_config_dest)
            print(f"✅ Copied base config to: {base_config_dest}")
        else:
            print(f"✅ Base config already exists: {base_config_dest}")
    else:
        print(f"❌ Base config source not found: {base_config_source}")
        return False
    
    # Verify the config can be loaded
    try:
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        
        # Initialize scope
        init_default_scope('mmpose')
        
        # Try to load the config
        config_path = os.path.join(deployment_dir, "config.py")
        cfg = Config.fromfile(config_path)
        print(f"✅ Configuration loads successfully!")
        
    except Exception as e:
        print(f"⚠️  Configuration still has issues: {e}")
        print("You may need to run the updated prepare_model_for_deployment.py script")
        return False
    
    print("\n🎉 Configuration dependencies fixed!")
    print("You can now run: python convert_to_onnx.py")
    return True

if __name__ == "__main__":
    fix_config_dependencies() 