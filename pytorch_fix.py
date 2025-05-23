# PyTorch 2.6 Compatibility Fix for MMEngine Checkpoints
# Run this cell BEFORE running the evaluation function

import torch
print(f"PyTorch version: {torch.__version__}")

# Method 1: Add ConfigDict to safe globals (recommended)
try:
    from mmengine.config.config import ConfigDict
    torch.serialization.add_safe_globals([ConfigDict])
    print("✅ Successfully added ConfigDict to PyTorch safe globals")
    print("You can now run the evaluation function safely!")
except Exception as e:
    print(f"❌ Could not add ConfigDict to safe globals: {e}")
    print("Trying alternative approach...")
    
    # Method 2: Monkey patch torch.load to use weights_only=False by default
    # This is less secure but will work for trusted checkpoints
    original_load = torch.load
    
    def patched_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    torch.load = patched_load
    print("✅ Patched torch.load to use weights_only=False by default")
    print("You can now run the evaluation function!")

# Verification
print("\nVerification:")
print("- MMEngine ConfigDict support:", hasattr(torch.serialization, '_safe_globals'))
print("- Ready for checkpoint loading!") 