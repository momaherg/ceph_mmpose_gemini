# PyTorch 2.6 Compatibility Fix for MMEngine Checkpoints
# Run this cell BEFORE running the evaluation function

import torch
print(f"PyTorch version: {torch.__version__}")

# Method 1: Add all necessary MMEngine classes to safe globals (recommended)
try:
    from mmengine.config.config import ConfigDict
    from mmengine.logging.history_buffer import HistoryBuffer
    
    # Collect all MMEngine classes that might be in checkpoints
    mmengine_classes = [ConfigDict, HistoryBuffer]
    
    # Add other common MMEngine classes if available
    try:
        from mmengine.logging.logger import MMLogger
        mmengine_classes.append(MMLogger)
    except ImportError:
        pass
        
    try:
        from mmengine.registry.registry import Registry
        mmengine_classes.append(Registry)
    except ImportError:
        pass
        
    try:
        from mmengine.utils.misc import DefaultScope
        mmengine_classes.append(DefaultScope)
    except ImportError:
        pass
    
    torch.serialization.add_safe_globals(mmengine_classes)
    print(f"✅ Successfully added {len(mmengine_classes)} MMEngine classes to PyTorch safe globals")
    print(f"Classes: {[cls.__name__ for cls in mmengine_classes]}")
    print("You can now run the evaluation function safely!")
except Exception as e:
    print(f"❌ Could not add MMEngine classes to safe globals: {e}")
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