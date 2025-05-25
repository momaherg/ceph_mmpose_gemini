"""
Quick Learning Rate Analysis Script for Cephalometric Training

This script analyzes the learning rate configuration and provides specific
recommendations based on the observed training behavior.
"""

import numpy as np
from mmengine.config import Config

def analyze_learning_rate_schedule():
    """Analyze the current learning rate schedule and provide recommendations."""
    
    print("="*60)
    print("LEARNING RATE ANALYSIS")
    print("="*60)
    
    # Load config
    config_path = "/content/ceph_mmpose_gemini/configs/hrnetv2/hrnetv2_w18_cephalometric_224x224.py"
    
    try:
        cfg = Config.fromfile(config_path)
        print("✓ Config loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return
    
    # Current settings
    current_lr = cfg.optim_wrapper.optimizer.lr
    optimizer_type = cfg.optim_wrapper.optimizer.type
    max_epochs = cfg.train_cfg.max_epochs
    batch_size = cfg.train_dataloader.batch_size
    
    print(f"\nCURRENT SETTINGS:")
    print(f"Learning Rate: {current_lr}")
    print(f"Optimizer: {optimizer_type}")
    print(f"Max Epochs: {max_epochs}")
    print(f"Batch Size: {batch_size}")
    
    # Analyze scheduler
    schedulers = cfg.param_scheduler
    print(f"\nSCHEDULER ANALYSIS:")
    for i, scheduler in enumerate(schedulers):
        print(f"  {i+1}. {scheduler['type']}")
        if scheduler['type'] == 'LinearLR':
            print(f"     Warmup: {scheduler['start_factor']} -> 1.0 over {scheduler['end']} iterations")
        elif scheduler['type'] == 'MultiStepLR':
            print(f"     Milestones: {scheduler['milestones']}, Gamma: {scheduler['gamma']}")
    
    # Calculate effective learning rates
    print(f"\nEFFECTIVE LEARNING RATES:")
    
    # After warmup (epoch 1-2)
    lr_after_warmup = current_lr
    print(f"  After warmup: {lr_after_warmup}")
    
    # After first milestone (epoch 40)
    lr_epoch_40 = current_lr * 0.1
    print(f"  Epoch 40+: {lr_epoch_40}")
    
    # After second milestone (epoch 55) 
    lr_epoch_55 = current_lr * 0.01
    print(f"  Epoch 55+: {lr_epoch_55}")
    
    # DIAGNOSIS
    print(f"\n🔍 DIAGNOSIS:")
    
    # Check if LR is too high
    if current_lr >= 5e-4:
        print(f"❌ PROBLEM: Learning rate {current_lr} is TOO HIGH for fine-tuning!")
        print(f"   Your model is likely overshooting the optimal weights.")
        print(f"   Evidence: Model predictions clustered at image boundaries (220+ pixels)")
        print(f"   Recommendation: Use 1e-4 to 2e-4 for fine-tuning pretrained models")
    
    # Check if warmup is appropriate
    if schedulers[0]['start_factor'] == 0.001:
        print(f"✓ Warmup factor is reasonable")
    else:
        print(f"⚠️  Warmup factor might need adjustment")
    
    # Check scheduler type
    multistep_present = any(s['type'] == 'MultiStepLR' for s in schedulers)
    if multistep_present:
        print(f"⚠️  MultiStepLR can be aggressive - consider CosineAnnealingLR")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"1. IMMEDIATE FIX:")
    print(f"   - Reduce learning rate to 1e-4 or 2e-4")
    print(f"   - This should fix the prediction clustering issue")
    
    print(f"\n2. IMPROVED SCHEDULE:")
    print(f"   - Use CosineAnnealingLR instead of MultiStepLR")
    print(f"   - More gradual decay prevents sudden performance drops")
    
    print(f"\n3. VALIDATION:")
    print(f"   - Add validation set monitoring")
    print(f"   - Stop training when validation performance plateaus")
    
    # Generate corrected config snippet
    print(f"\n📝 CORRECTED CONFIG SNIPPET:")
    print(f"""
# Corrected optimizer settings
optim_wrapper = dict(optimizer=dict(
    type='AdamW',
    lr=2e-4,  # Reduced from {current_lr}
    weight_decay=0.0001
))

# Improved scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='CosineAnnealingLR',  # Changed from MultiStepLR
        begin=0,
        end={max_epochs},
        by_epoch=True,
        T_max={max_epochs},
        eta_min=1e-6
    )
]
""")

if __name__ == "__main__":
    analyze_learning_rate_schedule() 