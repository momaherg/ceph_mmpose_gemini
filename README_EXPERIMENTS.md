# Cephalometric Landmark Detection Experiments

This repository contains experiments to improve cephalometric landmark detection accuracy using different loss functions and resolutions.

## 📊 Current Baseline (V4)
- **Model**: HRNetV2-W18 with AdaptiveWingLoss
- **Resolution**: 384×384
- **Overall MRE**: 2.348 pixels
- **Sella MRE**: 4.674 pixels
- **Gonion MRE**: 4.281 pixels

## 🧪 Experiments Overview

### Experiment A: AdaptiveWing + OHKM Hybrid ✅ Ready
Combines AdaptiveWingLoss with Online Hard Keypoint Mining to focus on difficult landmarks.
- **Target**: <2.3 pixels MRE
- **Expected**: 10%+ improvement on Sella/Gonion

### Experiment B: FocalHeatmapLoss 🔄 Pending
Adapts focal loss for heatmap regression to focus on hard-to-predict pixels.
- **Target**: <2.25 pixels MRE
- **Expected**: 3-7% overall improvement

### Experiment C: OHKMMSELoss 🔄 Pending
Pure OHKM with MSE loss and larger sigma for smoother gradients.
- **Target**: <2.3 pixels MRE
- **Expected**: 2-4% overall improvement

### Experiment D: CombinedTargetMSE 512×512 🔄 Pending
Higher resolution with combined heatmap and coordinate regression.
- **Target**: <2.15 pixels MRE
- **Expected**: 5-10% improvement from resolution

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure you have the following installed:
- Python 3.8+
- PyTorch with CUDA support
- MMPose and dependencies
- Custom modules (already in project)
```

### Running Experiment A (Ready)

1. **Check experiment status**:
```bash
python3 experiment_summary.py
```

2. **Start training**:
```bash
python3 train_experiment_a.py
```

3. **Monitor training** (in another terminal):
```bash
# Watch training logs
tail -f work_dirs/experiment_a_adaptive_wing_ohkm_hybrid/vis_data/scalars.json

# Check GPU usage
nvidia-smi -l 1
```

4. **Evaluate after training**:
```bash
python3 evaluate_experiment.py --experiment A
```

## 📁 Project Structure
```
.
├── configs/                          # Experiment configurations
│   └── experiment_a_adaptive_wing_ohkm_hybrid.py
├── custom_losses.py                  # Custom loss implementations
├── custom_cephalometric_dataset.py   # Dataset handling
├── custom_transforms.py              # Data transforms
├── cephalometric_dataset_info.py     # Landmark definitions
├── train_experiment_a.py             # Training script for Exp A
├── evaluate_experiment.py            # Unified evaluation script
├── experiment_summary.py             # Status overview
└── work_dirs/                        # Training outputs
    └── experiment_a_*/               # Checkpoints and logs
```

## 📈 Expected Results

| Experiment | Resolution | Expected MRE | Key Innovation |
|------------|------------|--------------|----------------|
| Baseline   | 384×384    | 2.348 px     | AdaptiveWingLoss |
| A          | 384×384    | <2.30 px     | +OHKM for hard landmarks |
| B          | 384×384    | <2.25 px     | Focal loss adaptation |
| C          | 384×384    | <2.30 px     | Pure OHKM with σ=4 |
| D          | 512×512    | <2.15 px     | Resolution + coord regression |

## 🔍 Key Metrics to Monitor

1. **Overall MRE**: Mean Radial Error across all landmarks
2. **Sella/Gonion MRE**: Performance on hardest landmarks
3. **Training Loss Convergence**: Should decrease smoothly
4. **Validation NME**: Should improve over epochs

## 💡 Tips

- **GPU Memory**: Reduce batch size if OOM errors occur
- **Training Time**: ~3 hours for 384×384, ~5-6 hours for 512×512
- **Best Practices**: 
  - Monitor loss curves for instability
  - Check per-landmark errors, not just overall
  - Save intermediate checkpoints

## 📝 Notes

- All experiments start from the V4 pre-trained weights
- Data is loaded from Google Drive JSON file
- Results are saved in CSV format for easy comparison
- Visualization plots are generated automatically

## 🐛 Troubleshooting

**Import Errors**: Ensure all custom modules are in the project root
```bash
ls custom_*.py cephalometric_dataset_info.py
```

**CUDA Errors**: Check GPU availability
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

**Data Loading Issues**: Verify data file path
```bash
# Check if data file exists (update path as needed)
ls "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json"
```

## 📊 Results Tracking

After each experiment:
1. Results saved to: `work_dirs/experiment_*/evaluation_results/`
2. Summary CSV: `experiment_*_summary.csv`
3. Detailed CSV: `experiment_*_results.csv`
4. Comparison plots: `experiment_*_comparison.png`

---

For questions or issues, check the experiment logs in the work_dirs folder. 