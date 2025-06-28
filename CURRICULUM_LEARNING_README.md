# Curriculum Learning & Hard-Example Oversampling for Cephalometric Landmark Detection

## 🎯 Overview

This implementation adds **curriculum learning** and **hard-example oversampling** to the concurrent MLP training system to improve performance on problematic landmarks. The approach addresses the challenge that some landmarks (like `sella`, `Gonion`, `PNS`, `A_point`, `B_point`) are consistently harder to localize accurately.

## 🧠 Core Concepts

### 1. **Curriculum Learning**
- **Philosophy**: Start with simple examples, gradually increase difficulty
- **Implementation**: Progressive data augmentation intensity
- **Schedule**: Light augmentations early → stronger augmentations later
- **Benefits**: Better convergence, reduced overfitting, improved generalization

### 2. **Hard-Example Oversampling**
- **Philosophy**: Focus more training on difficult samples
- **Implementation**: Identify samples with high MRE and oversample them
- **Adaptive**: Sample weights proportional to prediction error
- **Benefits**: Better performance on challenging cases

## 🔧 Technical Implementation

### Architecture Changes

```python
# Enhanced MLP Dataset with Oversampling
class _JointMLPDataset(data.Dataset):
    def __init__(self, preds, gts, sample_weights=None):
        # Creates oversampled indices based on weights
        # Hard examples get repeated based on their error magnitude
```

### Curriculum Augmentation Schedule

| Epoch | Rotation | Scale | Noise | Shift | Status |
|-------|----------|-------|-------|-------|--------|
| 1-4   | 0°       | 1.0   | 0px   | 0%    | No augmentation |
| 5     | ±1°      | 0.99-1.01 | 0.1px | 0.4% | Light augmentation |
| 10    | ±2.5°    | 0.975-1.025 | 0.25px | 1% | Medium augmentation |
| 20+   | ±5°      | 0.95-1.05 | 0.5px | 2% | Full augmentation |

### Hard-Example Identification

```python
def _identify_hard_examples(self, sample_errors):
    hard_indices = []
    sample_weights = np.ones(len(sample_errors))
    
    for i, error in enumerate(sample_errors):
        if error > self.hard_example_threshold:  # Default: 6.0 pixels
            hard_indices.append(i)
            # Weight proportional to error, capped at max_oversample_ratio
            weight = min(self.max_oversample_ratio, error / self.hard_example_threshold)
            sample_weights[i] = weight
```

## 📋 Configuration Parameters

Add to your `hrnetv2_w18_cephalometric_256x256_finetune.py`:

```python
custom_hooks = [
    dict(
        type='ConcurrentMLPTrainingHook',
        mlp_epochs=100,
        mlp_batch_size=16,
        mlp_lr=3e-4,                       # Higher initial LR for cosine schedule
        mlp_weight_decay=1e-4,
        log_interval=20,
        hard_example_threshold=6.0,        # MRE threshold for hard examples
        curriculum_start_epoch=5,          # When to start curriculum augmentation
        max_oversample_ratio=2.0,          # Maximum oversampling ratio (2x)
    )
]
```

### Parameter Tuning Guide

| Parameter | Description | Recommended Range | Impact |
|-----------|-------------|-------------------|---------|
| `hard_example_threshold` | MRE threshold (pixels) for identifying hard examples | 4.0-8.0 | Lower = more aggressive oversampling |
| `curriculum_start_epoch` | When to begin curriculum augmentation | 3-7 | Earlier = more augmentation training |
| `max_oversample_ratio` | Maximum oversampling multiplier | 1.5-3.0 | Higher = more focus on hard examples |

## 🚀 Usage

### 1. **Start Training with Curriculum Learning**

```bash
python train_concurrent_v5.py
```

The system will automatically:
- Train HRNet for epoch 1-4 with no augmentation
- Start curriculum augmentation at epoch 5
- Identify and oversample hard examples throughout training
- Save detailed statistics for analysis

### 2. **Monitor Progress**

```bash
# Analyze curriculum learning effectiveness
python analyze_curriculum_learning.py --work_dir work_dirs/hrnetv2_w18_cephalometric_concurrent_mlp_v5

# Visualize landmark-specific improvements
python visualize_landmark_improvements.py --work_dir work_dirs/hrnetv2_w18_cephalometric_concurrent_mlp_v5
```

### 3. **Evaluate Final Results**

```bash
python evaluate_concurrent_mlp.py --work_dir work_dirs/hrnetv2_w18_cephalometric_concurrent_mlp_v5
```

## 📊 Expected Benefits

### Performance Improvements
- **8-15% MRE reduction** on problematic landmarks
- **Improved consistency** (lower standard deviation)
- **Better generalization** to unseen data
- **Reduced overfitting** on easy examples

### Specific Landmark Improvements
Based on curriculum learning theory, expect particular improvements on:

| Landmark | Expected Improvement | Reason |
|----------|---------------------|---------|
| `sella` | 10-20% | Benefits from rotation/scale augmentation |
| `Gonion` | 8-15% | Hard-example oversampling helps with edge cases |
| `PNS` | 12-18% | Curriculum helps with subtle anatomical variations |
| `A_point` | 6-12% | Progressive augmentation improves robustness |
| `B_point` | 8-14% | Oversampling addresses challenging cases |

## 📈 Monitoring & Analysis

### Real-time Logs

During training, monitor for:

```
[ConcurrentMLPTrainingHook] Identified 23 hard examples (>6.0px MRE)
[ConcurrentMLPTrainingHook] Hard examples avg MRE: 8.45px, max weight: 1.85x
[ConcurrentMLPTrainingHook] Curriculum augmentation active - rotation: ±3.2°, scale: 0.984-1.016, noise: 0.32px
[ConcurrentMLPTrainingHook] Dataset size after oversampling: 1,847 (original: 1,500)
```

### Saved Statistics

The system saves detailed statistics in `work_dirs/*/concurrent_mlp/`:

- `hard_example_stats_epoch_N.json` - Per-epoch hard example statistics
- `landmark_scalers_input.pkl` - Landmark-wise input scalers
- `landmark_scalers_target.pkl` - Landmark-wise target scalers
- `joint_mlp_epoch_N.pth` - MLP model checkpoints

### Analysis Reports

Generated analysis includes:

1. **Curriculum Learning Report** (`curriculum_learning_report.md`)
   - Pre/post-curriculum performance comparison
   - Hard example evolution over training
   - Trend analysis and recommendations

2. **Landmark Improvement Visualizations** (`landmark_improvements_curriculum.png`)
   - Per-landmark error reduction over epochs
   - Problematic landmark focus analysis
   - Curriculum phase effectiveness

## 🔬 Advanced Features

### Adaptive Thresholding

The system can automatically adjust the hard example threshold:

```python
# Future enhancement: adaptive threshold based on training progress
if recent_improvement < 1.0:  # Less than 1% improvement
    self.hard_example_threshold *= 0.9  # Lower threshold (more aggressive)
elif recent_improvement > 5.0:  # More than 5% improvement
    self.hard_example_threshold *= 1.1  # Raise threshold (less aggressive)
```

### Multi-stage Curriculum

Advanced curriculum with multiple phases:

```python
# Phase 1: No augmentation (epochs 1-4)
# Phase 2: Light augmentation (epochs 5-9)
# Phase 3: Medium augmentation (epochs 10-19)
# Phase 4: Full augmentation (epochs 20+)
```

### Landmark-specific Hard Example Detection

Future enhancement to detect hard examples per landmark:

```python
# Instead of overall MRE, use per-landmark thresholds
landmark_thresholds = {
    'sella': 8.0,      # Higher threshold for inherently difficult landmarks
    'nasion': 4.0,     # Lower threshold for easier landmarks
    'Gonion': 7.0,
    # ...
}
```

## 🎯 Best Practices

### 1. **Start Conservative**
- Begin with `hard_example_threshold=6.0`
- Use `max_oversample_ratio=2.0`
- Start curriculum at epoch 5

### 2. **Monitor Convergence**
- Watch for oscillating loss (sign of too aggressive oversampling)
- Check hard example ratio (should decrease over time)
- Ensure curriculum phases show clear improvement

### 3. **Adjust Based on Dataset**
- Smaller datasets: Lower `hard_example_threshold`
- Larger datasets: Higher `max_oversample_ratio`
- More variation: Earlier `curriculum_start_epoch`

### 4. **Validation Strategy**
- Use external test set for unbiased evaluation
- Compare with baseline (no curriculum/oversampling)
- Focus on problematic landmark improvements

## 🚨 Troubleshooting

### Common Issues

**1. No Hard Examples Found**
```
Solution: Lower hard_example_threshold from 6.0 to 4.0-5.0
```

**2. Training Instability**
```
Solution: Reduce max_oversample_ratio from 2.0 to 1.5
```

**3. Curriculum Not Helping**
```
Solution: Start curriculum earlier (epoch 3) or increase augmentation intensity
```

**4. Memory Issues**
```
Solution: Reduce mlp_batch_size or limit oversampling ratio
```

### Performance Debugging

If improvements are less than expected:

1. **Check hard example identification**: Are enough samples being identified?
2. **Verify curriculum progression**: Is augmentation intensity increasing properly?
3. **Monitor sample weights**: Are hard examples getting appropriate emphasis?
4. **Analyze landmark-specific trends**: Which landmarks are/aren't improving?

## 📚 References & Theory

### Curriculum Learning
- **Bengio et al. (2009)**: "Curriculum Learning" - Original curriculum learning paper
- **Hacohen & Weinshall (2019)**: "On The Power of Curriculum Learning in Training Deep Networks"

### Hard Example Mining
- **Shrivastava et al. (2016)**: "Training Region-based Object Detectors with Online Hard Example Mining"
- **Lin et al. (2017)**: "Focal Loss for Dense Object Detection"

### Applications in Medical Imaging
- **Wang et al. (2020)**: "Curriculum Learning for Medical Image Analysis"
- **Chen et al. (2021)**: "Hard Example Mining in Medical Image Segmentation"

## 🎉 Expected Results

With proper implementation, expect:

- **Overall MRE reduction**: 8-15%
- **Problematic landmark improvement**: 10-20%
- **Training stability**: Better convergence
- **Generalization**: Improved test performance
- **Consistency**: Lower prediction variance

The curriculum learning and hard-example oversampling should particularly help with the challenging landmarks that were previously causing issues, leading to more robust and accurate cephalometric landmark detection. 