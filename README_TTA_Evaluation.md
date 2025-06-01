# Test-Time Augmentation (TTA) Evaluation System

This directory contains a comprehensive Test-Time Augmentation (TTA) evaluation system for cephalometric landmark detection models. TTA applies multiple augmentations during inference and averages the results to improve accuracy.

## Files Overview

### Core Evaluation Scripts

1. **`evaluate_detailed_metrics.py`** - Original detailed evaluation script (baseline)
2. **`evaluate_with_tta.py`** - Heavy TTA evaluation script (NEW)
3. **`compare_tta_performance.py`** - Comparison script to show TTA improvements (NEW)

### TTA Features

The TTA system includes the following augmentation strategies:

- **Horizontal flipping** (with proper landmark symmetry handling)
- **Multi-scale testing** (0.9x, 1.1x, 1.2x)
- **Small rotations** (-10°, -5°, +5°, +10°)
- **Brightness variations** (0.9x, 1.1x)
- **Contrast variations** (0.9x, 1.1x)
- **Small spatial shifts** (±5 pixels in x/y directions)
- **Combined augmentations** (flip+scale, scale+rotation combinations)

**Total: ~20 different augmentations per image**

## Usage

### 1. Run TTA Evaluation Only

```bash
python evaluate_with_tta.py --checkpoint path/to/your/checkpoint.pth
```

**Options:**
- `--checkpoint`: Path to specific checkpoint file
- `--config`: Config file path (default: uses existing config)
- `--work_dir`: Work directory to search for checkpoints (if --checkpoint not specified)
- `--output_dir`: Custom output directory

**Example:**
```bash
python evaluate_with_tta.py \
    --checkpoint work_dirs/hrnetv2_w18_cephalometric_384x384_adaptive_wing_loss_v4/best_NME_epoch_45.pth \
    --output_dir tta_results_experiment1
```

### 2. Compare Regular vs TTA Performance

```bash
python compare_tta_performance.py --checkpoint path/to/your/checkpoint.pth
```

This will:
1. Run regular evaluation (using `evaluate_detailed_metrics.py`)
2. Run TTA evaluation (using `evaluate_with_tta.py`)
3. Generate comparison plots and detailed reports

**Options:**
- `--checkpoint`: Path to checkpoint file (required)
- `--output_dir`: Output directory for comparison results
- `--skip_regular`: Skip regular evaluation (use existing results)
- `--skip_tta`: Skip TTA evaluation (use existing results)

**Example:**
```bash
python compare_tta_performance.py \
    --checkpoint work_dirs/hrnetv2_w18_cephalometric_384x384_adaptive_wing_loss_v4/best_NME_epoch_45.pth \
    --output_dir comparison_results_v1
```

### 3. Quick TTA Test with Auto-Detection

If you don't specify a checkpoint, the scripts will automatically find the best checkpoint in the default work directory:

```bash
python evaluate_with_tta.py
python compare_tta_performance.py --checkpoint auto
```

## Output Structure

### TTA Evaluation Output (`evaluate_with_tta.py`)

```
output_dir/
├── per_landmark_mre_results_tta.csv    # Per-landmark detailed metrics
├── overall_summary_tta.csv             # Overall summary statistics  
├── error_analysis_tta.png             # Error distribution plots
└── (console output with detailed statistics)
```

### Comparison Output (`compare_tta_performance.py`)

```
comparison_results/
├── regular_evaluation/
│   ├── per_landmark_mre_results.csv
│   ├── overall_summary.csv
│   └── error_analysis.png
├── tta_evaluation/
│   ├── per_landmark_mre_results_tta.csv
│   ├── overall_summary_tta.csv
│   └── error_analysis_tta.png
├── tta_comparison.png                  # Side-by-side comparison plots
└── tta_comparison_report.txt           # Detailed improvement analysis
```

## Expected Performance Improvements

TTA typically provides:

- **Overall MRE reduction**: 2-8% improvement on average
- **Per-landmark improvements**: Some landmarks may improve by 10-20%
- **Robustness**: Better handling of edge cases and challenging images
- **Stability**: More consistent predictions across similar images

## Technical Details

### TTA Pipeline

1. **Image Preprocessing**: Apply augmentation to input image
2. **Model Inference**: Run model prediction on augmented image
3. **Coordinate Transformation**: Transform predicted keypoints back to original coordinate system
4. **Symmetry Handling**: Handle landmark symmetry for horizontal flips
5. **Averaging**: Average all predictions to get final result

### Performance Considerations

- **Runtime**: TTA is ~20x slower than regular inference (due to multiple augmentations)
- **Memory**: Requires same GPU memory as regular inference
- **Accuracy**: Typically improves MRE by 2-8%

### Symmetric Landmark Handling

For cephalometric landmarks, you may want to define symmetric pairs for proper flip handling. Edit the `flip_indices` parameter in `evaluate_with_tta.py` if your landmarks have symmetric correspondences.

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: The script automatically handles this, but you can reduce batch processing if needed
2. **Missing dependencies**: Ensure OpenCV (`cv2`) is installed: `pip install opencv-python`
3. **Path issues**: Make sure your checkpoint and config files exist and are accessible

### Performance Optimization

- Use GPU for faster inference: `CUDA_VISIBLE_DEVICES=0 python evaluate_with_tta.py ...`
- For CPU-only: The script automatically detects and uses CPU if CUDA is unavailable

## Example Results

```
TTA EVALUATION RESULTS
============================================================
Overall MRE: 2.847 ± 1.923 pixels (vs 3.021 ± 2.156 pixels regular)
Valid predictions: 1,140/1,140 (100.0%)
Median error: 2.234 pixels
90th percentile: 5.612 pixels
95th percentile: 7.891 pixels

Improvement: 5.76% reduction in overall MRE
```

## Advanced Usage

### Custom TTA Configuration

You can modify the TTA transforms in `evaluate_with_tta.py` by editing the `get_tta_transforms()` method in the `TTATransforms` class. Add or remove specific augmentations based on your needs.

### Batch Processing Multiple Checkpoints

```bash
# Evaluate multiple checkpoints
for checkpoint in work_dirs/*/best_*.pth; do
    echo "Evaluating $checkpoint"
    python evaluate_with_tta.py --checkpoint "$checkpoint" --output_dir "tta_$(basename $(dirname $checkpoint))"
done
```

## Citation

If you use this TTA evaluation system in your research, please cite the relevant papers and acknowledge the MMPose framework. 