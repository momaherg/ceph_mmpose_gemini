# Cephalometric Landmark Detection Experiments

This experiment system allows you to systematically test different configurations for cephalometric landmark detection. Each experiment starts from the same pre-trained HRNetV2-W18 checkpoint and explores different hyperparameters.

## Available Experiments

The system includes 10 pre-configured experiments:

0. **baseline_256x256_mse** - Baseline configuration with 256x256 resolution and MSE loss
1. **highres_384x384_mse** - Higher resolution 384x384 with MSE loss
2. **adaptive_wing_256x256** - AdaptiveWingLoss with standard 256x256 resolution
3. **adaptive_wing_384x384** - AdaptiveWingLoss with high resolution 384x384
4. **sgd_momentum_256x256** - SGD with momentum optimizer
5. **low_lr_long_train** - Very low learning rate with extended training
6. **adamw_weight_decay** - AdamW optimizer with weight decay regularization
7. **small_batch_adaptive** - Smaller batch size with adaptive wing loss
8. **no_augmentation** - Baseline without data augmentation
9. **ultra_highres_512x512** - Ultra high resolution 512x512 experiment

## Running Experiments

### Run a Single Experiment

```bash
# Run experiment 0 (baseline)
python run_experiment.py --index 0

# List all available experiments
python run_experiment.py --list
```

### Run Multiple Experiments

```bash
# Run experiments 0, 2, and 3 sequentially
python run_batch_experiments.py --experiments 0 2 3 --sequential

# Run experiments in parallel (2 workers by default)
python run_batch_experiments.py --experiments 0 1 2 3

# Run with more parallel workers
python run_batch_experiments.py --experiments 0 1 2 3 --max-workers 4

# Skip already completed experiments
python run_batch_experiments.py --experiments 0 1 2 3 --skip-completed
```

## Managing Experiments

### List All Experiments with Status

```bash
# Show all experiments and their current status
python experiment_utils.py --list
```

### Compare Experiment Results

```bash
# Compare all completed experiments
python experiment_utils.py --compare

# Compare specific experiments
python experiment_utils.py --compare 0 2 3
```

### Get Detailed Information

```bash
# Get details for experiment 0
python experiment_utils.py --details 0
```

### Clean Up Experiments

```bash
# Delete experiment 0 directory (will ask for confirmation)
python experiment_utils.py --clean 0
```

## Experiment Results

Each experiment creates its own directory under `work_dirs/`:
- `work_dirs/experiment_0_baseline_256x256_mse/`
- `work_dirs/experiment_1_highres_384x384_mse/`
- etc.

Each directory contains:
- `best_NME_epoch_*.pth` - Best model checkpoint
- `experiment_info.json` - Experiment configuration
- `experiment_summary.txt` - Training summary
- `training_progress.png` - Loss and NME plots
- `vis_data/scalars.json` - Detailed training logs

## Evaluating Results

After training, you can evaluate detailed metrics:

```bash
# Evaluate experiment 0
python evaluate_detailed_metrics.py --work_dir work_dirs/experiment_0_baseline_256x256_mse
```

## Best Practices

1. **Start with key experiments**: Run experiments 0, 2, 3 first to compare baseline vs AdaptiveWingLoss and resolution impact

2. **Monitor GPU memory**: Higher resolution experiments (384x384, 512x512) use more memory

3. **Use parallel execution carefully**: Don't run too many high-resolution experiments in parallel

4. **Compare incrementally**: Use `experiment_utils.py --compare` after each batch to track progress

5. **Document findings**: The system saves all configurations and results for reproducibility

## Adding New Experiments

To add a new experiment, edit `experiments_config.py` and add a new configuration to the `experiments` list:

```python
{
    "name": "your_experiment_name",
    "description": "Description of what this experiment tests",
    "config": {
        "input_size": (256, 256),
        "heatmap_size": (64, 64),
        "loss_type": "KeypointMSELoss",
        "loss_config": {},
        "optimizer": "Adam",
        "lr": 5e-4,
        "batch_size": 32,
        "max_epochs": 50,
        "augmentation": {
            "rotate_factor": 15,
            "scale_factor": (0.85, 1.15),
        }
    }
}
```

## Troubleshooting

- **Out of memory**: Reduce batch_size or use lower resolution
- **Training fails**: Check `experiment_error.txt` in the work directory
- **Slow training**: Higher resolutions (384x384, 512x512) take longer
- **Can't compare results**: Make sure experiments completed successfully 