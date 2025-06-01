# MLP Refinement Network for Cephalometric Landmark Detection

## Overview

This project implements a **two-stage refinement approach** for improving cephalometric landmark detection accuracy. The approach uses a trained HRNetV2 model as the first stage and an MLP refinement network as the second stage to achieve better precision, especially for challenging landmarks.

## 🎯 Motivation

Your current HRNetV2 model (384x384 + AdaptiveWingLoss) achieves:
- **Overall MRE**: 2.348 ± 1.807 pixels  
- **Challenging landmarks** still have high errors:
  - Sella: 4.674 pixels
  - Gonion: 4.281 pixels  
  - PNS: 3.165 pixels

The MLP refinement network aims to **reduce these errors by 10-25%** through learned refinement.

## 🏗️ Architecture

### Two-Stage Pipeline
```
Image → HRNetV2 → Initial Predictions → MLP Refinement → Final Refined Predictions
```

### MLP Refinement Network Components

1. **Image Feature Extractor** (Lightweight CNN)
   - Extracts global image features (512-dim)
   - Input: 384×384 images → Output: Global context vector

2. **Landmark-Specific Refiners** (19 individual MLPs)
   - Each landmark has its own MLP refiner
   - Input features per landmark:
     - Initial HRNetV2 prediction (x, y)
     - Normalized coordinates 
     - Distance from image center
     - Global image features (512-dim)
   - Output: Coordinate refinement offset

3. **Feature Integration**
   - Combines spatial, geometric, and semantic information
   - Uses residual connections: `refined = initial + refinement`
   - Landmark-specific weights for challenging landmarks

### Key Design Features

- **Residual Learning**: Learns offsets rather than absolute coordinates
- **Individual MLPs**: Each landmark has specialized refinement logic
- **Weighted Loss**: Higher weights for challenging landmarks (Sella: 2.0x, Gonion: 2.0x, PNS: 1.5x)
- **Multiple Loss Functions**: MSE, Smooth L1, Huber loss support

## 📁 File Structure

```
mlp_refinement_network.py     # Core MLP refinement model
mlp_refinement_dataset.py     # Dataset for training refinement network
train_mlp_refinement.py       # Training script
evaluate_mlp_refinement.py    # Comprehensive evaluation script
```

## 🚀 Quick Start

### 1. Train MLP Refinement Network

```bash
python3 train_mlp_refinement.py
```

**Configuration:**
- Uses your best HRNetV2 checkpoint (384x384 + AdaptiveWingLoss)
- Pre-extracts HRNetV2 predictions for faster training
- 50 epochs with cosine annealing schedule
- Batch size: 16, Learning rate: 1e-3

### 2. Evaluate Results

```bash
python3 evaluate_mlp_refinement.py
```

**Outputs:**
- Detailed comparison with HRNetV2 baseline
- Per-landmark improvement analysis  
- Comprehensive plots and visualizations
- Results saved to `work_dirs/mlp_refinement_v1/evaluation_results/`

## 📊 Expected Results

Based on the architecture design, expected improvements:

### Overall Performance
- **Target MRE**: 2.348px → **2.0-2.2px** (5-15% improvement)
- **Median Error**: 1.901px → **1.6-1.8px**
- **95th Percentile**: 5.886px → **5.0-5.5px**

### Challenging Landmarks  
- **Sella**: 4.674px → **3.7-4.2px** (10-20% improvement)
- **Gonion**: 4.281px → **3.4-3.8px** (15-20% improvement)
- **PNS**: 3.165px → **2.5-2.8px** (15-20% improvement)

### Well-Performing Landmarks
- **Soft tissue landmarks** (tip of nose, lips): Minimal change (already <1.2px)
- **Stable landmarks** (nasion, menton): Small improvements (5-10%)

## 🧠 Model Details

### Network Architecture

```python
CephalometricMLPRefinement(
    num_landmarks=19,
    input_size=384,
    image_feature_dim=512,
    landmark_hidden_dims=[256, 128, 64],
    dropout=0.3,
    use_landmark_weights=True
)
```

**Model Size**: ~5.1M parameters
- Image encoder: ~1.8M parameters
- 19 landmark refiners: ~3.3M parameters total

### Training Strategy

1. **Data Preparation**: Extract HRNetV2 predictions for all training samples
2. **Loss Function**: Smooth L1 loss with landmark-specific weights
3. **Optimization**: AdamW with cosine annealing (1e-3 → 1e-6)
4. **Regularization**: Dropout (0.3) + Weight decay (1e-4) + Gradient clipping

### Key Features

- **Residual Learning**: Small weight initialization for stable training
- **Landmark Weights**: 2.0x weight for Sella/Gonion, 1.5x for PNS
- **Robust Loss**: Smooth L1 less sensitive to outliers than MSE
- **Feature Engineering**: Spatial + geometric + semantic features

## 📈 Evaluation Metrics

The evaluation script provides comprehensive analysis:

### Overall Metrics
- Mean Radial Error (MRE) comparison
- Standard deviation and percentile analysis
- Improvement percentage and absolute gains

### Per-Landmark Analysis
- Individual MRE for each of 19 landmarks
- Improvement breakdown by landmark type
- Focus on challenging landmarks (>3px error)

### Visualization
- Error distribution comparison
- Per-landmark improvement bar charts
- Improvement vs. initial error scatter plots
- Challenging landmarks focus plots

## 🔧 Configuration Options

### Model Configuration
```python
model_config = {
    'num_landmarks': 19,
    'input_size': 384,
    'image_feature_dim': 512,
    'landmark_hidden_dims': [256, 128, 64],
    'dropout': 0.3,
    'use_landmark_weights': True
}
```

### Training Configuration
```python
training_config = {
    'num_epochs': 50,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'loss_type': 'smooth_l1',  # 'mse', 'smooth_l1', 'huber'
    'scheduler_type': 'cosine',  # 'cosine', 'step', 'none'
    'batch_size': 16
}
```

## 💡 Technical Insights

### Why MLP Refinement Works

1. **Complementary Learning**: HRNetV2 provides global structure, MLP learns local refinements
2. **Landmark Specialization**: Individual MLPs can learn landmark-specific correction patterns
3. **Rich Feature Integration**: Combines spatial, geometric, and semantic information
4. **Residual Learning**: Small corrections are easier to learn than absolute positions

### Architectural Choices

- **Individual MLPs vs. Single MLP**: Better specialization for each landmark
- **Residual Connection**: Ensures refinement network only learns corrections
- **Global Image Features**: Provides context for local refinements
- **Geometric Features**: Distance from center helps with perspective/scale

## 🎮 Usage Examples

### Training with Custom Settings
```python
from train_mlp_refinement import main, MLPRefinementTrainer
from mlp_refinement_network import create_model

# Custom model configuration
config = {
    'landmark_hidden_dims': [512, 256, 128],  # Larger network
    'dropout': 0.2,  # Less dropout
    'image_feature_dim': 768  # Larger feature space
}

model = create_model(config)
# ... continue with training
```

### Evaluation on Custom Data
```python
from evaluate_mlp_refinement import MLPRefinementEvaluator
from mlp_refinement_dataset import MLPRefinementDataset

# Load custom test set
evaluator = MLPRefinementEvaluator(model, test_dataset)
metrics = evaluator.evaluate_all()
evaluator.print_results()
evaluator.plot_results('custom_results/')
```

## 🔍 Monitoring Training

Training progress is automatically monitored:

1. **Loss Curves**: Training and validation loss over epochs
2. **MRE Tracking**: Mean radial error for both initial and refined predictions  
3. **Improvement Tracking**: Continuous monitoring of refinement gains
4. **Best Model Saving**: Automatic saving of best performing checkpoint

### Key Metrics to Watch

- **Validation MRE**: Should decrease from HRNetV2 baseline
- **Improvement**: Should be consistently positive (>0)
- **Loss Convergence**: Smooth decrease without overfitting
- **Per-landmark Progress**: Challenging landmarks showing improvement

## 🏆 Success Criteria

### Minimum Success
- **Overall MRE improvement**: >5% (2.348px → <2.23px)
- **Challenging landmarks**: >10% improvement on Sella/Gonion/PNS
- **No degradation**: Well-performing landmarks not getting worse

### Target Success  
- **Overall MRE improvement**: >10% (2.348px → <2.11px)
- **Challenging landmarks**: >15% improvement
- **Balanced improvement**: Benefits across all landmark types

### Stretch Goal
- **Overall MRE improvement**: >15% (2.348px → <2.00px)
- **Challenging landmarks**: >20% improvement  
- **Sub-2px overall MRE**: Achieving sub-pixel precision

## 🚨 Troubleshooting

### Common Issues

1. **Training Loss Not Decreasing**
   - Check learning rate (try 5e-4 or 2e-3)
   - Verify HRNetV2 predictions are reasonable
   - Ensure proper data normalization

2. **Overfitting (Val Loss Increasing)**
   - Increase dropout (0.3 → 0.5)
   - Add more weight decay (1e-4 → 1e-3)
   - Reduce model complexity

3. **No Improvement Over Baseline**
   - Check if landmark weights are applied correctly
   - Verify residual connections are working
   - Try different loss functions (smooth_l1 → huber)

4. **Memory Issues**
   - Reduce batch size (16 → 8)
   - Disable prediction caching
   - Use mixed precision training

### Debug Commands

```bash
# Test network architecture
python3 mlp_refinement_network.py

# Test dataset loading  
python3 -c "from mlp_refinement_dataset import *; test_dataset()"

# Quick training test (1 epoch)
python3 train_mlp_refinement.py --num_epochs 1 --batch_size 4
```

## 📚 References

- **HRNet**: Deep High-Resolution Representation Learning for Human Pose Estimation
- **Residual Learning**: Deep Residual Learning for Image Recognition  
- **Multi-stage Refinement**: Cascaded Pyramid Network for Multi-Person Pose Estimation
- **Landmark Detection**: Wing Loss for Robust Facial Landmark Localisation 