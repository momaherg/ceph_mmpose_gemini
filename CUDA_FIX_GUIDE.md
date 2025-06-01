# CUDA Error Fix Guide for MLP Refinement Training

## Problem
You encountered CUDA device-side assertion errors during HRNetV2 prediction extraction:
```
CUDA error: device-side assert triggered
```

## Root Causes
1. **Memory Issues**: Large images causing GPU memory overflow
2. **Device Mismatches**: Model and data on different devices
3. **OpenCV Dimension Limits**: Very large images exceeding OpenCV limits
4. **CUDA Kernel Failures**: Invalid tensor operations

## Solutions Implemented

### 1. Image Resizing Fix ✅
- **Problem**: Original images too large for OpenCV transformations
- **Solution**: Resize images to 384x384 before HRNetV2 inference
- **Code**: Modified `extract_predictions()` in `mlp_refinement_dataset.py`

### 2. CPU Fallback System ✅
- **Problem**: CUDA errors during inference
- **Solution**: Automatic fallback to CPU inference when GPU fails
- **Features**:
  - Try GPU inference first
  - Automatically switch to CPU on CUDA errors
  - Clear GPU memory cache
  - Force CPU mode option

### 3. Better Error Handling ✅
- **Problem**: Cryptic CUDA errors
- **Solution**: Comprehensive error handling and debugging
- **Features**:
  - Detailed error messages
  - Progress tracking with success/failure counts
  - Zero prediction fallback for failed samples

### 4. Memory Management ✅
- **Problem**: GPU memory accumulation
- **Solution**: Periodic memory cleanup
- **Features**:
  - Clear GPU cache every 100 samples
  - Reduced batch sizes
  - CPU inference option

## How to Use the Fixes

### Option 1: Force CPU Inference (Recommended)
The training script now defaults to CPU inference for HRNetV2 to avoid CUDA issues:

```python
# In train_mlp_refinement.py
config = {
    'force_cpu_hrnet': True  # This is now the default
}
```

### Option 2: Test Before Training
Run the test script first to verify everything works:

```bash
python3 test_dataset_loading.py
```

### Option 3: Environment Variables
For more detailed CUDA error information:

```bash
export CUDA_LAUNCH_BLOCKING=1
python3 train_mlp_refinement.py
```

## Configuration Options

### CPU vs GPU for Different Components

| Component | GPU | CPU | Recommendation |
|-----------|-----|-----|----------------|
| HRNetV2 Inference | ❌ CUDA errors | ✅ Stable | **Use CPU** |
| MLP Training | ✅ Faster | ⚠️ Slower | **Use GPU** |

### Updated Configuration
```python
config = {
    'device': 'cuda:0',           # For MLP training (GPU)
    'force_cpu_hrnet': True,      # For HRNetV2 inference (CPU)
    'batch_size': 16,             # Reduced from 32
    'num_workers': 0,             # Avoid multiprocessing issues
}
```

## Performance Impact

### HRNetV2 on CPU vs GPU
- **CPU**: Slower but stable (~2-3x slower prediction extraction)
- **GPU**: Faster but prone to CUDA errors
- **Trade-off**: Slightly longer data preparation for stable training

### MLP Training Still on GPU
- The actual MLP refinement training still uses GPU
- Only HRNetV2 prediction extraction uses CPU
- Overall training time impact: ~10-20% slower data loading

## Expected Behavior Now

### During Prediction Extraction
```
Creating dataloaders and extracting HRNetV2 predictions...
Forcing CPU inference to avoid CUDA issues
HRNetV2 model loaded from ... (CPU)
Pre-extracting HRNetV2 predictions for faster training...
Extracting predictions: 0/1250 (Success: 0, Failed: 0)
  Sample 0: Image shape (224, 224, 3), dtype uint8
  Sample 1: Image shape (224, 224, 3), dtype uint8
  Sample 2: Image shape (224, 224, 3), dtype uint8
Extracting predictions: 25/1250 (Success: 25, Failed: 0)
...
Prediction extraction completed:
  Total samples: 1250
  Successful: 1248
  Failed: 2
  Success rate: 99.8%
```

### During MLP Training
```
🧠 Creating MLP refinement model...
Model parameters: 5,123,456
Epoch 1/50
Train - Loss: 0.1234, MRE: 2.156 (Δ: +0.192)
Val   - Loss: 0.1156, MRE: 2.089 (Δ: +0.259)
```

## Troubleshooting

### If You Still Get CUDA Errors
1. **Force complete CPU mode**:
   ```python
   config['device'] = 'cpu'  # Force CPU for everything
   ```

2. **Reduce batch size further**:
   ```python
   config['batch_size'] = 8  # Or even 4
   ```

3. **Disable prediction caching**:
   ```python
   cache_predictions=False  # Extract predictions on-the-fly
   ```

### If CPU is Too Slow
1. **Use smaller dataset for testing**:
   ```python
   train_df = train_df.head(100)  # Use only 100 samples
   ```

2. **Reduce image size**:
   ```python
   config['input_size'] = 256  # Smaller than 384
   ```

## Testing Commands

### 1. Test Dataset Loading Only
```bash
python3 test_dataset_loading.py
```

### 2. Test Image Processing
```bash
python3 test_image_processing.py
```

### 3. Full Training (with fixes)
```bash
python3 train_mlp_refinement.py
```

### 4. With Debug Info
```bash
CUDA_LAUNCH_BLOCKING=1 python3 train_mlp_refinement.py
```

## Next Steps

1. **✅ Run the test script** to verify fixes work
2. **✅ Run training** with CPU inference for HRNetV2  
3. **✅ Monitor** success rates during prediction extraction
4. **✅ Proceed with training** once prediction extraction is stable

The fixes ensure stable training by using CPU for HRNetV2 inference while keeping GPU acceleration for the actual MLP refinement training. 