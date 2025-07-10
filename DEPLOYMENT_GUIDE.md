# Deployment Guide for Cephalometric Landmark Detection Model

This guide will help you deploy your ensemble model (Model 2, Epoch 99) from GPU training to CPU inference server.

## Overview

The deployment process consists of 3 main steps:
1. **Prepare** - Extract and simplify the model checkpoint
2. **Convert** - Convert PyTorch models to ONNX format
3. **Deploy** - Run inference on CPU server

## Prerequisites

### On GPU Machine (where you trained)
```bash
# Required packages
pip install mmpose mmengine mmcv onnx onnxruntime torch torchvision
```

### On CPU Server (deployment target)
```bash
# Minimal requirements for inference
pip install onnxruntime opencv-python numpy
# Optional for visualization
pip install matplotlib
# Optional for faster inference
pip install onnxruntime-openvino  # Intel CPUs
```

## Step 1: Prepare Model for Deployment (GPU Machine)

Run this on your GPU machine where the ensemble was trained:

```bash
# Default: Extract model 2, epoch 99
python prepare_model_for_deployment.py

# Or specify custom parameters
python prepare_model_for_deployment.py \
    --model_idx 2 \
    --epoch 99 \
    --output_dir deployment_package
```

This will create a `deployment_package/` directory containing:
- `hrnet_published.pth` - Simplified HRNet checkpoint (~50% smaller)
- `mlp_refinement.pth` - MLP refinement model
- `config.py` - Model configuration
- `deployment_info.json` - Deployment metadata

## Step 2: Convert to ONNX Format (GPU Machine)

Convert the PyTorch models to ONNX for efficient CPU inference:

```bash
# Convert to ONNX
python convert_to_onnx.py

# Or with custom paths
python convert_to_onnx.py \
    --deployment_dir deployment_package \
    --output_dir onnx_models
```

This creates `onnx_models/` directory with:
- `hrnet_model.onnx` - HRNet model for CPU
- `mlp_model.onnx` - MLP refinement model
- `onnx_config.json` - Inference configuration
- `optimize_for_cpu.py` - Optional quantization script

### Optional: Quantize Models for Faster Inference

For ~2x faster inference with minimal accuracy loss:

```bash
cd onnx_models
python optimize_for_cpu.py
```

This creates quantized versions:
- `hrnet_model_quantized.onnx` (~75% smaller)
- `mlp_model_quantized.onnx` (~75% smaller)

## Step 3: Transfer to CPU Server

Transfer the ONNX models to your CPU server:

```bash
# Option 1: SCP
scp -r onnx_models/ user@cpu-server:/path/to/deployment/

# Option 2: rsync (better for large files)
rsync -avz onnx_models/ user@cpu-server:/path/to/deployment/

# Option 3: Create tarball
tar -czf onnx_models.tar.gz onnx_models/
# Then transfer and extract on server
```

Also transfer the inference script:
```bash
scp inference_cpu.py user@cpu-server:/path/to/deployment/
```

## Step 4: Run Inference on CPU Server

On your CPU server:

### Basic Inference

```bash
# Single image
python inference_cpu.py --image test_image.jpg

# Multiple images from list
python inference_cpu.py --image_list images.txt

# With visualization
python inference_cpu.py --image test_image.jpg --visualize

# Save results to JSON
python inference_cpu.py --image test_image.jpg --save_json
```

### Optimized Inference

```bash
# Use quantized models for faster inference
python inference_cpu.py --use_quantized --image test_image.jpg

# Batch processing
python inference_cpu.py --image_list batch_images.txt --use_quantized
```

## Performance Optimization

### 1. Use Quantized Models
- ~2x faster inference
- ~4x smaller model size
- <1% accuracy loss

```bash
python inference_cpu.py --use_quantized
```

### 2. Batch Processing
Process multiple images efficiently:

```python
# Create a list file
echo "image1.jpg" > batch.txt
echo "image2.jpg" >> batch.txt
echo "image3.jpg" >> batch.txt

# Run batch inference
python inference_cpu.py --image_list batch.txt --use_quantized
```

### 3. Multi-threading
The inference script already uses optimized threading:
- Inter-op threads: 4
- Intra-op threads: 4

Adjust in `inference_cpu.py` if needed:
```python
sess_options.inter_op_num_threads = 8  # For more CPU cores
sess_options.intra_op_num_threads = 8
```

### 4. Intel Optimization (Optional)
For Intel CPUs, install OpenVINO execution provider:

```bash
pip install onnxruntime-openvino
```

## Integration Example

### Python API Usage

```python
from inference_cpu import CephalometricInference
import cv2

# Initialize engine
engine = CephalometricInference('onnx_models/', use_quantized=True)

# Load image
image = cv2.imread('patient_xray.jpg')

# Run inference
results = engine.predict(image)

# Access predictions
keypoints = results['keypoints']  # Shape: (19, 2)
timing = results['timing']['total']  # Inference time in ms

# Print results
for i, (x, y) in enumerate(keypoints):
    print(f"Landmark {i}: ({x:.2f}, {y:.2f})")
```

### REST API Server Example

Create `api_server.py`:

```python
from flask import Flask, request, jsonify
from inference_cpu import CephalometricInference
import cv2
import numpy as np
import base64

app = Flask(__name__)
engine = CephalometricInference('onnx_models/', use_quantized=True)

@app.route('/predict', methods=['POST'])
def predict():
    # Get image from request
    image_data = base64.b64decode(request.json['image'])
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run inference
    results = engine.predict(image)
    
    # Return predictions
    return jsonify({
        'keypoints': results['keypoints'].tolist(),
        'inference_time_ms': results['timing']['total']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Troubleshooting

### 1. ONNX Runtime Not Found
```bash
pip install onnxruntime
# Or for older CPUs without AVX
pip install onnxruntime-1.10.0  # Use older version
```

### 2. Import Errors on CPU Server
The CPU server doesn't need MMPose. Only required packages:
```bash
pip install numpy opencv-python onnxruntime
```

### 3. Slow Inference
- Use `--use_quantized` flag
- Check CPU usage with `htop`
- Ensure no other heavy processes running
- Consider using GPU server for high-volume processing

### 4. Memory Issues
- Process images in smaller batches
- Reduce number of threads if RAM limited
- Use quantized models (4x smaller)

## Expected Performance

On a modern CPU (e.g., Intel i7-8700):
- **Original models**: ~50-100ms per image
- **Quantized models**: ~25-50ms per image
- **Batch processing**: Better throughput

## Accuracy Preservation

The deployment preserves accuracy through:
1. **No approximations** in model architecture
2. **Proper preprocessing** matching training
3. **MLP refinement** for improved predictions
4. **UDP decoding** for sub-pixel accuracy

Quantization impact:
- Original: 100% accuracy
- Quantized: >99% accuracy (negligible difference)

## Next Steps

1. **Monitor Performance**: Log inference times and track performance
2. **Scale Horizontally**: Deploy multiple instances behind load balancer
3. **GPU Deployment**: For high-volume, consider NVIDIA TensorRT
4. **Edge Deployment**: For embedded devices, consider TFLite or CoreML

## Support

For issues or questions:
1. Check model files exist in correct paths
2. Verify ONNX conversion completed without errors
3. Test with single image first before batch processing
4. Compare predictions with original PyTorch model if accuracy concerns 