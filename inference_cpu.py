#!/usr/bin/env python3
"""
CPU Inference Script for Cephalometric Landmark Detection
Runs HRNetV2 + MLP models using ONNX Runtime on CPU
"""

import os
import sys
import numpy as np
import cv2
import json
import argparse
import time
from typing import Dict, List, Tuple, Optional
import onnxruntime as ort

# For optional visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available. Visualization disabled.")


class CephalometricInference:
    """CPU inference engine for cephalometric landmark detection."""
    
    def __init__(self, onnx_dir: str, use_quantized: bool = False, skip_mlp: bool = False):
        """
        Initialize inference engine.
        
        Args:
            onnx_dir: Directory containing ONNX models and config
            use_quantized: Whether to use quantized models for faster inference
        """
        self.onnx_dir = onnx_dir
        self.use_quantized = use_quantized
        self.skip_mlp = skip_mlp
        
        # Load configuration
        config_path = os.path.join(onnx_dir, 'onnx_config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        print(f"✅ Loaded configuration from: {config_path}")
        print(f"📊 Model type: {self.config['model_type']}")
        print(f"📏 Input size: {self.config['input_size']}")
        print(f"🎯 Number of keypoints: {self.config['num_keypoints']}")
        
        # Initialize ONNX sessions
        self._init_sessions()
        
        # Load scalers if available
        self.scalers = None
        scalers_path = os.path.join(onnx_dir, 'mlp_model_scalers.pkl')
        if os.path.exists(scalers_path):
            try:
                import joblib
                self.scalers = joblib.load(scalers_path)
                print(f"✅ Loaded normalization scalers")
            except:
                print(f"⚠️  Could not load scalers, proceeding without normalization")
    
    def _init_sessions(self):
        """Initialize ONNX Runtime sessions."""
        # Session options for CPU optimization
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 4
        sess_options.intra_op_num_threads = 4
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # CPU providers
        providers = ['CPUExecutionProvider']
        
        # Load HRNet model
        hrnet_model_name = 'hrnet_model.onnx'
        if self.use_quantized and os.path.exists(os.path.join(self.onnx_dir, 'hrnet_model_quantized.onnx')):
            hrnet_model_name = 'hrnet_model_quantized.onnx'
            print(f"🚀 Using quantized HRNet model for faster inference")
        
        hrnet_path = os.path.join(self.onnx_dir, hrnet_model_name)
        self.hrnet_session = ort.InferenceSession(hrnet_path, sess_options, providers=providers)
        print(f"✅ Loaded HRNet model: {hrnet_model_name}")
        
        # Load MLP model if available and not skipped
        self.mlp_session = None
        if self.config['has_mlp_refinement'] and not self.skip_mlp:
            mlp_model_name = 'mlp_model.onnx'
            if self.use_quantized and os.path.exists(os.path.join(self.onnx_dir, 'mlp_model_quantized.onnx')):
                mlp_model_name = 'mlp_model_quantized.onnx'
                print(f"🚀 Using quantized MLP model for faster inference")
            
            mlp_path = os.path.join(self.onnx_dir, mlp_model_name)
            if os.path.exists(mlp_path):
                self.mlp_session = ort.InferenceSession(mlp_path, sess_options, providers=providers)
                print(f"✅ Loaded MLP refinement model: {mlp_model_name}")
            else:
                print(f"⚠️  MLP model not found, using HRNet predictions only")
        elif self.skip_mlp:
            print(f"ℹ️  MLP refinement skipped by user request (--no_mlp flag)")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image (H, W, C) in BGR format
            
        Returns:
            Preprocessed image tensor (1, C, H, W)
        """
        # Resize to model input size
        input_size = tuple(self.config['input_size'])
        image_resized = cv2.resize(image, input_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB if needed
        if self.config['preprocessing']['bgr_to_rgb']:
            image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        # Convert to float and normalize
        image_float = image_resized.astype(np.float32)
        
        # Apply ImageNet normalization
        mean = np.array(self.config['preprocessing']['mean'], dtype=np.float32)
        std = np.array(self.config['preprocessing']['std'], dtype=np.float32)
        image_normalized = (image_float - mean) / std
        
        # Transpose to (C, H, W) and add batch dimension
        image_tensor = np.transpose(image_normalized, (2, 0, 1))
        image_batch = np.expand_dims(image_tensor, axis=0)
        
        return image_batch
    
    def decode_heatmaps(self, heatmaps: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """
        Decode heatmaps to keypoint coordinates in original image space.
        
        Args:
            heatmaps: Model output heatmaps (B, K, H, W)
            original_shape: Original image shape (H, W)
            
        Returns:
            Keypoint coordinates (B, K, 2) in original image coordinates
        """
        batch_size, num_keypoints, h, w = heatmaps.shape
        keypoints = np.zeros((batch_size, num_keypoints, 2))
        
        # Model input size (what the image was resized to)
        model_input_h, model_input_w = self.config['input_size']
        
        # Original image size
        original_h, original_w = original_shape
        
        for b in range(batch_size):
            for k in range(num_keypoints):
                heatmap = heatmaps[b, k]
                
                # Find peak location
                idx = np.argmax(heatmap)
                y, x = np.unravel_index(idx, heatmap.shape)
                
                # Apply UDP (Ultra-Decoding Process) for sub-pixel accuracy
                if self.config['postprocessing']['use_udp'] and x > 0 and x < w-1 and y > 0 and y < h-1:
                    # Get surrounding values
                    dx = 0.5 * (heatmap[y, x+1] - heatmap[y, x-1])
                    dy = 0.5 * (heatmap[y+1, x] - heatmap[y-1, x])
                    
                    # Refine position
                    x_refined = x + np.clip(dx, -0.5, 0.5)
                    y_refined = y + np.clip(dy, -0.5, 0.5)
                else:
                    x_refined = x
                    y_refined = y
                
                # First scale from heatmap size to model input size
                x_model = x_refined * (model_input_w / w)
                y_model = y_refined * (model_input_h / h)
                
                # Then scale from model input size to original image size
                x_original = x_model * (original_w / model_input_w)
                y_original = y_model * (original_h / model_input_h)
                
                keypoints[b, k, 0] = x_original
                keypoints[b, k, 1] = y_original
        
        return keypoints
    
    def refine_with_mlp(self, keypoints: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """
        Refine keypoints using MLP model.
        
        Args:
            keypoints: Initial keypoints (B, K, 2) in original image coordinates
            original_shape: Original image shape (H, W)
            
        Returns:
            Refined keypoints (B, K, 2) in original image coordinates
        """
        if self.mlp_session is None:
            return keypoints
        
        batch_size = keypoints.shape[0]
        refined_keypoints = np.zeros_like(keypoints)
        
        # MLP was trained on specific scale - need to determine this
        # Based on your training, it's likely 224x224 or 384x384
        # Check the training scale from config
        mlp_training_scale = 224  # Default, will be overridden if found in config
        if hasattr(self, 'config') and 'mlp_training_scale' in self.config:
            mlp_training_scale = self.config['mlp_training_scale']
        
        # Scale keypoints to MLP training scale before processing
        original_h, original_w = original_shape
        scale_to_mlp_x = mlp_training_scale / original_w
        scale_to_mlp_y = mlp_training_scale / original_h
        
        for b in range(batch_size):
            # Scale keypoints to MLP training scale
            kpts_mlp_scale = keypoints[b].copy()
            kpts_mlp_scale[:, 0] *= scale_to_mlp_x  # Scale X coordinates
            kpts_mlp_scale[:, 1] *= scale_to_mlp_y  # Scale Y coordinates
            
            # Flatten keypoints to 38-D vector
            kpts_flat = kpts_mlp_scale.flatten().astype(np.float32)
            
            # Apply normalization if scalers available
            if self.scalers and 'input' in self.scalers:
                kpts_normalized = self.scalers['input'].transform(kpts_flat.reshape(1, -1))
                kpts_input = kpts_normalized.astype(np.float32)
            else:
                kpts_input = kpts_flat.reshape(1, -1)
            
            # Run MLP inference
            mlp_inputs = {self.mlp_session.get_inputs()[0].name: kpts_input}
            mlp_outputs = self.mlp_session.run(None, mlp_inputs)
            refined_flat = mlp_outputs[0][0]
            
            # Denormalize if needed
            if self.scalers and 'target' in self.scalers:
                refined_flat = self.scalers['target'].inverse_transform(refined_flat.reshape(1, -1))[0]
            
            # Reshape back to keypoints
            refined_mlp_scale = refined_flat.reshape(-1, 2)
            
            # Scale back to original image coordinates
            refined_keypoints[b] = refined_mlp_scale.copy()
            refined_keypoints[b][:, 0] /= scale_to_mlp_x  # Scale X back
            refined_keypoints[b][:, 1] /= scale_to_mlp_y  # Scale Y back
        
        return refined_keypoints
    
    def predict(self, image: np.ndarray) -> Dict:
        """
        Run inference on an image.
        
        Args:
            image: Input image (H, W, C) in BGR format
            
        Returns:
            Dictionary with predictions and metadata
        """
        start_time = time.time()
        
        # Store original shape
        original_shape = image.shape[:2]
        
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        preprocess_time = time.time() - start_time
        
        # Run HRNet inference
        hrnet_start = time.time()
        hrnet_inputs = {self.hrnet_session.get_inputs()[0].name: image_tensor}
        hrnet_outputs = self.hrnet_session.run(None, hrnet_inputs)
        heatmaps = hrnet_outputs[0]
        hrnet_time = time.time() - hrnet_start
        
        # Decode heatmaps to keypoints
        decode_start = time.time()
        keypoints = self.decode_heatmaps(heatmaps, original_shape)
        decode_time = time.time() - decode_start
        
        # Refine with MLP if available
        mlp_time = 0
        if self.mlp_session is not None:
            mlp_start = time.time()
            keypoints_refined = self.refine_with_mlp(keypoints, original_shape)
            mlp_time = time.time() - mlp_start
        else:
            keypoints_refined = keypoints
        
        total_time = time.time() - start_time
        
        # Prepare results
        results = {
            'keypoints': keypoints_refined[0],  # Remove batch dimension
            'keypoints_initial': keypoints[0],  # HRNet predictions before refinement
            'original_shape': original_shape,
            'input_shape': self.config['input_size'],
            'timing': {
                'preprocess': preprocess_time * 1000,  # ms
                'hrnet': hrnet_time * 1000,
                'decode': decode_time * 1000,
                'mlp': mlp_time * 1000,
                'total': total_time * 1000
            }
        }
        
        return results
    
    def visualize_predictions(self, image: np.ndarray, results: Dict, save_path: Optional[str] = None):
        """Visualize predictions on image."""
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  Visualization not available (matplotlib not installed)")
            return
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Convert BGR to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Plot initial predictions (HRNet only)
        ax1.imshow(image_rgb)
        keypoints_initial = results['keypoints_initial']
        ax1.scatter(keypoints_initial[:, 0], keypoints_initial[:, 1], 
                   c='red', s=80, marker='o', label='HRNet', edgecolors='white', linewidth=1)
        
        # Add landmark numbers for easier identification
        for i, (x, y) in enumerate(keypoints_initial):
            ax1.text(x+5, y+5, str(i), fontsize=8, color='red', weight='bold')
        
        ax1.set_title(f'Initial Predictions (HRNet) - Image: {image.shape[1]}x{image.shape[0]}')
        ax1.legend()
        ax1.axis('off')
        
        # Plot refined predictions (HRNet + MLP)
        ax2.imshow(image_rgb)
        keypoints_refined = results['keypoints']
        ax2.scatter(keypoints_refined[:, 0], keypoints_refined[:, 1], 
                   c='green', s=80, marker='o', label='HRNet + MLP', edgecolors='white', linewidth=1)
        
        # Add landmark numbers
        for i, (x, y) in enumerate(keypoints_refined):
            ax2.text(x+5, y+5, str(i), fontsize=8, color='green', weight='bold')
        
        # Show refinement arrows if MLP was used
        if 'mlp' in results['timing'] and results['timing']['mlp'] > 0:
            for i in range(len(keypoints_initial)):
                dx = keypoints_refined[i, 0] - keypoints_initial[i, 0]
                dy = keypoints_refined[i, 1] - keypoints_initial[i, 1]
                distance = np.sqrt(dx**2 + dy**2)
                if distance > 1.0:  # Show refinements larger than 1 pixel
                    ax2.arrow(keypoints_initial[i, 0], keypoints_initial[i, 1],
                             dx, dy, head_width=5, head_length=3, 
                             fc='blue', ec='blue', alpha=0.5, linewidth=1)
        
        ax2.set_title(f'Refined Predictions (HRNet + MLP) - Model Input: {self.config["input_size"]}')
        ax2.legend()
        ax2.axis('off')
        
        # Add coordinate info
        fig.text(0.5, 0.02, 
                f'Scaling: {image.shape[1]}x{image.shape[0]} → {self.config["input_size"][0]}x{self.config["input_size"][1]} → {image.shape[1]}x{image.shape[0]}',
                ha='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Run inference on multiple images."""
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"\n📸 Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Failed to load image: {image_path}")
                continue
            
            # Run inference
            result = self.predict(image)
            result['image_path'] = image_path
            results.append(result)
            
            # Print timing
            print(f"⏱️  Inference time: {result['timing']['total']:.2f}ms")
            print(f"   - Preprocessing: {result['timing']['preprocess']:.2f}ms")
            print(f"   - HRNet: {result['timing']['hrnet']:.2f}ms")
            print(f"   - Decoding: {result['timing']['decode']:.2f}ms")
            if result['timing']['mlp'] > 0:
                print(f"   - MLP refinement: {result['timing']['mlp']:.2f}ms")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='CPU inference for cephalometric landmark detection')
    parser.add_argument('--onnx_dir', type=str, default='onnx_models',
                        help='Directory containing ONNX models (default: onnx_models)')
    parser.add_argument('--image', type=str, 
                        help='Path to single image for inference')
    parser.add_argument('--image_list', type=str,
                        help='Path to text file containing list of images')
    parser.add_argument('--use_quantized', action='store_true',
                        help='Use quantized models for faster inference')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predictions')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Output directory for results (default: inference_results)')
    parser.add_argument('--save_json', action='store_true',
                        help='Save predictions to JSON file')
    parser.add_argument('--no_mlp', action='store_true',
                        help='Skip MLP refinement and use only HRNet predictions')
    
    args = parser.parse_args()
    
    # Check ONNX Runtime
    try:
        import onnxruntime
        print(f"✅ ONNX Runtime version: {onnxruntime.__version__}")
    except ImportError:
        print("❌ ERROR: onnxruntime not installed")
        print("Install with: pip install onnxruntime")
        return
    
    # Initialize inference engine
    print(f"\n{'='*70}")
    print(f"🚀 INITIALIZING CPU INFERENCE ENGINE")
    print(f"{'='*70}")
    
    engine = CephalometricInference(args.onnx_dir, use_quantized=args.use_quantized, skip_mlp=args.no_mlp)
    
    # Prepare image list
    if args.image:
        image_paths = [args.image]
    elif args.image_list:
        with open(args.image_list, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
    else:
        # Demo with a test image if available
        print("\n⚠️  No input specified. Looking for test images...")
        test_patterns = ['test*.jpg', 'test*.png', 'demo*.jpg', 'demo*.png']
        image_paths = []
        import glob
        for pattern in test_patterns:
            image_paths.extend(glob.glob(pattern))
        
        if not image_paths:
            print("❌ No test images found. Please specify --image or --image_list")
            return
        
        print(f"📸 Found {len(image_paths)} test images")
    
    # Run inference
    print(f"\n{'='*70}")
    print(f"🎯 RUNNING INFERENCE")
    print(f"{'='*70}")
    
    results = engine.predict_batch(image_paths)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Debug: Print coordinate ranges
    if results:
        print(f"\n📊 Coordinate Ranges:")
        for i, result in enumerate(results[:1]):  # Just first result
            kpts = result['keypoints']
            orig_shape = result['original_shape']
            print(f"   Image {i+1} shape: {orig_shape}")
            print(f"   Model input: {result['input_shape']}")
            print(f"   X range: {kpts[:, 0].min():.1f} - {kpts[:, 0].max():.1f}")
            print(f"   Y range: {kpts[:, 1].min():.1f} - {kpts[:, 1].max():.1f}")
    
    # Save results
    if args.save_json:
        output_json = os.path.join(args.output_dir, 'predictions.json')
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = []
        for result in results:
            json_result = {
                'image_path': result['image_path'],
                'keypoints': result['keypoints'].tolist(),
                'keypoints_initial': result['keypoints_initial'].tolist(),
                'original_shape': result['original_shape'],
                'input_shape': result['input_shape'],
                'timing': result['timing']
            }
            json_results.append(json_result)
        
        with open(output_json, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n✅ Predictions saved to: {output_json}")
    
    # Visualize if requested
    if args.visualize:
        print(f"\n📊 Creating visualizations...")
        for i, result in enumerate(results):
            image = cv2.imread(result['image_path'])
            vis_path = os.path.join(args.output_dir, f'visualization_{i+1}.png')
            engine.visualize_predictions(image, result, save_path=vis_path)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"📊 INFERENCE SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Processed {len(results)} images")
    
    if results:
        avg_time = np.mean([r['timing']['total'] for r in results])
        print(f"⏱️  Average inference time: {avg_time:.2f}ms ({1000/avg_time:.1f} FPS)")
        
        # Breakdown
        avg_preprocess = np.mean([r['timing']['preprocess'] for r in results])
        avg_hrnet = np.mean([r['timing']['hrnet'] for r in results])
        avg_decode = np.mean([r['timing']['decode'] for r in results])
        
        print(f"\n📊 Average timing breakdown:")
        print(f"   - Preprocessing: {avg_preprocess:.2f}ms ({avg_preprocess/avg_time*100:.1f}%)")
        print(f"   - HRNet: {avg_hrnet:.2f}ms ({avg_hrnet/avg_time*100:.1f}%)")
        print(f"   - Decoding: {avg_decode:.2f}ms ({avg_decode/avg_time*100:.1f}%)")
        
        if engine.mlp_session is not None:
            avg_mlp = np.mean([r['timing']['mlp'] for r in results])
            print(f"   - MLP refinement: {avg_mlp:.2f}ms ({avg_mlp/avg_time*100:.1f}%)")
    
    print(f"\n💡 Tips for faster inference:")
    print(f"   - Use --use_quantized flag for ~2x speedup")
    print(f"   - Batch processing for multiple images")
    print(f"   - Use multi-threading for parallel processing")


if __name__ == "__main__":
    main() 