from mmpose.datasets.transforms.loading import LoadImage
from mmpose.datasets.transforms.formatting import PackPoseInputs
from mmpose.registry import TRANSFORMS
import numpy as np
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData

# Initialize scope to ensure registry is ready
init_default_scope('mmpose')

print("Registering LoadImageNumpy in the TRANSFORMS registry...")

@TRANSFORMS.register_module(force=True) # Force registration to override any existing
class LoadImageNumpy(LoadImage):
    """Load an image from results['img'] which is already a numpy array.
    The 'img' key is populated by CustomCephalometricDataset.
    """
    
    def __init__(self, **kwargs):
        print("LoadImageNumpy.__init__ called")
        super().__init__(**kwargs)

    def transform(self, results: dict) -> dict:
        """Load an image from results['img'] which is already a numpy array.
        """
        img = results['img']
        if not isinstance(img, np.ndarray):
            raise TypeError(f"Image should be a NumPy array, but got {type(img)}")
        if img is None: # Should not happen if dataset guarantees 'img'
            raise ValueError('Image is not loaded, results["img"] is None.')

        # Assuming image from dataset is HWC.
        # MMPose pipelines usually handle channel order (e.g. BGR) and ToTensor conversion later.
        
        results['img_shape'] = img.shape[:2] # (h, w)
        results['ori_shape'] = img.shape[:2] # (h, w)
        # 'img_path' should be set by the dataset if needed for metainfo,
        # otherwise, it can be a placeholder or the 'patient_id'.
        # results['img_path'] = results.get('img_path', str(results.get('img_id', '')))
        return results

print("Registering CustomPackPoseInputs in the TRANSFORMS registry...")

@TRANSFORMS.register_module(force=True)
class CustomPackPoseInputs(PackPoseInputs):
    """Custom PackPoseInputs that properly handles bbox_scores and class labels for cephalometric dataset."""
    
    def _compute_class_from_anb(self, keypoints: np.ndarray) -> int:
        """Compute class from ANB angle calculated from ground truth landmarks.
        
        Args:
            keypoints: [num_keypoints, 2] array of landmark coordinates
            
        Returns:
            int: Class label (0=Class I, 1=Class II, 2=Class III)
        """
        try:
            # Import landmark info
            import cephalometric_dataset_info
            landmark_names = cephalometric_dataset_info.landmark_names_in_order
            
            # Create landmark index map
            landmark_idx = {name: i for i, name in enumerate(landmark_names)}
            
            # Get required landmarks
            if keypoints.shape[0] == 1:  # Handle (1, num_keypoints, 2) shape
                keypoints = keypoints[0]
                
            sella = keypoints[landmark_idx['sella']]
            nasion = keypoints[landmark_idx['nasion']]
            a_point = keypoints[landmark_idx['A_point']]
            b_point = keypoints[landmark_idx['B_point']]
            
            # Check if landmarks are valid (not zero)
            if np.all(sella == 0) or np.all(nasion == 0) or np.all(a_point == 0) or np.all(b_point == 0):
                return 0  # Default to Class I if landmarks are invalid
            
            # Calculate angles
            def angle(p1, p2, p3):
                """Calculate angle at p2 between vectors p2->p1 and p2->p3."""
                v1 = p1 - p2
                v2 = p3 - p2
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                return np.degrees(np.arccos(cos_angle))
            
            # Calculate SNA and SNB angles
            sna = angle(sella, nasion, a_point)
            snb = angle(sella, nasion, b_point)
            
            # Calculate ANB angle
            anb = sna - snb
            
            # Classify based on ANB angle
            # Skeletal Class I:  0<x<4 
            # Skeletal Class II: x≥4 
            # Skeletal Class III: x≤0
            if anb >= 4:
                return 1  # Class II
            elif anb <= 0:
                return 2  # Class III
            else:
                return 0  # Class I
                
        except Exception as e:
            # If anything goes wrong, default to Class I
            print(f"Warning: Could not compute ANB angle, defaulting to Class I. Error: {e}")
            return 0
    
    def transform(self, results: dict) -> dict:
        """Transform function to pack pose inputs, including bbox_scores and class labels."""
        
        # Call parent transform first
        packed_results = super().transform(results)
        
        # Ensure bbox_scores are properly added to gt_instances
        if 'data_samples' in packed_results:
            data_sample = packed_results['data_samples']
            
            # Check if gt_instances exists and bbox_scores is in the original results
            if hasattr(data_sample, 'gt_instances') and 'bbox_scores' in results:
                gt_instances = data_sample.gt_instances
                
                # Add bbox_scores to gt_instances
                if isinstance(results['bbox_scores'], np.ndarray):
                    import torch
                    gt_instances.bbox_scores = torch.from_numpy(results['bbox_scores']).float()
                else:
                    gt_instances.bbox_scores = results['bbox_scores']
                
                # Debug print removed - bbox_scores successfully added
            
            # Compute class labels from ground truth landmarks
            if hasattr(data_sample, 'gt_instances') and 'keypoints' in results:
                gt_instances = data_sample.gt_instances
                
                # Get ground truth keypoints
                gt_keypoints = results['keypoints']
                if isinstance(gt_keypoints, np.ndarray):
                    # Compute class from ANB angle
                    class_label = self._compute_class_from_anb(gt_keypoints)
                    
                    # Add as labels field for the model
                    import torch
                    gt_instances.labels = torch.tensor([class_label], dtype=torch.long)
                    
                    # Also store the computed class in results for debugging
                    results['computed_class'] = class_label
        
        return packed_results

# Print confirmation that the module has been loaded and registration attempted
print(f"LoadImageNumpy registered: {TRANSFORMS.get('LoadImageNumpy') is not None}")

# This will output all registered transforms for debugging
print("Registered transforms:", list(TRANSFORMS.module_dict.keys())) 