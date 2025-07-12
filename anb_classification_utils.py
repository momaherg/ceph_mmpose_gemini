import numpy as np

# Try to import torch, but make the module work without it
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    

def calculate_anb_angle(landmarks):
    """
    Calculate ANB angle from landmarks using the method from train_improved_v4.py.
    
    This matches the implementation used for training data labeling to ensure consistency.
    
    Args:
        landmarks: numpy array or torch tensor of shape (..., 19, 2) containing landmark coordinates
    
    Returns:
        ANB angles in degrees with the same batch shape as input
    """
    if HAS_TORCH and isinstance(landmarks, torch.Tensor):
        landmarks_np = landmarks.detach().cpu().numpy()
        is_tensor = True
    else:
        landmarks_np = landmarks
        is_tensor = False
    
    # Get the relevant points
    # Sella is index 0, Nasion is index 1, A-point is index 2, B-point is index 3
    Sella = landmarks_np[..., 0, :]    # Shape: (..., 2)
    Nasion = landmarks_np[..., 1, :]   # Shape: (..., 2)
    A_point = landmarks_np[..., 2, :]  # Shape: (..., 2)
    B_point = landmarks_np[..., 3, :]  # Shape: (..., 2)
    
    # Calculate angle at a vertex (matching _angle function from train_improved_v4.py)
    def angle_at_vertex(p1, vertex, p2):
        """Return the angle (deg) at vertex formed by p1-vertex-p2."""
        v1 = p1 - vertex
        v2 = p2 - vertex
        
        # Calculate norms
        v1_norm = np.linalg.norm(v1, axis=-1, keepdims=True)
        v2_norm = np.linalg.norm(v2, axis=-1, keepdims=True)
        
        # Handle zero vectors
        valid_mask = (v1_norm.squeeze(-1) > 1e-8) & (v2_norm.squeeze(-1) > 1e-8)
        
        # Initialize result
        angle_degrees = np.zeros(v1.shape[:-1])
        
        if np.any(valid_mask):
            # Normalize vectors
            v1_normalized = v1[valid_mask] / v1_norm[valid_mask]
            v2_normalized = v2[valid_mask] / v2_norm[valid_mask]
            
            # Calculate dot product
            dot_product = np.sum(v1_normalized * v2_normalized, axis=-1)
            
            # Clip to valid range for arccos
            dot_product = np.clip(dot_product, -1.0, 1.0)
            
            # Calculate angle
            angle_radians = np.arccos(dot_product)
            angle_degrees[valid_mask] = np.degrees(angle_radians)
        
        return angle_degrees
    
    # Calculate SNA and SNB angles
    sna = angle_at_vertex(Sella, Nasion, A_point)
    snb = angle_at_vertex(Sella, Nasion, B_point)
    
    # ANB = SNA - SNB (as in train_improved_v4.py)
    anb_angle = sna - snb
    
    if is_tensor and HAS_TORCH:
        return torch.from_numpy(anb_angle).to(landmarks.device)
    else:
        return anb_angle


def classify_from_anb_angle(anb_angle):
    """
    Classify skeletal pattern based on ANB angle.
    
    NOTE: These thresholds match those used in train_improved_v4.py to ensure consistency
    with the training data labeling.
    
    Args:
        anb_angle: ANB angle in degrees (numpy array or torch tensor)
    
    Returns:
        Classification labels:
        - 0: Skeletal Class I (2 <= ANB <= 4)
        - 1: Skeletal Class II (ANB > 4)
        - 2: Skeletal Class III (ANB < 2)
    """
    if HAS_TORCH and isinstance(anb_angle, torch.Tensor):
        # For torch tensors
        labels = torch.zeros_like(anb_angle, dtype=torch.long)
        labels[anb_angle > 4] = 1   # Class II
        labels[anb_angle < 2] = 2   # Class III
        # Class I (2 <= ANB <= 4) remains as 0
    else:
        # For numpy arrays
        labels = np.zeros_like(anb_angle, dtype=np.int64)
        labels[anb_angle > 4] = 1   # Class II
        labels[anb_angle < 2] = 2   # Class III
        # Class I (2 <= ANB <= 4) remains as 0
    
    return labels


def get_class_name(class_label):
    """
    Get human-readable class name from label.
    
    Args:
        class_label: Integer class label (0, 1, or 2)
    
    Returns:
        String class name
    """
    class_names = {
        0: "Skeletal Class I",
        1: "Skeletal Class II", 
        2: "Skeletal Class III"
    }
    return class_names.get(class_label, "Unknown") 