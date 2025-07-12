import numpy as np
import torch

def calculate_anb_angle(landmarks):
    """
    Calculate ANB angle from landmarks.
    
    Args:
        landmarks: numpy array or torch tensor of shape (..., 19, 2) containing landmark coordinates
    
    Returns:
        ANB angles in degrees with the same batch shape as input
    """
    if isinstance(landmarks, torch.Tensor):
        landmarks_np = landmarks.detach().cpu().numpy()
        is_tensor = True
    else:
        landmarks_np = landmarks
        is_tensor = False
    
    # Get the relevant points
    # A-point is index 2, Nasion is index 1, B-point is index 3
    A_point = landmarks_np[..., 2, :]  # Shape: (..., 2)
    Nasion = landmarks_np[..., 1, :]   # Shape: (..., 2)
    B_point = landmarks_np[..., 3, :]  # Shape: (..., 2)
    
    # Calculate vectors
    # Vector from Nasion to A-point
    NA_vector = A_point - Nasion  # Shape: (..., 2)
    # Vector from Nasion to B-point
    NB_vector = B_point - Nasion  # Shape: (..., 2)
    
    # Calculate angle between vectors using dot product
    # angle = arccos(dot(NA, NB) / (|NA| * |NB|))
    dot_product = np.sum(NA_vector * NB_vector, axis=-1)  # Shape: (...)
    
    NA_norm = np.linalg.norm(NA_vector, axis=-1) + 1e-8  # Add epsilon to avoid division by zero
    NB_norm = np.linalg.norm(NB_vector, axis=-1) + 1e-8
    
    cos_angle = np.clip(dot_product / (NA_norm * NB_norm), -1.0, 1.0)  # Clip to valid range for arccos
    angle_radians = np.arccos(cos_angle)
    
    # Convert to degrees
    angle_degrees = np.degrees(angle_radians)
    
    # Determine sign of angle using cross product
    # If cross product is negative, B is behind A (positive ANB)
    # If cross product is positive, B is in front of A (negative ANB)
    cross_product = NA_vector[..., 0] * NB_vector[..., 1] - NA_vector[..., 1] * NB_vector[..., 0]
    angle_degrees = np.where(cross_product > 0, -angle_degrees, angle_degrees)
    
    if is_tensor:
        return torch.from_numpy(angle_degrees).to(landmarks.device)
    else:
        return angle_degrees


def classify_from_anb_angle(anb_angle):
    """
    Classify skeletal pattern based on ANB angle.
    
    Args:
        anb_angle: ANB angle in degrees (numpy array or torch tensor)
    
    Returns:
        Classification labels:
        - 0: Skeletal Class I (0 < ANB < 4)
        - 1: Skeletal Class II (ANB >= 4)
        - 2: Skeletal Class III (ANB <= 0)
    """
    if isinstance(anb_angle, torch.Tensor):
        # For torch tensors
        labels = torch.zeros_like(anb_angle, dtype=torch.long)
        labels[anb_angle >= 4] = 1  # Class II
        labels[anb_angle <= 0] = 2  # Class III
        # Class I (0 < ANB < 4) remains as 0
    else:
        # For numpy arrays
        labels = np.zeros_like(anb_angle, dtype=np.int64)
        labels[anb_angle >= 4] = 1  # Class II
        labels[anb_angle <= 0] = 2  # Class III
        # Class I (0 < ANB < 4) remains as 0
    
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