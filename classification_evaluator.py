"""
Custom evaluator for skeletal classification metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Sequence
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
from collections import defaultdict
import torch


@METRICS.register_module()
class ClassificationMetric(BaseMetric):
    """Evaluation metric for skeletal classification.
    
    This metric calculates:
    - Overall accuracy
    - Per-class accuracy 
    - Confusion matrix
    - Classification report
    
    Args:
        num_classes (int): Number of classes (default: 3 for skeletal classes)
        collect_device (str): Device name used for collecting results from 
            different ranks during distributed training. Must be 'cpu' or 'gpu'.
            Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix 
            will be used instead. Default: None.
    """
    
    default_prefix = 'classification'
    
    def __init__(self,
                 num_classes: int = 3,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.num_classes = num_classes
        self.class_names = ['Class I', 'Class II', 'Class III']
        
    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.
        
        Args:
            data_batch (Sequence[dict]): A batch of data from dataloader.
            data_samples (Sequence[dict]): A batch of outputs from model.
        """
        for data_sample in data_samples:
            # Get ground truth classification
            if hasattr(data_sample, 'metainfo') and 'gt_classification' in data_sample.metainfo:
                gt_class = data_sample.metainfo['gt_classification']
            elif hasattr(data_sample, 'gt_classification'):
                gt_class = data_sample.gt_classification
            else:
                # If gt_classification is not set, compute from ground truth keypoints
                try:
                    import sys
                    sys.path.insert(0, '.')
                    from anb_classification_utils import calculate_anb_angle, classify_from_anb_angle
                    
                    gt_keypoints = data_sample.gt_instances.keypoints  # Shape: (1, 19, 2)
                    anb_angle = calculate_anb_angle(gt_keypoints)
                    gt_class = classify_from_anb_angle(anb_angle)
                    if isinstance(gt_class, torch.Tensor):
                        gt_class = gt_class.item()
                except Exception as e:
                    # Skip if we can't compute classification
                    continue
            
            # Get predicted classification
            pred_class = None
            if hasattr(data_sample, 'pred_classification'):
                pred_class = data_sample.pred_classification
            elif hasattr(data_sample, 'pred_instances'):
                # Try to compute from predicted keypoints
                try:
                    import sys
                    sys.path.insert(0, '.')
                    from anb_classification_utils import calculate_anb_angle, classify_from_anb_angle
                    
                    pred_keypoints = data_sample.pred_instances.keypoints  # Shape: (1, 19, 2)
                    anb_angle = calculate_anb_angle(pred_keypoints)
                    pred_class = classify_from_anb_angle(anb_angle)
                    if isinstance(pred_class, torch.Tensor):
                        pred_class = pred_class.item()
                except Exception as e:
                    # Skip if we can't compute classification
                    continue
            
            if gt_class is not None and pred_class is not None:
                # Store result
                result = {
                    'gt_class': int(gt_class),
                    'pred_class': int(pred_class),
                }
                
                # Also store probabilities if available
                if hasattr(data_sample, 'pred_classification_probs'):
                    result['pred_probs'] = data_sample.pred_classification_probs
                    
                self.results.append(result)
    
    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.
        
        Args:
            results (List[dict]): The processed results of each batch.
            
        Returns:
            Dict[str, float]: The computed metrics. The keys are the names
                of the metrics, and the values are corresponding results.
        """
        if not results:
            return {}
            
        # Extract predictions and ground truths
        gt_classes = np.array([r['gt_class'] for r in results])
        pred_classes = np.array([r['pred_class'] for r in results])
        
        # Calculate overall accuracy
        accuracy = np.mean(gt_classes == pred_classes)
        
        # Calculate per-class metrics
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        
        for gt, pred in zip(gt_classes, pred_classes):
            class_total[gt] += 1
            if gt == pred:
                class_correct[gt] += 1
            confusion_matrix[gt, pred] += 1
        
        # Compute per-class accuracy
        per_class_acc = {}
        for i in range(self.num_classes):
            if class_total[i] > 0:
                per_class_acc[f'acc_{self.class_names[i]}'] = class_correct[i] / class_total[i]
            else:
                per_class_acc[f'acc_{self.class_names[i]}'] = 0.0
        
        # Calculate macro-averaged metrics
        valid_classes = [i for i in range(self.num_classes) if class_total[i] > 0]
        if valid_classes:
            macro_acc = np.mean([class_correct[i] / class_total[i] for i in valid_classes])
        else:
            macro_acc = 0.0
        
        # Build metrics dictionary
        metrics = {
            'accuracy': accuracy,
            'macro_accuracy': macro_acc,
        }
        metrics.update(per_class_acc)
        
        # Add confusion matrix info
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                metrics[f'conf_{self.class_names[i]}_to_{self.class_names[j]}'] = int(confusion_matrix[i, j])
        
        # Add class distribution
        for i in range(self.num_classes):
            metrics[f'n_{self.class_names[i]}'] = int(class_total[i])
        
        return metrics 