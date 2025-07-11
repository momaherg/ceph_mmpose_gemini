#!/usr/bin/env python3
"""
Classification Metric for Multi-task Cephalometric Model
Evaluates patient classification performance (Class I, II, III)
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmpose.registry import METRICS
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings


@METRICS.register_module()
class ClassificationMetric(BaseMetric):
    """Classification metric for multi-task cephalometric model.
    
    Evaluates classification performance including accuracy, precision, recall, and F1-score.
    
    Args:
        num_classes (int): Number of classes (default: 3 for Class I, II, III)
        mode (str): Averaging mode for metrics ('macro', 'micro', 'weighted')
        compute_per_class (bool): Whether to compute per-class metrics
        collect_device (str): Device to collect results
    """
    
    def __init__(self,
                 num_classes: int = 3,
                 mode: str = 'macro',
                 compute_per_class: bool = True,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.num_classes = num_classes
        self.mode = mode
        self.compute_per_class = compute_per_class
        self.class_names = ['Class I', 'Class II', 'Class III']
    
    def process(self, data_batch: Sequence[dict], data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.
        
        Args:
            data_batch: A batch of data from dataloader
            data_samples: A batch of outputs from model
        """
        for data_sample in data_samples:
            # Extract ground truth class
            gt_class = None
            if hasattr(data_sample, 'gt_instances') and hasattr(data_sample.gt_instances, 'labels'):
                gt_class = data_sample.gt_instances.labels.cpu().numpy()
                if len(gt_class) > 0:
                    gt_class = int(gt_class[0])
                else:
                    continue
            else:
                # Skip if no ground truth class
                continue
            
            # Extract predicted class
            pred_class = None
            if hasattr(data_sample, 'pred_instances') and hasattr(data_sample.pred_instances, 'class_label'):
                pred_class = data_sample.pred_instances.class_label
                if isinstance(pred_class, np.ndarray):
                    pred_class = int(pred_class[0]) if len(pred_class) > 0 else pred_class
                else:
                    pred_class = int(pred_class)
            else:
                # Skip if no predicted class
                continue
            
            # Store results
            result = {
                'gt_class': gt_class,
                'pred_class': pred_class,
            }
            
            # Also store class probabilities if available
            if hasattr(data_sample, 'pred_instances') and hasattr(data_sample.pred_instances, 'class_probs'):
                result['class_probs'] = data_sample.pred_instances.class_probs
            
            self.results.append(result)
    
    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute classification metrics.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Dict[str, float]: Computed metrics
        """
        if not results:
            warnings.warn('No valid classification results found')
            return {}
        
        # Extract predictions and ground truth
        gt_classes = np.array([r['gt_class'] for r in results])
        pred_classes = np.array([r['pred_class'] for r in results])
        
        # Basic metrics
        metrics = {}
        
        # Accuracy
        accuracy = accuracy_score(gt_classes, pred_classes)
        metrics['accuracy'] = float(accuracy)
        
        # Precision, Recall, F1-score
        precision, recall, f1, support = precision_recall_fscore_support(
            gt_classes, pred_classes, 
            average=self.mode,
            zero_division=0
        )
        
        metrics['precision'] = float(precision)
        metrics['recall'] = float(recall)
        metrics['f1_score'] = float(f1)
        
        # Per-class metrics
        if self.compute_per_class:
            precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
                gt_classes, pred_classes,
                average=None,
                zero_division=0
            )
            
            for i in range(self.num_classes):
                class_name = self.class_names[i] if i < len(self.class_names) else f'Class_{i}'
                metrics[f'{class_name}_precision'] = float(precision_per_class[i])
                metrics[f'{class_name}_recall'] = float(recall_per_class[i])
                metrics[f'{class_name}_f1'] = float(f1_per_class[i])
                metrics[f'{class_name}_support'] = int(support_per_class[i])
        
        # Confusion matrix
        conf_matrix = confusion_matrix(gt_classes, pred_classes, labels=list(range(self.num_classes)))
        
        # Add confusion matrix as flattened metrics for logging
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                true_class = self.class_names[i] if i < len(self.class_names) else f'Class_{i}'
                pred_class = self.class_names[j] if j < len(self.class_names) else f'Class_{j}'
                metrics[f'conf_matrix_{true_class}_as_{pred_class}'] = int(conf_matrix[i, j])
        
        # Print confusion matrix
        print("\nClassification Confusion Matrix:")
        print(f"{'True/Pred':>12}", end='')
        for class_name in self.class_names[:self.num_classes]:
            print(f"{class_name:>12}", end='')
        print()
        
        for i in range(self.num_classes):
            true_class = self.class_names[i] if i < len(self.class_names) else f'Class_{i}'
            print(f"{true_class:>12}", end='')
            for j in range(self.num_classes):
                print(f"{conf_matrix[i, j]:>12}", end='')
            print()
        
        # Summary statistics
        print(f"\nClassification Metrics Summary:")
        print(f"  Overall Accuracy: {accuracy:.3f}")
        print(f"  Macro Precision: {precision:.3f}")
        print(f"  Macro Recall: {recall:.3f}")
        print(f"  Macro F1-Score: {f1:.3f}")
        
        if self.compute_per_class:
            print("\nPer-Class Metrics:")
            for i in range(self.num_classes):
                class_name = self.class_names[i] if i < len(self.class_names) else f'Class_{i}'
                print(f"  {class_name}:")
                print(f"    Precision: {precision_per_class[i]:.3f}")
                print(f"    Recall: {recall_per_class[i]:.3f}")
                print(f"    F1-Score: {f1_per_class[i]:.3f}")
                print(f"    Support: {support_per_class[i]}")
        
        return metrics 