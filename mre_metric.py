import numpy as np
from typing import Any, List, Dict, Sequence, Union

from mmengine.evaluator import BaseMetric
from mmpose.registry import METRICS
from mmpose.evaluation.functional import keypoint_pck_accuracy # Can be adapted for distance

@METRICS.register_module()
class MeanRadialError(BaseMetric):
    """Mean Radial Error (MRE) metric for keypoint evaluation.

    Calculates the average Euclidean distance between predicted and ground truth keypoints.

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used. Defaults to None.
    """
    default_prefix: str = 'mre' # Metric prefix in the results dictionary

    def __init__(self, 
                 collect_device: str = 'cpu', 
                 prefix: Union[str, None] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        # We don't necessarily need a num_keypoints here if we infer from data
        # or if we calculate per keypoint and then average.

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
                Each item is a dictionary that should contain `pred_instances`
                and `gt_instances` keys.
        """
        results = []
        for data_sample in data_samples:
            pred_instances = data_sample.get('pred_instances', None)
            gt_instances = data_sample.get('gt_instances', None)

            if pred_instances is None or gt_instances is None:
                # Skip if no prediction or ground truth
                # Or handle as an error/warning
                print("Warning: Missing pred_instances or gt_instances in a data sample.")
                continue

            pred_keypoints = pred_instances.get('keypoints', None)
            gt_keypoints = gt_instances.get('keypoints', None)
            
            # keypoints_visible might be useful to only calculate error for visible keypoints
            # gt_keypoints_visible = gt_instances.get('keypoints_visible', None)
            # For MRE, typically all provided GT keypoints are used unless specified.

            if pred_keypoints is None or gt_keypoints is None:
                print("Warning: Missing keypoints in pred_instances or gt_instances.")
                continue

            # Ensure keypoints are numpy arrays and have compatible shapes
            # pred_keypoints shape: (num_instances, num_keypoints, 2)
            # gt_keypoints shape: (num_instances, num_keypoints, 2)
            # We expect num_instances = 1 for top-down
            if not isinstance(pred_keypoints, np.ndarray):
                pred_keypoints = np.array(pred_keypoints)
            if not isinstance(gt_keypoints, np.ndarray):
                gt_keypoints = np.array(gt_keypoints)

            if pred_keypoints.shape[0] != 1 or gt_keypoints.shape[0] != 1:
                print(f"Warning: Expected single instance, but got pred: {pred_keypoints.shape[0]} and gt: {gt_keypoints.shape[0]}. Skipping sample.")
                continue
                
            if pred_keypoints.shape != gt_keypoints.shape:
                print(f"Warning: Shape mismatch between pred ({pred_keypoints.shape}) and gt ({gt_keypoints.shape}) keypoints. Skipping sample.")
                continue

            # Calculate Euclidean distance for each keypoint for this sample
            # distances shape: (num_keypoints,)
            distances = np.linalg.norm(pred_keypoints[0] - gt_keypoints[0], axis=1)
            
            # Store individual keypoint distances for this sample
            results.append({
                'distances': distances, 
                'num_keypoints': pred_keypoints.shape[1]
            })

        self.results.extend(results)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results from `process` method.
                Each item is a dictionary containing 'distances' and 'num_keypoints'.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of the metric,
            and the values are corresponding results.
        """
        if not results:
            return {'MRE': 0.0, 'MRE_per_keypoint_avg': 0.0}

        all_distances = [] # Stores distances for ALL keypoints across ALL samples
        total_keypoints_evaluated = 0
        sum_of_mre_per_sample = 0.0
        num_samples = len(results)
        
        num_keypoints_per_sample = results[0]['num_keypoints'] if num_samples > 0 else 0
        per_keypoint_sum_distances = np.zeros(num_keypoints_per_sample)
        per_keypoint_counts = np.zeros(num_keypoints_per_sample) # if considering visibility in future

        for res in results:
            sample_distances = res['distances'] # (num_keypoints,)
            all_distances.extend(sample_distances)
            sum_of_mre_per_sample += np.mean(sample_distances) # MRE for this sample
            total_keypoints_evaluated += res['num_keypoints']
            
            if res['num_keypoints'] == num_keypoints_per_sample:
                per_keypoint_sum_distances += sample_distances
                per_keypoint_counts += 1 # Assuming all keypoints processed for each sample
        
        # Overall MRE: average of all keypoint distances across all samples
        overall_mre = np.mean(all_distances) if all_distances else 0.0

        # MRE per keypoint (averaged across samples)
        mre_per_keypoint = per_keypoint_sum_distances / num_samples if num_samples > 0 else np.zeros(num_keypoints_per_sample)
        
        metrics = {f'MRE': overall_mre}
        for i in range(num_keypoints_per_sample):
            # You can get keypoint names from metainfo if needed for logging
            # For now, just use index
            metrics[f'MRE_kp{i:02d}'] = mre_per_keypoint[i]

        # Average of per-sample MREs (can be slightly different from overall_mre)
        # metrics['MRE_avg_per_sample'] = sum_of_mre_per_sample / num_samples if num_samples > 0 else 0.0
        
        return metrics 