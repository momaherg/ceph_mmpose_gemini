import copy
import os
import json
import numpy as np
from typing import Any, List, Optional, Union, Dict

from mmpose.registry import DATASETS
from mmpose.datasets.datasets.base import BaseTopdownDataset
from mmpose.datasets.datasets.utils import parse_pose_metainfo


@DATASETS.register_module()
class CephalometricDataset(BaseTopdownDataset):
    """Cephalometric Landmark Dataset.

    Args:
        ann_file (str): Annotation file path. Default: ''.
        bbox_file (str, optional): Detection result file path. If
            ``bbox_file`` is set, keypoints will be derived from
            ``bbox_file``. Default: None.
        data_mode (str): Specifies the mode of data samples: 'topdown' or
            'bottomup'. Default: 'topdown'.
        metainfo_file (str, optional): Path to the dataset meta information file.
            This file should define `dataset_info` dictionary. Default: None.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Default: None.
        data_prefix (dict, optional): Path prefix for image files. Default:
            ``dict(img_path='')``.
        filter_cfg (dict, optional): Config for filtering samples. Default: None.
        indices (int or Sequence[int], optional): Support loading a subset of
            the dataset. Default: None.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loading is faster.
            Default: True.
        pipeline (list, optional): Processing pipeline. Default: [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Default: False.
        lazy_init (bool, optional): Whether to load annotation during init.
            Default: False.
        max_refetch (int, optional): Times to refetch data. Default: 1000.
    """

    METAINFO: dict = dict() # Will be populated from metainfo_file

    def __init__(self,
                 ann_file: str = '',
                 bbox_file: Optional[str] = None,
                 data_mode: str = 'topdown',
                 metainfo_file: Optional[str] = None, # Added this
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, List[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[dict] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):

        if metainfo_file:
            import importlib.util
            spec = importlib.util.spec_from_file_location("dataset_info_module", metainfo_file)
            metainfo_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(metainfo_module)
            dataset_metainfo = metainfo_module.dataset_info
            
            parsed_metainfo = parse_pose_metainfo(dataset_metainfo)
            # Update instance metainfo. Class METAINFO can also be updated if this is the first instance.
            self.metainfo = copy.deepcopy(self.METAINFO) # Start with class METAINFO (if any base definitions)
            self.metainfo.update(parsed_metainfo) # Then update with specifics from file

        super().__init__(
            ann_file=ann_file,
            bbox_file=bbox_file,
            data_mode=data_mode,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)

    def _load_data_list(self) -> List[Dict]:
        """Load annotations from the JSON file specified in ``self.ann_file``."""
        if not self.ann_file:
            raise ValueError(
                f'{self.__class__.__name__}: \'ann_file\' cannot be empty when \'ann_data\' is not provided.')

        with open(self.ann_file, 'r') as f:
            raw_anns = json.load(f)

        data_list = []
        for ann_entry in raw_anns:
            full_img_path = os.path.join(self.data_root, ann_entry['img_path'])

            keypoints_xy = np.array(ann_entry['keypoints'], dtype=np.float32)
            visibility = np.array(ann_entry['keypoints_visible'], dtype=np.float32).reshape(-1, 1)
            joints_3d = np.concatenate([keypoints_xy, visibility], axis=1)
            joints_3d_visible = joints_3d.copy()

            bbox_xywh = ann_entry['bbox']

            data_info = {
                'img_id': ann_entry['img_id'],
                'img_path': full_img_path,
                'bbox': bbox_xywh,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'iscrowd': 0,
                'id': ann_entry['id'],
            }
            data_list.append(data_info)

        return data_list

    def _is_valid_instance(self, instance: dict) -> bool:
        """Check if the instance is valid.
        Override this method to filter invalid instances.
        The instance is a dictionary item from the list returned by ``_load_data_list()``.
        """
        if not instance.get('bbox') or \
           instance['bbox'][2] <= 0 or \
           instance['bbox'][3] <= 0:
            return False

        if np.sum(instance['joints_3d'][:, 2] > 0) == 0:
            return False
            
        return True

    # OPTIONAL: If your bounding boxes are very precise and you don't want
    # the TopDownGetBboxCenterScale to modify them too much, you could pre-calculate
    # 'center' and 'scale' here and ensure they are passed.
    # However, it's usually simpler to let the pipeline derive them.

    # OPTIONAL: If you have specific evaluation metrics not covered by standard ones,
    # you might need to override `evaluate` method. For PCK, NME, AUC, EPE,
    # standard evaluators should work if `dataset_info` (sigmas, etc.) is correct. 