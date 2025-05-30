dataset_info = dict(
    dataset_name='cephalometric',
    paper_info=dict(
        author='Lala & Mohamed Maher', # You can change this
        title='Cephalometric Landmark Prediction Dataset',
        container='Internal',
        year='2024', # Or the correct year
        homepage='', # Optional: link to dataset source/project page
    ),
    # landmark_cols from your description
    # Total 19 landmarks, each with x and y (38 columns)
    # sella, nasion, A point, B point, upper 1 tip, upper 1 apex, lower 1 tip, lower 1 apex,
    # ANS, PNS, Gonion, Menton, ST Nasion, Tip of the nose, Subnasal, Upper lip, Lower lip,
    # ST Pogonion, gnathion
    keypoint_info={
        0: dict(name='sella', id=0, color=[255, 128, 0], type='face', swap=''),
        1: dict(name='nasion', id=1, color=[255, 128, 0], type='face', swap=''),
        2: dict(name='A_point', id=2, color=[255, 128, 0], type='face', swap=''),
        3: dict(name='B_point', id=3, color=[255, 128, 0], type='face', swap=''),
        4: dict(name='upper_1_tip', id=4, color=[0, 255, 0], type='mouth', swap=''),
        5: dict(name='upper_1_apex', id=5, color=[0, 255, 0], type='mouth', swap=''),
        6: dict(name='lower_1_tip', id=6, color=[0, 255, 0], type='mouth', swap=''),
        7: dict(name='lower_1_apex', id=7, color=[0, 255, 0], type='mouth', swap=''),
        8: dict(name='ANS', id=8, color=[51, 153, 255], type='face', swap=''),
        9: dict(name='PNS', id=9, color=[51, 153, 255], type='face', swap=''),
        10: dict(name='Gonion', id=10, color=[102, 0, 204], type='face', swap=''),
        11: dict(name='Menton', id=11, color=[102, 0, 204], type='face', swap=''),
        12: dict(name='ST_Nasion', id=12, color=[255, 51, 51], type='face', swap=''),
        13: dict(name='Tip_of_the_nose', id=13, color=[255, 51, 51], type='face', swap=''),
        14: dict(name='Subnasal', id=14, color=[255, 51, 51], type='face', swap=''),
        15: dict(name='Upper_lip', id=15, color=[0, 102, 204], type='mouth', swap=''),
        16: dict(name='Lower_lip', id=16, color=[0, 102, 204], type='mouth', swap=''),
        17: dict(name='ST_Pogonion', id=17, color=[102, 0, 204], type='face', swap=''),
        18: dict(name='gnathion', id=18, color=[102, 0, 204], type='face', swap=''),
    },
    skeleton_info={}, # No skeleton defined for now, can be added later if needed for visualization
    # Updated joint weights based on MRE analysis - higher weights for difficult landmarks
    joint_weights=[
        2.0,  # 0: sella (high error ~8px, needs more weight)
        1.0,  # 1: nasion (good ~2.6px)
        1.0,  # 2: A_point (good ~2.8px)
        1.0,  # 3: B_point (moderate ~3.5px)
        1.0,  # 4: upper_1_tip (moderate ~3.7px)
        1.0,  # 5: upper_1_apex (moderate ~3.1px)
        1.0,  # 6: lower_1_tip (moderate ~3.4px)
        1.0,  # 7: lower_1_apex (moderate ~3.9px)
        1.0,  # 8: ANS (moderate ~3.2px)
        1.5,  # 9: PNS (high error ~4.6px, needs more weight)
        2.0,  # 10: Gonion (high error ~8.3px, needs more weight)
        1.0,  # 11: Menton (moderate ~3.7px)
        1.0,  # 12: ST_Nasion (good ~2.1px)
        1.0,  # 13: Tip_of_the_nose (excellent ~1.4px)
        1.0,  # 14: Subnasal (excellent ~1.8px)
        1.0,  # 15: Upper_lip (excellent ~2.2px)
        1.0,  # 16: Lower_lip (excellent ~2.1px)
        1.0,  # 17: ST_Pogonion (good ~2.9px)
        1.0   # 18: gnathion (moderate ~3.3px)
    ],
    # Sigmas are crucial for OKS calculation. These are initial guesses and might need tuning.
    # Smaller sigmas mean higher precision is penalized more.
    # Values are typically related to the scale/difficulty of localizing each point.
    # For 224x224 images, a common starting point (like COCO's person keypoints) is around 0.025 to 0.085
    # We can use a uniform value for now, or try to be slightly more specific if some points are harder.
    # Given all are facial landmarks, variation might not be extreme.
    sigmas=[0.035] * 19, # A slightly more generous sigma than pure COCO keypoints, adjust as needed.
    # Flip indices for horizontal flipping augmentation
    # Since these are cephalometric landmarks from lateral (side) view,
    # most landmarks don't have clear left/right pairs, so they map to themselves
    # You may need to adjust these based on your specific landmarks and how they should behave during flipping
    flip_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    # Add keypoint_id2name and keypoint_name2id for robustness
    keypoint_id2name = {
        0: 'sella', 1: 'nasion', 2: 'A_point', 3: 'B_point',
        4: 'upper_1_tip', 5: 'upper_1_apex', 6: 'lower_1_tip', 7: 'lower_1_apex',
        8: 'ANS', 9: 'PNS', 10: 'Gonion', 11: 'Menton',
        12: 'ST_Nasion', 13: 'Tip_of_the_nose', 14: 'Subnasal',
        15: 'Upper_lip', 16: 'Lower_lip', 17: 'ST_Pogonion', 18: 'gnathion'
    },
    keypoint_name2id = {
        'sella': 0, 'nasion': 1, 'A_point': 2, 'B_point': 3,
        'upper_1_tip': 4, 'upper_1_apex': 5, 'lower_1_tip': 6, 'lower_1_apex': 7,
        'ANS': 8, 'PNS': 9, 'Gonion': 10, 'Menton': 11,
        'ST_Nasion': 12, 'Tip_of_the_nose': 13, 'Subnasal': 14,
        'Upper_lip': 15, 'Lower_lip': 16, 'ST_Pogonion': 17, 'gnathion': 18
    }
)

landmark_names_in_order = [
    'sella', 'nasion', 'A_point', 'B_point', 'upper_1_tip', 'upper_1_apex',
    'lower_1_tip', 'lower_1_apex', 'ANS', 'PNS', 'Gonion', 'Menton',
    'ST_Nasion', 'Tip_of_the_nose', 'Subnasal', 'Upper_lip', 'Lower_lip',
    'ST_Pogonion', 'gnathion'
]

# Mapping from the landmark_cols in your description to the simplified names in keypoint_info
# This will be useful for the dataset class when parsing the JSON
landmark_column_map = {
    'sella_x': 'sella', 'sella_y': 'sella',
    'nasion_x': 'nasion', 'nasion_y': 'nasion',
    'A point_x': 'A_point', 'A point_y': 'A_point', # Note the space in 'A point'
    'B point_x': 'B_point', 'B point_y': 'B_point', # Note the space in 'B point'
    'upper 1 tip_x': 'upper_1_tip', 'upper 1 tip_y': 'upper_1_tip',
    'upper 1 apex_x': 'upper_1_apex', 'upper 1 apex_y': 'upper_1_apex',
    'lower 1 tip_x': 'lower_1_tip', 'lower 1 tip_y': 'lower_1_tip',
    'lower 1 apex_x': 'lower_1_apex', 'lower 1 apex_y': 'lower_1_apex',
    'ANS_x': 'ANS', 'ANS_y': 'ANS',
    'PNS_x': 'PNS', 'PNS_y': 'PNS',
    'Gonion _x': 'Gonion', 'Gonion _y': 'Gonion', # Note the space in 'Gonion x'
    'Menton_x': 'Menton', 'Menton_y': 'Menton',
    'ST Nasion_x': 'ST_Nasion', 'ST Nasion_y': 'ST_Nasion',
    'Tip of the nose_x': 'Tip_of_the_nose', 'Tip of the nose_y': 'Tip_of_the_nose',
    'Subnasal_x': 'Subnasal', 'Subnasal_y': 'Subnasal',
    'Upper lip_x': 'Upper_lip', 'Upper lip_y': 'Upper_lip',
    'Lower lip_x': 'Lower_lip', 'Lower lip_y': 'Lower_lip',
    'ST Pogonion_x': 'ST_Pogonion', 'ST Pogonion_y': 'ST_Pogonion',
    'gnathion_x': 'gnathion', 'gnathion_y': 'gnathion'
}

# Original landmark column names as per your description (for parsing)
# This order must correspond to how they appear if you were to iterate through them
# when constructing the keypoint array.
# For now, we will rely on the landmark_column_map and landmark_names_in_order.
original_landmark_cols = [
    'sella_x', 'sella_y', 'nasion_x', 'nasion_y', 'A point_x', 'A point_y',
    'B point_x', 'B point_y', 'upper 1 tip_x', 'upper 1 tip_y',
    'upper 1 apex_x', 'upper 1 apex_y', 'lower 1 tip_x', 'lower 1 tip_y',
    'lower 1 apex_x', 'lower 1 apex_y', 'ANS_x', 'ANS_y', 'PNS_x', 'PNS_y',
    'Gonion _x', 'Gonion _y', 'Menton_x', 'Menton_y', 'ST Nasion_x',
    'ST Nasion_y', 'Tip of the nose_x', 'Tip of the nose_y', 'Subnasal_x',
    'Subnasal_y', 'Upper lip_x', 'Upper lip_y', 'Lower lip_x',
    'Lower lip_y', 'ST Pogonion_x', 'ST Pogonion_y', 'gnathion_x',
    'gnathion_y'
] 