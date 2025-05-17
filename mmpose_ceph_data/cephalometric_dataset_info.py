dataset_info = dict(
    dataset_name='cephalometric',
    paper_info=dict(
        author='Lala & MMPose User',
        title='Cephalometric Landmark Prediction Dataset',
        container='Internal Project',
        year='2024',
        homepage='None',
    ),
    keypoint_info={
        0: dict(name='sella', id=0, color=[255, 128, 0], type='upper', swap=''),
        1: dict(name='nasion', id=1, color=[255, 128, 0], type='upper', swap=''),
        2: dict(name='A point', id=2, color=[255, 128, 0], type='upper', swap=''),
        3: dict(name='B point', id=3, color=[255, 128, 0], type='lower', swap=''),
        4: dict(name='upper 1 tip', id=4, color=[0, 255, 0], type='upper', swap=''),
        5: dict(name='upper 1 apex', id=5, color=[0, 255, 0], type='upper', swap=''),
        6: dict(name='lower 1 tip', id=6, color=[0, 255, 0], type='lower', swap=''),
        7: dict(name='lower 1 apex', id=7, color=[0, 255, 0], type='lower', swap=''),
        8: dict(name='ANS', id=8, color=[51, 153, 255], type='upper', swap=''),
        9: dict(name='PNS', id=9, color=[51, 153, 255], type='upper', swap=''),
        10: dict(name='Gonion', id=10, color=[102, 0, 204], type='lower', swap=''),
        11: dict(name='Menton', id=11, color=[102, 0, 204], type='lower', swap=''),
        12: dict(name='ST Nasion', id=12, color=[255, 51, 51], type='facial', swap=''),
        13: dict(name='Tip of the nose', id=13, color=[255, 51, 51], type='facial', swap=''),
        14: dict(name='Subnasal', id=14, color=[255, 51, 51], type='facial', swap=''),
        15: dict(name='Upper lip', id=15, color=[0, 102, 204], type='facial', swap=''),
        16: dict(name='Lower lip', id=16, color=[0, 102, 204], type='facial', swap=''),
        17: dict(name='ST Pogonion', id=17, color=[102, 0, 204], type='facial', swap=''),
        18: dict(name='gnathion', id=18, color=[102, 0, 204], type='lower', swap=''),
    },
    skeleton_info={
        # Example: basic profile outline (can be expanded)
        0: dict(link=('sella', 'nasion'), id=0, color=[255,128,0]),
        1: dict(link=('nasion', 'ST Nasion'), id=1, color=[255,128,0]),
        2: dict(link=('ST Nasion', 'Tip of the nose'), id=2, color=[255,51,51]),
        3: dict(link=('Tip of the nose', 'Subnasal'), id=3, color=[255,51,51]),
        4: dict(link=('Subnasal', 'Upper lip'), id=4, color=[255,51,51]),
        5: dict(link=('Upper lip', 'Lower lip'), id=5, color=[0,102,204]),
        6: dict(link=('Lower lip', 'ST Pogonion'), id=6, color=[0,102,204]),
        7: dict(link=('ST Pogonion', 'Menton'), id=7, color=[102,0,204]),
        8: dict(link=('Menton', 'Gonion'), id=8, color=[102,0,204]), # Simplified
        9: dict(link=('nasion', 'ANS'), id=9, color=[51,153,255]),
        10: dict(link=('ANS', 'A point'), id=10, color=[51,153,255]),
        11: dict(link=('A point', 'upper 1 tip'), id=11, color=[0,255,0]),
        12: dict(link=('B point', 'lower 1 tip'), id=12, color=[0,255,0]),
        13: dict(link=('Menton', 'gnathion'), id=13, color=[102,0,204]),
    },
    joint_weights=[1.] * 19, # Start with equal weights
    # Sigmas are important for evaluation (OKS) and some heatmap generation methods.
    # These are initial guesses and might need tuning.
    # For 256x256 image, a sigma of 0.025 means roughly 6.4 pixels.
    # For 224x224 image, 0.025 * 224 = 5.6 pixels.
    # Let's use values related to a fraction of the image size.
    # Or a common default like those used in COCO (around 0.025 to 0.085 for different joints).
    # A common starting point for heatmap based methods is sigma=2 (pixels) for target generation,
    # which is different from OKS sigmas.
    # For OKS sigmas, a typical value for facial landmarks might be smaller than for body joints.
    sigmas = [0.025, 0.025, 0.025, 0.025, 0.025, # sella, nasion, A, B, upper 1 tip
              0.025, 0.025, 0.025, 0.025, 0.025, # upper 1 apex, lower 1 tip, lower 1 apex, ANS, PNS
              0.035, 0.035, 0.025, 0.025, 0.025, # Gonion, Menton, ST Nasion, Tip of nose, Subnasal
              0.025, 0.025, 0.035, 0.035]       # Upper lip, Lower lip, ST Pogonion, gnathion
              # Using slightly larger for Gonion, Menton, ST Pogonion, gnathion as they might have more variance
) 