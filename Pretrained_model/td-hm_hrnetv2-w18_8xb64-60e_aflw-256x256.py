auto_scale_lr = dict(base_batch_size=512)
backend_args = dict(backend='local')
codec = dict(
    heatmap_size=(
        64,
        64,
    ),
    input_size=(
        256,
        256,
    ),
    sigma=2,
    type='MSRAHeatmap')
custom_hooks = [
    dict(type='SyncBuffersHook'),
]
data_mode = 'topdown'
data_root = 'data/aflw/'
dataset_type = 'AFLWDataset'
default_hooks = dict(
    badcase=dict(
        badcase_thr=5,
        enable=False,
        metric_type='loss',
        out_dir='badcase',
        type='BadCaseAnalysisHook'),
    checkpoint=dict(
        interval=1, rule='less', save_best='NME', type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='PoseVisualizationHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        extra=dict(
            stage1=dict(
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_branches=1,
                num_channels=(64, ),
                num_modules=1),
            stage2=dict(
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                ),
                num_branches=2,
                num_channels=(
                    18,
                    36,
                ),
                num_modules=1),
            stage3=dict(
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                    4,
                ),
                num_branches=3,
                num_channels=(
                    18,
                    36,
                    72,
                ),
                num_modules=4),
            stage4=dict(
                block='BASIC',
                multiscale_output=True,
                num_blocks=(
                    4,
                    4,
                    4,
                    4,
                ),
                num_branches=4,
                num_channels=(
                    18,
                    36,
                    72,
                    144,
                ),
                num_modules=3),
            upsample=dict(align_corners=False, mode='bilinear')),
        in_channels=3,
        init_cfg=dict(
            checkpoint='open-mmlab://msra/hrnetv2_w18', type='Pretrained'),
        type='HRNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        conv_kernel_sizes=(1, ),
        conv_out_channels=(270, ),
        decoder=dict(
            heatmap_size=(
                64,
                64,
            ),
            input_size=(
                256,
                256,
            ),
            sigma=2,
            type='MSRAHeatmap'),
        deconv_out_channels=None,
        in_channels=270,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        out_channels=19,
        type='HeatmapHead'),
    neck=dict(concat=True, type='FeatureMapProcessor'),
    test_cfg=dict(flip_mode='heatmap', flip_test=True, shift_heatmap=True),
    type='TopdownPoseEstimator')
optim_wrapper = dict(optimizer=dict(lr=0.002, type='Adam'))
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=60,
        gamma=0.1,
        milestones=[
            40,
            55,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='annotations/face_landmarks_aflw_test.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root='data/aflw/',
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                256,
                256,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='AFLWDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    norm_item='bbox_size', norm_mode='use_norm_item', type='NME')
train_cfg = dict(by_epoch=True, max_epochs=60, val_interval=1)
train_dataloader = dict(
    batch_size=64,
    dataset=dict(
        ann_file='annotations/face_landmarks_aflw_train.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root='data/aflw/',
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(direction='horizontal', type='RandomFlip'),
            dict(
                rotate_factor=60,
                scale_factor=(
                    0.75,
                    1.25,
                ),
                shift_prob=0,
                type='RandomBBoxTransform'),
            dict(input_size=(
                256,
                256,
            ), type='TopdownAffine'),
            dict(
                encoder=dict(
                    heatmap_size=(
                        64,
                        64,
                    ),
                    input_size=(
                        256,
                        256,
                    ),
                    sigma=2,
                    type='MSRAHeatmap'),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='AFLWDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(direction='horizontal', type='RandomFlip'),
    dict(
        rotate_factor=60,
        scale_factor=(
            0.75,
            1.25,
        ),
        shift_prob=0,
        type='RandomBBoxTransform'),
    dict(input_size=(
        256,
        256,
    ), type='TopdownAffine'),
    dict(
        encoder=dict(
            heatmap_size=(
                64,
                64,
            ),
            input_size=(
                256,
                256,
            ),
            sigma=2,
            type='MSRAHeatmap'),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='annotations/face_landmarks_aflw_test.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root='data/aflw/',
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                256,
                256,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='AFLWDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    norm_item='bbox_size', norm_mode='use_norm_item', type='NME')
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(input_size=(
        256,
        256,
    ), type='TopdownAffine'),
    dict(type='PackPoseInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
