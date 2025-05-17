# yapf:disable
log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10) # Save checkpoint every 10 epochs
evaluation = dict(interval=1, metric='NME', save_best='NME') # Evaluate every epoch

# yapf:enable

# runtime settings
total_epochs = 210 # default total epochs

# optimizer
optimizer = dict(type='Adam', lr=5e-4)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200]) # Default learning rate steps

# log config
log_config = dict(
    interval=50, # Log every 50 iterations
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook') # Uncomment to use TensorBoard
    ]) 