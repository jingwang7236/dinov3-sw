max_inchannels = 3

experiment_name = '002_channel_vit_dino_wo_hcs_lr_2e-6_debug'
work_dir = f'./work_dirs/{experiment_name}'


model = dict(
    type='DINOHCS',
    enable_sample=False,
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='mmpretrain.HCSChannelViT', 
        enable_sample=False,
        img_size=(224, 224),
        patch_size=16,
        in_channels=max_inchannels,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=(2, 5, 8, 11),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cls_token=True,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        interpolate_mode='bicubic',
        out_type='cls_token'),
    neck=dict(
        type='DINONeck',
        in_channels=768,
        out_channels=65536,
        hidden_channels=2048,
        bottleneck_channels=256),
    head=dict(
        type='DINOHead',
        out_channels=65536,
        num_crops=10,
        student_temp=0.1,
        center_momentum=0.9))

        
# train_pipeline = [
#     # dict(type='LoadRSImageFromFileWithChannelInfo'),
#     dict(
#         type='LoadImageFromFileWithChannelInfo',
#         imdecode_backend='tifffile',
#         color_type='unchanged',
#         to_float32=True,
#     ),
#     dict(
#         type='DINOMultiCrop',
#         global_crops_scale=(0.4, 1.0),
#         local_crops_scale=(0.05, 0.4),
#         local_crops_number=8),
#     dict(type='PackInputs', meta_keys=('sample_idx', 'img_path', 'ori_shape', 'img_shape',
#                          'scale_factor', 'flip', 'flip_direction', 'channels'))
# ]

 
train_pipeline = [
    dict(
        type='LoadImageFromFileWithChannelInfo',
        imdecode_backend='tifffile',
        color_type='unchanged',
        to_float32=True,
    ),
    dict(
        type='DINOMultiCropCV2',
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=8,
        global_size=224,
        local_size=96,
        use_solarize=False),
    dict(type='PackInputs', meta_keys=('sample_idx', 'img_path', 'ori_shape', 'img_shape',
                         'scale_factor', 'flip', 'flip_direction', 'channels'))
]



data_root = '/mnt/mty/FM_DATA/'
one_bands_dataset = dict(
    type='ChinaSiweiFmDataset',
    with_label=False,
    data_root=data_root,
    ann_file=data_root + '1bands_images.txt',
    # ann_file=data_root + '1bands_images_tiny_test.txt',
    data_prefix=dict(img_path=''),
    pipeline=train_pipeline,
)

three_bands_dataset = dict(
    type='ChinaSiweiFmDataset',
    with_label=False,
    data_root=data_root,
    ann_file=data_root + '3bands_images.txt',
    # ann_file=data_root + '3bands_images_tiny_test.txt',
    data_prefix=dict(img_path=''),
    pipeline=train_pipeline,
)

eight_bands_dataset = dict(
    type='ChinaSiweiFmDataset',
    with_label=False,
    data_root=data_root,
    ann_file=data_root + '8bands_images.txt',
    data_prefix=dict(img_path=''),
    pipeline=train_pipeline,
)



train_dataloader = dict(
    # batch_size=8,
    batch_size=48,
    # batch_size=32,
    num_workers=16,
    persistent_workers=True,
    # num_workers=0,
    # persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='ConcatDataset', 
        # datasets=[eight_bands_dataset]
        datasets=[three_bands_dataset]
    ))

optimizer = dict(type='AdamW', lr=2e-6, betas=(0.9, 0.95), weight_decay=0.05)
# optimizer = dict(type='AdamW', lr=0.0024, betas=(0.9, 0.95), weight_decay=0.05)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys=dict(
            ln=dict(decay_mult=0.0),
            bias=dict(decay_mult=0.0),
            pos_embed=dict(decay_mult=0.0),
            mask_token=dict(decay_mult=0.0),
            cls_token=dict(decay_mult=0.0))),
    loss_scale='dynamic',
    clip_grad=dict(max_norm=1.0, norm_type=2))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-09,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=90,
        by_epoch=True,
        begin=10,
        end=100,
        convert_to_iter_based=True)
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100)
default_scope = 'mmpretrain'
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(
    window_size=10,
    custom_cfg=[dict(data_src='', method='mean', window_size='global')])
vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_level = 'INFO'
load_from = None
resume = True
randomness = dict(seed=2, diff_rank_seed=True)
custom_hooks = [
    dict(
        type='DINOTeacherTempWarmupHook',
        warmup_teacher_temp=0.04,
        teacher_temp=0.04,
        teacher_temp_warmup_epochs=0,
        max_epochs=100),
]
