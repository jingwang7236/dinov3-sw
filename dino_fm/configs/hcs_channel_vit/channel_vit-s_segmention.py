
_base_ = [
    # '../_base_/models/upernet_vit-b16_ln_mln.py',
    # '../_base_/datasets/multi_class.py', 
    '../_base_/default_runtime.py',
    # '../_base_/schedules/schedule_80k.py'
]

experiment_name = '003_channel_vit-s__load_from_dino_pretrain'
work_dir = f'./work_dirs/{experiment_name}'
crop_size = (512, 512)


# 全要素地物分类
# dataset settings
dataset_type = 'Muti_Class_Dataset'
data_root = '/mnt/myf/dataset/multi-class51520/'
image_scale = (512, 512)
# crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFileWithChannelInfo'),
    # dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=image_scale,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label', 'channels'))
]
test_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(type='LoadImageFromFileWithChannelInfo'),
    dict(type='Resize', scale=image_scale, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label', 'channels'))
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(type='LoadImageFromFileWithChannelInfo'),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    # num_workers=0,
    # persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/images', seg_map_path='train/label'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=12,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val/images',
            seg_map_path='val/label'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size
)

num_classes = 10
max_inchannels = 3

model = dict(
    type='ChannelEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='/mnt/mty/open_source_mm/chinasiwei_fm/ChannelViT-main/pretrain/imagenet_channelvit_small_p16_DINO.pth',
    # pretrained='/mnt/mty/open_source_mm/chinasiwei_fm/ChannelViT-main/pretrain/imagenet_channelvit_small_p16_with_hcs_supervised.pth',
    # pretrained=None,
    backbone=dict(
        type='mmpretrain.HCSChannelViT',
        # type='VisionTransformer',
        enable_sample=False,
        img_size=(512, 512),
        patch_size=16,
        in_channels=max_inchannels,
        embed_dims=384,
        num_layers=6,
        num_heads=6,
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
        interpolate_mode='bicubic'),
    decode_head=dict(
        type='HCSSegmenterMaskTransformerHead',
        in_channels=384,
        channels=384,
        num_classes=num_classes,
        num_layers=2,
        num_heads=6,
        embed_dims=384,
        dropout_ratio=0.0,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
    # neck=dict(
    #     type='MultiLevelNeck',
    #     in_channels=[768, 768],
    #     # in_channels=[768, 768, 768, 768],
    #     out_channels=768,
    #     scales=[4, 2, 1, 0.5]),
    # decode_head=dict(
    #     type='UPerHead',
    #     in_channels=[768, 768],
    #     # in_channels=[768, 768, 768, 768],
    #     in_index=[0, 1],
    #     # in_index=[0, 1, 2, 3],
    #     pool_scales=(1, 2, 3, 6),
    #     channels=512,
    #     dropout_ratio=0.1,
    #     num_classes=num_classes,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # auxiliary_head=dict(
    #     type='FCNHead',
    #     in_channels=768,
    #     in_index=3,
    #     channels=256,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=num_classes,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # yapf: disable


# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    # _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=1),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

# training schedule for 80k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=8000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
