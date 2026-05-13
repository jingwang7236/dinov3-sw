_base_ = [
    '_base_/default_runtime.py',
]

experiment_name = '008_dinov3_vit-l_FIX_CONFIG_load_from_chinasiwei_150w_lr1e-4_pretrain'
work_dir = f'./work_dirs/dinov3/multiclass/{experiment_name}'


crop_size = (512, 512)
checkpoint = None  # noqa


# model settings
backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

num_classes = 10
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=checkpoint,
    backbone=dict(
        type='mmpretrain.DinoV3Backbone',
        # img_size=(512, 512),
        # checkpoint_path='/mnt/mty/open_source_mm/chinasiwei_fm/DinoV3LightningTraining/output/001_dinov3lightning_vit-l_chinasiweidataset_load_from_sat493m_pretrain/final_ssl_model_only_teacher-8aa4cbdd.pth',
        # checkpoint_path='/mnt/mty/open_source_mm/chinasiwei_fm/DinoV3LightningTraining/output/002_dinov3lightning_vit-l_chinasiweidataset_all_bands_20251023_load_from_sat493m_pretrain/002_final_ssl_model_66w_only_teacher-eadcf0ff.pth',
        # checkpoint_path='/mnt/mty/open_source_mm/chinasiwei_fm/dinov3-main/work_dirs/003_dinov3_vit-l_20251117_all_bands_datset_with_downstream_task_data_150w/eval/training_124999/003_150w_teacher_checkpoint-eadcf0ff.pth',
        # checkpoint_path='/mnt/mty/open_source_mm/chinasiwei_fm/DinoV3LightningTraining/output/005_dinov3lightning_vit-l_chinasiweidataset_all_bands_20251117_add_downstream_dataset_load_from_sat493m_pretrain_lr_1e-4/005_final_ssl_model-eadcf0ff.pth',
        checkpoint_path='/mnt/mty/open_source_mm/chinasiwei_fm/DinoV3LightningTraining/output/006_dinov3lightning_vit-l_FIX_CONFIG_chinasiwei_20251117_add_downstream_dataset_load_from_sat493m_pretrain_lr_1e-4/006_final_ssl_model-eadcf0ff.pth',

        # checkpoint_path='/mnt/mty/open_source_mm/chinasiwei_fm/dinov3-main/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
        freeze_backbone=False,
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_global_and_local_cls_norm=True,
        # n_storage_tokens=0,
        # mask_k_bias=False,
        # untie_global_and_local_cls_norm=False,
        # freeze_backbone=True
    ),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=1024,
        channels=1024,
        num_classes=num_classes,
        num_layers=2,
        num_heads=16,
        embed_dims=1024,
        dropout_ratio=0.0,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(480, 480)),
)



# 全要素地物分类
# dataset settings
dataset_type = 'Muti_Class_Dataset'
data_root = '/mnt/myf/dataset/multi-class51520/'
image_scale = (512, 512)
# crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=image_scale,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=image_scale, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile'),
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
    batch_size=6,
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



# optimizer = dict(lr=0.001, weight_decay=0.0)
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
# train_dataloader = dict(
#     # num_gpus: 8 -> batch_size: 8
#     batch_size=1)
# val_dataloader = dict(batch_size=1)

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
