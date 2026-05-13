_base_ = [
    '_base_/default_runtime.py',
    '_base_/datasets/potsdam.py'
]

# experiment_name = '009_dinov3_vit-l_potsdam_noBoundaryLabel_load_from_sat493m_pretrain'
experiment_name = '010_dinov3_vit-l_potsdam_noBoundaryLabel_load_from_chinasiweirgb29w_pretrain'
work_dir = f'./work_dirs/dinov3/{experiment_name}'


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

num_classes = 6
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=checkpoint,
    backbone=dict(
        type='mmpretrain.DinoV3Backbone',
        # img_size=(512, 512),
        checkpoint_path='/mnt/mty/open_source_mm/chinasiwei_fm/DinoV3LightningTraining/output/001_dinov3lightning_vit-l_chinasiweidataset_load_from_sat493m_pretrain/final_ssl_model_only_teacher-8aa4cbdd.pth',
        # checkpoint_path='/mnt/mty/open_source_mm/chinasiwei_fm/dinov3-main/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
        freeze_backbone=False,
        n_storage_token=False,
        mask_k_bias=False,
        untie_global_and_local_cls_norm=False,
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




# optimizer = dict(lr=0.001, weight_decay=0.0)
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
train_dataloader = dict(batch_size=6)
val_dataloader = dict(batch_size=1)

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
