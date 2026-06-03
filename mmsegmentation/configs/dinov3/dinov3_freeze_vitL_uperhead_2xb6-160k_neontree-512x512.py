_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/geo_bench_neontree.py'
]

experiment_name = 'dinov3_sat493m_vit-l_uperhead_neontree_downstream'
work_dir = f'./work_dirs/dinov3/geo_bench_neontree/{experiment_name}'

crop_size = (512, 512)
checkpoint = ''

# model settings
backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)  # multi gpus

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],  # ImageNet
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

num_classes = 2
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='DinoV3BackboneSimple',  # 或 'DinoV3Backbone'
        model_name='vit_large',  # 可选: vit_large, vit_giant, vit_7b
        checkpoint_path='/mnt/ht2-nas2/00-model/00-wj/Codes/checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
        freeze_backbone=True,  # True: 冻结, False: 微调
        use_grad_checkpoint=True,  # 开启梯度检查点，省 30% 显存
    ),
    neck=dict(
        type='MultiLevelNeck',
        in_channels=[1024, 1024, 1024, 1024],
        out_channels=512,
        scales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[512, 512, 512, 512],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 4),
        channels=256,
        dropout_ratio=0.2,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            # use_sigmoid=False → 只支持 num_classes ≥ 2
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=None,
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(480, 480)),
)

train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    drop_last=True,
)

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    # _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01),
    accumulative_counts=2,
    clip_grad=dict(max_norm=35, norm_type=1),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=1e-6, 
        by_epoch=False, 
        begin=0, 
        end=1500),
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
    checkpoint=dict(type='CheckpointHook', 
        by_epoch=False,
        interval=8000,
        save_best='mIoU',
        max_keep_ckpts=3,
        save_optimizer=False,  # 节省显存，不保存优化器状态
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# ============== 环境优化 ==============
env_cfg = dict(
    cudnn_benchmark=True,           # 自动选择最优卷积算法
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# ============== 混合精度训练 (FP16) - 关键优化 ==============
# 开启混合精度可减少约50%显存，batch_size可提升至12-16
fp16 = dict(
    loss_scale='dynamic',           # 动态损失缩放
    init_scale=512,                 # 初始缩放因子
)