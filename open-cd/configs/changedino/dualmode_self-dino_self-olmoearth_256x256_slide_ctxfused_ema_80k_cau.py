
_base_ = ['../_base_/default_runtime.py']
olmoearth_model_dir = "/mnt/qh2-nas3/00-model/00-wrs/zhejiang_earth_results/olmoearth10m_base/step23400"
OlmoEarth_config_path = "{}/config.json".format(olmoearth_model_dir)
OlmoEarth_weights_path = "{}/weights.pth".format(olmoearth_model_dir)
Dino_weights_path = "/mnt/qh2-nas3/00-model/00-limx/Dinov3/ckpt/stage2+stage3-zhejiang/23999.pth"
Dino_weights_type = "self_trained"

# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
dataset_type = 'CAUFloodDataset'
data_root = '/mnt/qh2-nas3/EO_test/datasets/CAU-Flood'
crop_size = (256, 256)

# ============ 归一化参数 ============
_OPT_MEAN = [54.730560606596306, 87.64151816023418, 75.88863827067563,]       # ← 光学 R/G/B 均值
_OPT_STD  = [45.622372646527445, 40.93144444024192, 41.774725322913284,]       # ← 光学 R/G/B 标准差
_SAR_MEAN = [158.02073291645573, 158.02073291645573, 158.02073291645573]       # ← SAR 三通道均值（单通道重复3次）
_SAR_STD  = [68.08039319156036, 68.08039319156036, 68.08039319156036]       # ← SAR 三通道标准差

data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=_OPT_MEAN + _SAR_MEAN,
    std=_OPT_STD + _SAR_STD,
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32))

# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
model = dict(
    type='DualModeBranchEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone_opt=dict(
        type='DINOv3AdapterBackbone',
        out_channels=128,
        extract_ids=[5, 11, 17, 23],
        dino_weight=Dino_weights_path,
        weights_type=Dino_weights_type,
        freeze_mode='frozen',          # 冻结 DINOv3 ViT
    ),
    backbone_sar = dict(
        type='OlmoEarthSAREncoder',
        model_dir=olmoearth_model_dir,
        config_path=OlmoEarth_config_path,
        weights_path=OlmoEarth_weights_path,
        model_variant='base',
        in_channels=3,
        out_channels=128,
        freeze_backbone=True,           # 冻结 SAR backbone
        adaptive_pool=False,
        native_inference=True,
        native_size=256,
        modality='sentinel1',
        input_res=10,
        default_month=0,
        load_projection=True,
        strict_load=True,
    ),
    decode_head=dict(
        type='ChangeDinoHybridCrossAttnDecoder',
        fpn_channels=128,
        n_layers=[1, 1, 1, 1],
        num_classes=2,
        align_corners=False,
        ignore_index=255,
        cross_num_heads=4,
        window_size=8,
        use_context=True,
        context_weight=1.0,
        gate_init=-1.0,
        focal_alpha=[0.5, 1.0],
        focal_gamma=4.0,
        lovasz_weight=1.0,
    ),
    # ③ 多尺度推理融合: p2/p3/p4/p5 logits 加权 (训练不受影响)
    ms_inference=True,
    ms_inference_weights=[0.5, 0.3, 0.1, 0.1],
    tta_flips=[3, 2],                  # TTA: 水平 + 垂直翻转
    train_cfg=dict(),
    test_cfg=dict(
        mode='slide',
        crop_size=crop_size,
        stride=(170, 170)),
)

# ---------------------------------------------------------------------------
# 数据流水线
# ---------------------------------------------------------------------------
train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotate', prob=0.5, degree=20),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='MultiImgPhotoMetricDistortion',
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10),
    dict(type='MultiImgPackSegInputs')
]

test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_set.txt',
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val_set.txt',
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test_set.txt',
        pipeline=test_pipeline))

val_evaluator = dict(type='mmseg.IoUMetric', iou_metrics=['mFscore', 'mIoU'])
test_evaluator = dict(type='mmseg.IoUMetric', iou_metrics=['mFscore', 'mIoU'])

# ---------------------------------------------------------------------------
# 优化器与训练策略 (双 backbone 全冻结, 无 paramwise 分组)
# ---------------------------------------------------------------------------
optimizer = dict(
    type='AdamW', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ④ 80k 余弦调度: 5k warmup -> cosine 到 1e-6
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=5000),
    dict(type='CosineAnnealingLR', T_max=75000, eta_min=1e-6,
         by_epoch=False, begin=5000, end=80000),
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000,
                    save_best='mIoU', max_keep_ckpts=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='CDVisualizationHook', interval=1,
                       img_shape=(256, 256, 3)))

# ① EMA: 权重平滑, val/test 自动换入 EMA 权重评估
custom_hooks = [
    dict(type='EMAHook', momentum=2e-4, update_buffers=True, priority='LOWEST'),
]
