# DFC2025 BRIGHT 前光后 SAR 变化检测 — 4 分类 + 多 trick (双 backbone 全冻结, 显存友好)
# ============================================================================
# 实验目的: 在 ctxfused 基线之上进一步缩小与 SOTA 的 mIoU 差距 (4 个类别普遍
#   偏低 => 倾向"欠拟合/域适配不足"而非类别不平衡, 故重点提升泛化能力)。
#   因显存不足, 两个 backbone (DINOv3 光学 + OlmoEarth SAR) 全部冻结, 仅训练
#   decoder/adapter/投影层, 显著降低显存占用; 同时保留其余有效 trick:
#     ① EMA 权重平滑 (均匀提升各类 IoU, +0.3~1.0 mIoU);
#     ③ 多尺度推理融合 (p2/p3/p4/p5 logits 加权, 改善边界与碎小目标);
#     ④ 训练 80k + 更长 warmup / 更低 eta_min 的余弦调度;
#     ⑤ Lovász + 类别加权 Focal (直接优化 mIoU 与稀有类)。
#   (② 部分解冻因显存不足而关闭。)
# 兼容性: 仅新增配置与新增类 (EMAHook / ms_inference 选项默认关闭), 完全向后兼容。
# ============================================================================

_base_ = ['../_base_/default_runtime.py']
olmoearth_model_dir = "/mnt/qh2-nas3/00-model/00-wrs/zhejiang_earth_results/olmoearth10m_base/step23400"
OlmoEarth_config_path = "{}/config.json".format(olmoearth_model_dir)
OlmoEarth_weights_path = "{}/weights.pth".format(olmoearth_model_dir)
Dino_weights_path = "/mnt/qh2-nas3/00-model/00-limx/Dinov3/ckpt/stage2+stage3-zhejiang/23999.pth"
Dino_weights_type = "self_trained"

# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
dataset_type = 'DFC2025BRIGHTDataset'
data_root = '/mnt/ht2-nas2/EO_test/dataset/ChangeDetection/BRIGHT'
crop_size = (512, 512)

data_prefix = dict(
    img_path_from='pre-event',
    img_path_to='post-event',
    seg_map_path='target')

# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=[73.17, 82.67, 86.77, 56.38, 56.38, 56.38],
    std=[53.27, 58.12, 67.28, 44.49, 44.49, 44.49],
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32))

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
        patch_size=4,
        image_size=512,
        in_channels=3,
        out_channels=128,
        freeze_backbone=True,           # 冻结 SAR backbone
        adaptive_pool=False,
        native_inference=True,
        native_size=256,
    ),
    decode_head=dict(
        type='ChangeDinoHybridCrossAttnDecoder',
        fpn_channels=128,
        n_layers=[1, 1, 1, 1],
        num_classes=4,
        align_corners=False,
        ignore_index=255,
        cross_num_heads=4,
        window_size=8,
        use_context=True,
        context_weight=1.0,
        gate_init=-1.0,
        # 类别顺序: [0=background, 1=intact, 2=damaged, 3=destroyed]
        focal_alpha=[0.5, 1.0, 0.75, 0.75],
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
        crop_size=(512, 512),
        stride=(341, 341)),
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
        data_prefix=data_prefix,
        binarize_label=False,
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
        data_prefix=data_prefix,
        binarize_label=False,
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
        data_prefix=data_prefix,
        binarize_label=False,
        pipeline=test_pipeline))

val_evaluator = dict(type='mmseg.IoUMetric', iou_metrics=['mFscore', 'mIoU'])
test_evaluator = dict(type='mmseg.IoUMetric', iou_metrics=['mFscore', 'mIoU'])

# ---------------------------------------------------------------------------
# 优化器与训练策略 (双 backbone 全冻结, 无 paramwise 分组)
# ---------------------------------------------------------------------------
optimizer = dict(
    type='AdamW', lr=0.0005, betas=(0.9, 0.999), weight_decay=0.0005)
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
                       img_shape=(512, 512, 3)))

# ① EMA: 权重平滑, val/test 自动换入 EMA 权重评估
custom_hooks = [
    dict(type='EMAHook', momentum=2e-4, update_buffers=True, priority='LOWEST'),
]
