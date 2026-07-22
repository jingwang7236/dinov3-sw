# DFC2025 BRIGHT 前光后 SAR 变化检测 — 4 分类 + 512 滑窗 + TTA + Lovász
# ============================================================================
# 实验目的: 在 dualmode_dino_olmoearth_512x512_slide_40k_dfc2025bright.py 基础上
#   优化 mIoU, 重点提升 Intact(未损毁建筑) 类别 IoU。
#
# 根因分析: 原 ChangeDinoCrossAttnDecoder 仅使用交叉注意力的"差异"特征
#   (a1 - a2), 对未变化区域(Intact 建筑)信号极弱, 导致其 IoU(75.9) 远低于
#   直接拼接特征的 UNet / DeepLabV3+(~90)。
#
# 优化手段(两个 backbone 均保持冻结, 仅训练 decoder / 数据流):
#   ① 新解码器 ChangeDinoHybridCrossAttnDecoder: 在差异特征之上叠加
#      "上下文(绝对外观)"分支, 门控融合, 补足未变化类别信号;
#   ② focal_alpha 按类别加权, 适度上权 Intact(class=1);
#   ③ TTA 增加垂直翻转 (tta_flips=[3,2]);
#   ④ Lovász Loss 继续启用以直接优化 mIoU。
# 兼容性: 不改动旧配置与旧类, 完全向后兼容。
# ============================================================================

_base_ = ['../_base_/default_runtime.py']
olmoearth_model_dir = "/mnt/ht2-nas2/00-model/00-wj/Codes/dinov3-sw/open-cd/olmoearth_pretrain-main/OlmoEarth-v1-Base"
OlmoEarth_config_path = "{}/config.json".format(olmoearth_model_dir)
OlmoEarth_weights_path = "{}/weights.pth".format(olmoearth_model_dir)

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
        dino_weight='/mnt/ht2-nas2/00-model/00-wj/Codes/checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
        freeze_mode='frozen',          # backbone 冻结
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
        freeze_backbone=True,           # backbone 冻结
        adaptive_pool=False,
        native_inference=True,
        native_size=256,
    ),
    decode_head=dict(
        # 关键改动: 叠加上下文(绝对外观)分支, 补足未变化类别(Intact)信号
        type='ChangeDinoHybridCrossAttnDecoder',
        fpn_channels=128,
        n_layers=[1, 1, 1, 1],
        num_classes=4,
        align_corners=False,
        ignore_index=255,
        cross_num_heads=4,
        window_size=8,
        use_context=True,              # 启用上下文融合分支
        context_weight=1.0,            # 上下文分支固定缩放
        gate_init=-1.0,                # 门控初值(sigmoid≈0.27), 训练中自适应
        # 类别顺序: [0=background, 1=intact, 2=damaged, 3=destroyed]
        # 适度上权 Intact, 改善其 IoU
        focal_alpha=[0.5, 1.0, 0.75, 0.75],
        focal_gamma=4.0,
        lovasz_weight=1.0,             # Lovász Loss 直接优化 mIoU
    ),
    tta_flips=[3, 2],                  # TTA: 水平 + 垂直翻转
    train_cfg=dict(),
    # 滑窗推理: 原始图 1024×1024, crop=512, stride=341 → 3×3 重叠网格
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

# 测试流水线: 不做 resize, 保留原始 1024×1024 分辨率交给滑窗推理处理
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

# 滑窗推理需逐图处理, batch_size=1
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
# 优化器与训练策略
# ---------------------------------------------------------------------------
optimizer = dict(
    type='AdamW', lr=0.0005, betas=(0.9, 0.999), weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=3000),
    dict(type='CosineAnnealingLR', T_max=37000, eta_min=1e-6,
         by_epoch=False, begin=3000, end=40000),
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000,
                    save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='CDVisualizationHook', interval=1,
                       img_shape=(512, 512, 3)))
