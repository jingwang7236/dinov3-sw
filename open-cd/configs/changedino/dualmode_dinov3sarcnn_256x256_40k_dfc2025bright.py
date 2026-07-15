# DFC2025 BRIGHT 前光后 SAR 变化检测 (非对称双流: 光学 DINOv3 + SAR CNN)
# - 光学侧: DINOv3AdapterBackbone (冻结 DINOv3 ViT-L)
# - SAR  侧: SARCNNEncoder (可训练 CNN)
# - 融合 : ChangeDinoCrossAttnDecoder (双向交叉注意力)
_base_ = ['../_base_/default_runtime.py']

# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
dataset_type = 'DFC2025BRIGHTDataset'
data_root = '/mnt/ht2-nas2/EO_test/dataset/ChangeDetection/BRIGHT'
crop_size = (256, 256)

# 注意: ann_file 指向 split txt; data_prefix 的 from=光学(3ch), to=SAR(1ch)
data_prefix = dict(
    img_path_from='pre-event',
    img_path_to='post-event',
    seg_map_path='target')

# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
norm_cfg = dict(type='SyncBN', requires_grad=True)
# mean/std 前 3 为光学, 后 3 为 SAR(灰度复制为 3 通道, 用相同统计量)
# 已在 train_set 上抽样统计得到 (0-255):
#   光学 mean=[73.17, 82.67, 86.77] std=[53.27, 58.12, 67.28]
#   SAR  mean=56.38 std=44.49
# 标签 building_damage 为 4 值(0=背景, 1=未损毁intact, 2=损毁damaged,
# 3=完全损毁destroyed)。设 binarize_label=False 做 4 分类损毁分级,
# 配合 num_classes=4 以复现论文 Table 中的 mIoU/F1 指标。
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
        freeze_mode='frozen',
    ),
    backbone_sar=dict(
        type='SARCNNEncoder',
        in_channels=3,        # SAR 灰度已复制为 3 通道
        out_channels=128,     # 与光学侧 / decode_head 对齐
        base_channels=32,
        n_blocks=2,
    ),
    decode_head=dict(
        type='ChangeDinoCrossAttnDecoder',   # 双向交叉注意力, 适合异构模态
        fpn_channels=128,
        n_layers=[1, 1, 1, 1],
        num_classes=4,
        align_corners=False,
        ignore_index=255,
        cross_num_heads=4,
        window_size=8,
        focal_alpha=None,        # 多分类: 关闭二分类 alpha 权重, 依赖 gamma
    ),
    # refiner(LearnableSoftMorph) 仅支持二分类(断言 C==2), 4 分类下移除
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ---------------------------------------------------------------------------
# 数据流水线 (注意: 异构模态, 不使用 MultiImgExchangeTime)
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
    dict(type='MultiImgResize', scale=crop_size, keep_ratio=True),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs')
]

train_dataloader = dict(
    batch_size=64,
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
# 优化器与训练策略
# ---------------------------------------------------------------------------
optimizer = dict(
    type='AdamW', lr=0.0005, betas=(0.9, 0.999), weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    # 阶段1: 预热
    dict(type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=3000),
    # 阶段2: 余弦退火
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
                       img_shape=(256, 256, 3)))
