# 蒸馏 Swin Transformer + Adapter backbone — LEVIR-CD 变化检测
# ============================================================================
# 基于 self-swin_standard_light_freeze_256x256_40k_levircd.py
# 将 SwinDistillBackbone (简单 1x1 Conv 投影) 替换为 SwinDistillAdapterBackbone
# (1x1 reduce → 3x3 dw → 1x1 proj)。
#
# 训练模式: Swin 主干始终冻结, 仅训练 adapter + decode_head。
# 可选两阶段: 第二阶段解冻 Swin 全量微调 (取消注释即可)。
#
# 对比实验:
#   self-swin_standard_light_freeze_256x256_40k_levircd.py
#     → SwinDistillBackbone: 简单投影, frozen → full_finetune 两阶段
#   self-swin_standard_light_adapter_freeze_256x256_40k_levircd.py (本文件)
#     → SwinDistillAdapterBackbone: adapter 投影, Swin 始终冻结
# ============================================================================
_base_ = [
    '../common/standard_512x512_40k_levircd.py']

crop_size = (256, 256)
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=[109.65, 104.805, 75.435] * 2,
    std=[54.315, 39.78, 36.465] * 2,
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32))
model = dict(
    type='ChangeDinoEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SwinDistillAdapterBackbone',
        out_channels=128,
        swin_weight='/mnt/qh2-nas3/00-model/00-wrs/zhejiang_earth_results/swin-distill/30999/swintransformer-huge-upsample.pt',
        swin_model='swin_huge',
        freeze_mode='frozen',
        # adapter_r=128,  # adapter bottleneck 维度, 默认等于 out_channels
    ),
    decode_head=dict(
        type='ChangeDinoDecoder',
        fpn_channels=128,
        n_layers=[1, 1, 1, 1],
        num_classes=2,
        align_corners=False,
        ignore_index=255,
    ),
    refiner=dict(
        type='LearnableSoftMorph',
        k_open=3,
        k_close=5,
        tau=0.05,
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotate', prob=0.5, degree=20),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    # dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    dict(type='MultiImgExchangeTime', prob=0.5),
    dict(
        type='MultiImgPhotoMetricDistortion',
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10),
    dict(type='MultiImgPackSegInputs')
]
train_dataloader = dict(
    batch_size=24,
    dataset=dict(pipeline=train_pipeline))

test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgResize', scale=crop_size, keep_ratio=True),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs')
]

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(pipeline=test_pipeline))

optimizer=dict(
    type='AdamW', lr=0.0005, betas=(0.9, 0.999), weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# compile = True # use PyTorch 2.x

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)

# ============================================================================
# 单阶段学习率调度 (Swin 始终冻结, 仅训练 adapter):
#   预热 (0 ~ 3k) + 余弦退火 (3k ~ 40k)
# ============================================================================
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=3000),
    dict(type='CosineAnnealingLR', T_max=37000, eta_min=1e-6,
         by_epoch=False, begin=3000, end=40000),
    # ---- 可选: 两阶段调度 (第二阶段解冻 Swin 全量微调) ----
    # dict(type='LinearLR', start_factor=0.1, end_factor=1.0,
    #      by_epoch=False, begin=20000, end=23000),
    # dict(type='CosineAnnealingLR', T_max=17000, eta_min=1e-7,
    #      by_epoch=False, begin=23000, end=40000),
]

# ============================================================================
# 冻结调度 Hook
#
# 默认: 全程冻结 Swin, 仅训练 adapter。
# 可选: 取消注释第二阶段条目以在 iter=20000 时全量微调。
# ============================================================================
custom_hooks = [
    dict(
        type='FreezeScheduleHook',
        backbone_attr='backbone',
        schedule=[
            # 全程冻结 Swin, 仅训练 adapter + decode_head
            dict(iter=0, freeze_mode='frozen'),
            # ---- 可选: 第二阶段全量微调 ----
            # dict(iter=20000, freeze_mode='full_finetune',
            #      load_from='{work_dir}/best_mIoU_*.pth'),
        ],
        verbose=True,
    )
]
