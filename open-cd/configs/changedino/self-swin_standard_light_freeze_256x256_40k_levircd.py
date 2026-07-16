# 蒸馏 Swin Transformer backbone — LEVIR-CD 变化检测
# ============================================================================
# 基于 self-dino_standard_light_adapter_freeze_256x256_40k_levircd.py
# 将 DINOv3 backbone 替换为自研蒸馏得到的 Swin-huge backbone。
#
# Swin 天然有 4 级层级特征 (/4, /8, /16, /32)，无需 adapter，
# 直接提取 + 1x1 卷积投影即可。支持三种训练模式通过 FreezeScheduleHook 切换:
#   阶段1 (0~20k):    frozen — 冻结 Swin，仅训练投影层 + decode_head
#   阶段2 (20k~40k):  full_finetune — 全量微调 (lr 降至 1/10)
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
        type='SwinDistillBackbone',
        out_channels=128,
        swin_weight='/mnt/qh2-nas3/00-model/00-wrs/zhejiang_earth_results/swin-distill/30999/swintransformer-huge-upsample.pt',
        swin_model='swin_huge',
        freeze_mode='frozen',
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
# 两阶段学习率调度:
#   阶段1 (0 ~ 3k):    线性预热
#   阶段1 (3k ~ 20k):  余弦退火
#   阶段2 (20k ~ 23k): 重新预热（新解冻参数需要小学习率起步）
#   阶段2 (23k ~ 40k): 余弦退火
# ============================================================================
param_scheduler = [
    # 阶段1: 预热 + 余弦退火
    dict(type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=3000),
    dict(type='CosineAnnealingLR', T_max=17000, eta_min=1e-6,
         by_epoch=False, begin=3000, end=20000),
    # 阶段2: 重新预热 + 余弦退火 (lr 降至 1/10 适配全量微调)
    dict(type='LinearLR', start_factor=0.1, end_factor=1.0,
         by_epoch=False, begin=20000, end=23000),
    dict(type='CosineAnnealingLR', T_max=17000, eta_min=1e-7,
         by_epoch=False, begin=23000, end=40000),
]

# ============================================================================
# 动态冻结调度 Hook — 在指定迭代次数切换 Swin 训练模式
# ============================================================================
custom_hooks = [
    dict(
        type='FreezeScheduleHook',
        backbone_attr='backbone',
        schedule=[
            # 阶段1: 冻结 Swin 主干
            dict(iter=0, freeze_mode='frozen'),
            # 阶段2: iter=20000 时全量微调
            dict(iter=20000, freeze_mode='full_finetune',
                 load_from='{work_dir}/best_mIoU_*.pth'),
            # 替代方案: 仅解冻最后 N 个 block（显存更友好）
            # dict(iter=20000, freeze_mode='unfreeze_last_n',
            #      unfreeze_last_n=4,
            #      load_from='{work_dir}/best_mIoU_*.pth'),
        ],
        verbose=True,
    )
]
