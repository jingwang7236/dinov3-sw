# ChangeDino + DINOv3AdapterBackbone (LEVIRD-CD) — 动态冻结策略
# ============================================================================
# 本配置演示 DINOv3AdapterBackbone 的三种 freeze_mode 及两阶段训练调度:
#
#   freeze_mode 选项:
#     'frozen'          — 完全冻结 ViT 主干，仅训练 adapter/proj/decoder (默认)
#     'full_finetune'   — 全量微调 ViT 主干所有参数
#     'unfreeze_last_n' — 仅解冻最后 n 层 transformer block + norm
#
#   两阶段训练调度 (通过 FreezeScheduleHook):
#     阶段1 (0 ~ 20k iters):   frozen    — adapter 先学好变化检测
#     阶段2 (20k ~ 40k iters): full_finetune — 加载阶段1 best 权重后全量微调
#
#   注意: full_finetune 会大幅增加显存（ViT-L 需存储前向激活），
#   如 OOM 请减小 batch_size 或改用 'unfreeze_last_n'。
# ============================================================================
_base_ = [
    '../common/standard_512x512_40k_levircd.py']

crop_size = (256, 256)
# model settings
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
    type='ChangeDinoEncoderDecoder',  # ChangeDino网络结构,同论文
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='DINOv3AdapterBackbone',
        out_channels=128,
        extract_ids=[5, 11, 17, 23],
        dino_weight='/mnt/ht2-nas2/00-model/00-wj/Codes/checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
        freeze_mode="frozen",
    ),
    decode_head=dict(
        type='ChangeDinoDecoder',
        fpn_channels=128,
        n_layers=[1, 1, 1, 1],
        num_classes=2,
        align_corners=False,
        ignore_index=255,
    ),
    # 可选：后处理模块
    refiner=dict(
        type='LearnableSoftMorph',
        k_open=3,
        k_close=5,
        tau=0.05,
    ),
    # model training and testing settings
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
# 动态冻结调度 Hook
# ============================================================================
custom_hooks = [
    dict(
        type='FreezeScheduleHook',
        backbone_attr='backbone',
        schedule=[
            # 阶段1: 冻结 ViT 主干
            dict(iter=0, freeze_mode='frozen'),
            # 阶段2: iter=20000 时，加载阶段1 best checkpoint，然后全量微调
            #   load_from 支持 glob 模式，自动匹配 best_mIoU_*.pth
            #   如不想加载 checkpoint，删去 load_from 即可从当前权重继续
            dict(iter=20000, freeze_mode='full_finetune',
                 load_from='{work_dir}/best_mIoU_*.pth'),
            # 替代方案: 仅解冻最后 4 层（显存更友好）
            # dict(iter=20000, freeze_mode='unfreeze_last_n',
            #      unfreeze_last_n=4,
            #      load_from='{work_dir}/best_mIoU_*.pth'),
        ],
        verbose=True,
    )
]
