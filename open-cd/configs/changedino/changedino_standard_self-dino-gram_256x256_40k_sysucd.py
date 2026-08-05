# ChangeDino + DINOv3Backbone (SYSU-CD) — 自研权重 + 两阶段冻结, 40k
# ============================================================================
# 基于 changedino_standard_256x256_40k_sysucd.py 版本替换自研权重，对比效果。
# 训练策略 (通过 FreezeScheduleHook):
#   阶段1 (0 ~ 20k iters):     DINO 分支 frozen，仅训练 CNN/FPN/adapter/decoder
#   阶段2 (20k ~ 40k iters):   解冻 DINO 最后 2 层 transformer block 微调
# 新增 Trick: Focal+Lovász 损失 / EMA / SoftMorph refiner。
# ============================================================================
_base_ = [
    '../common/standard_256x256_40k_sysucd.py']
Dino_weights_path = "/mnt/qh2-nas3/00-model/00-wrs/zhejiang_earth_results/zhejiang_DinoViT_large_Olmoearth10m_128gpu_stage2_stage3_no_cl_gram_nofusion/4999_new.pt"
Dino_weights_type = "self_trained"
Mobilenet_weights_path = "/mnt/ht2-nas2/00-model/00-wj/Codes/checkpoints/mobilenet_v2_imagenet.pth"
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
        type='ChangeDinoEncoder',
        backbone="mobilenetv2",
        fpn_channels=128,
        extract_ids=[5, 11, 17, 23],
        dino_weight=Dino_weights_path,  # 自研权重
        mobilenet_pretrained=Mobilenet_weights_path,  # ImageNet 预训练权重
    ),
    decode_head=dict(
        type='ChangeDinoDecoder',
        fpn_channels=128,
        n_layers=[1, 1, 1, 1],
        num_classes=2,
        align_corners=False,
        ignore_index=255,
        # Trick: 类别加权 Focal (增强正样本/变化类) + Lovász (直接优化 mIoU/IoU)
        focal_alpha=[0.5, 1.0],
        focal_gamma=4.0,
        lovasz_weight=1.0,
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


train_dataloader = dict(
    batch_size=36,
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
)

optimizer=dict(
    type='AdamW', lr=0.0005, betas=(0.9, 0.999), weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# compile = True # use PyTorch 2.x

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ============================================================================
# 两阶段学习率调度 (配合 FreezeScheduleHook):
#   阶段1 (0 ~ 3k):     线性预热
#   阶段1 (3k ~ 20k):   余弦退火
#   阶段2 (20k ~ 23k):  重新预热（新解冻的 ViT 后2层需小学习率起步）
#   阶段2 (23k ~ 40k):  余弦退火
# ============================================================================
param_scheduler = [
    # 阶段1: 预热 + 余弦退火
    dict(type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=3000),
    dict(type='CosineAnnealingLR', T_max=17000, eta_min=1e-6,
         by_epoch=False, begin=3000, end=20000),
    # 阶段2: 重新预热 + 余弦退火 (lr 降至 1/10 适配微调)
    dict(type='LinearLR', start_factor=0.1, end_factor=1.0,
         by_epoch=False, begin=20000, end=23000),
    dict(type='CosineAnnealingLR', T_max=17000, eta_min=1e-7,
         by_epoch=False, begin=23000, end=40000),
]

# checkpoint 保留 best + 最近 5 个
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000,
                    save_best='mIoU', max_keep_ckpts=5))

# ============================================================================
# Trick: EMA 权重平滑 (val/test 自动换入 EMA 权重评估)
# ============================================================================
# Trick: FreezeScheduleHook 两阶段冻结调度
#   阶段1 (0 ~ 20k iters):     DINO 分支完全 frozen，仅训练 CNN/FPN/adapter/decoder
#   阶段2 (20k ~ 40k iters):   解冻 DINO 最后 2 层 transformer block 微调
# ============================================================================
custom_hooks = [
    dict(type='EMAHook', momentum=2e-4, update_buffers=True, priority='LOWEST'),
    dict(
        type='FreezeScheduleHook',
        backbone_attr='backbone',
        schedule=[
            # 阶段1: 前半段冻结 DINO 分支
            dict(iter=0, freeze_mode='frozen'),
            # 阶段2: 后半段解冻最后 2 层微调
            #   可选 load_from 加载阶段1 best checkpoint (glob 自动匹配)
            dict(iter=20000, freeze_mode='unfreeze_last_n',
                 unfreeze_last_n=2),
        ],
        verbose=True,
    ),
]