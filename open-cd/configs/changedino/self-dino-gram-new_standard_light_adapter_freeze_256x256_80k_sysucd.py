# ChangeDino + DINOv3AdapterBackbone (SYSU-CD) — 全程冻结 ViT 主干, 80k
# ============================================================================
# 基于 40k 版本延长训练 (40k 时 best 出现在最后一个 iter_40000, 疑未收敛)。
# 调整: max_iters 80k; warmup 5k; 余弦退火 75k -> eta_min=1e-6;
#       val/checkpoint 间隔 8000; 保留 EMA / Focal+Lovász / SoftMorph refiner。
# ViT 主干全程冻结, 仅训练 adapter / 投影层 / decoder / refiner。
# ============================================================================
_base_ = [
    '../common/standard_256x256_40k_sysucd.py']
Dino_weights_path = "/mnt/qh2-nas3/00-model/00-wrs/zhejiang_earth_results/zhejiang_DinoViT_large_Olmoearth10m_128gpu_stage2_stage3_no_cl_gram_nofusion/4999_new.pt"
Dino_weights_type = "self_trained"

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
    type='ChangeDinoEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='DINOv3AdapterBackbone',
        out_channels=128,
        extract_ids=[5, 11, 17, 23],
        dino_weight=Dino_weights_path,
        weights_type=Dino_weights_type,  # 'official' / 'self_trained' / 'auto'
        # dino_weight='/mnt/ht2-nas2/00-model/00-wj/Codes/checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
        # weights_type='official',
        freeze_mode="frozen",
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
    refiner=dict(
        type='LearnableSoftMorph',
        k_open=3,
        k_close=5,
        tau=0.05,
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


train_dataloader = dict(
    batch_size=24,
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

# 80k 训练: max_iters 80000, 每 8000 iter 验证一次
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ============================================================================
# 单阶段学习率调度 (ViT 全程冻结):
#   0 ~ 5k:   线性预热 (延长 warmup, 适配更长训练)
#   5k ~ 80k: 余弦退火到 eta_min
# ============================================================================
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=5000),
    dict(type='CosineAnnealingLR', T_max=75000, eta_min=1e-6,
         by_epoch=False, begin=5000, end=80000),
]

# checkpoint 间隔对齐到 8000, 仅保留 best + 最近 5 个
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000,
                    save_best='mIoU', max_keep_ckpts=5))

# ============================================================================
# Trick: EMA 权重平滑 (val/test 自动换入 EMA 权重评估, 通常 +0.3~1.0 mIoU)
# ============================================================================
custom_hooks = [
    dict(type='EMAHook', momentum=2e-4, update_buffers=True, priority='LOWEST'),
]
