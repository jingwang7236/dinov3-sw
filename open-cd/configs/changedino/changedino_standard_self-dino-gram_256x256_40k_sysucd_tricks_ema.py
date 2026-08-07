# ChangeDino + 自研 DINOv3-gram (SYSU-CD) — 全 trick 汇总, 40k
# ============================================================================
# 基于 changedino_standard_self-dino-gram_256x256_40k_sysucd_new.py, 同时启用:
#
# [推理 trick] (零训练成本, 已加入 ChangeDinoEncoderDecoder):
#   ① ms_inference: p2/p3/p4/p5 logits 加权融合 [0.5,0.3,0.1,0.1], 改善边界/小目标
#   ② TTA: 水平 + 垂直翻转 [3,2] 推理后翻回取平均
#
# [训练 trick]:
#   ③ 数据增强补垂直翻转 (航拍无方向偏好, 双翻增益明显)
#   ④ 两阶段冻结: 阶段1(0~20k) DINO frozen; 阶段2(20k~40k) 解冻最后 4 层微调
#      (比原解冻 2 层更充分)
#   ⑤ EMA 权重平滑 (mobilenetv2 已加载 ImageNet 权重, EMA 影子不再贴近随机,
#      此前 val 全 0 问题已解决)
#
# 训练规模: 单卡 bs=16, lr=0.0005, 40k iters。SYSU-CD train=12000 对 =>
#           1 epoch=750 iters, 40k iters ≈ 53 epoch (<=100, 充分且不冗余)。
#
# 保留既有 trick: mobilenet ImageNet 预训练 / 自研 DINOv3-gram / Focal+Lovász /
#                 SoftMorph refiner / 余弦退火。
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
    # ===== 推理 trick =====
    ms_inference=True,                         # ① 多尺度 logits 加权融合
    ms_inference_weights=[0.5, 0.3, 0.1, 0.1],
    tta_flips=[3, 2],                          # ② 水平 + 垂直翻转 TTA
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))                # whole 推理; 大图可改 slide

train_dataloader = dict(
    batch_size=16,
    # ③ 数据增强补垂直翻转 (覆盖 base 的 train_pipeline)
    dataset=dict(pipeline=[
        dict(type='MultiImgLoadImageFromFile'),
        dict(type='MultiImgLoadAnnotations'),
        dict(type='MultiImgRandomRotate', prob=0.5, degree=20),
        dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
        dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
        dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
        dict(type='MultiImgExchangeTime', prob=0.5),
        dict(
            type='MultiImgPhotoMetricDistortion',
            brightness_delta=10,
            contrast_range=(0.8, 1.2),
            saturation_range=(0.8, 1.2),
            hue_delta=10),
        dict(type='MultiImgPackSegInputs')
    ]),
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

# 训练规模: 40k iters (≈53 epoch @ bs=16)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ============================================================================
# 两阶段学习率调度 (配合 FreezeScheduleHook):
#   阶段1 (0 ~ 3k):    线性预热
#   阶段1 (3k ~ 20k):  余弦退火
#   阶段2 (20k ~ 23k): 重新预热 (新解冻的 ViT 后4层需小学习率起步)
#   阶段2 (23k ~ 40k): 余弦退火
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
# Trick: EMA 权重平滑 + FreezeScheduleHook 两阶段冻结调度
#   EMA 延迟启动 (ema_start_iter=5000): warmup(0~3000) 与首 val(4000) 之间不启动
#   EMA, val 直接用真实权重, 避免把 warmup 期欠训练的随机预测链路(FPN/adapter/
#   pff/decoder/refiner)平均进影子导致早期 val 塌缩到多数类(changed=0)。
#   到达 iter 5000 (warmup 后、阶段1余弦稳定区) 用"当前已训练权重"初始化影子,
#   之后正常 EMA。这样从源头避免"随机预测链路"污染影子。
#   (注: mmengine hook 顺序下, 日志 val 指标=EMA 权重, save_best 存的是真实权重,
#    两者可能略有差异, 非 NaN)
#   阶段1 (0 ~ 20k iters):     DINO 分支完全 frozen, 仅训练 CNN/FPN/adapter/decoder
#   阶段2 (20k ~ 40k iters):   解冻 DINO 最后 4 层 transformer block 微调
# ============================================================================
custom_hooks = [
    dict(type='EMAHook', momentum=2e-4, update_buffers=True,
         ema_start_iter=5000, priority='LOWEST'),
    dict(
        type='FreezeScheduleHook',
        backbone_attr='backbone',
        schedule=[
            # 阶段1: 前半段冻结 DINO 分支
            dict(iter=0, freeze_mode='frozen'),
            # 阶段2: 后半段解冻最后 4 层微调 (比解冻 2 层更充分)
            dict(iter=20000, freeze_mode='unfreeze_last_n',
                 unfreeze_last_n=4),
        ],
        verbose=True,
    ),
]
