# DFC2025 BRIGHT 前光后 SAR 变化检测 — 4 分类 + 两阶段微调 (解冻双 backbone 最后 2 层)
# ============================================================================
# 基于 *_optim.py, 在保留 native_size=384 / 双 backbone 初始冻结 / EMA /
# ms_inference+TTA / Lovász+Focal / 80k 的前提下, 通过 FreezeScheduleHook
# 进行两阶段训练:
#   阶段1 (0 ~ 40k iters):   双 backbone 完全冻结, 仅训练 adapter/decoder/
#                            feature_extractor/input_adapter (与 _optim 一致)
#   阶段2 (40k ~ 80k iters): 解冻 DINOv3 ViT 与 OlmoEarth SAR ViT 各最后 2 个
#                            transformer block (+ 最终 norm), 以极小学习率微调,
#                            提升域适配能力, 弥补"全冻结欠拟合/各类 IoU 普遍偏低"。
#
# 关键改动 (相对 _optim.py):
#   ① optimizer 增加 paramwise_cfg: backbone ViT blocks 用 1/10 lr, 避免解冻后
#     大学习率破坏预训练表征;
#   ② param_scheduler 改为两阶段 (阶段2 重新 warmup, 适配新解冻参数);
#   ③ custom_hooks 新增两个 FreezeScheduleHook (分别作用于 backbone_opt /
#      backbone_sar), 在 iter=40000 同步解冻最后 2 层;
#   ④ 不开启 use_checkpoint (保持 _optim 的高速), 显存上升靠降 bs/native 兜底。
#
# 注意: 解冻后显存显著上升 (尤其 SAR native 384 + ViT-L/12 反向)。
#       OOM 应对 (按需): bs 8->4~6, 或 native_size 384->320, 或减少解冻层数。
# ============================================================================

_base_ = ['../_base_/default_runtime.py']
olmoearth_model_dir = "/mnt/qh2-nas3/00-model/00-wrs/zhejiang_earth_results/olmoearth10m_base/step23400"
OlmoEarth_config_path = "{}/config.json".format(olmoearth_model_dir)
OlmoEarth_weights_path = "{}/weights.pth".format(olmoearth_model_dir)
Dino_weights_path = "/mnt/qh2-nas3/00-model/00-wrs/zhejiang_earth_results/zhejiang_DinoViT_large_Olmoearth10m_128gpu_stage2_stage3_no_cl_gram_nofusion/4999_new.pt"
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
        freeze_mode='frozen',          # 阶段1 全冻结; 阶段2 由 Hook 切到 unfreeze_last_n
    ),
    backbone_sar = dict(
        type='OlmoEarthSAREncoder',
        model_dir=olmoearth_model_dir,
        config_path=OlmoEarth_config_path,
        weights_path=OlmoEarth_weights_path,
        model_variant='base',
        in_channels=3,
        out_channels=128,
        freeze_backbone=True,           # 阶段1 全冻结; 阶段2 由 Hook 切到 unfreeze_last_n
        adaptive_pool=False,
        native_inference=True,
        native_size=384,                # 保留 384, 让 SAR ViT 真正处理更细分辨率
        use_checkpoint=False,           # ★ 关闭梯度检查点 (提速), 显存上升
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
        stride=(256, 256)),
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
# 优化器: decoder/adapter 用基准 lr; backbone ViT blocks 用 1/10 lr
# (paramwise_cfg 在 optimizer 构建时按参数名分组, 冻结参数 grad=None 会被
#  optimizer 跳过, 解冻后自动以该组 lr 更新)
# ---------------------------------------------------------------------------
optimizer = dict(
    type='AdamW', lr=0.0005, betas=(0.9, 0.999), weight_decay=0.0005)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    # paramwise_cfg 属于 OptimWrapperConstructor: 按参数名分组分配 lr_mult。
    # 冻结参数 grad=None 会被 optimizer 跳过, 解冻后自动以该组 lr 更新。
    paramwise_cfg=dict(
        custom_keys={
            # 光学 DINOv3 ViT blocks (ViT-L 共 24 层, 阶段2 仅最后 2 层解冻)
            'backbone.adapter.backbone.blocks': dict(lr_mult=0.1),
            # SAR OlmoEarth ViT blocks (base 共 12 层, 阶段2 仅最后 2 层解冻)
            'backbone_sar.blocks': dict(lr_mult=0.1),
        }))

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ---------------------------------------------------------------------------
# 两阶段学习率调度 (配合 FreezeScheduleHook, iter=40000 解冻):
#   阶段1 (0 ~ 5k):     线性预热
#   阶段1 (5k ~ 40k):   余弦退火到 5e-5 (阶段1 末, 为阶段2 低 lr 衔接)
#   阶段2 (40k ~ 43k):  重新预热 (新解冻的 ViT 后 2 层需小学习率起步)
#   阶段2 (43k ~ 80k):  余弦退火到 1e-6
# ---------------------------------------------------------------------------
param_scheduler = [
    # 阶段1: 预热 + 余弦退火
    dict(type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=5000),
    dict(type='CosineAnnealingLR', T_max=35000, eta_min=5e-5,
         by_epoch=False, begin=5000, end=40000),
    # 阶段2: 重新预热 (start_factor 相对当前 lr 再降到 1/10) + 余弦退火
    dict(type='LinearLR', start_factor=0.1, end_factor=1.0,
         by_epoch=False, begin=40000, end=43000),
    dict(type='CosineAnnealingLR', T_max=37000, eta_min=1e-6,
         by_epoch=False, begin=43000, end=80000),
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

# ---------------------------------------------------------------------------
# ① EMA 权重平滑 (val/test 自动换入 EMA 权重评估)
# ② 两个 FreezeScheduleHook: 分别对 backbone_opt (属性名 'backbone') 与
#    backbone_sar (属性名 'backbone_sar') 在 iter=40000 同步解冻最后 2 层。
#    两个 backbone 的 set_freeze_mode 接口签名一致, 均支持 unfreeze_last_n。
# ---------------------------------------------------------------------------
custom_hooks = [
    dict(type='EMAHook', momentum=2e-4, update_buffers=True, priority='LOWEST'),
    # 光学 DINOv3 分支: 模型属性为 self.backbone
    dict(
        type='FreezeScheduleHook',
        backbone_attr='backbone',
        schedule=[
            dict(iter=0, freeze_mode='frozen'),
            dict(iter=40000, freeze_mode='unfreeze_last_n',
                 unfreeze_last_n=2),
        ],
        verbose=True,
    ),
    # SAR OlmoEarth 分支: 模型属性为 self.backbone_sar
    dict(
        type='FreezeScheduleHook',
        backbone_attr='backbone_sar',
        schedule=[
            dict(iter=0, freeze_mode='frozen'),
            dict(iter=40000, freeze_mode='unfreeze_last_n',
                 unfreeze_last_n=2),
        ],
        verbose=True,
    ),
]
