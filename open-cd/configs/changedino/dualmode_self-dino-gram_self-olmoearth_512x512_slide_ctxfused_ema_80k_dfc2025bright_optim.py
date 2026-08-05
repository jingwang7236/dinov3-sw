# DFC2025 BRIGHT 前光后 SAR 变化检测 — 4 分类 + 多 trick (双 backbone 全冻结, 显存友好)
# ============================================================================
# 本版改进 (提高有效分辨率以提 IoU):
#   - 光学 DINOv3 (RoPE) 原生支持任意分辨率, crop 保持 512 即可拿到细特征;
#   - 关键瓶颈在 SAR: OlmoEarth 原本 native_size=256 会把任意输入先下采样到
#     256 再过 ViT, 空间细节丢失。本版 native_size=256->384, 让 SAR ViT 真正
#     编码更细分辨率, 与光学分支细节对齐 (此前仅提 crop 不提 native 实测 test
#     mIoU 不升反降 67.09->66.74)。
#   - 显存控制: OlmoEarth 开 use_checkpoint (梯度检查点, transformer block 重算)
#     + train batch_size 8->4, 适配 native 384 (token 1024->2304, attn O(N^2))。
#   - 推理 stride 已是 (256,256) 50% 重叠, 改善边界。
# 其它 trick 同基线: 双 backbone 全冻结 / EMA / ms_inference+TTA / Lovász+Focal / 80k。
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
        freeze_mode='frozen',          # 冻结 DINOv3 ViT
    ),
    backbone_sar = dict(
        type='OlmoEarthSAREncoder',
        model_dir=olmoearth_model_dir,
        config_path=OlmoEarth_config_path,
        weights_path=OlmoEarth_weights_path,
        model_variant='base',
        in_channels=3,
        out_channels=128,
        freeze_backbone=True,           # 冻结 SAR backbone
        adaptive_pool=False,
        native_inference=True,
        native_size=384,                # 提高到 384, 让 SAR ViT 真正处理更细分辨率
        use_checkpoint=True,            # 梯度检查点, 降低 native↑ 带来的显存峰值
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
    batch_size=12,                      # native 384 显存增大
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
