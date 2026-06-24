# ChangeDINO-Light + LoRA 插件: 冻结 DINOv3 全部权重，注入 LoRA 低秩适配器高效微调
_base_ = [
    '../common/standard_256x256_40k_sysucd.py']

crop_size = (256, 256)
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=[123.675, 116.28, 103.53] * 2,
    std=[58.395, 57.12, 57.375] * 2,
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32))

model = dict(
    type='ChangeDinoEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ChangeDinoEncoderLoRA',
        out_channels=128,
        extract_ids=[5, 11, 17, 23],
        dino_weight='/mnt/ht2-nas2/00-model/00-wj/Codes/checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
        # ====== LoRA 插件参数 ======
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_target_modules=["qkv", "proj", "fc1", "fc2"],
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

train_dataloader = dict(batch_size=8)
val_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False))
test_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False))

# 注意: paramwise_cfg 必须放在 optim_wrapper 下，不能放在 optimizer 内部
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.0005),
    paramwise_cfg=dict(
        custom_keys={
            'lora_': dict(lr_mult=1.0),
            'decode_head': dict(lr_mult=0.5),
            'dense_adp': dict(lr_mult=0.5),
            'refiner': dict(lr_mult=0.5),
        }
    ))

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', T_max=39000, eta_min=1e-7,
         by_epoch=False, begin=1000, end=40000),
]
