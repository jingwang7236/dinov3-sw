_base_ = [
    '../common/standard_256x256_40k_sysucd.py']

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
        # freeze_mode: ViT 主干冻结模式
        #   'frozen'          — 完全冻结，仅训练 adapter/proj (默认，最省显存)
        #   'full_finetune'   — 全量微调 ViT 所有参数 (显存需求大)
        #   'unfreeze_last_n' — 仅解冻最后 N 层 block + norm (需配合 unfreeze_last_n=N)
        freeze_mode="frozen",
        # unfreeze_last_n=4,  # 仅 freeze_mode='unfreeze_last_n' 时生效
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


train_dataloader = dict(
    batch_size=64,
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

param_scheduler = [
    # 阶段1: 长预热 (3000 iterations)
    dict(
        type='LinearLR',
        start_factor=1e-5,
        by_epoch=False,
        begin=0,
        end=3000,
    ),
    # 阶段2: 余弦退火 (从最高学习率衰减到最低)
    dict(
        type='CosineAnnealingLR',
        T_max=37000,  # 总迭代数 - 预热迭代数
        eta_min=1e-6,
        by_epoch=False,
        begin=3000,
        end=40000,
    )
]