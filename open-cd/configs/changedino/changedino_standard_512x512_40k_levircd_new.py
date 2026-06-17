# changedino_standard_512x512_40k_levircd.py比论文指标低10个点左右，需要优化配置和代码
_base_ = [
    # '../_base_/models/changer_s50.py', 
    '../common/standard_512x512_40k_levircd.py']

crop_size = (512, 512)
# model settings
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
    type='ChangeDinoEncoderDecoder',  # ChangeDino网络结构,同论文
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ChangeDinoEncoder',
        backbone="mobilenetv2",
        fpn_channels=128,
        extract_ids=[5, 11, 17, 23],
        dino_weight='/mnt/ht2-nas2/00-model/00-wj/Codes/checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
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
    dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
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
]
train_dataloader = dict(
    batch_size=12,
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
    type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.05)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# compile = True # use PyTorch 2.x

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)