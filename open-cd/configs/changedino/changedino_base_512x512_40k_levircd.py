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
    type='DIEncoderDecoder',  # 孪生网络结构
    data_preprocessor=data_preprocessor,
    pretrained=None,
    # backbone=dict(
    #     type='IA_ResNeSt',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     dilations=(1, 1, 1, 1),
    #     strides=(1, 2, 2, 2),
    #     norm_cfg=norm_cfg,
    #     norm_eval=False,
    #     style='pytorch',
    #     contract_dilation=True,
    #     stem_channels=64,
    #     radix=2,
    #     reduction_factor=4,
    #     avg_down_stride=True,
    #     interaction_cfg=(None, None, None, None)),
    backbone=dict(
        type='SiameseDinoV3Backbone',
        model_name='vit_large',  # 可选: vit_large, vit_giant, vit_7b
        checkpoint_path='/mnt/ht2-nas2/00-model/00-wj/Codes/checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
        freeze_backbone=True,  # 训练后期可以考虑解冻后1-2层训练
        fpn=True,
        patch_size=16,
        img_size=512,
        use_grad_checkpoint=True,  # 内存不足时可开启
    ),
    decode_head=dict(
        type='Changer',
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotFlip', rotate_prob=0.5, flip_prob=0.5, degree=(-20, 20)),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
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
    batch_size=48,
    dataset=dict(pipeline=train_pipeline))

optimizer=dict(
    type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# compile = True # use PyTorch 2.x