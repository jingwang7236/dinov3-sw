
_base_ = [
    '/mnt/mty/open_source_mm/open-cd/configs/common/standard_256x256_100e_levircd.py']

experiment_name = '008_levircd_dinov3_load_from_FIX_CONFIG_chinasiwei_153w_lr_1e-4_pretrain_512x512_mlphead_lr_1e-6_bs4_amp'
work_dir = f'./work_dirs/dinov3/levircd/{experiment_name}'


crop_size = (512, 512)
input_size = 512

train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    # dict(type='MultiImgExchangeTime', prob=0.5),
    dict(
        type='MultiImgPhotoMetricDistortion',
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10),
    dict(type='MultiImgPackSegInputs')
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(
    batch_size=4,
    # batch_size=2,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(pipeline=train_pipeline))


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
    type='SiamEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='mmpretrain.DinoV3Backbone',
        checkpoint_path='/mnt/mty/open_source_mm/chinasiwei_fm/DinoV3LightningTraining/output/006_dinov3lightning_vit-l_FIX_CONFIG_chinasiwei_20251117_add_downstream_dataset_load_from_sat493m_pretrain_lr_1e-4/006_final_ssl_model-eadcf0ff.pth',
        # checkpoint_path='/mnt/mty/open_source_mm/chinasiwei_fm/dinov3-main/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
        freeze_backbone=False,
        fpn=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_global_and_local_cls_norm=True,
    ),
    neck=dict(
        type='SequentialNeck',
        necks=[
            dict(
                type='DualInputsFPN',
                in_channels=[1024, 1024, 1024, 1024],
                out_channels=256,
                num_outs=5,
                # init_cfg=dict(
                #     type='Pretrained',
                #     checkpoint=pretrained_ckpt,
                #     prefix='pre_neck.'),
            ),
            dict(
                type='DualInputsSimpleFusionNeck',
                in_channels=[256, 256, 256, 256, 256],
                return_tuple=True,
            ),
            dict(
                type='UNetDecodeNeck',
                in_channels=[256] * 5,
                dec_num_convs=(2, 2, 2, 2),
                dec_dilations=(1, 1, 1, 1),
                with_cp=False,
                conv_cfg=None,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU'),
                upsample_cfg=dict(type='mmseg.InterpConv'),
            )
        ]
    ),
    decode_head=dict(
        type='MLPSegHead',
        out_size=(input_size // 4, input_size // 4),
        in_channels=[256] * 5,
        in_index=[0, 1, 2, 3, 4],
        channels=256,
        dropout_ratio=0,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='mmseg.DiceLoss', loss_weight=3.0)
        ]
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(input_size, input_size), stride=(input_size // 2, input_size // 2))
)
# optimizer
max_epochs = 300

# optim_wrapper = dict(
#     _delete_=True,
#     type='OptimWrapper',
#     optimizer=dict(
#         type='AdamW', lr=0.0004, betas=(0.9, 0.999), weight_decay=0.05))

# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=1e-4, by_epoch=True, begin=0, end=5, convert_to_iter_based=True),
#     dict(
#         type='CosineAnnealingLR',
#         T_max=max_epochs,
#         begin=5,
#         by_epoch=True,
#         end=max_epochs,
#         convert_to_iter_based=True
#     ),
# ]


optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.05))

param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=1e-6, 
        by_epoch=True, 
        begin=0, 
        end=5, 
        convert_to_iter_based=True
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs,
        begin=5,
        by_epoch=True,
        end=max_epochs,
        convert_to_iter_based=True
    ),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)
default_hooks = dict(checkpoint=dict(interval=5))