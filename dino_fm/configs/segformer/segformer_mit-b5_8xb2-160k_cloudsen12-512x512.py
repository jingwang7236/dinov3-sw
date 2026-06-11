
_base_ = [
    './_base_/datasets/cloudsen12.py',
    './_base_/default_runtime.py', 
    './_base_/schedules/schedule_160k.py'
]

experiment_name = '004_segformer_b5_cloudsen12_rgb_uint8_input_add_20260107_all_and_no_cloud_data'
work_dir = f'./work_dirs/cloud_seg/{experiment_name}'

checkpoint = '/mnt/mty/ISASeg/mmseg-main/mit_b5_20220624-658746d9.pth'  # noqa

norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)
num_classes = 4

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=False,
    # bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

# model settings
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformer',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 6, 40, 3],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))



optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=10, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
# test_dataloader = val_dataloader


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='PackSegInputs')
]


test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CloudTestDataset',
        # data_root='/mnt/CLOUD/data/cloud-inspur/m_text/Cloud_0_train/add_buaa_snow_train_data_0909_parse_label/mtytestdata/2293',
        # data_root='/mnt/CLOUD/data/20260105_test_image',
        data_root='//mnt/CLOUD/data/cloud_shadow__thincloud_test_image',
        ann_file='',
        data_prefix=dict(
            img_path='',
            seg_map_path=''),
        pipeline=test_pipeline))

default_hooks = dict(
    visualization=dict(type='SegVisualizationHook'))
# default_hooks = dict(
#     visualization=dict(type='CloudSegVisualizationHook'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='CloudSegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], format_only=True, output_dir=f'{work_dir}/results')
