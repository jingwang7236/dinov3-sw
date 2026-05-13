# Chinasiwei Foundation Model


## DINOV3 pretrain
dinov3预训练仅支持torch2.7.1，其需要GLIBC_2.28,目前公司服务器系统均为2.17版本，因此目前需要在docker中进行预训练，可以使用`pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime`该镜像。

由于该镜像是runtime镜像，无法使用torch2+的compile加速，因此训练需设置`TORCH_COMPILE_DISABLE=1`。

可以使用dinov3原生代码预训练或使用DinoV3LightningTraining进行训练。区别在于加载，前者加载预训练权重的地方是自己实现的（@mutongyao）

使用`DINOv3`预训练的命令（单卡A800）：
```shell
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 TORCH_COMPILE_DISABLE=1 PYTHONPATH=${PWD}  python ./dinov3/train/train.py \
--config-file /mnt/mty/open_source_mm/chinasiwei_fm/dinov3-main/dinov3/configs/train/vit-large_chinasiwei/vitl_im1k_lin834_chinasiweidataset.yaml \
--output-dir ./work_dirs/003_dinov3_vit-l_20251117_all_bands_datset_with_downstream_task_data_150w

```


使用`DINOv3PytorchLightning`预训练的命令（单卡A800）：
```shell
CUDA_VISIBLE_DEVICES=1 TORCH_COMPILE_DISABLE=1 python src/training/train_dinov3_lightning.py \
--config-file configs/config_lightning_finetuning_chinasiweidataset_vit-l.yaml \
--output-dir ./output/003_dinov3lightning_vit-l_chinasiweidataset_all_bands_20251117_add_downstream_dataset_load_from_sat493m_pretrain \
--gpus 1 \
--sampler-type infinite \
# --sampler-type distributed \
--strategy ddp \
--checkpoint-path /mnt/mty/open_source_mm/chinasiwei_fm/dinov3-main/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth

```

## ChannelAdaptiveDINO + DINOV3 pretrain

已完成ChannelAdaptiveDINO（DINO-BoC）+ DINOv3的改造。

预训练的命令（单卡A800）：
```shell
cd /mnt/mty/open_source_mm/chinasiwei_fm/dinov3-main
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 TORCH_COMPILE_DISABLE=1 PYTHONPATH=${PWD}  python ./dinov3/train/train.py --config-file /mnt/mty/open_source_mm/chinasiwei_fm/dinov3-main/dinov3/configs/train/vit-large_chinasiwei/channel_adaptive_vitl_im1k_lin834_chinasiweidataset.yaml --output-dir ./work_dirs/005_channel_adaptive_dinov3_vit-l_20251226_all_bands_datset_with_downstream_task_data

```

## DINOv3 distillation

DINOv3论文对蒸馏的描述为，为节省资源，设计了`高效的多学生蒸馏`，即一次训练中，利用教师模型（7B）的一次推理，同时蒸馏多个版本的student。

相关配置的位置如下，其中`ranks_range`代表的是该学生训练时使用的gpu序号。
```yaml
distillation:
  enabled: true
  full_cfg_path: <PATH/TO/TEACHER/CONFIG/config.yaml>
  checkpoint_path: <PATH/TO/TEACHER/checkpoint.pth>
multidistillation:
  enabled: true
  global_batch_size: 1920
  students:
  - name: vits_mlp4_4
    config_path:  <PATH/TO/STUDENT/CONFIG/vits_mlp4_4.yaml>
    ranks_range:
    - 0
    - 48
  - name: vitsp_swiglu6_1
    config_path: <PATH/TO/STUDENT/CONFIG/vitsp_swiglu6_1.yaml>
    ranks_range:
    - 48
    - 96
  - name: vitb_mlp4_3
    config_path: <PATH/TO/STUDENT/CONFIG/vitb_mlp4_3.yaml>
    ranks_range:
    - 96
    - 176
  - name: vitl_mlp4_1
    config_path:  <PATH/TO/STUDENT/CONFIG/vitl_mlp4_1.yaml>
    ranks_range:
    - 176
    - 296
```

由于目前公司只能使用单卡进行训练，因此只能同时蒸馏一个学生模型，`ranks_range`需设为0,1。

具体命令如下：
```shell
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 TORCH_COMPILE_DISABLE=1 PYTHONPATH=${PWD} \
python ./dinov3/train/train.py \
--config-file /mnt/mty/open_source_mm/chinasiwei_fm/dinov3-main/dinov3/configs/train/vit-large_chinasiwei/multi_distillation.yaml \
--output-dir ./work_dirs/101_multidistillation_test \
--multi-distillation
```

目前使用ViT-L蒸馏ViT-S，`global_crops_size=256`, `local_crops_size=112`, `global_batch_size=256`, 在单张A800上需要消耗约72GB显存。

## 下游任务微调

下游任务微调部分的代码已适配到低版本pytorch，因此下游任务微调无需使用docker，直接用服务器上的环境即可，目前验证过程的下游任务微调均是在单张4090上完成的。

```shell
conda activate fm_env
```

目前下游任务微调的config都放在：`/mnt/mty/open_source_mm/chinasiwei_fm/configs/dinov3/`目录下

可修改的config参数如下：
```python
model = dict(
    backbone=dict(
        type='mmpretrain.DinoV3Backbone',
        checkpoint_path='/mnt/mty/open_source_mm/chinasiwei_fm/dinov3-main/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
        freeze_backbone=False,
        fpn=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_global_and_local_cls_norm=True,

        # (@mutongyao) 使用channelAdaptive方式训练的预训练模型时需配置下面两项。
        channel_adaptive=False,
        feat_fuse_method='mean'  # feat_fuse_method in 'mean' / 'conv'
    ),
)
```
- `checkpoint_path`: 加载的预训练权重
- `freeze_backbone`: 是否冻结骨干网络
- `fpn`: 是否对FPN层的init_weight执行单独的操作（主要是目标检测需要开启）
- `n_storage_tokens`、`mask_k_bias`、`untie_global_and_local_cls_norm`: 需根据预训练权重配置，详见本readme最后预训练权重章节
- `channel_adaptive`: 预训练模型是否开启了channel_adaptive
- `feat_fuse_method`: 在使用channel_adaptive时的特征融合方式




**请注意：**

目前`/mnt/mty/open_source_mm/chinasiwei_fm/dinov3-main/dinov3/models/vision_transformer.py#187` 存在HACK写法。在feat_fuse_method为conv时，默认了数据集是三通道影像。 后期待优化。


**说明：**

config中设置的backbone的位置在：/mnt/mty/open_source_mm/mmpretrain/mmpretrain/models/backbones/dinov3.py

其通过：

```python
    self.encoder = torch.hub.load(
        REPO_DIR, 
        model=model_name, 
        source='local', 
        weights=checkpoint_path,
        n_storage_tokens=n_storage_tokens,
        mask_k_bias=mask_k_bias,
        untie_global_and_local_cls_norm=untie_global_and_local_cls_norm,
    )
```
来加载dinov3官方的bockbone。所load的是`/mnt/mty/open_source_mm/chinasiwei_fm/dinov3-main/dinov3/hub/backbones.py`的DINOv3。

更多细节可以查看飞书云文档中有关下游任务微调改造的部分以及官方的issue。


### 语义分割

目前集成了`SegmenterMaskTransformerHead`和`UperHead`两种分割头。

训练全要素分割的命令：
```shell
cd /mnt/mty/open_source_mm/mmsegmentation
python ./tools/train.py /mnt/mty/open_source_mm/chinasiwei_fm/configs/dinov3/seg/dinov3_vit-l_1xb16-160k_multiclass-512x512.py

```


### 目标检测（旋转框）

目前参考MTP的目标检测下游任务配置，在mmrotate 1.x中集成了DINOv3 + Oriented-rcnn，已在DOTAv1.0 和 DIOR-R上跑通。


需要开启fpn。

训练命令：
```shell
cd /mnt/mty/open_source_mm/mmrotate-1.x
python ./tools/train.py /mnt/mty/open_source_mm/chinasiwei_fm/configs/dinov3/det/oriented_rcnn_dinov3_l_1024_dotav10.py

```

### 变化检测

目前Levir-CD上训练了`UNetHead`、`MLPSegHead`两种变化检测头，

训练命令：
```shell
cd /mnt/mty/open_source_mm/open-cd
python ./tools/train.py /mnt/mty/open_source_mm/chinasiwei_fm/configs/dinov3/cd/dinov3_vit-l_512x512_300e_levircd.py

python ./tools/train.py /mnt/mty/open_source_mm/chinasiwei_fm/configs/dinov3/cd/dinov3_vit-l_512x512_300e_levircd_mlphead.py

```


## 预训练权重

- DINOv3官方的预训练权重(ViT-L)：/mnt/mty/open_source_mm/chinasiwei_fm/dinov3-main/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
    ```python
    n_storage_tokens=4,
    mask_k_bias=True,
    untie_global_and_local_cls_norm=True,
    ```

- 使用29wRGB数据预训练的权重(ViT-L): /mnt/mty/open_source_mm/chinasiwei_fm/DinoV3LightningTraining/output/001_dinov3lightning_vit-l_chinasiweidataset_load_from_sat493m_pretrain/final_ssl_model_only_teacher-8aa4cbdd.pth
    ```python
    n_storage_tokens=0,
    mask_k_bias=False,
    untie_global_and_local_cls_norm=False,
    ```

- 使用66w数据预训练的权重(ViT-L): /mnt/mty/open_source_mm/chinasiwei_fm/DinoV3LightningTraining/output/002_dinov3lightning_vit-l_chinasiweidataset_all_bands_20251023_load_from_sat493m_pretrain/002_final_ssl_model_66w_only_teacher-eadcf0ff.pth
    ```python
    n_storage_tokens=0,
    mask_k_bias=False,
    untie_global_and_local_cls_norm=False,
    ```

- 使用153w数据,学习率1e-4预训练的权重(ViT-L): /mnt/mty/open_source_mm/chinasiwei_fm/DinoV3LightningTraining/output/005_dinov3lightning_vit-l_chinasiweidataset_all_bands_20251117_add_downstream_dataset_load_from_sat493m_pretrain_lr_1e-4/005_final_ssl_model-eadcf0ff.pth
    ```python
    n_storage_tokens=0,
    mask_k_bias=False,
    untie_global_and_local_cls_norm=False,
    ``` 

- 使用153w数据,学习率1e-4,修复训练config后预训练的权重(ViT-L): /mnt/mty/open_source_mm/chinasiwei_fm/DinoV3LightningTraining/output/006_dinov3lightning_vit-l_FIX_CONFIG_chinasiwei_20251117_add_downstream_dataset_load_from_sat493m_pretrain_lr_1e-4/006_final_ssl_model-eadcf0ff.pth
    ```python
    n_storage_tokens=4,
    mask_k_bias=True,
    untie_global_and_local_cls_norm=True,
    ```

- 使用裁剪为512*512的切片加下游任务数据集,学习率1e-4,修复训练config后预训练的权重(ViT-L): /mnt/mty/open_source_mm/chinasiwei_fm/dinov3-main/work_dirs/005_channel_adaptive_dinov3_vit-l_20251226_all_bands_datset_with_downstream_task_data/eval/training_374999/005_teacher_checkpoint-08c60483.pth
    ```python
    n_storage_tokens=4,
    mask_k_bias=True,
    untie_global_and_local_cls_norm=True,
    channel_adaptive=True,
    feat_fuse_method='mean' # 'conv'
    ```

NOTE: 将预训练权重处理成下游任务load支持的格式的代码在`test.ipynb`中