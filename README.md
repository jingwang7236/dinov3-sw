# Foundation Model and Downstream task based on DINOv3


## DINOV3 pretrain

使用`DINOv3`预训练的命令（单卡A800）：
```shell
# 使用png/tiff图像从头训练vit-Large模型,单机单卡
cd dino_fm/dinov3-main
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 TORCH_COMPILE_DISABLE=1 PYTHONPATH=${PWD}  python ./dinov3/train/train.py \
--config-file ./dinov3/configs/train/vit-large_chinasiwei/vitl_swdata_pretrain_from_scracth.yaml \
--output-dir work_dirs

# 使用png/tiff图像从头训练vit-Large模型,单机多卡
cd dino_fm/dinov3-main
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARN
export MASTER_PORT=39500
export PYTHONPATH=${PWD}
export CUDA_LAUNCH_BLOCKING=1 
export TORCH_COMPILE_DISABLE=1
# 使用 torchrun 启动分布式训练
torchrun \
    --nproc_per_node=4 \
    --master_port=$MASTER_PORT \
    ./dinov3/train/train.py \
    --config-file ./dinov3/configs/train/vitlarge_chinasiwei/vitl_swdata_pretrain_from_scracth.yaml \
    --output-dir work_dirs_multi_gpu
```


## 下游任务微调


```shell
conda activate dinov3-mmlab-wj2
```


### 语义分割

目前主要测试了`UperHead`分割头。

训练全要素分割的命令：
```shell
cd mmsegmentation
CUDA_VISIBLE_DEVICES=3,5 \
bash ./tools/dist_train.sh \
./configs/dinov3/dinov3_freeze_vitL_uperhead_2xb6-160k_neontree-512x512.py 2

```


### 变化检测

目前Levir-CD上训练了`Changer`、`ChangeDino`两种变化检测头，

训练命令：
```shell
cd open-cd
python tools/train.py configs/changer/changer_ex_s101_512x512_40k_levircd.py \
--work-dir work_dirs/changer_ex_s101_512x512_40k_levircd

CUDA_VISIBLE_DEVICES=5 python tools/train.py configs/changedino/changedino_standard_512x512_40k_levircd.py \
--work-dir work_dirs/changedino_standard_512x512_40k_levircd

```


## 预训练权重

- DINOv3官方的预训练权重(ViT-L)：dino_fm/dinov3-main/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
    ```python
    n_storage_tokens=4,
    mask_k_bias=True,
    untie_global_and_local_cls_norm=True,
    ```


NOTE: 将预训练权重处理成下游任务load支持的格式的代码在`test.ipynb`中