
# 基于dinov3的下游任务微调

## 语义分割
export PYTHONPATH=/mnt/ht2-nas2/00-model/00-wj/Codes/dinov3-sw:$PYTHONPATH
cd /mnt/ht2-nas2/00-model/00-wj/Codes/dinov3-sw/mmsegmentation
CUDA_VISIBLE_DEVICES=1 python ./tools/train.py  \
/mnt/ht2-nas2/00-model/00-wj/Codes/dinov3-sw/chinasiwei_fm/configs/dinov3/seg/dinov3_vit-l_uperhead_1xb16-160k_neontree-512x512.py