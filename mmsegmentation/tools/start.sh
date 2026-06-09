# 50.11 conda: dinov3-mmlab-wj2
# 下游任务微调，语义分割
export PYTHONPATH="/mnt/ht2-nas2/00-model/00-wj/Codes/dinov3-sw:$PYTHONPATH"
export PYTHONPATH="/mnt/ht2-nas2/00-model/00-wj/Codes/dinov3-sw/mmsegmentation:$PYTHONPATH"
# 执行上述两行之后，当前终端就能找到自定义模块 DinoV3BackboneSimple
# cd /mnt/ht2-nas2/00-model/00-wj/Codes/dinov3-sw/mmsegmentation

# CUDA_VISIBLE_DEVICES=5 \
# python tools/train.py ./configs/dinov3/dinov3_freeze_vitL_uperhead_1xb2-160k_neontree-512x512.py


CUDA_VISIBLE_DEVICES=3,5 \
bash ./tools/dist_train.sh \
./configs/dinov3/dinov3_freeze_vitL_uperhead_2xb6-160k_neontree-512x512.py 2