# 50.11 conda: dinov3-mmlab-wj2

# 复现levircd数据集指标，done，same with paper experiment
# cd /mnt/ht2-nas2/00-model/00-wj/Codes/dinov3-sw/open-cd
# python tools/train.py configs/changer/changer_ex_s101_512x512_40k_levircd.py \
# --work-dir work_dirs/changer_ex_s101_512x512_40k_levircd

# backbone 替换为dinov3，冻结权重；
# CUDA_VISIBLE_DEVICES=3 python tools/train.py configs/changedino/changedino_base_512x512_40k_levircd.py \
# --work-dir work_dirs/changedino_base_512x512_40k_levircd

# 测试模型
# TORCH_LOAD_WEIGHTS_ONLY=0 python tools/test.py \
# configs/changedino/changedino_base_512x512_40k_levircd.py \
# work_dirs/changedino_base_512x512_40k_levircd/iter_4000.pth


# CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/changedino/changedino_standard_512x512_40k_levircd_new.py

OPENCV_LOG_LEVEL=ERROR CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/changedino/dualmode_dinov3sarcnn_256x256_40k_dfc2025bright.py