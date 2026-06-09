# 50.11 conda: dinov3-mmlab-wj2

# 复现levircd数据集指标，done，same with paper experiment
# cd /mnt/ht2-nas2/00-model/00-wj/Codes/dinov3-sw/open-cd
python tools/train.py configs/changer/changer_ex_s101_512x512_40k_levircd.py \
--work-dir work_dirs/changer_ex_s101_512x512_40k_levircd

# backbone 替换为dinov3，冻结权重；
# python tools/train.py configs/changedino/changedino_base_512x512_40k_levircd.py \
# --work-dir work_dirs/changedino_base_512x512_40k_levircd