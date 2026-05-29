# 单机单卡，从头预训练
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 TORCH_COMPILE_DISABLE=1 PYTHONPATH=${PWD} \
# python ./dinov3/train/train.py \
#     --config-file ./dinov3/configs/train/vit-large_chinasiwei/vitl_swdata_pretrain.yaml \
#     --output-dir work_dirs/vitl_swdata_pretrain

# 单机多卡，从头预训练
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export NCCL_DEBUG=WARN
# export MASTER_PORT=39500
# export PYTHONPATH=${PWD}
# export CUDA_LAUNCH_BLOCKING=1 
# export TORCH_COMPILE_DISABLE=1
# # 使用 torchrun 启动分布式训练
# torchrun \
#     --nproc_per_node=4 \
#     --master_port=$MASTER_PORT \
#     ./dinov3/train/train.py \
#     --config-file ./dinov3/configs/train/vit-large_chinasiwei/vitl_swdata_pretrain.yaml \
#     --output-dir work_dirs/multi_gpu

# 二阶段gram模型微调
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE_DISABLE=1 PYTHONPATH=${PWD} \
# python ./dinov3/train/train.py \
#     --config-file ./dinov3/configs/train/vit-large_chinasiwei/vitl_swdata_gram_anchor.yaml \
#     --output-dir work_dirs/vitl_gram_anchor

# 单机多卡，7b模型lora微调，预训练权重，bs=1,至少需要4卡,,4卡训练时在eval阶段报错显存不足，导致没有保存pth权重
export CUDA_VISIBLE_DEVICES=2,3,4,5,6
export NCCL_DEBUG=WARN
export MASTER_PORT=39600
export PYTHONPATH=${PWD}
export CUDA_LAUNCH_BLOCKING=1 
export TORCH_COMPILE_DISABLE=1
# 使用 torchrun 启动分布式训练
torchrun \
    --nproc_per_node=5 \
    --master_port=$MASTER_PORT \
    ./dinov3/train/train.py \
    --config-file ./dinov3/configs/train/finetune/dinov3_vit7b16_lora_finetune.yaml \
    --output-dir work_dirs/finetune/dinov3_vit7b16_lora_finetune

    