# 单机单卡，从头预训练
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 TORCH_COMPILE_DISABLE=1 PYTHONPATH=${PWD} \
python ./dinov3/train/train.py \
    --config-file ./dinov3/configs/train/vit-large_chinasiwei/vitl_swdata_pretrain_from_scracth.yaml \
    --output-dir work_dirs

# 单机多卡，从头预训练
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
    --output-dir work_dirs/multi_gpu