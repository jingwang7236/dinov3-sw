import logging
from typing import List, Dict, Any
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger("dinov3")

def apply_lora_to_student(model_meta_arch, cfg):
    """
    将 LoRA 应用于 SSLMetaArch 中的 student 网络。
    
    Args:
        model_meta_arch: SSLMetaArch 实例
        cfg: 配置对象
    """
    if not hasattr(cfg, 'peft') or not cfg.peft.enabled:
        logger.info("PEFT/LoRA is disabled in config.")
        return

    logger.info(f"Applying LoRA to student network with rank {cfg.peft.lora_rank}")
    
    # 获取 student 网络 (通常是 VisionTransformer)
    # 注意：在 SSLMetaArch 中，student 可能是一个字典或多个网络，需根据具体实现调整
    # 假设 model_meta_arch.student 是主要的 ViT 模型
    student_net = model_meta_arch.student
    
    # 如果 student 是 DistributedDataParallel 或 FSDP 包裹前的原始模型
    # 在 prepare_for_distributed_training 之前调用此函数
    
    lora_config = LoraConfig(
        r=cfg.peft.lora_rank,
        lora_alpha=cfg.peft.lora_alpha,
        target_modules=cfg.peft.target_modules,
        lora_dropout=cfg.peft.lora_dropout,
        bias=cfg.peft.bias,
        task_type=TaskType.FEATURE_EXTRACTION, # DINOv3 是特征提取任务
        modules_to_save=cfg.peft.modules_to_save
    )
    
    # 应用 LoRA
    # 注意：get_peft_model 会返回一个 PeftModel，它包裹了原始模型
    model_meta_arch.student = get_peft_model(student_net, lora_config)
    
    # 打印可训练参数
    model_meta_arch.student.print_trainable_parameters()
    
    # 重要：确保 Teacher 网络不被 LoRA 影响，且保持冻结
    # Teacher 通常是 EMA 更新的，不需要梯度，所以默认没问题。
    # 但需确保 optimizer 只获取 student 的参数

def get_lora_param_groups(model_meta_arch, base_lr, weight_decay):
    """
    为 LoRA 模型构建优化器参数组。
    PEFT 模型会自动设置 requires_grad，但我们需要正确分组以应用不同的 LR/WD。
    """
    # 获取所有需要梯度的参数
    params = []
    for name, param in model_meta_arch.student.named_parameters():
        if param.requires_grad:
            params.append(param)
            
    # 简单起见，将所有 LoRA 参数放在一个组里
    # 如果需要更精细的控制（如 bias 不同 WD），可在此扩展
    param_groups = [
        {
            "params": params,
            "lr": base_lr,
            "weight_decay": weight_decay,
            "is_last_layer": False, # LoRA 通常不视为 last layer，除非特定设计
            "lr_multiplier": 1.0,
            "wd_multiplier": 1.0
        }
    ]
    return param_groups