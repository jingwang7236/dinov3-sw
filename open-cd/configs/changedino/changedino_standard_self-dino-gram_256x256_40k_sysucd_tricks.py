# ChangeDino + 自研 DINOv3-gram (SYSU-CD) — EMA 消融实验 (无 EMA)
# ============================================================================
# 消融目的: 验证 EMA 机制是否有效。
#   - 本配置: 关闭 EMA (custom_hooks 仅保留 FreezeScheduleHook)。
#   - 对照组: changedino_standard_self-dino-gram_256x256_40k_sysucd_tricks_ema.py
#             (其余完全一致, 仅多一个 EMAHook)。
#
# 其它设置 (模型 / 推理 trick / 数据增强 / 两阶段冻结 / 损失 / 学习率 / 训练规模)
# 全部继承自 _tricks 配置, 单变量控制, 仅 EMA 不同, 便于公平对比。
# ============================================================================
_base_ = [
    './changedino_standard_self-dino-gram_256x256_40k_sysucd_tricks_ema.py']

# 关闭 EMA: 仅保留两阶段冻结调度, 移除 EMAHook
custom_hooks = [
    dict(
        type='FreezeScheduleHook',
        backbone_attr='backbone',
        schedule=[
            # 阶段1: 前半段冻结 DINO 分支
            dict(iter=0, freeze_mode='frozen'),
            # 阶段2: 后半段解冻最后 4 层微调
            dict(iter=20000, freeze_mode='unfreeze_last_n',
                 unfreeze_last_n=4),
        ],
        verbose=True,
    ),
]
