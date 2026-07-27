# Copyright (c) Open-CD. All rights reserved.
"""自包含的 EMA (Exponential Moving Average) Hook.

动机: BRIGHT 等灾害变化检测任务上, EMA 权重平滑可显著且**均匀**地提升所有
类别的 IoU (典型 +0.3~1.0 mIoU), 是性价比最高的 trick 之一。

实现为「参数影子」式 EMA, 完全不依赖 mmengine 的 BaseAveragedModel, 因此:
  * 不需要改动任何 detector / backbone / decode_head;
  * 对任意模型结构 (含自定义的 DualModeBranchEncoderDecoder) 均成立;
  * 旧配置不引入本 Hook, 行为完全不变 (完全向后兼容)。

工作流程:
  * before_run:        若无可训练参数则直接跳过 (如纯评估);
  * after_train_iter:  shadow = (1-momentum)*shadow + momentum*model (仅训练态参数);
  * before_val/val_iter/test: 备份 model 权重 -> 载入 shadow 权重;
  * after_val/val_iter/test:  恢复 model 原始权重 (训练继续用真实权重)。
"""
import logging

import torch
from mmengine.hooks import Hook

from opencd.registry import HOOKS


@HOOKS.register_module()
class EMAHook(Hook):
    """参数影子式 EMA Hook.

    Args:
        momentum (float): EMA 动量 (新权重占比). 经验上 1e-4~1e-3 较合适,
            数据量大/训练长用更小 (如 1e-4), 数据量小用更大 (如 2e-4).
        priority (str): Hook 优先级, 设为 'LOWEST' 以保证在 ``after_val_epoch``
            中, EMA 权重在 CheckpointHook (VERY_LOW) 保存 best 之后才换回,
            从而使 ``save_best`` 保存的是 EMA 权重。
        update_buffers (bool): 是否对非参数 buffer (如 BN 的 running_mean) 也
            做 EMA. 训练态 backbone 用 SyncBN 时建议 True, 以平滑统计量。
        valid_interval (int): 仅当 ``runner.iter % valid_interval == 0`` 时才
            在 val 前换入 EMA 权重, 避免每次小间隔 val 重复拷贝开销过大。
            默认 0 表示每次 val/test 都换入。
    """

    priority = 'LOWEST'

    def __init__(self,
                 momentum: float = 1e-4,
                 update_buffers: bool = True,
                 valid_interval: int = 0):
        super().__init__()
        self.momentum = momentum
        self.update_buffers = update_buffers
        self.valid_interval = valid_interval
        self._shadow = None          # dict[name -> tensor]
        self._backup = None          # dict[name -> tensor]
        self._ema_enabled = False
        self._swapped = False        # 当前是否已换入 EMA 权重(防重复换入)
        # 影子是否已被训练步填充。纯推理/评测(如 tools/test.py)时 before_run 会用
        # "刚构建、未训练"的模型初始化影子, 此时绝不能把这份随机影子换入覆盖
        # checkpoint, 否则 test 会跑在未训练权重上。只有发生至少一次
        # after_train_iter 更新后, 影子才被视为有效。
        self._shadow_populated = False
        self.logger = logging.getLogger('mmengine')

    # ------------------------------------------------------------------
    def _unwrap(self, model):
        while hasattr(model, 'module'):
            model = model.module
        return model

    def _named_tensors(self, model):
        """返回需要 EMA 的 (name, tensor) 列表 (参数 + 可选 buffer)."""
        items = list(model.named_parameters())
        if self.update_buffers:
            # 跳过非数值 / 形状特殊的 buffer (如 num_batches_tracked)
            for name, buf in model.named_buffers():
                if name.endswith('num_batches_tracked'):
                    continue
                if buf.is_floating_point():
                    items.append((name, buf))
        return items

    @torch.no_grad()
    def _init_shadow(self, model):
        self._shadow = {}
        for name, t in self._named_tensors(model):
            self._shadow[name] = t.detach().clone()
        self.logger.info(
            f'[EMAHook] Initialized EMA shadow for '
            f'{len(self._shadow)} tensors, momentum={self.momentum}, '
            f'update_buffers={self.update_buffers}.')

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def before_run(self, runner) -> None:
        model = self._unwrap(runner.model)
        # 只在有可训练参数时启用 (避免纯推理场景报错)
        has_params = any(p.requires_grad for p in model.parameters())
        if not has_params:
            self.logger.info('[EMAHook] No trainable params, EMA disabled.')
            self._ema_enabled = False
            return
        self._init_shadow(model)
        self._ema_enabled = True

    def after_train_iter(self, runner, batch_idx, data_batch=None,
                         outputs=None) -> None:
        if not self._ema_enabled:
            return
        model = self._unwrap(runner.model)
        m = self.momentum
        with torch.no_grad():
            for name, t in self._named_tensors(model):
                if name not in self._shadow:
                    self._shadow[name] = t.detach().clone()
                    continue
                self._shadow[name].mul_(1.0 - m).add_(t.detach(), alpha=m)
        # 至少经过一次训练更新后, 影子才被视为有效(可参与评测换入)
        self._shadow_populated = True

    # ---- 评测前: 换入 EMA 权重 ----
    def _swap_in(self, runner):
        if not self._ema_enabled:
            return
        # 影子尚未被训练步填充(如纯推理 tools/test.py / 纯 val 未训练)时,
        # 影子里只是 "刚构建的随机/预训练权重", 换入会覆盖已加载的 checkpoint,
        # 导致评测跑在未训练权重上。此时跳过换入, 直接用模型当前(checkpoint)权重。
        if not self._shadow_populated:
            self.logger.info(
                '[EMAHook] EMA shadow not populated yet (no training step); '
                'skip swap-in, evaluate with current model weights.')
            return
        if self._swapped:
            return  # 已换入, 避免重复覆盖 backup
        if self.valid_interval and (runner.iter % self.valid_interval != 0):
            return
        model = self._unwrap(runner.model)
        self._backup = {}
        with torch.no_grad():
            for name, t in self._named_tensors(model):
                self._backup[name] = t.detach().clone()
                if name in self._shadow:
                    t.copy_(self._shadow[name])
        self._swapped = True
        self.logger.info(f'[EMAHook] iter={runner.iter}: '
                         f'swapped EMA weights in for evaluation.')

    def _swap_out(self, runner):
        if not self._ema_enabled or not self._swapped:
            return
        model = self._unwrap(runner.model)
        with torch.no_grad():
            for name, t in self._named_tensors(model):
                if name in self._backup:
                    t.copy_(self._backup[name])
        self._backup = None
        self._swapped = False
        self.logger.info(f'[EMAHook] iter={runner.iter}: '
                         f'restored training weights after evaluation.')

    def before_val_epoch(self, runner) -> None:
        self._swap_in(runner)

    def after_val_epoch(self, runner, metrics) -> None:
        self._swap_out(runner)

    def before_val(self, runner) -> None:
        self._swap_in(runner)

    def after_val(self, runner) -> None:
        self._swap_out(runner)

    def before_test_epoch(self, runner) -> None:
        self._swap_in(runner)

    def after_test_epoch(self, runner, metrics) -> None:
        self._swap_out(runner)

    def before_test(self, runner) -> None:
        self._swap_in(runner)

    def after_test(self, runner) -> None:
        self._swap_out(runner)
