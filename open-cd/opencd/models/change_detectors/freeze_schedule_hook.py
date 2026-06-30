# Copyright (c) Open-CD. All rights reserved.
"""训练过程中动态切换 backbone 冻结模式的 Hook。

典型用法: 两阶段训练
  - 阶段1 (0 ~ N iters): 冻结 backbone，仅训练 adapter / decoder
  - 阶段2 (N ~ end):    解冻 backbone（全量微调或解冻最后 N 层）

由于 PyTorch optimizer 在初始化时已包含所有参数（含 requires_grad=False
的），切换时无需修改 optimizer：冻结参数 grad=None 被自动跳过，解冻后
梯度正常计算、更新。
"""
import glob
import os

from mmengine.hooks import Hook

from opencd.registry import HOOKS


@HOOKS.register_module()
class FreezeScheduleHook(Hook):
    """按迭代数动态切换 DINOv3AdapterBackbone 的冻结模式。

    Args:
        schedule (list[dict]): 调度列表，按时间顺序排列，每个元素包含:
            - iter (int): 切换时刻（全局迭代数）。
            - freeze_mode (str): 'frozen' / 'full_finetune' / 'unfreeze_last_n'。
            - unfreeze_last_n (int, optional): mode='unfreeze_last_n' 时的层数。
            - load_from (str, optional): 切换前加载的 checkpoint 路径。
              支持 '{work_dir}' 占位符和 glob 模式（如 'best_mIoU_*.pth'）。
        backbone_attr (str): backbone 在模型中的属性路径，默认 'backbone'。
            对于 DualModeBranchEncoderDecoder 的光学侧也是 'backbone'。
        verbose (bool): 是否打印可训练参数统计。
    """

    priority = 'HIGH'

    def __init__(self, schedule, backbone_attr='backbone', verbose=True):
        # 确保 schedule 按 iter 排序
        self.schedule = sorted(schedule, key=lambda x: x['iter'])
        self.backbone_attr = backbone_attr
        self.verbose = verbose
        self._schedule_idx = 0

    def _get_backbone(self, runner):
        """从 runner.model 中提取 backbone 对象（自动解包 DDP wrapper）。"""
        model = runner.model
        # 解包 MMDataParallel / MMDistributedDataParallel
        if hasattr(model, 'module'):
            model = model.module
        obj = model
        for attr in self.backbone_attr.split('.'):
            obj = getattr(obj, attr)
        return obj

    def _load_checkpoint(self, runner, load_from):
        """加载 checkpoint（仅权重，不重置 optimizer / scheduler）。"""
        # 支持 {work_dir} 占位符
        path = load_from.replace('{work_dir}', runner.work_dir)
        # 支持 glob 模式（如 best_mIoU_*.pth）
        if '*' in path or '?' in path:
            matches = sorted(glob.glob(path))
            if not matches:
                runner.logger.warning(
                    f'[FreezeScheduleHook] No checkpoint found for '
                    f'pattern: {path}')
                return
            path = matches[-1]
        if not os.path.isfile(path):
            runner.logger.warning(
                f'[FreezeScheduleHook] Checkpoint not found: {path}')
            return
        runner.logger.info(f'[FreezeScheduleHook] Loading checkpoint: {path}')
        runner.load_checkpoint(path)

    def _count_trainable(self, backbone):
        """统计 backbone 中可训练参数数量。"""
        total = sum(p.numel() for p in backbone.parameters())
        trainable = sum(p.numel()
                        for p in backbone.parameters() if p.requires_grad)
        return trainable, total

    def _apply_schedule_entry(self, runner, entry):
        """应用单个调度条目。"""
        mode = entry['freeze_mode']
        n = entry.get('unfreeze_last_n', 0)

        # 可选: 切换前加载 checkpoint
        load_from = entry.get('load_from', None)
        if load_from:
            self._load_checkpoint(runner, load_from)

        # 切换冻结模式
        backbone = self._get_backbone(runner)
        backbone.set_freeze_mode(mode, n)

        runner.logger.info(
            f'[FreezeScheduleHook] iter={runner.iter}: '
            f'switched to freeze_mode=\'{mode}\''
            + (f', unfreeze_last_n={n}' if mode == 'unfreeze_last_n' else ''))

        if self.verbose:
            trainable, total = self._count_trainable(backbone)
            runner.logger.info(
                f'[FreezeScheduleHook]   backbone trainable params: '
                f'{trainable:,} / {total:,} ({100*trainable/total:.1f}%)')

    def before_run(self, runner):
        """训练开始前，应用 iter <= 当前 iter 的调度（如有）。"""
        while self._schedule_idx < len(self.schedule):
            entry = self.schedule[self._schedule_idx]
            if entry['iter'] <= runner.iter:
                self._apply_schedule_entry(runner, entry)
                self._schedule_idx += 1
            else:
                break

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        """每次迭代前检查是否需要切换冻结模式。"""
        while self._schedule_idx < len(self.schedule):
            entry = self.schedule[self._schedule_idx]
            if runner.iter >= entry['iter']:
                self._apply_schedule_entry(runner, entry)
                self._schedule_idx += 1
            else:
                break
