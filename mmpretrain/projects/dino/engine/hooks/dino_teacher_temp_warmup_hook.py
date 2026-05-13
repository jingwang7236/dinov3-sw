# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmengine.hooks import Hook

from mmpretrain.registry import HOOKS


@HOOKS.register_module()
class DINOTeacherTempWarmupHook(Hook):
    """Warmup teacher temperature for DINO.

    This hook warmups the temperature for teacher to stabilize the training
    process.

    Args:
        warmup_teacher_temp (float): Warmup temperature for teacher.
        teacher_temp (float): Temperature for teacher.
        teacher_temp_warmup_epochs (int): Warmup epochs for teacher
            temperature.
        max_epochs (int): Maximum epochs for training.
    """

    def __init__(self, warmup_teacher_temp: float, teacher_temp: float,
                 teacher_temp_warmup_epochs: int, max_epochs: int) -> None:
        super().__init__()
        self.teacher_temps = np.concatenate(
            (np.linspace(warmup_teacher_temp, teacher_temp,
                         teacher_temp_warmup_epochs),
             np.ones(max_epochs - teacher_temp_warmup_epochs) * teacher_temp))

    @staticmethod
    def _unwrap_model(model):
        """
        兼容单卡、DP、DDP
        """
        return getattr(model, 'module', model)

    def before_train_epoch(self, runner) -> None:
        # runner.model.module.head.teacher_temp = self.teacher_temps[
        #     runner.epoch]

        model = self._unwrap_model(runner.model)
        idx = min(runner.epoch, len(self.teacher_temps) - 1)
        temp = float(self.teacher_temps[idx])
        head = getattr(model, 'head', None)

        if head is None:
            raise AttributeError("Model has no attribute 'head' , cannot set `teacher_temp`")
        
        if not hasattr(head, 'teacher_temp'):
            raise AttributeError("head has no arrtibute `teacher_temp`, Please ensure you are using DINOHead or expose a `teacher_temp` attr.")

        head.teacher_temp = temp
        runner.logger.info(
            f"[DINOTeacherTempWarmupHook] epoch={runner.epoch}"
            f"set teacher_temp={temp:.5f}"
        )

