"""Lovász-Softmax loss for semantic segmentation.

Reference: Berman et al., "Optimizing Intersection-Over-Union in Deep
Neural Networks for Image Segmentation", CVPR 2016.

该 loss 直接优化 IoU/Jaccard 指标的上确界紧凸松弛 (Lovász extension)，
对类别不平衡的变化检测任务尤其有效，常与 Focal/Dice 联合使用。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovász extension w.r.t sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _flatten_probas(probas, labels, ignore=None):
    """[B,C,H,W] probas + [B,H,W] labels -> [N,C] + [N], 排除 ignore 像素。"""
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    return probas[valid], labels[valid]


def _lovasz_softmax_flat(probas, labels, classes='present'):
    """Multi-class Lovász-Softmax loss on flattened tensors."""
    if probas.numel() == 0:
        return probas.new_zeros(1).squeeze()
    C = probas.size(1)
    class_to_sum = (list(range(C)) if classes == 'all'
                    else torch.unique(labels).tolist())
    losses = []
    for c in class_to_sum:
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    if len(losses) == 0:
        return probas.new_zeros(1).squeeze()
    return torch.stack(losses).mean()


class LovaszSoftmaxLoss(nn.Module):
    """Lovász-Softmax loss.

    Args:
        ignore_index (int): 忽略的标签值, 默认 255。
        classes (str): 'present' (仅对出现的类别求平均) 或 'all'。
        per_image (bool): True 则逐图计算后取平均, False 则在 batch 整体计算。
        reduction (str): 'mean' 或 'sum'。
    """

    def __init__(self, ignore_index=255, classes='present',
                 per_image=False, reduction='mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.classes = classes
        self.per_image = per_image
        self.reduction = reduction

    def forward(self, logits, target):
        """
        Args:
            logits (Tensor): [N, C, H, W] raw logits。
            target (Tensor): [N, H, W] integer labels。
        """
        probas = F.softmax(logits, dim=1)
        if self.per_image:
            total = 0.0
            for p, t in zip(probas, target):
                total += _lovasz_softmax_flat(
                    *_flatten_probas(p.unsqueeze(0), t.unsqueeze(0),
                                     self.ignore_index),
                    classes=self.classes)
            loss = total / probas.size(0)
        else:
            loss = _lovasz_softmax_flat(
                *_flatten_probas(probas, target, self.ignore_index),
                classes=self.classes)
        if self.reduction == 'sum':
            return loss * probas.size(0)
        return loss
