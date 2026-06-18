import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """计算 Focal Loss."""
        # 1. 创建有效像素掩码
        valid_mask = target != self.ignore_index
        
        # 2. 将 target 中的 ignore_index 临时替换为 0（避免 one_hot 越界）
        target_safe = target.clone()
        target_safe[~valid_mask] = 0  # 将忽略像素临时设为 0
        
        # 3. 对于二分类，直接使用 sigmoid（避免 one_hot）
        if pred.shape[1] == 1:
            pred_sigmoid = torch.sigmoid(pred)
            pred_prob = torch.where(target_safe == 1, pred_sigmoid, 1 - pred_sigmoid)
            loss = -self.alpha * (1 - pred_prob) ** self.gamma * torch.log(pred_prob.clamp(1e-8, 1 - 1e-8))
        else:
            # 多分类：使用 softmax + one_hot
            pred_softmax = F.softmax(pred, dim=1)
            # 使用安全的 target 进行 one_hot
            target_one_hot = F.one_hot(target_safe, num_classes=pred.shape[1]).permute(0, 3, 1, 2)
            pred_prob = (pred_softmax * target_one_hot).sum(dim=1)
            
            pt = pred_prob.clamp(1e-8, 1 - 1e-8)
            focal_weight = (1 - pt) ** self.gamma
            if self.alpha is not None:
                # 与官方 [alpha, 1-alpha] gather(target) 对齐：
                # 变化类(1)->0.75, 背景类(0)->0.25
                alpha_weight = torch.where(target_safe == 1, 1 - self.alpha, self.alpha)
                focal_weight = alpha_weight * focal_weight
            loss = -focal_weight * torch.log(pt)
        
        # 4. 将无效像素的 loss 设为 0
        loss = loss * valid_mask.float()
        
        # 5. 计算平均损失（仅对有效像素）
        if self.reduction == 'mean':
            return loss.sum() / (valid_mask.float().sum() + 1e-8)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DICELoss(nn.Module):
    def __init__(self, smooth=1e-5, reduction='mean', ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """计算 Dice Loss."""
        # 1. 创建有效像素掩码
        valid_mask = target != self.ignore_index
        
        # 2. 将 target 中的 ignore_index 临时替换为 0
        target_safe = target.clone()
        target_safe[~valid_mask] = 0
        
        # 3. 对于二分类
        if pred.shape[1] == 1:
            pred_sigmoid = torch.sigmoid(pred)
            pred_prob = pred_sigmoid
            target_float = target_safe.float()
        else:
            pred_softmax = F.softmax(pred, dim=1)
            target_one_hot = F.one_hot(target_safe, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
            pred_prob = (pred_softmax * target_one_hot).sum(dim=1)
            target_float = target_one_hot[:, 1] if pred.shape[1] > 1 else target_safe.float()
        
        # 4. 计算 Dice
        intersection = (pred_prob * target_float).sum(dim=(1, 2))
        union = pred_prob.sum(dim=(1, 2)) + target_float.sum(dim=(1, 2))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # 5. 只对有效像素计算
        dice = dice * valid_mask.float().mean(dim=(1, 2))
        
        if self.reduction == 'mean':
            return 1 - dice.mean()
        elif self.reduction == 'sum':
            return 1 - dice.sum()
        else:
            return 1 - dice