import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# Binary Cross Entropy Loss
# --------------------------
def compute_mask_loss(pred_masks, gt_masks):
    """
    pred_masks: [B, N, H, W]  # 模型输出
    gt_masks:   [B, N, H, W]  # GT 掩膜
    """
    loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks)
    return loss

# --------------------------
# Dice Loss (可选)
# --------------------------
def dice_loss(pred_masks, gt_masks, eps=1e-6):
    pred_probs = torch.sigmoid(pred_masks)
    intersection = (pred_probs * gt_masks).sum(dim=(2, 3))
    union = pred_probs.sum(dim=(2, 3)) + gt_masks.sum(dim=(2, 3)) + eps
    loss = 1 - (2 * intersection / union)
    return loss.mean()

# --------------------------
# Box L1 Loss
# --------------------------
def compute_box_l1_loss(pred_boxes, gt_boxes):
    """
    pred_boxes: [B, N, 4]
    gt_boxes:   [B, N, 4]
    """
    loss = F.l1_loss(pred_boxes, gt_boxes)
    return loss

# --------------------------
# IoU Loss (GIoU optional)
# --------------------------
def compute_iou_loss(pred_boxes, gt_boxes):
    """
    计算 IoU 损失（使用 simplified 版本）
    """
    # 转为 [x1, y1, x2, y2]
    x1 = torch.max(pred_boxes[..., 0], gt_boxes[..., 0])
    y1 = torch.max(pred_boxes[..., 1], gt_boxes[..., 1])
    x2 = torch.min(pred_boxes[..., 2], gt_boxes[..., 2])
    y2 = torch.min(pred_boxes[..., 3], gt_boxes[..., 3])

    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area_pred = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    area_gt   = (gt_boxes[..., 2] - gt_boxes[..., 0]) * (gt_boxes[..., 3] - gt_boxes[..., 1])
    union = area_pred + area_gt - inter + 1e-6
    iou = inter / union
    loss = 1 - iou
    return loss.mean()

# --------------------------
# Total Loss Wrapper
# --------------------------
def compute_total_loss(pred_masks, gt_masks, pred_boxes, gt_boxes, weights=None):
    """
    聚合总损失
    weights: dict, e.g. {"mask": 1.0, "box": 1.0}
    """
    if weights is None:
        weights = {"mask": 1.0, "box": 1.0}

    mask_loss = compute_mask_loss(pred_masks, gt_masks)
    # 或使用 dice_loss(pred_masks, gt_masks)

    box_loss = compute_iou_loss(pred_boxes, gt_boxes)

    total_loss = weights["mask"] * mask_loss + weights["box"] * box_loss
    return total_loss, {"mask_loss": mask_loss.item(), "box_loss": box_loss.item()}