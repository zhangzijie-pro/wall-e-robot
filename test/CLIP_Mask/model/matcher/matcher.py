import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms


class RegionMatcher(nn.Module):
    def __init__(self, iou_threshold=0.5, top_k=100, use_positional_encoding=True):
        """
        Args:
            iou_threshold: NMS 阈值
            top_k: 返回前 K 个匹配框
            use_positional_encoding: 是否使用位置编码增强 patch 特征
        """
        super().__init__()
        self.iou_threshold = iou_threshold
        self.top_k = top_k
        self.use_positional_encoding = use_positional_encoding

        # 可学习的位置编码投影（2D 位置 → 特征维度）
        self.position_proj = None  # 初始化后再设置维度

    def add_positional_encoding(self, patch_features, anchors):
        """
        Args:
            patch_features: [N, D]
            anchors: [N, 4] in (cx, cy, w, h), 归一化 [0, 1]
        Returns:
            enhanced_features: [N, D]
        """
        if self.position_proj is None:
            # 初始化位置投影层
            self.position_proj = nn.Linear(2, patch_features.shape[1]).to(patch_features.device)

        cx, cy = anchors[:, 0], anchors[:, 1]
        pos = torch.stack([cx, cy], dim=-1)  # [N, 2]
        pos = (pos - 0.5) * 2  # normalize to [-1, 1]

        pos_embed = self.position_proj(pos)  # [N, D]
        return patch_features + pos_embed

    def forward(self, text_features: torch.Tensor, patch_features: torch.Tensor, anchors: torch.Tensor):
        """
        Args:
            text_features: [M, D] 文本特征（多个类别），已归一化
            patch_features: [N, D] 区域特征，已归一化
            anchors: [N, 4] anchor 区域 (cx, cy, w, h)，归一化坐标

        Returns:
            boxes: Tensor[K, 4]，选中 anchor 的 (x1, y1, x2, y2)
            scores: Tensor[K]，匹配分数
            labels: Tensor[K]，匹配的类别 index（text_features 的下标）
            indices: Tensor[K]，原始 patch 的索引
        """
        if self.use_positional_encoding:
            patch_features = self.add_positional_encoding(patch_features, anchors)

        # 计算余弦相似度： [N, M]
        sim_scores = patch_features @ text_features.T

        # 取每个 patch 最相关的 text
        max_scores, labels = sim_scores.max(dim=1)  # [N], [N]

        # 转换 anchor 坐标： (cx, cy, w, h) → (x1, y1, x2, y2)
        cx, cy, w, h = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [N, 4]

        # NMS 筛选
        selected = nms(boxes, max_scores, self.iou_threshold)
        selected = selected[:self.top_k]

        return boxes[selected], max_scores[selected], labels[selected], selected