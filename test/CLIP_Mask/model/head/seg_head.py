import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_prototypes=32, mask_size=160):
        """
        Args:
            in_channels: 输入特征图的通道数
            num_prototypes: 原型掩膜数量
            mask_size: 输出掩膜的尺寸（方形）
        """
        super().__init__()
        self.mask_size = mask_size
        self.num_prototypes = num_prototypes

        # 原型掩膜生成网络（对整张特征图）
        self.proto_net = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_prototypes, kernel_size=1)
        )

        # 掩膜系数预测（基于 ROI 特征）
        self.mask_coef_proj = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, num_prototypes)
        )

    def forward(self, feature_map: torch.Tensor, boxes: torch.Tensor):
        """
        Args:
            feature_map: [B, C, H, W]，特征图
            boxes: Tensor[K, 5]，候选框 (batch_idx, x1, y1, x2, y2)，坐标为原图尺度映射到 feature map 的尺度

        Returns:
            mask_prototypes: [B, P, H, W]，原型掩膜
            mask_coefs: [K, P]，每个候选框对应的掩膜权重系数
        """
        B, C, H, W = feature_map.shape

        # 构建原型掩膜：对整张特征图卷积
        mask_prototypes = self.proto_net(feature_map)  # [B, P, H, W]

        # ROI Align 提取每个 box 的区域特征
        # boxes: [K, 5]，每个为 [batch_idx, x1, y1, x2, y2]，坐标需与 feature_map 对齐
        roi_features = roi_align(
            input=feature_map,
            boxes=boxes,
            output_size=(1, 1),
            spatial_scale=1.0,  # 若输入 boxes 已是 feature map 尺度
            aligned=True
        ).squeeze(-1).squeeze(-1)  # [K, C]

        # 掩膜系数预测
        mask_coefs = self.mask_coef_proj(roi_features)  # [K, P]

        return mask_prototypes, mask_coefs


def build_masks(mask_prototypes: torch.Tensor, mask_coefs: torch.Tensor, boxes: torch.Tensor = None):
    """
    Args:
        mask_prototypes: Tensor[B, P, H, W]
        mask_coefs: Tensor[K, P]
        boxes (optional): 如果需要将掩膜投影回图像空间，可以接入坐标信息

    Returns:
        masks: Tensor[K, H, W]，每个候选区域的掩膜
    """
    # 展平原型掩膜到 [P, H, W]
    if mask_prototypes.dim() == 4:
        # 若 batch size == 1，可直接取第一个
        mask_prototypes = mask_prototypes[0]  # [P, H, W]

    masks = torch.einsum("kp,phw->khw", mask_coefs, mask_prototypes)  # [K, H, W]
    masks = torch.sigmoid(masks)

    return masks