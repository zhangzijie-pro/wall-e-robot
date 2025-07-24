import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Union

class AnchorGenerator(nn.Module):
    def __init__(self, 
                 image_size: Tuple[int, int] = (640, 640), 
                 grid_sizes: List[int] = [8, 16, 32], 
                 anchor_templates: Dict[int, List[Tuple[int, int]]] = None,
                 normalize: bool = False):
        """
        Args:
            image_size: 原始图像大小 (H, W)
            grid_sizes: 网格步长（对应不同特征图尺度）
            anchor_templates: 每个尺度对应的 anchor 模板尺寸 (w, h)
            normalize: 是否将 anchor 的中心坐标归一化到 [0, 1]
        """
        super().__init__()
        self.image_h, self.image_w = image_size
        self.grid_sizes = grid_sizes
        self.normalize = normalize

        if anchor_templates is None:
            self.anchor_templates = {
                8: [(10, 10), (15, 20)],
                16: [(30, 40), (40, 60)],
                32: [(60, 80), (80, 100)]
            }
        else:
            self.anchor_templates = anchor_templates

    def forward(self) -> torch.Tensor:
        """
        生成所有尺度下的 anchor。

        Returns:
            anchors: Tensor [N, 4]，格式为 (cx, cy, w, h)
        """
        all_anchors = []

        for stride in self.grid_sizes:
            grid_h = self.image_h // stride
            grid_w = self.image_w // stride
            anchor_sizes = self.anchor_templates.get(stride, [])

            # 生成中心点坐标
            y_coords = (torch.arange(grid_h) + 0.5) * stride
            x_coords = (torch.arange(grid_w) + 0.5) * stride
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")  # [H, W]

            # 展平为 [HW, 2]
            centers = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # [HW, 2]

            for aw, ah in anchor_sizes:
                # 每个模板复制对应所有网格点
                wh = torch.tensor([aw, ah], dtype=torch.float32).expand(centers.shape[0], 2)  # [HW, 2]
                anchor = torch.cat([centers, wh], dim=1)  # [HW, 4]
                all_anchors.append(anchor)

        anchors = torch.cat(all_anchors, dim=0)  # [N, 4]

        if self.normalize:
            anchors[:, 0] /= self.image_w  # cx
            anchors[:, 1] /= self.image_h  # cy
            anchors[:, 2] /= self.image_w  # w
            anchors[:, 3] /= self.image_h  # h

        return anchors  # [N, 4]