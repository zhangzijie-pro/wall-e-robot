import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.ops import roi_align
import clip

class CLIPExtractor(nn.Module):
    def __init__(self, clip_model, mode='image'):
        super().__init__()
        assert mode in ['image', 'text']
        self.clip_model = clip_model.eval()
        self.mode = mode
        self.image_preprocess = T.Compose([
            T.Resize((224, 224)),  # CLIP 默认输入尺寸
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711))
        ])

    def forward(self, inputs, boxes=None):
        """
        Args:
            inputs:
                - 如果 mode='image': tensor 图像 [B, 3, H, W]
                - 如果 mode='text': list of str
            boxes: anchor box [B, N, 4] in (x1,y1,x2,y2) format
                   仅 image 模式下使用
        Returns:
            features: [B, N, D] 或 [B, D]
        """
        if self.mode == 'text':
            # 文本指令编码
            with torch.no_grad():
                return self.clip_model.encode_text(
                    clip.tokenize(inputs).to(inputs.device)
                ).float()  # [B, D]

        elif self.mode == 'image':
            B, C, H, W = inputs.shape
            N = boxes.shape[1]
            boxes_reshape = boxes.view(-1, 4)  # [B*N, 4]

            # 为每个 box 添加 batch 索引
            batch_idx = torch.arange(B, device=inputs.device).view(B, 1).expand(-1, N).reshape(-1)
            roi_boxes = torch.cat([batch_idx[:, None].float(), boxes_reshape], dim=1)  # [B*N, 5]

            # 使用 roi_align 提取每个 box 的区域
            region_crops = roi_align(inputs, roi_boxes, output_size=(224, 224))  # [B*N, 3, 224, 224]
            region_crops = self.image_preprocess(region_crops)

            # 使用 CLIP 编码每个区域
            with torch.no_grad():
                region_features = self.clip_model.encode_image(region_crops)  # [B*N, D]
            return region_features.view(B, N, -1).float()  # [B, N, D]