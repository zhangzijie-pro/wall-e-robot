import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class SegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels=256,  # FPN/BiFPN输出通道数
        proto_channels=32,  # 掩码原型通道数
        mask_size=28       # 输出掩码分辨率（例如，28x28）
    ):
        super(SegmentationHead, self).__init__()
        self.in_channels = in_channels
        self.proto_channels = proto_channels
        self.mask_size = mask_size
        
        # ProtoNet：生成掩码原型
        self.proto_conv = nn.Sequential(
            nn.Conv2d(in_channels, proto_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(proto_channels, proto_channels, kernel_size=1)
        )
        
        # 掩码系数预测（与检测头边界框对应）
        self.coef_conv = nn.Conv2d(
            in_channels,
            proto_channels,  # 系数数量与原型通道数一致
            kernel_size=1
        )
        
    def forward(self, features, det_predictions):
        """
        输入多尺度特征图和检测头预测，输出分割掩码
        Args:
            features: List of feature maps [P3, P4, P5], each of shape (batch_size, in_channels, H_i, W_i)
            det_predictions: List of detection predictions, each of shape (batch_size, num_anchors, H_i, W_i, 5+num_classes)
        Returns:
            masks: List of masks for each detection, shape (batch_size, num_detections, mask_size, mask_size)
        """
        # 从最高分辨率特征图（P3）生成掩码原型
        proto = self.proto_conv(features[0])  # (batch_size, proto_channels, H_3, W_3)
        proto = F.interpolate(proto, size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False)
        
        masks = []
        for i, pred in enumerate(det_predictions):
            batch_size, num_anchors, h, w, _ = pred.shape
            # 预测掩码系数
            coef = self.coef_conv(features[i])  # (batch_size, proto_channels, H_i, W_i)
            coef = coef.view(batch_size, self.proto_channels, h, w)
            coef = coef.unsqueeze(1).expand(-1, num_anchors, -1, -1, -1)
            coef = coef.permute(0, 1, 3, 4, 2).contiguous()  # (batch_size, num_anchors, H_i, W_i, proto_channels)
            
            # 计算掩码
            coef_flat = coef.view(batch_size, -1, self.proto_channels)  # (B, N, C)
            proto_flat = proto.view(batch_size, self.proto_channels, -1)  # (B, C, 784)
            mask = torch.bmm(coef_flat, proto_flat)  # (B, N, 784)
            mask = mask.view(batch_size, -1, self.mask_size, self.mask_size)  # (B, N, 28, 28)
            mask = torch.sigmoid(mask)  # 确保掩码值在[0, 1]
            masks.append(mask)
        
        # 合并所有尺度的掩码
        masks = torch.cat(masks, dim=1)  # (batch_size, total_detections, mask_size, mask_size)
        return masks
    
    def align_with_boxes(self, masks, boxes):
        """
        将掩码与解码后的边界框对齐
        Args:
            masks: Tensor of shape (batch_size, total_detections, mask_size, mask_size)
            boxes: Tensor of shape (batch_size, total_detections, 5+num_classes)
        Returns:
            aligned_masks: List of aligned masks for each batch
        """
        aligned_masks = []
        for i in range(masks.shape[0]):
            mask = masks[i]  # (total_detections, mask_size, mask_size)
            box = boxes[i]   # (total_detections, 5+num_classes)
            valid = box[:, 4] > 0  # 使用置信度过滤
            aligned_masks.append(mask[valid])
        return aligned_masks

# 测试代码
if __name__ == "__main__":
    # 模拟FPN输出和检测头预测
    random.seed(42)
    features = [
        torch.randn(1, 256, 56, 56),  # P3
        torch.randn(1, 256, 28, 28),  # P4
        torch.randn(1, 256, 14, 14)   # P5
    ]
    det_predictions = [
        torch.randn(1, 3, 56, 56, 6),  # P3
        torch.randn(1, 3, 28, 28, 6),  # P4
        torch.randn(1, 3, 14, 14, 6)   # P5
    ]
    
    # 初始化分割头
    seg_head = SegmentationHead(in_channels=256, proto_channels=32, mask_size=28)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_head = seg_head.to(device)
    features = [f.to(device) for f in features]
    det_predictions = [p.to(device) for p in det_predictions]
    
    # 前向传播
    masks = seg_head(features, det_predictions)
    
    # 模拟解码后的边界框
    from detection_head import DetectionHead
    det_head = DetectionHead(in_channels=256, num_anchors=3, num_classes=1).to(device)
    boxes = det_head.decode_boxes(det_predictions)
    
    # 对齐掩码
    aligned_masks = seg_head.align_with_boxes(masks, boxes)
    
    # 打印输出
    print("Masks Shape:", masks.shape)
    print("Aligned Masks (first batch):", aligned_masks[0].shape if aligned_masks[0].shape[0] > 0 else "Empty")