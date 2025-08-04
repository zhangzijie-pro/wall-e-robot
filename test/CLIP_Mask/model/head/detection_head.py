import torch
import torch.nn as nn
from torchvision.ops import nms
import torch.nn.functional as F

def generate_dynamic_anchors(base_sizes, num_layers, anchors_per_layer=3, max_layers=5):
    """
    生成每层的 anchor 尺寸（宽, 高）
    Args:
        base_sizes: list，例如 [(10,13), (30,61), (116,90)]
        num_layers: 实际的 FPN 层数（例如 3）
        anchors_per_layer: 每层 anchor 数
        max_layers: 最大支持几层
    Returns:
        anchors: List[List[Tuple]]，每层一个 anchor list
    """
    num_layers = min(num_layers, max_layers)
    anchors = []
    for i in range(num_layers):
        scale = 2 ** i
        layer_anchors = []
        for base_w, base_h in base_sizes:
            anchor = (int(base_w * scale), int(base_h * scale))
            layer_anchors.append(anchor)
        anchors.append(layer_anchors[:anchors_per_layer])
    return anchors

class DetectionHead(nn.Module):
    def __init__(
        self,
        in_channels=256,  # FPN/BiFPN输出通道数
        num_fpn_layers=3,
        num_anchors=3,    # 每个网格的anchor数量
        num_classes=1     # 动态类别数（占位，实际由CLIP文本特征决定）
    ):
        super(DetectionHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.out_channels = num_anchors * (5 + num_classes)  # 4坐标 + 1置信度 + 类别分数
        
        # 卷积层预测边界框、置信度和类别
        self.conv = nn.Conv2d(
            in_channels,
            self.out_channels,
            kernel_size=1,  # 1x1卷积，保持空间分辨率
            stride=1
        )
        
        base_anchor_sizes = [(10, 13), (30, 61), (62, 45)]
        self.anchors = generate_dynamic_anchors(
            base_anchor_sizes,
            num_layers=num_fpn_layers,
            anchors_per_layer=num_anchors
        )
        # 预定义anchor尺寸（宽、高），按尺度调整
        # self.anchors = [
        #     [(10, 13), (16, 30), (33, 23)],  # P3 (高分辨率，小目标)
        #     [(30, 61), (62, 45), (59, 119)], # P4 (中分辨率)
        #     [(116, 90), (156, 198), (373, 326)] # P5 (低分辨率，大目标)
        # ]
        
    def forward(self, features):
        """
        输入多尺度特征图，输出边界框预测
        Args:
            features: List of feature maps [P3, P4, P5], each of shape (batch_size, in_channels, H_i, W_i)
        Returns:
            predictions: List of predictions, each of shape (batch_size, num_anchors, H_i, W_i, 5+num_classes)
        """
        predictions = []
        
        for i, feat in enumerate(features):
            # 卷积预测
            pred = self.conv(feat)  # (batch_size, num_anchors * (5+num_classes), H_i, W_i)
            
            # 重塑输出
            batch_size, _, h, w = pred.shape
            pred = pred.view(batch_size, self.num_anchors, 5 + self.num_classes, h, w)
            pred = pred.permute(0, 1, 3, 4, 2)  # (batch_size, num_anchors, H_i, W_i, 5+num_classes)
            
            # 应用sigmoid激活
            pred[..., :4] = torch.sigmoid(pred[..., :4])  # 坐标 (x, y, w, h)
            pred[..., 4] = torch.sigmoid(pred[..., 4])    # 置信度
            predictions.append(pred)
        
        return predictions
    
    def decode_boxes(self, predictions, strides=None):
        """
        解码预测边界框，转换为图像坐标（支持动态FPN层数）
        Args:
            predictions: List of shape (B, num_anchors, H_i, W_i, 5+C)
            strides: List[int], 每层特征图对应的下采样倍数（如[8, 16, 32, ...]）
        Returns:
            boxes: Tensor of shape (B, total_boxes, 5+C)
        """
        device = predictions[0].device
        num_levels = len(predictions)

        # 自动设置 strides（如果未传入）
        if strides is None:
            strides = [8 * (2 ** i) for i in range(num_levels)]  # 如 [8,16,32,64,...]

        print("Strides:", len(strides))
        print("Number of FPN Levels:", num_levels)
        print("Number of Anchors per Level:", len(self.anchors))
        
        assert len(strides) == num_levels == len(self.anchors), \
            "strides、FPN层数、anchor层数不匹配"

        boxes_list = []

        for i, pred in enumerate(predictions):
            B, A, H, W, C = pred.shape
            stride = strides[i]
            anchors = torch.tensor(self.anchors[i], device=device).float()  # shape: [A, 2]
            anchors = anchors.view(1, A, 1, 1, 2).expand(B, A, H, W, 2)  # broadcast

            # 构建grid
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing="ij"
            )
            grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(B, A, H, W)
            grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(B, A, H, W)

            # 解码坐标
            decoded = pred.clone()
            decoded[..., 0] = (decoded[..., 0] + grid_x) * stride  # cx
            decoded[..., 1] = (decoded[..., 1] + grid_y) * stride  # cy
            decoded[..., 2] = decoded[..., 2] * anchors[..., 0]    # w
            decoded[..., 3] = decoded[..., 3] * anchors[..., 1]    # h

            # reshape 到统一格式
            decoded = decoded.reshape(B, -1, C)  # (B, A*H*W, 5+C)
            boxes_list.append(decoded)

        return torch.cat(boxes_list, dim=1)  # (B, total_boxes, 5+C)
    
    def nms(self, boxes, iou_threshold=0.5, conf_threshold=0.4):
        """
        非极大值抑制，过滤重叠边界框
        Args:
            boxes: Tensor of shape (batch_size, total_boxes, 5+num_classes)
            iou_threshold: IoU阈值
            conf_threshold: 置信度阈值
        Returns:
            keep_boxes: List of filtered boxes for each batch
        """
        keep_boxes = []
        for i in range(boxes.shape[0]):
            box = boxes[i]  # (total_boxes, 5+num_classes)
            scores = box[:, 4]  # 置信度
            valid = scores > conf_threshold
            box = box[valid]
            
            if box.shape[0] == 0:
                keep_boxes.append(box)
                continue
            
            # 转换为(x1, y1, x2, y2)格式
            x1 = box[:, 0] - box[:, 2] / 2
            y1 = box[:, 1] - box[:, 3] / 2
            x2 = box[:, 0] + box[:, 2] / 2
            y2 = box[:, 1] + box[:, 3] / 2
            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
            
            # NMS
            keep_idx = nms(boxes_xyxy, box[:, 4], iou_threshold)
            keep_boxes.append(box[keep_idx])
        
        return keep_boxes

# 测试代码
if __name__ == "__main__":
    # 模拟FPN输出
    features = [
        torch.randn(1, 256, 56, 56),  # P3
        torch.randn(1, 256, 28, 28),  # P4
        torch.randn(1, 256, 14, 14),   # P5
        torch.randn(1, 256, 7,7)   # P6
    ]
    
    # 初始化检测头
    det_head = DetectionHead(in_channels=256, num_fpn_layers=len(features),num_anchors=3, num_classes=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    det_head = det_head.to(device)
    features = [f.to(device) for f in features]
    
    # 前向传播
    predictions = det_head(features)
    
    # 解码边界框
    boxes = det_head.decode_boxes(predictions)
    
    # 应用NMS
    nms_boxes = det_head.nms(boxes)
    
    # 打印输出
    print("Predictions Shapes:")
    for i, pred in enumerate(predictions):
        print(f"P{i+3} Shape: {pred.shape}")
    print("\nDecoded Boxes Shape:", boxes.shape)
    print("NMS Boxes (first batch):", nms_boxes[0].shape if nms_boxes[0].shape[0] > 0 else "Empty")