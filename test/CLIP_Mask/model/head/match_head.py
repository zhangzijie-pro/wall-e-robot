import torch
import torch.nn as nn
import torch.nn.functional as F

class MatchingHead(nn.Module):
    def __init__(
        self,
        in_channels=256,
        hidden_dim=512
    ):
        super(MatchingHead, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        self.projection = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=1
        )
        
    def forward(self, features, det_boxes, text_features):
        """
        输入FPN特征图、解码后的边界框和CLIP文本特征，输出匹配分数
        Args:
            features: List of feature maps [P3, P4, P5]
            det_boxes: Tensor, shape (batch_size, total_boxes, 5+num_classes)
            text_features: Tensor, shape (batch_size, num_texts, hidden_dim)
        Returns:
            scores: Tensor, shape (batch_size, total_boxes, num_texts)
        """
        batch_size = det_boxes.shape[0]
        total_boxes = det_boxes.shape[1]
        num_texts = text_features.shape[1]
        
        proj_features = []
        for feat in features:
            proj = self.projection(feat)
            proj_features.append(proj)
        
        feat = proj_features[0]
        stride = 2 ** 3  # P3
        boxes_scaled = det_boxes[:, :, :4] / stride
        box_features = self.extract_box_features(feat, boxes_scaled)
        
        box_features = F.normalize(box_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        scores = torch.bmm(box_features, text_features.transpose(1, 2))
        scores = torch.sigmoid(scores)
        
        return scores
    
    def extract_box_features(self, feature_map, boxes):
        """
        从特征图中提取边界框对应的特征
        Args:
            feature_map: Tensor, shape (batch_size, hidden_dim, H_i, W_i)
            boxes: Tensor, shape (batch_size, total_boxes, 4)
        Returns:
            box_features: Tensor, shape (batch_size, total_boxes, hidden_dim)
        """
        batch_size, _, h, w = feature_map.shape
        grid_x = boxes[..., 0].clamp(0, w - 1).long()
        grid_y = boxes[..., 1].clamp(0, h - 1).long()

        # Generate batch indices
        batch_idx = torch.arange(batch_size, device=feature_map.device).view(-1, 1).expand(-1, boxes.size(1))

        # Gather features using advanced indexing
        box_features = feature_map[batch_idx, :, grid_y, grid_x]  # (batch_size, total_boxes, hidden_dim)
        return box_features

if __name__ == "__main__":
    features = [
        torch.randn(1, 256, 56, 56),
        torch.randn(1, 256, 28, 28),
        torch.randn(1, 256, 14, 14)
    ]
    det_boxes = torch.randn(1, 10584, 6)
    text_features = torch.randn(1, 2, 512)
    
    matching_head = MatchingHead(in_channels=256, hidden_dim=512)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    matching_head = matching_head.to(device)
    features = [f.to(device) for f in features]
    det_boxes = det_boxes.to(device)
    text_features = text_features.to(device)
    
    scores = matching_head(features, det_boxes, text_features)
    
    print("Matching Scores Shape:", scores.shape)
    print("Matching Scores Sample:", scores[0, :5, :])
