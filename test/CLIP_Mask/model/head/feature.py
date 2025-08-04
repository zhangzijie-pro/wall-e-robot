import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoder(nn.Module):
    def __init__(self, input_dim=4, embed_dim=512, mode='add'):
        """
        为边界框添加位置编码
        Args:
            input_dim: anchor输入维度 (cx, cy, w, h)，默认4
            embed_dim: 特征嵌入维度，默认512（匹配CLIP文本维度）
            mode: 融合方式，'add'（加性）或'concat'（拼接后投影）
        """
        super().__init__()
        self.mode = mode
        self.embed_dim = embed_dim

        if mode == 'add':
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.ReLU(),
                nn.LayerNorm(embed_dim)
            )
        elif mode == 'concat':
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, embed_dim // 2),
                nn.ReLU(),
                nn.LayerNorm(embed_dim // 2)
            )
            self.projector = nn.Linear(embed_dim + embed_dim // 2, embed_dim)
        else:
            raise ValueError("mode must be 'add' or 'concat'")

    def forward(self, patch_embeddings: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_embeddings: Tensor [batch_size, num_anchors, embed_dim]
            anchors: Tensor [batch_size, num_anchors, 4] (cx, cy, w, h)，归一化坐标[0,1]
        Returns:
            enhanced_embeddings: Tensor [batch_size, num_anchors, embed_dim]
        """
        pos_feat = self.encoder(anchors)  # [batch_size, num_anchors, embed_dim] 或 [batch_size, num_anchors, embed_dim//2]

        if self.mode == 'add':
            return patch_embeddings + pos_feat
        else:  # concat mode
            fused = torch.cat([patch_embeddings, pos_feat], dim=-1)
            return self.projector(fused)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        """
        多头自注意力机制，捕捉特征的全局上下文
        Args:
            in_channels: 输入特征通道数
            num_heads: 注意力头数
        """
        super(MultiHeadSelfAttention, self).__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x):
        """
        Args:
            x: Tensor [batch_size, in_channels, height, width]
        Returns:
            out: Tensor [batch_size, in_channels, height, width]
        """
        batch_size, C, H, W = x.shape
        
        Q = self.query(x).view(batch_size, self.num_heads, self.head_dim, H * W)
        K = self.key(x).view(batch_size, self.num_heads, self.head_dim, H * W)
        V = self.value(x).view(batch_size, self.num_heads, self.head_dim, H * W)
        
        Q = Q.transpose(-2, -1)  # [B, num_heads, H*W, head_dim]
        scores = torch.matmul(Q, K) * self.scale  # [B, num_heads, H*W, H*W]
        attn = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn, V.transpose(-2, -1))  # [B, num_heads, H*W, head_dim]
        context = context.transpose(-2, -1).contiguous().view(batch_size, C, H, W)
        
        return self.out(context)

class FeatureEnhancer(nn.Module):
    def __init__(
        self,
        in_channels=256,
        embed_dim=512,
        num_heads=8,
        feature_sizes=[(80, 80), (40, 40), (20, 20)],
        mode='add'
    ):
        """
        特征增强模块，结合位置编码和自注意力
        Args:
            in_channels: FPN输出通道数
            embed_dim: 位置编码嵌入维度（匹配CLIP文本维度）
            num_heads: 注意力头数
            feature_sizes: FPN特征图尺寸
            mode: 位置编码融合模式（'add'或'concat'）
        """
        super(FeatureEnhancer, self).__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.feature_sizes = feature_sizes
        
        # 投影特征图到embed_dim
        self.projection = nn.ModuleList([
            nn.Conv2d(in_channels, embed_dim, kernel_size=1) for _ in feature_sizes
        ])
        
        # 位置编码器
        self.pos_encoders = nn.ModuleList([
            PositionalEncoder(input_dim=4, embed_dim=embed_dim, mode=mode) for _ in feature_sizes
        ])
        
        # 自注意力模块
        self.attentions = nn.ModuleList([
            MultiHeadSelfAttention(embed_dim, num_heads) for _ in feature_sizes
        ])
        
        # 层归一化和投影回in_channels
        # self.norm = nn.ModuleList([
        #     nn.LayerNorm([embed_dim, h, w]) for h, w in feature_sizes
        # ])
        self.norm = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in feature_sizes
        ])
        self.out_projection = nn.ModuleList([
            nn.Conv2d(embed_dim, in_channels, kernel_size=1) for _ in feature_sizes
        ])
    
    def extract_box_features(self, feature_map, anchors, stride):
        """
        从特征图中提取边界框对应的特征
        Args:
            feature_map: Tensor [batch_size, embed_dim, H_i, W_i]
            anchors: Tensor [batch_size, num_anchors, H_i, W_i, 5+num_classes]
            stride: 特征图缩放比例
        Returns:
            box_features: Tensor [batch_size, num_anchors*H_i*W_i, embed_dim]
        """
        batch_size, embed_dim, H, W = feature_map.shape
        anchors = anchors.reshape(batch_size, -1, anchors.shape[-1])
        box_coords = anchors[:, :, :4] / stride

        grid_x = box_coords[..., 0].clamp(0, W - 1).long()
        grid_y = box_coords[..., 1].clamp(0, H - 1).long()

        feature_map = feature_map.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        box_features = []
        for b in range(batch_size):
            coords = grid_y[b] * W + grid_x[b]  # [N]
            feat = feature_map[b].view(-1, embed_dim)[coords]  # [N, C]
            box_features.append(feat)

        return torch.stack(box_features, dim=0)  # [B, N, C]

    def forward(self, features, det_predictions):
        """
        输入FPN特征图和检测头预测，输出增强后的特征图
        Args:
            features: List of feature maps [P3, P4, P5], shape [batch_size, in_channels, H_i, W_i]
            det_predictions: List of detection predictions, shape [batch_size, num_anchors, H_i, W_i, 5+num_classes]
        Returns:
            enhanced_features: List of enhanced feature maps, shape [batch_size, in_channels, H_i, W_i]
        """
        enhanced_features = []
        # strides = [8, 16, 32]  # P3, P4, P5的缩放比例
        strides = [8 * (2 ** i) for i in range(len(features))]  # 如 [8,16,32,64,...]
        
        for i, (feat, pred) in enumerate(zip(features, det_predictions)):
            # 投影到embed_dim
            feat_proj = self.projection[i](feat)  # [batch_size, embed_dim, H_i, W_i]
            
            # 提取边界框特征
            box_features = self.extract_box_features(feat_proj, pred, strides[i])  # [batch_size, num_anchors*H_i*W_i, embed_dim]
            
            # 添加位置编码
            anchors = pred.reshape(pred.shape[0], -1, pred.shape[-1])[:, :, :4]  # [batch_size, num_anchors*H_i*W_i, 4]
            anchors = anchors / self.feature_sizes[i][0]  # 归一化到[0,1]
            enhanced_box_features = self.pos_encoders[i](box_features, anchors)  # [batch_size, num_anchors*H_i*W_i, embed_dim]
            
            # 重塑回特征图
            feat_enhanced = feat_proj.clone()
            batch_size, _, H, W = feat_proj.shape
            for b in range(batch_size):
                grid_x = (anchors[b, :, 0] * W).long().clamp(0, W-1)
                grid_y = (anchors[b, :, 1] * H).long().clamp(0, H-1)
                feat_enhanced[b, :, grid_y, grid_x] = enhanced_box_features[b].permute(1, 0)
            
            # 自注意力增强
            feat_attn = self.attentions[i](feat_enhanced)
            
            # 残差连接和归一化
            # feat = self.norm[i](feat_enhanced + feat_attn)
            feat = self.norm[i]((feat_enhanced + feat_attn).permute(0, 2, 3, 1))  # [B, H, W, C]
            feat = feat.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
            
            # 投影回in_channels
            feat = self.out_projection[i](feat)
            enhanced_features.append(feat)
        
        return enhanced_features

# 测试代码
if __name__ == "__main__":
    # 模拟FPN输出和检测头预测
    features = [
        torch.randn(1, 256, 80, 80),  # P3
        torch.randn(1, 256, 40, 40),  # P4
        torch.randn(1, 256, 20, 20),   # P5
        torch.randn(1, 256, 10, 10)   # P5
    ]
    det_predictions = [
        torch.randn(1, 3, 80, 80, 6),  # P3
        torch.randn(1, 3, 40, 40, 6),  # P4
        torch.randn(1, 3, 20, 20, 6),   # P5
        torch.randn(1, 3, 10, 10, 6)   # P5
    ]
    
    # 初始化FeatureEnhancer
    enhancer = FeatureEnhancer(
        in_channels=256,
        embed_dim=512,
        num_heads=8,
        feature_sizes=[(80, 80), (40, 40), (20, 20), (10, 10)],
        mode='concat'
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enhancer = enhancer.to(device)
    features = [f.to(device) for f in features]
    det_predictions = [p.to(device) for p in det_predictions]
    
    # 前向传播
    enhanced_features = enhancer(features, det_predictions)
    
    # 打印输出
    print("Enhanced Features Shapes:")
    for i, feat in enumerate(enhanced_features):
        print(f"P{i+3} Shape: {feat.shape}")