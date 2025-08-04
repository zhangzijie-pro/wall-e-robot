import torch
import torch.nn as nn
import torch.nn.functional as F
# --------------------------- BiFPN Implementation ---------------------------
class BiFPN(nn.Module):
    def __init__(
        self,
        in_channels=768,
        out_channels=256,
        num_levels=3,
        input_size=(14, 14)
    ):
        super(BiFPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.input_size = input_size

        # 横向连接
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in range(num_levels)
        ])
        # 下采样模块
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        # 上采样模块
        self.upsample = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        # 输出卷积
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in range(num_levels)
        ])
        # bottom-up融合卷积
        self.bottomup_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in range(num_levels)
        ])
        # 计算各层目标尺寸（从低到高分辨率）
        self.target_sizes = [(input_size[0] // (2 ** i), input_size[1] // (2 ** i)) for i in range(num_levels)][::-1]

    def forward(self, inputs):
        """
        输入: (batch_size, in_channels, H, W)
        输出: List of feature maps, each shape (batch_size, out_channels, H_i, W_i)
        """
        # 动态生成多尺度输入
        features = [inputs]
        for _ in range(1, self.num_levels):
            features.append(self.downsample(features[-1]))
        features = features[::-1]  # 从低到高分辨率

        # 横向连接
        laterals = [self.lateral_convs[i](features[i]) for i in range(self.num_levels)]

        # --------- Top-Down Path ---------
        td = [laterals[-1]]  # 最低分辨率开始
        for i in range(self.num_levels-2, -1, -1):
            up = self.upsample(td[0], size=laterals[i].shape[2:])
            fused = laterals[i] + up
            td.insert(0, self.output_convs[i](fused))
        # td: [P3, P4, P5, ...] (高到低分辨率)

        # --------- Bottom-Up Path ---------
        bu = [td[0]]
        for i in range(1, self.num_levels):
            down = self.downsample(bu[-1])
            down_resized = F.interpolate(down, size=td[i].shape[2:], mode='nearest')
            fused = td[i] + down_resized
            bu.append(self.bottomup_convs[i](fused))
        # bu: [P3, P4, P5, ...] (高到低分辨率)
        return bu


class FPN(nn.Module):
    def __init__(
        self,
        in_channels=768,  # CLIP视觉编码器输出通道数 (ViT-B/32)
        out_channels=256,  # FPN输出通道数，统一为256以适配YOLO
        num_levels=3,      # 输出尺度数（例如，P3, P4, P5）
        input_size=(14, 14)  # 输入特征图的尺寸 (H', W')
    ):
        super(FPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.input_size = input_size
        
        # 横向连接：将CLIP特征图转换为统一通道数
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in range(num_levels)
        ])
        
        # 自顶向下路径：上采样融合
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in range(num_levels)
        ])
        
        # 下采样模块
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 计算各层目标尺寸（从低到高分辨率）
        self.target_sizes = [(input_size[0] // (2 ** i), input_size[1] // (2 ** i)) for i in range(num_levels)][::-1]
    
    def forward(self, inputs):
        """
        输入CLIP视觉编码器的特征图，输出多尺度特征图
        Args:
            inputs: Tensor of shape (batch_size, in_channels, H', W') (e.g., [batch_size, 768, 14, 14])
        Returns:
            outputs: List of feature maps [P3, P4, P5], each of shape (batch_size, out_channels, H_i, W_i)
        """
        # 动态生成多尺度输入
        features = [inputs]
        for _ in range(1, self.num_levels):
            features.append(self.downsample(features[-1]))
        features = features[::-1]  # 从低到高分辨率
        
        # 横向连接：统一通道数
        laterals = [self.lateral_convs[i](features[i]) for i in range(self.num_levels)]
        
        # 自顶向下路径：上采样融合
        outputs = [laterals[-1]]  # P5 = C5的横向连接结果
        for i in range(self.num_levels-2, -1, -1):
            # 上采样到目标尺寸
            target_size = self.target_sizes[i+1]  # P4对应C4, P3对应C3
            upsampled = F.interpolate(
                outputs[0],
                size=target_size,
                mode='nearest'
            )
            # 融合当前层的横向连接
            fused = laterals[i] + F.interpolate(outputs[0], size=laterals[i].shape[2:], mode='nearest')
            outputs.insert(0, self.output_convs[i](fused))
        
        return outputs

if __name__ == "__main__":
    input_features = torch.randn(1, 768, 28, 28)  # (batch_size, in_channels, H', W')
    
    I, O, size = input_features.shape[1], 256, (input_features.shape[2], input_features.shape[3])

    fpn = FPN(in_channels=I, out_channels=O, num_levels=5, input_size=size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fpn = fpn.to(device)
    input_features = input_features.to(device)
    
    outputs = fpn(input_features)
    
    print("FPN Outputs:")
    print(len(outputs))  
    for i, feat in enumerate(outputs):
        print(f"P{i+3} Shape: {feat.shape}")

    # BiFPN 测试
    # print("\nBiFPN Outputs:")
    # bifpn = BiFPN(in_channels=I, out_channels=O, num_levels=4).to(device)
    # outputs_bi = bifpn(input_features)
    # for i, feat in enumerate(outputs_bi):
    #     print(f"BiP{i+3} Shape: {feat.shape}")