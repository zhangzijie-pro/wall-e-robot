import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class CLIPVisionBackbone(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", freeze_early_layers=True):
        super(CLIPVisionBackbone, self).__init__()
        # 加载预训练CLIP模型
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # 获取视觉编码器
        self.vision_model = self.clip_model.vision_model
        
        # 冻结早期层（可选，保留通用特征）
        if freeze_early_layers:
            for param in self.vision_model.encoder.layers[:6].parameters():
                param.requires_grad = False
        
        # 获取输出维度（特征图通道数）
        self.out_channels = self.clip_model.vision_model.config.hidden_size  # 例如，768 for ViT-B/32
        
    def forward(self, images):
        """
        输入图像，输出特征图
        Args:
            images: Tensor of shape (batch_size, 3, H, W)
        Returns:
            feature_map: Tensor of shape (batch_size, out_channels, H', W')
        """
        # 通过CLIP视觉编码器提取特征
        vision_outputs = self.vision_model(pixel_values=images)
        feature_map = vision_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # 转换特征图形状（ViT输出需重塑）
        batch_size = feature_map.size(0)
        seq_len = feature_map.size(1)
        grid_size = int(seq_len ** 0.5)  # 假设特征图为方形（例如，14x14 for 224x224输入）
        feature_map = feature_map[:, 1:, :].view(batch_size, grid_size, grid_size, self.out_channels)
        feature_map = feature_map.permute(0, 3, 1, 2)  # (batch_size, out_channels, H', W')
        
        return feature_map
    
    def preprocess(self, images):
        """
        预处理输入图像
        Args:
            images: List of PIL images or numpy arrays
        Returns:
            pixel_values: Tensor of shape (batch_size, 3, H, W)
        """
        return self.processor(images=images, return_tensors="pt")["pixel_values"]

# 测试代码
if __name__ == "__main__":
    import torchvision.transforms as transforms
    from PIL import Image
    
    # 加载示例图像
    image = Image.open("bus.jpg").convert("RGB")
    transform = transforms.Compose([transforms.Resize((224, 224))])
    image = transform(image)
    
    # 初始化模型
    backbone = CLIPVisionBackbone()
    
    # 预处理图像
    pixel_values = backbone.preprocess([image]).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # 前向传播
    backbone = backbone.to("cuda" if torch.cuda.is_available() else "cpu")
    feature_map = backbone(pixel_values)
    
    # 打印输出
    print("Feature Map Shape:", feature_map.shape)
    print("Feature Map Sample:", feature_map[0, :5, :5, :5])  # 打印部分特征