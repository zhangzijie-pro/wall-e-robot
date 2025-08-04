import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer


# preprocess -> image (224, 224) -> pathch 16 -> (224/32) -> (7,7)
class CLIPBackbone(nn.Module):
    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32",
        freeze_vision_early_layers=True,
        freeze_text=True
    ):
        super(CLIPBackbone, self).__init__()
        # 加载预训练CLIP模型
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        
        # 获取视觉和文本编码器
        self.vision_model = self.clip_model.vision_model
        self.text_model = self.clip_model.text_model
        
        # 冻结视觉编码器早期层（可选，保留通用特征）
        if freeze_vision_early_layers:
            for param in self.vision_model.encoder.layers[:6].parameters():
                param.requires_grad = False
        
        # 冻结文本编码器（通常冻结以保留预训练特征）
        if freeze_text:
            for param in self.text_model.parameters():
                param.requires_grad = False
        
        # 获取输出维度
        self.vision_out_channels = self.clip_model.vision_model.config.hidden_size  # 例如，768
        self.text_out_dim = self.clip_model.text_model.config.hidden_size  # 例如，512
    
    def forward_vision(self, images):
        """
        输入图像，输出视觉特征图
        Args:
            images: Tensor of shape (batch_size, 3, H, W)
        Returns:
            feature_map: Tensor of shape (batch_size, vision_out_channels, H', W')
        """
        # 通过CLIP视觉编码器提取特征
        vision_outputs = self.vision_model(pixel_values=images)
        feature_map = vision_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        print("Feature Map Shape:", feature_map.shape)
        # 转换特征图形状（去除CLS token，重塑为2D特征图）
        batch_size = feature_map.size(0)
        seq_len = feature_map.size(1)
        grid_size = int((seq_len - 1) ** 0.5)  # 假设特征图为方形（例如，14x14 for 224x224输入）
        feature_map = feature_map[:, 1:, :].view(batch_size, grid_size, grid_size, self.vision_out_channels)
        feature_map = feature_map.permute(0, 3, 1, 2)  # (batch_size, vision_out_channels, H', W')
        
        return feature_map
    
    def forward_text(self, input_ids, attention_mask=None):
        """
        输入分词后的input_ids和attention_mask，输出文本特征向量
        Args:
            input_ids: Tensor of shape (batch_size, seq_length), 通常seq_length=77
            attention_mask: Tensor of shape (batch_size, seq_length)
        Returns:
            text_features: Tensor of shape (batch_size, text_out_dim)
        """
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output  # (batch_size, hidden_size)
        return text_features
    
    def preprocess_images(self, images):
        """
        预处理输入图像
        Args:
            images: List of PIL images or numpy arrays
        Returns:
            pixel_values: Tensor of shape (batch_size, 3, H, W)
        """
        return self.processor(images=images, return_tensors="pt")["pixel_values"]
    
    def preprocess_text(self, texts):
        """
        预处理输入文本，进行分词并生成input_ids和attention_mask
        Args:
            texts: List of strings (支持中文)
        Returns:
            inputs: Dict with input_ids (batch_size, 77) and attention_mask (batch_size, 77)
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",  # 填充到最大长度77
            max_length=77,         # CLIP默认最大序列长度
            truncation=True        # 截断超长文本
        )
        return {
            "input_ids": inputs["input_ids"],        # (batch_size, 77)
            "attention_mask": inputs["attention_mask"]  # (batch_size, 77)
        }

# 测试代码
if __name__ == "__main__":
    import torchvision.transforms as transforms
    from PIL import Image
    
    # 示例输入
    image = Image.open("bus.jpg").convert("RGB")
    transform = transforms.Compose([transforms.Resize((224, 224))])
    image = transform(image)
    texts = ["红色苹果", "绿色瓶子"]
    
    # 初始化模型
    backbone = CLIPBackbone()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = backbone.to(device)
    
    # 预处理图像和文本
    pixel_values = backbone.preprocess_images([image]).to(device)
    print("Pixel Values Shape:", pixel_values.shape)
    text_inputs = backbone.preprocess_text(texts)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    
    # 前向传播
    vision_features = backbone.forward_vision(pixel_values)
    text_features = backbone.forward_text(text_inputs["input_ids"], text_inputs["attention_mask"])
    
    # 打印输出
    print("Vision Feature Map Shape:", vision_features.shape)
    # print("Vision Feature Map Sample:", vision_features[0, :5, :5, :5])
    print("Text Input IDs Shape:", text_inputs["input_ids"].shape)
    print("Text Attention Mask Shape:", text_inputs["attention_mask"].shape)
    print("Text Features Shape:", text_features.shape)
    print("Text Features Sample:", text_features[0, :5])