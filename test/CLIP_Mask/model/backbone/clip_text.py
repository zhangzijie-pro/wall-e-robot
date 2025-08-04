import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer

class CLIPTextBackbone(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", freeze=True):
        super(CLIPTextBackbone, self).__init__()
        # 加载预训练CLIP模型
        self.clip_model = CLIPModel.from_pretrained(model_name)
        # 加载专门的Tokenizer用于文本处理
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        
        # 获取文本编码器
        self.text_model = self.clip_model.text_model
        
        # 冻结文本编码器（通常冻结以保留预训练特征）
        if freeze:
            for param in self.text_model.parameters():
                param.requires_grad = False
        
        # 获取输出维度
        self.out_dim = self.clip_model.text_model.config.hidden_size  # 例如，512
        
    def forward(self, input_ids, attention_mask=None):
        """
        输入分词后的input_ids和attention_mask，输出文本特征向量
        Args:
            input_ids: Tensor of shape (batch_size, seq_length), 通常seq_length=77
            attention_mask: Tensor of shape (batch_size, seq_length)
        Returns:
            text_features: Tensor of shape (batch_size, out_dim)
        """
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output  # (batch_size, hidden_size)
        return text_features
    
    def preprocess(self, texts):
        """
        预处理输入文本，进行分词并生成input_ids和attention_mask
        Args:
            texts: List of strings (支持中文)
        Returns:
            inputs: Dict with input_ids (batch_size, 77) and attention_mask (batch_size, 77)
        """
        # 使用CLIPTokenizer处理文本
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
    # 示例中文文本
    texts = ["hello world", "read cup!"]
    
    # 初始化模型
    # backbone = CLIPTextBackbone()
    
    # 预处理文本
    # inputs = backbone.preprocess(texts)
    # inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}
    
    # # 前向传播
    # backbone = backbone.to("cuda" if torch.cuda.is_available() else "cpu")
    # text_features = backbone(inputs["input_ids"], inputs["attention_mask"])
    
    # # 打印输出
    # print("Input IDs Shape:", inputs["input_ids"].shape)
    # print("Input IDs Sample:", inputs["input_ids"][0, :10])  # 打印前10个token ID
    # print("Attention Mask Shape:", inputs["attention_mask"].shape)
    # print("Attention Mask Sample:", inputs["attention_mask"][0, :10])  # 打印前10个掩码
    # print("Text Features Shape:", text_features.shape)
    # print("Text Features Sample:", text_features[0, :5])  # 打印部分特征
    
    
    from transformers import CLIPModel

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    print("Text Model \n",model.text_model)
    print("Vision Model \n",model.vision_model)