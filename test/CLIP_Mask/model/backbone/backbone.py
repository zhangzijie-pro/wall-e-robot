import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer, CLIPFeatureExtractor
from PIL import Image
import math
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict

class CLIPBackbone(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch16", freeze_vision=False, freeze_text=True):
        super(CLIPBackbone, self).__init__()
        self.model_name = model_name

        # 加载CLIP模型（transformers版本）
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(model_name)

        self.vision_out_channels = self.clip_model.vision_model.config.hidden_size  # 768
        self.text_out_dim = self.clip_model.text_model.config.hidden_size  # 512
        self.image_size = self.feature_extractor.size  # 通常为224
        self.patch_size = self.clip_model.vision_model.config.patch_size  # 通常为32

        # 添加 FPN 模块：将 ViT 输出的 14x14 单尺度特征图映射为多尺度特征
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[self.vision_out_channels],
            out_channels=256
        )

        if freeze_vision:
            for p in self.clip_model.vision_model.parameters():
                p.requires_grad = False
        if freeze_text:
            for p in self.clip_model.text_model.parameters():
                p.requires_grad = False

    def preprocess_images(self, images):
        if isinstance(images, (Image.Image, torch.Tensor)):
            images = [images]
        pixel_values = self.feature_extractor(images=images, return_tensors="pt")["pixel_values"]
        return pixel_values  # [B, 3, 224, 224]

    def preprocess_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        tokenized = self.tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        }

    def forward_vision(self, pixel_values):
        outputs = self.clip_model.vision_model(pixel_values)
        # print("Vision Model Outputs:", outputs.shape)
        hidden_states = outputs.last_hidden_state  # [B, 1 + HW, C]
        patch_tokens = hidden_states[:, 1:, :]     # 去除CLS
        H = W = int(math.sqrt(patch_tokens.shape[1]))
        vision_features = patch_tokens.view(patch_tokens.size(0), H, W, -1).permute(0, 3, 1, 2)
        # 将 14x14 的特征图作为 C5 输入 FPN，构建多尺度特征金字塔
        fpn_input = {"0": vision_features}  # "0" 是 level 名称
        fpn_features = self.fpn(fpn_input)  # 返回 OrderedDict，如 {"0": ..., "1": ..., ...}
        return fpn_features  # dict of [B, 256, H, W]

    def forward_text(self, input_ids, attention_mask):
        outputs = self.clip_model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # [B, 512]
        return pooled.unsqueeze(1)  # [B, 1, 512]
    
if __name__ == "__main__":
    from PIL import Image

    image = Image.open("bus.jpg").convert("RGB")
    print(image.size)
    texts = ["a bus on the road"]

    backbone = CLIPBackbone(
        model_name="openai/clip-vit-base-patch16",
        freeze_vision=False,
        freeze_text=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = backbone.to(device)

    pixel_values = backbone.preprocess_images([image]).to(device)
    text_inputs = backbone.preprocess_text(texts)
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device)

    vision_features = backbone.forward_vision(pixel_values)
    for level_name, feature in vision_features.items():
        print(f"FPN Level {level_name}: Shape = {feature.shape}")
    text_features = backbone.forward_text(input_ids, attention_mask)

    print("Text Features Shape:", text_features.shape)      # [B, 1, 512]