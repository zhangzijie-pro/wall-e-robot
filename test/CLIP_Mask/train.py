import os
import yaml
import torch
import clip  # 官方CLIP库
import argparse
from torch.utils.data import DataLoader
from torch import optim

from datasets.dataset import GroundedSegDataset
from utils.loss import compute_total_loss
from model.detector import CLIPSegDetector
from model.backbone.PostionEncoder import PositionalEncoder
from model.matcher.matcher import SimilarityMatcher
from model.head.seg_head import SegmentationHead

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_model(config, device):
    # 加载CLIP模型
    clip_model, _ = clip.load(config["backbone"]["clip_variant"], device=device, jit=False)
    if config["backbone"].get("freeze_vision", True):
        for param in clip_model.visual.parameters():
            param.requires_grad = False
    if config["backbone"].get("freeze_text", True):
        for param in clip_model.transformer.parameters():
            param.requires_grad = False

    # 编码器函数封装
    class TextEncoder:
        def __call__(self, texts):
            tokens = clip.tokenize(texts).to(device)
            with torch.no_grad():
                feats = clip_model.encode_text(tokens)
            return feats

    class ImageEncoder:
        def __call__(self, images, anchors=None):
            # anchors 参数是为了接口一致，暂时未用
            with torch.no_grad():
                feats = clip_model.encode_image(images)
            return feats

    text_encoder = TextEncoder()
    image_encoder = ImageEncoder()

    matcher = SimilarityMatcher(
        iou_threshold=config["region_matcher"]["nms_threshold"],
        top_k=config["region_matcher"]["top_k"]
    )
    pos_encoding = PositionalEncoder(embed_dim=512, mode='add')  # 或根据配置调整
    seg_head = SegmentationHead(in_channels=512, num_prototypes=32)

    model = CLIPSegDetector(
        clip_model=clip_model,
        text_encoder=text_encoder,
        matcher=matcher,
        pos_encoding=pos_encoding,
        seg_head=seg_head,
        image_size=(config["dataset"]["input_size"][1], config["dataset"]["input_size"][0])  # W, H
    ).to(device)

    return model

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        images = batch["image"].to(device)
        text_prompts = batch["text_prompt"]
        gt_masks = batch["masks"].to(device)
        gt_boxes = batch["boxes"].to(device)

        optimizer.zero_grad()

        # 生成anchors的过程可在dataset中实现，或者此处传入，这里假设dataset已经返回anchors
        anchors = batch["anchors"].to(device)

        outputs = model(images, anchors, text_prompts)

        loss, loss_dict = compute_total_loss(
            pred_masks=outputs["masks"],
            gt_masks=gt_masks,
            pred_boxes=outputs.get("boxes", None),
            gt_boxes=gt_boxes
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Train loss: {avg_loss:.4f}")
    return avg_loss

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    for batch in dataloader:
        images = batch["image"].to(device)
        text_prompts = batch["text_prompt"]
        gt_masks = batch["masks"].to(device)
        gt_boxes = batch["boxes"].to(device)
        anchors = batch["anchors"].to(device)

        outputs = model(images, anchors, text_prompts)

        loss, loss_dict = compute_total_loss(
            pred_masks=outputs["masks"],
            gt_masks=gt_masks,
            pred_boxes=outputs.get("boxes", None),
            gt_boxes=gt_boxes
        )

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Validation loss: {avg_loss:.4f}")
    return avg_loss

def main(config_path):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = GroundedSegDataset(config["dataset"], split="train")
    val_dataset = GroundedSegDataset(config["dataset"], split="val")

    train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size", 8), shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.get("batch_size", 8), shuffle=False, num_workers=4)

    model = build_model(config["model"], device)

    optimizer = optim.AdamW(model.parameters(), lr=config["model"]["training"]["lr"], weight_decay=config["model"]["training"]["weight_decay"])

    epochs = config["model"]["training"]["epochs"]
    best_val_loss = float("inf")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("checkpoints", exist_ok=True)
            save_path = os.path.join("checkpoints", "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/model_config.yaml")
    args = parser.parse_args()
    main(args.config)