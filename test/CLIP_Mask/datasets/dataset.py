import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from transformers import CLIPProcessor
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomDataset(Dataset):
    def __init__(self, dataset_dir, split='train', image_size=224, mask_size=28, augment=True):
        """
        自定义数据集，加载图像、文本提示、边界框和掩码，支持数据增强
        Args:
            dataset_dir: 数据集根目录
            split: 数据集划分（'train' 或 'val'）
            image_size: 图像调整大小（默认224）
            mask_size: 掩码调整大小（默认28）
            augment: 是否应用数据增强（训练时开启，验证时关闭）
        """
        super(CustomDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.split = split
        self.image_size = image_size
        self.mask_size = mask_size
        self.augment = augment
        
        # 加载CLIP处理器
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # 基本图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        # 掩码预处理
        self.mask_transform = transforms.Compose([
            transforms.Resize((mask_size, mask_size)),
            transforms.ToTensor()
        ])
        
        # 数据增强（仅训练时）
        self.augment_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels'])) if augment else None
        
        # 加载annotations
        with open(os.path.join(dataset_dir, 'annotations.json'), 'r') as f:
            self.annotations = json.load(f)
        
        self.data = [ann for ann in self.annotations]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取单条数据
        Returns:
            Dict containing:
                - image: Tensor [3, image_size, image_size]
                - text_inputs: Dict {input_ids, attention_mask}
                - boxes: Tensor [num_objects, 4] (cx, cy, w, h)
                - masks: Tensor [num_objects, mask_size, mask_size]
                - labels: List of strings
        """
        ann = self.data[idx]
        image_id = ann['image_id']
        text_prompt = ann['text_prompt']
        objects = ann['objects']
        
        # 加载图像
        image_path = os.path.join(self.dataset_dir, 'images', image_id)
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # 加载掩码和边界框
        boxes = []
        masks = []
        labels = []
        for obj in objects:
            x, y, w, h = obj['box']
            boxes.append([x, y, w, h])
            mask_path = os.path.join(self.dataset_dir, obj['mask_path'])
            mask = Image.open(mask_path).convert('L')
            masks.append(np.array(mask))
            labels.append(obj['label'])
        
        # 数据增强
        if self.augment and self.augment_transform:
            augmented = self.augment_transform(
                image=image_np,
                bboxes=boxes,
                labels=[0] * len(boxes),  # 占位，albumentations需要
                masks=masks
            )
            image_np = augmented['image']  # 已转换为张量
            boxes = augmented['bboxes']
            masks = augmented['masks']
        else:
            image_np = self.image_transform(image)
            masks = [self.mask_transform(Image.fromarray(m)).squeeze(0) for m in masks]
        
        # 转换为(cx, cy, w, h)，归一化到[0,1]
        img_width, img_height = image.size
        boxes_norm = []
        for x, y, w, h in boxes:
            cx = (x + w / 2) / img_width
            cy = (y + h / 2) / img_height
            w_norm = w / img_width
            h_norm = h / img_height
            boxes_norm.append([cx, cy, w_norm, h_norm])
        
        # 处理掩码
        masks = [torch.tensor(m, dtype=torch.float32) if isinstance(m, np.ndarray) else m for m in masks]
        masks = [m > 0.5 for m in masks]  # 二值化
        masks = torch.stack(masks, dim=0) if masks else torch.zeros((0, self.mask_size, self.mask_size))
        
        # 文本预处理
        text_inputs = self.clip_processor(
            text=text_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
        
        # 转换为张量
        boxes = torch.tensor(boxes_norm, dtype=torch.float32) if boxes_norm else torch.zeros((0, 4))
        
        return {
            'image': image_np if self.augment else image_np,
            'text_inputs': text_inputs,
            'boxes': boxes,
            'masks': masks,
            'labels': labels
        }

def collate_fn(batch):
    """
    自定义collate函数，处理变长的边界框和掩码
    """
    images = torch.stack([item['image'] for item in batch])
    text_inputs = {
        'input_ids': torch.stack([item['text_inputs']['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['text_inputs']['attention_mask'] for item in batch])
    }
    boxes = [item['boxes'] for item in batch]
    masks = [item['masks'] for item in batch]
    labels = [item['labels'] for item in batch]
    return {
        'images': images,
        'text_inputs': text_inputs,
        'boxes': boxes,
        'masks': masks,
        'labels': labels
    }

if __name__ == "__main__":
    dataset = CustomDataset(
        dataset_dir='dataset',
        split='train',
        image_size=224,
        mask_size=28,
        augment=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    for batch in dataloader:
        print("Batch Keys:", batch.keys())
        print("Images Shape:", batch['images'].shape)
        print("Text Input IDs Shape:", batch['text_inputs']['input_ids'].shape)
        print("Boxes:", [b.shape for b in batch['boxes']])
        print("Masks:", [m.shape for m in batch['masks']])
        print("Labels:", batch['labels'])
        break