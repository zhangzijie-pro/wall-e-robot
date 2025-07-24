import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class GroundedSegDataset(Dataset):
    def __init__(self, annotation_file, image_root, mask_root, image_size=224):
        super().__init__()
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        self.image_root = image_root
        self.mask_root = mask_root
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_path = os.path.join(self.image_root, ann["image_id"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        prompt = ann["text_prompt"]
        if isinstance(prompt, list):
            prompt = " ".join(prompt)

        objects = ann["objects"]
        boxes = []
        masks = []
        labels = []

        for obj in objects:
            x, y, w, h = obj["box"]
            boxes.append([x, y, x + w, y + h])
            labels.append(obj["label"])

            mask_path = os.path.join(self.mask_root, obj["mask_path"])
            mask = Image.open(mask_path).convert("L").resize((self.image_size, self.image_size))
            mask = torch.from_numpy(np.array(mask)).float() / 255.0
            masks.append(mask)

        boxes = torch.tensor(boxes).float()  # [N, 4]
        masks = torch.stack(masks, dim=0)    # [N, H, W]

        return {
            "image": image,
            "prompt": prompt,
            "boxes": boxes,
            "masks": masks,
            "labels": labels
        }