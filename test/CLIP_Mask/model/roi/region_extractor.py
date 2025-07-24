import torch
import torchvision.transforms as T
from torchvision.transforms.functional import pad
from PIL import Image, ImageOps
from typing import Union

class ImagePatchExtractor:
    def __init__(self, image_size=(640, 640), patch_size=(224, 224), padding_mode='constant', padding_value=0):
        self.image_h, self.image_w = image_size
        self.patch_h, self.patch_w = patch_size
        self.padding_mode = padding_mode
        self.padding_value = padding_value

        self.transform = T.Compose([
            T.Resize((self.patch_h, self.patch_w), antialias=True),
            T.ToTensor()
        ])

    def extract(self, image: Union[Image.Image, torch.Tensor], anchors: torch.Tensor) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            image = T.ToPILImage()(image.cpu())

        patches = []

        for anchor in anchors.cpu():
            cx, cy, w, h = anchor.tolist()

            left = int(round(cx - w / 2))
            top = int(round(cy - h / 2))
            right = int(round(cx + w / 2))
            bottom = int(round(cy + h / 2))

            # 计算需要的 padding 边距
            pad_left = max(0, -left)
            pad_top = max(0, -top)
            pad_right = max(0, right - self.image_w)
            pad_bottom = max(0, bottom - self.image_h)

            # 先 pad 图像保证裁剪区域完整
            if any([pad_left, pad_top, pad_right, pad_bottom]):
                image_padded = ImageOps.expand(image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=self.padding_value)
                # 调整裁剪坐标到 padded 图像空间
                left += pad_left
                top += pad_top
                right += pad_left
                bottom += pad_top
            else:
                image_padded = image

            cropped = image_padded.crop((left, top, right, bottom))
            resized = self.transform(cropped)
            patches.append(resized)

        return torch.stack(patches)  # [N, C, patch_h, patch_w]