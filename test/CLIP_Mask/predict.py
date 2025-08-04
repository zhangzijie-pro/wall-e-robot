import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from model.model import CLIPYOLOSeg
from transformers import CLIPProcessor
import torchvision.transforms as transforms

class InferencePipeline:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化推理管道，加载模型和CLIP处理器
        Args:
            model_path: 预训练模型权重路径
            device: 推理设备（'cuda' 或 'cpu'）
        """
        self.device = device
        
        # 加载模型
        self.model = CLIPYOLOSeg().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # CLIP处理器
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])
    
    def preprocess_image(self, image):
        """
        预处理单张图像
        Args:
            image: PIL Image 或文件路径
        Returns:
            image_tensor: Tensor [1, 3, 224, 224]
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        image_tensor = self.image_transform(image).unsqueeze(0)  # [1, 3, 224, 224]
        return image_tensor.to(self.device)
    
    def preprocess_texts(self, texts):
        """
        预处理文本提示
        Args:
            texts: List of strings
        Returns:
            text_inputs: Dict {input_ids, attention_mask}
        """
        if isinstance(texts, str):
            texts = [texts]
        text_inputs = self.clip_processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        return text_inputs
    
    def denormalize_box(self, box, img_width, img_height):
        """
        将归一化边界框转换为像素坐标
        Args:
            box: Tensor [4] (cx, cy, w, h)
            img_width, img_height: 原始图像尺寸
        Returns:
            box: List [x, y, w, h]
        """
        cx, cy, w, h = box
        x = (cx - w / 2) * img_width
        y = (cy - h / 2) * img_height
        w *= img_width
        h *= img_height
        return [x, y, w, h]
    
    def infer(self, image, texts, conf_threshold=0.5, nms_iou_threshold=0.5):
        """
        执行推理
        Args:
            image: PIL Image 或文件路径
            texts: List of strings
            conf_threshold: 置信度阈值
            nms_iou_threshold: NMS IoU 阈值
        Returns:
            results: Dict containing det_boxes, masks, scores, grasp_points
        """
        # 预处理
        image_tensor = self.preprocess_image(image)
        text_inputs = self.preprocess_texts(texts)
        original_image = Image.open(image).convert('RGB') if isinstance(image, str) else image
        img_width, img_height = original_image.size
        
        # 推理
        with torch.no_grad():
            outputs = self.model([original_image], texts)
        
        # 后处理
        results = {
            'det_boxes': [],
            'masks': [],
            'scores': [],
            'grasp_points': []
        }
        
        # 处理边界框
        det_boxes = outputs['det_boxes'][0]  # [num_boxes, 6]
        if det_boxes.shape[0] > 0:
            for box in det_boxes:
                confidence = box[4].item()
                if confidence >= conf_threshold:
                    box_coords = box[:4].cpu().numpy()
                    box_coords = self.denormalize_box(box_coords, img_width, img_height)
                    results['det_boxes'].append(box_coords + [confidence, box[5].item()])
        
        # 处理掩码
        masks = outputs['masks'][0]  # [num_boxes, 28, 28]
        if masks.shape[0] > 0:
            results['masks'] = masks.cpu().numpy()
        
        # 处理匹配分数
        scores = outputs['scores'][0]  # [num_boxes, num_texts]
        if scores.shape[0] > 0:
            results['scores'] = scores.cpu().numpy()
        
        # 处理抓取点
        grasp_points = outputs['grasp_points'][0]  # [num_boxes, 2]
        if grasp_points.shape[0] > 0:
            for gp in grasp_points:
                x, y = gp.cpu().numpy()
                x *= img_width / 28  # 缩放回原始尺寸
                y *= img_height / 28
                results['grasp_points'].append([x, y])
        
        return results, original_image
    
    def visualize_results(self, image, results, texts, save_path=None):
        """
        可视化推理结果
        Args:
            image: PIL Image
            results: Dict containing det_boxes, masks, scores, grasp_points
            texts: List of strings
            save_path: 保存路径（可选）
        """
        img_np = np.array(image)
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img_np)
        
        for i, (box, mask, score, gp) in enumerate(zip(
            results['det_boxes'], results['masks'], results['scores'], results['grasp_points']
        )):
            # 绘制边界框
            x, y, w, h, conf, _ = box
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y - 10, f'{texts[np.argmax(score)]}: {conf:.2f}', 
                    color='r', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
            
            # 绘制掩码
            mask_resized = Image.fromarray(mask * 255).resize((int(w), int(h)))
            mask_np = np.array(mask_resized) / 255
            mask_alpha = np.zeros((int(h), int(w), 4))
            mask_alpha[mask_np > 0.5, :] = [0, 0, 1, 0.5]  # 蓝色半透明
            ax.imshow(mask_alpha, extent=(x, x+w, y+h, y))
            
            # 绘制抓取点
            gp_x, gp_y = gp
            ax.plot(gp_x + x, gp_y + y, 'g*', markersize=15, label='Grasp Point' if i == 0 else None)
        
        ax.legend()
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    # 初始化推理管道
    pipeline = InferencePipeline(model_path='model.pth')
    
    # 输入数据
    image_path = 'dataset/images/0001.jpg'
    texts = ["the red mug on the table"]
    
    # 推理
    results, original_image = pipeline.infer(
        image=image_path,
        texts=texts,
        conf_threshold=0.5,
        nms_iou_threshold=0.5
    )
    
    print("Detected Boxes:", results['det_boxes'])
    print("Masks Shape:", results['masks'].shape if len(results['masks']) > 0 else "Empty")
    print("Scores Shape:", results['scores'].shape if len(results['scores']) > 0 else "Empty")
    print("Grasp Points:", results['grasp_points'])
    
    # 可视化
    pipeline.visualize_results(
        image=original_image,
        results=results,
        texts=texts,
        save_path='output_0001.jpg'
    )