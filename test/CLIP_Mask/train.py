import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CLIPYOLOSeg
from dataset import CustomDataset, collate_fn
import numpy as np
import os
from tqdm import tqdm

class DetectionLoss(nn.Module):
    def __init__(self):
        super(DetectionLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')
    
    def forward(self, pred_boxes, gt_boxes, pred_conf, gt_conf):
        """
        检测损失：边界框回归 + 置信度
        Args:
            pred_boxes: Tensor [batch_size, num_anchors*H*W, 4]
            gt_boxes: Tensor [batch_size, num_anchors*H*W, 4]
            pred_conf: Tensor [batch_size, num_anchors*H*W, 1]
            gt_conf: Tensor [batch_size, num_anchors*H*W, 1]
        Returns:
            loss: Scalar
        """
        box_loss = self.smooth_l1_loss(pred_boxes, gt_boxes)
        conf_loss = self.bce_loss(pred_conf, gt_conf)
        return box_loss + conf_loss

class SegmentationLoss(nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
    
    def forward(self, pred_masks, gt_masks):
        """
        分割损失：二值交叉熵
        Args:
            pred_masks: Tensor [batch_size, num_detections, 28, 28]
            gt_masks: Tensor [batch_size, num_detections, 28, 28]
        Returns:
            loss: Scalar
        """
        return self.bce_loss(pred_masks, gt_masks)

class MatchingLoss(nn.Module):
    def __init__(self):
        super(MatchingLoss, self).__init__()
    
    def forward(self, pred_scores, gt_labels, text_features):
        """
        匹配损失：优化边界框与文本特征的余弦相似度
        Args:
            pred_scores: Tensor [batch_size, num_boxes, num_texts]
            gt_labels: List[List[str]], 真实标签
            text_features: Tensor [batch_size, num_texts, 512]
        Returns:
            loss: Scalar
        """
        batch_size, num_boxes, num_texts = pred_scores.shape
        loss = 0.0
        for b in range(batch_size):
            # 假设每个样本一个正样本标签
            target = torch.zeros_like(pred_scores[b])
            for i, label in enumerate(gt_labels[b]):
                # 简化：假设标签与texts顺序一致
                target[:, i] = 1.0 if label in gt_labels[b] else 0.0
            loss += nn.functional.cross_entropy(pred_scores[b], target.argmax(dim=-1))
        return loss / batch_size

class GraspLoss(nn.Module):
    def __init__(self):
        super(GraspLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')
    
    def forward(self, pred_grasp_points, gt_grasp_points):
        """
        抓取点损失：L2损失
        Args:
            pred_grasp_points: Tensor [batch_size, num_detections, 2]
            gt_grasp_points: Tensor [batch_size, num_detections, 2]
        Returns:
            loss: Scalar
        """
        return self.mse_loss(pred_grasp_points, gt_grasp_points)

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, batch_size=4, lr=1e-4, num_epochs=50, save_dir='checkpoints'):
        """
        初始化训练器
        Args:
            model: CLIPYOLOSeg模型
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            batch_size: 批量大小
            lr: 学习率
            num_epochs: 训练轮数
            save_dir: 模型保存目录
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        
        # 损失函数
        self.det_loss_fn = DetectionLoss()
        self.seg_loss_fn = SegmentationLoss()
        self.match_loss_fn = MatchingLoss()
        self.grasp_loss_fn = GraspLoss()
        
        # DataLoader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn
        )
    
    def compute_grasp_points(self, masks):
        """
        从掩码计算抓取点（中心点）
        Args:
            masks: Tensor [batch_size, num_detections, 28, 28]
        Returns:
            grasp_points: Tensor [batch_size, num_detections, 2]
        """
        batch_size, num_detections, H, W = masks.shape
        grasp_points = torch.zeros(batch_size, num_detections, 2, device=masks.device)
        for b in range(batch_size):
            for i in range(num_detections):
                mask = masks[b, i]
                if mask.sum() > 0:
                    y, x = torch.nonzero(mask, as_tuple=True)
                    grasp_points[b, i, 0] = x.float().mean()
                    grasp_points[b, i, 1] = y.float().mean()
        return grasp_points
    
    def train_one_epoch(self, epoch):
        """
        训练一个epoch
        """
        self.model.train()
        total_loss = 0.0
        det_loss_total = 0.0
        seg_loss_total = 0.0
        match_loss_total = 0.0
        grasp_loss_total = 0.0
        
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
            images = batch['images'].to(self.device)
            text_inputs = {k: v.to(self.device) for k, v in batch['text_inputs'].items()}
            gt_boxes = [b.to(self.device) for b in batch['boxes']]
            gt_masks = [m.to(self.device) for m in batch['masks']]
            gt_labels = batch['labels']
            
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(images, [item for sublist in gt_labels for item in sublist])
            
            # 准备检测损失输入
            det_predictions = outputs['det_predictions']
            total_det_loss = 0.0
            for i, (pred, gt_box) in enumerate(zip(det_predictions, gt_boxes)):
                pred = pred.view(pred.size(0), -1, pred.size(-1))  # [batch_size, num_anchors*H*W, 5+num_classes]
                gt_conf = (gt_box.sum(dim=-1) > 0).float().unsqueeze(-1)  # 简单假设：有box则conf=1
                pred_boxes = pred[..., :4]
                pred_conf = pred[..., 4:5]
                total_det_loss += self.det_loss_fn(pred_boxes, gt_box, pred_conf, gt_conf)
            det_loss = total_det_loss / len(det_predictions)
            
            # 分割损失
            pred_masks = outputs['masks']
            total_seg_loss = 0.0
            for pred_m, gt_m in zip(pred_masks, gt_masks):
                if gt_m.shape[0] > 0:
                    total_seg_loss += self.seg_loss_fn(pred_m, gt_m)
            seg_loss = total_seg_loss / max(1, len([m for m in gt_masks if m.shape[0] > 0]))
            
            # 匹配损失
            pred_scores = outputs['scores']  # [batch_size, num_boxes, num_texts]
            text_features = self.model.backbone.forward_text(
                text_inputs['input_ids'],
                text_inputs['attention_mask']
            )
            match_loss = self.match_loss_fn(pred_scores, gt_labels, text_features)
            
            # 抓取点损失
            pred_grasp_points = outputs['grasp_points']
            total_grasp_loss = 0.0
            for pred_gp, gt_m in zip(pred_grasp_points, gt_masks):
                if gt_m.shape[0] > 0:
                    gt_gp = self.compute_grasp_points(gt_m)
                    total_grasp_loss += self.grasp_loss_fn(pred_gp, gt_gp)
            grasp_loss = total_grasp_loss / max(1, len([m for m in gt_masks if m.shape[0] > 0]))
            
            # 总损失
            loss = det_loss + seg_loss + match_loss + grasp_loss
            loss.backward()
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            det_loss_total += det_loss.item()
            seg_loss_total += seg_loss.item()
            match_loss_total += match_loss.item()
            grasp_loss_total += grasp_loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch+1}, Total Loss: {avg_loss:.4f}, Det: {det_loss_total/len(self.train_loader):.4f}, "
              f"Seg: {seg_loss_total/len(self.train_loader):.4f}, Match: {match_loss_total/len(self.train_loader):.4f}, "
              f"Grasp: {grasp_loss_total/len(self.train_loader):.4f}")
        return avg_loss
    
    def validate(self, epoch):
        """
        验证一个epoch
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch['images'].to(self.device)
                text_inputs = {k: v.to(self.device) for k, v in batch['text_inputs'].items()}
                gt_boxes = [b.to(self.device) for b in batch['boxes']]
                gt_masks = [m.to(self.device) for m in batch['masks']]
                gt_labels = batch['labels']
                
                outputs = self.model(images, [item for sublist in gt_labels for item in sublist])
                
                # 检测损失
                det_predictions = outputs['det_predictions']
                total_det_loss = 0.0
                for pred, gt_box in zip(det_predictions, gt_boxes):
                    pred = pred.view(pred.size(0), -1, pred.size(-1))
                    gt_conf = (gt_box.sum(dim=-1) > 0).float().unsqueeze(-1)
                    pred_boxes = pred[..., :4]
                    pred_conf = pred[..., 4:5]
                    total_det_loss += self.det_loss_fn(pred_boxes, gt_box, pred_conf, gt_conf)
                det_loss = total_det_loss / len(det_predictions)
                
                # 分割损失
                pred_masks = outputs['masks']
                total_seg_loss = 0.0
                for pred_m, gt_m in zip(pred_masks, gt_masks):
                    if gt_m.shape[0] > 0:
                        total_seg_loss += self.seg_loss_fn(pred_m, gt_m)
                seg_loss = total_seg_loss / max(1, len([m for m in gt_masks if m.shape[0] > 0]))
                
                # 匹配损失
                pred_scores = outputs['scores']
                text_features = self.model.backbone.forward_text(
                    text_inputs['input_ids'],
                    text_inputs['attention_mask']
                )
                match_loss = self.match_loss_fn(pred_scores, gt_labels, text_features)
                
                # 抓取点损失
                pred_grasp_points = outputs['grasp_points']
                total_grasp_loss = 0.0
                for pred_gp, gt_m in zip(pred_grasp_points, gt_masks):
                    if gt_m.shape[0] > 0:
                        gt_gp = self.compute_grasp_points(gt_m)
                        total_grasp_loss += self.grasp_loss_fn(pred_gp, gt_gp)
                grasp_loss = total_grasp_loss / max(1, len([m for m in gt_masks if m.shape[0] > 0]))
                
                loss = det_loss + seg_loss + match_loss + grasp_loss
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss
    
    def train(self):
        """
        主训练循环
        """
        best_val_loss = float('inf')
        for epoch in range(self.num_epochs):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss
                }, os.path.join(self.save_dir, 'best_model.pth'))
                print(f"Saved best model with validation loss: {val_loss:.4f}")

# 测试代码
if __name__ == "__main__":
    # 初始化数据集
    dataset = CustomDataset(
        dataset_dir='dataset',
        split='train',
        image_size=224,
        mask_size=28
    )
    
    # 简单划分训练和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 初始化模型
    model = CLIPYOLOSeg()
    
    # 初始化训练器
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=4,
        lr=1e-4,
        num_epochs=50,
        save_dir='checkpoints'
    )
    
    # 开始训练
    trainer.train()