import torch
import torch.nn as nn

class GraspHead(nn.Module):
    def __init__(self, mask_size=28):
        super(GraspHead, self).__init__()
        self.mask_size = mask_size
        
        # 可选：回归抓取点偏移
        self.offset_conv = nn.Conv2d(
            1,  # 输入为掩码（单通道）
            2,  # 输出x, y偏移
            kernel_size=3,
            padding=1
        )
        
    def forward(self, masks):
        """
        输入分割掩码，预测抓取点
        Args:
            masks: Tensor, shape (batch_size, total_detections, mask_size, mask_size)
        Returns:
            grasp_points: Tensor, shape (batch_size, total_detections, 2), (x, y)
        """
        batch_size, total_detections, _, _ = masks.shape
        grasp_points = []
        
        for b in range(batch_size):
            mask = masks[b]  # (total_detections, mask_size, mask_size)
            
            # 计算掩码几何中心
            coords_y, coords_x = torch.meshgrid(
                torch.arange(self.mask_size, device=mask.device),
                torch.arange(self.mask_size, device=mask.device),
                indexing='ij'
            )
            coords_x = coords_x.float()
            coords_y = coords_y.float()
            
            mask_sum = mask.sum(dim=(1, 2), keepdim=True) + 1e-6  # 防止除零
            center_x = (mask * coords_x).sum(dim=(1, 2)) / mask_sum.squeeze(-1)  # (total_detections,)
            center_y = (mask * coords_y).sum(dim=(1, 2)) / mask_sum.squeeze(-1)  # (total_detections,)
            
            # 可选：回归偏移
            mask_reshaped = mask.unsqueeze(1)  # (total_detections, 1, mask_size, mask_size)
            offsets = self.offset_conv(mask_reshaped)  # (total_detections, 2, mask_size, mask_size)
            offsets = offsets[:, :, self.mask_size//2, self.mask_size//2]  # 取中心点偏移
            center_x = center_x + offsets[:, 0]
            center_y = center_y + offsets[:, 1]
            
            grasp_points.append(torch.stack([center_x, center_y], dim=-1))  # (total_detections, 2)
        
        grasp_points = torch.stack(grasp_points, dim=0)  # (batch_size, total_detections, 2)
        return grasp_points

# 测试代码
if __name__ == "__main__":
    # 模拟分割掩码
    masks = torch.randn(1, 10584, 28, 28).sigmoid()  # 模拟掩码
    grasp_head = GraspHead(mask_size=28)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grasp_head = grasp_head.to(device)
    masks = masks.to(device)
    
    # 前向传播
    grasp_points = grasp_head(masks)
    
    # 打印输出
    print("Grasp Points Shape:", grasp_points.shape)
    print("Grasp Points Sample:", grasp_points[0, :5, :])