import torch
import torch.nn as nn
from backbone.clip_backbone import CLIPBackbone
from neck.FPN import FPN
from head.detection_head import DetectionHead
from head.segment_head import SegmentationHead
from head.match_head import MatchingHead
from head.grasp_head import GraspHead
from head.feature import FeatureEnhancer, MultiHeadSelfAttention
import cv2

import torch
import torch.nn as nn


class CLIPYOLOSeg(nn.Module):
    def __init__(self):
        super(CLIPYOLOSeg, self).__init__()
        self.layer = 3
        self.backbone = CLIPBackbone(
            model_name="openai/clip-vit-base-patch32",
            freeze_vision_early_layers=True,
            freeze_text=True
        )
        
        self.neck = FPN(
            in_channels=self.backbone.vision_out_channels,
            out_channels=256,
            num_levels=self.layer
        )
        
        self.feature_enhancer = FeatureEnhancer(
            in_channels=256,
            embed_dim=512,
            num_heads=8,
            feature_sizes=[(80, 80), (40, 40), (20, 20)]
        )
        
        self.detection_head = DetectionHead(
            in_channels=256,
            num_fpn_layers=self.layer,
            num_anchors=3,
            num_classes=1
        )
        self.segmentation_head = SegmentationHead(
            in_channels=256,
            proto_channels=32,
            mask_size=28
        )
        self.matching_head = MatchingHead(
            in_channels=256,
            hidden_dim=self.backbone.text_out_dim
        )
        self.grasp_head = GraspHead(
            mask_size=28
        )
    
    def forward(self, images, texts):
        pixel_values = self.backbone.preprocess_images(images)
        text_inputs = self.backbone.preprocess_text(texts)
        
        vision_features = self.backbone.forward_vision(pixel_values)
        text_features = self.backbone.forward_text(
            text_inputs["input_ids"],
            text_inputs["attention_mask"]
        ).unsqueeze(0)
        
        neck_features = self.neck(vision_features)
        
        det_predictions = self.detection_head(neck_features)
        
        neck_features = self.feature_enhancer(neck_features, det_predictions)
        
        det_boxes = self.detection_head.decode_boxes(det_predictions)
        nms_boxes = self.detection_head.nms(det_boxes)
        
        masks = self.segmentation_head(neck_features, det_predictions)
        aligned_masks = self.segmentation_head.align_with_boxes(masks, det_boxes)
        
        scores = self.matching_head(neck_features, det_boxes, text_features)
        
        grasp_points = self.grasp_head(masks)
        
        return {
            "det_boxes": nms_boxes,
            "masks": aligned_masks,
            "scores": scores,
            "grasp_points": grasp_points
        }
        
def denormalize_box(box, img_width, img_height):
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


"""
Feature Map Shape: torch.Size([1, 50, 768])
Strides: 3
Number of FPN Levels: 3
Number of Anchors per Level: 3
NMS Boxes (first batch): torch.Size([114, 6])
Aligned Masks (first batch): torch.Size([177, 28, 28])
Matching Scores Shape: torch.Size([1, 177, 2])
Grasp Points Shape: torch.Size([1, 177, 177, 2])
"""
if __name__ == "__main__":
    from PIL import Image
    import torchvision.transforms as transforms
    
    
    conf_threshold = 0
    # image = Image.open("./backbone/bus.jpg").convert("RGB")
    image = cv2.imread("./backbone/bus.jpg")
    img_height,img_width = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    image = Image.fromarray(image)
    
    transform = transforms.Compose([transforms.Resize((224, 224))])
    image = transform(image)
    texts = ["person", "bus"]
    
    model = CLIPYOLOSeg()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    with torch.no_grad():
        outputs = model([image], texts)
    
    print("NMS Boxes (first batch):", outputs["det_boxes"][0].shape if outputs["det_boxes"][0].shape[0] > 0 else "Empty")
    print("Aligned Masks (first batch):", outputs["masks"][0].shape if outputs["masks"][0].shape[0] > 0 else "Empty")
    print("Matching Scores Shape:", outputs["scores"].shape)
    print("Grasp Points Shape:", outputs["grasp_points"].shape)
    
    # results = {
    #     'det_boxes': [],
    #     'masks': [],
    #     'scores': [],
    #     'grasp_points': []
    # }
 
    # det_boxes = outputs['det_boxes'][0]  # [num_boxes, 6]
    # if det_boxes.shape[0] > 0:
    #     for box in det_boxes:
    #         confidence = float(box[4])
    #         if confidence >= conf_threshold:
    #             box_coords = [float(x) for x in box[:4]]
    #             box_coords = denormalize_box(box_coords, img_width, img_height)
    #             # Round to 2 decimal places for clarity
    #             box_coords = [round(x, 2) for x in box_coords]
    #             results['det_boxes'].append(box_coords + [round(confidence, 2), int(box[5])])

    # masks = outputs['masks'][0]  # [num_boxes, 28, 28]
    # if masks.shape[0] > 0:
    #     results['masks'] = [[round(float(x), 2) for x in mask.flatten()] for mask in masks]
    
    # # Process matching scores
    # scores = outputs['scores'][0]  # [num_boxes, num_texts]
    # if scores.shape[0] > 0:
    #     results['scores'] = [[round(float(x), 2) for x in score] for score in scores]
        
    # # grasp_points = outputs['grasp_points'][0]  # [num_boxes, 2]
    # # if grasp_points.shape[0] > 0:
    # #     for gp in grasp_points:
    # #         x, y = [float(x) for x in gp]
    # #         x = round(x * img_width / 28, 2)  # Scale to original dimensions
    # #         y = round(y * img_height / 28, 2)
    # #         results['grasp_points'].append([x, y])
            
    # print("\nDetection Results:")
    # print("Bounding Boxes:")
    # for i, box in enumerate(results['det_boxes']):
    #     x, y, w, h, conf, cls = box
    #     print(f"Box {i+1}: x={x:.2f}, y={y:.2f}, w={w:.2f}, h={h:.2f}, "
    #           f"confidence={conf:.2f}, class={cls}")
    
    # print("\nMatching Scores:")
    # for i, score in enumerate(results['scores']):
    #     print(f"Object {i+1}: {[f'{s:.2f}' for s in score]}")