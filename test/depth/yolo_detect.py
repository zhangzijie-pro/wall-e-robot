import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2
import matplotlib

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(DEVICE)

def load_yolo_model(model_path='yolo11n.pt'):
    return YOLO(model_path)

def detect_objects(image, model):
    results = model(image)
    boxes = results[0].boxes
    detections = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy()
    return detections, scores, classes, results[0].names

def draw_boxes_on_image(image, detections, scores, classes, class_names):
    image = image.copy()
    for box, score, cls in zip(detections, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[int(cls)]} {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image

def load_depth_model():
    model_configs = {
        'vits': {
            'encoder': 'vits',
            'features': 64,
            'out_channels': [48, 96, 192, 384]
        }
    }
    model = DepthAnythingV2(**model_configs['vits'])
    model.load_state_dict(torch.load('./depth_anything_v2_vits.pth', map_location='cpu'), strict=False)
    return model.to(DEVICE).eval()

# 执行深度估计并返回伪彩色图
def get_depth_colormap(raw_image, model, input_size=518, grayscale=False):
    cmap = matplotlib.colormaps['Spectral_r']
    depth = model.infer_image(raw_image, input_size)  # 转RGB
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)

    if grayscale:
        return np.repeat(depth[..., np.newaxis], 3, axis=-1)
    else:
        return (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    yolo_model = load_yolo_model()
    depth_model = load_depth_model()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_resized = cv2.resize(frame, (640, 480))
        detections, scores, classes, class_names = detect_objects(frame_resized, yolo_model)
        yolo_result = draw_boxes_on_image(frame_resized, detections, scores, classes, class_names)

        depth_colormap = get_depth_colormap(frame_resized, depth_model, grayscale=False)
        depth_with_boxes = draw_boxes_on_image(depth_colormap, detections, scores, classes, class_names)

        combined = np.hstack((yolo_result, depth_with_boxes))
        cv2.imshow('YOLO Detection | Depth Estimation', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
