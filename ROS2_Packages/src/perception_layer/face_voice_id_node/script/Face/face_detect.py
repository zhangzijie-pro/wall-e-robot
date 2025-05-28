# speaker_recognition/Face/Face_detect.py

import os
import time
from math import ceil

import MNN
import cv2
import numpy as np
import torch
import box_utils_numpy as box_utils


min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
strides = [8, 16, 32, 64]


def define_img_size(image_size):
    """生成先验框（priors）用于目标检测
    
    Args:
        image_size (tuple or list): 输入图像的尺寸，格式为 (width, height)
    
    Returns:
        torch.Tensor: 先验框张量，形状为 (N, 4)，每行表示 [x_center, y_center, w, h]
    
    Notes:
        - 假设全局变量 `strides` 和 `min_boxes` 已定义，分别表示特征图的步幅和最小框尺寸
        - `strides` 是一个列表，表示不同特征图的步幅（如 [8, 16, 32]）
        - `min_boxes` 是一个列表，表示每个特征图上的最小框尺寸
    """
    shrinkage_list = []
    feature_map_w_h_list = []
    for size in image_size:
        feature_map = [ceil(size / stride) for stride in strides]
        feature_map_w_h_list.append(feature_map)

    for i in range(0, len(image_size)):
        shrinkage_list.append(strides)
    priors = generate_priors(feature_map_w_h_list, shrinkage_list, image_size, min_boxes)
    return priors


def generate_priors(feature_map_list, shrinkage_list, image_size, min_boxes, clamp=True):
    """
    根据特征图生成先验框（priors）
    
    Args:
        feature_map_list (list): 特征图尺寸列表，每个元素为 [width, height]
        shrinkage_list (list): 步幅列表，每个元素为特征图的步幅 [stride_w, stride_h]
        image_size (tuple or list): 输入图像尺寸，格式为 (width, height)
        min_boxes (list): 每个特征图的最小框尺寸列表
        clamp (bool, optional): 是否将先验框值限制在 [0, 1] 范围内，默认为 True
    
    Returns:
        torch.Tensor: 先验框张量，形状为 (N, 4)，每行表示 [x_center, y_center, w, h]
    
    Notes:
        - 先验框的坐标和尺寸是相对于输入图像的归一化值
        - x_center, y_center 表示框中心点的归一化坐标
        - w, h 表示框的归一化宽度和高度
    """
    priors = []
    for index in range(0, len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]
        for j in range(0, feature_map_list[1][index]):
            for i in range(0, feature_map_list[0][index]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h

                for min_box in min_boxes[index]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([
                        x_center,
                        y_center,
                        w,
                        h
                    ])
    print("priors nums:{}".format(len(priors)))
    priors = torch.tensor(priors)
    if clamp:
        torch.clamp(priors, 0.0, 1.0, out=priors)
    return priors


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    """
    根据模型输出预测边界框、标签和置信度
    
    Args:
        width (int): 输入图像的宽度
        height (int): 输入图像的高度
        confidences (np.ndarray): 置信度分数，形状为 (N, num_classes)
        boxes (np.ndarray): 边界框坐标，形状为 (N, 4)，格式为 [x1, y1, x2, y2] 或归一化值
        prob_threshold (float): 置信度阈值，低于此值的框被过滤
        iou_threshold (float, optional): 非极大值抑制（NMS）的 IoU 阈值，默认为 0.3
        top_k (int, optional): NMS 保留的最大框数，-1 表示无限制，默认为 -1
    
    Returns:
        tuple:
            - np.ndarray: 最终边界框坐标，形状为 (M, 4)，格式为 [x1, y1, x2, y2]，整数值
            - np.ndarray: 标签数组，形状为 (M,)，表示每个框的类别索引
            - np.ndarray: 置信度数组，形状为 (M,)，表示每个框的置信度分数
    
    Notes:
        - 假设 box_utils.hard_nms 已定义，用于执行非极大值抑制
        - confidences 和 boxes 是模型输出的原始数据，通常需要后处理
    """
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

def crop_save_boxes(image_ori, boxes, save_dir):
    """
    Args:
        image_ori (np.ndarray): 输入的原始图像
        boxes (np.ndarray): 边界框坐标，形状为 (N, 4)，每行为 [x1, y1, x2, y2]
        save_dir (str): 保存裁剪图片的目录
    
    Returns:
        None
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if image_ori is None:
        return 
    if boxes.shape[0] == 0 or boxes is None:
        return
    
    h,w = image_ori.shape[:2]
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        x1,y1,x2,y2 = map(int, box)

        x1 = max(0,x1)
        y1 = max(0,y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        cropped = image_ori[y1:y2, x1:x2]
        print("cropped:",cropped)

        save_path = os.path.join(save_dir, f"crop_{i:03d}.png")
        img = cv2.resize(cropped,(125,125))
        cv2.imwrite(save_path, img)
    
    return None

def Get_crop_Tensor(image_ori, boxes):
    """
    得到裁剪内容的张量信息
    """
    if image_ori is None:
        return
    if boxes.shape[0] == 0 or boxes is None:
        return
    

    image_crops = []
    h,w = image_ori[:2]
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        x1,y1,x2,y2 = map(int, box)
        x1 = max(0,x1)
        y1 = max(0,y1)
        x2 = min(w,x2)
        y2 = min(h,y2)

        if x2<=x1 or y2<=y1:
            continue
    
        cropped = image_ori[y1:y2, x1:x2]
        cropped_tensor = torch.tensor(cropped)
        image_crops.append(cropped_tensor)
    
    return image_crops


def inference(
    input_img_path,
    save_path,
    model_path,
    input_size=[320, 240],
    image_mean=127.5,
    image_std=128.0,
    center_variance=0.1,
    size_variance=0.2,
    threshold=0.7,
    return_tensor=False,
):
    """
    使用 MNN 模型进行推理并裁剪检测到的边界框区域
    进行验证测试集使用
    """
    
    priors = define_img_size(input_size)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for file_path in os.listdir(input_img_path):
        img_path = os.path.join(input_img_path, file_path)
        image_ori = cv2.imread(img_path)
        
        if image_ori is None:
            print(f"警告：无法加载图像 {img_path}，跳过")
            continue
        
        interpreter = MNN.Interpreter(model_path)
        session = interpreter.createSession()
        input_tensor = interpreter.getSessionInput(session)
        
        # 图像预处理
        image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, tuple(input_size))
        image = image.astype(np.float32)  # 明确转换为 np.float32
        image = (image - image_mean) / image_std
        image = image.transpose((2, 0, 1))  # 转换为 CHW 格式
        
        # 确保输入形状为 (1, 3, height, width)
        image = np.expand_dims(image, axis=0)  # 添加 batch 维度
        if image.shape != (1, 3, input_size[1], input_size[0]):
            print(f"错误：图像形状 {image.shape} 不符合预期 (1, 3, {input_size[1]}, {input_size[0]})")
            continue
        
        # 准备输入张量
        tmp_input = MNN.Tensor(
            (1, 3, input_size[1], input_size[0]),
            MNN.Halide_Type_Float,
            image,
            MNN.Tensor_DimensionType_Caffe
        )
        input_tensor.copyFrom(tmp_input)
        
        # 运行推理并计时
        start_time = time.time()
        interpreter.runSession(session)
        
        # 获取输出
        scores = interpreter.getSessionOutput(session, "scores").getData()
        boxes = interpreter.getSessionOutput(session, "boxes").getData()
        
        # 重塑输出
        boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
        scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
        
        print(f"推理时间: {round(time.time() - start_time, 4)} 秒")
        
        # 后处理边界框
        boxes = box_utils.convert_locations_to_boxes(
            boxes, priors, center_variance, size_variance
        )
        boxes = box_utils.center_form_to_corner_form(boxes)
        boxes, labels, probs = predict(
            image_ori.shape[1], image_ori.shape[0], scores, boxes, threshold
        )
        
        return crop_save_boxes(image_ori, boxes, save_path) if return_tensor else Get_crop_Tensor(image_ori, boxes)
        # 裁剪并保存边界框区域



def inference_img(
    img_tensor,
    model_path,
    input_size=[320, 240],
    image_mean=127.5,
    image_std=128.0,
    center_variance=0.1,
    size_variance=0.2,
    threshold=0.7,
    save_path=None,
    return_tensor=True,
):
    """
    Detect Face

    Args:
        img_tensor: cv2.imread(img_path)->  MatLike
        model_path
    """
    
    priors = define_img_size(input_size)
    
    if save_path is None:
        pass
    else:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    image_ori = img_tensor
    
    
    interpreter = MNN.Interpreter(model_path)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    
    image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, tuple(input_size))
    image = image.astype(np.float32)
    image = (image - image_mean) / image_std
    image = image.transpose((2, 0, 1))
    
    image = np.expand_dims(image, axis=0)
    if image.shape != (1, 3, input_size[1], input_size[0]):
        print(f"错误：图像形状 {image.shape} 不符合预期 (1, 3, {input_size[1]}, {input_size[0]})")
    
    tmp_input = MNN.Tensor(
        (1, 3, input_size[1], input_size[0]),
        MNN.Halide_Type_Float,
        image,
        MNN.Tensor_DimensionType_Caffe
    )
    input_tensor.copyFrom(tmp_input)
    
    interpreter.runSession(session)
    
    scores = interpreter.getSessionOutput(session, "scores").getData()
    boxes = interpreter.getSessionOutput(session, "boxes").getData()
    
    boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
    scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
    
    boxes = box_utils.convert_locations_to_boxes(
        boxes, priors, center_variance, size_variance
    )
    boxes = box_utils.center_form_to_corner_form(boxes)
    boxes, labels, probs = predict(
        image_ori.shape[1], image_ori.shape[0], scores, boxes, threshold
    )
    
    return crop_save_boxes(image_ori, boxes, save_path) if return_tensor else Get_crop_Tensor(image_ori, boxes)



# test
def test():
    path = r"C:\Users\Admin\Desktop\work\Valle\Face-Detector\imgs"
    save_path = r"C:\Users\Admin\Desktop\work\Valle\Face-Detector\cut_pic"
    model_path = r"C:\Users\Admin\Desktop\work\Valle\Face-Detector\model\version-RFB\RFB-320.mnn"

    inference(path, save_path=save_path,model_path=model_path)


test()
