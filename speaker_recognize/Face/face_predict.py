import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import MNN
import cv2
import time
from log import logger

current_dir = os.path.dirname(__file__)
face_pth = os.path.abspath(os.path.join(current_dir,"..","..","model","facenet_fp16.mnn"))
img = os.path.abspath(os.path.join(current_dir,"..","..","Dataset","face_dataset","Colin_Powell","Colin_Powell_0002.jpg"))

def extract_embedding(model_path,image=None,input_size=(125, 125),image_path=None):

    if model_path is None or os.path.exists(model_path):
        logger.error(f"model_path: {model_path} Error")
        return

    # 加载模型
    interpreter = MNN.Interpreter(model_path)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)

    if image is not None:
        img = image
    elif image_path is not None:
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return
        img = cv2.imread(image_path)
    else:
        logger.error("Either image_path or img_array must be provided.")
        return
    
    img = cv2.resize(img, input_size)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)

    tmp_input = MNN.Tensor((1, 3, *input_size), MNN.Halide_Type_Float, img, MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFrom(tmp_input)

    start_time = time.time()
    interpreter.runSession(session)
    output = interpreter.getSessionOutput(session)

    embedding = np.array(output.getData(), dtype=np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    
    print(f"推理时间: {round(time.time() - start_time, 4)} 秒")
    return embedding
