import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import MNN
import cv2
import time

current_dir = os.path.dirname(__file__)
face_pth = os.path.abspath(os.path.join(current_dir,"..","..","model","facenet_fp16.mnn"))
img = os.path.abspath(os.path.join(current_dir,"..","..","Dataset","face_dataset","Colin_Powell","Colin_Powell_0002.jpg"))

def extract_embedding(image_path, model_path, input_size=(125, 125)):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # 加载模型
    interpreter = MNN.Interpreter(model_path)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)

    img = cv2.imread(image_path)
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

embedding = extract_embedding(img, face_pth)
print(embedding.shape)
print(embedding)