import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import MNN
import cv2
import json
import time
from log import logger

current_dir = os.path.dirname(__file__)
# img = os.path.abspath(os.path.join(current_dir,"..","..","Dataset","face_dataset","Colin_Powell","Colin_Powell_0008.jpg")) # 


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

    interpreter.runSession(session)
    output = interpreter.getSessionOutput(session)

    embedding = np.array(output.getData(), dtype=np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding


def build_face_database(image_dir, model_path, input_size=(125, 125)):
    face_db = {}

    for person_name in os.listdir(image_dir):
        person_path = os.path.join(image_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        face_db[person_name] = []
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            try:
                embedding = extract_embedding(img_path, model_path, input_size)
                face_db[person_name].append(embedding.tolist())
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")

    return face_db

from scipy.spatial.distance import cosine

def recognize_face(img, model_path, face_db, input_size=(125, 125), threshold=0.5):
    """
    Args:
        img: img_path or img_tensor
        model_path: only support MNN model
        face_db: face_lib data
    """
    query_embedding = extract_embedding(model_path, img, input_size) # [128]

    best_match = None
    best_score = 1.0

    for name, embeddings in face_db.items():
        for emb in embeddings:
            score = cosine(query_embedding, np.array(emb))
            if score < best_score:
                best_score = score
                best_match = name

    if best_score < threshold:
        return best_match, best_score
    else:
        return None, best_score