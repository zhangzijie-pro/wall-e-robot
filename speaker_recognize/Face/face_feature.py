# speaker_recognition/Face/face_feature.py

import torch
import cv2
from torch.utils.data import Dataset, DataLoader

import os

current_dir = os.path.dirname(__file__)
face_dataset = os.path.abspath(os.path.join(current_dir,"..","face_pic"))

class dataset(Dataset):
    pass

