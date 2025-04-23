import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os

current_dir = os.path.dirname(__file__)
face_dataset = os.path.abspath(os.path.join(current_dir,"..","face_pic"))

class Data_Preprocessing:
    """
    图像文件预处理  面部特征提取模型
    """
    def __init__(self,dataset_path, img_size=(125,125)):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.image_Preprocess()
    
    def image_Preprocess(self):
        for dir in os.listdir(self.dataset_path):

            dir_path = os.path.join(self.dataset_path, dir)
            if not os.path.isdir(dir_path):
                continue
            img_files = [f for f in os.listdir(dir_path)]
            for i, file in enumerate(img_files):
                img_path = os.path.join(dir_path, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                resized = cv2.resize(img, self.img_size)
                new_name = f"{i:03d}.jpg"
                new_path = os.path.join(dir_path, new_name)
                cv2.imwrite(new_path, resized)

class FaceDataset(Dataset):
    def __init__(self, dataset_path, img_size=(125, 125)):
        self.data = []
        self.label_map = {}
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])
        label_id = 0
        for person in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person)
            if not os.path.isdir(person_path): continue
            if person not in self.label_map:
                self.label_map[person] = label_id
                label_id += 1
            for img_file in os.listdir(person_path):
                self.data.append((os.path.join(person_path, img_file), self.label_map[person]))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return image, label
