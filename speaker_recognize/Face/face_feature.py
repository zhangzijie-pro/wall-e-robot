import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import os
import random

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
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])
        self.data = []
        self.labels = []
        self.class_to_imgs = {}
        for label, person in enumerate(os.listdir(dataset_path)):
            person_path = os.path.join(dataset_path, person)
            if not os.path.isdir(person_path): continue

            imgs = [os.path.join(person_path, f) for f in os.listdir(person_path)]
            self.class_to_imgs[label]=imgs
            self.data.extend(imgs)
            self.labels.extend([label] * len(imgs))
        
        self.transform = transform
        self.labels = np.array(self.labels)

    def __getitem__(self, index):
        anchor_path = self.data[index]
        anchor_label = self.labels[index]

        positive_path = random.choice([img for img in self.class_to_imgs[anchor_label] if img != anchor_path])
        negative_label = random.choice([l for l in self.class_to_imgs if l != anchor_label])
        negative_path = random.choice(self.class_to_imgs[negative_label])

        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative
    
    def __len__(self):
        return len(self.data)

def conv_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6()
    )

def conv_dw(iup, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(iup, iup, 3, stride, 1, groups=iup, bias=False),
        nn.BatchNorm2d(iup),
        nn.ReLU6(),

        nn.Conv2d(iup, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6()
    )

class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size = 1000):
        super(MobileFaceNet, self).__init__()
        self.stage1 = nn.Sequential(
            # 125*125 3 -> 63*63 32
            conv_bn(3, 32, 2),
            conv_dw(32,64,1),

            # 63*63 64 -> 32*32 128
            conv_dw(64, 128, 2),
            conv_dw(128,128, 1),
            
            # 32*32 128 -> 16*16 256
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1)
        )
        self.stage2 = nn.Sequential(
            conv_dw(256, 512, 2),    # 16x16 -> 8x8
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
        )
        self.stage3 = nn.Sequential(
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, embedding_size)
    
    def forward(self,x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # [B, 1024]
        x = self.fc(x)
        return x

def mobilenet():
    return MobileFaceNet(embedding_size=1024)  # 输出特征维度为1024


class Face_Detect(nn.Module):
    def __init__(self, dropout_keep_prob=0.5, embedding_size=128, num_classes=None, mode="train"):
        super(Face_Detect, self).__init__()
        self.backbone = mobilenet()
        flat_shape = 1024
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.Dropout = nn.Dropout2d(1-dropout_keep_prob)
        self.Bottleneck = nn.Linear(flat_shape, embedding_size, bias=False)
        self.last_bn = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)

        if mode=="train" and num_classes is not None:
            self.classifier = nn.Linear(embedding_size, num_classes)
        else:
            self.classifier = None

    def forward(self,x):
        x = self.backbone(x) # [B ,1024] 
        x = x.view(x.size(0), -1) #[ B,1024]
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        x = self.last_bn(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
    def forward_classifier(self,x):
        x = self.classifier(x)
        return x
    
class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss,self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
    

def train_tripletloss(model, dataloader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for anchor, positive, negative in dataloader:
        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)

        loss = loss_fn(anchor_emb, positive_emb, negative_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)
        

dataset = FaceDataset(face_dataset)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = Face_Detect( embedding_size=128)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = TripletLoss(margin=0.2)

for epoch in range(10):
    loss = train_tripletloss(model, dataloader, optimizer, loss_fn)
    print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

torch.save(model.state_dict(), 'face_model.pth')
