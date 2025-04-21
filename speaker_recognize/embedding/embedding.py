import os
import random
import torch
import torch.nn as nn   
import torch.nn.functional as F
import librosa
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import torchvision.models as models
import logging
import torchaudio
import torchaudio.transforms as T
import time

current_dir = os.path.dirname(__file__)
voice_path = os.path.abspath(os.path.join(current_dir ,'..','..', 'voice'))

class Data_Preprocessing:
    """
    预处理数据
    """
    def __init__(self, voice_path):
        self.root_voice_path = voice_path
        self.voice = []

    def m4a_to_wav(self):
        """
        转换m4a -> wav
        """
        import pydub

        for dir in os.listdir(self.root_voice_path):
            dir_path = os.path.join(self.root_voice_path, dir)
            
            if not os.path.isdir(dir_path):
                continue
            file_names = []

            m4a_files = [f for f in os.listdir(dir_path) if f.endswith('.m4a')]
            for i,file in enumerate(m4a_files):
                new_name = f"{i:03d}.wav"
                audio = pydub.AudioSegment.from_file(os.path.join(dir_path, file), format="m4a")
                audio.export(os.path.join(dir_path, new_name), format="wav")
                os.remove(os.path.join(dir_path, file))  # 删除原始文件
                file_names.append(new_name)
            
            self.voice.append(file_names)

class TripDataSet(Dataset):
    """
    数据集
    特征提取
    """
    def __init__(self, voice_path):
        self.voice_path = voice_path
        self.pre_data = Data_Preprocessing(voice_path)
        self.pre_data.m4a_to_wav()
        self.voice = self.pre_data.voice
        self.data = {}  # 每类人的音频路径
        for speaker in os.listdir(voice_path):
            paths = [os.path.join(voice_path, speaker, f) for f in os.listdir(os.path.join(voice_path, speaker)) if f.endswith('.wav')]
            self.data[speaker] = paths
        self.speakers = list(self.data.keys())

    def _pad_feature(self, feat, max_len=300):
        """
        统一长度
        """
        if feat.shape[0] > max_len:
            feat = feat[:max_len]
        else:
            feat = np.pad(feat, ((0, max_len - feat.shape[0]), (0, 0)))
        return feat.astype(np.float32)

    def extract_mel_feature(self, file_path, sr=16000, n_mels=40):
        """
        提取特征
        """
        y, _ = librosa.load(file_path, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        log_mel = librosa.power_to_db(mel)
        return log_mel.T  # shape: (time_steps, n_mels)


    def __len__(self):
        return 1000  # 随便一个大数

    def __getitem__(self, idx):
        anchor_spk = random.choice(self.speakers)
        negative_spk = random.choice([s for s in self.speakers if s != anchor_spk])

        anchor_path, positive_path = random.sample(self.data[anchor_spk], 2)
        negative_path = random.choice(self.data[negative_spk])

        anchor = self.extract_mel_feature(anchor_path)
        positive = self.extract_mel_feature(positive_path)
        negative = self.extract_mel_feature(negative_path)

        # pad 成相同长度
        anchor = self._pad_feature(anchor)
        positive = self._pad_feature(positive)
        negative = self._pad_feature(negative)

        return torch.tensor(anchor), torch.tensor(positive), torch.tensor(negative)

class Config:
    def __init__(self):
        self.__start_time = time.time()
        self.hours = 0
        self.minute = 0
        self.second = 0

    def get_spend_time(self):
        """ 
        获取花费时间
        """
        self.second = int(time.time().self.start_time)+1
        
        if self.second > 60:
            self.minute = self.second/60
            self.second = self.second-(self.minute*60)
            if self.minute>60:
                self.hours = self.minute/60
                self.minute = self.minute-(self.hours*60)

        return (self.hours,self.minute,self.second)

class SpeakerNet(nn.Module):
    """
    声纹识别分类模型
    """
    def __init__(self, n_mel=40, embeding_num=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16,  kernel_size=(5,5), padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(32,embeding_num)    #  向量

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x) # 提取主要特征输出为4D张量
        x = x.view(x.size(0),-1)  # 展平为2D张量
        embedding = F.normalize(self.fc1(x), p=2, dim=1)  # (B, 128), L2归一化
        return 

class TrckNet(nn.Module):
    """
    声纹识别分类模型
    resnet
    """
    def __init__(self, embeding_num=128):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embeding_num)

    def forward(self, x):
        x = self.resnet(x)
        return F.normalize(x, p=2, dim=1)

def model_train(
    resnet_model=1,
    lr=0.001,
    batch_size = 16, 
    epochs = 100
    ):

    config_time = Config()
    if resnet_model:
        model =TrckNet()
    else:
        model = SpeakerNet()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=1.0)

    
    dataset = TripDataSet(voice_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Total training samples:", len(dataset))
    print(f"{len(dataset)/batch_size}/epoch")


    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            anchor = anchor.unsqueeze(1)
            positive = positive.unsqueeze(1)
            negative = negative.unsqueeze(1)

            emb_anchor = model(anchor)
            emb_positive = model(positive)
            emb_negative = model(negative)

            loss = criterion(emb_anchor, emb_positive, emb_negative)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    (hours, minutes, sec) = config_time.get_spend_time()
    print(f"spend {hours} hours {minutes} min {sec} s")
    torch.save(model.state_dict(), 'tvector_model.pth')


model_train()