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
import torchaudio.transforms as T
from log import logger
import config

current_dir = os.path.dirname(__file__)
voice_path = os.path.abspath(os.path.join(current_dir ,'..','..', 'Dataset','voice'))

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
        统一数据长度
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
        return 1000

    def __getitem__(self, idx):
        anchor_spk = random.choice(self.speakers)
        negative_spk = random.choice([s for s in self.speakers if s != anchor_spk])

        anchor_path, positive_path = random.sample(self.data[anchor_spk], 2)
        negative_path = random.choice(self.data[negative_spk])

        anchor = self.extract_mel_feature(anchor_path)
        positive = self.extract_mel_feature(positive_path)
        negative = self.extract_mel_feature(negative_path)

        anchor = self._pad_feature(anchor)
        positive = self._pad_feature(positive)
        negative = self._pad_feature(negative)

        return torch.tensor(anchor), torch.tensor(positive), torch.tensor(negative)
 

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
        return embedding

class StatsPool(nn.Module):
    """统计池化层"""
    def forward(self, x):
        # x: [B, C, T, F] -> reshape to [B, C, T*F]
        x = x.view(x.size(0), x.size(1), -1)
        mean = x.mean(dim=2)
        std = x.std(dim=2)
        return torch.cat((mean, std), dim=1)  # 输出: [B, 2C]

class TrckNet(nn.Module):
    """
    改进版 ResNet 声纹识别模型，带统计池化
    """
    def __init__(self, embeding_num=128):
        super(TrckNet, self).__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Identity()  # 去掉原来的全连接层

        self.pool = StatsPool()
        self.bn = nn.BatchNorm1d(512 * 2)
        self.fc = nn.Linear(512 * 2, embeding_num)

    def forward(self, x):
        # 输入: [B, 1, T, F]
        features = self.backbone(x)               # 输出: [B, 512, H, W]
        pooled = self.pool(features)              # 输出: [B, 1024]
        pooled = self.bn(pooled)
        emb = self.fc(pooled)                     # 输出: [B, embeding_num]
        return F.normalize(emb, p=2, dim=1)       # 单位化

def model_train(
    resnet_model=1,
    lr=0.001,
    batch_size = 16, 
    epochs = 10,
    device = "cpu"
    ):
    """
    Train Model
    Args:
        resnet_model: 1: embedding model / 0:classify model
        lr: optimizer learn rate
        batch_size: default=16
        epochs: training num
        device: "cpu" or gpu_num
        
    Example of device in Config:
    .. code-block:: python
        config = Config()
        device = config.device()  # get train device
        model_train(device=device)
    """
    config_time = config.Config()
    if resnet_model:
        model =TrckNet()
    else:
        model = SpeakerNet()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=1.0)

    
    dataset = TripDataSet(voice_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    logger.info("Total training samples:", len(dataset))
    logger.info(f"{len(dataset)/batch_size}/epoch")
    # print("Total training samples:", len(dataset))
    # print(f"{len(dataset)/batch_size}/epoch")


    for epoch in range(epochs):
        model.to(device).train()
        total_loss = 0
        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            anchor = anchor.to(torch.device(device)).unsqueeze(1)
            positive = positive.to(torch.device(device)).unsqueeze(1)
            negative = negative.to(torch.device(device)).unsqueeze(1)
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
    # print(f"🟢 spend {hours}hours-{minutes}minutes-{sec}second")
    logger.info(f"🟢 spend {hours}hours-{minutes}minutes-{sec}second")
    torch.save(model.state_dict(), 'tvector_model.pth')


model_train(epochs=100)