import os
import torch
import torch.nn as nn   
import torch.nn.functional as F
import librosa
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np

import torchaudio
import torchaudio.transforms as T

os.path.curdir = os.path.dirname(os.path.abspath(__file__))
voice_path = os.path.join(os.path.curdir, "voice")

class Pre_data:
    """
    预处理数据
    """
    def __init__(self, voice_path):
        self.root_voice_path = voice_path
        self.voice = []

    def m4a_to_wav(self):
        """
        Convert m4a files to wav format.
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

class DataSet:
    """
    数据集类
    特征提取
    """
    def __init__(self, voice_path):
        self.voice_path = voice_path
        self.pre_data = Pre_data(voice_path)
        self.pre_data.m4a_to_wav()
        self.voice = self.pre_data.voice
        self.mel_transfrom = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=40
        )


    def extract_mel_feature(self, file_path, sr=16000, n_mels=40):
        """
        提取特征
        """
        y, _ = librosa.load(file_path, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        log_mel = librosa.power_to_db(mel)
        return log_mel.T  # shape: (time_steps, n_mels)

    def load_dataset(self):
        label_map = {}
        features = []
        labels = []
        speaker_dirpath = os.listdir(self.voice_path)

        for idx,speaker in enumerate(speaker_dirpath):
            label_map[speaker] = idx
            speaker_path = os.path.join(self.voice_path, speaker)
            for file in os.listdir(speaker_path):
                if not file.endswith('.wav'):
                    continue
                file_path = os.path.join(speaker_path, file)
                feat = self.extract_mel_feature(file_path)
                features.append(feat)
                labels.append(idx)
        return features, labels, label_map

# 同一数据长度
def padd_features(features, max_len = 300):
    padded = []
    for feat in features:
        if feat.shape[0]>max_len:
            feat = feat[:max_len]
        else:
            pad_width = max_len-feat.shape[0]
            feat = np.pad(feat, ((0,pad_width),(0,0)))
        padded.append(torch.tensor(feat, dtype=torch.float))
    return torch.stack(padded)

class SpeakerNet(nn.Module):
    """
    声纹识别模型
    """
    def __init__(self, n_mel=40, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16,  kernel_size=(5,5), padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(32,128)
        self.fc2 = nn.Linear(128, num_classes) # 分类

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x) # 提取主要特征输出为4D张量
        x = x.view(x.size(0),-1)  # 展平为2D张量
        x = F.relu(self.fc1(x))
        return self.fc2(x)


dataset_ready = DataSet(voice_path)
X, y, label_map = dataset_ready.load_dataset()
X = padd_features(X)
y = torch.tensor(y, dtype=torch.long)

batch_size = 16
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = SpeakerNet(num_classes=len(label_map))
opitimer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

for epoch in range(100):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for x,y in dataloader:
        opitimer.zero_grad()
        output = model(x)
        loss = loss_func(output, y)
        loss.backward()
        opitimer.step()
        total_loss+=loss.item()
        _, predicted = torch.max(output, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        print(f"Accuracy: {100 * correct / total:.2f}%")

torch.save({
    'model_state_dict': model.state_dict(),
    'label_map': label_map
}, "speaker_label_model.pth")
torch.save(label_map, "label_map.pth")
torch.save(model.state_dict(), "speaker_model.pth")