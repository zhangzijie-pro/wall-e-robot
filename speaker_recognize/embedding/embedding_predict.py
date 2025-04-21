import torch
import torchaudio
import librosa
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from embedding import TrckNet
import os

def extract_mel(file_path, sr=16000, n_mels=128, n_fft=512, hop_length=160, win_length=400):
    y, _ = librosa.load(file_path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                         n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    log_mel = librosa.power_to_db(mel)
    return log_mel

def preprocess(file_path):
    mel = extract_mel(file_path)  # [n_mels, time]
    mel = torch.tensor(mel).unsqueeze(0)  # [1, n_mels, T]
    mel = torch.nn.functional.interpolate(mel.unsqueeze(0), size=(128, 128), mode='bilinear')  # [1, 1, 128, 128]
    return mel.squeeze(0)

def cosine_similarity(a, b):
    return F.cosine_similarity(a, b).item()

# 加载训练好的模型
model = TrckNet(embeding_num=128)
model_path = r"C:\Users\lenovo\Desktop\python\tvector_model.pth"
model.load_state_dict(torch.load(model_path, map_location='cpu'))  # 替换为你的模型路径
model.eval()

x1_path = r"C:\Users\lenovo\Desktop\python\音频信号处理\voice\l\004.wav"
x2_path = r"C:\Users\lenovo\Desktop\python\音频信号处理\1.wav"

# 提取两个语音的特征
x1 = preprocess(x1_path)
x2 = preprocess(x2_path)

with torch.no_grad():
    emb1 = model(x1.unsqueeze(0))  # [1, 128]
    emb2 = model(x2.unsqueeze(0))

sim = cosine_similarity(emb1, emb2)
print(f"相似度: {sim:.4f}")
if sim < 0.8:
    print("⚠️ 此人未注册")
else:
    print(f"✅")
    


