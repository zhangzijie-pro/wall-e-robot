import os
import torch
import torch.nn.functional as F
import librosa
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchaudio.transforms as T
import random


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
    def __init__(self, voice_path):
        self.voice_path = voice_path
        self.pre_data = Data_Preprocessing(voice_path)
        self.pre_data.m4a_to_wav()
        self.voice = self.pre_data.voice
        self.data = {}  # 每类人的音频路径
        for speaker in os.listdir(voice_path):
            paths = [os.path.join(voice_path, speaker, f) for f in os.listdir(os.path.join(voice_path, speaker)) if f.endswith('.wav')]
            if len(paths) >= 2:  # 至少两条语音才考虑做 anchor/positive
                self.data[speaker] = paths
        self.speakers = list(self.data.keys())

    def __len__(self):
        return 100

    def _pad_feature(self, feat, max_len=300):
        if feat.shape[1] > max_len:
            feat = feat[:, :max_len]
        else:
            feat = np.pad(feat, ((0, 0), (0, max_len - feat.shape[1])), mode='constant')
        return feat.astype(np.float32)

    def extract_mel_feature(self, file_path, sr=16000, n_mels=80):
        y, _ = librosa.load(file_path, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        log_mel = librosa.power_to_db(mel)
        log_mel = self._pad_feature(log_mel)
        return log_mel.T

    def __getitem__(self, index):
        anchor_spk = random.choice(self.speakers)
        positive_spk = anchor_spk
        negative_spk = random.choice([s for s in self.speakers if s != anchor_spk])

        anchor, positive = random.sample(self.data[anchor_spk], 2)
        negative = random.choice(self.data[negative_spk])

        anchor_feat = self.extract_mel_feature(anchor)
        positive_feat = self.extract_mel_feature(positive)
        negative_feat = self.extract_mel_feature(negative)

        return (
            torch.tensor(anchor_feat),
            torch.tensor(positive_feat),
            torch.tensor(negative_feat),
        )