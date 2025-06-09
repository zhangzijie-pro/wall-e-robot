import os
import numpy as np
import librosa
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class DatasetProcessor:
    def __init__(self, train_path, output_dir, sr=16000, n_mels=80, max_len=300):
        self.train_path = Path(train_path)
        self.output_dir = Path(output_dir)
        self.sr = sr
        self.n_mels = n_mels
        self.max_len = max_len
        self.speaker_to_id = {}
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
        log_mel = self._pad_feature(log_mel, self.max_len)
        return log_mel.T

    def assign_speaker_ids(self):
        """Assign a unique ID to each speaker across all DR1 to DR8 directories."""
        speaker_id = 0
        for dr_dir in sorted(self.train_path.glob("DR[1-8]")):
            if not dr_dir.is_dir():
                continue
            for speaker_dir in dr_dir.iterdir():
                if speaker_dir.is_dir() and speaker_dir.name not in self.speaker_to_id:
                    self.speaker_to_id[speaker_dir.name] = speaker_id
                    speaker_id += 1
        
        # Save speaker-to-ID mapping
        mapping_path = self.output_dir / "speaker_to_id.json"
        with open(mapping_path, 'w') as f:
            json.dump(self.speaker_to_id, f, indent=4)
        print(f"Saved speaker-to-ID mapping to {mapping_path} with {len(self.speaker_to_id)} speakers")

    def process_dataset(self):
        """Process all audio files in DR1 to DR8 and save mel-spectrogram features."""
        self.assign_speaker_ids()
        
        for dr_dir in sorted(self.train_path.glob("DR[1-8]")):
            if not dr_dir.is_dir():
                continue
            print(f"Processing {dr_dir.name}...")
            for speaker_dir in dr_dir.iterdir():
                if not speaker_dir.is_dir():
                    continue
                speaker_id = self.speaker_to_id[speaker_dir.name]
                speaker_output_dir = self.output_dir / f"speaker_{speaker_id}"
                speaker_output_dir.mkdir(exist_ok=True)
                
                for audio_file in speaker_dir.glob("*.wav"):
                    if audio_file.name == "merge_result.wav":
                        continue
                    try:
                        mel_features = self.extract_mel_feature(audio_file, self.sr, self.n_mels)
                        output_path = speaker_output_dir / f"{dr_dir.name}_{audio_file.stem}_mel.npy"
                        np.save(output_path, mel_features)
                        print(f"Processed {audio_file} -> {output_path}")
                    except Exception as e:
                        print(f"Error processing {audio_file}: {e}")
        
        print("Dataset processing completed.")

class SpeakerDataset(Dataset):
    def __init__(self, data_dir, speaker_to_id_path):
        self.data_dir = Path(data_dir)
        with open(speaker_to_id_path, 'r') as f:
            self.speaker_to_id = json.load(f)
        self.files = []
        self.labels = []
        
        # Collect all .npy files and their corresponding speaker IDs
        for speaker_dir in self.data_dir.glob("speaker_*"):
            speaker_id = int(speaker_dir.name.split('_')[1])
            for npy_file in speaker_dir.glob("*.npy"):
                self.files.append(npy_file)
                # Map file to speaker ID
                speaker_name = [name for name, sid in self.speaker_to_id.items() if sid == speaker_id][0]
                self.labels.append(speaker_id)
        
        print(f"Loaded {len(self.files)} samples from {len(self.speaker_to_id)} speakers")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load mel-spectrogram features
        mel_features = np.load(self.files[idx])
        # Convert to torch tensor
        mel_features = torch.from_numpy(mel_features).float()
        # Ensure shape is (channels, height, width) for CNN input if needed
        mel_features = mel_features # Add channel dimension: (time, n_mels)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel_features, label