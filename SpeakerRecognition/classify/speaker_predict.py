import os
import torch
import torch.nn.functional as F
import librosa
import numpy as np
from embedding.embedding import SpeakerNet
os.path.curdir = os.path.dirname(os.path.abspath(__file__))
now_path = os.path.join(os.path.curdir)

def extract_mel_feature(file_path, sr=16000, n_mels=40, max_len=300):
    y, _ = librosa.load(file_path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel)
    log_mel = log_mel.T

    if log_mel.shape[0] > max_len:
        log_mel = log_mel[:max_len]
    else:
        pad_width = max_len - log_mel.shape[0]
        log_mel = np.pad(log_mel, ((0, pad_width), (0, 0)))

    return torch.tensor(log_mel, dtype=torch.float).unsqueeze(0)  # shape: (1, time, mel)

def predict(audio_path, model_path="speaker_model.pth", threshold=0.8):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    label_map = checkpoint['label_map']
    label_inv_map = {v: k for k, v in label_map.items()}

    model = SpeakerNet(num_classes=len(label_map))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    features = extract_mel_feature(audio_path)

    with torch.no_grad():
        output = model(features)
        probs = F.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        confidence = confidence.item()
        pred_idx = pred_idx.item()

        if confidence < threshold:
            print(f"识别结果：未知说话人（置信度 {confidence:.2f} < 阈值 {threshold}）")
        else:
            print(f"识别结果：说话人是 \"{label_inv_map[pred_idx]}\"（置信度 {confidence:.2f}）")

if __name__ == "__main__":
    model_path = os.path.join(now_path,"model/speaker_label_model.pth")
    test_file = os.path.join(now_path,"voice/l/005.wav")
    #test_file = r"C:\Users\lenovo\Desktop\python\音频信号处理\Guide\Voice\流行 u.wav"
    predict(test_file,model_path )