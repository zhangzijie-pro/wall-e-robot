import random
import numpy as np
import torch
import torchaudio

TARGET_SR = 16000

def load_wav_mono(path: str, target_sr: int = TARGET_SR) -> torch.Tensor:
    """
    返回: wav [T] float32, 16kHz, mono
    """
    wav, sr = torchaudio.load(path)  # [C,T]
    wav = wav.mean(dim=0)            # [T]
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.contiguous()

def random_crop_or_repeat(wav: torch.Tensor, length: int) -> torch.Tensor:
    """
    训练时常用：随机截取固定长度，不足则循环拼接
    """
    T = wav.numel()
    if T == length:
        return wav
    if T < length:
        reps = int(np.ceil(length / T))
        wav = wav.repeat(reps)[:length]
        return wav
    start = random.randint(0, T - length)
    return wav[start:start+length]

def wav_to_fbank(wav_16k: torch.Tensor, n_mels: int = 80) -> torch.Tensor:
    """
    输入 wav: [T], 16kHz
    输出 feat: [T_frames, 80]
    """
    wav_16k = wav_16k.unsqueeze(0)  # [1,T]
    feat = torchaudio.compliance.kaldi.fbank(
        wav_16k,
        sample_frequency=TARGET_SR,
        num_mel_bins=n_mels,
        frame_length=25,
        frame_shift=10,
        use_energy=False,
        window_type="povey",
        dither=0.0
    )
    return feat
