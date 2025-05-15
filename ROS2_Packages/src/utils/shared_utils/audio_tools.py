import numpy as np
import wave
import io

def pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1, sampwidth: int = 2) -> bytes:
    """
    pcm -> wav
    """
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return wav_io.getvalue()

def normalize_audio(audio: np.ndarray, target_level: float = 0.9) -> np.ndarray:
    """
    将音频数据归一化到指定振幅水平。
    """
    max_val = np.max(np.abs(audio))
    if max_val == 0:
        return audio
    return audio * (target_level / max_val)

def bytes_to_np(pcm_data: bytes, dtype=np.int16) -> np.ndarray:
    """
    pcm byte -> numpy array
    """
    return np.frombuffer(pcm_data, dtype=dtype)

def np_to_bytes(audio_np: np.ndarray, dtype=np.int16) -> bytes:
    """
    numpy array -> pcm byte
    """
    return audio_np.astype(dtype).tobytes()
