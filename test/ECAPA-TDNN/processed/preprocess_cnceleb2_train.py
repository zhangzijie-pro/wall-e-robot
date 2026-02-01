import os
import json
import random
import subprocess
from collections import defaultdict

import torch
import torchaudio
from tqdm import tqdm

# ========= 你需要改这里 =========
CN_ROOT = r"..\CN-Celeb_flac"              # 数据集根目录
OUT_DIR = r"..\processed\cn_celeb2"     # 输出目录
USE_DEV = True                         # 是否把 dev 也加入训练
VAL_SPK_RATIO = 0.1                    # 按说话人划分验证集比例
FFMPEG = "ffmpeg"                      # ffmpeg 在 PATH 就写 "ffmpeg"，否则写绝对路径
# =================================

TARGET_SR = 16000
N_MELS = 80
MIN_SEC = 1.0
SEED = 1234

random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)
FEAT_DIR = os.path.join(OUT_DIR, "fbank_pt")
FIXED_WAV_DIR = os.path.join(OUT_DIR, "fixed_wav")  # flac读不了时转wav
os.makedirs(FEAT_DIR, exist_ok=True)
os.makedirs(FIXED_WAV_DIR, exist_ok=True)

def try_load_audio(path):
    """
    尝试用 torchaudio 直接读（支持 flac/wav）
    返回 wav[T] float, sr
    """
    wav, sr = torchaudio.load(path)  # [C,T]
    wav = wav.mean(dim=0)            # mono [T]
    return wav, sr

def ffmpeg_to_wav(src_path, dst_path):
    cmd = [FFMPEG, "-y", "-i", src_path, "-ac", "1", "-ar", str(TARGET_SR), "-acodec", "pcm_s16le", dst_path]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode == 0, p.stdout

def load_audio_16k_mono(path):
    """
    优先 torchaudio 直接读；失败则 ffmpeg 转码成 wav 再读
    返回 wav[T] 16k mono
    """
    try:
        wav, sr = try_load_audio(path)
    except Exception:
        # 转码
        base = os.path.splitext(os.path.basename(path))[0]
        dst = os.path.join(FIXED_WAV_DIR, base + ".wav")
        ok, log = ffmpeg_to_wav(path, dst)
        if not ok:
            raise RuntimeError(f"ffmpeg failed: {path}\n{log}")
        wav, sr = try_load_audio(dst)

    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.contiguous()

def wav_to_fbank(wav_16k):
    wav_16k = wav_16k.unsqueeze(0)  # [1,T]
    feat = torchaudio.compliance.kaldi.fbank(
        wav_16k,
        sample_frequency=TARGET_SR,
        num_mel_bins=N_MELS,
        frame_length=25,
        frame_shift=10,
        use_energy=False,
        window_type="povey",
        dither=0.0
    )  # [T_frames, 80]
    return feat

def scan_split_dir(split_dir):
    """
    split_dir: CN_ROOT/data 或 CN_ROOT/dev
    返回 spk2files: dict{spk: [paths]}
    """
    spk2files = defaultdict(list)
    for spk in os.listdir(split_dir):
        spk_path = os.path.join(split_dir, spk)
        if not os.path.isdir(spk_path):
            continue
        for fn in os.listdir(spk_path):
            if fn.lower().endswith(".flac") or fn.lower().endswith(".wav"):
                spk2files[spk].append(os.path.join(spk_path, fn))
    return spk2files

def main():
    # 1) 扫描 data/dev
    data_dir = os.path.join(CN_ROOT, "data")
    dev_dir  = os.path.join(CN_ROOT, "dev")

    spk2files = scan_split_dir(data_dir)
    if USE_DEV and os.path.isdir(dev_dir):
        spk2files_dev = scan_split_dir(dev_dir)
        for spk, files in spk2files_dev.items():
            spk2files[spk].extend(files)

    spks = sorted([s for s in spk2files.keys() if len(spk2files[s]) > 0])
    print("speakers:", len(spks))

    # 2) 按说话人划分 train/val
    spks_shuf = spks[:]
    random.shuffle(spks_shuf)
    n_val = max(1, int(len(spks_shuf) * VAL_SPK_RATIO))
    val_spks = set(spks_shuf[:n_val])

    spk2id = {spk: i for i, spk in enumerate(spks)}

    train_list_path = os.path.join(OUT_DIR, "train_fbank_list.txt")
    val_list_path   = os.path.join(OUT_DIR, "val_fbank_list.txt")

    ok, bad = 0, 0
    with open(train_list_path, "w", encoding="utf-8") as ftrain, open(val_list_path, "w", encoding="utf-8") as fval:
        for spk in tqdm(spks, desc="Extract fbank"):
            label = spk2id[spk]
            is_val = spk in val_spks
            out_f = fval if is_val else ftrain

            for ap in spk2files[spk]:
                try:
                    wav = load_audio_16k_mono(ap)
                    if wav.numel() < int(TARGET_SR * MIN_SEC):
                        bad += 1
                        continue
                    feat = wav_to_fbank(wav)  # [T,80]
                    base = os.path.splitext(os.path.basename(ap))[0]
                    feat_path = os.path.join(FEAT_DIR, f"{spk}__{base}.pt")
                    torch.save(feat, feat_path)

                    feat_path_norm = feat_path.replace("\\", "/")
                    out_f.write(f"{label} {feat_path_norm}\n")


                    ok += 1
                except Exception:
                    bad += 1
                    continue

    with open(os.path.join(OUT_DIR, "spk2id.json"), "w", encoding="utf-8") as f:
        json.dump(spk2id, f, ensure_ascii=False, indent=2)

    print("Done!")
    print("ok:", ok, "bad:", bad)
    print("train_list:", train_list_path)
    print("val_list:", val_list_path)
    print("feat_dir:", FEAT_DIR)

if __name__ == "__main__":
    main()
