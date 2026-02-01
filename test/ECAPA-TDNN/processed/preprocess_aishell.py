import os
import sys
import json
import math
import random
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
from tqdm import tqdm

# 读音频：优先 soundfile（严格），其次 torchaudio（更兼容）
import soundfile as sf
import torch
import torchaudio


# =========================
# 配置区：你只需要改这里
# =========================
DATA_ROOT = r"..\data_aishell\wav\train"   # AISHELL 解压后的 wav 根目录（Windows路径）
OUT_DIR   = r"..\processed\aishell"  # 输出目录（特征、列表等）

TARGET_SR = 16000
N_MELS    = 80
CHUNK_SEC = 2.0        # 训练时随机截取的时长（秒）
MIN_SEC   = 1.0        # 小于这个时长的音频直接丢弃
MAX_SEC   = 20.0       # 太长的也可以丢弃或截断（这里默认保留，提特征时截取）
VAL_SPK_RATIO = 0.1    # 按“说话人”切 val 的比例（推荐）

SEED = 1234

# 可选：如果你的 wav 很多不能读，开启自动修复（需要 ffmpeg）
ENABLE_FFMPEG_FIX = True
FFMPEG_PATH = "ffmpeg"   # 如果 PATH 没配置，改成 r"C:\ffmpeg\bin\ffmpeg.exe"


# =========================
# 工具函数
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def run_cmd(cmd: List[str]) -> Tuple[int, str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return p.returncode, p.stdout
    except Exception as e:
        return -1, str(e)

def is_silence(x: np.ndarray, thr_rms: float = 1e-4) -> bool:
    # x: float32 in [-1, 1]
    if x.size == 0:
        return True
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    return rms < thr_rms

def safe_read_audio(path: str) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    """
    返回 (audio_float32_mono, sr, err)
    audio: np.float32, range approx [-1,1]
    """
    # 1) soundfile 严格读
    try:
        audio, sr = sf.read(path, always_2d=False)
        # audio 可能是 float64 / int16 / (T, C)
        if audio is None:
            return None, None, "soundfile returned None"
        audio = np.asarray(audio)

        # 处理多声道
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        # 转 float32
        if audio.dtype.kind in ("i", "u"):
            # int16/int32 -> float
            maxv = np.iinfo(audio.dtype).max
            audio = audio.astype(np.float32) / float(maxv)
        else:
            audio = audio.astype(np.float32)

        # 去掉 NaN/Inf
        if not np.isfinite(audio).all():
            return None, None, "audio has NaN/Inf"
        return audio, int(sr), None
    except Exception as e:
        # 2) soundfile 失败，尝试 torchaudio 更兼容
        try:
            wav, sr = torchaudio.load(path)  # [C,T]
            wav = wav.mean(dim=0)            # mono [T]
            audio = wav.numpy().astype(np.float32)
            if not np.isfinite(audio).all():
                return None, None, "torchaudio audio has NaN/Inf"
            return audio, int(sr), None
        except Exception as e2:
            return None, None, f"read failed: soundfile={e} | torchaudio={e2}"

def ffmpeg_fix_to_wav(src_path: str, dst_path: str, target_sr: int = 16000) -> Tuple[bool, str]:
    """
    用 ffmpeg 强制转成标准 PCM16 wav，修复头/编码问题。
    """
    cmd = [
        FFMPEG_PATH, "-y",
        "-i", src_path,
        "-ac", "1",
        "-ar", str(target_sr),
        "-acodec", "pcm_s16le",
        dst_path
    ]
    code, out = run_cmd(cmd)
    return code == 0, out

def resample_np(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio
    # 用 torchaudio 做高质量重采样
    wav = torch.from_numpy(audio).float().unsqueeze(0)  # [1,T]
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
    y = resampler(wav).squeeze(0).numpy()
    return y.astype(np.float32)

def random_crop(audio: np.ndarray, sr: int, chunk_sec: float) -> np.ndarray:
    target_len = int(sr * chunk_sec)
    if audio.shape[0] == target_len:
        return audio
    if audio.shape[0] < target_len:
        # 不足则循环填充（比补零更不容易让模型学坏）
        reps = int(math.ceil(target_len / audio.shape[0]))
        tiled = np.tile(audio, reps)[:target_len]
        return tiled.astype(np.float32)
    # 够长则随机截取
    start = random.randint(0, audio.shape[0] - target_len)
    return audio[start:start + target_len].astype(np.float32)

def fbank_kaldi(audio_16k: np.ndarray) -> torch.Tensor:
    """
    返回 [T, 80] 的 FBank
    """
    wav = torch.from_numpy(audio_16k).float().unsqueeze(0)  # [1,T]
    feat = torchaudio.compliance.kaldi.fbank(
        wav,
        sample_frequency=TARGET_SR,
        num_mel_bins=N_MELS,
        frame_length=25,
        frame_shift=10,
        use_energy=False,
        window_type="povey",
        dither=0.0
    )  # [T, 80]
    return feat

def scan_wavs(data_root: str) -> List[Tuple[str, str]]:
    """
    返回 [(spk, wav_path), ...]
    """
    items = []
    for spk in sorted(os.listdir(data_root)):
        spk_path = os.path.join(data_root, spk)
        if not os.path.isdir(spk_path):
            continue
        for fn in os.listdir(spk_path):
            if fn.lower().endswith(".wav"):
                items.append((spk, os.path.join(spk_path, fn)))
    return items


# =========================
# 主流程
# =========================
@dataclass
class CheckResult:
    spk: str
    path: str
    ok: bool
    reason: str
    sec: float
    sr: int
    rms: float
    fixed_path: Optional[str] = None


def check_and_optionally_fix(items: List[Tuple[str, str]], fixed_dir: str) -> List[CheckResult]:
    ensure_dir(fixed_dir)
    results: List[CheckResult] = []

    for spk, p in tqdm(items, desc="检查音频有效性"):
        audio, sr, err = safe_read_audio(p)
        if audio is None:
            # 读失败：尝试 ffmpeg 修复
            if ENABLE_FFMPEG_FIX:
                outp = os.path.join(fixed_dir, f"{spk}__{os.path.basename(p)}")
                ok, log = ffmpeg_fix_to_wav(p, outp, TARGET_SR)
                if ok:
                    audio2, sr2, err2 = safe_read_audio(outp)
                    if audio2 is not None:
                        sec = audio2.shape[0] / float(sr2)
                        rms = float(np.sqrt(np.mean(audio2 * audio2) + 1e-12))
                        # 长度/静音检查
                        if sec < MIN_SEC:
                            results.append(CheckResult(spk, p, False, "too_short_after_fix", sec, sr2, rms, outp))
                        elif sec > MAX_SEC:
                            results.append(CheckResult(spk, p, True, "ok_after_fix_long", sec, sr2, rms, outp))
                        elif is_silence(audio2):
                            results.append(CheckResult(spk, p, False, "silence_after_fix", sec, sr2, rms, outp))
                        else:
                            results.append(CheckResult(spk, p, True, "ok_after_fix", sec, sr2, rms, outp))
                    else:
                        results.append(CheckResult(spk, p, False, f"read_failed_even_after_fix: {err2}", 0.0, 0, 0.0, outp))
                else:
                    results.append(CheckResult(spk, p, False, f"read_failed: {err}", 0.0, 0, 0.0, None))
            else:
                results.append(CheckResult(spk, p, False, f"read_failed: {err}", 0.0, 0, 0.0, None))
            continue

        sec = audio.shape[0] / float(sr)
        rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))

        if sec < MIN_SEC:
            results.append(CheckResult(spk, p, False, "too_short", sec, sr, rms, None))
        elif is_silence(audio):
            results.append(CheckResult(spk, p, False, "silence", sec, sr, rms, None))
        else:
            # ok
            results.append(CheckResult(spk, p, True, "ok", sec, sr, rms, None))

    return results


def split_by_speaker(valid_items: List[Tuple[str, str]], val_ratio: float) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], Dict[str, int]]:
    spks = sorted(list({spk for spk, _ in valid_items}))
    random.shuffle(spks)
    n_val = max(1, int(len(spks) * val_ratio))
    val_spks = set(spks[:n_val])
    train, val = [], []
    for spk, p in valid_items:
        (val if spk in val_spks else train).append((spk, p))

    # map speaker -> int id
    all_spks = sorted(list({spk for spk, _ in valid_items}))
    spk2id = {spk: i for i, spk in enumerate(all_spks)}
    return train, val, spk2id


def extract_and_save_features(pairs: List[Tuple[str, str]], spk2id: Dict[str, int], feat_dir: str, index_path: str):
    ensure_dir(feat_dir)

    with open(index_path, "w", encoding="utf-8") as f:
        for spk, path in tqdm(pairs, desc=f"提取FBank -> {os.path.basename(index_path)}"):
            label = spk2id[spk]

            # 读音频（如果之前修复过，这里传的是修复后的路径）
            audio, sr, err = safe_read_audio(path)
            if audio is None:
                continue

            # 重采样到16k
            audio = resample_np(audio, sr, TARGET_SR)

            # 随机截取固定长度（训练用）；如果你想全长提特征，可在这里改掉
            audio = random_crop(audio, TARGET_SR, CHUNK_SEC)

            # 提特征 [T,80]
            feat = fbank_kaldi(audio)  # torch [T, 80]

            # 保存特征为 .pt
            # 文件名：spk__原wav名.pt
            base = os.path.splitext(os.path.basename(path))[0]
            feat_path = os.path.join(feat_dir, f"{spk}__{base}.pt")
            torch.save(feat, feat_path)

            feat_path = feat_path.replace("\\", "/")
            f.write(f"{label} {feat_path}\n")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ensure_dir(OUT_DIR)
    fixed_dir = os.path.join(OUT_DIR, "fixed_wav")
    feat_dir  = os.path.join(OUT_DIR, "fbank_pt")

    # 1) 扫描 wav
    items = scan_wavs(DATA_ROOT)
    print(f"扫描到 wav 数量: {len(items)}")

    # 2) 检查有效性 + 可选修复
    results = check_and_optionally_fix(items, fixed_dir=fixed_dir)

    # 输出统计报告
    ok = [r for r in results if r.ok]
    bad = [r for r in results if not r.ok]
    print(f"有效音频: {len(ok)} | 无效音频: {len(bad)}")

    report_path = os.path.join(OUT_DIR, "check_report.jsonl")
    with open(report_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
    print("检测报告已保存:", report_path)

    # 3) 组装有效数据（优先用修复后的路径）
    valid_pairs: List[Tuple[str, str]] = []
    for r in ok:
        use_path = r.fixed_path if r.fixed_path else r.path
        valid_pairs.append((r.spk, use_path))

    if len(valid_pairs) == 0:
        print("没有可用的音频。请先看 check_report.jsonl 里失败原因。")
        sys.exit(1)

    # 4) 按说话人切分 train/val
    train_pairs, val_pairs, spk2id = split_by_speaker(valid_pairs, VAL_SPK_RATIO)
    print(f"说话人数: {len(spk2id)} | 训练条目: {len(train_pairs)} | 验证条目: {len(val_pairs)}")

    with open(os.path.join(OUT_DIR, "spk2id.json"), "w", encoding="utf-8") as f:
        json.dump(spk2id, f, ensure_ascii=False, indent=2)

    # 5) 提取并保存特征 + 生成索引列表
    train_index = os.path.join(OUT_DIR, "train_fbank_list.txt")
    val_index   = os.path.join(OUT_DIR, "val_fbank_list.txt")

    extract_and_save_features(train_pairs, spk2id, feat_dir, train_index)
    extract_and_save_features(val_pairs, spk2id, feat_dir, val_index)

    print("\n全部完成！")
    print("训练索引:", train_index)
    print("验证索引:", val_index)
    print("特征目录:", feat_dir)
    print("说话人映射:", os.path.join(OUT_DIR, "spk2id.json"))
    print("无效/修复报告:", report_path)


if __name__ == "__main__":
    main()
