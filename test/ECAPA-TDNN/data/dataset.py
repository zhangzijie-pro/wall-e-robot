import os
import random
import torch
from torch.utils.data import Dataset


class FbankPtDataset(Dataset):
    def __init__(self, list_path: str):
        self.list_path = list_path
        self.max_frames = 200      # 固定帧长：200帧≈2秒（推荐 200~300）
        self.random_crop = True    # 训练随机裁剪
        self.base_dir = os.path.dirname(os.path.abspath(list_path))
        self.items = []
        with open(list_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                lab, p = line.split(maxsplit=1)
                p = p.strip().strip('"').strip("'")
                self.items.append((int(lab), p))

    def __len__(self):
        return len(self.items)

    def _try_candidates(self, p_raw: str):
        """
        生成一组候选路径，按顺序尝试 exists
        目的：兼容各种写错的前缀/相对路径/重复 processed 的情况
        """
        p_norm = p_raw.replace("\\", "/").strip()

        cands = []

        # 1) 如果本来就是绝对路径
        if os.path.isabs(p_norm):
            cands.append(os.path.abspath(p_norm))

        # 2) 相对路径：以 list 文件所在目录为基准
        cands.append(os.path.abspath(os.path.join(self.base_dir, p_norm)))

        # 3) 常见错误：路径里重复出现 processed/processed
        if "processed/processed/" in p_norm:
            p_fix = p_norm.replace("processed/processed/", "processed/")
            if os.path.isabs(p_fix):
                cands.append(os.path.abspath(p_fix))
            cands.append(os.path.abspath(os.path.join(self.base_dir, p_fix)))

        for prefix in [
            "processed/cn_celeb2/",
            "processed/aishell/",
            "processed_aishell/",
            "processed_cnceleb2/",
        ]:
            if p_norm.startswith(prefix):
                p_fix = p_norm[len(prefix):]
                cands.append(os.path.abspath(os.path.join(self.base_dir, p_fix)))

        fname = os.path.basename(p_norm)
        if fname:
            cands.append(os.path.abspath(os.path.join(self.base_dir, "fbank_pt", fname)))
            cands.append(os.path.abspath(os.path.join(self.base_dir, "..", "fbank_pt", fname)))

        uniq = []
        seen = set()
        for c in cands:
            c = os.path.abspath(c)
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq

    def __getitem__(self, idx):
        label, p_raw = self.items[idx]

        cands = self._try_candidates(p_raw)
        p_hit = None
        for c in cands:
            if os.path.exists(c):
                p_hit = c
                break

        if p_hit is None:
            # 把候选路径都打印出来，方便你定位到底错在哪
            msg = (
                f"Feature not found for raw path: {p_raw}\n"
                f"from list: {self.list_path}\n"
                f"base_dir: {self.base_dir}\n"
                f"candidates tried:\n  - " + "\n  - ".join(cands)
            )
            raise FileNotFoundError(msg)

        try:
            feat = torch.load(p_hit, map_location="cpu", weights_only=True)  # [T,80]
        except TypeError:
            feat = torch.load(p_hit, map_location="cpu")

        if feat.dim() != 2:
            raise ValueError(f"Bad feature shape at {p_hit}: {tuple(feat.shape)} (expect [T,80])")
        
        # 防止坏特征把训练炸掉
        if not torch.isfinite(feat).all():
            # 把坏样本直接替换成全 0（或抛异常）
            feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        mean = feat.mean(dim=0, keepdim=True)
        std = feat.std(dim=0, keepdim=True).clamp_min(1e-5)
        feat = (feat - mean) / std

        T = feat.size(0)
        L = self.max_frames

        if T > L:
            if self.random_crop:
                start = random.randint(0, T - L)
            else:
                start = 0
            feat = feat[start:start + L]
        elif T < L:
            pad = torch.zeros(L - T, feat.size(1), dtype=feat.dtype)
            feat = torch.cat([feat, pad], dim=0)

        feat = spec_augment(feat, p=0.5)

        return feat, label

def spec_augment(feat, time_mask=20, freq_mask=8, p=0.5):
    # feat: [T,80]
    if random.random() > p:
        return feat
    T, F = feat.size(0), feat.size(1)

    # time mask
    t = random.randint(0, time_mask)
    t0 = random.randint(0, max(0, T - t))
    feat[t0:t0+t, :] = 0

    # freq mask
    f = random.randint(0, freq_mask)
    f0 = random.randint(0, max(0, F - f))
    feat[:, f0:f0+f] = 0
    return feat


def pad_collate(batch):
    """
    batch: List[(feat[T,80], label)]
    输出:
      x: [B, T_max, 80]
      y: [B]
      lengths: [B]
    """
    feats, labels = zip(*batch)
    lengths = torch.tensor([f.size(0) for f in feats], dtype=torch.long)
    T_max = int(lengths.max().item())
    B = len(feats)

    x = torch.zeros(B, T_max, feats[0].size(1), dtype=feats[0].dtype)
    for i, f in enumerate(feats):
        x[i, : f.size(0)] = f

    y = torch.tensor(labels, dtype=torch.long)
    return x, y, lengths
