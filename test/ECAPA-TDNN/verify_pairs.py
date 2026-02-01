import os
import random
from collections import defaultdict

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from models.ecapa import ECAPA_TDNN

# t-SNE 可视化（需要 pip install scikit-learn）
try:
    from sklearn.manifold import TSNE
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


def _clean_path_str(p: str) -> str:
    # 去掉引号、常规空白、BOM
    p = p.strip().strip('"').strip("'").strip()
    p = p.lstrip("\ufeff")
    # 统一分隔符
    p = p.replace("\\", "/")
    return p


def read_list(list_path: str):
    """
    读取 list: 每行 "label path"
    并做路径纠错：
      - 相对路径以 list 所在目录为基准
      - 自动修复 Windows 常见的 processed/processed 重复前缀
    """
    items = []
    base_dir = os.path.dirname(os.path.abspath(list_path))

    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lab, p = line.split(maxsplit=1)
            p = _clean_path_str(p)

            # 相对路径 -> 以 list 所在目录为基准转绝对路径
            if not os.path.isabs(p):
                p = os.path.abspath(os.path.join(base_dir, p))
            else:
                p = os.path.abspath(p)

            # 规范化 + ✅关键修复：processed/processed -> processed
            p = os.path.normpath(p).replace("\\", "/")
            p = p.replace("/processed/processed/", "/processed/")

            items.append((int(lab), p))

    return items


def build_pairs(items, num_pos=2000, num_neg=2000, seed=1234):
    """
    items: [(label, feat_path)]
    返回 pairs: [(is_same(1/0), p1, p2)]
    """
    random.seed(seed)
    spk2paths = defaultdict(list)
    for lab, p in items:
        spk2paths[lab].append(p)

    spks = [s for s in spk2paths.keys() if len(spk2paths[s]) >= 2]
    all_spks = list(spk2paths.keys())

    pairs = []

    # 正对：同一说话人不同语句
    for _ in range(num_pos):
        spk = random.choice(spks)
        p1, p2 = random.sample(spk2paths[spk], 2)
        pairs.append((1, p1, p2))

    # 负对：不同说话人
    for _ in range(num_neg):
        s1, s2 = random.sample(all_spks, 2)
        p1 = random.choice(spk2paths[s1])
        p2 = random.choice(spk2paths[s2])
        pairs.append((0, p1, p2))

    random.shuffle(pairs)
    return pairs


@torch.no_grad()
def embed_from_fbank_pt(model, feat_path, device):
    feat_path = os.path.normpath(feat_path)

    # 如果路径太长/奇怪，先常规 exists；不存在就返回 None
    if not os.path.exists(feat_path):
        return None

    # 兼容新旧 torch.load
    try:
        feat = torch.load(feat_path, map_location="cpu", weights_only=True)  # [T,80]
    except TypeError:
        feat = torch.load(feat_path, map_location="cpu")
    except Exception:
        return None

    if not torch.is_tensor(feat) or feat.dim() != 2:
        return None

    x = feat.unsqueeze(0).to(device)  # [1,T,80]
    emb = model(x)                    # [1,192]（你模型里是否 normalize 取决于你的训练头，通常在训练头或外面做）
    return emb.squeeze(0).cpu()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    # 如果你训练时做了 embedding L2 normalize，这里点积=cos
    # 如果没有 normalize，你可以在这里先 normalize 再点积（先不强制改你的流程）
    return float(torch.sum(a * b).item())


def compute_eer(labels, scores):
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)

    P = sum(labels)
    N = len(labels) - P
    fa = N
    fr = 0

    best_diff = 1.0
    eer = 1.0
    best_th = None

    for th, lab in pairs:
        if lab == 1:
            fr += 1
        else:
            fa -= 1

        far = fa / max(1, N)
        frr = fr / max(1, P)
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            eer = (far + frr) / 2.0
            best_th = th

    return eer, best_th


def roc_points(labels, scores, num_th=200):
    mn, mx = min(scores), max(scores)
    ths = [mn + (mx - mn) * i / (num_th - 1) for i in range(num_th)]
    P = sum(labels)
    N = len(labels) - P

    tpr, fpr = [], []
    for th in ths:
        tp = sum(1 for l, s in zip(labels, scores) if l == 1 and s >= th)
        fp = sum(1 for l, s in zip(labels, scores) if l == 0 and s >= th)
        tpr.append(tp / max(1, P))
        fpr.append(fp / max(1, N))
    return fpr, tpr


def det_points(labels, scores, num_th=400):
    mn, mx = min(scores), max(scores)
    ths = [mn + (mx - mn) * i / (num_th - 1) for i in range(num_th)]
    P = sum(labels)
    N = len(labels) - P

    fars, frrs = [], []
    for th in ths:
        fa = sum(1 for l, s in zip(labels, scores) if l == 0 and s >= th)
        fr = sum(1 for l, s in zip(labels, scores) if l == 1 and s < th)
        fars.append(fa / max(1, N))
        frrs.append(fr / max(1, P))
    return fars, frrs


@torch.no_grad()
def collect_embeddings_for_tsne(model, items, device, max_spk=20, per_spk=25, seed=1234):
    random.seed(seed)
    spk2paths = defaultdict(list)
    for lab, p in items:
        spk2paths[lab].append(p)

    spks = [s for s in spk2paths.keys() if len(spk2paths[s]) >= 2]
    random.shuffle(spks)
    spks = spks[:max_spk]

    X_list, y_list = [], []
    for spk in spks:
        paths = spk2paths[spk][:]
        random.shuffle(paths)
        paths = paths[:per_spk]
        for p in paths:
            emb = embed_from_fbank_pt(model, p, device)
            if emb is None:
                continue
            X_list.append(emb.numpy())
            y_list.append(spk)

    if len(X_list) == 0:
        return None, None
    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.int64)


def recall_at_k(embeddings, labels, ks=(1, 5, 10)):
    sims = embeddings @ embeddings.t()
    sims.fill_diagonal_(-1e9)
    idx = torch.argsort(sims, dim=1, descending=True)

    M = sims.size(0)
    res = {}
    for k in ks:
        hit = 0
        for i in range(M):
            topk = idx[i, :k]
            if (labels[topk] == labels[i]).any().item():
                hit += 1
        res[k] = hit / M
    return res


def main():
    VAL_LIST = r"processed/cn_celeb2/val_fbank_list.txt"
    CKPT = r"outputs/best.pt"

    VAL_LIST = os.path.abspath(VAL_LIST)
    CKPT = os.path.abspath(CKPT)

    print("VAL_LIST =", VAL_LIST)
    print("CKPT     =", CKPT)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    items = read_list(VAL_LIST)
    print("items:", len(items))

    # 简单抽样检查存在性
    sample = random.sample(items, k=min(5, len(items)))
    exist_cnt = sum(1 for _, p in sample if os.path.exists(p))
    print(f"sample exists: {exist_cnt}/{len(sample)}")
    if exist_cnt == 0:
        print("[ERROR] Your list paths still do not exist on disk. Example paths:")
        for _, p in sample:
            print(" ", p)
        return

    pairs = build_pairs(items, num_pos=3000, num_neg=3000)
    print("pairs:", len(pairs))

    # 加载模型
    ckpt = torch.load(CKPT, map_location="cpu")
    model = ECAPA_TDNN(in_channels=80, channels=512, embd_dim=192).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    emb_cache = {}
    labels, scores = [], []
    missing = 0
    used = 0

    for is_same, p1, p2 in tqdm(pairs, desc="Scoring"):
        if p1 not in emb_cache:
            emb_cache[p1] = embed_from_fbank_pt(model, p1, device)
        if p2 not in emb_cache:
            emb_cache[p2] = embed_from_fbank_pt(model, p2, device)

        e1 = emb_cache[p1]
        e2 = emb_cache[p2]
        if e1 is None or e2 is None:
            missing += 1
            continue

        scores.append(cosine(e1, e2))
        labels.append(is_same)
        used += 1

    print(f"Scoring used pairs: {used}, skipped(missing feats): {missing}")

    if used == 0:
        print("[ERROR] used==0: still cannot load any feature. Showing first 20 existence checks:")
        for i, (_, p) in enumerate(items[:20]):
            print(i, os.path.exists(p), p)
        return

    eer, th = compute_eer(labels, scores)
    print(f"EER = {eer*100:.2f}%  (best_th≈{th:.4f})")

    os.makedirs("outputs_eval", exist_ok=True)

    # ROC
    fpr, tpr = roc_points(labels, scores, num_th=200)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC (EER={eer*100:.2f}%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs_eval/roc.png")
    plt.close()

    # DET
    fars, frrs = det_points(labels, scores, num_th=400)
    plt.figure()
    plt.plot(fars, frrs)
    plt.xlabel("FAR")
    plt.ylabel("FRR")
    plt.title(f"DET (EER={eer*100:.2f}%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs_eval/det.png")
    plt.close()

    # Score histogram
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    plt.figure()
    plt.hist(pos, bins=50, alpha=0.6, label="same")
    plt.hist(neg, bins=50, alpha=0.6, label="diff")
    plt.legend()
    plt.title("Score distribution (cosine)")
    plt.tight_layout()
    plt.savefig("outputs_eval/score_hist.png")
    plt.close()

    # t-SNE + Recall@K
    X, y_tsne = collect_embeddings_for_tsne(model, items, device, max_spk=20, per_spk=25)
    if X is not None and y_tsne is not None:
        if _HAS_SKLEARN:
            tsne = TSNE(
                n_components=2,
                perplexity=min(30, max(5, (len(X) // 3))),
                init="pca",
                learning_rate="auto",
                random_state=1234,
            )
            Z = tsne.fit_transform(X)

            plt.figure()
            uniq = sorted(set(y_tsne.tolist()))
            for spk in uniq:
                mask = (y_tsne == spk)
                plt.scatter(Z[mask, 0], Z[mask, 1], s=10, alpha=0.8)
            plt.title("t-SNE of Speaker Embeddings (sampled)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("outputs_eval/tsne.png")
            plt.close()
        else:
            print("[WARN] sklearn not found, skip t-SNE. Install: pip install scikit-learn")

        emb_t = torch.from_numpy(X).float()
        emb_t = emb_t / (emb_t.norm(dim=1, keepdim=True) + 1e-12)
        lab_t = torch.from_numpy(y_tsne).long()
        r = recall_at_k(emb_t, lab_t, ks=(1, 5, 10))
        print("Recall@K (sampled):", {f"R@{k}": round(v * 100, 2) for k, v in r.items()})
        with open("outputs_eval/recall_at_k.txt", "w", encoding="utf-8") as f:
            for k, v in r.items():
                f.write(f"Recall@{k}: {v*100:.2f}%\n")
    else:
        print("[WARN] No embeddings collected for t-SNE/Recall@K")

    print("Saved: outputs_eval/roc.png, det.png, score_hist.png (+ tsne/recall if enabled)")


if __name__ == "__main__":
    main()
