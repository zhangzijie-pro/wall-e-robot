import os
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from configs.train_config import TrainConfig
from models.ecapa import ECAPA_TDNN
from loss.aamsoftmax import AAMSoftmax
from data.dataset import FbankPtDataset, pad_collate
from utils.seed import set_seed
from utils.meters import AverageMeter, top1_accuracy
from utils.plot import plot_curves


def infer_num_classes(list_path: str) -> int:
    mx = -1
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lab = int(line.split()[0])
                mx = max(mx, lab)
    return mx + 1


def label_stats(list_path: str):
    labels = set()
    mn = 10**18
    mx = -10**18
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            lab = int(line.split()[0])
            labels.add(lab)
            mn = min(mn, lab)
            mx = max(mx, lab)
    return mn, mx, len(labels)


@torch.no_grad()
def validate(model, head, loader, device, num_classes: int):
    model.eval()
    head.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # label 范围检查（避免 silent 错误）
        if y.min().item() < 0 or y.max().item() >= num_classes:
            raise RuntimeError(
                f"[VAL] label out of range: min={y.min().item()}, max={y.max().item()}, C={num_classes}"
            )

        emb = model(x)

        # 数值检查（遇到 NaN/Inf 直接定位）
        if not torch.isfinite(emb).all():
            raise RuntimeError("[VAL] Non-finite embedding detected (NaN/Inf).")

        loss, logits = head(emb, y)

        if not torch.isfinite(loss).all() or not torch.isfinite(logits).all():
            raise RuntimeError("[VAL] Non-finite loss/logits detected (NaN/Inf).")

        acc = top1_accuracy(logits, y)  # float

        bs = y.size(0)
        loss_meter.update(float(loss.item()), bs)
        acc_meter.update(float(acc), bs)

    return loss_meter.avg, acc_meter.avg


def main():
    cfg = TrainConfig()
    set_seed(1234)

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)

    # ✅ device 统一规范
    use_cuda = torch.cuda.is_available() and (cfg.device.startswith("cuda") if isinstance(cfg.device, str) else True)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device:", device)

    # 数据
    train_ds = FbankPtDataset(cfg.train_list)
    val_ds = FbankPtDataset(cfg.val_list)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=pad_collate,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=pad_collate,
        pin_memory=(device.type == "cuda"),
        drop_last=False
    )

    # 类别数（说话人数量）
    num_classes = infer_num_classes(cfg.train_list)
    mn, mx, uniq = label_stats(cfg.train_list)
    print("num_classes =", num_classes)
    print(f"train label stats: min={mn}, max={mx}, unique={uniq}")
    if mn != 0:
        print("[WARN] train labels min != 0. If labels start from 1, accuracy will be near 0 unless adjusted.")
    if mx != num_classes - 1:
        print("[WARN] max label != num_classes-1, check label continuity.")
    if uniq != num_classes:
        print("[WARN] unique labels != num_classes. Labels may be non-continuous; training may be harder.")

    # 模型 + 头
    model = ECAPA_TDNN(in_channels=cfg.feat_dim, channels=cfg.channels, embd_dim=cfg.emb_dim).to(device)
    head = AAMSoftmax(cfg.emb_dim, num_classes, s=cfg.scale , m=cfg.margin).to(device)

    # 优化器 / 调度
    params = list(model.parameters()) + list(head.parameters())
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    # ✅ 正确 AMP 用法（device_type 只能是 "cuda" 或 "cpu"）
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        head.train()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", ncols=100)
        for x, y, _ in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # label 范围检查
            if y.min().item() < 0 or y.max().item() >= num_classes:
                raise RuntimeError(
                    f"[TRAIN] label out of range: min={y.min().item()}, max={y.max().item()}, C={num_classes}"
                )

            optim.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                emb = model(x)

                # embedding 数值检查
                if not torch.isfinite(emb).all():
                    raise RuntimeError("[TRAIN] Non-finite embedding detected (NaN/Inf).")

                loss, logits = head(emb, y)

            # loss/logits 数值检查
            if not torch.isfinite(loss).all() or not torch.isfinite(logits).all():
                raise RuntimeError("[TRAIN] Non-finite loss/logits detected (NaN/Inf).")

            # backward + step
            if use_amp:
                scaler.scale(loss).backward()

                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)

                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                optim.step()

            acc = top1_accuracy(logits, y)  # float

            bs = y.size(0)
            loss_meter.update(float(loss.item()), bs)
            acc_meter.update(float(acc), bs)

            pbar.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                acc=f"{acc_meter.avg:.4f}",
                lr=f"{optim.param_groups[0]['lr']:.2e}"
            )

        scheduler.step()

        # 验证
        val_loss, val_acc = validate(model, head, val_loader, device, num_classes=num_classes)

        history["train_loss"].append(loss_meter.avg)
        history["train_acc"].append(acc_meter.avg)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[Epoch {epoch}] "
            f"train_loss={loss_meter.avg:.4f}, train_acc={acc_meter.avg:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        # 保存 checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "head": head.state_dict(),
            "optim": optim.state_dict(),
            "history": history,
            "num_classes": num_classes,
        }
        torch.save(ckpt, os.path.join(cfg.out_dir, "last.pt"))

        if cfg.save_best and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, os.path.join(cfg.out_dir, "best.pt"))
            print(">> saved best.pt, best_val_acc =", best_val_acc)

        # 每个 epoch 更新曲线图
        plot_curves(cfg.out_dir, history)

    # 最终保存 history
    with open(os.path.join(cfg.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("训练完成！曲线图在：", cfg.out_dir)


if __name__ == "__main__":
    main()
