from dataclasses import dataclass

@dataclass
class TrainConfig:
    # 数据
    train_list: str = r"processed/cn_celeb2/train_fbank_list.txt"
    val_list: str   = r"processed/cn_celeb2/val_fbank_list.txt"

    # 模型
    feat_dim: int = 80
    channels: int = 512
    emb_dim: int = 192

    # AAM-Softmax
    margin: float = 0.2
    scale: float = 30.0

    # 训练
    epochs: int = 100
    # batch_size: int = 8
    batch_size: int = 32
    num_workers: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0

    # 设备
    device: str = "cuda"
    amp: bool = True

    # 输出
    out_dir: str = "outputs"
    save_best: bool = True
