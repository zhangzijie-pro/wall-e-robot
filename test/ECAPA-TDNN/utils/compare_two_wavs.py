import argparse
import torch

from models.ecapa import ECAPA_TDNN
from utils.audio import load_wav_mono, wav_to_fbank


@torch.no_grad()
def embed_wav(model, wav_path, device):
    wav = load_wav_mono(wav_path, target_sr=16000)      # [T]
    feat = wav_to_fbank(wav, n_mels=80)                 # [T_frames,80]
    x = feat.unsqueeze(0).to(device)                    # [1,T,80]
    emb = model(x).squeeze(0).cpu()                     # [192] normalized
    return emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="outputs/best.pt")
    parser.add_argument("--wav1", type=str, required=True)
    parser.add_argument("--wav2", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.35, help="cosine阈值，需用验证脚本估计")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = ECAPA_TDNN(in_channels=80, channels=512, embd_dim=192).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    e1 = embed_wav(model, args.wav1, device)
    e2 = embed_wav(model, args.wav2, device)

    score = float(torch.sum(e1 * e2).item())  # normalize后点积=cosine
    same = score >= args.threshold

    print("cosine =", score)
    print("threshold =", args.threshold)
    print("same person ?" , same)


if __name__ == "__main__":
    main()
