# loss/aam_softmax.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AAMSoftmax(nn.Module):
    """
    Additive Angular Margin Softmax (ArcFace-style)
    输入：
      emb: [B, D] 已经 L2 normalize
      label: [B]
    输出：
      logits: [B, C]
      loss: 标量
    """
    def __init__(self, emb_dim: int, num_classes: int, s: float = 30.0, m: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.randn(num_classes, emb_dim))
        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb: torch.Tensor, label: torch.Tensor):
        # normalize class weights
        W = F.normalize(self.weight, p=2, dim=1)  # [C,D]
        # cosine: [B,C]
        cosine = torch.matmul(emb, W.t()).clamp(-1.0, 1.0)

        # cos(theta+m) = cosθ cos m - sinθ sin m
        sine = torch.sqrt((1.0 - cosine * cosine).clamp(min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m

        # optional: easy-margin off, use standard arcface condition
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits = logits * self.s

        loss = F.cross_entropy(logits, label)
        # return logits, loss
        return loss, logits