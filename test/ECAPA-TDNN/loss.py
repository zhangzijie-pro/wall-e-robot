import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AAMSoftmaxLoss(nn.Module):
    def __init__(self, embedding_dim, num_classes, margin=0.2, scale=30.0):
        super(AAMSoftmaxLoss, self).__init__()
        self.margin = margin
        self.scale = scale
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        print(f"[DEBUG] Embeddings shape: {embeddings.shape}")
        print(f"[DEBUG] Labels shape: {labels.shape}, dtype: {labels.dtype}, max: {labels.max()}, min: {labels.min()}")
        assert labels.dim() == 1, f"Expected 1D labels, got shape {labels.shape}"
        if labels.dtype != torch.long:
            labels = labels.long()
        batch_size = embeddings.size(0)
        assert labels.size(0) == batch_size, "Mismatch between batch size and labels"

        # L2 normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weights = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity
        cosine = F.linear(embeddings, weights)  # [B, num_classes]

        # Compute phi (cos(θ + m))
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Combine phi and cosine for the final logits
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        # Cross entropy loss
        loss = F.cross_entropy(output, labels)
        return loss
