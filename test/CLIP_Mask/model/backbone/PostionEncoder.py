import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):
    def __init__(self, input_dim=4, embed_dim=512, mode='add'):
        """
        Args:
            input_dim: anchor 输入维度 (cx, cy, w, h)
            embed_dim: CLIP embedding 维度，默认512或768
            mode: 融合方式：'add'（加性）或 'concat'（拼接后线性投影）
        """
        super().__init__()
        self.mode = mode
        self.embed_dim = embed_dim

        if mode == 'add':
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.ReLU(),
                nn.LayerNorm(embed_dim)
            )
        elif mode == 'concat':
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, embed_dim // 2),
                nn.ReLU(),
                nn.LayerNorm(embed_dim // 2)
            )
            self.projector = nn.Linear(embed_dim + embed_dim // 2, embed_dim)
        else:
            raise ValueError("mode must be 'add' or 'concat'")

    def forward(self, patch_embeddings: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_embeddings: Tensor[N, D]
            anchors: Tensor[N, 4] (cx, cy, w, h)，归一化坐标推荐 [0,1]

        Returns:
            enhanced_embeddings: Tensor[N, D]
        """
        pos_feat = self.encoder(anchors)  # [N, D] 或 [N, D//2]

        if self.mode == 'add':
            return patch_embeddings + pos_feat
        else:  # concat mode
            fused = torch.cat([patch_embeddings, pos_feat], dim=-1)
            return self.projector(fused)