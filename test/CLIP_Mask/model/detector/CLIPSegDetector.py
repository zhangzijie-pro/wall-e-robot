import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPSegDetector(nn.Module):
    def __init__(
        self,
        clip_model,
        text_encoder,
        matcher,
        pos_encoding,
        seg_head,
        image_size=(224, 224)
    ):
        super().__init__()
        self.clip_model = clip_model
        self.text_encoder = text_encoder
        self.matcher = matcher
        self.pos_encoding = pos_encoding
        self.seg_head = seg_head
        self.image_size = torch.tensor(image_size).float()  # e.g., [224., 224.]

    def forward(self, image, anchors, prompt_text, selected_indices=None):
        """
        Args:
            image: Tensor[B, 3, H, W]
            anchors: Tensor[B, K, 4] (cx, cy, w, h), in absolute pixel
            prompt_text: list[str]
            selected_indices: Optional[List[List[int]]], e.g. top-k matched anchors per image

        Returns:
            dict with:
                - "masks": Tensor[B, K, H, W]
                - "matched_scores": Tensor[B, K]
        """
        B, K, _ = anchors.shape

        # Encode image features from CLIP visual backbone
        feature_map = self._extract_feature_map(image)  # [B, D, H', W']
        B, D, H, W = feature_map.shape

        # Anchor -> feature vector per region
        region_features = self._extract_region_features(feature_map, anchors)  # [B, K, D]

        # Normalize anchors to [0, 1]
        norm_anchors = anchors / self.image_size.to(anchors.device)  # [B, K, 4]

        # Add positional encoding
        region_features = self.pos_encoding(region_features, norm_anchors)  # [B, K, D]

        # Text encoding
        text_features = self.text_encoder(prompt_text)  # [T, D]
        text_features = F.normalize(text_features, dim=-1)
        region_features = F.normalize(region_features, dim=-1)

        # Matching
        matched_scores = self.matcher(region_features, text_features)  # [B, K]

        # Select top regions (or use provided indices)
        if selected_indices is None:
            selected_indices = torch.topk(matched_scores, k=min(10, K), dim=1).indices  # [B, k]

        # Build mask for selected anchors
        mask_protos, mask_coefs = self.seg_head(feature_map, selected_indices)  # [B, C, H, W], [B, k, C]
        masks = self._build_masks(mask_protos, mask_coefs)  # [B, k, H, W]

        return {
            "masks": masks,
            "matched_scores": matched_scores,
            "selected_indices": selected_indices
        }

    def _extract_feature_map(self, image):
        """
        Extract patch token feature map from CLIP (ViT only).
        """
        B = image.size(0)
        with torch.no_grad():
            tokens = self.clip_model.encode_image(image)  # [B, num_patches+1, D]
        patch_tokens = tokens[:, 1:]  # remove CLS
        D = patch_tokens.size(-1)
        S = patch_tokens.size(1)
        H = W = int(S ** 0.5)
        return patch_tokens.permute(0, 2, 1).reshape(B, D, H, W)  # [B, D, H, W]

    def _extract_region_features(self, feature_map, anchors):
        """
        Sample RoI features for each anchor.
        Args:
            feature_map: [B, D, H, W]
            anchors: [B, K, 4] in absolute coordinates

        Returns:
            region_features: [B, K, D]
        """
        B, D, H, W = feature_map.shape
        K = anchors.shape[1]

        device = feature_map.device
        spatial_feats = []

        for b in range(B):
            fm = feature_map[b]  # [D, H, W]
            feat_list = []
            for k in range(K):
                cx, cy, w, h = anchors[b, k]
                x1 = max(int((cx - w / 2) / self.image_size[0] * W), 0)
                y1 = max(int((cy - h / 2) / self.image_size[1] * H), 0)
                x2 = min(int((cx + w / 2) / self.image_size[0] * W), W)
                y2 = min(int((cy + h / 2) / self.image_size[1] * H), H)

                patch = fm[:, y1:y2, x1:x2]
                if patch.numel() == 0:
                    pooled = torch.zeros(D, device=device)
                else:
                    pooled = F.adaptive_avg_pool2d(patch, 1).squeeze(-1).squeeze(-1)  # [D]
                feat_list.append(pooled)
            spatial_feats.append(torch.stack(feat_list))  # [K, D]

        return torch.stack(spatial_feats)  # [B, K, D]

    def _build_masks(self, mask_proto, mask_coefs):
        """
        Generate masks from prototypes and coefficients
        Args:
            mask_proto: [B, C, H, W]
            mask_coefs: [B, K, C]

        Returns:
            masks: [B, K, H, W]
        """
        B, C, H, W = mask_proto.shape
        K = mask_coefs.shape[1]
        masks = []

        for b in range(B):
            mp = mask_proto[b]  # [C, H, W]
            coef = mask_coefs[b]  # [K, C]
            m = torch.einsum('kc,chw->k hw', coef, mp)  # [K, H, W]
            m = torch.sigmoid(m)
            masks.append(m)
        return torch.stack(masks)  # [B, K, H, W]