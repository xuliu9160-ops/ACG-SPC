# model.py
# ConvNeXt-Tiny + Quality-based Multiple Instance Learning (QMIL)
# - torchvision ConvNeXt-Tiny (IMAGENET1K) or timm ConvNeXt-Tiny (IN22K)
# - input : [B, S, F, C, H, W]
# - output: frame_logits [B, S, F, num_classes], video_logits [B, num_classes]

import torch
import torch.nn as nn
try:
    from torchvision.models.convnext import convnext_tiny, ConvNeXt_Tiny_Weights
except Exception:
    convnext_tiny = None
    ConvNeXt_Tiny_Weights = None


try:
    import timm
except Exception:
    timm = None


def topk_softmax(scores: torch.Tensor, k: int) -> torch.Tensor:
    """
    Args:
        scores: [B, S]
        k: keep top-k segments per sample
    Returns:
        weights: [B, S], softmax-normalized on segment axis
    """
    if scores.dim() != 2:
        raise ValueError(f"Expected scores shape [B,S], got {scores.shape}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    B, S = scores.shape
    k_eff = min(int(k), int(S))

    topk_idx = torch.topk(scores, k=k_eff, dim=1, largest=True, sorted=False).indices
    # AMP/FP16-safe mask value (e.g., avoid -1e9 overflow in float16)
    neg_large = torch.finfo(scores.dtype).min
    masked_scores = torch.full_like(scores, neg_large)
    masked_scores.scatter_(1, topk_idx, scores.gather(1, topk_idx))
    return torch.softmax(masked_scores, dim=1)


class SegmentQualityMLP(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,S,D] -> [B,S,1]
        return self.net(x)


# -----------------------------
# Helpers for timm ConvNeXt
# -----------------------------
class _TimmConvNeXtStemStages(nn.Module):
    """
    Wrap timm ConvNeXt variants into:
      stem(x)  -> low-level maps [N, C_low, H', W']
      stages(low) -> pooled features [N, D]
    Works for multiple timm API variants.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # Determine feature dim robustly
        num_features = getattr(model, "num_features", None)
        if num_features is None:
            num_features = getattr(model, "feature_info", None)
        self.feature_dim = int(getattr(model, "num_features", 768))

        # Case A: newer timm with stem + stages
        if hasattr(model, "stem") and hasattr(model, "stages"):
            self.stem = model.stem

            # Build a stages pipeline that ALWAYS ends in pooled [N, D]
            parts = [*model.stages]

            # Some timm convnext have norm + head.global_pool
            if hasattr(model, "norm"):
                parts.append(model.norm)

            # If there is a head with global_pool, use it. Otherwise fallback to GAP.
            if hasattr(model, "head") and hasattr(model.head, "global_pool"):
                parts.append(model.head.global_pool)
                self.stages = nn.Sequential(*parts)
                self._force_gap = False
            else:
                # We will force GAP in forward if needed
                self.stages = nn.Sequential(*parts)
                self._force_gap = True

        # Case B: older timm with downsample_layers + stages
        elif hasattr(model, "downsample_layers") and hasattr(model, "stages"):
            self.stem = model.downsample_layers[0]

            # Interleave downsample + stage blocks
            remaining_down = list(model.downsample_layers[1:])
            stage_blocks = list(model.stages)

            combined = []
            for down, stage in zip(remaining_down, stage_blocks):
                combined.extend([down, stage])
            if len(stage_blocks) > len(remaining_down):
                combined.append(stage_blocks[-1])

            self.stages = nn.Sequential(*combined)
            self._force_gap = True

        # Case C: fallback if model exposes features
        elif hasattr(model, "features"):
            self.stem = nn.Sequential(model.features[0])
            self.stages = nn.Sequential(*model.features[1:])
            self._force_gap = True

        else:
            raise RuntimeError("Unsupported timm ConvNeXt structure. Please check timm version/model name.")

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward_stem(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)

    def forward_stages(self, low: torch.Tensor) -> torch.Tensor:
        out = self.stages(low)

        # Ensure pooled [N, D]
        if out.dim() == 4:
            # If spatial not 1x1, GAP it
            if self._force_gap or out.shape[-2] != 1 or out.shape[-1] != 1:
                out = self.gap(out)
            out = out.flatten(1)  # [N, D]
        elif out.dim() == 2:
            # already [N, D]
            pass
        else:
            raise RuntimeError(f"Unexpected stages output shape: {out.shape}")

        return out


class _TorchvisionConvNeXtStemStages(nn.Module):
    """
    Wrap torchvision ConvNeXt-Tiny into:
      stem(x)   -> [N, C_low, H', W']
      stages(x) -> [N, D, 1, 1]
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.stem = nn.Sequential(model.features[0])
        self.stages = nn.Sequential(
            *model.features[1:],
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.feature_dim = int(model.classifier[2].in_features)


def create_qmil_backbone(backbone_source, timm_name, pretrained):
    backbone_source = str(backbone_source).lower()
    if backbone_source not in ("torchvision", "timm"):
        raise ValueError("backbone_source must be 'torchvision' or 'timm'")

    if backbone_source == "torchvision":
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        cnxt = convnext_tiny(weights=weights)
        return _TorchvisionConvNeXtStemStages(cnxt)

    if timm is None:
        raise ImportError("timm is not installed. Install with: pip install timm")
    raw_timm_name = str(timm_name)
    timm_name_aliases = {
        "convnext_tiny.fb_in22k": "convnext_tiny_in22k",
    }
    resolved_timm_name = raw_timm_name
    try:
        cnxt = timm.create_model(resolved_timm_name, pretrained=bool(pretrained), num_classes=0)
    except RuntimeError as e:
        if "Unknown model" in str(e) and raw_timm_name in timm_name_aliases:
            resolved_timm_name = timm_name_aliases[raw_timm_name]
            print(f"[timm] Fallback model name: {raw_timm_name} -> {resolved_timm_name}")
            cnxt = timm.create_model(resolved_timm_name, pretrained=bool(pretrained), num_classes=0)
        else:
            raise
    return _TimmConvNeXtStemStages(cnxt)


def get_qmil_backbone_feature_dim(backbone):
    if not hasattr(backbone, "feature_dim"):
        raise AttributeError("Backbone wrapper must provide `feature_dim`.")
    return int(backbone.feature_dim)


def create_qmil_frame_head(feature_dim, num_classes):
    return nn.Linear(int(feature_dim), int(num_classes))


# -----------------------------
# Video Classifier
# -----------------------------
class VideoClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        segment_count: int = 6,
        frames_per_segment: int = 3,
        video_dropout: float = 0.3,
        backbone_source: str = "torchvision",   # "torchvision" or "timm"
        timm_name: str = "convnext_tiny.fb_in22k",
        pretrained: bool = True,
        selection_mode: str = "topk_softmax",
        topk: int = 2,
    ):
        super().__init__()
        self.segment_count = int(segment_count)
        self.frames_per_segment = int(frames_per_segment)
        self.selection_mode = str(selection_mode)
        self.topk = int(topk)

        backbone = create_qmil_backbone(
            backbone_source=backbone_source,
            timm_name=timm_name,
            pretrained=pretrained,
        )
        self.stem = backbone.stem
        self.stages = backbone.stages
        self.feature_dim = get_qmil_backbone_feature_dim(backbone)

        self.quality_head = SegmentQualityMLP(in_features=self.feature_dim)
        self.frame_head = create_qmil_frame_head(self.feature_dim, num_classes)
        self.video_head = nn.Sequential(nn.Dropout(float(video_dropout)), nn.Linear(self.feature_dim, num_classes))

    def forward(self, x: torch.Tensor):
        # x: [B,S,F,C,H,W]
        if x.dim() != 6:
            raise ValueError(f"Expected [B,S,F,C,H,W], got {x.shape}")

        B, S, F, C, H, W = x.shape
        if S != self.segment_count or F != self.frames_per_segment:
            raise ValueError(f"Expected S={self.segment_count}, F={self.frames_per_segment}; got S={S}, F={F}")

        x_flat = x.view(B * S * F, C, H, W)  # [N,C,H,W], N=B*S*F

        # Full backbone deep feature extraction: [B*S*F,C,H,W] -> [B*S*F,D]
        low = self.stem(x_flat)
        feat_flat = self.stages(low)
        if feat_flat.dim() == 4:
            feat_flat = feat_flat.flatten(1)
        elif feat_flat.dim() != 2:
            raise RuntimeError(f"Unexpected backbone output shape: {feat_flat.shape}")

        feats = feat_flat.view(B, S, F, -1)  # [B,S,F,D]
        frame_logits = self.frame_head(feats)

        seg_tokens = feats.mean(dim=2)        # [B,S,D]
        seg_scores = self.quality_head(seg_tokens).squeeze(-1)  # [B,S]

        if self.selection_mode == "topk_softmax":
            seg_w = topk_softmax(seg_scores, k=self.topk)         # [B,S]
        else:
            raise ValueError(f"Unsupported selection_mode: {self.selection_mode}")

        video_feat = (seg_tokens * seg_w.unsqueeze(-1)).sum(dim=1)  # [B,D]
        video_logits = self.video_head(video_feat)

        return frame_logits, video_logits


if __name__ == "__main__":
    B, S, F, C, H, W = 2, 6, 3, 3, 224, 224
    x = torch.randn(B, S, F, C, H, W)

    m_tv = VideoClassifier(
        num_classes=2,
        segment_count=S,
        frames_per_segment=F,
        backbone_source="torchvision",
        pretrained=False,
        video_dropout=0.3,
        selection_mode="topk_softmax",
        topk=2,
    )
    fl, vl = m_tv(x)
    print("torchvision frame_logits:", fl.shape, "video_logits:", vl.shape)

    # Uncomment if timm is installed and model exists in your timm build
    # m_timm = VideoClassifier(
    #     num_classes=2,
    #     segment_count=S,
    #     frames_per_segment=F,
    #     backbone_source="timm",
    #     timm_name="convnext_tiny.fb_in22k",
    #     pretrained=False,
    #     video_dropout=0.3,
    #     selection_mode="topk_softmax",
    #     topk=2,
    # )
    # fl2, vl2 = m_timm(x)
    # print("timm frame_logits:", fl2.shape, "video_logits:", vl2.shape)
