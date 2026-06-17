# model.py
# ConvNeXt-Tiny + Quality-based Multiple Instance Learning (QMIL)
# - torchvision ConvNeXt-Tiny (IMAGENET1K) or timm ConvNeXt-Tiny (IN22K)
# - input : [B, S, F, C, H, W]
# - output: frame_logits [B, S, F, num_classes], video_logits [B, num_classes]

import torch
import torch.nn as nn
import torch.nn.functional as F
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
# ACG-STG: Adaptive Gating Modules
# -----------------------------
class _Stats3Gate(nn.Module):
    """Low-parameter gate: Linear(3,1) + Sigmoid on 3 core statistics.

    Input statistics (order-sensitive):
      [seg_prob_std, neighbor_smoothness, support_mean]
    """
    def __init__(self, init_bias: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(3, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, float(init_bias))

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        # stats: [B, 3] -> gate: [B, 1] in (0,1)
        return torch.sigmoid(self.linear(stats))


class _MLPGate(nn.Module):
    """Small MLP gate: Linear(7,16) + ReLU + Linear(16,1) + Sigmoid.

    Input statistics (order-sensitive):
      [seg_prob_std, top1_minus_mean, prob_range,
       neighbor_smoothness, support_mean, support_top1,
       entropy_or_concentration]
    """
    def __init__(self, init_bias: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(7, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )
        # Zero-init final layer: training starts at g = sigmoid(init_bias)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.constant_(self.mlp[2].bias, float(init_bias))

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        # stats: [B, 7] -> gate: [B, 1] in (0,1)
        return torch.sigmoid(self.mlp(stats))


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


# -----------------------------
#  Generic 2D backbone wrappers
# -----------------------------
class _ResNetStemStages(nn.Module):
    """Wrap torchvision ResNet (18/34) into stem + stages.  feature_dim=512."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.stem = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
        )
        self.stages = nn.Sequential(
            model.layer1, model.layer2, model.layer3, model.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.feature_dim = 512


class _EfficientNetStemStages(nn.Module):
    """Wrap torchvision EfficientNet-B0 into stem + stages.  feature_dim=1280."""
    def __init__(self, model: nn.Module):
        super().__init__()
        feats = list(model.features)
        self.stem = nn.Sequential(feats[0], feats[1])
        self.stages = nn.Sequential(
            *feats[2:],
            model.avgpool if hasattr(model, "avgpool") else nn.AdaptiveAvgPool2d((1, 1)),
        )
        # classifier is [Dropout(p), Linear(1280, nc)]
        self.feature_dim = int(model.classifier[1].in_features)


class _MobileNetV3StemStages(nn.Module):
    """Wrap torchvision MobileNetV3-Large into stem + stages.  feature_dim=960."""
    def __init__(self, model: nn.Module):
        super().__init__()
        feats = list(model.features)
        self.stem = nn.Sequential(feats[0], feats[1])
        self.stages = nn.Sequential(
            *feats[2:],
            model.avgpool if hasattr(model, "avgpool") else nn.AdaptiveAvgPool2d((1, 1)),
        )
        # classifier is [Linear(960,1280), Hardswish, Dropout, Linear(1280,nc)]
        self.feature_dim = int(model.classifier[0].in_features)


def create_qmil_backbone(backbone_source, timm_name, pretrained, backbone_name="convnext_tiny"):
    backbone_name = str(backbone_name).lower()

    # --- 2D backbone dispatch (non-ConvNeXt) ---
    if backbone_name == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights
        w = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        return _ResNetStemStages(resnet18(weights=w))

    if backbone_name == "resnet34":
        from torchvision.models import resnet34, ResNet34_Weights
        w = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        return _ResNetStemStages(resnet34(weights=w))

    if backbone_name == "efficientnet_b0":
        from torchvision.models import efficientnet_b0
        try:
            from torchvision.models import EfficientNet_B0_Weights
            w = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        except ImportError:
            w = pretrained  # older torchvision: pretrained is bool
        return _EfficientNetStemStages(efficientnet_b0(weights=w) if w is not None else efficientnet_b0())

    if backbone_name == "mobilenet_v3_large":
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        w = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        return _MobileNetV3StemStages(mobilenet_v3_large(weights=w))

    # --- ConvNeXt-Tiny branch (existing, default) ---
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
#  Key-compatible state_dict loader
# -----------------------------
def _load_state_dict_key_compatible(target, source_state_dict, module_name):
    """
    Load source_state_dict into target, handling torchvision/timm key prefix
    differences in Sequential wrappers.

    Attempt sequence:
      1. strict=True direct load — works if both use same wrapper type.
      2. Strip one "0." prefix level from source keys and retry strict=True.
      3. Strip "features.0." prefix level and retry strict=True.
      4. Raise RuntimeError with key samples on failure.
    """
    target_keys = set(target.state_dict().keys())
    source_keys = set(source_state_dict.keys())

    # Attempt 1: direct
    if source_keys == target_keys:
        target.load_state_dict(source_state_dict, strict=True)
        return

    # Attempt 2: strip single "0." prefix  (e.g. 0.0.weight -> 0.weight)
    remapped = {}
    for k, v in source_state_dict.items():
        if k.startswith("0."):
            remapped[k[2:]] = v
        else:
            remapped[k] = v
    if set(remapped.keys()) == target_keys:
        print(f"[MAE] {module_name}: remapped keys (stripped '0.' prefix) — strict load")
        target.load_state_dict(remapped, strict=True)
        return

    # Attempt 3: strip "features.0." prefix
    remapped2 = {}
    prefix = "features.0."
    for k, v in source_state_dict.items():
        if k.startswith(prefix):
            remapped2[k[len(prefix):]] = v
        else:
            remapped2[k] = v
    if set(remapped2.keys()) == target_keys:
        print(f"[MAE] {module_name}: remapped keys (stripped 'features.0.' prefix) — strict load")
        target.load_state_dict(remapped2, strict=True)
        return

    # Failed — report
    raise RuntimeError(
        f"[MAE] Cannot load {module_name}: key mismatch after all remap attempts.\n"
        f"  Target keys (sample): {sorted(target_keys)[:5]}\n"
        f"  Source keys (sample): {sorted(source_keys)[:5]}"
    )


def load_mae_encoder_from_checkpoint(model, checkpoint_path):
    """
    Load MAE encoder weights (stem/stages) into a VideoClassifier model.

    Works for any backbone (ConvNeXt-Tiny, EfficientNet-B0, etc.) as long as:
      - MAE checkpoint backbone_name matches model._qm_backbone_name
      - MAE checkpoint feature_dim matches model.feature_dim
      - checkpoint contains stem_state_dict and stages_state_dict

    Loading is strict=True — the MAE stem/stages are byte-identical to
    downstream VideoClassifier stem/stages by construction.

    Does NOT load frame_head / quality_head / video_head.
    """
    raw_ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(raw_ckpt, dict):
        raise ValueError(f"[MAE] Invalid checkpoint format: {checkpoint_path}")

    pretrain_type = raw_ckpt.get("pretrain_type", None)
    if pretrain_type != "mae":
        raise ValueError(
            f"[MAE] Expected pretrain_type='mae', got '{pretrain_type}'. "
            f"Checkpoint: {checkpoint_path}"
        )

    ckpt_bb = raw_ckpt.get("backbone_name", None)
    model_bb = getattr(model, "_qm_backbone_name", None)
    if model_bb is None:
        raise ValueError(
            "[MAE] model._qm_backbone_name is not set. "
            "Set it before calling load_mae_encoder_from_checkpoint()."
        )
    if ckpt_bb != model_bb:
        raise ValueError(
            f"[MAE] Backbone mismatch: checkpoint='{ckpt_bb}', "
            f"model='{model_bb}'. "
            f"Checkpoint: {checkpoint_path}"
        )

    ckpt_fd = raw_ckpt.get("feature_dim", None)
    if ckpt_fd is not None and int(ckpt_fd) != model.feature_dim:
        raise ValueError(
            f"[MAE] Feature dim mismatch: checkpoint={ckpt_fd}, "
            f"model={model.feature_dim}. "
            f"Checkpoint: {checkpoint_path}"
        )

    stem_sd = raw_ckpt.get("stem_state_dict", None)
    stages_sd = raw_ckpt.get("stages_state_dict", None)
    if stem_sd is None or stages_sd is None:
        raise ValueError(
            "[MAE] Checkpoint missing 'stem_state_dict'/'stages_state_dict'. "
            "Re-train MAE to generate a compatible checkpoint."
        )

    model.stem.load_state_dict(stem_sd, strict=True)
    model.stages.load_state_dict(stages_sd, strict=True)
    print(f"[MAE] Loaded stem/stages from MAE checkpoint with strict=True")
    print(f"[MAE] Backbone: {ckpt_bb}  feature_dim: {model.feature_dim}")
    print("[MAE] frame_head / quality_head / video_head are randomly initialized "
          "(not loaded from MAE checkpoint)")

    ckpt_source = raw_ckpt.get("backbone_source", None)
    return {
        "pretrain_type": "mae",
        "backbone_name": ckpt_bb,
        "backbone_source": ckpt_source,
        "timm_name": raw_ckpt.get("timm_name", None),
        "feature_dim": model.feature_dim,
        "mask_ratio": raw_ckpt.get("mask_ratio", None),
        "patch_size": raw_ckpt.get("patch_size", None),
        "mae_best_loss": raw_ckpt.get("best_loss", None),
        "mae_best_epoch": raw_ckpt.get("best_epoch", None),
        "mae_epochs": raw_ckpt.get("epochs", None),
        "encoder_params": raw_ckpt.get("encoder_params", None),
        "decoder_params": raw_ckpt.get("decoder_params", None),
    }


# Backward-compatible alias — existing callers can use either name.
load_convnext_encoder_from_mae_checkpoint = load_mae_encoder_from_checkpoint


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
        backbone_name: str = "convnext_tiny",    # see create_qmil_backbone
        selection_mode: str = "topk_softmax",
        topk: int = 2,
        # --- STG-CRSA hyperparameters ---
        stg_temperature: float = 1.0,
        stg_tau: float = 0.0,
        stg_gamma: float = 1.0,
        stg_floor: float = 0.05,
        crsa_alpha: float = 1.0,
        crsa_center: str = "median",
        stg_residual_lambda: float = 0.25,
        # --- ACG-STG: Adaptive Gate hyperparameters ---
        stg_adaptive_gate: str = "none",        # "none" | "stats3" | "mlp"
        stg_gate_detach_stats: bool = True,
        stg_gate_init_bias: float = 0.0,
        stg_gate_log: bool = True,
        **_unused,
    ):
        super().__init__()
        self.segment_count = int(segment_count)
        self.frames_per_segment = int(frames_per_segment)
        self.selection_mode = str(selection_mode)
        self.topk = int(topk)

        # STG-CRSA config
        self.stg_temperature = float(stg_temperature)
        self.stg_tau = float(stg_tau)
        self.stg_gamma = float(stg_gamma)
        self.stg_floor = float(stg_floor)
        self.crsa_alpha = float(crsa_alpha)
        self.crsa_center = str(crsa_center).lower()
        if self.crsa_center not in ("median", "mean"):
            raise ValueError(f"crsa_center must be 'median' or 'mean', got '{crsa_center}'")
        self.stg_residual_lambda = float(stg_residual_lambda)

        # ACG-STG: Adaptive Gate config
        self.stg_adaptive_gate = str(stg_adaptive_gate).lower()
        if self.stg_adaptive_gate not in ("none", "stats3", "mlp"):
            raise ValueError(
                f"stg_adaptive_gate must be 'none', 'stats3', or 'mlp', got '{stg_adaptive_gate}'"
            )
        self.stg_gate_detach_stats = bool(stg_gate_detach_stats)
        self.stg_gate_init_bias = float(stg_gate_init_bias)
        self.stg_gate_log = bool(stg_gate_log)

        # Build adaptive gate module (only when selection_mode is stg_crsa_residual
        # and stg_adaptive_gate is not "none")
        if self.selection_mode == "stg_crsa_residual" and self.stg_adaptive_gate != "none":
            if self.stg_adaptive_gate == "stats3":
                self.acg_gate = _Stats3Gate(init_bias=self.stg_gate_init_bias)
            elif self.stg_adaptive_gate == "mlp":
                self.acg_gate = _MLPGate(init_bias=self.stg_gate_init_bias)
            else:
                self.acg_gate = None
        else:
            self.acg_gate = None

        # Debug tensors (populated per forward)
        self.last_segment_weights = None
        self.last_stg_raw_weights = None
        self.last_stg_gate = None
        self.last_stg_support = None
        self.last_stg_reliability = None
        self.last_stg_trust = None
        self.last_seg_probs = None

        # ACG-STG gate stats (populated per forward)
        self.last_gate_stats = None   # dict: gate, lambda_eff (detached, for logging)
        self.last_gate_raw = None     # [B,1] tensor (non-detached, for budget loss)

        backbone = create_qmil_backbone(
            backbone_source=backbone_source,
            timm_name=timm_name,
            pretrained=pretrained,
            backbone_name=backbone_name,
        )
        self.stem = backbone.stem
        self.stages = backbone.stages
        self.feature_dim = get_qmil_backbone_feature_dim(backbone)

        # --- Backbone identity metadata (used by MAE loader for integrity checks) ---
        # Must reflect the ACTUAL implementation source, not just the caller's arg.
        # resnet/efficientnet/mobilenet always use torchvision internally.
        _TORCHVISION_ONLY = {"resnet18", "resnet34", "efficientnet_b0", "mobilenet_v3_large"}
        self._qm_backbone_name = str(backbone_name)
        if self._qm_backbone_name in _TORCHVISION_ONLY:
            self._qm_backbone_source = "torchvision"
            self._qm_timm_name = None
        else:
            self._qm_backbone_source = str(backbone_source)
            self._qm_timm_name = str(timm_name) if timm_name else None

        self.quality_head = SegmentQualityMLP(in_features=self.feature_dim)
        self.frame_head = create_qmil_frame_head(self.feature_dim, num_classes)
        self.video_head = nn.Sequential(nn.Dropout(float(video_dropout)), nn.Linear(self.feature_dim, num_classes))

    # ----------------------------------------------------------------
    #  ACG-STG: Gate statistics computation
    # ----------------------------------------------------------------
    def _compute_gate_statistics(self, seg_probs, support, stg_raw_w):
        """Compute per-video statistics for the adaptive gate.

        Args:
            seg_probs: [B,S] segment-level malignant probabilities
            support:   [B,S] per-segment feature support (cosine-similarity based)
            stg_raw_w: [B,S] STG-CRSA raw weights (L1-normalized, sums to 1)

        Returns:
            dict with keys:
              seg_prob_std, top1_minus_mean, prob_range,
              neighbor_smoothness, support_mean, support_top1,
              entropy_or_concentration
            Each value is [B] float tensor.
        """
        B, S = seg_probs.shape

        # 1. seg_prob_std: standard deviation of segment malignant probs
        seg_prob_std = seg_probs.std(dim=1, unbiased=False)  # [B]

        # 2. top1_minus_mean
        seg_max = seg_probs.max(dim=1).values  # [B]
        seg_mean = seg_probs.mean(dim=1)       # [B]
        top1_minus_mean = seg_max - seg_mean   # [B]

        # 3. prob_range
        prob_range = seg_max - seg_probs.min(dim=1).values  # [B]

        # 4. neighbor_smoothness: 1 - mean(|p[i+1] - p[i]|)
        neighbor_diff = (seg_probs[:, 1:] - seg_probs[:, :-1]).abs().mean(dim=1)  # [B]
        neighbor_smoothness = 1.0 - neighbor_diff  # [B], approx in [0,1]

        # 5. support_mean
        support_mean = support.mean(dim=1)  # [B]

        # 6. support_top1: support of the top1 segment (by seg_prob)
        top1_idx = seg_probs.argmax(dim=1, keepdim=True)      # [B,1]
        support_top1 = support.gather(1, top1_idx).squeeze(-1)  # [B]

        # 7. entropy_or_concentration: 1 - normalized_entropy
        #    higher = more concentrated segment distribution
        eps = 1e-8
        probs_norm = seg_probs / (seg_probs.sum(dim=1, keepdim=True) + eps)
        log_probs = torch.log(probs_norm + eps)
        entropy = -(probs_norm * log_probs).sum(dim=1)  # [B]
        max_entropy = torch.log(torch.tensor(float(S), dtype=seg_probs.dtype, device=seg_probs.device))
        entropy_norm = entropy / (max_entropy + eps)    # [B] in [0,1]
        concentration = 1.0 - entropy_norm              # [B], higher = spikier

        return {
            "seg_prob_std": seg_prob_std,
            "top1_minus_mean": top1_minus_mean,
            "prob_range": prob_range,
            "neighbor_smoothness": neighbor_smoothness,
            "support_mean": support_mean,
            "support_top1": support_top1,
            "entropy_or_concentration": concentration,
        }

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

        if self.selection_mode == "topk_softmax":
            seg_scores = self.quality_head(seg_tokens).squeeze(-1)  # [B,S]
            seg_w = topk_softmax(seg_scores, k=self.topk)         # [B,S]
            self.last_stg_raw_weights = None
            self.last_stg_gate = None
            self.last_stg_support = None
            self.last_stg_reliability = None
            self.last_stg_trust = None
            self.last_seg_probs = None
            self.last_gate_stats = None
            self.last_gate_raw = None
        elif self.selection_mode == "mean":
            # Uniform mean pooling: no quality_head, no top-k selection
            seg_w = torch.full((B, S), 1.0 / S, device=seg_tokens.device, dtype=seg_tokens.dtype)
            self.last_stg_raw_weights = None
            self.last_stg_gate = None
            self.last_stg_support = None
            self.last_stg_reliability = None
            self.last_stg_trust = None
            self.last_seg_probs = None
            self.last_gate_stats = None
            self.last_gate_raw = None
        elif self.selection_mode in ("stg_crsa", "stg_crsa_residual"):
            # ---- STG-CRSA: Segment Trustworthiness Gating + CRSA ----
            # Steps 1-5 are shared between v1 (stg_crsa) and v2 (stg_crsa_residual).
            #
            # Step 1: Feature support via cosine similarity (CS-FSS heritage)
            tokens_norm = torch.nn.functional.normalize(seg_tokens, p=2, dim=-1)  # [B,S,D]
            cos_mat = tokens_norm @ tokens_norm.transpose(1, 2)       # [B,S,S]
            cos_sum = cos_mat.sum(dim=2)                              # [B,S]
            diag = cos_mat.diagonal(dim1=-2, dim2=-1)                # [B,S]
            support = (cos_sum - diag) / max(S - 1, 1)               # [B,S]  mean cos with others

            # Step 2: Z-score normalize support, then gate via sigmoid
            support_mean = support.mean(dim=1, keepdim=True)          # [B,1]
            support_std = support.std(dim=1, keepdim=True, unbiased=False)  # [B,1]
            support_z = (support - support_mean) / (support_std + 1e-6)     # [B,S]
            gate = torch.sigmoid((support_z - self.stg_tau) / self.stg_temperature)  # [B,S]

            # Step 3: Trust = floor + gate^gamma  (low-trust segments suppressed)
            trust = self.stg_floor + gate.pow(self.stg_gamma)         # [B,S]

            # Step 4: CRSA reliability = exp(-alpha * |prob - center|)
            seg_logits = self.video_head(seg_tokens)                  # [B,S,2]
            seg_probs = torch.softmax(seg_logits, dim=-1)[..., 1]     # [B,S]
            if self.crsa_center == "median":
                center = seg_probs.median(dim=1, keepdim=True).values  # [B,1]
            else:
                center = seg_probs.mean(dim=1, keepdim=True)           # [B,1]
            reliability = torch.exp(-self.crsa_alpha * torch.abs(seg_probs - center))  # [B,S]

            # Step 5: Gated aggregation — trust × reliability, then L1 normalize
            raw_w = trust * reliability                                # [B,S]
            stg_raw_w = raw_w / (raw_w.sum(dim=1, keepdim=True) + 1e-8)  # [B,S]

            if self.selection_mode == "stg_crsa_residual":
                # ---- v2: Residual mix with uniform weights ----
                # Mean pooling retains clean baseline stability;
                # STG-CRSA acts as a weak calibration term via lambda.
                uniform_w = torch.full((B, S), 1.0 / S, device=stg_raw_w.device, dtype=stg_raw_w.dtype)

                if self.acg_gate is not None:
                    # ---- ACG-STG v1: adaptive video-level gate ----
                    stats_dict = self._compute_gate_statistics(seg_probs, support, stg_raw_w)

                    # Build input tensor in canonical order
                    if self.stg_adaptive_gate == "stats3":
                        stats_keys = ["seg_prob_std", "neighbor_smoothness", "support_mean"]
                    else:  # mlp
                        stats_keys = [
                            "seg_prob_std", "top1_minus_mean", "prob_range",
                            "neighbor_smoothness", "support_mean", "support_top1",
                            "entropy_or_concentration",
                        ]
                    stats_tensor = torch.stack([stats_dict[k] for k in stats_keys], dim=1)  # [B, n_stats]

                    if self.stg_gate_detach_stats:
                        stats_tensor = stats_tensor.detach()

                    g = self.acg_gate(stats_tensor)  # [B, 1] in (0,1)
                    lambda_eff = self.stg_residual_lambda * g  # [B, 1]

                    seg_w = (1.0 - lambda_eff) * uniform_w + lambda_eff * stg_raw_w
                    seg_w = seg_w / (seg_w.sum(dim=1, keepdim=True) + 1e-8)

                    # Store gate stats
                    self.last_gate_raw = g
                    if self.stg_gate_log:
                        self.last_gate_stats = {
                            "gate": g.detach(),
                            "lambda_eff": lambda_eff.detach(),
                        }
                    else:
                        self.last_gate_stats = None
                else:
                    # Original fixed-lambda path (unchanged)
                    lam = self.stg_residual_lambda
                    seg_w = (1.0 - lam) * uniform_w + lam * stg_raw_w
                    seg_w = seg_w / (seg_w.sum(dim=1, keepdim=True) + 1e-8)
                    self.last_gate_stats = None
                    self.last_gate_raw = None
            else:
                # v1: pure STG-CRSA (no residual)
                seg_w = stg_raw_w
                self.last_gate_stats = None
                self.last_gate_raw = None

            # Store debug tensors (detached)
            self.last_stg_raw_weights = stg_raw_w.detach()
            self.last_stg_gate = gate.detach()
            self.last_stg_support = support.detach()
            self.last_stg_reliability = reliability.detach()
            self.last_stg_trust = trust.detach()
            self.last_seg_probs = seg_probs.detach()
        else:
            raise ValueError(f"Unsupported selection_mode: {self.selection_mode}")

        self.last_segment_weights = seg_w.detach()

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
