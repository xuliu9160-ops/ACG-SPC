# mae_model.py
# MAE (Masked Autoencoder) — mask-then-encode image reconstruction
#
# Supported backbones:
#   - convnext_tiny  (timm convnext_tiny.fb_in22k or torchvision)
#   - efficientnet_b0 (torchvision EfficientNet-B0)
#
# Design (mask-then-encode):
#   x [N,3,224,224]
#   1. build patch mask → pixel_mask [N,1,224,224]
#   2. x_masked = x * (1 - pixel_mask)   # masked patches → 0
#   3. feat = stages_body(stem(x_masked))  # encoder sees NO masked content
#   4. recon = decoder(feat)              # [N,3,224,224]
#   5. loss = MSE(recon * pixel_mask, x * pixel_mask)   # only on masked regions
#
# Stem/stages are byte-identical to downstream VideoClassifier by construction.
# This guarantees strict key compatibility when loading MAE weights into downstream.

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------
#  Mask utilities
# -------------------------------------------------------
def build_patch_mask(batch_size: int, grid_size: int = 7, mask_ratio: float = 0.75,
                     img_size: int = 224, device=None, dtype=torch.float32):
    """
    Create per-sample random patch masks.

    Args:
        batch_size: N
        grid_size:  img_size // patch_size (e.g. 7 for 224/32)
        mask_ratio: fraction of patches to mask (default 0.75)
        img_size:   spatial size (must be divisible by patch_size)

    Returns:
        pixel_mask: [N, 1, img_size, img_size] — 1.0 = masked, 0.0 = visible
        patch_mask: [N, grid_size * grid_size] — bool, True = masked
    """
    num_patches = grid_size * grid_size
    num_mask = max(1, int(num_patches * mask_ratio))

    noise = torch.rand(batch_size, num_patches, device=device)
    ids_shuffle = noise.argsort(dim=1)
    ids_restore = ids_shuffle.argsort(dim=1)
    patch_mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    patch_mask[:, :num_mask] = True
    patch_mask = patch_mask.gather(1, ids_restore)

    patch_size = img_size // grid_size
    pixel_mask = patch_mask.float().view(batch_size, grid_size, grid_size)
    pixel_mask = pixel_mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    pixel_mask = pixel_mask.unsqueeze(1)  # [N, 1, img_size, img_size]

    return pixel_mask, patch_mask


# -------------------------------------------------------
#  ConvNeXt-MAE model (timm default — downstream-compatible)
# -------------------------------------------------------
class ConvNeXtMAE(nn.Module):
    """Mask-then-encode MAE using timm ConvNeXt-Tiny encoder (default)."""

    def __init__(self, backbone_source="timm", timm_name="convnext_tiny.fb_in22k",
                 backbone_name="convnext_tiny", img_size=224, patch_size=32,
                 mask_ratio=0.75, pretrained=True):
        super().__init__()

        backbone_source = str(backbone_source).lower()
        backbone_name = str(backbone_name).lower()

        if int(img_size) != 224:
            raise ValueError(
                f"Only img_size=224 is supported in this MAE implementation, got {img_size}"
            )
        if int(img_size) % int(patch_size) != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
            )

        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.grid_size = self.img_size // self.patch_size
        self.mask_ratio = float(mask_ratio)
        self.backbone_source = backbone_source
        self.timm_name = str(timm_name)
        self.backbone_name = backbone_name

        if backbone_source == "timm":
            import timm
            raw = timm.create_model(self.timm_name, pretrained=bool(pretrained), num_classes=0)
            self.stem = raw.stem
            self.stages = raw.stages
            self.feature_dim = 768
        elif backbone_source == "torchvision":
            from torchvision.models.convnext import convnext_tiny, ConvNeXt_Tiny_Weights
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            raw = convnext_tiny(weights=weights)
            self.stem = nn.Sequential(raw.features[0])
            self.stages = nn.Sequential(*raw.features[1:])
            self.feature_dim = 768
        else:
            raise ValueError(f"Unsupported backbone_source: {backbone_source}")

        self.decoder = self._build_decoder()

    # ---------------------------------------------------
    #  Decoder: 768-dim 7×7 → 3-channel 224×224
    # ---------------------------------------------------
    def _build_decoder(self):
        return nn.Sequential(
            nn.Conv2d(768, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, 3, padding=1),
            nn.PixelShuffle(2),                     # 7→14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.PixelShuffle(2),                     # 14→28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.PixelShuffle(2),                     # 28→56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.PixelShuffle(2),                     # 56→112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 12, 3, padding=1),
            nn.PixelShuffle(2),                     # 112→224
        )

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        pixel_mask, _patch_mask = build_patch_mask(
            B, self.grid_size, self.mask_ratio,
            img_size=self.img_size, device=x.device, dtype=x.dtype,
        )
        x_masked = x * (1.0 - pixel_mask)

        # Encode: stem + stages (same as downstream VideoClassifier)
        low = self.stem(x_masked)
        feat = self.stages(low)  # [N, 768, grid_size, grid_size]

        recon = self.decoder(feat)  # [N, 3, img_size, img_size]

        diff = (recon - x) ** 2
        masked_diff = diff * pixel_mask
        masked_pixels = pixel_mask.sum() + 1e-8
        masked_loss = masked_diff.sum() / masked_pixels

        full_pixels = B * 3 * self.img_size * self.img_size
        full_loss = diff.sum() / full_pixels

        return masked_loss, recon, pixel_mask, masked_loss.detach(), full_loss.detach()

    def count_params(self):
        enc = sum(p.numel() for p in self.stem.parameters()) + \
              sum(p.numel() for p in self.stages.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": enc + dec,
            "trainable": trainable,
            "encoder": enc,
            "decoder": dec,
        }


# -------------------------------------------------------
#  EfficientNet-B0 MAE model (torchvision)
# -------------------------------------------------------
class EfficientNetB0MAE(nn.Module):
    """
    Mask-then-encode MAE using torchvision EfficientNet-B0 encoder.

    Stem/stages are byte-identical to downstream VideoClassifier
    (_EfficientNetStemStages wrapper in model.py), guaranteeing
    strict=True key compatibility when loading MAE weights into downstream.

    Encoder: stem + stages  →  [N, 1280, 7, 7] feature map
    Decoder: PixelShuffle ×5 →  [N, 3, 224, 224] reconstruction
    """

    def __init__(self, img_size=224, patch_size=32, mask_ratio=0.75, pretrained=True):
        super().__init__()

        if int(img_size) != 224:
            raise ValueError(
                f"Only img_size=224 is supported in this MAE implementation, got {img_size}"
            )
        if int(img_size) % int(patch_size) != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
            )

        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.grid_size = self.img_size // self.patch_size  # 7
        self.mask_ratio = float(mask_ratio)
        self.backbone_source = "torchvision"
        self.backbone_name = "efficientnet_b0"
        self.feature_dim = 1280

        # ---------------------------------------------------
        #  Build encoder: identical to downstream _EfficientNetStemStages
        #  (model.py create_qmil_backbone → _EfficientNetStemStages)
        # ---------------------------------------------------
        from torchvision.models import efficientnet_b0
        try:
            from torchvision.models import EfficientNet_B0_Weights
            w = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        except ImportError:
            w = pretrained  # older torchvision: pretrained is bool

        raw = efficientnet_b0(weights=w) if w is not None else efficientnet_b0()
        feats = list(raw.features)

        # stem: feats[0] + feats[1]  (same as downstream)
        self.stem = nn.Sequential(feats[0], feats[1])

        # stages: feats[2:] + avgpool  (same as downstream)
        # avgpool (AdaptiveAvgPool2d) has 0 parameters → 0 state_dict keys.
        # Including it makes stages byte-identical to downstream.
        self.stages = nn.Sequential(
            *feats[2:],
            raw.avgpool if hasattr(raw, "avgpool") else nn.AdaptiveAvgPool2d((1, 1)),
        )

        # --- Sanity check: feature_dim == 1280 ---
        actual_fd = int(raw.classifier[1].in_features)
        if actual_fd != 1280:
            raise AssertionError(
                f"Expected EfficientNet-B0 feature_dim=1280, got {actual_fd}. "
                f"This code only supports EfficientNet-B0."
            )

        # --- Sanity check: spatial grid == 7×7 ---
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            dummy_stem = self.stem(dummy)
            # stages_body = all layers except final avgpool
            dummy_body = dummy_stem
            for layer in self.stages[:-1]:
                dummy_body = layer(dummy_body)
            actual_grid = tuple(dummy_body.shape[2:])
            actual_channels = int(dummy_body.shape[1])
            if actual_grid != (7, 7):
                raise AssertionError(
                    f"Expected EfficientNet-B0 encoder spatial grid (7,7), "
                    f"got {actual_grid} with shape {tuple(dummy_body.shape)}. "
                    f"Input must be 224×224."
                )
            if actual_channels != 1280:
                raise AssertionError(
                    f"Expected EfficientNet-B0 encoder channels=1280, "
                    f"got {actual_channels}. Unexpected model structure."
                )
        print(f"[EfficientNetB0MAE] Encoder verified: feature_dim={self.feature_dim}, "
              f"spatial_grid={actual_grid}, channels={actual_channels}")

        # ---------------------------------------------------
        #  Decoder: 1280-dim 7×7 → 3-channel 224×224
        # ---------------------------------------------------
        self.decoder = self._build_decoder()

    def _build_decoder(self):
        """1280-dim 7×7 → 3-channel 224×224 via 5× PixelShuffle."""
        return nn.Sequential(
            nn.Conv2d(1280, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.PixelShuffle(2),                     # 7→14  (1024→256)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.PixelShuffle(2),                     # 14→28  (512→128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.PixelShuffle(2),                     # 28→56  (256→64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.PixelShuffle(2),                     # 56→112  (128→32)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 12, 3, padding=1),
            nn.PixelShuffle(2),                     # 112→224  (12→3)
        )

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        pixel_mask, _patch_mask = build_patch_mask(
            B, self.grid_size, self.mask_ratio,
            img_size=self.img_size, device=x.device, dtype=x.dtype,
        )
        x_masked = x * (1.0 - pixel_mask)

        # Encode: stem → stages_body (before avgpool)
        low = self.stem(x_masked)
        # self.stages[:-1] = EfficientNet feature blocks only (no avgpool)
        feat = low
        for layer in self.stages[:-1]:
            feat = layer(feat)
        # feat: [N, 1280, 7, 7]

        recon = self.decoder(feat)  # [N, 3, 224, 224]

        diff = (recon - x) ** 2
        masked_diff = diff * pixel_mask
        masked_pixels = pixel_mask.sum() + 1e-8
        masked_loss = masked_diff.sum() / masked_pixels

        full_pixels = B * 3 * self.img_size * self.img_size
        full_loss = diff.sum() / full_pixels

        return masked_loss, recon, pixel_mask, masked_loss.detach(), full_loss.detach()

    def count_params(self):
        # Encoder params = stem + stages (including avgpool which has 0 params)
        enc = sum(p.numel() for p in self.stem.parameters()) + \
              sum(p.numel() for p in self.stages.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": enc + dec,
            "trainable": trainable,
            "encoder": enc,
            "decoder": dec,
        }
