# mae_model.py
# ConvNeXt-Tiny MAE: mask-then-encode image reconstruction
#
# Design (mask-then-encode):
#   x [N,3,224,224]
#   1. build patch mask → pixel_mask [N,1,224,224]
#   2. x_masked = x * (1 - pixel_mask)   # masked patches → 0
#   3. feat = stages(stem(x_masked))      # [N,768,7,7]  (encoder sees NO masked content)
#   4. recon = decoder(feat)              # [N,3,224,224]
#   5. loss = MSE(recon * pixel_mask, x * pixel_mask)   # only on masked regions
#
# Default encoder: timm convnext_tiny.fb_in22k (same as downstream VideoClassifier).
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
