# pretrain_busi_mae.py
# BUSI-MAE self-supervised pretraining for ConvNeXt-Tiny
# Mask-then-encode: random patches → 0, encoder sees only visible content
#
# Data: BUVImageDataset → all benign/malignant images, labels ignored
# Output: checkpoint with encoder_state_dict for downstream fine-tuning

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BUVImageDataset
from mae_model import ConvNeXtMAE


# -------------------------------------------------------
#  Helpers
# -------------------------------------------------------
def get_git_info():
    info = {"git_branch": "unknown", "git_commit": "unknown", "git_dirty": False}
    try:
        info["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
    except Exception:
        pass
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        pass
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        info["git_dirty"] = bool(out.strip())
    except Exception:
        pass
    return info


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------
#  Main
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("BUSI-MAE pretraining — ConvNeXt-Tiny")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root dir with benign/ and malignant/ subdirs")
    parser.add_argument("--save_path", type=str, default="busi_mae_convnext_tiny_best.pth")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Init encoder from ImageNet pretrained weights")
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MAE] Device: {device}")

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # ---------------------------------------------------------------
    #  Data
    # ---------------------------------------------------------------
    dataset = BUVImageDataset(
        root_dir=args.data_root,
        img_size=args.img_size,
        train_mode=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
    )
    print(f"[MAE] Dataset size: {len(dataset)} images")
    print(f"[MAE] Class distribution (ignored): "
          f"{dict(zip(*torch.bincount(torch.tensor(dataset.get_labels(), dtype=torch.long)).unique(return_counts=True)))}")

    # ---------------------------------------------------------------
    #  Model
    # ---------------------------------------------------------------
    model = ConvNeXtMAE(
        img_size=args.img_size,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        pretrained=args.pretrained,
    ).to(device)
    params = model.count_params()
    print(f"[MAE] Backbone: convnext_tiny  feature_dim={model.feature_dim}")
    print(f"[MAE] Params — total={params['total']:,}  encoder={params['encoder']:,}  "
          f"decoder={params['decoder']:,}  trainable={params['trainable']:,}")
    print(f"[MAE] mask_ratio={args.mask_ratio}  patch_size={args.patch_size}  "
          f"img_size={args.img_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # ---------------------------------------------------------------
    #  Training
    # ---------------------------------------------------------------
    best_loss = float("inf")
    best_epoch = -1
    loss_history = []
    save_path_best = args.save_path
    save_path_last = save_path_best.replace(".pth", "_last.pth")
    if save_path_last == save_path_best:
        save_path_last = save_path_best.replace(".pth", "") + "_last.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_full = 0.0

        pbar = tqdm(loader, desc=f"MAE Epoch {epoch}/{args.epochs}", leave=False)
        for images, _labels in pbar:
            images = images.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                masked_loss, recon, pixel_mask, ml_val, fl_val = model(images)
            scaler.scale(masked_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += ml_val.item() * images.size(0)
            running_full += fl_val.item() * images.size(0)

            pbar.set_postfix(
                masked=f"{ml_val.item():.6f}",
                full=f"{fl_val.item():.6f}",
            )

        train_masked = running_loss / len(dataset)
        train_full = running_full / len(dataset)
        loss_history.append({
            "epoch": epoch,
            "masked_loss": float(train_masked),
            "full_loss": float(train_full),
            "lr": float(optimizer.param_groups[0]["lr"]),
        })

        print(f"[Epoch {epoch:04d}]  masked_loss={train_masked:.6f}  "
              f"full_loss={train_full:.6f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        is_best = train_masked < best_loss
        if is_best:
            best_loss = train_masked
            best_epoch = epoch

        scheduler.step()

        # Save last checkpoint
        _save_checkpoint(model, params, args, save_path_last,
                         best_loss=best_loss, best_epoch=best_epoch,
                         latest_epoch=epoch, loss_history=loss_history,
                         is_best=False)

        # Save best checkpoint
        if is_best:
            _save_checkpoint(model, params, args, save_path_best,
                             best_loss=best_loss, best_epoch=best_epoch,
                             latest_epoch=epoch, loss_history=loss_history,
                             is_best=True)
            print(f"  -> Saved best checkpoint to: {save_path_best}")

    print(f"[MAE] Finished.  best_epoch={best_epoch}  best_masked_loss={best_loss:.6f}")


def _save_checkpoint(model, params, args, path, best_loss, best_epoch,
                     latest_epoch, loss_history, is_best):
    git_info = get_git_info()
    ckpt = {
        "pretrain_type": "mae",
        "backbone_name": "convnext_tiny",
        "feature_dim": model.feature_dim,
        "encoder_state_dict": model.encoder.state_dict(),
        "decoder_state_dict": model.decoder.state_dict(),
        "mask_ratio": args.mask_ratio,
        "patch_size": args.patch_size,
        "img_size": args.img_size,
        "total_params": int(params["total"]),
        "trainable_params": int(params["trainable"]),
        "encoder_params": int(params["encoder"]),
        "decoder_params": int(params["decoder"]),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "data_root": str(args.data_root),
        "git_branch": git_info["git_branch"],
        "git_commit": git_info["git_commit"],
        "git_dirty": git_info["git_dirty"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "best_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "latest_epoch": int(latest_epoch),
        "loss_history": loss_history,
        "is_best": bool(is_best),
    }
    torch.save(ckpt, path)


if __name__ == "__main__":
    main()
