import os
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import BUVImageDataset
from model import (
    create_qmil_backbone,
    get_qmil_backbone_feature_dim,
    create_qmil_frame_head,
)


class ImageClassifier(nn.Module):
    def __init__(
        self,
        num_classes=2,
        backbone_source="timm",
        timm_name="convnext_tiny.fb_in22k",
        pretrained=True,
    ):
        super().__init__()
        backbone = create_qmil_backbone(
            backbone_source=backbone_source,
            timm_name=timm_name,
            pretrained=pretrained,
        )
        self.stem = backbone.stem
        self.stages = backbone.stages
        self.feature_dim = get_qmil_backbone_feature_dim(backbone)
        self.frame_head = create_qmil_frame_head(self.feature_dim, num_classes)

    def forward(self, x):
        low = self.stem(x)
        feat = self.stages(low)
        if feat.dim() == 4:
            feat = feat.flatten(1)
        elif feat.dim() != 2:
            raise RuntimeError(f"Unexpected backbone output shape: {feat.shape}")
        return self.frame_head(feat)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_num = 0.0, 0, 0
    all_probs, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((preds == labels).sum().item())
        total_num += int(labels.size(0))

        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    mean_loss = total_loss / max(1, total_num)
    mean_acc = total_correct / max(1, total_num)

    probs_np = np.concatenate(all_probs, axis=0) if len(all_probs) > 0 else np.array([0.5], dtype=np.float32)
    labels_np = np.concatenate(all_labels, axis=0) if len(all_labels) > 0 else np.array([0], dtype=np.int64)
    try:
        auc = float(roc_auc_score(labels_np, probs_np))
    except ValueError:
        auc = 0.5

    return mean_loss, mean_acc, auc


def main():
    parser = argparse.ArgumentParser("Pretrain frame branch with static images")
    parser.add_argument("--data_root", type=str, default="busimage")
    parser.add_argument("--save_path", type=str, default="frame_branch_pretrain_best.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone_source", type=str, default="timm", choices=["torchvision", "timm"])
    parser.add_argument("--timm_name", type=str, default="convnext_tiny.fb_in22k")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    all_dataset = BUVImageDataset(
        root_dir=args.data_root,
        img_size=args.img_size,
        train_mode=True,
    )
    all_labels = np.array(all_dataset.get_labels(), dtype=np.int64)
    all_samples = list(all_dataset.samples)
    if len(all_samples) == 0:
        raise RuntimeError(
            f"No images found under {args.data_root}. "
            "Expected class folders: benign/ and malignant/ (flat or nested)."
        )

    idx = np.arange(len(all_samples))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=args.val_ratio,
        random_state=args.seed,
        shuffle=True,
        stratify=all_labels if len(np.unique(all_labels)) > 1 else None,
    )

    train_samples = [all_samples[i] for i in train_idx]
    val_samples = [all_samples[i] for i in val_idx]

    train_dataset = BUVImageDataset(
        root_dir=args.data_root,
        img_size=args.img_size,
        train_mode=True,
        samples=train_samples,
    )
    val_dataset = BUVImageDataset(
        root_dir=args.data_root,
        img_size=args.img_size,
        train_mode=False,
        samples=val_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
    )

    model = ImageClassifier(
        num_classes=2,
        backbone_source=args.backbone_source,
        timm_name=args.timm_name,
        pretrained=args.pretrained,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_auc = -1.0
    best_epoch = -1
    save_path = args.save_path

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = logits.argmax(dim=1)
            running_loss += float(loss.item()) * images.size(0)
            running_correct += int((preds == labels).sum().item())
            running_total += int(labels.size(0))

            pbar.set_postfix(
                loss=float(loss.item()),
                acc=float(running_correct / max(1, running_total)),
                lr=float(optimizer.param_groups[0]["lr"]),
            )

        train_loss = running_loss / max(1, running_total)
        train_acc = running_correct / max(1, running_total)

        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best checkpoint to: {save_path}")

    print(f"Finished. best_epoch={best_epoch}, best_val_auc={best_auc:.4f}, ckpt={save_path}")


if __name__ == "__main__":
    main()
