# train_cv_with_your_model_last_epoch.py
# - No early stopping
# - DO NOT save best model
# - Evaluate and report using LAST epoch checkpoint only (per fold)
# - Logs per-epoch AUC + metrics @0.5 and @Youden for analysis

import os
import json
import random
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, accuracy_score,
    confusion_matrix, roc_curve
)
from sklearn.model_selection import StratifiedKFold

from dataset import BUVFolderDataset
from model import VideoClassifier
from loss import combined_loss


# ==========================================================
#  REPRODUCIBILITY
# ==========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ==========================================================
#  DEVICE
# ==========================================================
GPU_IDS = [0]
if GPU_IDS is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, GPU_IDS))
    print(f"Set CUDA_VISIBLE_DEVICES to: {GPU_IDS}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_MULTI_GPU = torch.cuda.device_count() > 1
print(f"Device: {DEVICE} | GPUs visible: {torch.cuda.device_count()}")

torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# ==========================================================
#  PATHS (server: BUV under /data/niyit/)
# ==========================================================
DATA_ROOT = os.environ.get("VIDEO_ROOT", "/data/niyit/Miccai 2022 BUV Dataset/rawframes")
SAVE_DIR = os.environ.get("SAVE_DIR", "cross_val_results_last_epoch")
os.makedirs(SAVE_DIR, exist_ok=True)


# ==========================================================
#  CONFIG
# ==========================================================
SEGMENTS = 6
FRAMES_PER_SEG = 3
IMG_SIZE = 224

N_FOLDS = 3
EPOCHS = 50

BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = torch.cuda.is_available()

MAX_NORM = 1.0

LR = 1e-4
WEIGHT_DECAY = 1e-3
ETA_MIN = 1e-6

# Frame-branch static-image pretrain controls
ENABLE_FRAME_BRANCH_PRETRAIN = True
FRAME_BRANCH_PRETRAIN_CKPT = os.environ.get("FRAME_BRANCH_PRETRAIN_CKPT", "frame_branch_pretrain_best.pth")
FRAME_BRANCH_PRETRAIN_LOAD_BACKBONE = True
FRAME_BRANCH_PRETRAIN_LOAD_FRAME_HEAD = True


# ==========================================================
#  METRICS
# ==========================================================
def compute_metrics(probs, y, threshold=0.5):
    try:
        auc = roc_auc_score(y, probs)
    except ValueError:
        auc = 0.5

    preds = (probs >= threshold).astype(int)
    f1 = f1_score(y, preds, zero_division=0)
    sens = recall_score(y, preds, pos_label=1, zero_division=0)
    spec = recall_score(y, preds, pos_label=0, zero_division=0)
    acc = accuracy_score(y, preds)
    cm = confusion_matrix(y, preds)

    return {
        "auc": float(auc),
        "threshold": float(threshold),
        "f1": float(f1),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
    }


def youden_threshold(probs, y):
    fpr, tpr, thresholds = roc_curve(y, probs)
    if len(thresholds) == 0:
        return 0.5
    idx = int(np.argmax(tpr - fpr))
    return float(thresholds[idx])


def _get_base_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


# ==========================================================
#  MODEL / OPTIM
# ==========================================================
def create_model():
    model = VideoClassifier(
        num_classes=2,
        segment_count=SEGMENTS,
        frames_per_segment=FRAMES_PER_SEG,
        backbone_source="timm",
        timm_name="convnext_tiny.fb_in22k",
        pretrained=True,
        selection_mode="topk_softmax",
        topk=2,
        video_dropout=0.5,
    )

    if ENABLE_FRAME_BRANCH_PRETRAIN:
        if os.path.isfile(FRAME_BRANCH_PRETRAIN_CKPT):
            ckpt = torch.load(FRAME_BRANCH_PRETRAIN_CKPT, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]

            if not isinstance(ckpt, dict):
                print(f"[FrameBranchPretrain] Invalid checkpoint format: {FRAME_BRANCH_PRETRAIN_CKPT}")
            else:
                normalized = {}
                for k, v in ckpt.items():
                    nk = k[7:] if k.startswith("module.") else k
                    normalized[nk] = v

                if FRAME_BRANCH_PRETRAIN_LOAD_BACKBONE:
                    backbone_state = {
                        k: v for k, v in normalized.items()
                        if k.startswith("stem.") or k.startswith("stages.")
                    }
                    if len(backbone_state) > 0:
                        load_info = model.load_state_dict(backbone_state, strict=False)
                        print(
                            "[FrameBranchPretrain] Backbone load: SUCCESS | "
                            f"loaded_keys={len(backbone_state)} | "
                            f"missing={len(load_info.missing_keys)} | "
                            f"unexpected={len(load_info.unexpected_keys)}"
                        )
                    else:
                        print("[FrameBranchPretrain] Backbone load: SKIPPED (no stem/stages keys in ckpt)")
                else:
                    print("[FrameBranchPretrain] Backbone load: DISABLED by config")

                if FRAME_BRANCH_PRETRAIN_LOAD_FRAME_HEAD:
                    frame_head_state = {}
                    if "frame_head.weight" in normalized:
                        frame_head_state["weight"] = normalized["frame_head.weight"]
                    if "frame_head.bias" in normalized:
                        frame_head_state["bias"] = normalized["frame_head.bias"]

                    if len(frame_head_state) > 0:
                        load_info = model.frame_head.load_state_dict(frame_head_state, strict=False)
                        print(
                            "[FrameBranchPretrain] Frame Head load: SUCCESS | "
                            f"loaded_keys={len(frame_head_state)} | "
                            f"missing={len(load_info.missing_keys)} | "
                            f"unexpected={len(load_info.unexpected_keys)}"
                        )
                    else:
                        print("[FrameBranchPretrain] Frame Head load: SKIPPED (no frame_head keys in ckpt)")
                else:
                    print("[FrameBranchPretrain] Frame Head load: DISABLED by config")
        else:
            print(f"[FrameBranchPretrain] Checkpoint not found: {FRAME_BRANCH_PRETRAIN_CKPT}")
    else:
        print("[FrameBranchPretrain] Disabled.")

    model = model.to(DEVICE)

    if USE_MULTI_GPU:
        model = torch.nn.DataParallel(model)
        print(f"Model wrapped with DataParallel (using {torch.cuda.device_count()} GPUs)")
    return model


def create_optimizer(model):
    base = _get_base_model(model)
    return torch.optim.AdamW(base.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    all_probs, all_y = [], []
    for videos, labels in loader:
        videos = videos.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        _, video_logits = model(videos)
        probs = torch.softmax(video_logits, dim=1)[:, 1]

        all_probs.append(probs.detach().cpu().numpy())
        all_y.append(labels.detach().cpu().numpy())

    return np.concatenate(all_probs, axis=0), np.concatenate(all_y, axis=0)


# ==========================================================
#  LOAD FULL INDEX
# ==========================================================
print("Loading full dataset index...")
full_dataset = BUVFolderDataset(
    root_dir=DATA_ROOT,
    segment_count=SEGMENTS,
    frames_per_segment=FRAMES_PER_SEG,
    img_size=IMG_SIZE,
    augment=False,
    test_mode=False,
)

all_video_ids = full_dataset.get_video_ids()
all_labels = np.array(full_dataset.get_labels(), dtype=np.int64)

print(f"Total videos: {len(all_video_ids)}")
print("Overall class distribution:", Counter(all_labels))


# ==========================================================
#  CV
# ==========================================================
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_results = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_video_ids, all_labels), start=1):
    print("\n" + "=" * 90)
    print(f"FOLD {fold_idx}/{N_FOLDS}")
    print("=" * 90)

    train_video_ids = [all_video_ids[i] for i in train_idx]
    val_video_ids = [all_video_ids[i] for i in val_idx]

    train_dataset = BUVFolderDataset(
        root_dir=DATA_ROOT,
        video_ids=train_video_ids,
        segment_count=SEGMENTS,
        frames_per_segment=FRAMES_PER_SEG,
        img_size=IMG_SIZE,
        augment=True,
        test_mode=False,
        enable_vflip=True,
        enable_rotation=True,
        rotation_angles=(-15, -10, -5, 5, 10, 15),
    )

    val_dataset = BUVFolderDataset(
        root_dir=DATA_ROOT,
        video_ids=val_video_ids,
        segment_count=SEGMENTS,
        frames_per_segment=FRAMES_PER_SEG,
        img_size=IMG_SIZE,
        augment=False,
        test_mode=True,
        enable_vflip=False,
        enable_rotation=False,
    )

    print("Train class distribution:", Counter(train_dataset.get_labels()))
    print("Val class distribution:  ", Counter(val_dataset.get_labels()))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )

    # Class weights for VIDEO loss only
    tr_counts = np.bincount(train_dataset.get_labels(), minlength=2)
    w = tr_counts.sum() / np.maximum(tr_counts, 1)
    class_weights_video = torch.tensor(w, dtype=torch.float32, device=DEVICE)

    model = create_model()
    optimizer = create_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=ETA_MIN)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    fold_dir = os.path.join(SAVE_DIR, f"fold{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    last_path = os.path.join(fold_dir, "last.pth")
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Fold {fold_idx} Epoch {epoch}/{EPOCHS}", leave=False)
        for videos, labels in pbar:
            videos = videos.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                frame_logits, video_logits = model(videos)
                loss, frame_loss_val, video_loss_val = combined_loss(
                    frame_logits, video_logits, labels,
                    class_weights_frame=None,
                    class_weights_video=class_weights_video,
                    frame_weight=0.6,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * videos.size(0)
            preds = video_logits.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

            pbar.set_postfix(
                loss=float(loss.item()),
                acc=float(running_correct / max(1, running_total)),
                frame=float(frame_loss_val),
                video=float(video_loss_val),
                lr=float(optimizer.param_groups[0]["lr"]),
            )

        train_loss = running_loss / max(1, running_total)
        train_acc = running_correct / max(1, running_total)

        # ---------------- VALIDATE (for logging only) ----------------
        val_probs, val_y = predict_probs(model, val_loader)
        try:
            val_auc = roc_auc_score(val_y, val_probs)
        except ValueError:
            val_auc = 0.5

        m05 = compute_metrics(val_probs, val_y, threshold=0.5)
        thr_y = youden_threshold(val_probs, val_y)
        my = compute_metrics(val_probs, val_y, threshold=thr_y)

        p = val_probs
        y = val_y
        pos_mean = float(p[y == 1].mean()) if np.any(y == 1) else float("nan")
        neg_mean = float(p[y == 0].mean()) if np.any(y == 0) else float("nan")

        log_row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_auc": float(val_auc),
            "val_metrics@0.5": m05,
            "val_threshold_youden": float(thr_y),
            "val_metrics@youden": my,
            "prob_stats": {
                "min": float(p.min()),
                "mean": float(p.mean()),
                "max": float(p.max()),
                "mean_pos": pos_mean,
                "mean_neg": neg_mean,
            },
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(log_row)

        print("\n" + "-" * 90)
        print(f"Fold {fold_idx} | Epoch {epoch}/{EPOCHS}")
        print(f"Train: loss={train_loss:.4f}, acc={train_acc:.3f}")
        print(f"Val:   AUC={val_auc:.4f}")
        print(f"Val@0.5:        F1={m05['f1']:.4f}, Sens={m05['sensitivity']:.4f}, Spec={m05['specificity']:.4f}, Acc={m05['accuracy']:.4f}")
        print(f"Val@Youden({thr_y:.3f}): F1={my['f1']:.4f}, Sens={my['sensitivity']:.4f}, Spec={my['specificity']:.4f}, Acc={my['accuracy']:.4f}")
        print("CM@0.5:\n", np.array(m05["confusion_matrix"]))
        print("CM@Youden:\n", np.array(my["confusion_matrix"]))
        print("-" * 90)

        scheduler.step()

        # Save ONLY LAST checkpoint (overwrite each epoch)
        torch.save(model.state_dict(), last_path)

    # ======================================================
    # Final evaluation USING LAST EPOCH
    # ======================================================
    model.load_state_dict(torch.load(last_path, map_location=DEVICE))
    val_probs, val_y = predict_probs(model, val_loader)
    try:
        final_auc = roc_auc_score(val_y, val_probs)
    except ValueError:
        final_auc = 0.5

    final_m05 = compute_metrics(val_probs, val_y, threshold=0.5)
    final_thr_y = youden_threshold(val_probs, val_y)
    final_my = compute_metrics(val_probs, val_y, threshold=final_thr_y)

    fold_summary = {
        "fold": fold_idx,
        "final_eval_auc_last": float(final_auc),
        "final_eval@0.5": final_m05,
        "final_threshold_youden": float(final_thr_y),
        "final_eval@youden": final_my,
        "segment_count": SEGMENTS,
        "frames_per_segment": FRAMES_PER_SEG,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "notes": "No best checkpoint; report last epoch only. Also report both @0.5 and @Youden.",
    }
    fold_results.append(fold_summary)

    with open(os.path.join(fold_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(fold_dir, "results_last.json"), "w") as f:
        json.dump(fold_summary, f, indent=2)

    print("\n" + "=" * 90)
    print(f"FOLD {fold_idx} FINAL (LAST EPOCH)")
    print(f"AUC: {final_auc:.4f}")
    print(f"@0.5:          F1={final_m05['f1']:.4f}, Sens={final_m05['sensitivity']:.4f}, Spec={final_m05['specificity']:.4f}, Acc={final_m05['accuracy']:.4f}")
    print(f"@Youden({final_thr_y:.3f}): F1={final_my['f1']:.4f}, Sens={final_my['sensitivity']:.4f}, Spec={final_my['specificity']:.4f}, Acc={final_my['accuracy']:.4f}")
    print("=" * 90)

# ---------------- CV Summary (LAST EPOCH) ----------------
def ms(x):
    return float(np.mean(x)), float(np.std(x))

vals_auc = [fr["final_eval_auc_last"] for fr in fold_results]

# @0.5
vals_f1_05 = [fr["final_eval@0.5"]["f1"] for fr in fold_results]
vals_se_05 = [fr["final_eval@0.5"]["sensitivity"] for fr in fold_results]
vals_sp_05 = [fr["final_eval@0.5"]["specificity"] for fr in fold_results]
vals_ac_05 = [fr["final_eval@0.5"]["accuracy"] for fr in fold_results]

# @Youden
vals_f1_y = [fr["final_eval@youden"]["f1"] for fr in fold_results]
vals_se_y = [fr["final_eval@youden"]["sensitivity"] for fr in fold_results]
vals_sp_y = [fr["final_eval@youden"]["specificity"] for fr in fold_results]
vals_ac_y = [fr["final_eval@youden"]["accuracy"] for fr in fold_results]

auc_m, auc_s = ms(vals_auc)

f1_05_m, f1_05_s = ms(vals_f1_05)
se_05_m, se_05_s = ms(vals_se_05)
sp_05_m, sp_05_s = ms(vals_sp_05)
ac_05_m, ac_05_s = ms(vals_ac_05)

f1_y_m, f1_y_s = ms(vals_f1_y)
se_y_m, se_y_s = ms(vals_se_y)
sp_y_m, sp_y_s = ms(vals_sp_y)
ac_y_m, ac_y_s = ms(vals_ac_y)

print("\n" + "=" * 90)
print("3-FOLD CV SUMMARY (LAST EPOCH)")
print(f"AUC: {auc_m*100:.2f} ± {auc_s*100:.2f}%")

print("\nMetrics @0.5")
print(f"F1         : {f1_05_m*100:.2f} ± {f1_05_s*100:.2f}%")
print(f"Sensitivity: {se_05_m*100:.2f} ± {se_05_s*100:.2f}%")
print(f"Specificity: {sp_05_m*100:.2f} ± {sp_05_s*100:.2f}%")
print(f"Accuracy   : {ac_05_m*100:.2f} ± {ac_05_s*100:.2f}%")

print("\nMetrics @Youden")
print(f"F1         : {f1_y_m*100:.2f} ± {f1_y_s*100:.2f}%")
print(f"Sensitivity: {se_y_m*100:.2f} ± {se_y_s*100:.2f}%")
print(f"Specificity: {sp_y_m*100:.2f} ± {sp_y_s*100:.2f}%")
print(f"Accuracy   : {ac_y_m*100:.2f} ± {ac_y_s*100:.2f}%")
print("=" * 90)

with open(os.path.join(SAVE_DIR, "cross_validation_summary_last.json"), "w") as f:
    json.dump(
        {
            "model": "VideoClassifier",
            "n_folds": N_FOLDS,
            "segment_count": SEGMENTS,
            "frames_per_segment": FRAMES_PER_SEG,
            "fold_results": fold_results,
            "summary": {
                "auc": {"mean": auc_m, "std": auc_s},
                "metrics@0.5": {
                    "f1": {"mean": f1_05_m, "std": f1_05_s},
                    "sensitivity": {"mean": se_05_m, "std": se_05_s},
                    "specificity": {"mean": sp_05_m, "std": sp_05_s},
                    "accuracy": {"mean": ac_05_m, "std": ac_05_s},
                },
                "metrics@youden": {
                    "f1": {"mean": f1_y_m, "std": f1_y_s},
                    "sensitivity": {"mean": se_y_m, "std": se_y_s},
                    "specificity": {"mean": sp_y_m, "std": sp_y_s},
                    "accuracy": {"mean": ac_y_m, "std": ac_y_s},
                },
            },
        },
        f,
        indent=2,
    )

print(f"\nSaved everything to: {SAVE_DIR}")
