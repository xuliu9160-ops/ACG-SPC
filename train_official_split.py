import json
import os
import random
import subprocess
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BUVFolderDataset
from loss import combined_loss
from model import VideoClassifier


# ==========================================================
#  REPRODUCIBILITY
# ==========================================================
SEED = int(os.environ.get("SEED", "42"))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


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
#  PATHS
# ==========================================================
DATA_ROOT = os.environ.get("VIDEO_ROOT", "/data/niyit/Miccai 2022 BUV Dataset/rawframes")
DATASET_ROOT = os.path.dirname(DATA_ROOT.rstrip("/\\"))
OFFICIAL_TRAIN_JSON = os.environ.get(
    "OFFICIAL_TRAIN_JSON",
    os.path.join(DATASET_ROOT, "imagenet_vid_train_15frames.json"),
)
FIXED_TEST_JSON = os.environ.get(
    "FIXED_TEST_JSON",
    os.path.join(DATASET_ROOT, "imagenet_vid_val.json"),
)
SAVE_DIR = os.environ.get("SAVE_DIR", "official_split_results")
os.makedirs(SAVE_DIR, exist_ok=True)


# ==========================================================
#  CONFIG
# ==========================================================
SEGMENTS = 6
FRAMES_PER_SEG = 3
IMG_SIZE = 224

EPOCHS = int(os.environ.get("EPOCHS", "50"))
BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = torch.cuda.is_available()

MAX_NORM = 1.0
LR = 1e-4
WEIGHT_DECAY = 1e-3
ETA_MIN = 1e-6

N_REPEATS = int(os.environ.get("N_REPEATS", "5"))
INTERNAL_VAL_RATIO = float(os.environ.get("INTERNAL_VAL_RATIO", "0.15"))

# Frame loss mode controls (default keeps baseline behavior)
FRAME_LOSS_MODE = os.environ.get("FRAME_LOSS_MODE", "all_frame")
MIL_SELECT_LEVEL = os.environ.get("MIL_SELECT_LEVEL", "segment")
MIL_POS_TOPK_SEGMENTS = int(os.environ.get("MIL_POS_TOPK_SEGMENTS", "1"))
MIL_MALIGNANT_CLASS_IDX = int(os.environ.get("MIL_MALIGNANT_CLASS_IDX", "1"))
MIL_SELECTION_DETACH = os.environ.get("MIL_SELECTION_DETACH", "1") not in ["0", "false", "False"]

# Frame-branch static-image pretrain controls
ENABLE_FRAME_BRANCH_PRETRAIN = True
FRAME_BRANCH_PRETRAIN_CKPT = os.environ.get("FRAME_BRANCH_PRETRAIN_CKPT", "frame_branch_pretrain_best.pth")
FRAME_BRANCH_PRETRAIN_LOAD_BACKBONE = True
FRAME_BRANCH_PRETRAIN_LOAD_FRAME_HEAD = True


# ==========================================================
#  METRICS
# ==========================================================
def safe_auc(y, probs):
    try:
        return float(roc_auc_score(y, probs))
    except ValueError:
        return 0.5


def safe_auprc(y, probs):
    try:
        return float(average_precision_score(y, probs))
    except ValueError:
        # Fallback for degenerate label distributions
        return float(np.mean(y)) if len(y) > 0 else 0.0


def compute_metrics(probs, y, threshold=0.5):
    auc = safe_auc(y, probs)
    auprc = safe_auprc(y, probs)

    preds = (probs >= threshold).astype(int)
    f1 = f1_score(y, preds, zero_division=0)
    sens = recall_score(y, preds, pos_label=1, zero_division=0)
    spec = recall_score(y, preds, pos_label=0, zero_division=0)
    acc = accuracy_score(y, preds)
    cm = confusion_matrix(y, preds, labels=[0, 1])

    return {
        "auc": float(auc),
        "auprc": float(auprc),
        "threshold": float(threshold),
        "f1": float(f1),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
    }


def youden_threshold(probs, y):
    try:
        fpr, tpr, thresholds = roc_curve(y, probs)
    except ValueError:
        return 0.5
    if len(thresholds) == 0:
        return 0.5
    idx = int(np.argmax(tpr - fpr))
    return float(thresholds[idx])


def _get_base_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


# ==========================================================
#  JSON PARSING
# ==========================================================
def parse_official_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "videos" not in data or not isinstance(data["videos"], list):
        raise ValueError(f"Invalid JSON schema in {json_path}: missing 'videos' list")

    label_map = {"benign": 0, "malignant": 1}
    entries = []
    for vid in data["videos"]:
        full_name = str(vid.get("name", "")).strip()
        if "/" not in full_name:
            raise ValueError(f"Invalid video name '{full_name}' in {json_path}")

        class_name, video_id = full_name.split("/", 1)
        class_name = class_name.strip().lower()
        video_id = video_id.strip()

        if class_name not in label_map:
            raise ValueError(f"Unknown class prefix '{class_name}' in {json_path}")
        if len(video_id) == 0:
            raise ValueError(f"Empty video_id in {json_path} for name='{full_name}'")

        entries.append(
            {
                "id": vid.get("id", None),
                "full_name": full_name,
                "class_name": class_name,
                "video_id": video_id,
                "label": int(label_map[class_name]),
            }
        )
    return entries


def ensure_rawframes_exist(entries, root_dir):
    missing = []
    for e in entries:
        folder = os.path.join(root_dir, e["class_name"], e["video_id"])
        if not os.path.isdir(folder):
            missing.append(folder)
    return missing


def entries_to_video_ids(entries):
    return [e["video_id"] for e in entries]


def entries_to_labels(entries):
    return np.array([int(e["label"]) for e in entries], dtype=np.int64)


def class_counter_from_entries(entries):
    return Counter(entries_to_labels(entries))


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
                        k: v for k, v in normalized.items() if k.startswith("stem.") or k.startswith("stages.")
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
        probs = torch.softmax(video_logits, dim=1)[:, 1]  # malignant prob

        all_probs.append(probs.detach().cpu().numpy())
        all_y.append(labels.detach().cpu().numpy())

    return np.concatenate(all_probs, axis=0), np.concatenate(all_y, axis=0)


def ms(x):
    return float(np.mean(x)), float(np.std(x))


def get_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    print("Official split protocol started.")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"OFFICIAL_TRAIN_JSON: {OFFICIAL_TRAIN_JSON}")
    print(f"FIXED_TEST_JSON: {FIXED_TEST_JSON}")
    print(f"SAVE_DIR: {SAVE_DIR}")
    print(
        "Frame loss config: "
        f"mode={FRAME_LOSS_MODE}, "
        f"mil_select_level={MIL_SELECT_LEVEL}, "
        f"mil_pos_topk_segments={MIL_POS_TOPK_SEGMENTS}, "
        f"mil_malignant_class_idx={MIL_MALIGNANT_CLASS_IDX}, "
        f"mil_selection_detach={MIL_SELECTION_DETACH}"
    )

    train_entries = parse_official_json(OFFICIAL_TRAIN_JSON)
    fixed_test_entries = parse_official_json(FIXED_TEST_JSON)

    missing_train = ensure_rawframes_exist(train_entries, DATA_ROOT)
    missing_test = ensure_rawframes_exist(fixed_test_entries, DATA_ROOT)
    if len(missing_train) > 0 or len(missing_test) > 0:
        msg = {
            "missing_train_folders": missing_train[:10],
            "missing_test_folders": missing_test[:10],
            "missing_train_count": len(missing_train),
            "missing_test_count": len(missing_test),
        }
        raise FileNotFoundError(f"Found missing rawframes folders: {json.dumps(msg, ensure_ascii=False)}")

    train_set_keys = {(e["class_name"], e["video_id"]) for e in train_entries}
    test_set_keys = {(e["class_name"], e["video_id"]) for e in fixed_test_entries}
    overlap = train_set_keys & test_set_keys
    if len(overlap) > 0:
        raise ValueError(f"Official train and fixed test overlap detected: {len(overlap)}")

    print(f"Official train videos: {len(train_entries)} | class dist: {class_counter_from_entries(train_entries)}")
    print(f"Fixed test videos: {len(fixed_test_entries)} | class dist: {class_counter_from_entries(fixed_test_entries)}")
    print(f"Train/Test overlap: {len(overlap)}")

    fixed_test_dataset = BUVFolderDataset(
        root_dir=DATA_ROOT,
        video_ids=entries_to_video_ids(fixed_test_entries),
        segment_count=SEGMENTS,
        frames_per_segment=FRAMES_PER_SEG,
        img_size=IMG_SIZE,
        augment=False,
        test_mode=True,
        enable_vflip=False,
        enable_rotation=False,
    )
    fixed_test_loader = DataLoader(
        fixed_test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )

    repeat_results = []
    all_train_labels = entries_to_labels(train_entries)
    all_indices = np.arange(len(train_entries))

    for repeat_idx in range(1, N_REPEATS + 1):
        repeat_seed = SEED + repeat_idx
        set_seed(repeat_seed)

        print("\n" + "=" * 100)
        print(f"REPEAT {repeat_idx}/{N_REPEATS} | split seed = {repeat_seed}")
        print("=" * 100)

        tr_idx, iv_idx = train_test_split(
            all_indices,
            test_size=INTERNAL_VAL_RATIO,
            random_state=repeat_seed,
            shuffle=True,
            stratify=all_train_labels,
        )
        tr_entries = [train_entries[i] for i in tr_idx]
        iv_entries = [train_entries[i] for i in iv_idx]

        train_dataset = BUVFolderDataset(
            root_dir=DATA_ROOT,
            video_ids=entries_to_video_ids(tr_entries),
            segment_count=SEGMENTS,
            frames_per_segment=FRAMES_PER_SEG,
            img_size=IMG_SIZE,
            augment=True,
            test_mode=False,
            enable_vflip=True,
            enable_rotation=True,
            rotation_angles=(-15, -10, -5, 5, 10, 15),
        )
        internal_val_dataset = BUVFolderDataset(
            root_dir=DATA_ROOT,
            video_ids=entries_to_video_ids(iv_entries),
            segment_count=SEGMENTS,
            frames_per_segment=FRAMES_PER_SEG,
            img_size=IMG_SIZE,
            augment=False,
            test_mode=True,
            enable_vflip=False,
            enable_rotation=False,
        )

        print(f"Train subset videos: {len(train_dataset)} | dist: {Counter(train_dataset.get_labels())}")
        print(f"Internal val videos:  {len(internal_val_dataset)} | dist: {Counter(internal_val_dataset.get_labels())}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=(NUM_WORKERS > 0),
        )
        internal_val_loader = DataLoader(
            internal_val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=(NUM_WORKERS > 0),
        )

        tr_counts = np.bincount(train_dataset.get_labels(), minlength=2)
        w = tr_counts.sum() / np.maximum(tr_counts, 1)
        class_weights_video = torch.tensor(w, dtype=torch.float32, device=DEVICE)

        model = create_model()
        optimizer = create_optimizer(model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=ETA_MIN)
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        repeat_dir = os.path.join(SAVE_DIR, f"repeat{repeat_idx}")
        os.makedirs(repeat_dir, exist_ok=True)
        best_path = os.path.join(repeat_dir, "best.pth")
        last_path = os.path.join(repeat_dir, "last.pth")

        history = []
        best_val_auc = -float("inf")
        best_epoch = -1
        best_val_auprc = 0.0
        best_youden_thr = 0.5
        best_m05 = None
        best_my = None

        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss, running_correct, running_total = 0.0, 0, 0

            pbar = tqdm(train_loader, desc=f"Repeat {repeat_idx} Epoch {epoch}/{EPOCHS}", leave=False)
            for videos, labels in pbar:
                videos = videos.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    frame_logits, video_logits = model(videos)
                    loss, frame_loss_val, video_loss_val = combined_loss(
                        frame_logits,
                        video_logits,
                        labels,
                        class_weights_frame=None,
                        class_weights_video=class_weights_video,
                        frame_weight=0.6,
                        frame_loss_mode=FRAME_LOSS_MODE,
                        mil_select_level=MIL_SELECT_LEVEL,
                        mil_pos_topk_segments=MIL_POS_TOPK_SEGMENTS,
                        mil_malignant_class_idx=MIL_MALIGNANT_CLASS_IDX,
                        mil_selection_detach=MIL_SELECTION_DETACH,
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
                    frame=float(frame_loss_val.item()),
                    video=float(video_loss_val.item()),
                    lr=float(optimizer.param_groups[0]["lr"]),
                )

            train_loss = running_loss / max(1, running_total)
            train_acc = running_correct / max(1, running_total)

            iv_probs, iv_y = predict_probs(model, internal_val_loader)
            iv_auc = safe_auc(iv_y, iv_probs)
            iv_auprc = safe_auprc(iv_y, iv_probs)
            iv_m05 = compute_metrics(iv_probs, iv_y, threshold=0.5)
            iv_thr_y = youden_threshold(iv_probs, iv_y)
            iv_my = compute_metrics(iv_probs, iv_y, threshold=iv_thr_y)

            is_best = iv_auc > best_val_auc
            if is_best:
                best_val_auc = float(iv_auc)
                best_val_auprc = float(iv_auprc)
                best_epoch = int(epoch)
                best_youden_thr = float(iv_thr_y)
                best_m05 = iv_m05
                best_my = iv_my
                torch.save(model.state_dict(), best_path)

            torch.save(model.state_dict(), last_path)

            history.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "internal_val_auc": float(iv_auc),
                    "internal_val_auprc": float(iv_auprc),
                    "internal_val_metrics@0.5": iv_m05,
                    "internal_val_youden_threshold": float(iv_thr_y),
                    "internal_val_metrics@youden": iv_my,
                    "is_best_epoch": bool(is_best),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
            )

            print("\n" + "-" * 100)
            print(
                f"Repeat {repeat_idx} | Epoch {epoch}/{EPOCHS} | "
                f"Train loss={train_loss:.4f}, acc={train_acc:.3f}"
            )
            print(
                f"Internal Val | AUC={iv_auc:.4f}, AUPRC={iv_auprc:.4f}, "
                f"best_auc={best_val_auc:.4f} @epoch {best_epoch}"
            )
            print("-" * 100)

            scheduler.step()

        best_val_result = {
            "best_epoch": int(best_epoch),
            "best_val_auc": float(best_val_auc),
            "best_val_auprc": float(best_val_auprc),
            "metrics@0.5": best_m05,
            "youden_threshold_on_internal_val": float(best_youden_thr),
            "metrics@youden_on_internal_val": best_my,
        }

        with open(os.path.join(repeat_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        with open(os.path.join(repeat_dir, "best_val_result.json"), "w", encoding="utf-8") as f:
            json.dump(best_val_result, f, indent=2)

        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        ft_probs, ft_y = predict_probs(model, fixed_test_loader)
        ft_auc = safe_auc(ft_y, ft_probs)
        ft_auprc = safe_auprc(ft_y, ft_probs)
        ft_m05 = compute_metrics(ft_probs, ft_y, threshold=0.5)
        ft_m_val_youden = compute_metrics(ft_probs, ft_y, threshold=best_youden_thr)

        fixed_test_result = {
            "best_epoch": int(best_epoch),
            "best_val_auc": float(best_val_auc),
            "best_val_auprc": float(best_val_auprc),
            "fixed_test_auc": float(ft_auc),
            "fixed_test_auprc": float(ft_auprc),
            "fixed_test_metrics@0.5": ft_m05,
            "internal_val_youden_threshold_used_on_test": float(best_youden_thr),
            "fixed_test_metrics@val_youden_on_test": ft_m_val_youden,
        }
        with open(os.path.join(repeat_dir, "fixed_test_result_best.json"), "w", encoding="utf-8") as f:
            json.dump(fixed_test_result, f, indent=2)

        repeat_results.append(
            {
                "repeat": int(repeat_idx),
                "split_seed": int(repeat_seed),
                "train_count": int(len(tr_entries)),
                "internal_val_count": int(len(iv_entries)),
                "fixed_test_count": int(len(fixed_test_entries)),
                "best_val_result": best_val_result,
                "fixed_test_result_best": fixed_test_result,
            }
        )

        print("\n" + "=" * 100)
        print(
            f"REPEAT {repeat_idx} DONE | "
            f"Best epoch={best_epoch}, Best internal AUC={best_val_auc:.4f}, "
            f"Fixed test AUC={ft_auc:.4f}, AUPRC={ft_auprc:.4f}"
        )
        print("=" * 100)

    vals_auc = [r["fixed_test_result_best"]["fixed_test_auc"] for r in repeat_results]
    vals_auprc = [r["fixed_test_result_best"]["fixed_test_auprc"] for r in repeat_results]
    vals_f1_05 = [r["fixed_test_result_best"]["fixed_test_metrics@0.5"]["f1"] for r in repeat_results]
    vals_se_05 = [r["fixed_test_result_best"]["fixed_test_metrics@0.5"]["sensitivity"] for r in repeat_results]
    vals_sp_05 = [r["fixed_test_result_best"]["fixed_test_metrics@0.5"]["specificity"] for r in repeat_results]
    vals_ac_05 = [r["fixed_test_result_best"]["fixed_test_metrics@0.5"]["accuracy"] for r in repeat_results]

    vals_f1_vy = [r["fixed_test_result_best"]["fixed_test_metrics@val_youden_on_test"]["f1"] for r in repeat_results]
    vals_se_vy = [
        r["fixed_test_result_best"]["fixed_test_metrics@val_youden_on_test"]["sensitivity"] for r in repeat_results
    ]
    vals_sp_vy = [
        r["fixed_test_result_best"]["fixed_test_metrics@val_youden_on_test"]["specificity"] for r in repeat_results
    ]
    vals_ac_vy = [r["fixed_test_result_best"]["fixed_test_metrics@val_youden_on_test"]["accuracy"] for r in repeat_results]

    auc_m, auc_s = ms(vals_auc)
    auprc_m, auprc_s = ms(vals_auprc)
    f1_05_m, f1_05_s = ms(vals_f1_05)
    se_05_m, se_05_s = ms(vals_se_05)
    sp_05_m, sp_05_s = ms(vals_sp_05)
    ac_05_m, ac_05_s = ms(vals_ac_05)
    f1_vy_m, f1_vy_s = ms(vals_f1_vy)
    se_vy_m, se_vy_s = ms(vals_se_vy)
    sp_vy_m, sp_vy_s = ms(vals_sp_vy)
    ac_vy_m, ac_vy_s = ms(vals_ac_vy)

    summary_payload = {
        "repeat_results": repeat_results,
        "fixed_test_summary": {
            "auc": {"mean": auc_m, "std": auc_s},
            "auprc": {"mean": auprc_m, "std": auprc_s},
            "metrics@0.5": {
                "f1": {"mean": f1_05_m, "std": f1_05_s},
                "sensitivity": {"mean": se_05_m, "std": se_05_s},
                "specificity": {"mean": sp_05_m, "std": sp_05_s},
                "accuracy": {"mean": ac_05_m, "std": ac_05_s},
            },
            "metrics@val_youden_on_test": {
                "f1": {"mean": f1_vy_m, "std": f1_vy_s},
                "sensitivity": {"mean": se_vy_m, "std": se_vy_s},
                "specificity": {"mean": sp_vy_m, "std": sp_vy_s},
                "accuracy": {"mean": ac_vy_m, "std": ac_vy_s},
            },
        },
        "config_snapshot": {
            "repeat_count": int(N_REPEATS),
            "train_internal_val_ratio": {"train": float(1.0 - INTERNAL_VAL_RATIO), "internal_val": float(INTERNAL_VAL_RATIO)},
            "official_train_json_path": OFFICIAL_TRAIN_JSON,
            "fixed_test_json_path": FIXED_TEST_JSON,
            "video_root": DATA_ROOT,
            "frame_loss_mode": FRAME_LOSS_MODE,
            "mil_pos_topk_segments": int(MIL_POS_TOPK_SEGMENTS),
            "mil_select_level": MIL_SELECT_LEVEL,
            "mil_malignant_class_idx": int(MIL_MALIGNANT_CLASS_IDX),
            "mil_selection_detach": bool(MIL_SELECTION_DETACH),
            "seed": int(SEED),
            "commit_hash": get_commit_hash(),
        },
    }

    summary_path = os.path.join(SAVE_DIR, "official_split_summary_best_test.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\n" + "=" * 100)
    print("OFFICIAL SPLIT SUMMARY (FIXED TEST, BEST INTERNAL-VAL AUC CHECKPOINT)")
    print(f"AUC   : {auc_m*100:.2f} ± {auc_s*100:.2f}%")
    print(f"AUPRC : {auprc_m*100:.2f} ± {auprc_s*100:.2f}%")
    print("Metrics@0.5")
    print(f"F1         : {f1_05_m*100:.2f} ± {f1_05_s*100:.2f}%")
    print(f"Sensitivity: {se_05_m*100:.2f} ± {se_05_s*100:.2f}%")
    print(f"Specificity: {sp_05_m*100:.2f} ± {sp_05_s*100:.2f}%")
    print(f"Accuracy   : {ac_05_m*100:.2f} ± {ac_05_s*100:.2f}%")
    print("Metrics@val_youden_on_test")
    print(f"F1         : {f1_vy_m*100:.2f} ± {f1_vy_s*100:.2f}%")
    print(f"Sensitivity: {se_vy_m*100:.2f} ± {se_vy_s*100:.2f}%")
    print(f"Specificity: {sp_vy_m*100:.2f} ± {sp_vy_s*100:.2f}%")
    print(f"Accuracy   : {ac_vy_m*100:.2f} ± {ac_vy_s*100:.2f}%")
    print("=" * 100)
    print(f"Saved all outputs to: {SAVE_DIR}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
