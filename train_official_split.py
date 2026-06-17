import json
import os
import random
import subprocess
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
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
from loss import combined_loss, pairwise_ranking_loss
from model import VideoClassifier, load_mae_encoder_from_checkpoint


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

EXTERNAL_CUDA = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if EXTERNAL_CUDA:
    print(f"[GPU] Respect external CUDA_VISIBLE_DEVICES={EXTERNAL_CUDA}")
else:
    gpu_ids_env = os.environ.get("GPU_IDS", "").strip()
    if gpu_ids_env:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_env
        print(f"[GPU] Set CUDA_VISIBLE_DEVICES from GPU_IDS env: {gpu_ids_env}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, GPU_IDS))
        print(f"[GPU] Set CUDA_VISIBLE_DEVICES from default GPU_IDS: {GPU_IDS}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_MULTI_GPU = torch.cuda.device_count() > 1
print(f"[GPU] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"[GPU] DEVICE: {DEVICE} | GPUs visible: {torch.cuda.device_count()}")

torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


DATASET_NAME = os.environ.get("DATASET_NAME", "BUV")

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
SEGMENTS = int(os.environ.get("SEGMENTS", "6"))
FRAMES_PER_SEG = int(os.environ.get("FRAMES_PER_SEG", "3"))
IMG_SIZE = 224

EPOCHS = int(os.environ.get("EPOCHS", "50"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
NUM_WORKERS = 4
PIN_MEMORY = torch.cuda.is_available()

MAX_NORM = 1.0
LR = 1e-4
WEIGHT_DECAY = 1e-3
ETA_MIN = 1e-6

N_REPEATS = int(os.environ.get("N_REPEATS", "5"))
INTERNAL_VAL_RATIO = float(os.environ.get("INTERNAL_VAL_RATIO", "0.15"))

# Experiment identity
EXPERIMENT_TAG = os.environ.get("EXPERIMENT_TAG", "buv_qmil_reverse_ablation")

# Frame loss mode controls
FRAME_LOSS_MODE = os.environ.get("FRAME_LOSS_MODE", "mil_selective")
MIL_SELECT_LEVEL = os.environ.get("MIL_SELECT_LEVEL", "segment")
MIL_POS_TOPK_SEGMENTS = int(os.environ.get("MIL_POS_TOPK_SEGMENTS", "1"))
MIL_MALIGNANT_CLASS_IDX = int(os.environ.get("MIL_MALIGNANT_CLASS_IDX", "1"))
MIL_SELECTION_DETACH = os.environ.get("MIL_SELECTION_DETACH", "1").lower() not in ("0", "false", "no")

FRAME_WEIGHT = float(os.environ.get("FRAME_WEIGHT", "0.6"))

# Frame-branch static-image pretrain controls
ENABLE_FRAME_BRANCH_PRETRAIN = os.environ.get("ENABLE_FRAME_BRANCH_PRETRAIN", "1").lower() in ("1", "true", "yes")
FRAME_BRANCH_PRETRAIN_CKPT = os.environ.get("FRAME_BRANCH_PRETRAIN_CKPT", "frame_branch_pretrain_best.pth")
FRAME_BRANCH_PRETRAIN_LOAD_BACKBONE = True
FRAME_BRANCH_PRETRAIN_LOAD_FRAME_HEAD = True

# Backbone selection
BACKBONE_NAME = os.environ.get("BACKBONE_NAME", "convnext_tiny")

# QMIL / aggregation controls
SELECTION_MODE = os.environ.get("SELECTION_MODE", "topk_softmax")
QMIL_TOPK = int(os.environ.get("QMIL_TOPK", "2"))

# ---- STG-CRSA hyperparameters (only used when SELECTION_MODE == "stg_crsa") ----
STG_TEMPERATURE = float(os.environ.get("STG_TEMPERATURE", "1.0"))
STG_TAU = float(os.environ.get("STG_TAU", "0.0"))
STG_GAMMA = float(os.environ.get("STG_GAMMA", "1.0"))
STG_FLOOR = float(os.environ.get("STG_FLOOR", "0.05"))
CRSA_ALPHA = float(os.environ.get("CRSA_ALPHA", "1.0"))
CRSA_CENTER = os.environ.get("CRSA_CENTER", "median").strip().lower()
assert CRSA_CENTER in ("median", "mean"), f"CRSA_CENTER must be 'median' or 'mean', got '{CRSA_CENTER}'"
STG_RESIDUAL_LAMBDA = float(os.environ.get("STG_RESIDUAL_LAMBDA", "0.25"))

# ---- ACG-STG: Adaptive Gate hyperparameters ----
STG_ADAPTIVE_GATE = os.environ.get("STG_ADAPTIVE_GATE", "none").strip().lower()
assert STG_ADAPTIVE_GATE in ("none", "stats3", "mlp"), \
    f"STG_ADAPTIVE_GATE must be 'none', 'stats3', or 'mlp', got '{STG_ADAPTIVE_GATE}'"
STG_GATE_DETACH_STATS = os.environ.get("STG_GATE_DETACH_STATS", "1").lower() not in ("0", "false", "no")
STG_GATE_BUDGET_WEIGHT = float(os.environ.get("STG_GATE_BUDGET_WEIGHT", "0.0"))
STG_GATE_BUDGET_RHO = float(os.environ.get("STG_GATE_BUDGET_RHO", "0.7"))
STG_GATE_INIT_BIAS = float(os.environ.get("STG_GATE_INIT_BIAS", "0.0"))
STG_GATE_LOG = os.environ.get("STG_GATE_LOG", "1").lower() not in ("0", "false", "no")

# Derived flag: any STG-CRSA mode
_STG_CRSA_ACTIVE = SELECTION_MODE.startswith("stg_crsa")

# Derived flag: ACG-STG active
_ACG_STG_ACTIVE = (
    SELECTION_MODE == "stg_crsa_residual"
    and STG_ADAPTIVE_GATE != "none"
)

# ---- Ranking Loss hyperparameters ----
RANK_LOSS_WEIGHT = float(os.environ.get("RANK_LOSS_WEIGHT", "0.0"))
RANK_LOSS_TAU = float(os.environ.get("RANK_LOSS_TAU", "0.2"))
RANK_LOSS_MARGIN = float(os.environ.get("RANK_LOSS_MARGIN", "0.0"))
RANK_LOSS_WARMUP_EPOCHS = int(os.environ.get("RANK_LOSS_WARMUP_EPOCHS", "5"))
RANK_LOSS_USE_PROB = os.environ.get("RANK_LOSS_USE_PROB", "0").lower() not in ("0", "false", "no")

# Derived flag: ranking loss active
_RANK_LOSS_ACTIVE = RANK_LOSS_WEIGHT > 0.0


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


# -------------------------------------------------------
#  Git info / param count helpers
# -------------------------------------------------------
def _get_git_info():
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


def _count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


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
_MODEL_META = {}  # populated by create_model() for snapshot/manifest use

def create_model():
    model = VideoClassifier(
        num_classes=2,
        segment_count=SEGMENTS,
        frames_per_segment=FRAMES_PER_SEG,
        backbone_source="timm",
        timm_name="convnext_tiny.fb_in22k",
        pretrained=True,
        backbone_name=BACKBONE_NAME,
        selection_mode=SELECTION_MODE,
        topk=QMIL_TOPK,
        video_dropout=0.5,
        stg_temperature=STG_TEMPERATURE,
        stg_tau=STG_TAU,
        stg_gamma=STG_GAMMA,
        stg_floor=STG_FLOOR,
        crsa_alpha=CRSA_ALPHA,
        crsa_center=CRSA_CENTER,
        stg_residual_lambda=STG_RESIDUAL_LAMBDA,
        stg_adaptive_gate=STG_ADAPTIVE_GATE,
        stg_gate_detach_stats=STG_GATE_DETACH_STATS,
        stg_gate_init_bias=STG_GATE_INIT_BIAS,
        stg_gate_log=STG_GATE_LOG,
    )
    total_params, trainable_params = _count_params(model)
    print(
        f"[Model] backbone={BACKBONE_NAME}  "
        f"feature_dim={model.feature_dim}  "
        f"total_params={total_params:,}  "
        f"trainable_params={trainable_params:,}"
    )

    if ENABLE_FRAME_BRANCH_PRETRAIN:
        if os.path.isfile(FRAME_BRANCH_PRETRAIN_CKPT):
            raw_ckpt = torch.load(FRAME_BRANCH_PRETRAIN_CKPT, map_location="cpu")

            # --- backbone_name guard ---
            ckpt_backbone = None
            ckpt_feature_dim = None
            if isinstance(raw_ckpt, dict):
                ckpt_backbone = raw_ckpt.get("backbone_name", None)
                ckpt_feature_dim = raw_ckpt.get("feature_dim", None)

            if isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt:
                ckpt = raw_ckpt["state_dict"]
            else:
                ckpt = raw_ckpt

            if not isinstance(ckpt, dict):
                print(f"[FrameBranchPretrain] Invalid checkpoint format: {FRAME_BRANCH_PRETRAIN_CKPT}")
            else:
                # backbone_name check
                if ckpt_backbone is not None and ckpt_backbone != BACKBONE_NAME:
                    raise ValueError(
                        f"[FrameBranchPretrain] BACKBONE MISMATCH: "
                        f"checkpoint backbone='{ckpt_backbone}' vs current BACKBONE_NAME='{BACKBONE_NAME}'. "
                        f"Cannot load {ckpt_backbone} pretrain weights into {BACKBONE_NAME} model. "
                        f"Checkpoint: {FRAME_BRANCH_PRETRAIN_CKPT}"
                    )
                elif ckpt_backbone is None:
                    print(
                        "[FrameBranchPretrain] WARNING: checkpoint has no backbone_name metadata. "
                        "Cannot verify backbone match — proceed at your own risk."
                    )
                else:
                    print(f"[FrameBranchPretrain] Backbone match confirmed: {ckpt_backbone}")

                # feature_dim check
                if ckpt_feature_dim is not None and int(ckpt_feature_dim) != model.feature_dim:
                    raise ValueError(
                        f"[FrameBranchPretrain] FEATURE_DIM MISMATCH: "
                        f"checkpoint feature_dim={ckpt_feature_dim} vs model feature_dim={model.feature_dim}. "
                        f"Checkpoint: {FRAME_BRANCH_PRETRAIN_CKPT}"
                    )

                # --- pretrain_type dispatch ---
                pretrain_type = raw_ckpt.get("pretrain_type", None)
                if pretrain_type == "mae":
                    # MAE path: load encoder only, no frame_head
                    mae_meta = load_mae_encoder_from_checkpoint(
                        model, FRAME_BRANCH_PRETRAIN_CKPT
                    )
                    # skip supervised stanza below
                    model = model.to(DEVICE)
                    if USE_MULTI_GPU:
                        model = torch.nn.DataParallel(model)
                        print(f"Model wrapped with DataParallel (using {torch.cuda.device_count()} GPUs)")
                    # attach metadata
                    model._qm_backbone_name = BACKBONE_NAME
                    model._qm_total_params = total_params
                    model._qm_trainable_params = trainable_params
                    model._qm_pretrain_type = "mae"
                    model._qm_mae_meta = mae_meta
                    _MODEL_META.update({
                        "backbone_name": BACKBONE_NAME,
                        "feature_dim": int(model.feature_dim),
                        "total_params": int(total_params),
                        "trainable_params": int(trainable_params),
                        "pretrain_type": "mae",
                        "selection_mode": SELECTION_MODE,
                        "stg_adaptive_gate": STG_ADAPTIVE_GATE if _ACG_STG_ACTIVE else None,
                        "stg_gate_detach_stats": STG_GATE_DETACH_STATS if _ACG_STG_ACTIVE else None,
                        "stg_gate_init_bias": STG_GATE_INIT_BIAS if _ACG_STG_ACTIVE else None,
                        "stg_gate_budget_weight": STG_GATE_BUDGET_WEIGHT if _ACG_STG_ACTIVE else None,
                        "stg_gate_budget_rho": STG_GATE_BUDGET_RHO if _ACG_STG_ACTIVE else None,
                        "rank_loss_weight": RANK_LOSS_WEIGHT if _RANK_LOSS_ACTIVE else None,
                        "rank_loss_tau": RANK_LOSS_TAU if _RANK_LOSS_ACTIVE else None,
                        "rank_loss_margin": RANK_LOSS_MARGIN if _RANK_LOSS_ACTIVE else None,
                        "rank_loss_warmup_epochs": RANK_LOSS_WARMUP_EPOCHS if _RANK_LOSS_ACTIVE else None,
                        "rank_loss_use_prob": RANK_LOSS_USE_PROB if _RANK_LOSS_ACTIVE else None,
                    })
                    return model

                if pretrain_type is not None and pretrain_type != "supervised_frame_branch":
                    raise ValueError(
                        f"[FrameBranchPretrain] Unknown pretrain_type='{pretrain_type}'. "
                        f"Expected 'mae' or 'supervised_frame_branch'. "
                        f"Checkpoint: {FRAME_BRANCH_PRETRAIN_CKPT}"
                    )

                # --- supervised_frame_branch path (existing, unchanged) ---
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
                        if len(load_info.missing_keys) > 20 or len(load_info.unexpected_keys) > 20:
                            print(
                                "[FrameBranchPretrain] WARNING: large mismatch in backbone load — "
                                "possible backbone/structure mismatch. "
                                f"missing_keys sample: {load_info.missing_keys[:5]}  "
                                f"unexpected_keys sample: {load_info.unexpected_keys[:5]}"
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
        print("[INFO] Frame branch pretrain disabled.")

    model = model.to(DEVICE)

    if USE_MULTI_GPU:
        model = torch.nn.DataParallel(model)
        print(f"Model wrapped with DataParallel (using {torch.cuda.device_count()} GPUs)")

    # attach metadata for manifest
    model._qm_backbone_name = BACKBONE_NAME
    model._qm_total_params = total_params
    model._qm_trainable_params = trainable_params
    model._qm_pretrain_type = "supervised_frame_branch" if ENABLE_FRAME_BRANCH_PRETRAIN else "none"
    # store globally for config_snapshot / manifest / result_summary
    _MODEL_META.update({
        "backbone_name": BACKBONE_NAME,
        "feature_dim": int(model.feature_dim),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "pretrain_type": ("supervised_frame_branch" if ENABLE_FRAME_BRANCH_PRETRAIN else "none"),
        "selection_mode": SELECTION_MODE,
        "stg_temperature": STG_TEMPERATURE if _STG_CRSA_ACTIVE else None,
        "stg_tau": STG_TAU if _STG_CRSA_ACTIVE else None,
        "stg_gamma": STG_GAMMA if _STG_CRSA_ACTIVE else None,
        "stg_floor": STG_FLOOR if _STG_CRSA_ACTIVE else None,
        "crsa_alpha": CRSA_ALPHA if _STG_CRSA_ACTIVE else None,
        "crsa_center": CRSA_CENTER if _STG_CRSA_ACTIVE else None,
        "stg_residual_lambda": STG_RESIDUAL_LAMBDA if _STG_CRSA_ACTIVE else None,
        "stg_adaptive_gate": STG_ADAPTIVE_GATE if _ACG_STG_ACTIVE else None,
        "stg_gate_detach_stats": STG_GATE_DETACH_STATS if _ACG_STG_ACTIVE else None,
        "stg_gate_init_bias": STG_GATE_INIT_BIAS if _ACG_STG_ACTIVE else None,
        "stg_gate_budget_weight": STG_GATE_BUDGET_WEIGHT if _ACG_STG_ACTIVE else None,
        "stg_gate_budget_rho": STG_GATE_BUDGET_RHO if _ACG_STG_ACTIVE else None,
        "rank_loss_weight": RANK_LOSS_WEIGHT if _RANK_LOSS_ACTIVE else None,
        "rank_loss_tau": RANK_LOSS_TAU if _RANK_LOSS_ACTIVE else None,
        "rank_loss_margin": RANK_LOSS_MARGIN if _RANK_LOSS_ACTIVE else None,
        "rank_loss_warmup_epochs": RANK_LOSS_WARMUP_EPOCHS if _RANK_LOSS_ACTIVE else None,
        "rank_loss_use_prob": RANK_LOSS_USE_PROB if _RANK_LOSS_ACTIVE else None,
    })
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
    """Legacy wrapper — kept for backward compat."""
    info = _get_git_info()
    return info["git_commit"][:8] if info["git_commit"] != "unknown" else None


def main():
    train_log_path = os.path.join(SAVE_DIR, "train.log")
    train_log_fp = open(train_log_path, "w", encoding="utf-8", buffering=1)  # line-buffered
    def _log(msg):
        print(msg)
        train_log_fp.write(msg + "\n")

    print("=" * 80)
    print("REVERSE ABLATION CONFIG")
    print(f"EXPERIMENT_TAG:               {EXPERIMENT_TAG}")
    print(f"BACKBONE_NAME:                {BACKBONE_NAME}")
    print(f"ENABLE_FRAME_BRANCH_PRETRAIN: {ENABLE_FRAME_BRANCH_PRETRAIN}")
    print(f"FRAME_BRANCH_PRETRAIN_CKPT:   {FRAME_BRANCH_PRETRAIN_CKPT}")
    print(f"PRETRAIN_TYPE:                {_MODEL_META.get('pretrain_type', 'unknown')}")
    print(f"FRAME_LOSS_MODE:              {FRAME_LOSS_MODE}")
    print(f"MIL_SELECT_LEVEL:             {MIL_SELECT_LEVEL}")
    print(f"MIL_POS_TOPK_SEGMENTS:        {MIL_POS_TOPK_SEGMENTS}")
    print(f"MIL_MALIGNANT_CLASS_IDX:      {MIL_MALIGNANT_CLASS_IDX}")
    print(f"MIL_SELECTION_DETACH:         {MIL_SELECTION_DETACH}")
    print(f"FRAME_WEIGHT:                 {FRAME_WEIGHT}")
    print(f"QMIL selection_mode:          {SELECTION_MODE}  topk={QMIL_TOPK}")
    if _STG_CRSA_ACTIVE:
        print(f"STG_TEMPERATURE:              {STG_TEMPERATURE}")
        print(f"STG_TAU:                      {STG_TAU}")
        print(f"STG_GAMMA:                    {STG_GAMMA}")
        print(f"STG_FLOOR:                    {STG_FLOOR}")
        print(f"CRSA_ALPHA:                   {CRSA_ALPHA}")
        print(f"CRSA_CENTER:                  {CRSA_CENTER}")
        if SELECTION_MODE == "stg_crsa_residual":
            print(f"STG_RESIDUAL_LAMBDA:          {STG_RESIDUAL_LAMBDA}")
    if _ACG_STG_ACTIVE:
        print(f"STG_ADAPTIVE_GATE:            {STG_ADAPTIVE_GATE}")
        print(f"STG_GATE_DETACH_STATS:        {STG_GATE_DETACH_STATS}")
        print(f"STG_GATE_INIT_BIAS:           {STG_GATE_INIT_BIAS}")
        print(f"STG_GATE_BUDGET_WEIGHT:       {STG_GATE_BUDGET_WEIGHT}")
        print(f"STG_GATE_BUDGET_RHO:          {STG_GATE_BUDGET_RHO}")
        print(f"STG_GATE_LOG:                 {STG_GATE_LOG}")
    if _RANK_LOSS_ACTIVE:
        print(f"RANK_LOSS_WEIGHT:             {RANK_LOSS_WEIGHT}")
        print(f"RANK_LOSS_TAU:                {RANK_LOSS_TAU}")
        print(f"RANK_LOSS_MARGIN:             {RANK_LOSS_MARGIN}")
        print(f"RANK_LOSS_WARMUP_EPOCHS:      {RANK_LOSS_WARMUP_EPOCHS}")
        print(f"RANK_LOSS_USE_PROB:           {RANK_LOSS_USE_PROB}")
    print(f"DATASET_NAME:                 {DATASET_NAME}")
    print(f"VIDEO_ROOT:                   {DATA_ROOT}")
    print(f"OFFICIAL_TRAIN_JSON:          {OFFICIAL_TRAIN_JSON}")
    print(f"FIXED_TEST_JSON:              {FIXED_TEST_JSON}")
    print(f"SEGMENTS:                     {SEGMENTS}")
    print(f"FRAMES_PER_SEG:               {FRAMES_PER_SEG}")
    print(f"EPOCHS:                       {EPOCHS}")
    print(f"BATCH_SIZE:                   {BATCH_SIZE}")
    print(f"N_REPEATS:                    {N_REPEATS}")
    print(f"SEED:                         {SEED}")
    print(f"SAVE_DIR:                     {SAVE_DIR}")
    print("=" * 80)

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
            epoch_gate_vals = []      # accumulated gate values for logging
            epoch_lambda_vals = []    # accumulated lambda_eff values for logging
            epoch_budget_vals = []    # accumulated budget loss values
            epoch_rank_vals = []      # accumulated ranking loss values

            # ---- Ranking loss warmup ----
            if _RANK_LOSS_ACTIVE:
                rank_weight_eff = RANK_LOSS_WEIGHT
                if epoch <= RANK_LOSS_WARMUP_EPOCHS:
                    rank_weight_eff = RANK_LOSS_WEIGHT * float(epoch) / float(max(1, RANK_LOSS_WARMUP_EPOCHS))
            else:
                rank_weight_eff = 0.0

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
                        frame_weight=FRAME_WEIGHT,
                        frame_loss_mode=FRAME_LOSS_MODE,
                        mil_select_level=MIL_SELECT_LEVEL,
                        mil_pos_topk_segments=MIL_POS_TOPK_SEGMENTS,
                        mil_malignant_class_idx=MIL_MALIGNANT_CLASS_IDX,
                        mil_selection_detach=MIL_SELECTION_DETACH,
                    )

                    # ---- Ranking loss ----
                    rank_loss_val = video_logits.new_tensor(0.0)
                    if _RANK_LOSS_ACTIVE:
                        rank_loss_val = pairwise_ranking_loss(
                            video_logits,
                            labels,
                            tau=RANK_LOSS_TAU,
                            margin=RANK_LOSS_MARGIN,
                            use_prob=RANK_LOSS_USE_PROB,
                        )
                        if rank_weight_eff > 0.0 and rank_loss_val.item() > 0.0:
                            loss = loss + rank_weight_eff * rank_loss_val
                        epoch_rank_vals.append(float(rank_loss_val.detach().item()))

                    # ---- ACG-STG: gate stats accumulation & budget loss ----
                    if _ACG_STG_ACTIVE:
                        base_model = _get_base_model(model)
                        gate_raw = getattr(base_model, "last_gate_raw", None)
                        if gate_raw is not None:
                            # Budget loss (only when beta > 0)
                            if STG_GATE_BUDGET_WEIGHT > 0:
                                g_mean = gate_raw.mean()
                                budget_loss = (
                                    STG_GATE_BUDGET_WEIGHT
                                    * F.relu(g_mean - STG_GATE_BUDGET_RHO) ** 2
                                )
                                loss = loss + budget_loss
                                epoch_budget_vals.append(float(budget_loss.detach().item()))

                            # Accumulate for logging
                            gate_stats = getattr(base_model, "last_gate_stats", None)
                            if gate_stats is not None:
                                epoch_gate_vals.append(gate_stats["gate"].detach().cpu())
                                epoch_lambda_vals.append(gate_stats["lambda_eff"].detach().cpu())

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
                    rank=float(rank_loss_val.detach().item()) if _RANK_LOSS_ACTIVE else None,
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

            # ---- ACG-STG: gate stats logging ----
            gate_log_entry = None
            if _ACG_STG_ACTIVE and len(epoch_gate_vals) > 0:
                all_gates = torch.cat(epoch_gate_vals)       # [total_B]
                all_lambdas = torch.cat(epoch_lambda_vals)   # [total_B]
                gate_log_entry = {
                    "gate_mean": float(all_gates.mean().item()),
                    "gate_std": float(all_gates.std().item()),
                    "gate_min": float(all_gates.min().item()),
                    "gate_max": float(all_gates.max().item()),
                    "lambda_eff_mean": float(all_lambdas.mean().item()),
                    "lambda_eff_std": float(all_lambdas.std().item()),
                }
                if STG_GATE_BUDGET_WEIGHT > 0 and len(epoch_budget_vals) > 0:
                    gate_log_entry["budget_loss_mean"] = float(
                        sum(epoch_budget_vals) / max(1, len(epoch_budget_vals))
                    )

            # ---- Ranking loss: epoch summary ----
            rank_log_entry = None
            if _RANK_LOSS_ACTIVE and len(epoch_rank_vals) > 0:
                rank_loss_mean = float(sum(epoch_rank_vals) / max(1, len(epoch_rank_vals)))
                rank_log_entry = {
                    "rank_loss": rank_loss_mean,
                    "rank_weight_eff": float(rank_weight_eff),
                }

            _hist_entry = {
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
            if gate_log_entry is not None:
                _hist_entry["acg_stg_gate"] = gate_log_entry
            if rank_log_entry is not None:
                _hist_entry["rank_loss"] = rank_log_entry
            history.append(_hist_entry)

            print("\n" + "-" * 100)
            print(
                f"Repeat {repeat_idx} | Epoch {epoch}/{EPOCHS} | "
                f"Train loss={train_loss:.4f}, acc={train_acc:.3f}"
            )
            if gate_log_entry is not None:
                _log(
                    f"[ACG-STG] gate_mean={gate_log_entry['gate_mean']:.4f} "
                    f"gate_std={gate_log_entry['gate_std']:.4f} "
                    f"gate_min={gate_log_entry['gate_min']:.4f} "
                    f"gate_max={gate_log_entry['gate_max']:.4f} "
                    f"lambda_eff_mean={gate_log_entry['lambda_eff_mean']:.4f} "
                    f"lambda_eff_std={gate_log_entry['lambda_eff_std']:.4f}"
                    + (
                        f" budget_loss={gate_log_entry.get('budget_loss_mean', 0.0):.6f}"
                        if STG_GATE_BUDGET_WEIGHT > 0 else ""
                    )
                )
            if rank_log_entry is not None:
                _log(
                    f"[RankLoss] weight_eff={rank_log_entry['rank_weight_eff']:.4f} "
                    f"rank_loss={rank_log_entry['rank_loss']:.6f}"
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

    git_info = _get_git_info()

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
            "experiment_tag": EXPERIMENT_TAG,
            "backbone_name": BACKBONE_NAME,
            "feature_dim": int(_MODEL_META.get("feature_dim", -1)),
            "total_params": int(_MODEL_META.get("total_params", -1)),
            "trainable_params": int(_MODEL_META.get("trainable_params", -1)),
            "dataset_name": DATASET_NAME,
            "segments": int(SEGMENTS),
            "frames_per_seg": int(FRAMES_PER_SEG),
            "batch_size": int(BATCH_SIZE),
            "epochs": int(EPOCHS),
            "n_repeats": int(N_REPEATS),
            "enable_frame_branch_pretrain": bool(ENABLE_FRAME_BRANCH_PRETRAIN),
            "pretrain_type": _MODEL_META.get("pretrain_type", "unknown"),
            "frame_branch_pretrain_ckpt": FRAME_BRANCH_PRETRAIN_CKPT,
            "frame_loss_mode": FRAME_LOSS_MODE,
            "mil_select_level": MIL_SELECT_LEVEL,
            "mil_pos_topk_segments": int(MIL_POS_TOPK_SEGMENTS),
            "mil_malignant_class_idx": int(MIL_MALIGNANT_CLASS_IDX),
            "mil_selection_detach": bool(MIL_SELECTION_DETACH),
            "selection_mode": SELECTION_MODE,
            "qmil_topk": int(QMIL_TOPK),
            "frame_weight": FRAME_WEIGHT,
            "stg_temperature": STG_TEMPERATURE if _STG_CRSA_ACTIVE else None,
            "stg_tau": STG_TAU if _STG_CRSA_ACTIVE else None,
            "stg_gamma": STG_GAMMA if _STG_CRSA_ACTIVE else None,
            "stg_floor": STG_FLOOR if _STG_CRSA_ACTIVE else None,
            "crsa_alpha": CRSA_ALPHA if _STG_CRSA_ACTIVE else None,
            "crsa_center": CRSA_CENTER if _STG_CRSA_ACTIVE else None,
            "stg_residual_lambda": STG_RESIDUAL_LAMBDA if _STG_CRSA_ACTIVE else None,
            "stg_adaptive_gate": STG_ADAPTIVE_GATE if _ACG_STG_ACTIVE else None,
            "stg_gate_detach_stats": STG_GATE_DETACH_STATS if _ACG_STG_ACTIVE else None,
            "stg_gate_init_bias": STG_GATE_INIT_BIAS if _ACG_STG_ACTIVE else None,
            "stg_gate_budget_weight": STG_GATE_BUDGET_WEIGHT if _ACG_STG_ACTIVE else None,
            "stg_gate_budget_rho": STG_GATE_BUDGET_RHO if _ACG_STG_ACTIVE else None,
            "rank_loss_weight": RANK_LOSS_WEIGHT if _RANK_LOSS_ACTIVE else None,
            "rank_loss_tau": RANK_LOSS_TAU if _RANK_LOSS_ACTIVE else None,
            "rank_loss_margin": RANK_LOSS_MARGIN if _RANK_LOSS_ACTIVE else None,
            "rank_loss_warmup_epochs": RANK_LOSS_WARMUP_EPOCHS if _RANK_LOSS_ACTIVE else None,
            "rank_loss_use_prob": RANK_LOSS_USE_PROB if _RANK_LOSS_ACTIVE else None,
            "official_train_json_path": OFFICIAL_TRAIN_JSON,
            "fixed_test_json_path": FIXED_TEST_JSON,
            "video_root": DATA_ROOT,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "device": str(DEVICE),
            "repeat_count": int(N_REPEATS),
            "train_internal_val_ratio": {"train": float(1.0 - INTERNAL_VAL_RATIO), "internal_val": float(INTERNAL_VAL_RATIO)},
            "seed": int(SEED),
            "commit_hash": git_info["git_commit"],
            "git_branch": git_info["git_branch"],
            "git_dirty": git_info["git_dirty"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    summary_path = os.path.join(SAVE_DIR, "official_split_summary_best_test.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    # --- experiment_manifest.json ---
    manifest = {
        "experiment_tag": EXPERIMENT_TAG,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_branch": git_info["git_branch"],
        "git_commit": git_info["git_commit"],
        "git_dirty": git_info["git_dirty"],
        "seed": int(SEED),
        "dataset_name": DATASET_NAME,
        "backbone_name": BACKBONE_NAME,
        "feature_dim": int(_MODEL_META.get("feature_dim", -1)),
        "total_params": int(_MODEL_META.get("total_params", -1)),
        "trainable_params": int(_MODEL_META.get("trainable_params", -1)),
        "batch_size": BATCH_SIZE,
        "segments": SEGMENTS,
        "frames_per_seg": FRAMES_PER_SEG,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "selection_mode": SELECTION_MODE,
        "qmil_topk": int(QMIL_TOPK),
        "frame_weight": FRAME_WEIGHT,
        "stg_temperature": STG_TEMPERATURE if _STG_CRSA_ACTIVE else None,
        "stg_tau": STG_TAU if _STG_CRSA_ACTIVE else None,
        "stg_gamma": STG_GAMMA if _STG_CRSA_ACTIVE else None,
        "stg_floor": STG_FLOOR if _STG_CRSA_ACTIVE else None,
        "crsa_alpha": CRSA_ALPHA if _STG_CRSA_ACTIVE else None,
        "crsa_center": CRSA_CENTER if _STG_CRSA_ACTIVE else None,
        "stg_residual_lambda": STG_RESIDUAL_LAMBDA if _STG_CRSA_ACTIVE else None,
        "stg_adaptive_gate": STG_ADAPTIVE_GATE if _ACG_STG_ACTIVE else None,
        "stg_gate_detach_stats": STG_GATE_DETACH_STATS if _ACG_STG_ACTIVE else None,
        "stg_gate_init_bias": STG_GATE_INIT_BIAS if _ACG_STG_ACTIVE else None,
        "stg_gate_budget_weight": STG_GATE_BUDGET_WEIGHT if _ACG_STG_ACTIVE else None,
        "stg_gate_budget_rho": STG_GATE_BUDGET_RHO if _ACG_STG_ACTIVE else None,
        "rank_loss_weight": RANK_LOSS_WEIGHT if _RANK_LOSS_ACTIVE else None,
        "rank_loss_tau": RANK_LOSS_TAU if _RANK_LOSS_ACTIVE else None,
        "rank_loss_margin": RANK_LOSS_MARGIN if _RANK_LOSS_ACTIVE else None,
        "rank_loss_warmup_epochs": RANK_LOSS_WARMUP_EPOCHS if _RANK_LOSS_ACTIVE else None,
        "rank_loss_use_prob": RANK_LOSS_USE_PROB if _RANK_LOSS_ACTIVE else None,
        "mil_pos_topk_segments": int(MIL_POS_TOPK_SEGMENTS),
        "mil_select_level": MIL_SELECT_LEVEL,
        "mil_selection_detach": bool(MIL_SELECTION_DETACH),
        "frame_loss_mode": FRAME_LOSS_MODE,
        "enable_frame_branch_pretrain": bool(ENABLE_FRAME_BRANCH_PRETRAIN),
        "pretrain_type": _MODEL_META.get("pretrain_type", "unknown"),
        "frame_branch_pretrain_ckpt": FRAME_BRANCH_PRETRAIN_CKPT,
        "official_train_json_path": OFFICIAL_TRAIN_JSON,
        "fixed_test_json_path": FIXED_TEST_JSON,
        "video_root": DATA_ROOT,
        "output_dir": SAVE_DIR,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device": str(DEVICE),
        "repeat_count": int(N_REPEATS),
        "internal_val_ratio": float(INTERNAL_VAL_RATIO),
        "n_repeats": int(N_REPEATS),
    }
    manifest_path = os.path.join(SAVE_DIR, "experiment_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # --- result_summary.json ---
    result_summary = {
        "experiment_tag": EXPERIMENT_TAG,
        "backbone_name": BACKBONE_NAME,
        "pretrain_type": _MODEL_META.get("pretrain_type", "unknown"),
        "feature_dim": int(_MODEL_META.get("feature_dim", -1)),
        "total_params": int(_MODEL_META.get("total_params", -1)),
        "trainable_params": int(_MODEL_META.get("trainable_params", -1)),
        "selection_mode": SELECTION_MODE,
        "qmil_topk": int(QMIL_TOPK),
        "frame_weight": FRAME_WEIGHT,
        "frame_loss_mode": FRAME_LOSS_MODE,
        "enable_frame_branch_pretrain": bool(ENABLE_FRAME_BRANCH_PRETRAIN),
        "pretrain_ckpt": FRAME_BRANCH_PRETRAIN_CKPT,
        "n_repeats": int(N_REPEATS),
        "stg_adaptive_gate": STG_ADAPTIVE_GATE if _ACG_STG_ACTIVE else None,
        "stg_gate_budget_weight": STG_GATE_BUDGET_WEIGHT if _ACG_STG_ACTIVE else None,
        "stg_gate_budget_rho": STG_GATE_BUDGET_RHO if _ACG_STG_ACTIVE else None,
        "stg_gate_init_bias": STG_GATE_INIT_BIAS if _ACG_STG_ACTIVE else None,
        "rank_loss_weight": RANK_LOSS_WEIGHT if _RANK_LOSS_ACTIVE else None,
        "rank_loss_tau": RANK_LOSS_TAU if _RANK_LOSS_ACTIVE else None,
        "rank_loss_margin": RANK_LOSS_MARGIN if _RANK_LOSS_ACTIVE else None,
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
        "per_repeat": [
            {
                "repeat": r["repeat"],
                "split_seed": r["split_seed"],
                "best_epoch": r["best_val_result"]["best_epoch"],
                "best_val_auc": r["best_val_result"]["best_val_auc"],
                "best_val_auprc": r["best_val_result"]["best_val_auprc"],
                "youden_threshold": r["best_val_result"]["youden_threshold_on_internal_val"],
                "fixed_test_auc": r["fixed_test_result_best"]["fixed_test_auc"],
                "fixed_test_auprc": r["fixed_test_result_best"]["fixed_test_auprc"],
            }
            for r in repeat_results
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["git_commit"],
    }
    result_summary_path = os.path.join(SAVE_DIR, "result_summary.json")
    with open(result_summary_path, "w", encoding="utf-8") as f:
        json.dump(result_summary, f, indent=2)

    print("\n" + "=" * 100)
    print("OFFICIAL SPLIT SUMMARY (FIXED TEST, BEST INTERNAL-VAL AUC CHECKPOINT)")
    print(f"Backbone: {BACKBONE_NAME}")
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
    print(f"Summary JSON:      {summary_path}")
    print(f"Manifest JSON:     {manifest_path}")
    print(f"Result Summary:    {result_summary_path}")
    print(f"Train Log:         {train_log_path}")

    # --- env_snapshot.txt ---
    env_snapshot_path = os.path.join(SAVE_DIR, "env_snapshot.txt")
    with open(env_snapshot_path, "w", encoding="utf-8") as f:
        _env_keys = [
            "DATASET_NAME", "EXPERIMENT_TAG", "BACKBONE_NAME", "SELECTION_MODE",
            "QMIL_TOPK", "EPOCHS", "BATCH_SIZE", "N_REPEATS", "SEED",
            "SEGMENTS", "FRAMES_PER_SEG", "FRAME_LOSS_MODE", "FRAME_WEIGHT",
            "MIL_SELECT_LEVEL", "MIL_POS_TOPK_SEGMENTS", "MIL_MALIGNANT_CLASS_IDX",
            "MIL_SELECTION_DETACH", "ENABLE_FRAME_BRANCH_PRETRAIN",
            "STG_TEMPERATURE", "STG_TAU", "STG_GAMMA", "STG_FLOOR",
            "CRSA_ALPHA", "CRSA_CENTER", "STG_RESIDUAL_LAMBDA",
            "STG_ADAPTIVE_GATE", "STG_GATE_DETACH_STATS", "STG_GATE_BUDGET_WEIGHT",
            "STG_GATE_BUDGET_RHO", "STG_GATE_INIT_BIAS", "STG_GATE_LOG",
            "RANK_LOSS_WEIGHT", "RANK_LOSS_TAU", "RANK_LOSS_MARGIN",
            "RANK_LOSS_WARMUP_EPOCHS", "RANK_LOSS_USE_PROB",
        ]
        for k in _env_keys:
            v = os.environ.get(k, "")
            f.write(f"{k}={v}\n")
    print(f"Env Snapshot:      {env_snapshot_path}")

    train_log_fp.close()


if __name__ == "__main__":
    main()
