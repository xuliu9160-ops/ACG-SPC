# STG-CRSA v1 Implementation Report

> 日期：2026-06-16
> 分支：`exp/stg-crsa-v1`
> Base：`de753d1e76e570dc1d80697431d9914f3c3a01c6` (clean baseline)
> 状态：**实现完成，已验证，未 commit**

---

## 1. 修改文件

| 文件 | 改动 | 行数 |
|---|---|---|
| `model.py` | +76 行 | `import torch.nn.functional as F`; `__init__` 新增 STG-CRSA 超参数 + debug tensor 初始化; `forward` 新增 `stg_crsa` elif 分支; 非 STG 模式清零 debug tensors |
| `train_official_split.py` | +41 行 | 新增 6 个环境变量; `create_model` 传入 STG-CRSA 参数; `main()` 打印 STG-CRSA config; `_MODEL_META`/`config_snapshot`/`manifest`/`result_summary` 记录 STG-CRSA 字段 |

**未修改文件**：`loss.py`, `dataset.py`

---

## 2. 新增/修改代码位置

### 2.1 model.py

| 行号 | 内容 |
|---|---|
| L9 | `import torch.nn.functional as F` |
| L435-440 | `__init__` 签名新增 6 个 STG-CRSA 参数 |
| L450-457 | 超参数存储 + `crsa_center` 校验 |
| L461-465 | 5 个 debug tensor 初始化为 `None` |
| L519-524 | `topk_softmax` 分支清零 stg tensors |
| L527-532 | `mean` 分支清零 stg tensors |
| **L533-569** | **`stg_crsa` 核心实现** |
| L573 | `self.last_segment_weights = seg_w.detach()` (统一外提) |

### 2.2 train_official_split.py

| 行号 | 内容 |
|---|---|
| L136-143 | 6 个 STG-CRSA 环境变量定义 |
| L301-312 | `create_model()` 传入 STG-CRSA 参数 |
| L475-482 | `_MODEL_META` 记录 STG-CRSA 字段 |
| L525-531 | `main()` 打印 STG-CRSA config |
| L906-911 | `config_snapshot` 记录 STG-CRSA 字段 |
| L953-958 | `experiment_manifest` 记录 STG-CRSA 字段 |
| L988-993 | `result_summary` 记录 STG-CRSA 字段 |

---

## 3. 新 selection_mode 公式

### `selection_mode = "stg_crsa"`

```
输入: seg_tokens = mean(feats, dim=frames)     # [B,S,D]

Step 1 — Feature Support:
  tokens_norm = L2_normalize(seg_tokens, dim=-1)
  cos_mat = tokens_norm @ tokens_norm.T          # [B,S,S]
  support_i = mean_{j≠i}(cos_ij)                # [B,S]

Step 2 — Gate:
  support_z = (support - μ) / (σ + 1e-6)         # Z-score normalize
  gate = sigmoid((support_z - τ) / T)             # [B,S], τ=stg_tau, T=stg_temperature

Step 3 — Trust:
  trust = floor + gate^γ                           # [B,S], floor=stg_floor, γ=stg_gamma

Step 4 — CRSA Reliability:
  seg_logits = video_head(seg_tokens)              # [B,S,2]
  seg_probs = softmax(seg_logits)[:,:,1]           # [B,S]
  center = median(seg_probs) or mean(seg_probs)
  reliability = exp(-α * |seg_probs - center|)     # [B,S], α=crsa_alpha

Step 5 — Gated Fusion:
  raw_w = trust × reliability                      # [B,S]
  seg_w = raw_w / (sum(raw_w) + 1e-8)             # L1 normalize

输出:
  video_feat = sum(seg_tokens × seg_w)            # [B,D]
  video_logits = video_head(video_feat)            # [B,2]
```

---

## 4. 默认超参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `STG_TEMPERATURE` | 1.0 | sigmoid 温度 |
| `STG_TAU` | 0.0 | sigmoid 偏移 |
| `STG_GAMMA` | 1.0 | trust 中的 gate 指数 |
| `STG_FLOOR` | 0.05 | trust 最低值 |
| `CRSA_ALPHA` | 1.0 | CRSA 一致性强度 |
| `CRSA_CENTER` | `median` | robustness center (`median`/`mean`) |

---

## 5. 与旧机制的对比

| 维度 | 旧 CS-FSS | 旧 CRSA | **STG-CRSA (新)** |
|---|---|---|---|
| 信号来源 | cosine similarity + video_head probs | video_head probs only | cosine similarity (gate) + video_head probs (reliability) |
| 聚合方式 | penalty × uniform_w | exp(-α·|p-center|) | trust × reliability, L1 normalize |
| frame_head 使用 | 否 | 否 | **否** |
| quality_head 使用 | 否 | 否 | **否** |
| gate 来源 | 无 | 无 | feature support (Z-score → sigmoid) |
| 与 evidence 的关系 | 无关 | 无关 | 无关（不使用 frame_head malignant prob) |
| 核心直觉 | "feature-inconsistent segments are unreliable" | "segments disagreeing with consensus are unreliable" | "gated fusion: trust decisions in feature space, aggregate in probability space" |

**与旧 CRSA + CS-FSS 直接叠加的关键区别**：
- 旧叠加：`seg_w = gate * w_crsa + (1-gate) * w_uniform` → gate 是 soft mix，低 gate 的 segment 仍有 uniform weight 贡献
- STG-CRSA：`seg_w ∝ trust × reliability` → trust 是**乘法**因子，低 gate→低 trust→segment 在乘积中被大幅度抑制（接近 floor × reliability，而非 uniform）

**不使用 frame_head / quality_head**：
- Gate 信号来自 `seg_tokens` 的 cosine similarity（特征空间自支持度）
- Reliability 信号来自 `video_head(seg_tokens)` 的 per-segment prob（概率空间一致性）
- `frame_head` 仍计算 `frame_logits`（兼容现有训练代码），但**不参与聚合**
- `quality_head` 在 STG-CRSA 模式下完全未使用

---

## 6. py_compile 结果

```
python3 -m py_compile model.py train_official_split.py
PASS
```

---

## 7. diff --check 结果

```
git diff --check
(无输出 = 无 whitespace 错误)
```

---

## 8. forward sanity 结果

```
===== mode = mean =====
  weights sum=1.000000  min=0.166667  all non-neg=True
  PASSED

===== mode = topk_softmax =====
  weights sum=1.000000  min=0.000000  all non-neg=True
  PASSED

===== mode = stg_crsa =====
  weights sum=1.000000  min=0.067599  all non-neg=True
  gate: [0.1784, 0.7831]  all in [0,1]: True
  trust: min=0.228389  all>0: True
  seg_probs: mean=0.5016  std=0.0133
  support: mean=0.0197
  reliability: mean=0.9902
  no NaN: True  no Inf: True
  PASSED

===== stg tensors None for non-STG modes PASSED =====
===== ALL TESTS PASSED =====
```

验证要点：
- ✅ `mean` 和 `topk_softmax` 不被破坏
- ✅ `stg_crsa` 权重和 ≈ 1，全非负
- ✅ `gate ∈ [0,1]`
- ✅ `trust > 0` (受 stg_floor=0.05 保证)
- ✅ 无 NaN / Inf
- ✅ 非 STG 模式 stg tensors 为 None

---

## 9. git status

```
branch: exp/stg-crsa-v1
HEAD:   de753d1e76e570dc1d80697431d9914f3c3a01c6
Modified (unstaged):
  M model.py                (+76 lines)
  M train_official_split.py (+41 lines)
Not committed, not pushed
```

---

## 10. 下一步建议的 smoke 命令

```bash
# BUV smoke test (1 repeat, 10 epochs, quick check gate doesn't collapse)
SELECTION_MODE=stg_crsa \
BACKBONE_NAME=efficientnet_b0 \
FRAME_WEIGHT=0.0 \
ENABLE_FRAME_BRANCH_PRETRAIN=0 \
EPOCHS=10 \
N_REPEATS=1 \
BATCH_SIZE=8 \
STG_TEMPERATURE=1.0 \
STG_TAU=0.0 \
STG_GAMMA=1.0 \
STG_FLOOR=0.05 \
CRSA_ALPHA=1.0 \
CRSA_CENTER=median \
DATASET_NAME=BUV \
VIDEO_ROOT=/path/to/rawframes \
OFFICIAL_TRAIN_JSON=/path/to/train.json \
FIXED_TEST_JSON=/path/to/test.json \
python3 train_official_split.py

# WHBUS smoke test (same params)
DATASET_NAME=WHBUS \
SEGMENTS=8 \
FRAMES_PER_SEG=7 \
BATCH_SIZE=4 \
VIDEO_ROOT=/path/to/whbus/unpacked \
OFFICIAL_TRAIN_JSON=/path/to/whbus_train.json \
FIXED_TEST_JSON=/path/to/whbus_val.json \
python3 train_official_split.py
```

Smoke test 应该关注：
1. `stg_gate_mean` 不塌缩到 0 或 1
2. `stg_trust_mean` 在训练过程中稳定
3. `segment_weight_entropy` 不退化到接近 0
4. loss 正常下降
5. internal_val AUC > 0.5

如果 gate 塌缩到全 0.5（无信息），可以调大 `STG_TAU` 或调小 `STG_TEMPERATURE`。
如果 trust 全部接近 floor=0.05（gate 太小），可以减小 `STG_GAMMA`。
