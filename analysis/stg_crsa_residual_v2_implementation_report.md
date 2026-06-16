# STG-CRSA Residual v2 Implementation Report

> 日期：2026-06-16
> 分支：`exp/stg-crsa-residual-v2`
> Base：`exp/stg-crsa-v1` (commit `88855ca`)
> 状态：**实现完成，已验证，未 commit**

---

## 1. 修改文件

| 文件 | 改动 | 行数 |
|---|---|---|
| `model.py` | +24 / -2 | 新增 `stg_residual_lambda` 参数 + `last_stg_raw_weights` debug 字段; `stg_crsa`/`stg_crsa_residual` 共享 Steps 1-5; v2 新增 Step 6 residual mix |
| `train_official_split.py` | +48 / -21 | 新增 `STG_RESIDUAL_LAMBDA` 环境变量; `_STG_CRSA_ACTIVE` 派生标志; 所有 STG 条件扩展为 `startswith("stg_crsa")`; config_snapshot/manifest/result_summary 记录 `stg_residual_lambda` |

---

## 2. 新公式

### selection_mode = "stg_crsa_residual"

```
Step 1-5: 与 v1 stg_crsa 完全相同
  support = mean_{j≠i}(cos(seg_i, seg_j))
  support_z = (support - μ) / (σ + 1e-6)
  gate = sigmoid((support_z - τ) / T)
  trust = floor + gate^γ
  seg_probs = softmax(video_head(seg_tokens))[:,:,1]
  center = median(seg_probs) or mean(seg_probs)
  reliability = exp(-α * |seg_probs - center|)
  stg_raw_w = L1_normalize(trust × reliability)

Step 6 (NEW — Residual Mix):
  uniform_w = 1/S
  final_w = (1 - λ) × uniform_w + λ × stg_raw_w
  final_w = L1_normalize(final_w)

  where λ = stg_residual_lambda (default 0.25)

Aggregation (unchanged):
  video_feat = sum(seg_tokens × final_w)
  video_logits = video_head(video_feat)
```

### 默认超参数

| 参数 | v1 默认 | v2 默认 | 含义 |
|---|---|---|---|
| `STG_TEMPERATURE` | 1.0 | 1.0 | sigmoid 温度 |
| `STG_TAU` | 0.0 | 0.0 | sigmoid 偏移 |
| `STG_GAMMA` | 1.0 | 1.0 | trust 中 gate 指数 |
| `STG_FLOOR` | 0.05 | 0.05 | trust 最低值 |
| `CRSA_ALPHA` | 1.0 | 1.0 | CRSA 一致性强度 |
| `CRSA_CENTER` | `median` | `median` | robustness center |
| `STG_RESIDUAL_LAMBDA` | — | **0.25** | uniform↔STG 混合比例 |

---

## 3. 与 STG-CRSA v1 的区别

| 维度 | v1 (`stg_crsa`) | v2 (`stg_crsa_residual`) |
|---|---|---|
| 聚合权重 | `seg_w = stg_raw_w` | `seg_w = (1-λ)×uniform + λ×stg_raw_w` |
| λ=0 | — | `seg_w = uniform` (纯 mean pooling) |
| λ=1 | — | `seg_w = stg_raw_w` (等价于 v1) |
| λ=0.25 (default) | — | 75% uniform + 25% STG tuning |
| 权重范围 (sanity) | [0.075, 0.267] | [0.152, 0.200] |
| v1 模式 | ✅ 保留 | ✅ 保留（不变） |
| 选择 | `selection_mode="stg_crsa"` | `selection_mode="stg_crsa_residual"` |

**Sanity 证明**：
- `stg_crsa` v1: `seg_w == stg_raw_w` ✓
- `stg_crsa_residual` λ=0: `final_w == uniform` ✓
- `stg_crsa_residual` λ=1: `final_w == stg_raw_w` ✓
- `stg_crsa_residual` λ=0.25: `final_w ∈ [min(uniform, stg_raw), max(uniform, stg_raw)]` ✓

---

## 4. 为什么这是对 v1 失败的针对性修正

### v1 问题诊断

v1 formal 结果：BUV 下降，WHBUS AUC/Acc 小升但 AUPRC 下降。

最可能的失败模式：`trust × reliability` 乘积在少数但关键的恶性 segment 上产生了过低的权重。当恶性病灶只在一个 segment 中明显时，该 segment 可能在 feature space 中与其他（正常组织）segment 不相似 → 低 support → 低 gate → 低 trust → 该 segment 被严重压制 → video 级别恶性信号被削弱。

### v2 修正策略

**Residual connection 到 uniform mean pooling**：

- 即使 gate=0（极低 trust），segment 仍保留 `(1-λ)/S` 的基础权重
- λ=0.25 意味着 STG-CRSA 的校准最多贡献 25% 的权重调制
- 其余 75% 来自 mean pooling 的稳定性

这类似于 ResNet 的 skip connection 思想：STG-CRSA 学习的是**偏离 uniform 的残差**，而非从头决定所有权重。如果 STG 的可靠性信号是噪声（在 v1 中可能就是），残差连接削弱其影响。

### 消融优势

- λ=0 → clean mean baseline（可精确复现）
- λ=1 → v1 STG-CRSA（可精确复现）
- λ=0.25 → v2（保守校准）
- λ sweep → 连续调节 STG 贡献强度

---

## 5. py_compile 结果

```
python3 -m py_compile model.py train_official_split.py
PASS
```

---

## 6. diff check 结果

```
git diff --check
(无输出)
```

---

## 7. sanity 结果

```
===== mode = mean =====
  final_w sum=1.000000  min=0.166667  max=0.166667  all non-neg=True
  PASSED

===== mode = stg_crsa =====
  final_w sum=1.000000  min=0.075176  max=0.266602  all non-neg=True
  stg_raw_w sum=1.000000
  gate mean=0.4994 in [0,1]: True
  trust min=0.246502 all>0: True
  no NaN/Inf: True / True
  PASSED

===== mode = stg_crsa_residual =====
  final_w sum=1.000000  min=0.152236  max=0.200061  all non-neg=True
  stg_raw_w sum=1.000000
  gate mean=0.4793 in [0,1]: True
  trust min=0.345675 all>0: True
  no NaN/Inf: True / True
  residual mix check: max|final - expected|=0.00e+00 (should be ~0)
  lambda=0: final_w all=0.166667 == uniform  PASSED
  lambda=1: final_w == stg_raw_w  PASSED
  lambda=0.25: final_w between uniform and stg_w: True
  PASSED

v1 stg_crsa: seg_w == stg_raw_w: True

===== ALL TESTS PASSED =====
```

关键数值：
- v1 权重范围：[0.075, 0.267]（spread = 0.192）
- v2 权重范围：[0.152, 0.200]（spread = 0.048）
- v2 的 spread 仅为 v1 的 **25%** — 这正是 λ=0.25 的预期效果

---

## 8. git status

```
branch: exp/stg-crsa-residual-v2
HEAD:   88855ca (exp/stg-crsa-v1)
Modified (unstaged):
  M model.py                (+24 / -2 lines)
  M train_official_split.py (+48 / -21 lines)
Not committed, not pushed
```

---

## 9. 建议 smoke 命令

```bash
# v2 BUV smoke (1 repeat, 10 epochs)
SELECTION_MODE=stg_crsa_residual \
BACKBONE_NAME=efficientnet_b0 \
FRAME_WEIGHT=0.0 \
ENABLE_FRAME_BRANCH_PRETRAIN=0 \
EPOCHS=10 N_REPEATS=1 BATCH_SIZE=8 \
STG_TEMPERATURE=1.0 STG_TAU=0.0 STG_GAMMA=1.0 \
STG_FLOOR=0.05 CRSA_ALPHA=1.0 CRSA_CENTER=median \
STG_RESIDUAL_LAMBDA=0.25 \
DATASET_NAME=BUV \
VIDEO_ROOT=/path/to/rawframes \
OFFICIAL_TRAIN_JSON=/path/to/train.json \
FIXED_TEST_JSON=/path/to/test.json \
python3 train_official_split.py

# Parameter sweep (BUV 1-repeat each):
#   STG_RESIDUAL_LAMBDA ∈ {0.0, 0.1, 0.25, 0.5, 0.75, 1.0}
#   0.0 = pure mean baseline (reproducible)
#   1.0 = pure v1 (reproducible)

# Also test v1 vs v2 on WHBUS
SELECTION_MODE=stg_crsa_residual \
DATASET_NAME=WHBUS SEGMENTS=8 FRAMES_PER_SEG=7 BATCH_SIZE=4 \
... same STG params ...
```

Smoke test 关注：
1. v2 AUC/AUPRC 是否比 v1 更稳定（尤其 WHBUS AUPRC）
2. λ=0.25 的内部验证 AUC 是否至少不差于 clean mean
3. λ sweep 曲线形状：哪个 λ 最优？
