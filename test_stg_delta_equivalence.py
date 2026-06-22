#!/usr/bin/env python3
"""Equivalence test: fixed STG lambda=0.15  vs  mlp_delta lambda0=0.15 eps=0.0

Covers:
  Test A — constructor RNG preservation (global RNG state identical after init)
  Test B — post-constructor random sample equivalence
  Test C — forward pass equivalence (logits + seg_w)
  Test D — old modes unchanged
  Test E — delta mode with eps > 0 smoke
"""
import torch
import numpy as np

from model import VideoClassifier

SEED = 42
B, S, F, C, H, W = 2, 6, 3, 3, 224, 224


# ================================================================
#  Test A: constructor RNG preservation
# ================================================================
def test_rng_preservation():
    """Gate construction must NOT alter the global RNG state.

    After constructing a model with gate=none vs gate=mlp_delta (eps=0),
    the global RNG state must be identical, so that downstream operations
    (DataLoader shuffle, augmentation, dropout) produce the same sequence.
    """
    print("Test A: constructor RNG preservation")
    print("-" * 50)

    for gate_mode in ["mlp_delta", "stats3_delta"]:
        # --- Build none model, capture RNG after ---
        torch.manual_seed(SEED)
        _m_none = VideoClassifier(
            num_classes=2, segment_count=S, frames_per_segment=F,
            backbone_source="torchvision", pretrained=False,
            video_dropout=0.5, selection_mode="stg_crsa_residual",
            stg_adaptive_gate="none", stg_residual_lambda=0.15,
        )
        rng_after_none = torch.get_rng_state()

        if torch.cuda.is_available():
            cuda_rng_after_none = torch.cuda.get_rng_state_all()
        else:
            cuda_rng_after_none = None

        # --- Build delta model, capture RNG after ---
        torch.manual_seed(SEED)
        _m_delta = VideoClassifier(
            num_classes=2, segment_count=S, frames_per_segment=F,
            backbone_source="torchvision", pretrained=False,
            video_dropout=0.5, selection_mode="stg_crsa_residual",
            stg_adaptive_gate=gate_mode,
            stg_delta_base_lambda=0.15, stg_delta_eps=0.0,
        )
        rng_after_delta = torch.get_rng_state()

        if torch.cuda.is_available():
            cuda_rng_after_delta = torch.cuda.get_rng_state_all()
        else:
            cuda_rng_after_delta = None

        cpu_ok = torch.equal(rng_after_none, rng_after_delta)
        cuda_ok = True
        if cuda_rng_after_none is not None and cuda_rng_after_delta is not None:
            cuda_ok = all(
                torch.equal(a, b)
                for a, b in zip(cuda_rng_after_none, cuda_rng_after_delta)
            )

        status = "PASS" if (cpu_ok and cuda_ok) else "FAIL"
        print(f"  {gate_mode:16s}  CPU RNG match={cpu_ok}  CUDA RNG match={cuda_ok}  -> {status}")
        if not (cpu_ok and cuda_ok):
            return False
    return True


# ================================================================
#  Test B: post-constructor random sample equivalence
# ================================================================
def test_post_constructor_random():
    """After building the model, the next torch.randn() calls must produce
    identical values regardless of gate mode — confirming that gate
    construction does not perturb the downstream random sequence.
    """
    print("\nTest B: post-constructor random sample equivalence")
    print("-" * 50)

    for gate_mode in ["mlp_delta", "stats3_delta"]:
        torch.manual_seed(SEED)
        _m_none = VideoClassifier(
            num_classes=2, segment_count=S, frames_per_segment=F,
            backbone_source="torchvision", pretrained=False,
            video_dropout=0.5, selection_mode="stg_crsa_residual",
            stg_adaptive_gate="none", stg_residual_lambda=0.15,
        )
        x_none = torch.randn(10)

        torch.manual_seed(SEED)
        _m_delta = VideoClassifier(
            num_classes=2, segment_count=S, frames_per_segment=F,
            backbone_source="torchvision", pretrained=False,
            video_dropout=0.5, selection_mode="stg_crsa_residual",
            stg_adaptive_gate=gate_mode,
            stg_delta_base_lambda=0.15, stg_delta_eps=0.0,
        )
        x_delta = torch.randn(10)

        match = torch.equal(x_none, x_delta)
        status = "PASS" if match else "FAIL"
        diff = (x_none - x_delta).abs().max().item() if not match else 0.0
        print(f"  {gate_mode:16s}  torch.randn(10) match={match}  max_diff={diff:.2e}  -> {status}")
        if not match:
            return False
    return True


# ================================================================
#  Test C: forward pass equivalence
# ================================================================
def _forward_equiv_check(gate_mode):
    torch.manual_seed(SEED)
    model_fixed = VideoClassifier(
        num_classes=2, segment_count=S, frames_per_segment=F,
        backbone_source="torchvision", pretrained=False,
        video_dropout=0.5, selection_mode="stg_crsa_residual",
        stg_adaptive_gate="none", stg_residual_lambda=0.15,
    )

    torch.manual_seed(SEED)
    model_delta = VideoClassifier(
        num_classes=2, segment_count=S, frames_per_segment=F,
        backbone_source="torchvision", pretrained=False,
        video_dropout=0.5, selection_mode="stg_crsa_residual",
        stg_adaptive_gate=gate_mode,
        stg_delta_base_lambda=0.15, stg_delta_eps=0.0,
    )

    # Weights must be identical
    for (n1, p1), (n2, p2) in zip(
        model_fixed.named_parameters(), model_delta.named_parameters()
    ):
        if not torch.equal(p1, p2):
            print(f"    PARAM MISMATCH: {n1} max_diff={(p1-p2).abs().max():.2e}")
            return False

    # Forward pass
    torch.manual_seed(SEED + 999)
    x = torch.randn(B, S, F, C, H, W)
    model_fixed.eval()
    model_delta.eval()

    with torch.no_grad():
        _, vl_fixed = model_fixed(x)
        _, vl_delta = model_delta(x)
        sw_fixed = model_fixed.last_segment_weights
        sw_delta = model_delta.last_segment_weights

    logits_diff = (vl_fixed - vl_delta).abs().max().item()
    segw_diff = (sw_fixed - sw_delta).abs().max().item()

    gate_stats_ok = (model_delta.last_gate_stats is None and model_delta.last_gate_raw is None)

    tol = 1e-6
    ok = logits_diff < tol and segw_diff < tol and gate_stats_ok
    status = "PASS" if ok else "FAIL"
    print(f"  {gate_mode:16s}  logits_diff={logits_diff:.2e}  segw_diff={segw_diff:.2e}  "
          f"gate_bypass={gate_stats_ok}  -> {status}")
    return ok


def test_forward_equivalence():
    print("\nTest C: forward pass equivalence")
    print("-" * 50)
    ok1 = _forward_equiv_check("mlp_delta")
    ok2 = _forward_equiv_check("stats3_delta")
    return ok1 and ok2


# ================================================================
#  Test D: old modes unchanged
# ================================================================
def test_old_modes_unchanged():
    print("\nTest D: old modes (none, stats3, mlp) unchanged")
    print("-" * 50)
    for gate in ["none", "stats3", "mlp"]:
        torch.manual_seed(SEED)
        m = VideoClassifier(
            num_classes=2, segment_count=4, frames_per_segment=3,
            backbone_source="torchvision", pretrained=False,
            video_dropout=0.5, selection_mode="stg_crsa_residual",
            stg_adaptive_gate=gate,
        )
        x = torch.randn(2, 4, 3, 3, 224, 224)
        with torch.no_grad():
            fl, vl = m(x)
        assert fl.shape == (2, 4, 3, 2), f"bad frame_logits shape for gate={gate}"
        assert vl.shape == (2, 2), f"bad video_logits shape for gate={gate}"
        print(f"  {gate:10s}  forward OK")
    print("  -> PASS")


# ================================================================
#  Test E: delta mode with eps > 0 smoke
# ================================================================
def test_delta_mode_with_nonzero_eps():
    print("\nTest E: delta mode with eps > 0 smoke")
    print("-" * 50)
    for gate in ["stats3_delta", "mlp_delta"]:
        torch.manual_seed(SEED)
        m = VideoClassifier(
            num_classes=2, segment_count=4, frames_per_segment=3,
            backbone_source="torchvision", pretrained=False,
            video_dropout=0.5, selection_mode="stg_crsa_residual",
            stg_adaptive_gate=gate,
            stg_delta_base_lambda=0.15, stg_delta_eps=0.05,
            stg_delta_min=0.0, stg_delta_max=1.0,
        )
        x = torch.randn(2, 4, 3, 3, 224, 224)
        with torch.no_grad():
            fl, vl = m(x)
        assert fl.shape == (2, 4, 3, 2)
        assert vl.shape == (2, 2)
        le = m.last_gate_stats["lambda_eff"]
        d = m.last_gate_stats["gate"]
        assert 0.14 < le.mean().item() < 0.16, \
            f"lambda_eff mean should be ~0.15, got {le.mean():.4f}"
        assert d.abs().max().item() < 1e-4, \
            f"delta should be ~0 with zero-init, got {d.abs().max():.6f}"
        print(f"  {gate:16s}  lambda_eff_mean={le.mean():.4f}  delta_max={d.abs().max():.6f}  OK")
    print("  -> PASS")


# ================================================================
if __name__ == "__main__":
    all_ok = True

    all_ok &= test_rng_preservation()
    all_ok &= test_post_constructor_random()
    all_ok &= test_forward_equivalence()
    test_old_modes_unchanged()
    test_delta_mode_with_nonzero_eps()

    print("\n" + "=" * 70)
    if all_ok:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED — see above")
    print("=" * 70)
