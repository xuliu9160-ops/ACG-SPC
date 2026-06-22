#!/usr/bin/env python3
"""Equivalence test: fixed STG lambda=0.15  vs  mlp_delta lambda0=0.15 eps=0.0

Verifies that with eps=0, the delta path produces identical output to fixed STG.
"""
import torch
import numpy as np

from model import VideoClassifier

SEED = 42


def test_equivalence():
    B, S, F, C, H, W = 2, 6, 3, 3, 224, 224

    # ---- Build fixed STG model ----
    torch.manual_seed(SEED)
    model_fixed = VideoClassifier(
        num_classes=2,
        segment_count=S,
        frames_per_segment=F,
        backbone_source="torchvision",
        pretrained=False,
        video_dropout=0.5,
        selection_mode="stg_crsa_residual",
        stg_adaptive_gate="none",
        stg_residual_lambda=0.15,
    )

    # ---- Build mlp_delta eps=0 model (same seed) ----
    torch.manual_seed(SEED)
    model_delta = VideoClassifier(
        num_classes=2,
        segment_count=S,
        frames_per_segment=F,
        backbone_source="torchvision",
        pretrained=False,
        video_dropout=0.5,
        selection_mode="stg_crsa_residual",
        stg_adaptive_gate="mlp_delta",
        stg_delta_base_lambda=0.15,
        stg_delta_eps=0.0,
    )

    # ---- Check backbone weights are identical ----
    for (n1, p1), (n2, p2) in zip(
        model_fixed.named_parameters(), model_delta.named_parameters()
    ):
        if not torch.equal(p1, p2):
            print(f"  PARAM MISMATCH: {n1} vs {n2}")
            print(f"    max_abs_diff = {(p1 - p2).abs().max().item():.2e}")
            return False

    print("  Backbone/classifier weights: IDENTICAL (same RNG state)")

    # ---- Forward pass with same input ----
    torch.manual_seed(SEED + 999)
    x = torch.randn(B, S, F, C, H, W)

    model_fixed.eval()
    model_delta.eval()

    with torch.no_grad():
        fl_fixed, vl_fixed = model_fixed(x)
        fl_delta, vl_delta = model_delta(x)

        # seg_w comparison (stored in last_segment_weights)
        seg_w_fixed = model_fixed.last_segment_weights
        seg_w_delta = model_delta.last_segment_weights

    max_diff_logits = (vl_fixed - vl_delta).abs().max().item()
    max_diff_seg_w = (seg_w_fixed - seg_w_delta).abs().max().item()

    print(f"  video_logits max_abs_diff: {max_diff_logits:.2e}")
    print(f"  seg_w       max_abs_diff: {max_diff_seg_w:.2e}")

    # ---- Check gate stats are None for eps=0 delta ----
    assert model_delta.last_gate_stats is None, (
        f"Expected last_gate_stats=None for eps=0 delta, got {model_delta.last_gate_stats}"
    )
    assert model_delta.last_gate_raw is None, (
        f"Expected last_gate_raw=None for eps=0 delta, got {model_delta.last_gate_raw}"
    )
    print("  Gate stats: correctly None (eps=0 bypass active)")

    # ---- Assertions ----
    tol = 1e-6
    logits_ok = max_diff_logits < tol
    segw_ok = max_diff_seg_w < tol

    print()
    if logits_ok and segw_ok:
        print("PASS: mlp_delta eps=0.0 is identical to fixed STG lambda=0.15")
        return True
    else:
        if not logits_ok:
            print(f"FAIL: video_logits diff {max_diff_logits:.2e} >= {tol}")
        if not segw_ok:
            print(f"FAIL: seg_w diff {max_diff_seg_w:.2e} >= {tol}")
        return False


def test_stats3_delta_equivalence():
    """Same test for stats3_delta eps=0"""
    B, S, F, C, H, W = 2, 6, 3, 3, 224, 224

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
        stg_adaptive_gate="stats3_delta",
        stg_delta_base_lambda=0.15, stg_delta_eps=0.0,
    )

    for (n1, p1), (n2, p2) in zip(
        model_fixed.named_parameters(), model_delta.named_parameters()
    ):
        if not torch.equal(p1, p2):
            print(f"  PARAM MISMATCH: {n1} vs {n2}")
            return False

    print("  Backbone/classifier weights: IDENTICAL (same RNG state)")

    torch.manual_seed(SEED + 999)
    x = torch.randn(B, S, F, C, H, W)
    model_fixed.eval()
    model_delta.eval()

    with torch.no_grad():
        _, vl_fixed = model_fixed(x)
        _, vl_delta = model_delta(x)
        seg_w_fixed = model_fixed.last_segment_weights
        seg_w_delta = model_delta.last_segment_weights

    max_diff_logits = (vl_fixed - vl_delta).abs().max().item()
    max_diff_seg_w = (seg_w_fixed - seg_w_delta).abs().max().item()

    print(f"  video_logits max_abs_diff: {max_diff_logits:.2e}")
    print(f"  seg_w       max_abs_diff: {max_diff_seg_w:.2e}")
    print(f"  Gate stats: {'correctly None' if model_delta.last_gate_stats is None else 'BUG: not None'}")

    tol = 1e-6
    if max_diff_logits < tol and max_diff_seg_w < tol:
        print("PASS: stats3_delta eps=0.0 is identical to fixed STG lambda=0.15")
        return True
    else:
        print("FAIL")
        return False


def test_old_modes_unchanged():
    """Smoke: old modes (none, stats3, mlp) still work."""
    B, S, F, C, H, W = 2, 4, 3, 3, 224, 224
    for gate in ["none", "stats3", "mlp"]:
        torch.manual_seed(SEED)
        m = VideoClassifier(
            num_classes=2, segment_count=S, frames_per_segment=F,
            backbone_source="torchvision", pretrained=False,
            video_dropout=0.5, selection_mode="stg_crsa_residual",
            stg_adaptive_gate=gate,
        )
        x = torch.randn(B, S, F, C, H, W)
        with torch.no_grad():
            fl, vl = m(x)
        assert fl.shape == (B, S, F, 2), f"bad frame_logits shape for gate={gate}"
        assert vl.shape == (B, 2), f"bad video_logits shape for gate={gate}"
    print("PASS: old modes (none, stats3, mlp) unchanged")


def test_delta_mode_with_nonzero_eps():
    """Smoke: delta mode with eps > 0 produces valid lambda_eff in range."""
    B, S, F, C, H, W = 2, 4, 3, 3, 224, 224
    for gate in ["stats3_delta", "mlp_delta"]:
        torch.manual_seed(SEED)
        m = VideoClassifier(
            num_classes=2, segment_count=S, frames_per_segment=F,
            backbone_source="torchvision", pretrained=False,
            video_dropout=0.5, selection_mode="stg_crsa_residual",
            stg_adaptive_gate=gate,
            stg_delta_base_lambda=0.15, stg_delta_eps=0.05,
            stg_delta_min=0.0, stg_delta_max=1.0,
        )
        x = torch.randn(B, S, F, C, H, W)
        with torch.no_grad():
            fl, vl = m(x)
        assert fl.shape == (B, S, F, 2)
        assert vl.shape == (B, 2)
        le = m.last_gate_stats["lambda_eff"]
        d = m.last_gate_stats["gate"]
        # With zero-init MLP, delta≈0 and lambda_eff≈0.15
        assert 0.14 < le.mean().item() < 0.16, f"lambda_eff mean should be ~0.15, got {le.mean().item():.4f}"
        assert d.abs().max().item() < 1e-4, f"delta should be ~0 with zero-init, got {d.abs().max().item():.6f}"
    print("PASS: delta mode with eps > 0 produces valid output")


if __name__ == "__main__":
    print("=" * 70)
    print("Test 1: mlp_delta eps=0 vs fixed STG equivalence")
    print("-" * 70)
    ok1 = test_equivalence()

    print()
    print("=" * 70)
    print("Test 2: stats3_delta eps=0 vs fixed STG equivalence")
    print("-" * 70)
    ok2 = test_stats3_delta_equivalence()

    print()
    print("=" * 70)
    print("Test 3: Old modes unchanged")
    print("-" * 70)
    test_old_modes_unchanged()

    print()
    print("=" * 70)
    print("Test 4: Delta mode with eps > 0")
    print("-" * 70)
    test_delta_mode_with_nonzero_eps()

    print()
    print("=" * 70)
    if ok1 and ok2:
        print("ALL EQUIVALENCE TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)
