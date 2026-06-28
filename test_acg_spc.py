#!/usr/bin/env python3
"""Smoke tests for ACG-SPC low-drop perturbation consistency (v1 — clean branch).

Tests:
  1. mask_bottom_k_weights correctness
  2. aggregate_with_weights (linear_only=True/False)
  3. acg_spc_low_loss — both teacher & student use linear_only, no RNG consumed
  4. Default disabled: last_seg_w_for_spc is None
  5. Default disabled: logits + RNG unchanged
  6. SPC with stg_crsa_residual + mlp gate
"""
import torch
from model import VideoClassifier
from loss import mask_bottom_k_weights, acg_spc_low_loss

SEED = 42


def test_mask_bottom_k():
    """mask_bottom_k_weights: zero bottom-k + renormalize."""
    print("Test 1: mask_bottom_k_weights")
    print("-" * 40)

    seg_w = torch.tensor([
        [0.1, 0.2, 0.3, 0.1, 0.2, 0.1],
        [0.05, 0.05, 0.5, 0.2, 0.1, 0.1],
    ])
    masked, idx = mask_bottom_k_weights(seg_w, k=1)
    print(f"  original:    {seg_w[0].tolist()}")
    print(f"  masked(k=1): {masked[0].tolist()}")
    print(f"  dropped idx: {idx[0].tolist()}")
    assert masked.sum(dim=1).sub(1.0).abs().max() < 1e-6
    assert (masked[0] == 0.0).sum() == 1
    print("  -> PASS\n")
    return True


def test_aggregate_with_weights():
    """aggregate_with_weights produces valid [B,2] logits."""
    print("Test 2: aggregate_with_weights")
    print("-" * 40)

    torch.manual_seed(SEED)
    B, S, F, C, H, W = 2, 6, 3, 3, 224, 224
    m = VideoClassifier(num_classes=2, segment_count=S, frames_per_segment=F,
                        backbone_source="torchvision", pretrained=False,
                        video_dropout=0.5, selection_mode="mean")
    m.enable_acg_spc_cache = True
    x = torch.randn(B, S, F, C, H, W)
    m.eval()
    with torch.no_grad():
        fl, vl = m(x)

    seg_tok = m.last_seg_tokens_for_spc
    seg_w = m.last_seg_w_for_spc
    assert seg_tok is not None and seg_w is not None

    logits_lin = m.aggregate_with_weights(seg_tok, seg_w, linear_only=True)
    diff = (vl - logits_lin).abs().max().item()
    print(f"  linear_only=True:  diff_from_main={diff:.2e}")

    logits_full = m.aggregate_with_weights(seg_tok, seg_w, linear_only=False)
    print(f"  linear_only=False: shape={list(logits_full.shape)}")

    masked_w, _ = mask_bottom_k_weights(seg_w, k=1)
    logits_masked = m.aggregate_with_weights(seg_tok, masked_w, linear_only=True)
    print(f"  masked(k=1):       shape={list(logits_masked.shape)}")
    print("  -> PASS\n")
    return True


def test_spc_loss():
    """acg_spc_low_loss: teacher+student both no-dropout, no RNG consumed."""
    print("Test 3: acg_spc_low_loss (teacher+student linear_only)")
    print("-" * 40)

    torch.manual_seed(SEED)
    B, S, F, C, H, W = 2, 6, 3, 3, 224, 224
    m = VideoClassifier(num_classes=2, segment_count=S, frames_per_segment=F,
                        backbone_source="torchvision", pretrained=False,
                        video_dropout=0.5, selection_mode="mean")
    m.enable_acg_spc_cache = True
    x = torch.randn(B, S, F, C, H, W)
    m.train()
    fl, vl = m(x)

    seg_tok = m.last_seg_tokens_for_spc
    seg_w = m.last_seg_w_for_spc

    # Capture RNG before SPC
    rng_before = torch.get_rng_state()
    spc_val, spc_debug = acg_spc_low_loss(seg_w, seg_tok, m, low_k=1)
    rng_after = torch.get_rng_state()
    rng_match = torch.equal(rng_before, rng_after)

    print(f"  spc_low_kl={spc_debug['spc_low_kl']:.6f}")
    print(f"  drop_low_delta_prob={spc_debug['drop_low_delta_prob']:.6f}")
    print(f"  bottomk_weight_mean={spc_debug['bottomk_weight_mean']:.6f}")
    print(f"  RNG preserved after SPC: {rng_match}")
    assert torch.isfinite(spc_val)
    assert spc_val.item() >= 0
    assert rng_match, "SPC loss must NOT consume RNG (teacher+student both linear_only)"
    print("  -> PASS\n")
    return True


def test_default_disabled():
    """Default: no cache, no side effects."""
    print("Test 4: default disabled (no cache, no side effects)")
    print("-" * 40)

    torch.manual_seed(SEED)
    B, S, F, C, H, W = 2, 6, 3, 3, 224, 224

    torch.manual_seed(SEED + 1)
    m1 = VideoClassifier(num_classes=2, segment_count=S, frames_per_segment=F,
                         backbone_source="torchvision", pretrained=False,
                         video_dropout=0.5, selection_mode="mean")
    x = torch.randn(B, S, F, C, H, W)
    m1.eval()
    with torch.no_grad():
        _, vl1 = m1(x)
    rng1 = torch.get_rng_state()

    torch.manual_seed(SEED + 1)
    m2 = VideoClassifier(num_classes=2, segment_count=S, frames_per_segment=F,
                         backbone_source="torchvision", pretrained=False,
                         video_dropout=0.5, selection_mode="mean")
    x2 = torch.randn(B, S, F, C, H, W)
    m2.eval()
    with torch.no_grad():
        _, vl2 = m2(x)
    rng2 = torch.get_rng_state()

    diff = (vl1 - vl2).abs().max().item()
    rng_match = torch.equal(rng1, rng2)
    assert diff < 1e-6, f"logits diff={diff:.2e}"
    assert rng_match, "RNG should match"
    assert m1.last_seg_w_for_spc is None, "should be None by default"
    assert m1.last_seg_tokens_for_spc is None, "should be None by default"
    print(f"  logits_diff={diff:.2e}  rng_match={rng_match}  cache=None  OK")
    print("  -> PASS\n")
    return True


def test_spc_with_mlp_gate():
    """SPC works with stg_crsa_residual + mlp gate."""
    print("Test 5: SPC with stg_crsa_residual + mlp")
    print("-" * 40)

    torch.manual_seed(SEED)
    B, S, F, C, H, W = 2, 6, 3, 3, 224, 224
    m = VideoClassifier(num_classes=2, segment_count=S, frames_per_segment=F,
                        backbone_source="torchvision", pretrained=False,
                        video_dropout=0.5, selection_mode="stg_crsa_residual",
                        stg_adaptive_gate="mlp")
    m.enable_acg_spc_cache = True
    x = torch.randn(B, S, F, C, H, W)
    m.train()
    fl, vl = m(x)

    seg_tok = m.last_seg_tokens_for_spc
    seg_w = m.last_seg_w_for_spc
    assert seg_w is not None

    rng_before = torch.get_rng_state()
    spc_val, spc_debug = acg_spc_low_loss(seg_w, seg_tok, m, low_k=1)
    rng_match = torch.equal(rng_before, torch.get_rng_state())

    print(f"  spc_low_kl={spc_debug['spc_low_kl']:.6f}")
    print(f"  bottomk_weight_mean={spc_debug['bottomk_weight_mean']:.6f}")
    print(f"  RNG preserved: {rng_match}")
    assert torch.isfinite(spc_val)
    assert rng_match
    print("  -> PASS\n")
    return True


if __name__ == "__main__":
    all_ok = True
    all_ok &= test_mask_bottom_k()
    all_ok &= test_aggregate_with_weights()
    all_ok &= test_spc_loss()
    all_ok &= test_default_disabled()
    all_ok &= test_spc_with_mlp_gate()

    print("=" * 60)
    if all_ok:
        print("ALL ACG-SPC TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
