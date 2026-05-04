"""
Tests for the P14 fused log_softmax+gather optimization in
``shared.per_token_loss.batched_per_token_log_probs``.

The function under test now uses ``F.cross_entropy(reduction='none')`` to
fuse the per-position ``log_softmax`` followed by ``gather`` into a single
kernel. These tests verify that:

  1. The fused output is numerically equivalent to the naive
     ``log_softmax + gather`` implementation (rtol=1e-5, atol=1e-6).
  2. The function correctly handles the empty-response edge case
     (``prompt_len == seq_len``).
  3. Output dtype is fp32 even when the model emits bf16 logits.
  4. Gradients flow through the fused path (``token_lp.grad_fn`` set,
     ``backward()`` populates ``slice_logits.grad``).

These run on CPU with synthetic logits (no real LM load), so they are
fast and not marked ``slow``.

Run with:
    pytest ppo_specs/tests/test_fused_log_probs.py -v
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Make the RLVR-Comparison root importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from shared.per_token_loss import batched_per_token_log_probs  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic stand-in for an HF CausalLM
# ---------------------------------------------------------------------------


class _SyntheticLM:
    """Minimal stand-in for an HF CausalLM.

    Ignores ``input_ids`` and returns a fixed pre-built logits tensor
    ``[B, S, V]`` wrapped in a ``SimpleNamespace`` so callers can do
    ``outputs.logits``. This lets us test ``batched_per_token_log_probs``
    deterministically against a synthetic ground-truth tensor without
    loading a real model.
    """

    def __init__(self, logits: torch.Tensor):
        self._logits = logits

    def __call__(self, input_ids=None, attention_mask=None, use_cache=False):
        return SimpleNamespace(logits=self._logits)


def _make_synthetic_inputs(
    B: int = 4,
    S: int = 20,
    V: int = 50,
    pad_token_id: int = 0,
    seed: int = 42,
    requires_grad: bool = False,
    dtype: torch.dtype = torch.float32,
) -> tuple:
    """Build ``(model, all_full_ids, prompt_lens, logits)`` for tests."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, S, V, generator=g, dtype=dtype)
    if requires_grad:
        logits.requires_grad_(True)

    # Construct varying-length sequences entirely within S:
    # prompt_len in [4, 7], full length in [12, S]. Token ids in [1, V-1]
    # (avoid pad_token_id=0 inside real positions).
    rng = torch.Generator().manual_seed(seed + 1)
    all_full_ids = []
    prompt_lens = []
    for i in range(B):
        pl = 4 + i % 4                # 4, 5, 6, 7, 4, ...
        full_len = 12 + (i * 3) % (S - 12 + 1)  # within [12, S]
        full_len = min(full_len, S)
        ids = (
            1 + torch.randint(0, V - 1, (full_len,), generator=rng)
        ).tolist()
        all_full_ids.append(ids)
        prompt_lens.append(pl)

    model = _SyntheticLM(logits)
    return model, all_full_ids, prompt_lens, logits


# ---------------------------------------------------------------------------
# 1. Numerical parity vs. the naive log_softmax+gather path
# ---------------------------------------------------------------------------


def test_fused_matches_naive_numerically():
    """Fused cross_entropy must match ``log_softmax + gather`` to fp32 tol.

    Builds a synthetic ``[B=4, S=20, V=50]`` logit tensor, runs the public
    ``batched_per_token_log_probs`` (which now uses cross_entropy) AND the
    legacy ``log_softmax + gather`` path inline against the same logits and
    same prompt/response slicing, then asserts close.
    """
    device = torch.device("cpu")
    pad_id = 0
    B, S, V = 4, 20, 50

    model, all_full_ids, prompt_lens, logits = _make_synthetic_inputs(
        B=B, S=S, V=V, pad_token_id=pad_id,
    )

    # Pad to max_len like the function does, for the inline naive path.
    max_len = max(len(ids) for ids in all_full_ids)
    padded = torch.full(
        (B, max_len), pad_id, dtype=torch.long, device=device,
    )
    for i, ids in enumerate(all_full_ids):
        padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    max_resp_len = max(
        len(all_full_ids[i]) - prompt_lens[i] for i in range(B)
    )
    assert max_resp_len > 0, "test setup error: need at least one response token"

    # ── Naive reference: log_softmax + gather, sample-by-sample ─────────
    expected_per_token = torch.zeros((B, max_resp_len), device=device)
    expected_mask = torch.zeros((B, max_resp_len), device=device)
    for i in range(B):
        pl = prompt_lens[i]
        seq_len = len(all_full_ids[i])
        resp_len = seq_len - pl
        if resp_len <= 0:
            continue
        slice_logits = logits[i, pl - 1 : seq_len - 1, :].float()
        lp_full = F.log_softmax(slice_logits, dim=-1)
        target_ids = padded[i, pl:seq_len]
        token_lp = lp_full.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
        expected_per_token[i, :resp_len] = token_lp
        expected_mask[i, :resp_len] = 1.0

    # ── Fused implementation under test ─────────────────────────────────
    got_per_token, got_mask = batched_per_token_log_probs(
        model, all_full_ids, prompt_lens,
        pad_token_id=pad_id, device=device,
    )

    torch.testing.assert_close(
        got_per_token, expected_per_token, rtol=1e-5, atol=1e-6,
    )
    torch.testing.assert_close(
        got_mask, expected_mask, rtol=0.0, atol=0.0,
    )


# ---------------------------------------------------------------------------
# 2. Empty-response edge case
# ---------------------------------------------------------------------------


def test_fused_handles_empty_response():
    """When ``prompt_len == seq_len`` for every sample, return zeros.

    Verifies the function does not crash and the returned tensors are
    elementwise zero (per the existing ``max_resp_len == 0`` early-return
    contract).
    """
    device = torch.device("cpu")
    pad_id = 0
    B, S, V = 3, 10, 25

    g = torch.Generator().manual_seed(7)
    logits = torch.randn(B, S, V, generator=g)
    model = _SyntheticLM(logits)

    rng = torch.Generator().manual_seed(8)
    all_full_ids = []
    prompt_lens = []
    for i in range(B):
        full_len = 5 + i  # 5, 6, 7
        ids = (
            1 + torch.randint(0, V - 1, (full_len,), generator=rng)
        ).tolist()
        all_full_ids.append(ids)
        prompt_lens.append(full_len)  # prompt_len == full_len → empty response

    per_token, mask = batched_per_token_log_probs(
        model, all_full_ids, prompt_lens,
        pad_token_id=pad_id, device=device,
    )

    # Function early-returns ``zeros((B, 1))`` for both tensors.
    assert per_token.shape == (B, 1)
    assert mask.shape == (B, 1)
    assert torch.all(per_token == 0.0), (
        f"expected all-zero per_token, got {per_token}"
    )
    assert torch.all(mask == 0.0), (
        f"expected all-zero mask, got {mask}"
    )


# ---------------------------------------------------------------------------
# 3. Output dtype is fp32 even with bf16 input logits
# ---------------------------------------------------------------------------


def test_fused_dtype():
    """Output dtype must be fp32 even when the LM emits bf16 logits.

    The cross_entropy call upcasts ``slice_logits.float()`` so the per-token
    log-prob has fp32 precision regardless of the model's output dtype.
    The pre-allocated ``per_token`` buffer is fp32 (default), so assigning
    fp32 values into it preserves fp32 throughout.
    """
    device = torch.device("cpu")
    pad_id = 0
    B, S, V = 2, 12, 30

    # Build bf16 logits — emulates a bf16 (CUDA) model on the CPU side.
    g = torch.Generator().manual_seed(123)
    logits_fp32 = torch.randn(B, S, V, generator=g)
    logits_bf16 = logits_fp32.to(torch.bfloat16)
    model = _SyntheticLM(logits_bf16)

    rng = torch.Generator().manual_seed(124)
    all_full_ids = []
    prompt_lens = []
    for i in range(B):
        pl = 3 + i
        full_len = 8 + i * 2
        ids = (
            1 + torch.randint(0, V - 1, (full_len,), generator=rng)
        ).tolist()
        all_full_ids.append(ids)
        prompt_lens.append(pl)

    per_token, mask = batched_per_token_log_probs(
        model, all_full_ids, prompt_lens,
        pad_token_id=pad_id, device=device,
    )

    assert per_token.dtype == torch.float32, (
        f"expected fp32 per_token, got {per_token.dtype}"
    )
    # mask is also fp32 (default zeros allocator).
    assert mask.dtype == torch.float32, (
        f"expected fp32 mask, got {mask.dtype}"
    )
    # All masked values must be finite (no -inf/NaN from bf16 precision).
    masked_vals = per_token[mask.bool()]
    assert torch.isfinite(masked_vals).all(), (
        f"non-finite values in fused output: {masked_vals}"
    )


# ---------------------------------------------------------------------------
# 4. Grad flow: backward() populates input logits.grad
# ---------------------------------------------------------------------------


def test_fused_grad_flow():
    """Gradients must flow from ``token_lp`` back to input logits.

    With ``requires_grad=True`` on the synthetic logits, the returned
    per-token tensor must have a ``grad_fn`` set, and calling
    ``.sum().backward()`` must populate ``logits.grad`` with non-NaN values.
    """
    device = torch.device("cpu")
    pad_id = 0
    B, S, V = 3, 14, 40

    model, all_full_ids, prompt_lens, logits = _make_synthetic_inputs(
        B=B, S=S, V=V, pad_token_id=pad_id, requires_grad=True,
    )

    per_token, mask = batched_per_token_log_probs(
        model, all_full_ids, prompt_lens,
        pad_token_id=pad_id, device=device,
    )

    assert per_token.requires_grad, "per_token must require grad"
    assert per_token.grad_fn is not None, (
        "per_token has no grad_fn — autograd graph is detached"
    )

    # Reduce to scalar through the masked positions and call backward.
    loss = (per_token * mask).sum()
    loss.backward()

    assert logits.grad is not None, (
        "input logits.grad is None after backward — no grad propagated"
    )
    assert torch.isfinite(logits.grad).all(), (
        "non-finite values in logits.grad after backward"
    )
    # At least one position should have a non-zero gradient (we backprop
    # through real response tokens).
    assert (logits.grad != 0).any(), (
        "logits.grad is all-zero — backward did not touch any logit"
    )
