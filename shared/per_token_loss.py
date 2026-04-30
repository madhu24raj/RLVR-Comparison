"""
Shared per-token loss utilities for PPO and GRPO trainers.

Provides the core math that both algorithms share:
  - batched_per_token_log_probs: per-token response log probs + mask
  - clipped_surrogate_loss: PPO-clip objective at the token level
  - per_token_kl: masked KL divergence between two log-prob tensors

These are pure functions (no trainer state) so both PPOTrainer and
GRPOTrainer can import them without coupling to each other.
"""
from __future__ import annotations

import torch
from typing import List, Optional, Tuple

from transformers import AutoModelForCausalLM


def batched_per_token_log_probs(
    model: AutoModelForCausalLM,
    all_full_ids: List[List[int]],
    prompt_lens: List[int],
    pad_token_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token response log-probs for a batch.

    Returns the log probability of each *response* token under the given
    model, right-padded across the batch, plus a 1/0 mask marking which
    positions correspond to real tokens.

    Args:
        model:         CausalLM to run the forward pass through.
        all_full_ids:  list of [prompt | response] token id lists.
        prompt_lens:   prompt lengths (so we can slice off the response).
        pad_token_id:  tokenizer's pad token id for padding.
        device:        torch device.

    Returns:
        log_probs: [B, T_max_response]  per-token log probs (response only)
        mask:      [B, T_max_response]  1 where real, 0 where padding
    """
    B = len(all_full_ids)
    max_len = max(len(ids) for ids in all_full_ids)
    padded = torch.full(
        (B, max_len), pad_token_id,
        dtype=torch.long, device=device,
    )
    attention_mask = torch.zeros(
        B, max_len, dtype=torch.long, device=device,
    )
    for i, ids in enumerate(all_full_ids):
        padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :len(ids)] = 1

    outputs = model(
        input_ids=padded, attention_mask=attention_mask, use_cache=False,
    )

    max_resp_len = max(
        max(len(all_full_ids[i]) - prompt_lens[i], 0) for i in range(B)
    )
    if max_resp_len == 0:
        zeros = torch.zeros((B, 1), device=device)
        return zeros, torch.zeros((B, 1), device=device)

    per_token = torch.zeros((B, max_resp_len), device=device)
    mask = torch.zeros((B, max_resp_len), device=device)

    # Process one sample at a time to avoid materializing the full
    # [B, S, V] log_softmax tensor. With Qwen2.5's 151k vocab, the
    # full tensor is ~4.7 GB in fp32 at batch=16, seq=400 — too large
    # for a T4 alongside two model copies.
    for i in range(B):
        pl = prompt_lens[i]
        seq_len = len(all_full_ids[i])
        resp_len = seq_len - pl
        if resp_len <= 0:
            continue
        # Only compute log_softmax on this sample's response positions
        response_logits = outputs.logits[i, pl - 1 : seq_len - 1, :].float()  # [R, V]
        response_log_probs = torch.log_softmax(response_logits, dim=-1)  # [R, V]
        response_ids = padded[i, pl:seq_len]  # [R]
        token_lp = response_log_probs.gather(
            1, response_ids.unsqueeze(-1)
        ).squeeze(-1)  # [R]
        per_token[i, :resp_len] = token_lp
        mask[i, :resp_len] = 1.0
        del response_logits, response_log_probs  # free immediately

    return per_token, mask


def clipped_surrogate_loss(
    new_per_token: torch.Tensor,   # [B, T]
    old_per_token: torch.Tensor,   # [B, T]
    advantages: torch.Tensor,      # [B]
    mask: torch.Tensor,            # [B, T]
    clip_epsilon: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token PPO-clip surrogate loss, masked mean.

    Args:
        new_per_token: per-token log probs under current policy (with grad).
        old_per_token: per-token log probs under old policy (detached).
        advantages:    per-sample advantages [B].
        mask:          response token mask [B, T].
        clip_epsilon:  PPO clip range.

    Returns:
        policy_loss:   scalar, the masked-mean clipped surrogate.
        clip_fraction: scalar, fraction of tokens where clipping engaged.
        ratio:         [B, T] importance sampling ratio (detached).
    """
    log_ratio = new_per_token - old_per_token
    log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
    ratio = torch.exp(log_ratio)
    A_expanded = advantages.unsqueeze(-1)  # [B, 1]

    unclipped = ratio * A_expanded
    clipped = torch.clamp(
        ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon
    ) * A_expanded
    per_token_pg = -torch.min(unclipped, clipped)

    mask_sum = mask.sum().clamp(min=1.0)
    policy_loss = (per_token_pg * mask).sum() / mask_sum

    ratio_detached = ratio.detach()
    clip_hits = ((ratio_detached - 1.0).abs() > clip_epsilon).float()
    clip_fraction = (clip_hits * mask).sum() / mask_sum

    return policy_loss, clip_fraction, ratio_detached


def per_token_kl(
    log_probs_a: torch.Tensor,  # [B, T]
    log_probs_b: torch.Tensor,  # [B, T]
    mask: torch.Tensor,         # [B, T]
) -> torch.Tensor:
    """Masked per-token KL: mean((a - b) * mask) / sum(mask).

    Returns KL(a || b) estimated as E_a[log a - log b].
    The direction depends on what you pass:
      - KL(old || new): pass (old, new)
      - KL(new || ref): pass (new, ref)
    """
    mask_sum = mask.sum().clamp(min=1.0)
    kl_per_token = (log_probs_a - log_probs_b) * mask
    return kl_per_token.sum() / mask_sum
