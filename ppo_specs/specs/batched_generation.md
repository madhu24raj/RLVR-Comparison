# Batched Generation Spec for PPO Pipeline

**Status:** Implemented  
**Author:** ML Engineering  
**Date:** 2026-04-08  

### Implementation Status

The following spec items have been implemented in `ppo_trainer.py` and `advantage.py`:

| Section | Component | Status |
|---------|-----------|--------|
| 1 | Tokenizer left-padding setup | **Implemented** in `load_ppo_trainer` |
| 2 | `generate_rollouts` batched | **Implemented** -- single `model.generate()` call |
| 3 | `_batched_sequence_log_probs` | **Implemented** -- replaces per-sample `_sequence_log_prob` |
| 4 | `_policy_log_probs` batched | **Implemented** -- delegates to `_batched_sequence_log_probs` |
| 5 | `_critic_forward` batched | **Implemented** -- single tokenization + forward pass |
| 6 | `_eval_critic_on_prompts` batched | **Implemented** |
| 7 | `evaluate` batched | **Implemented** -- uses configurable sub-batch size |
| 8 | `estimate_mc_advantages` batched | **Implemented** -- uses `repeat()` micro-batches |
| 9 | Combined forward pass (P2) | **Spec only** -- policy and critic still use separate passes |
| 10 | Memory budget analysis | **Spec only** -- reference documentation |

Tests: `tests/test_batched_ops.py` verifies log-prob consistency between batched and single-sample paths.

## Executive Summary

The current PPO pipeline processes every prompt sequentially — one `model.generate()` call, one `model()` forward pass, and one `critic()` call per sample. On a cluster with batch_size=16 and n_samples=50 for MC estimation, this means **800 sequential autoregressive generation calls** per MC evaluation, and **16 sequential calls per training step**. RLHF training [spends ~80% of wall time on generation](https://arxiv.org/pdf/2405.11143), making this the dominant bottleneck.

This spec converts every per-sample loop into a batched operation, yielding an estimated **8-16x speedup** on the generation path and **4-8x speedup** on forward-pass log-prob/critic computation, depending on hardware and sequence length variance.

---

## Table of Contents

1. [Prerequisite: Tokenizer Setup](#1-prerequisite-tokenizer-setup)
2. [generate_rollouts](#2-generate_rollouts)
3. [_sequence_log_prob](#3-_sequence_log_prob)
4. [_policy_log_probs](#4-_policy_log_probs)
5. [_critic_forward](#5-_critic_forward)
6. [_eval_critic_on_prompts](#6-_eval_critic_on_prompts)
7. [evaluate](#7-evaluate)
8. [estimate_mc_advantages](#8-estimate_mc_advantages)
9. [Combined Forward Pass Optimization](#9-combined-forward-pass-optimization)
10. [Memory Budget Analysis](#10-memory-budget-analysis)
11. [Migration Plan](#11-migration-plan)

---

## 1. Prerequisite: Tokenizer Setup

Left-padding is mandatory for batched generation with decoder-only models. The model was not trained to continue generation from padding tokens, so padding must appear on the left so that the rightmost (most recent) token is always a real token.

### Current code

```python
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### Batched replacement

```python
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
```

This single line must be set **before any batched tokenization call**. It should be placed in `load_ppo_trainer()` and in `estimate_mc_advantages()`.

> **Warning:** Right-padding with causal LMs causes the model to attend to pad tokens during generation, producing garbage. Always verify `tokenizer.padding_side == "left"` before batched `.generate()`.

---

## 2. generate_rollouts

### Current code (bottleneck)

```python
for prompt, gt in zip(prompts, ground_truths):
    enc = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                         max_length=512, padding=False).to(self.device)
    prompt_len = enc["input_ids"].shape[1]
    out = self.model.generate(**enc, max_new_tokens=..., ...)
    full_ids = out[0]
    completion = self.tokenizer.decode(full_ids[prompt_len:], ...)
    old_log_prob = self._sequence_log_prob(full_ids.unsqueeze(0), prompt_len).item()
    reward = self.reward_fn(completion, gt)
    value = self._critic_value_no_grad(enc["input_ids"])
    # ... append Rollout
```

**Cost:** B sequential `generate()` calls + B sequential forward passes for log-probs + B sequential forward passes for critic = **3B serial GPU operations**.

### Batched replacement

```python
@torch.no_grad()
def generate_rollouts(
    self,
    prompts: List[str],
    ground_truths: List[str],
) -> RolloutBatch:
    """
    Batched rollout generation: one generate() call, one forward pass for
    log-probs, one forward pass for critic values.
    """
    self.model.eval()
    B = len(prompts)

    # ── Step 1: Tokenize all prompts with left-padding ────────────────
    enc = self.tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,          # pads to longest in batch
    ).to(self.device)
    # enc["input_ids"]: [B, max_prompt_len]  (left-padded)
    # enc["attention_mask"]: [B, max_prompt_len]  (0 for pad, 1 for real)

    # Per-sample prompt lengths (number of non-pad tokens)
    prompt_lengths = enc["attention_mask"].sum(dim=1)  # [B], int

    # ── Step 2: Batched generation ────────────────────────────────────
    out_ids = self.model.generate(
        **enc,
        max_new_tokens=self.config.max_new_tokens,
        do_sample=self.config.do_sample,
        temperature=self.config.temperature,
        pad_token_id=self.tokenizer.pad_token_id,
    )
    # out_ids: [B, max_prompt_len + max_response_len]
    # Some samples may have hit EOS early; remaining tokens are pad_token_id.

    max_prompt_len = enc["input_ids"].shape[1]

    # ── Step 3: Extract per-sample completions ────────────────────────
    # The response portion starts at position max_prompt_len for all samples
    # (because prompts were left-padded to align on the right).
    response_ids = out_ids[:, max_prompt_len:]  # [B, max_response_len]

    # Build per-sample response mask: 1 for real response tokens, 0 for pad
    response_mask = (response_ids != self.tokenizer.pad_token_id).long()

    # Decode completions (batch_decode handles padding automatically)
    completions = self.tokenizer.batch_decode(
        response_ids, skip_special_tokens=True
    )

    # ── Step 4: Compute rewards (CPU-bound, cannot batch further) ─────
    rewards_list = [
        self.reward_fn(comp, gt)
        for comp, gt in zip(completions, ground_truths)
    ]

    # ── Step 5: Batched old log-probs ─────────────────────────────────
    old_log_probs = self._batched_sequence_log_prob(
        out_ids, prompt_lengths, max_prompt_len, response_mask
    )  # [B]

    # ── Step 6: Batched critic values ─────────────────────────────────
    critic_values = self._batched_critic_value_no_grad(
        enc["input_ids"], enc["attention_mask"]
    )  # [B]

    # ── Step 7: Build rollout objects ─────────────────────────────────
    rollouts: List[Rollout] = []
    for i in range(B):
        # Reconstruct full_ids without left-pad tokens for storage
        # (the Rollout stores unpadded token ids for replay)
        real_prompt_start = max_prompt_len - prompt_lengths[i].item()
        full_ids_i = out_ids[i, real_prompt_start:].tolist()

        # Trim trailing pad tokens from response
        resp_len = response_mask[i].sum().item()
        if resp_len < response_ids.shape[1]:
            # Trim trailing pads from full_ids
            trailing_pad = response_ids.shape[1] - resp_len
            full_ids_i = full_ids_i[:-trailing_pad] if trailing_pad > 0 else full_ids_i

        rollouts.append(Rollout(
            prompt=prompts[i],
            completion=completions[i],
            reward=rewards_list[i],
            old_log_prob=old_log_probs[i].item(),
            value=critic_values[i].item(),
            full_ids=full_ids_i,
            prompt_len=prompt_lengths[i].item(),
        ))

    self.total_rollouts += B
    return RolloutBatch(rollouts)
```

### Edge cases

| Case | Problem | Solution |
|------|---------|----------|
| Variable prompt lengths | Left-padding misaligns without attention mask | `attention_mask` from tokenizer handles this; `generate()` respects it |
| Early EOS in some samples | `generate()` pads remaining positions with `pad_token_id` | `response_mask` filters these out in log-prob computation |
| `pad_token == eos_token` | `response_mask` based on `pad_token_id` may incorrectly mask real EOS tokens | Use position-based masking: for each sample find the first EOS after the prompt and mask everything after it (see alternative below) |
| Empty response | Model generates EOS immediately | `_batched_sequence_log_prob` returns 0 for that sample via mask sum |

### Robust response masking (handles pad_token == eos_token)

```python
def _build_response_mask(
    self,
    response_ids: torch.Tensor,   # [B, R]
) -> torch.Tensor:
    """
    Build mask that is 1 for real response tokens, 0 for post-EOS padding.
    Handles the case where pad_token_id == eos_token_id by finding the
    FIRST occurrence of EOS in each row and masking everything after it.
    """
    B, R = response_ids.shape
    eos_id = self.tokenizer.eos_token_id

    # Find first EOS position per sample (R if no EOS found)
    is_eos = (response_ids == eos_id)
    # arange mask: position indices [0, 1, ..., R-1]
    positions = torch.arange(R, device=response_ids.device).unsqueeze(0).expand(B, R)

    # For each row, find minimum position where is_eos is True
    # If no EOS, set to R (include all tokens)
    eos_positions = torch.where(
        is_eos,
        positions,
        torch.full_like(positions, R),
    ).min(dim=1).values  # [B]

    # Include the EOS token itself, mask everything after
    # mask[i, j] = 1 if j <= eos_positions[i], else 0
    mask = (positions <= eos_positions.unsqueeze(1)).long()
    return mask
```

### Expected speedup

- **Generation:** 8-16x on a single GPU (batch_size=8-16). The autoregressive loop runs once instead of B times; GPU compute units are fully saturated. [Empirically, batched `generate()` achieves near-linear scaling up to the point where KV-cache memory is exhausted](https://docs.vllm.ai/).
- **Log-prob forward pass:** Eliminated as separate step; folded into batched call.
- **Critic forward pass:** Eliminated as separate step; folded into batched call.
- **Overall `generate_rollouts`:** ~10x wall-time reduction for batch_size=16.

### Memory impact

- KV-cache grows as `B * n_layers * 2 * n_heads * head_dim * seq_len * dtype_bytes`. For Qwen2.5-0.5B (24 layers, 896 hidden, float32) with B=16, max_seq=768: ~3.4 GB KV-cache vs ~0.2 GB sequential. This fits comfortably on a 24GB GPU.
- For Llama-3-8B (32 layers, 4096 hidden, bf16) with B=16, max_seq=768: ~12 GB KV-cache. Tight on 24GB; may need micro-batching (see Section 10).

---

## 3. _sequence_log_prob

### Current code

```python
def _sequence_log_prob(self, input_ids, prompt_len):
    outputs = self.model(input_ids=input_ids, use_cache=False)
    log_probs = torch.log_softmax(outputs.logits, dim=-1)  # [1, L, V]
    response_ids = input_ids[:, prompt_len:]
    response_log_probs = log_probs[:, prompt_len - 1 : -1, :]
    token_lp = response_log_probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)
    return token_lp.sum(dim=-1)
```

**Problem:** Called once per sample. Each call launches a full forward pass for a single sequence, wasting GPU parallelism.

### Batched replacement

```python
def _batched_sequence_log_prob(
    self,
    full_ids: torch.Tensor,       # [B, L]  (left-padded prompt + response + right-pad)
    prompt_lengths: torch.Tensor,  # [B]  actual (unpadded) prompt lengths
    max_prompt_len: int,           # padded prompt length (same for all)
    response_mask: torch.Tensor,   # [B, R]  1 for real response tokens, 0 for pad
) -> torch.Tensor:
    """
    Batched sequence log-probability computation.

    Key insight: because prompts are left-padded to max_prompt_len,
    the response tokens always start at position max_prompt_len in the
    padded sequence. The logits that predict these tokens are at positions
    [max_prompt_len - 1 : -1].

    However, attention must be masked correctly so that real prompt tokens
    do not attend to left-pad tokens.
    """
    B, L = full_ids.shape

    # Build full attention mask: 1 for all non-left-pad positions
    # Left-pad region: positions [0, max_prompt_len - prompt_lengths[i]) are pad
    prompt_mask = torch.zeros(B, max_prompt_len, device=full_ids.device)
    for i in range(B):
        pl = prompt_lengths[i].item()
        prompt_mask[i, max_prompt_len - pl:] = 1.0

    # Response portion mask (already provided), concat to form full mask
    full_attention_mask = torch.cat([prompt_mask, response_mask.float()], dim=1)  # [B, L]

    # Forward pass — single call for entire batch
    outputs = self.model(
        input_ids=full_ids,
        attention_mask=full_attention_mask,
        use_cache=False,
    )
    logits = outputs.logits  # [B, L, V]

    # Log-softmax over vocabulary
    log_probs = torch.log_softmax(logits, dim=-1)  # [B, L, V]

    # Response token ids and their predicting logits
    response_ids = full_ids[:, max_prompt_len:]  # [B, R]
    R = response_ids.shape[1]

    if R == 0:
        return torch.zeros(B, device=full_ids.device)

    # Logits at positions [max_prompt_len-1 : L-1] predict tokens at
    # positions [max_prompt_len : L]
    predicting_log_probs = log_probs[:, max_prompt_len - 1 : max_prompt_len - 1 + R, :]  # [B, R, V]

    # Gather log-probs of actual tokens
    token_lp = predicting_log_probs.gather(
        2, response_ids.unsqueeze(-1)
    ).squeeze(-1)  # [B, R]

    # Mask out padding positions and sum
    token_lp = token_lp * response_mask.float()  # zero out pad positions
    return token_lp.sum(dim=-1)  # [B]
```

### Vectorized attention mask construction (no Python loop)

```python
# Replace the Python for-loop above with:
positions = torch.arange(max_prompt_len, device=full_ids.device).unsqueeze(0)  # [1, P]
pad_lengths = (max_prompt_len - prompt_lengths).unsqueeze(1)  # [B, 1]
prompt_mask = (positions >= pad_lengths).float()  # [B, P]
```

### Expected speedup

- **Forward pass:** 4-8x for batch_size=8-16. Matrix multiplications scale efficiently with batch dimension.
- **Kernel launch overhead:** Eliminated B-1 redundant CUDA kernel launches.

### Memory impact

- Activations: `B * L * V * 4 bytes` for logits. With V=151936 (Qwen2.5), B=16, L=768: ~7.1 GB in float32. **This is the memory bottleneck.**
- Mitigation: Use `torch.float16` / `torch.bfloat16` for inference (halves to ~3.5 GB), or micro-batch (see Section 10).

### Edge cases

- **Different prompt lengths:** Handled by left-padding; attention mask prevents pad tokens from corrupting representations. The predicting positions are uniform because left-padding aligns all prompts to the right edge.
- **Different response lengths:** `response_mask` zeros out log-probs at padding positions before summation.
- **All-padding response:** Sum is 0.0, matching the single-sample fallback.

---

## 4. _policy_log_probs (with gradients)

### Current code

```python
def _policy_log_probs(self, batch: RolloutBatch) -> torch.Tensor:
    log_probs: List[torch.Tensor] = []
    for rollout in batch.rollouts:
        full_ids = torch.tensor([rollout.full_ids], dtype=torch.long, device=self.device)
        lp = self._sequence_log_prob(full_ids, rollout.prompt_len)
        log_probs.append(lp.squeeze(0))
    return torch.stack(log_probs)  # [B]
```

**Problem:** B sequential forward passes **with gradients** — each stores a full activation graph.

### Batched replacement

```python
def _policy_log_probs(self, batch: RolloutBatch) -> torch.Tensor:
    """
    Batched policy log-probs WITH gradients for PPO surrogate loss.

    Reconstructs the padded batch from stored rollout full_ids, runs a
    single forward pass, and extracts per-sample sequence log-probs.
    """
    B = len(batch.rollouts)

    # ── Reconstruct padded tensors from stored rollout data ───────────
    full_ids_list = [
        torch.tensor(r.full_ids, dtype=torch.long, device=self.device)
        for r in batch.rollouts
    ]
    prompt_lengths = torch.tensor(
        [r.prompt_len for r in batch.rollouts],
        dtype=torch.long, device=self.device,
    )

    # Pad to uniform length (right-pad since these are complete sequences)
    max_len = max(ids.shape[0] for ids in full_ids_list)
    padded_ids = torch.full(
        (B, max_len), self.tokenizer.pad_token_id,
        dtype=torch.long, device=self.device,
    )
    attention_mask = torch.zeros(B, max_len, device=self.device)

    for i, ids in enumerate(full_ids_list):
        L_i = ids.shape[0]
        padded_ids[i, :L_i] = ids
        attention_mask[i, :L_i] = 1.0

    # ── Single forward pass with gradients ────────────────────────────
    outputs = self.model(
        input_ids=padded_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    log_probs_all = torch.log_softmax(outputs.logits, dim=-1)  # [B, L, V]

    # ── Extract per-sample response log-probs ─────────────────────────
    # Each sample has a different prompt_len, so we must index per-sample.
    seq_log_probs = []
    for i in range(B):
        pl = prompt_lengths[i].item()
        seq_len = int(attention_mask[i].sum().item())
        resp_len = seq_len - pl

        if resp_len <= 0:
            seq_log_probs.append(torch.zeros(1, device=self.device).squeeze(0))
            continue

        # Logits at [pl-1 : seq_len-1] predict tokens [pl : seq_len]
        resp_logits = log_probs_all[i, pl - 1 : seq_len - 1, :]  # [R, V]
        resp_token_ids = padded_ids[i, pl : seq_len]               # [R]
        token_lp = resp_logits.gather(
            1, resp_token_ids.unsqueeze(-1)
        ).squeeze(-1)  # [R]
        seq_log_probs.append(token_lp.sum())

    return torch.stack(seq_log_probs)  # [B]
```

### Fully vectorized version (no inner Python loop)

The per-sample extraction loop above is unavoidable when prompt lengths differ and we need to maintain the gradient graph (we cannot use `torch.where` trivially because the predicting-position offset differs per sample). However, the loop is over B (typically 8-16) scalar indexing operations on an already-computed tensor — its cost is negligible compared to the single forward pass.

If prompt lengths happen to be uniform (e.g., fixed-format prompts), the loop can be replaced:

```python
# Only valid when all prompt_lengths are identical:
pl = prompt_lengths[0].item()
resp_ids = padded_ids[:, pl:]                              # [B, R_max]
pred_lp  = log_probs_all[:, pl - 1 : -1, :]               # [B, R_max, V]
token_lp = pred_lp.gather(2, resp_ids.unsqueeze(-1)).squeeze(-1)  # [B, R_max]
resp_mask = attention_mask[:, pl:]                          # [B, R_max]
seq_log_probs = (token_lp * resp_mask).sum(dim=-1)         # [B]
```

### Expected speedup

- **Forward pass:** 4-8x. This is the most impactful batching because the forward pass runs with gradients, storing activations.
- **Per-sample loop:** Negligible overhead (B integer index operations on GPU tensor).

### Memory impact

- **This is the most memory-sensitive function.** With gradients, PyTorch stores intermediate activations for backprop.
- Activation memory per sample: ~`n_layers * seq_len * hidden_size * 4 bytes` (for float32). For Qwen2.5-0.5B: 24 * 768 * 896 * 4 = ~63 MB/sample. At B=16: ~1.0 GB of activations.
- For Llama-3-8B: 32 * 768 * 4096 * 4 = ~385 MB/sample. At B=16: ~6.2 GB. Likely requires gradient checkpointing or micro-batching.
- **Recommendation:** Add a `micro_batch_size` config parameter. Accumulate gradients across micro-batches:

```python
def _policy_log_probs_microbatched(
    self, batch: RolloutBatch, micro_batch_size: int = 4
) -> torch.Tensor:
    """Gradient-accumulation-friendly micro-batching."""
    all_lp = []
    rollouts = batch.rollouts
    for start in range(0, len(rollouts), micro_batch_size):
        micro = RolloutBatch(rollouts[start : start + micro_batch_size])
        lp = self._policy_log_probs(micro)
        all_lp.append(lp)
    return torch.cat(all_lp)  # [B]
```

---

## 5. _critic_forward

### Current code

```python
def _critic_forward(self, batch, rewards):
    if not self.critic.is_trainable():
        return None, torch.tensor(0.0, device=self.device)
    values = []
    for rollout in batch.rollouts:
        enc = self.tokenizer(rollout.prompt, return_tensors="pt", ...).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=enc["input_ids"], output_hidden_states=True, ...)
        last_hidden = outputs.hidden_states[-1][:, -1, :].detach()
        v = self.critic(last_hidden).squeeze(0)
        values.append(v)
    critic_values = torch.stack(values)
    critic_loss = F.mse_loss(critic_values, rewards)
    return critic_values, critic_loss
```

### Batched replacement

```python
def _critic_forward(
    self,
    batch: RolloutBatch,
    rewards: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """
    Batched critic forward: one tokenization, one LM forward pass,
    one critic forward pass.
    """
    if not self.critic.is_trainable():
        return None, torch.tensor(0.0, device=self.device)

    self.critic.train()
    prompts = [r.prompt for r in batch.rollouts]
    B = len(prompts)

    # ── Tokenize with right-padding (prompts only, no generation) ─────
    # Right-padding is correct here: we need the LAST non-pad token's
    # hidden state, and right-padding keeps real tokens left-aligned,
    # making extraction straightforward.
    enc = self.tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
        # Note: temporarily override padding_side for this call
    ).to(self.device)

    # If tokenizer is left-padded (for generation), handle accordingly:
    # We need per-sample last-real-token positions.
    attention_mask = enc["attention_mask"]  # [B, P]

    with torch.no_grad():
        outputs = self.model(
            input_ids=enc["input_ids"],
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )

    last_hidden_states = outputs.hidden_states[-1]  # [B, P, H]

    # ── Extract last non-padding token per sample ─────────────────────
    # With left-padding: last real token is always at position P-1
    # With right-padding: last real token is at sum(attention_mask)-1
    seq_lengths = attention_mask.sum(dim=1).long()  # [B]

    # Vectorized extraction of last-token hidden state
    if self.tokenizer.padding_side == "left":
        # All real tokens are right-aligned; last real token = last position
        last_token_hidden = last_hidden_states[:, -1, :]  # [B, H]
    else:
        # Right-padded: last real token at variable positions
        last_positions = (seq_lengths - 1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        last_positions = last_positions.expand(-1, -1, last_hidden_states.shape[-1])  # [B, 1, H]
        last_token_hidden = last_hidden_states.gather(1, last_positions).squeeze(1)  # [B, H]

    last_token_hidden = last_token_hidden.detach()  # detach from policy graph

    # ── Critic forward (with grad for critic parameters) ──────────────
    critic_values = self.critic(last_token_hidden)  # [B]
    critic_loss = nn.functional.mse_loss(critic_values, rewards)

    return critic_values, critic_loss
```

### Helper: padding-side context manager

Since we need left-padding for generation but may prefer right-padding for non-generative forward passes (to keep last-token extraction simple), use a context manager:

```python
from contextlib import contextmanager

@contextmanager
def padding_side(tokenizer, side: str):
    """Temporarily switch tokenizer padding side."""
    original = tokenizer.padding_side
    tokenizer.padding_side = side
    try:
        yield
    finally:
        tokenizer.padding_side = original
```

Usage:

```python
# In _critic_forward:
with padding_side(self.tokenizer, "right"):
    enc = self.tokenizer(prompts, return_tensors="pt", padding=True, ...)
```

### Expected speedup

- **LM forward pass:** 4-8x (same as _sequence_log_prob).
- **Critic forward pass:** Already vectorized (critic takes `[B, H]`), so no change there. The gain is entirely from batching the LM backbone.

### Memory impact

- No gradients through the LM (torch.no_grad), so no activation storage for the backbone.
- Hidden states: `B * P * H * n_layers * 4 bytes`. For Qwen2.5-0.5B with `output_hidden_states=True`: 25 layers * 16 * 512 * 896 * 4 = ~750 MB. Optimization: only request the last layer's hidden states if the API supports it, or index immediately and delete.

---

## 6. _eval_critic_on_prompts

### Current code

```python
def _eval_critic_on_prompts(self, prompts):
    values = []
    for prompt in prompts:
        enc = self.tokenizer(prompt, return_tensors="pt", ...).to(self.device)
        outputs = self.model(input_ids=enc["input_ids"], output_hidden_states=True, ...)
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        v = self.critic(last_hidden).item()
        values.append(v)
    return np.array(values)
```

### Batched replacement

```python
@torch.no_grad()
def _eval_critic_on_prompts(self, prompts: List[str]) -> np.ndarray:
    """Batched critic evaluation on prompts."""
    self.model.eval()
    self.critic.eval()

    if not self.critic.is_trainable():
        return np.zeros(len(prompts))

    B = len(prompts)

    # Tokenize all prompts at once (left-padding is fine here)
    enc = self.tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(self.device)

    attention_mask = enc["attention_mask"]  # [B, P]

    outputs = self.model(
        input_ids=enc["input_ids"],
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
    )

    last_hidden_states = outputs.hidden_states[-1]  # [B, P, H]

    # Extract last real token per sample
    if self.tokenizer.padding_side == "left":
        last_token_hidden = last_hidden_states[:, -1, :]
    else:
        seq_lengths = attention_mask.sum(dim=1).long()
        idx = (seq_lengths - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, last_hidden_states.shape[-1])
        last_token_hidden = last_hidden_states.gather(1, idx).squeeze(1)

    values = self.critic(last_token_hidden)  # [B]
    return values.cpu().numpy()
```

### Expected speedup

- Identical to `_critic_forward`: 4-8x.

### Memory impact

- Same as `_critic_forward` but always under `torch.no_grad()`, so no activation storage. Very lightweight.

---

## 7. evaluate

### Current code

```python
def evaluate(self, prompts, ground_truths, n_eval=50):
    self.model.eval()
    rewards = []
    for prompt, gt in zip(prompts[:n_eval], ground_truths[:n_eval]):
        enc = self.tokenizer(prompt, return_tensors="pt", ...).to(self.device)
        out = self.model.generate(**enc, do_sample=False, ...)
        completion = self.tokenizer.decode(out[0][prompt_len:], ...)
        rewards.append(self.reward_fn(completion, gt))
    return compute_accuracy(rewards)
```

### Batched replacement

```python
@torch.no_grad()
def evaluate(
    self,
    prompts: List[str],
    ground_truths: List[str],
    n_eval: int = 50,
    eval_batch_size: int = 8,
) -> float:
    """
    Batched greedy evaluation. Processes prompts in micro-batches
    to control memory usage during evaluation.
    """
    self.model.eval()
    prompts = prompts[:n_eval]
    ground_truths = ground_truths[:n_eval]
    all_rewards: List[float] = []

    for start in range(0, len(prompts), eval_batch_size):
        batch_prompts = prompts[start : start + eval_batch_size]
        batch_gts = ground_truths[start : start + eval_batch_size]

        enc = self.tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        max_prompt_len = enc["input_ids"].shape[1]

        out_ids = self.model.generate(
            **enc,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,  # greedy for deterministic eval
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode responses (everything after the padded prompt region)
        response_ids = out_ids[:, max_prompt_len:]
        completions = self.tokenizer.batch_decode(
            response_ids, skip_special_tokens=True
        )

        for comp, gt in zip(completions, batch_gts):
            all_rewards.append(self.reward_fn(comp, gt))

    return compute_accuracy(all_rewards)
```

### Expected speedup

- **Per micro-batch:** 8x for eval_batch_size=8.
- **Overall:** n_eval=50 goes from 50 sequential generate calls to 7 batched calls. ~7x speedup.

### Memory impact

- Minimal: no gradients, KV-cache freed after each micro-batch. eval_batch_size=8 is conservative enough for any GPU.

---

## 8. estimate_mc_advantages

### Current code

```python
for prompt, gt in zip(prompts, ground_truths):       # N prompts
    for _ in range(n_samples):                         # K samples each
        out = policy.generate(**enc, do_sample=True, ...)
        completion = tokenizer.decode(out[0][prompt_len:], ...)
        sample_rewards.append(reward_fn(completion, gt))
```

**Cost:** `N * K` sequential generate calls. For N=20, K=50: **1000 calls**.

### Batched replacement

```python
def estimate_mc_advantages(
    policy,
    tokenizer,
    prompts: List[str],
    ground_truths: List[str],
    reward_fn: Callable[[str, str], float],
    n_samples: int = 50,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    device: str = "cpu",
    mc_batch_size: int = 16,
) -> Dict[str, float]:
    """
    Batched Monte Carlo baseline estimation.

    Strategy: repeat each prompt n_samples times, then process all
    N*K prompts in batches of mc_batch_size through generate().
    """
    policy.eval()

    # Ensure left-padding for batched generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    N = len(prompts)

    # ── Build expanded lists: each prompt repeated n_samples times ─────
    expanded_prompts = []
    expanded_gts = []
    prompt_indices = []  # track which original prompt each sample belongs to
    for i, (prompt, gt) in enumerate(zip(prompts, ground_truths)):
        expanded_prompts.extend([prompt] * n_samples)
        expanded_gts.extend([gt] * n_samples)
        prompt_indices.extend([i] * n_samples)

    total = len(expanded_prompts)  # N * n_samples

    # ── Batched generation ────────────────────────────────────────────
    all_completions: List[str] = []

    with torch.no_grad():
        for start in range(0, total, mc_batch_size):
            batch_prompts = expanded_prompts[start : start + mc_batch_size]

            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(device)

            max_prompt_len = enc["input_ids"].shape[1]

            out_ids = policy.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
            )

            response_ids = out_ids[:, max_prompt_len:]
            completions = tokenizer.batch_decode(
                response_ids, skip_special_tokens=True
            )
            all_completions.extend(completions)

    # ── Compute rewards and aggregate per prompt ──────────────────────
    all_rewards = [
        reward_fn(comp, gt)
        for comp, gt in zip(all_completions, expanded_gts)
    ]

    # Group by original prompt index
    mc_baselines: Dict[str, float] = {}
    for i, prompt in enumerate(prompts):
        start_idx = i * n_samples
        end_idx = start_idx + n_samples
        mc_baselines[prompt] = float(np.mean(all_rewards[start_idx:end_idx]))

    # Restore original padding side
    tokenizer.padding_side = original_padding_side

    return mc_baselines
```

### Expected speedup

- **Sequential:** N*K generate calls. For N=20, K=50: 1000 calls.
- **Batched (mc_batch_size=16):** ceil(1000/16) = 63 batched calls, each ~16x faster than a single call.
- **Net:** ~16x / (63/1000) = **~250x** effective speedup. In wall-clock terms, going from ~2 hours to ~30 seconds for a typical MC evaluation run.

### Memory impact

- Controlled by `mc_batch_size`. Setting it to 16 keeps memory usage identical to batched `generate_rollouts`.
- Peak memory is independent of N*K since we micro-batch.

### Edge cases

- **Identical prompts in a batch:** No issue — `generate()` with `do_sample=True` produces different outputs per sample even for identical inputs (different random seeds per position in the batch).
- **Mixed prompt lengths in a micro-batch:** Left-padding + attention_mask handles this correctly. However, grouping similar-length prompts together reduces padding waste. Optional optimization:

```python
# Sort by prompt length before batching, then unsort results
lengths = [len(tokenizer.encode(p)) for p in expanded_prompts]
sorted_indices = sorted(range(total), key=lambda i: lengths[i])
# ... generate in sorted order, then reorder results back
```

---

## 9. Combined Forward Pass Optimization

### Opportunity

During `ppo_update`, the current code runs:
1. `_critic_forward`: LM forward with `output_hidden_states=True` on prompts (no grad through LM)
2. `_policy_log_probs`: LM forward on full sequences (with grad)

These are two separate forward passes through the same model. Can they be merged?

### Analysis

| Aspect | _policy_log_probs | _critic_forward |
|--------|-------------------|-----------------|
| Input | full_ids (prompt + response) | prompt only |
| Needs | logits over full sequence | hidden states at last prompt token |
| Gradients | Yes (policy grad) | No (detached from policy) |

### Answer: Partial merge is possible

A single forward pass on the **full sequence** with `output_hidden_states=True` provides both:
- Logits at all positions (for policy log-probs)
- Hidden states at the last-prompt-token position (for critic)

```python
def _combined_policy_and_critic_forward(
    self,
    batch: RolloutBatch,
    rewards: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Single forward pass that computes both policy log-probs (with grad)
    and critic values (detached from policy graph).

    Returns:
        (new_log_probs [B], critic_values [B] or None, critic_loss scalar)
    """
    B = len(batch.rollouts)

    # ── Reconstruct padded batch ──────────────────────────────────────
    full_ids_list = [
        torch.tensor(r.full_ids, dtype=torch.long, device=self.device)
        for r in batch.rollouts
    ]
    prompt_lengths = torch.tensor(
        [r.prompt_len for r in batch.rollouts],
        dtype=torch.long, device=self.device,
    )

    max_len = max(ids.shape[0] for ids in full_ids_list)
    padded_ids = torch.full(
        (B, max_len), self.tokenizer.pad_token_id,
        dtype=torch.long, device=self.device,
    )
    attention_mask = torch.zeros(B, max_len, device=self.device)

    for i, ids in enumerate(full_ids_list):
        L_i = ids.shape[0]
        padded_ids[i, :L_i] = ids
        attention_mask[i, :L_i] = 1.0

    # ── Single forward pass ───────────────────────────────────────────
    need_hidden = self.critic.is_trainable()
    outputs = self.model(
        input_ids=padded_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=need_hidden,
    )

    # ── Policy log-probs (WITH grad) ──────────────────────────────────
    log_probs_all = torch.log_softmax(outputs.logits, dim=-1)  # [B, L, V]

    seq_log_probs = []
    for i in range(B):
        pl = prompt_lengths[i].item()
        seq_len = int(attention_mask[i].sum().item())
        resp_len = seq_len - pl
        if resp_len <= 0:
            seq_log_probs.append(torch.zeros(1, device=self.device).squeeze(0))
            continue
        resp_logits = log_probs_all[i, pl - 1 : seq_len - 1, :]
        resp_token_ids = padded_ids[i, pl : seq_len]
        token_lp = resp_logits.gather(1, resp_token_ids.unsqueeze(-1)).squeeze(-1)
        seq_log_probs.append(token_lp.sum())

    new_log_probs = torch.stack(seq_log_probs)  # [B]

    # ── Critic values (DETACHED from policy graph) ────────────────────
    if not need_hidden:
        return new_log_probs, None, torch.tensor(0.0, device=self.device)

    last_hidden_states = outputs.hidden_states[-1]  # [B, L, H]

    # Extract hidden state at the last prompt token (position prompt_len - 1)
    # These are right-padded sequences, so prompt starts at position 0.
    last_prompt_positions = (prompt_lengths - 1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
    last_prompt_positions = last_prompt_positions.expand(
        -1, -1, last_hidden_states.shape[-1]
    )  # [B, 1, H]
    last_prompt_hidden = last_hidden_states.gather(
        1, last_prompt_positions
    ).squeeze(1)  # [B, H]

    # Detach: critic loss must NOT flow gradients into policy weights
    last_prompt_hidden = last_prompt_hidden.detach()

    self.critic.train()
    critic_values = self.critic(last_prompt_hidden)  # [B]
    critic_loss = nn.functional.mse_loss(critic_values, rewards)

    return new_log_probs, critic_values, critic_loss
```

### Updated ppo_update using the combined pass

```python
def ppo_update(
    self,
    batch: RolloutBatch,
    precomputed_advantages: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    self.model.train()

    rewards = batch.rewards().to(self.device)
    old_log_probs = batch.old_log_probs().to(self.device)

    # ── Combined forward pass ─────────────────────────────────────────
    new_log_probs, critic_values, critic_loss = \
        self._combined_policy_and_critic_forward(batch, rewards)

    # ── Advantages ────────────────────────────────────────────────────
    if precomputed_advantages is not None:
        advantages = precomputed_advantages.detach()
    else:
        values_for_adv = critic_values.detach() if critic_values is not None else None
        advantages = compute_advantages(
            rewards, values_for_adv, gamma=self.config.gamma, normalize=True,
        )

    # ── PPO surrogate ─────────────────────────────────────────────────
    log_ratio = new_log_probs - old_log_probs.detach()
    log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(
        ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon,
    )
    policy_loss = -torch.mean(torch.min(ratio * advantages, clipped * advantages))

    kl = (old_log_probs.detach() - new_log_probs).mean()

    total_loss = (policy_loss
                  + self.config.critic_loss_coeff * critic_loss
                  + self.config.kl_coeff * kl)

    self.policy_optimizer.zero_grad()
    if self.critic_optimizer:
        self.critic_optimizer.zero_grad()

    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    if self.critic.is_trainable():
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
    self.policy_optimizer.step()
    if self.critic_optimizer:
        self.critic_optimizer.step()

    ratio_detached = ratio.detach()
    return {
        "policy_loss": policy_loss.item(),
        "critic_loss": critic_loss.item(),
        "kl_divergence": kl.item(),
        "mean_reward": rewards.mean().item(),
        "reward_variance": rewards.var().item(),
        "mean_advantage": advantages.mean().item(),
        "clip_fraction": ((ratio_detached - 1.0).abs() > self.config.clip_epsilon)
                         .float().mean().item(),
    }
```

### Savings from combined pass

| Metric | Separate passes | Combined pass |
|--------|----------------|---------------|
| LM forward passes per ppo_update | 2 (policy + critic) | 1 |
| Memory peak | max(policy_activations, critic_hidden) | policy_activations + critic_hidden (slightly higher) |
| Wall time | 2 * forward_time | 1 * forward_time (+ small overhead for `output_hidden_states`) |

**Net effect:** ~1.8x speedup on `ppo_update` (the `output_hidden_states=True` overhead is ~10% per forward pass).

**Trade-off:** The combined pass stores hidden states for all layers alongside the gradient activation graph, increasing peak memory by ~15-20%. For large models, it may be preferable to keep them separate and pay the wall-time cost. Add a config flag:

```python
# In PPOConfig:
fuse_policy_critic_forward: bool = True  # merge policy + critic into one pass
```

---

## 10. Memory Budget Analysis

### Per-function memory breakdown (Qwen2.5-0.5B, B=16, max_seq=768, float32)

| Component | Sequential (current) | Batched | Notes |
|-----------|---------------------|---------|-------|
| Model weights | 2.0 GB | 2.0 GB | No change |
| KV-cache (generate) | 0.2 GB | 3.4 GB | B=16 vs B=1 |
| Logits tensor | 0.4 GB | 7.1 GB | B*L*V*4; largest tensor |
| Activations (with grad) | 0.06 GB | 1.0 GB | For _policy_log_probs |
| Hidden states (all layers) | 0.03 GB | 0.75 GB | For critic; output_hidden_states |
| **Peak total** | **~2.7 GB** | **~14.3 GB** | Fits on 24GB GPU |

### Per-function memory breakdown (Llama-3-8B, B=16, max_seq=768, bfloat16)

| Component | Sequential (current) | Batched | Notes |
|-----------|---------------------|---------|-------|
| Model weights | 16.0 GB | 16.0 GB | No change |
| KV-cache (generate) | 0.8 GB | 12.0 GB | Tight on 24GB |
| Logits tensor | 1.5 GB | 24.0 GB | **Exceeds 24GB GPU** |
| **Recommendation** | - | micro_batch_size=4 | 4 micro-batches of 4 |

### Recommended config additions

```python
@dataclass
class PPOConfig:
    # ... existing fields ...

    # ── Batching controls ────────────────────────────────────────────
    generation_batch_size: int = 16      # max batch for model.generate()
    forward_micro_batch_size: int = 8    # micro-batch for gradient forward passes
    eval_batch_size: int = 8             # micro-batch for evaluation
    mc_batch_size: int = 16              # micro-batch for MC estimation
    fuse_policy_critic_forward: bool = True
    use_fp16_inference: bool = True      # fp16/bf16 for no-grad passes
```

### Automatic micro-batch sizing

```python
def _estimate_max_batch_size(self, seq_len: int, with_grad: bool) -> int:
    """
    Estimate maximum batch size that fits in GPU memory.
    Conservative heuristic based on model size and available memory.
    """
    if not torch.cuda.is_available():
        return 4  # CPU fallback

    free_mem = torch.cuda.mem_get_info(self.device)[0]
    model_mem = sum(p.numel() * p.element_size() for p in self.model.parameters())
    available = free_mem - model_mem * 1.1  # 10% headroom

    vocab_size = self.model.config.vocab_size
    hidden_size = self.model.config.hidden_size
    n_layers = self.model.config.num_hidden_layers

    # Per-sample memory estimate
    logits_per_sample = seq_len * vocab_size * 4  # float32 logits
    if with_grad:
        activations_per_sample = n_layers * seq_len * hidden_size * 4
    else:
        activations_per_sample = 0

    per_sample = logits_per_sample + activations_per_sample
    max_batch = int(available / per_sample * 0.8)  # 80% safety margin
    return max(1, max_batch)
```

---

## 11. Migration Plan

### Phase 1: Non-breaking additions (low risk)

1. Set `tokenizer.padding_side = "left"` in `load_ppo_trainer()`.
2. Add `padding_side()` context manager utility.
3. Add batching config fields to `PPOConfig`.
4. Implement `_build_response_mask()` helper.
5. Batch `evaluate()` — pure inference, no gradient impact, easy to validate.
6. Batch `_eval_critic_on_prompts()` — pure inference.

**Validation:** Assert batched evaluate accuracy == sequential evaluate accuracy on 10 prompts.

### Phase 2: Generation batching (medium risk)

7. Batch `generate_rollouts()` with micro-batching support.
8. Batch `estimate_mc_advantages()`.

**Validation:** For 5 prompts, compare batched vs sequential: completions should differ (sampling), but reward distributions should be statistically similar (KS test, p > 0.05 over 100 runs).

### Phase 3: Training batching (higher risk — affects gradients)

9. Batch `_policy_log_probs()`.
10. Batch `_critic_forward()`.
11. Implement `_combined_policy_and_critic_forward()`.

**Validation:** 
- Numerical: `|batched_log_prob - sequential_log_prob| < 1e-4` for each sample (fp32) or `< 1e-2` (fp16).
- Gradient: `cosine_similarity(batched_grad, sequential_grad) > 0.999` over model parameters.
- Training: Identical learning curves (within noise) over 20 steps with fixed seed.

### Phase 4: vLLM integration (cluster-scale, optional)

12. Replace HuggingFace `model.generate()` with vLLM `LLM.generate()` for rollout generation.
13. Use vLLM's PagedAttention for memory-efficient batched generation.
14. Separate generation and training onto different GPU groups (as in [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) and [verl](https://github.com/verl-project/verl)).

This phase is architecturally separate: vLLM handles generation on dedicated GPUs, results are sent back as token IDs for training on other GPUs. The batched `_policy_log_probs` and `_critic_forward` from Phase 3 still apply on the training side.

---

## Summary of Expected Speedups

| Function | Current calls per step (B=16) | Batched calls | Speedup |
|----------|-------------------------------|---------------|---------|
| `generate_rollouts` generation | 16 sequential | 1 batched | **~10x** |
| `generate_rollouts` log-probs | 16 sequential fwd | 1 batched fwd | **~6x** |
| `generate_rollouts` critic | 16 sequential fwd | 1 batched fwd | **~6x** |
| `_policy_log_probs` (per epoch) | 16 sequential fwd | 1 batched fwd | **~6x** |
| `_critic_forward` (per epoch) | 16 sequential fwd | 1 batched fwd | **~6x** |
| Combined fwd (replaces above two) | 2 batched fwd | 1 batched fwd | **~1.8x** |
| `evaluate` (n_eval=50) | 50 sequential | 7 batched | **~7x** |
| `estimate_mc_advantages` (N=20, K=50) | 1000 sequential | 63 batched | **~16x** |

**End-to-end training step estimate:** From ~48 sequential GPU operations to ~3-5 batched operations. **~10-15x overall wall-time reduction.**

---

## References

- [HuggingFace Generation Strategies — Left Padding](https://huggingface.co/docs/transformers/generation_strategies)
- [HuggingFace LLM Tutorial — Batched Generation](https://huggingface.co/docs/transformers/llm_tutorial)
- [OpenRLHF — Scalable RLHF Framework](https://github.com/OpenRLHF/OpenRLHF)
- [verl — Volcano Engine RL for LLMs](https://github.com/verl-project/verl)
- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenRLHF Performance Tuning](https://openrlhf.readthedocs.io/en/latest/performance.html)
- [HuggingFace Batched Inference Discussion](https://github.com/huggingface/transformers/issues/18478)
