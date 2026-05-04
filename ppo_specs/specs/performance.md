# Performance Fix Spec

Severity levels:
- **CRITICAL** — causes 10x or greater slowdown; unacceptable for cluster runs
- **CLUSTER BLOCKER** — required to run at full scale; not needed for local tests
- **MODERATE** — 2-5x improvement available
- **MINOR** — small gain, low effort

Each issue: **Problem -> Impact -> Step-by-step fix**.

---

## Forward Pass Accounting (Per Training Step)

This section provides an exact count of model forward passes per `train_step` call.
All counts assume `batch_size=B`, `n_ppo_epochs=K`, and `max_new_tokens=T`.

### Phase 1: `generate_rollouts` (all under `@torch.no_grad()`)

Now **batched** (fixed P1). All B prompts are processed together:
1. `model.generate()` — single batched call with left-padding. Cost: 1 batched
   generate over B prompts (equivalent to ~1 full-sequence forward pass total,
   not B sequential calls).
2. `_batched_sequence_log_probs()` — 1 batched forward pass on all B sequences
   (prompt + response). Right-padded to uniform length. No KV cache.
3. `_batched_critic_values()` — 1 batched forward pass with `output_hidden_states=True`
   on all B prompts. Skipped for `capacity="none"`.

**Subtotal Phase 1:** 1 batched generate + 1 batched full_seq_fwd + 1 batched prompt_fwd
= 3 batched forward passes (down from B * 3 sequential passes).

### Phase 1.5: Advantage precomputation in `train_step`

`train_step` calls `_critic_forward(batch, rewards)` under `torch.no_grad()` to
compute initial critic values for advantage estimation. This is:
- B forward passes with `output_hidden_states=True` (prompt only, no grad).
- **Redundant with the critic forward in Phase 2** (see P11 below).

**Subtotal Phase 1.5:** B prompt-only forward passes.

### Phase 2: `ppo_update` (repeated K times)

Each epoch (now **batched**, partial P2 fix):
1. `_critic_forward()` — 1 batched forward pass with `output_hidden_states=True`
   under `torch.no_grad()` on the model, with grad on the critic head only.
   Prompt only, all B samples in one call.
2. `_policy_log_probs()` — 1 batched forward pass **with gradients** on the full
   sequence (prompt + response), all B samples padded to uniform length.
   These are the most expensive passes because they build the computation graph.

**Subtotal Phase 2:** K * 2 batched forward passes per epoch (down from K * B * 2).

### Grand Total (after batching)

```
Total batched model forward passes per train_step:
  = 1 batched generate       [Phase 1: rollout]
  + 1 batched full_seq_fwd   [Phase 1: log probs]
  + 1 batched prompt_fwd     [Phase 1: critic]
  + 1 batched prompt_fwd     [Phase 1.5: advantage precompute]
  + K * 2 batched fwd        [Phase 2: PPO update epochs]
  = 4 + 2K batched forward passes

For K=1: 6 batched forward passes per step
For K=4: 12 batched forward passes per step

Each batched pass processes all B samples in parallel on GPU, providing
~10-15x throughput improvement over the previous sequential approach.
```

### Cost Breakdown by Type (after batching)

| Pass type | Count | Grad? | hidden_states? | Sequence length |
|-----------|-------|-------|----------------|-----------------|
| `generate()` (batched) | 1 | No | No | ~T autoregressive steps |
| `_batched_sequence_log_probs` (rollout) | 1 | No | No | prompt+response |
| `_batched_critic_values` | 1 | No | Yes | prompt only |
| `_critic_forward` (batched, precompute) | 1 | No (full) | Yes | prompt only |
| `_critic_forward` (batched, per epoch) | K | Critic only | Yes | prompt only |
| `_policy_log_probs` (batched, per epoch) | K | Full policy | No | prompt+response |

### Memory During PPO Update

During each `ppo_update` epoch, the following are in memory simultaneously:
- Model weights (~2 GB for 0.5B in float32, ~16 GB for 8B in bfloat16)
- Activations from `_policy_log_probs` (with grad, the largest cost)
- Critic head activations (small)
- Optimizer states (2x model weights for AdamW)
- Rollout buffer (token IDs, rewards, log probs)

Total for 0.5B float32: ~8 GB (manageable on any GPU)
Total for 8B bfloat16: ~48-64 GB (requires A100 80GB or gradient checkpointing)

---

## P1 — Per-sample rollout generation instead of batched [CRITICAL]

**Status**: **Fixed** (2026-04-08)
**Severity**: Critical

**Location:** `ppo_specs/ppo_trainer.py` (`generate_rollouts`)

**Problem:**
```python
for prompt, gt in zip(prompts, ground_truths):
    enc = self.tokenizer(prompt, ...)      # one sample
    out = self.model.generate(**enc, ...)  # one generate() call
```
For `batch_size=16` this issues 16 sequential `model.generate()` calls where a
single batched call would suffice.

**Impact:** ~10-15x throughput reduction. On an A100 with Llama-3-8B, a single
`generate()` call on 16 prompts takes ~2 s; 16 sequential calls take ~20 s.
For 200 training steps this is ~65 min vs ~6.5 min.

**Fix:**

Replace the per-sample loop with a batched call using left-padding:

```python
self.tokenizer.padding_side = "left"
enc = self.tokenizer(
    prompts,
    return_tensors="pt",
    truncation=True,
    max_length=self.config.tokenize_max_length,
    padding=True,
).to(self.device)

with torch.no_grad():
    out = self.model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_new_tokens=self.config.max_new_tokens,
        do_sample=self.config.do_sample,
        temperature=self.config.temperature,
        pad_token_id=self.tokenizer.pad_token_id,
    )
```

> **Note:** Left-padding is required for decoder-only LLMs so that attention masks
> are aligned. `tokenizer.padding_side = "left"` must be set before the batched call.

**Resolution:** `generate_rollouts` now uses a single batched `model.generate()` call
with left-padding. `_batched_sequence_log_probs` and `_batched_critic_values` replace
the per-sample loops. Legacy single-sample methods are kept for backward compatibility.

---

## P2 — Double forward pass per sample during PPO update [CRITICAL]

**Status**: **Partially Fixed** (2026-04-08) -- `_policy_log_probs` and `_critic_forward` are now batched but still separate passes
**Severity**: Moderate (downgraded from Critical)

**Location:** `ppo_specs/ppo_trainer.py` — `_critic_forward` and `_policy_log_probs`
both run separate forward passes on the same model.

**Problem:**
```python
# Call 1 (_critic_forward): extract hidden states for critic
outputs = self.model(input_ids=prompt_ids, output_hidden_states=True)

# Call 2 (_policy_log_probs): compute log-probs from logits
outputs = self.model(input_ids=full_ids, use_cache=False)
```
Note: these operate on DIFFERENT input lengths (prompt-only vs full sequence),
so they cannot be trivially merged. However, the policy forward pass on the full
sequence ALSO computes hidden states (if `output_hidden_states=True` is added).
A single forward pass on the full sequence can provide both logits (for log-probs)
and the last-token hidden state of the prompt portion (for the critic).

**Impact:** 2x the forward-pass cost during every PPO update step. For K=1, B=16:
saves 16 prompt-only forward passes per step.

**Fix:**

Create a combined `_ppo_forward` method that runs ONE forward pass per sample on
the full sequence with `output_hidden_states=True`, extracting both:
- logits for sequence log-prob computation
- hidden states at the prompt's last token for critic evaluation

```python
def _ppo_forward(self, batch: RolloutBatch) -> tuple:
    critic_values, log_probs_list = [], []
    for rollout in batch.rollouts:
        full_ids = torch.tensor([rollout.full_ids], ..., device=self.device)
        outputs = self.model(
            input_ids=full_ids, use_cache=False, output_hidden_states=True
        )
        # Log-probs from logits
        lp = self._log_prob_from_outputs(outputs, full_ids, rollout.prompt_len)
        log_probs_list.append(lp.squeeze(0))
        # Critic from hidden states (detached from policy grad graph)
        if self.critic.is_trainable():
            last_h = outputs.hidden_states[-1][:, rollout.prompt_len - 1, :].detach()
            critic_values.append(self.critic(last_h).squeeze())
    return (torch.stack(critic_values) if critic_values else None,
            torch.stack(log_probs_list))
```

---

## P3 — Model re-loaded from disk per critic capacity in E2.8 [MODERATE]

**Status**: Open
**Severity**: Moderate

**Location:** `ppo_specs/run_e2_8.py` (`run_one_capacity` calls `load_ppo_trainer`)

**Problem:** Each of the four capacity runs calls `AutoModelForCausalLM.from_pretrained(...)`,
which loads the model from disk (or HuggingFace cache).

**Impact:** ~30-60 s per load. For 4 capacities x 3 seeds = 12 extra loads = 6-12 min.

**Fix:** Load the base model once; deep-copy its `state_dict`; reset before each run.

---

## P4 — Sequential MC rollouts per prompt [MODERATE]

**Status**: **Fixed** (2026-04-08)
**Severity**: Moderate

**Location:** `ppo_specs/advantage.py` (`estimate_mc_advantages`)

**Problem:**
```python
for _ in range(n_samples):      # 50 or 1000 iterations
    out = policy.generate(**enc, ...)   # one sample at a time
```

**Impact:** For `n_samples=1000` and 5 prompts = 5,000 sequential `generate()` calls.
At ~0.5 s/call = ~42 min. Batching brings this to ~2-5 min.

**Fix:** Batch identical copies of the same prompt using `repeat()`.

**Resolution:** `estimate_mc_advantages` now processes micro-batches of `batch_size`
samples per prompt using `enc["input_ids"].repeat(n_batch, 1)` and
`enc["attention_mask"].repeat(n_batch, 1)`, with left-padding for batched generation.

---

## P5 — Redundant tokenisation in `_critic_forward` [MINOR]

**Status**: **Fixed** (2026-04-08) -- moot; `_critic_forward` is now batched
**Severity**: Minor

**Location:** `ppo_specs/ppo_trainer.py` — `_critic_forward` previously re-tokenised
prompts that were already stored as token IDs in `rollout.full_ids[:rollout.prompt_len]`.

**Resolution:** `_critic_forward` is now a single batched tokenization + forward pass
over all prompts in the batch. The per-sample tokenization loop no longer exists.

---

## P6 — No gradient checkpointing [CLUSTER BLOCKER]

**Status**: **Fixed** (2026-04-08)
**Severity**: Cluster Blocker

**Location:** `ppo_specs/ppo_trainer.py` (`load_ppo_trainer`)

**Problem:** Llama-3-8B requires ~40 GB for weights in bfloat16. Without gradient
checkpointing, activation memory during backward pushes VRAM to 80-120 GB.

**Fix:**
```python
if device.type == "cuda":
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
```

**Resolution:** `load_ppo_trainer` now enables gradient checkpointing when
`config.gradient_checkpointing` is True. The `gradient_checkpointing` field
was added to `PPOConfig` (default False).

---

## P7 — No distributed training / multi-GPU support [CLUSTER BLOCKER]

**Status**: Open
**Severity**: Cluster Blocker

**Location:** Entire codebase — no `accelerate`, no DDP.

**Fix:** Integrate HuggingFace `accelerate` with FSDP for 8B+ models.

---

## P8 — `torch.float32` on GPU: 2x slower than bfloat16 [MODERATE]

**Status**: **Fixed** (2026-04-08)
**Severity**: Moderate

**Location:** `ppo_specs/ppo_trainer.py` (`load_ppo_trainer`)

**Fix:**
```python
torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
```

**Resolution:** `load_ppo_trainer` now auto-selects dtype via `config.torch_dtype`
(`"auto"` = bfloat16 on GPU, float32 on CPU). The `log_softmax` computation is
always upcast to float32 for numerical stability, even when the model is in bfloat16.

---

## P9 — Wasted log probs from generation [MODERATE]

**Status**: Open
**Severity**: Moderate

**Location:** `ppo_specs/ppo_trainer.py` — `generate_rollouts`

**Problem:** `model.generate()` computes logits at every autoregressive step but
discards them. Then `_sequence_log_prob()` runs a SECOND full forward pass on the
entire sequence just to recompute those same logits. This wastes B full-sequence
forward passes per training step.

**Impact:** For B=16, this eliminates 16 full-sequence forward passes per step.
Each takes ~0.5 s on A100 with an 8B model = ~8 s savings per step, ~27 min over
200 steps.

**Fix:** Use `model.generate(..., output_scores=True, return_dict_in_generate=True)`
to capture logits during generation:

```python
gen_out = self.model.generate(
    **enc,
    max_new_tokens=self.config.max_new_tokens,
    do_sample=self.config.do_sample,
    temperature=self.config.temperature,
    pad_token_id=self.tokenizer.pad_token_id,
    output_scores=True,
    return_dict_in_generate=True,
)
# gen_out.scores is a tuple of (vocab_size,) tensors, one per generated token
# Compute log-probs directly from scores instead of re-running the model
```

Note: `output_scores=True` returns pre-softmax logits per generation step, which
can be converted to log-probs without a separate forward pass.

---

## P10 — Redundant advantage precomputation forward pass [MINOR]

**Status**: Open
**Severity**: Minor

**Location:** `ppo_specs/ppo_trainer.py` — `train_step` lines 423-424

**Problem:** `train_step` calls `self._critic_forward(batch, rewards)` under
`torch.no_grad()` to precompute advantages before the PPO loop. Then the first
iteration of `ppo_update` calls `_critic_forward` again (with grad on critic)
for the critic loss. The model forward passes in both calls are identical (both
under `torch.no_grad()` for the model; only the critic head differs in grad
tracking).

**Impact:** B extra prompt-only forward passes per training step. For B=16 with
an 8B model, this wastes ~4 s per step.

**Fix:** Extract critic values from the first `ppo_update` epoch's
`_critic_forward` call and use them for advantage computation. Alternatively,
after implementing P2, the combined forward pass handles this naturally.

---

## P11 — Evaluation overhead [MINOR]

**Status**: Open
**Severity**: Minor

**Location:** `ppo_specs/ppo_trainer.py` — `evaluate()`, called every `eval_every`
steps.

**Problem:** Each evaluation call runs `n_eval` sequential `model.generate()` calls
(20 during training, 50 for final). These are additional forward passes beyond
the training step count.

**Impact:** For `eval_every=20`, `n_steps=200`, `n_eval=20`:
- 10 eval calls x 20 generates = 200 extra generates during training
- Plus 1 final eval: 50 generates
- Total: 250 extra generate calls = ~2-4 min on A100

This is acceptable overhead (< 5% of total) but could be reduced by batching
eval generation.

---

## Deep Performance Review — Additional Items (added 2026-04-30)

These items (P12–P18) were identified by a quantitative deep review.
All wall-clock numbers are for A100 80GB single-GPU at Llama-3-8B bf16
B=16 S=600 unless noted. Wins compound; do them in this order.

## P12 — Epoch-0 redundant policy forward [HIGH]

**Status:** Open
**Severity:** High (1.4 s/step at 8B = 4.7 min over 200 steps; compounds
with DDP at 4× ranks for ~19 min savings)

**Location:** [ppo_trainer.py:402-405](../ppo_trainer.py#L402-L405)

**Problem:** The first PPO epoch's `new_per_token` is bitwise-identical
to `fixed_old_per_token` because no optimizer step has occurred yet.
The policy forward pass at `_batched_per_token_log_probs(model=self.model, ...)`
(with grad) is run anyway, computing values that are already cached.

**Fix:** in `ppo_update`, accept an `is_first_epoch: bool` flag. On
epoch 0, reuse `precomputed_old_per_token_log_probs` as `new_per_token`
while still building a fresh autograd graph through a sentinel forward
that's smaller, OR skip epoch 0 entirely:

```python
# In train_step, restructure the K-loop:
for epoch in range(self.config.n_ppo_epochs):
    metrics = self.ppo_update(
        batch,
        precomputed_advantages=fixed_advantages,
        is_first_epoch=(epoch == 0),
    )
```

In `ppo_update`, when `is_first_epoch=True`, skip the policy forward
and use the precomputed `old_log_probs` to construct `ratio = 1.0`
identically (since ratio = exp(new - old) = exp(0) = 1.0 on epoch 0).
The `policy_loss` reduces to `-mean(advantages)` which has zero gradient
through the policy (no log-prob graph). Net: epoch 0 only updates the
critic, not the policy. This matches the standard "no PPO ratio on epoch 0"
behavior in TRL.

**Savings:** 1 of K policy passes; for K=4 at 8B = ~1.4 s/step × 200 = 4.7 min
single-GPU; ~19 min on 4× A100 DDP if combined with §1.3.1 update sharding.

## P13 — Repeated tokenization in `_extract_last_hidden` [MODERATE]

**Status:** Open
**Severity:** Moderate (~30 ms/step; on the critical path with no GPU overlap)

**Location:** [ppo_trainer.py:298-301](../ppo_trainer.py#L298-L301)

**Problem:** The same B prompts are tokenized **6 times per train_step**
(rollout init + advantage precompute + K=4 epochs of `_critic_forward`).
The token IDs are already cached in `Rollout.full_ids[:prompt_len]`.

**Fix:** change `_extract_last_hidden(self, prompts: List[str])` to accept
`(prompt_ids: List[List[int]], prompt_lens: List[int])` and pad from the
cached IDs. Eliminates 5 redundant tokenization calls per step.

```python
def _extract_last_hidden_from_ids(self, prompt_ids, prompt_lens):
    max_len = max(prompt_lens)
    padded = torch.full(
        (len(prompt_ids), max_len), self.tokenizer.pad_token_id,
        dtype=torch.long, device=self.device,
    )
    attention_mask = torch.zeros_like(padded)
    for i, (ids, pl) in enumerate(zip(prompt_ids, prompt_lens)):
        # Left-pad
        padded[i, max_len - pl:] = torch.tensor(ids[:pl])
        attention_mask[i, max_len - pl:] = 1
    # ... rest as before
```

**Savings:** ~30 ms/step at B=16 (6 ms/step at 0.5B). Removes a divergent
code path between rollout-time and PPO-time prompt encoding.

## P14 — Fused log_softmax+gather over response slice [MODERATE]

**Status:** Open
**Severity:** Moderate (~150 ms/step at 8B)

**Location:** [shared/per_token_loss.py:75-90](../../shared/per_token_loss.py#L75-L90)

**Problem:** Per-sample Python loop with separate `log_softmax` + `gather`
+ `.float()` upcast. Each iteration is HBM-bound and forces a serialized
kernel launch.

**Fix:** use `torch.nn.functional.cross_entropy(reduction='none')` to fuse
log_softmax + gather into a single kernel. Saves ~3 GB transient (under
K=4 PPO epochs, fp32 [R,V] per sample is allocated K times under grad)
AND ~144 ms/step in kernel launch + memory bandwidth.

```python
# Replace the per-sample loop body:
# OLD:
#   slice_logits = outputs.logits[i, pl-1 : seq_len-1, :].float()
#   lp_full = F.log_softmax(slice_logits, dim=-1)
#   token_lp = lp_full.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
# NEW:
slice_logits = outputs.logits[i, pl-1 : seq_len-1, :]
token_lp = -F.cross_entropy(
    slice_logits.float(), target_ids, reduction="none",
)
```

Add a parity test in `tests/test_batched_ops.py` confirming the new
implementation matches the old one within `rtol=1e-5`.

## P15 — `output_hidden_states=True` returns full layer stack [MODERATE]

**Status:** Open
**Severity:** Moderate (~3 GB peak transient at 8B; can unlock B=24 from B=16)

**Location:** [ppo_trainer.py:303-308](../ppo_trainer.py#L303-L308)

**Problem:** Requesting all `n_layers+1` hidden states when only `[-1]`
is used. Allocates 33 × 96 MB = 3.17 GB transient at 8B (bf16).

**Fix:** use a forward hook on the last decoder layer to capture only
the final hidden state. See §11.6 of `memory_optimization.md` for the
implementation pattern.

**Caveat:** with `gradient_checkpointing=True`, fusing the policy+critic
forward (PERF P2) becomes problematic because hidden_states retention
through backward defeats GC. P15's hook-based approach AVOIDS this
because no `output_hidden_states=True` flag is passed — only the final
layer is captured. Combine P15 with P2 to keep activation memory bounded.

## P16 — Grad-norm clipping forces full-gradient sync [MODERATE]

**Status:** Open
**Severity:** Moderate at 8B + DDP (~430 ms/step blocking on 4× A100)

**Location:** [ppo_trainer.py:466-469](../ppo_trainer.py#L466-L469)
(via `accelerator.clip_grad_norm_` per DDP migration spec §1.3)

**Problem:** Global gradient-norm clipping requires
`sqrt(sum_p ||g_p||^2)` across ranks. Per K=4 epochs at 4× A100 8B, that's
~430 ms blocking per train_step.

**Mitigations (pick one):**

1. **Per-bucket clipping (DeepSpeed style):** clip each DDP bucket's
   gradient norm locally before all-reduce. Equivalent to `clip_grad_value_`
   semantically, not equivalent to global L2-norm clipping but standard
   in DeepSpeed/FSDP.
2. **Skip clipping every other step:** halve the sync overhead at the cost
   of higher gradient variance early in training. Safe with conservative
   `lr=1e-5`.
3. **Larger DDP bucket size** (`bucket_cap_mb=200` per the new spec note)
   reduces bucket count from 656 to ~82 for 8B, improving overlap.

For now, leave global clipping as default; document the cost.

## P17 — Re-tensorization across PPO epochs [LOW]

**Status:** Open
**Severity:** Low (~3 ms/step, but on the critical path)

**Location:** [shared/per_token_loss.py:46-55](../../shared/per_token_loss.py#L46-L55)

**Problem:** The `padded` tensor is rebuilt K times per `train_step` from
identical `full_ids` lists. Each rebuild is a host-to-device copy at
~50 µs/sample × 16 = 0.8 ms × 4-7 calls = 3-6 ms/step.

**Fix:** pad and pin once after `generate_rollouts`, pass through
`ppo_update` as a tensor:

```python
# In train_step, after generate_rollouts:
padded_ids, attention_mask, prompt_lens = _build_padded(
    [r.full_ids for r in batch.rollouts],
    [r.prompt_len for r in batch.rollouts],
    pad_token_id=self.tokenizer.pad_token_id,
    device=self.device,
)
# Pass to ppo_update as kwargs
```

**Savings:** 75% of tensor-build cost across K=4 = ~4 ms/step.

## P18 — `.item()` loops in `generate_rollouts` [LOW]

**Status:** Open
**Severity:** Low (~1.5 ms/step, but B-serialized syncs)

**Location:** [ppo_trainer.py:201, 236, 242](../ppo_trainer.py#L201)

**Problem:** Three Python loops with `.item()` per iteration force B
serialized CUDA stream syncs.

**Fix:**
- Line 201: `pad_lens = (enc["input_ids"] == pad_id).sum(dim=1).tolist()`
  (one sync, returns list directly)
- Line 224: defer `tolist()` — keep `out` on GPU; only convert to a Python
  list at Rollout construction, OR change `Rollout.full_ids` to a `torch.Tensor`
- Lines 236, 242: assign vector-then-loop instead of loop-then-item:
  ```python
  old_log_probs_cpu = old_log_probs.cpu().tolist()
  critic_values_cpu = critic_values.cpu().tolist()
  for i, r in enumerate(rollouts):
      r.old_log_prob = old_log_probs_cpu[i]
      r.value = critic_values_cpu[i]
  ```

**Savings:** B-1 syncs = ~1.5 ms/step at B=16; freed up CPU↔GPU pipeline
for other overlap.

## P19 — Length-bucketed generation [HIGH]

**Status:** Open
**Severity:** High (~3.2 s/step at 8B; ~64% of generated tokens past
the average length are wasted padding)

**Location:** [ppo_trainer.py:189-196](../ppo_trainer.py#L189-L196)

**Problem:** `model.generate()` continues until EVERY sample reaches
`max_new_tokens` or all hit EOS. With T_max=384 and average GSM8K
response ~140 tokens, ~64% of generated tokens past the 140-mark are
masked out (wasted).

**Fix:** sort prompts by token length, bucket into groups of ~4, call
`generate()` per bucket. Variance within a bucket drops by ~3×, padding
waste drops to ~25%. Net: ~40% of generate wall-time saved.

```python
# In generate_rollouts, before the generate call:
prompt_lens_t = enc["attention_mask"].sum(dim=1)
sort_idx = torch.argsort(prompt_lens_t)
inverse_sort = torch.argsort(sort_idx)  # for un-sorting later

bucket_size = 4
sorted_results = []
for start in range(0, B, bucket_size):
    bucket_idx = sort_idx[start : start + bucket_size]
    out_bucket = self.model.generate(
        input_ids=enc["input_ids"][bucket_idx],
        attention_mask=enc["attention_mask"][bucket_idx],
        ...
    )
    sorted_results.append(out_bucket)

# Reassemble in original order
out = torch.cat(sorted_results)[inverse_sort]
```

**Savings:** ~3.2 s/step at 8B × 200 = ~10 min per E2.7 run. Adds ~10
lines but is well-isolated.

**Caveat:** ordering must be undone before constructing `RolloutBatch`
to keep `cycle_batch` deterministic alignment with ground truths. The
`inverse_sort` index above handles this.

## P20 — DDP bucket_cap_mb tuning [LOW]

**Status:** Open
**Severity:** Low (~5-10 ms/step at 8B 4× A100)

**Location:** Accelerate's `Accelerator()` constructor (`ddp_cpu_gpu_migration.md` §1)

**Problem:** Default `bucket_cap_mb=25` results in ~656 DDP buckets for
an 8B bf16 model, each requiring a separate NCCL all-reduce. Bucket
count dominates fixed overhead.

**Fix:**
```python
from accelerate.utils import DistributedDataParallelKwargs
accelerator = Accelerator(kwargs_handlers=[
    DistributedDataParallelKwargs(
        bucket_cap_mb=200,           # ~82 buckets for 8B vs 656
        find_unused_parameters=False, # critical: ~5-10 ms/step waste otherwise
    ),
])
```

`find_unused_parameters=False` is correctness-critical: without it, DDP
traverses the autograd graph each step looking for params with no grad,
costing ~5-10 ms/step. The default in Accelerate may be True for safety.

## Wall-clock per-step budget table (deep-review numbers)

For a single train_step with K=4 PPO epochs, B=16, T=384, S≈600.
Numbers in seconds.

| Phase | Qwen-0.5B / 1× A100 | Qwen-0.5B / 4× DDP | Llama-3-8B / 1× A100 | Llama-3-8B / 4× DDP |
|-------|--------------------:|-------------------:|---------------------:|--------------------:|
| Generate | 0.90 | 0.27 | 8.0 | 2.2 |
| Old-log-probs (rollout) | 0.035 | 0.035 | 0.35 | 0.35 |
| Critic init forward | 0.012 | 0.012 | 0.11 | 0.11 |
| Adv precompute critic | 0.012 | 0.012 | 0.11 | 0.11 |
| Adv precompute frozen-old | 0.035 | 0.035 | 0.35 | 0.35 |
| Reference frozen-log-probs | 0.035 | 0.035 | 0.35 | 0.35 |
| K=4 × policy fwd (with grad) | 0.24 | 0.24 | 5.6 | 5.6 |
| K=4 × critic fwd | 0.048 | 0.048 | 0.44 | 0.44 |
| K=4 × backward + optim step | 0.14 | 0.54 (+NCCL) | 0.80 | 2.5 (+NCCL) |
| `all_gather_object` | 0 | 0.003 | 0 | 0.005 |
| CPU↔GPU sync overhead | 0.004 | 0.004 | 0.004 | 0.004 |
| **TOTAL per train_step** | **~1.45** | **~1.20** | **~16.1** | **~12.0** |

DDP at 8B gives only 1.3× speedup because the gathered-batch architecture
forces every rank to redo log-probs/critic/policy update on the full batch
— pure replication, not parallelization. The Optional update sharding
(§1.3.1 of ddp_cpu_gpu_migration.md) captures the missing 2-3× speedup.

## Top 5 perf gaps ranked by ROI for cluster runs

| # | Gap | Savings (8B, 4× A100) | Effort | Priority |
|---|-----|----------------------:|--------|----------|
| 1 | §1.3.1: Shard PPO update | ~16.8 s/step → ~5.6 s/step | Medium | **Critical for 4+ GPU runs** |
| 2 | P12: Skip epoch-0 redundancy | 1.4 s/step | Low | High |
| 3 | P9: Reuse `output_scores` from generate | 0.35 s/step | Low-Medium | High |
| 4 | RM Performance Gap (reward_model_integration.md): batch SelfJudgeRewardModel | 1.5 s/step (in self_judge mode) | Low | High when self_judge enabled |
| 5 | P19: Length-bucketed generation | 3.2 s/step | Medium | High |

Compounded: at 8B 4× A100 with all 5 applied, ~12.0 s/step → ~5.0 s/step
(2.4× speedup beyond the existing DDP). Over 200 steps: ~24 min saved per
E2.7 run.

---

## Implementation Priority

When adapting for the cluster, apply fixes in this order:

| Priority | Issue | Savings | Status |
|----------|-------|---------|--------|
| 1 | P1 — Batched rollout generation | 10-15x throughput gain | **Fixed** |
| 2 | P6 — Gradient checkpointing | Required to avoid OOM with 8B model | **Fixed** |
| 3 | P8 — bfloat16 | 2x speed + memory | **Fixed** |
| 4 | P9 — Reuse generation logits | Saves B full-sequence forward passes/step | Open |
| 5 | P2 — Combined forward pass | Saves K*B prompt-only forward passes/step | Partial (batched, still separate) |
| 6 | P7 — Multi-GPU (accelerate) | Further parallelism | Open |
| 7 | P4 — Batched MC rollouts | Speeds up baseline estimation | **Fixed** |
| 8 | P5 — Redundant tokenisation in critic | Moot after batching | **Fixed** |
| 9 | P3 — Shared model state_dict | Saves re-load time in E2.8 | Open |
| 10 | P10, P11 — Minor | Polish; minimal impact | Open |

---

## Reference Model / KL Analysis

### Current Implementation
The code computes `KL(pi_old || pi_new)` as a regularizer within each PPO step
(line 312), controlled by `kl_coeff` (default 0.0). This is the standard PPO KL
penalty between the collecting policy and the updated policy.

### Missing: RLHF-style Reference Model KL
Standard RLHF (Ouyang et al., 2022) uses a **frozen reference model** pi_ref
(the initial pretrained model) and adds `KL(pi_theta || pi_ref)` to the reward:
```
r_modified = r_verifier - beta * KL(pi_theta || pi_ref)
```
This prevents the policy from drifting too far from the pretrained model.

### Analysis for RLVR
For RLVR (reward from a verifier, not a learned reward model), the reference
model KL is **less critical** than in RLHF because:

1. **No reward hacking**: The verifier provides ground-truth reward. In RLHF,
   the reward model can be exploited; KL constrains exploitation. Here, binary
   correctness cannot be hacked.

2. **Mode collapse risk is lower**: The model only needs to produce correct
   numerical answers, not satisfy a noisy reward model. The diversity pressure
   from sampling (temperature=0.7) provides some regularisation.

3. **Memory cost**: A reference model doubles GPU memory (~32 GB for 8B in
   bfloat16). This is a significant cost for a minor benefit.

However, without reference KL, the policy can still degenerate:
- **Formatting collapse**: The model might find a shortcut format that gets
  reward=1 on training data but fails on test data.
- **Catastrophic forgetting**: Extended training could destroy the model's
  language capabilities.

### Recommendation
For the research paper's scope (100-200 steps, small models), omitting the
reference model is acceptable. Document this as a known limitation. If training
runs exceed ~500 steps or show accuracy degradation, add reference KL.

The PPO-style `KL(pi_old || pi_new)` penalty (when `kl_coeff > 0`) provides
step-to-step stability, which is sufficient for short training runs.
