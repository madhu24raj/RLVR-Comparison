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

For each of B prompts:
1. `model.generate()` — autoregressive generation with KV cache. This is
   ~T forward passes on 1 token each (with cached KV). Cost is roughly
   equivalent to 1 full-sequence forward pass per prompt.
2. `_sequence_log_prob()` — 1 full forward pass on the entire sequence
   (prompt + response, ~512+T tokens). **No KV cache** (`use_cache=False`).
3. `_critic_value_no_grad()` — 1 forward pass with `output_hidden_states=True`
   on the prompt only (~512 tokens). Skipped for `capacity="none"`.

**Subtotal Phase 1:** B * (1 generate + 1 full_seq_fwd + 1 prompt_fwd)
= B * (generate + 2) full-equivalent forward passes.

### Phase 1.5: Advantage precomputation in `train_step`

`train_step` calls `_critic_forward(batch, rewards)` under `torch.no_grad()` to
compute initial critic values for advantage estimation. This is:
- B forward passes with `output_hidden_states=True` (prompt only, no grad).
- **Redundant with the critic forward in Phase 2** (see P11 below).

**Subtotal Phase 1.5:** B prompt-only forward passes.

### Phase 2: `ppo_update` (repeated K times)

Each epoch:
1. `_critic_forward()` — B forward passes with `output_hidden_states=True`
   under `torch.no_grad()` on the model, but with grad on the critic head.
   Prompt only (~512 tokens each).
2. `_policy_log_probs()` — B forward passes **with gradients** on the full
   sequence (prompt + response). These are the most expensive passes because
   they build the computation graph for backpropagation.

**Subtotal Phase 2:** K * B * 2 forward passes per epoch.

### Grand Total

```
Total model forward passes per train_step:
  = B*(generate + 2)        [Phase 1: rollout]
  + B                       [Phase 1.5: advantage precompute]
  + K * B * 2               [Phase 2: PPO update epochs]
  = B*(generate + 3 + 2K)

For B=16, K=1, T~256:
  = 16 * (generate + 3 + 2)
  = 16 * (generate + 5)
  = 16 generates + 80 full forward passes

For B=16, K=4:
  = 16 * (generate + 3 + 8)
  = 16 generates + 176 full forward passes
```

### Cost Breakdown by Type

| Pass type | Count | Grad? | hidden_states? | Sequence length |
|-----------|-------|-------|----------------|-----------------|
| `generate()` | B | No | No | ~T autoregressive steps |
| `_sequence_log_prob` (rollout) | B | No | No | prompt+response |
| `_critic_value_no_grad` | B | No | Yes | prompt only |
| `_critic_forward` (precompute) | B | No (full) | Yes | prompt only |
| `_critic_forward` (per epoch) | K*B | Critic only | Yes | prompt only |
| `_policy_log_probs` (per epoch) | K*B | Full policy | No | prompt+response |

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

**Status**: Open
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

---

## P2 — Double forward pass per sample during PPO update [CRITICAL]

**Status**: Open
**Severity**: Critical

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

**Status**: Open
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

---

## P5 — Redundant tokenisation in `_critic_forward` [MINOR]

**Status**: Open (moot after P2 fix)
**Severity**: Minor

**Location:** `ppo_specs/ppo_trainer.py` — `_critic_forward` re-tokenises prompts
that are already stored as token IDs in `rollout.full_ids[:rollout.prompt_len]`.

**Fix:** Use stored token IDs directly. Becomes moot once P2 eliminates the
separate `_critic_forward` call.

---

## P6 — No gradient checkpointing [CLUSTER BLOCKER]

**Status**: Open
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

---

## P7 — No distributed training / multi-GPU support [CLUSTER BLOCKER]

**Status**: Open
**Severity**: Cluster Blocker

**Location:** Entire codebase — no `accelerate`, no DDP.

**Fix:** Integrate HuggingFace `accelerate` with FSDP for 8B+ models.

---

## P8 — `torch.float32` on GPU: 2x slower than bfloat16 [MODERATE]

**Status**: Open
**Severity**: Moderate

**Location:** `ppo_specs/ppo_trainer.py` (`load_ppo_trainer`)

**Fix:**
```python
torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
```

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

## Implementation Priority

When adapting for the cluster, apply fixes in this order:

| Priority | Issue | Savings |
|----------|-------|---------|
| 1 | P1 — Batched rollout generation | 10-15x throughput gain |
| 2 | P6 — Gradient checkpointing | Required to avoid OOM with 8B model |
| 3 | P8 — bfloat16 | 2x speed + memory |
| 4 | P9 — Reuse generation logits | Saves B full-sequence forward passes/step |
| 5 | P2 — Combined forward pass | Saves K*B prompt-only forward passes/step |
| 6 | P7 — Multi-GPU (accelerate) | Further parallelism |
| 7 | P4 — Batched MC rollouts | Speeds up baseline estimation |
| 8 | P3 — Shared model state_dict | Saves re-load time in E2.8 |
| 9 | P10, P11 — Minor | Polish; minimal impact |

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
