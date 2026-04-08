# Safety Fix Spec

Severity levels: **HIGH** (can silently corrupt training or crash), **MEDIUM** (can cause
OOM, incorrect results, or hard-to-debug failures), **LOW** (latent issues on edge hardware).

Each issue follows: **Problem → Failure mode → Step-by-step fix**.

---

## S1 — Exponential overflow in PPO ratio [HIGH]

**Location:** `ppo_specs/ppo_trainer.py:259`

**Problem:**
```python
ratio = torch.exp(new_log_probs - old_log_probs.detach())
```
When the model updates significantly between rollout and update (large log-prob
delta), `torch.exp` overflows to `inf`. For example, if `old_log_prob = -50` and
`new_log_prob = -0.5`, the delta is 49.5 → `exp(49.5) ≈ 5e21`.

**Failure mode:** `ratio` becomes `inf` → `ratio * advantages` is `nan` → policy
loss is `nan` → gradients are `nan` → all parameters silently become `nan` → training
appears to run but produces random outputs.

**Fix:**
```python
# Clamp log-ratio before exponentiation to prevent overflow/underflow.
# Values outside [-20, 20] represent a >1e8 policy shift — the PPO
# clip at ε=0.2 would discard these anyway.
log_ratio = new_log_probs - old_log_probs.detach()
log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
ratio = torch.exp(log_ratio)
clipped = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon)
policy_loss = -torch.mean(torch.min(ratio * advantages, clipped * advantages))
```

---

## S2 — Input length mismatch not validated [HIGH]

**Location:** `ppo_specs/ppo_trainer.py:140` (`generate_rollouts`), `ppo_trainer.py:379` (`evaluate`)

**Problem:**
```python
for prompt, gt in zip(prompts, ground_truths):
```
`zip` silently stops at the shorter list. A caller that accidentally passes
mismatched lengths processes fewer samples without any error.

**Failure mode:** Training on wrong prompt–answer pairs; evaluation accuracy computed
on a smaller set than reported; silent data corruption.

**Fix:**
Add at the top of both `generate_rollouts` and `evaluate`:
```python
if len(prompts) != len(ground_truths):
    raise ValueError(
        f"Length mismatch: {len(prompts)} prompts vs {len(ground_truths)} ground truths"
    )
```

---

## S3 — Zero-std advantage normalization inconsistency [HIGH]

**Location:** `ppo_specs/advantage.py:48–49`

**Problem:**
```python
if normalize and advantages.numel() > 1 and advantages.std() > 1e-8:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```
When all rewards in a batch are identical (e.g., all zeros after the model stalls,
or all ones on a trivial batch), `advantages.std()` is exactly 0 and the condition
skips normalization. On the *next* step with mixed rewards the normalization runs
normally. This causes gradient scales to vary wildly between steps.

**Failure mode:** Inconsistent advantage magnitudes across iterations → unstable
training, especially when the model temporarily converges to uniform outputs.

**Fix:** Compute `std` once and always apply normalization (with safe denominator):
```python
if normalize and advantages.numel() > 1:
    std = advantages.std()
    # If std is zero, advantages are all equal; subtracting mean gives all-zeros,
    # which is the correct zero-gradient signal. No need to divide.
    if std > 1e-8:
        advantages = (advantages - advantages.mean()) / (std + 1e-8)
    else:
        advantages = advantages - advantages.mean()  # all-zero result
```

---

## S4 — Empty response causes shape mismatch in `torch.stack` [MEDIUM]

**Location:** `ppo_specs/ppo_trainer.py:203–205`

**Problem:**
```python
if response_ids.shape[1] == 0:
    return torch.tensor(0.0, device=self.device)   # shape []
```
The caller `_policy_log_probs` does:
```python
log_probs.append(lp.squeeze(0))   # squeeze on shape [] → still shape []
...
return torch.stack(log_probs)     # fails if any element has shape [] vs [1]
```

**Failure mode:** If any prompt in the batch generates zero new tokens, `torch.stack`
raises a shape error and the entire training step crashes.

**Fix:**
```python
if response_ids.shape[1] == 0:
    return torch.zeros(1, device=self.device)   # shape [1], consistent with normal path
```
Also add a warning:
```python
import warnings
if response_ids.shape[1] == 0:
    warnings.warn("Empty response generated — assigning log_prob=0.", RuntimeWarning)
    return torch.zeros(1, device=self.device)
```

---

## S5 — Critic squeeze shape bug [MEDIUM]

**Location:** `ppo_specs/ppo_trainer.py:344`

**Problem:**
```python
v = self.critic(last_hidden).squeeze(0)
```
`self.critic(last_hidden)` returns shape `[1]` (batch-size 1). `.squeeze(0)` removes
dim 0 → scalar tensor (shape `[]`). BUT if `LargeCriticMLP` or a future critic returns
shape `[1, 1]`, `.squeeze(0)` gives `[1]`, and `torch.stack` later produces `[B, 1]`
instead of `[B]`, causing a shape mismatch in the MSE loss.

**Fix:** Use `.squeeze()` without arguments to remove all size-1 dimensions:
```python
v = self.critic(last_hidden).squeeze()   # always scalar regardless of critic output shape
```

---

## S6 — GPU memory leak in `_policy_log_probs` [MEDIUM]

**Location:** `ppo_specs/ppo_trainer.py:299–300`

**Problem:**
```python
for rollout in batch.rollouts:
    full_ids = torch.tensor([rollout.full_ids], dtype=torch.long, device=self.device)
    lp = self._sequence_log_prob(full_ids, rollout.prompt_len)
    log_probs.append(lp.squeeze(0))
    # full_ids tensor stays on GPU until Python GC runs
```
Each `full_ids` tensor (up to 768 tokens × int64 = ~6 KB per sample) is created
on GPU and not freed until CPython's GC runs, which may not happen within the training loop.

**Failure mode:** Slow VRAM growth over many steps; eventual OOM on long training runs.

**Fix:** Explicitly delete after use:
```python
full_ids = torch.tensor([rollout.full_ids], dtype=torch.long, device=self.device)
lp = self._sequence_log_prob(full_ids, rollout.prompt_len)
log_probs.append(lp.squeeze(0))
del full_ids
```

---

## S7 — GPU memory leak in `_critic_forward` [MEDIUM]

**Location:** `ppo_specs/ppo_trainer.py:328–334`

**Problem:**
```python
for rollout in batch.rollouts:
    enc = self.tokenizer(rollout.prompt, ...).to(self.device)
    # enc["input_ids"] stays on GPU after the loop body
```
`enc` is re-created every iteration and moved to GPU, but never explicitly freed.

**Failure mode:** Same as S6 — VRAM leak during extended training.

**Fix:**
```python
enc = self.tokenizer(rollout.prompt, ...).to(self.device)
with torch.no_grad():
    outputs = self.model(input_ids=enc["input_ids"], ...)
del enc   # free GPU tensor immediately
last_hidden = outputs.hidden_states[-1][:, -1, :]
```

---

## S8 — `del tmp_trainer` does not free GPU VRAM [MEDIUM]

**Location:** `ppo_specs/run_e2_8.py:200–202`

**Problem:**
```python
del tmp_trainer
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```
`del` in Python decrements the reference count but does not trigger the garbage
collector synchronously, especially for cyclic references. `empty_cache()` only
frees CUDA memory that is already unused — it cannot force collection.

**Failure mode:** The 0.5B (or 8B) model remains in GPU memory throughout the sweep,
causing OOM when loading the first per-capacity model.

**Fix:**
```python
import gc
tmp_trainer.model.cpu()       # move weights off GPU first
del tmp_trainer
gc.collect()                  # force CPython GC
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

---

## S9 — NaN in reward variance at first eval step [MEDIUM]

**Location:** `ppo_specs/run_e2_7.py:119–121`

**Problem:**
```python
window = reward_history[-config.eval_every:] if len(reward_history) >= config.eval_every \
         else reward_history
stability = float(np.var(window))
```
If `eval_every=2` and the first eval fires at step 0 before any rewards are
accumulated, `reward_history` is empty and `np.var([])` returns `nan`.

**Failure mode:** `nan` propagates into `logger.log_step` and the JSON output file;
downstream plotting code crashes when parsing the first entry.

**Fix:**
```python
if not reward_history:
    reward_variance = 0.0
else:
    window = reward_history[-config.eval_every:] if len(reward_history) >= config.eval_every \
             else reward_history
    reward_variance = float(np.var(window))
```

---

## S10 — Empty completion not handled [MEDIUM]

**Location:** `ppo_specs/ppo_trainer.py:161–162`

**Problem:** If the model immediately emits EOS, `completion` is an empty string.
`gsm8k_reward("", gt)` returns 0.0 (correct behaviour), but `_sequence_log_prob`
returns `tensor([0.0])` — which implies the policy assigns probability 1.0 to this
(empty) sequence. This is factually wrong and harms the log-ratio computation.

**Failure mode:** Model learns that generating nothing is "neutral" (log-prob 0),
which distorts the PPO ratio and can lead to a degenerate policy that generates
minimal tokens.

**Fix:**
```python
completion = self.tokenizer.decode(full_ids[prompt_len:], skip_special_tokens=True)
if not completion.strip():
    import warnings
    warnings.warn(
        f"Empty completion generated for prompt: {prompt[:60]!r} … "
        "Assigning reward=0 and skipping gradient for this sample.",
        RuntimeWarning,
    )
    # Store a flag so ppo_update can exclude this sample
    rollouts.append(Rollout(..., skip_grad=True))
```
Add `skip_grad: bool = False` field to the `Rollout` dataclass and mask these
samples in `_compute_policy_loss`:
```python
mask = torch.tensor([not r.skip_grad for r in batch.rollouts], dtype=torch.float32)
policy_loss = -torch.mean(torch.min(ratio * advantages, clipped * advantages) * mask)
```

---

## S11 — Tokenizer `pad_token` may overwrite a meaningful value [MEDIUM]

**Location:** `ppo_specs/ppo_trainer.py:414–416`

**Problem:**
```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```
Some models (e.g., future versions of Llama) may set `pad_token` to a value
distinct from `eos_token`. Overwriting it can cause incorrect padding behaviour or
affect generation stop conditions.

**Fix:**
```python
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
elif tokenizer.pad_token_id != tokenizer.eos_token_id:
    # Pad token exists but differs from eos — keep it; just ensure generate()
    # uses the right pad_token_id below.
    pass
```
In all `model.generate(...)` calls, pass `pad_token_id=self.tokenizer.pad_token_id`
(not the hardcoded `eos_token_id`).

---

## S12 — `n_eval` silently truncates without warning [LOW]

**Location:** `ppo_specs/ppo_trainer.py:379`

**Problem:**
```python
for prompt, gt in zip(prompts[:n_eval], ground_truths[:n_eval]):
```
If `n_eval=50` but `len(prompts)=20`, the loop runs 20 times; the returned
accuracy is on 20 prompts but the caller believes it used 50.

**Fix:**
```python
n_eval = min(n_eval, len(prompts))
if n_eval < n_eval:   # won't trigger; replace with explicit check:
actual_n = min(n_eval, len(prompts))
if actual_n < n_eval:
    print(f"[eval] Warning: n_eval={n_eval} > dataset size={len(prompts)}; using {actual_n}")
for prompt, gt in zip(prompts[:actual_n], ground_truths[:actual_n]):
```

---

## S13 — Log-prob computed over padding tokens [MEDIUM]

**Location:** `ppo_specs/ppo_trainer.py:200–213`

**Problem:** `generate()` may append padding tokens after EOS in some model/tokenizer
combinations. `_sequence_log_prob` then sums log-probs over those padding tokens,
computing an incorrect sequence probability.

**Fix:** After generation, mask out any tokens at or after the first EOS:
```python
response_ids = full_ids[0, prompt_len:]
# Find first EOS and truncate
eos_positions = (response_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
if len(eos_positions) > 0:
    response_ids = response_ids[: eos_positions[0] + 1]   # include EOS, drop trailing pad
```
Apply the same truncation before storing `full_ids` in the Rollout.

---

## S14 — `torch.float32` hardcoded; breaks on bf16-only hardware [LOW]

**Location:** `ppo_specs/ppo_trainer.py:420`

**Problem:**
```python
model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.float32)
```
Float32 is needed on CPU (float16 produces NaN there), but is 2× slower on A100/H100
and may not work on bf16-only accelerators.

**Fix:**
```python
torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch_dtype)
```
Also add `torch_dtype` as an optional field in `PPOConfig`:
```python
torch_dtype: str = "auto"   # "float32" | "bfloat16" | "auto"
```

---

## S15 — Redundant `.detach()` inside `torch.no_grad()` [LOW]

**Location:** `ppo_specs/ppo_trainer.py:343`

**Problem:**
```python
with torch.no_grad():
    outputs = self.model(...)
last_hidden = outputs.hidden_states[-1][:, -1, :].detach()
```
Inside `torch.no_grad()`, no gradients are computed and no computation graph is
built. The `.detach()` on line 343 is redundant and misleads readers into thinking
the tensor would otherwise carry a gradient.

**Fix:** Remove `.detach()`:
```python
last_hidden = outputs.hidden_states[-1][:, -1, :]   # no_grad() already stops graph
```

---

## S16 — Log-ratio clamp at ±20 is extremely loose [LOW]

### ID: S16
**Status**: Open (by design)
**Severity**: Low
**Description**: The log-ratio is clamped to [-20, 20] before exponentiation.
`exp(20) ≈ 4.85e8`, which is an astronomically large policy ratio. The PPO clip
at ε=0.2 means the gradient only flows through ratios in [0.8, 1.2]. Any ratio
outside this range is clipped to 1±ε anyway, so the log-ratio clamp at ±20 is
purely a NaN-prevention safety net, not a functional constraint.

**Analysis**: A tighter clamp (e.g., ±5, where exp(5)≈148) would still be far
outside the useful range and would provide earlier protection against numerical
issues. However, since the current clamp is correct and the PPO clip handles
the functional constraint, this is cosmetic. No change needed unless training
exhibits instability from very large (but not infinite) ratios.

---

## S17 — Potential gradient graph retention in metrics dictionary [LOW]

### ID: S17
**Status**: Fixed
**Severity**: Low
**Description**: In `ppo_update`, the clip_fraction metric was computed using
`ratio` which had gradients attached. Although `.item()` was called (extracting
a Python float), the intermediate computation `(ratio - 1.0).abs()` created
temporary tensors on the computation graph, which could delay memory freeing.

**Fix**: Now uses `ratio.detach()` before computing clip_fraction.

---

## Deep Review — Numerical Stability Assessment

### Sequence-level log-prob accumulation

`_sequence_log_prob` sums per-token log-probs over the full response. For long
responses (e.g., 256 tokens), this sum can be very negative (e.g., -500). The
difference `new_log_prob - old_log_prob` is then computed between two large
negative numbers, which can lose precision. However, since both are computed
from the same token sequence with similar models, the difference is typically
small (within ±10), so this is acceptable.

The use of `torch.log_softmax` (fused log+softmax) is numerically stable,
avoiding the separate `log(softmax(x))` pattern that can produce `-inf` for
rare tokens.

### Advantage normalization edge cases

The updated normalization handles three cases:
1. Normal: `(A - mean) / (std + eps)` -- full z-score
2. Zero-std (all advantages equal): `A - mean` = all zeros -- correct zero gradient
3. Single element: normalization skipped -- correct (no meaningful statistics)

This is robust against all batch compositions including all-correct, all-incorrect,
and single-sample batches.

---

## Training Dynamics Safety Review

### TD-S1: Optimizer gradient isolation verified
**Status**: Verified Safe
**Severity**: N/A (verification)
**Description**: With separate optimizers for policy and critic, `total_loss.backward()`
computes gradients for ALL parameters in the computation graph. The question is whether
each optimizer only updates the correct parameters.

**Analysis**:
- `policy_optimizer` is constructed over `model.parameters()`. After `backward()`,
  `model.parameters()` have `.grad` populated from policy_loss + kl terms only
  (critic_loss contributes zero gradient to model params due to `torch.no_grad()` +
  `.detach()` in `_critic_forward`). `policy_optimizer.step()` updates model params
  with the correct gradients.
- `critic_optimizer` is constructed over `critic.parameters()`. After `backward()`,
  `critic.parameters()` have `.grad` populated from critic_loss only (policy_loss
  and kl contribute zero gradient to critic params since they don't involve critic
  parameters). `critic_optimizer.step()` updates critic params with correct gradients.
- There is NO cross-contamination: each optimizer only sees gradients from the correct
  loss terms. The single `backward()` call is efficient and correct.

### TD-S2: Critic loss coefficient now configurable (was hardcoded)
**Status**: Fixed
**Severity**: Medium
**Description**: The critic loss coefficient was hardcoded as 0.5 in `ppo_update`.
If a researcher wanted to adjust the relative weight of critic training, they had to
modify source code. This was also undocumented, making it easy to miss in a paper.

**Fix**: Added `critic_loss_coeff: float = 0.5` to `PPOConfig`. The `ppo_update`
method now reads `self.config.critic_loss_coeff` instead of the hardcoded 0.5.

### TD-S3: No learning rate warmup (risk assessment)
**Status**: Open
**Severity**: Low
**Description**: Both policy and critic optimizers start at their full learning rates
from step 0. For LLM fine-tuning, a linear warmup over 5-10% of total steps is
standard practice to prevent large initial gradient updates that could destabilize
the pretrained representations.

**Analysis**: For the current experiment settings (100-200 steps, batch_size=8-16,
lr=1e-5), the risk is low:
- The learning rate 1e-5 is already conservative.
- PPO's clipping mechanism bounds the effective policy change per step.
- The gradient clipping (max_norm=1.0) provides additional protection.

For cluster runs with larger batches or higher learning rates, adding warmup is
recommended. Add `warmup_steps: int = 0` to PPOConfig and create a
`torch.optim.lr_scheduler.LinearLR` scheduler if warmup_steps > 0.

### TD-S4: Binary reward special considerations
**Status**: Verified Safe
**Severity**: N/A (verification)
**Description**: With binary rewards {0,1}, several aspects of the training dynamics
differ from continuous-reward settings.

**Analysis**:
- **Advantage distribution**: For batch_size=8 with binary rewards, advantages can
  only take a small number of distinct values. Normalization via z-score is still
  meaningful as long as at least one correct and one incorrect response exist.
- **Critic targets**: The critic learns to predict E[r|s] which is a probability in
  [0,1]. MSE loss is appropriate and bounded.
- **Reward variance**: `rewards.var()` with binary rewards equals p(1-p) where p is
  accuracy. This is a useful training stability metric.
- **All-same batches**: When all rewards are 0 or all are 1, advantages are all zero
  after normalization, producing zero policy gradient. This is correct -- there is no
  signal about which responses are better.
