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
