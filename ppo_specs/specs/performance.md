# Performance Fix Spec

Severity levels:
- **CRITICAL** — causes 10× or greater slowdown; unacceptable for cluster runs
- **CLUSTER BLOCKER** — required to run at full scale; not needed for local tests
- **MODERATE** — 2–5× improvement available
- **MINOR** — small gain, low effort

Each issue: **Problem → Impact → Step-by-step fix**.

---

## P1 — Per-sample rollout generation instead of batched [CRITICAL]

**Location:** `ppo_specs/ppo_trainer.py:140–187` (`generate_rollouts`)

**Problem:**
```python
for prompt, gt in zip(prompts, ground_truths):
    enc = self.tokenizer(prompt, ...)      # one sample
    out = self.model.generate(**enc, ...)  # one generate() call
```
For `batch_size=16` this issues 16 sequential `model.generate()` calls where a
single batched call would suffice.

**Impact:** ~10–15× throughput reduction. On an A100 with Llama-3-8B, a single
`generate()` call on 16 prompts takes ~2 s; 16 sequential calls take ~20 s.
For 200 training steps this is ~65 min vs ~6.5 min.

**Fix:**

Replace the per-sample loop with a batched call, then decode per-sample:

```python
def generate_rollouts(
    self,
    prompts: list[str],
    ground_truths: list[str],
) -> RolloutBatch:
    assert len(prompts) == len(ground_truths)
    self.model.eval()

    # ── Batch tokenise (left-pad so all prompts end at the same position) ──
    self.tokenizer.padding_side = "left"
    enc = self.tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=self.config.tokenize_max_length,
        padding=True,
    ).to(self.device)

    prompt_lens = enc["attention_mask"].sum(dim=1).tolist()   # actual token counts

    with torch.no_grad():
        out = self.model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    rollouts: list[Rollout] = []
    for i, (prompt, gt) in enumerate(zip(prompts, ground_truths)):
        plen = prompt_lens[i]
        full_ids_i = out[i]                    # [padded_prompt + response]
        # Strip left-padding: the actual prompt starts at (total_len - plen)
        prompt_start = out.shape[1] - self.config.max_new_tokens - plen
        actual_full = full_ids_i[prompt_start:]
        completion = self.tokenizer.decode(actual_full[plen:], skip_special_tokens=True)

        old_lp  = self._sequence_log_prob(actual_full.unsqueeze(0), plen).item()
        reward  = self.reward_fn(completion, gt)
        value   = self._critic_value_no_grad(actual_full[:plen].unsqueeze(0))

        rollouts.append(Rollout(
            prompt=prompt, completion=completion, reward=reward,
            old_log_prob=old_lp, value=value,
            full_ids=actual_full.tolist(), prompt_len=plen,
        ))

    self.total_rollouts += len(prompts)
    return RolloutBatch(rollouts)
```

> **Note:** Left-padding is required for decoder-only LLMs so that attention masks
> are aligned. `tokenizer.padding_side = "left"` must be set before the batched call.

---

## P2 — Double forward pass per sample during PPO update [CRITICAL]

**Location:** `ppo_specs/ppo_trainer.py` — `_critic_forward` (line 245) and `_policy_log_probs` (line 257) both run a separate forward pass on the same sequences.

**Problem:**
```python
# Call 1: extract hidden states for critic
outputs = self.model(input_ids=prompt_ids, output_hidden_states=True)

# Call 2 (separate loop): compute log-probs from logits
outputs = self.model(input_ids=full_ids, use_cache=False)
```
Both use the same model on overlapping inputs.

**Impact:** 2× the forward-pass cost during every PPO update step.

**Fix:**

Merge into a single combined forward pass method `_ppo_forward`:

```python
def _ppo_forward(
    self, batch: RolloutBatch
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """
    Single forward pass that yields both critic values and sequence log-probs.

    Returns:
        critic_values: [B] tensor (or None for capacity="none")
        log_probs:     [B] tensor with grad
    """
    self.model.train()
    critic_values: list[torch.Tensor] = []
    log_probs_list: list[torch.Tensor] = []

    for rollout in batch.rollouts:
        full_ids = torch.tensor(
            [rollout.full_ids], dtype=torch.long, device=self.device
        )
        # One pass: request hidden states + logits simultaneously
        outputs = self.model(
            input_ids=full_ids,
            use_cache=False,
            output_hidden_states=True,
        )

        # ── Sequence log-prob (from logits) ──────────────────────────────────
        lp = self._log_prob_from_outputs(outputs, full_ids, rollout.prompt_len)
        log_probs_list.append(lp.squeeze(0))

        # ── Critic value (from last hidden state, detached from policy grad) ──
        if self.critic.is_trainable():
            self.critic.train()
            last_h = outputs.hidden_states[-1][:, -1, :].detach()
            v = self.critic(last_h).squeeze()
            critic_values.append(v)

        del full_ids  # free GPU tensor

    critic_out = torch.stack(critic_values) if critic_values else None
    return critic_out, torch.stack(log_probs_list)
```

Add helper:
```python
def _log_prob_from_outputs(self, outputs, input_ids, prompt_len):
    log_probs = torch.log_softmax(outputs.logits, dim=-1)
    response_ids = input_ids[:, prompt_len:]
    if response_ids.shape[1] == 0:
        return torch.zeros(1, device=self.device)
    response_lp = log_probs[:, prompt_len - 1 : -1, :]
    return response_lp.gather(2, response_ids.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
```

Replace the two-call pattern in `ppo_update` with one call to `_ppo_forward`.

---

## P3 — Model re-loaded from disk per critic capacity in E2.8 [MODERATE]

**Location:** `ppo_specs/run_e2_8.py:99` (`run_one_capacity` calls `load_ppo_trainer`)

**Problem:** Each of the four capacity runs calls `AutoModelForCausalLM.from_pretrained(...)`,
which loads the model from disk (or HuggingFace cache) into a new Python object.

**Impact:** ~30–60 s per load (SSD) or longer (network). For 4 capacities across 3
seeds = 12 extra model loads ≈ 6–12 min overhead.

**Fix:**

Load the base model once; deep-copy its state dict; reset weights before each run:

```python
# In run_e2_8.py, before the capacity loop:
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy

print(f"[E2.8] Loading base model once: {config.model_name}")
base_model = AutoModelForCausalLM.from_pretrained(
    config.model_name, torch_dtype=torch_dtype
).to(device)
base_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# In run_one_capacity, instead of load_ppo_trainer:
base_model.load_state_dict({k: v.to(device) for k, v in base_state.items()})
critic = build_critic(capacity, base_model.config.hidden_size).to(device)
trainer = PPOTrainer(cfg, base_model, tokenizer, critic, gsm8k_reward, device)
```

---

## P4 — Sequential MC rollouts per prompt [MODERATE]

**Location:** `ppo_specs/advantage.py:92–116` (`estimate_mc_advantages`)

**Problem:**
```python
for _ in range(n_samples):      # 50 or 1000 iterations
    out = policy.generate(**enc, ...)   # one sample at a time
```
For `n_samples=1000` and 5 prompts this is 5,000 sequential `generate()` calls.

**Impact:** At ~0.5 s/call, MC estimation takes ~42 min on cluster for the full
n_samples=1000 setting. Batching brings this to ~2–5 min.

**Fix:**

Batch identical copies of the same prompt:
```python
def estimate_mc_advantages(
    policy, tokenizer, prompts, ground_truths, reward_fn,
    n_samples=50, max_new_tokens=128, device="cpu",
    gen_batch_size=8,      # new parameter
) -> dict[str, float]:
    policy.eval()
    mc_baselines = {}
    with torch.no_grad():
        for prompt, gt in zip(prompts, ground_truths):
            enc = tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=512, padding=False,
            )
            prompt_len = enc["input_ids"].shape[1]
            sample_rewards = []

            # Generate in sub-batches of gen_batch_size
            for batch_start in range(0, n_samples, gen_batch_size):
                actual_bs = min(gen_batch_size, n_samples - batch_start)
                batch_enc = {
                    k: v.repeat(actual_bs, 1).to(device) for k, v in enc.items()
                }
                out = policy.generate(
                    **batch_enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
                for row in out:
                    completion = tokenizer.decode(row[prompt_len:], skip_special_tokens=True)
                    sample_rewards.append(reward_fn(completion, gt))
                del batch_enc, out

            mc_baselines[prompt] = float(np.mean(sample_rewards))
    return mc_baselines
```

---

## P5 — Redundant tokenisation in `_critic_forward` [MINOR]

**Location:** `ppo_specs/ppo_trainer.py:328–334`

**Problem:** Each rollout's prompt is re-tokenised during the critic update, even though
`rollout.full_ids[:rollout.prompt_len]` already contains the prompt tokens.

**Fix:**

1. Add `prompt_ids: list[int]` to the `Rollout` dataclass.
2. In `generate_rollouts`, populate it: `prompt_ids=enc["input_ids"][0].tolist()`.
3. In `_critic_forward`, use stored ids directly:
```python
prompt_ids = torch.tensor(
    [rollout.prompt_ids], dtype=torch.long, device=self.device
)
```
Remove the `self.tokenizer(...)` call.

> **Note:** After fixing P2 (combined forward pass), `_critic_forward` is eliminated
> entirely, making P5 moot.

---

## P6 — No gradient checkpointing [CLUSTER BLOCKER]

**Location:** `ppo_specs/ppo_trainer.py:418–421` (`load_ppo_trainer`)

**Problem:** Llama-3-8B requires ~40 GB for weights in bfloat16. The backward pass
accumulates activation tensors that can push VRAM to ~80–120 GB on a single A100
(80 GB), causing OOM.

**Fix:**

Enable gradient checkpointing immediately after loading:
```python
model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    torch_dtype=torch_dtype,
)
if device.type == "cuda":
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
model = model.to(device)
```
This recomputes activations during backward instead of storing them, trading ~30%
compute time for ~50% VRAM reduction.

---

## P7 — No distributed training / multi-GPU support [CLUSTER BLOCKER]

**Location:** Entire codebase — no `accelerate`, no DDP, no `device_map`.

**Problem:** All training runs on a single GPU. Multi-GPU A100 nodes go unused.

**Impact:** Training that could run in 2 h on 4× A100 takes 8+ h on 1× A100.

**Fix:**

Integrate HuggingFace `accelerate`:

1. Install: `pip install accelerate` (already in `requirements.txt` via `accelerate`).
2. Add to `load_ppo_trainer`:
```python
from accelerate import Accelerator
accelerator = Accelerator()
device = accelerator.device
```
3. Wrap model, optimisers, and data loaders:
```python
model, policy_optimizer, critic_optimizer = accelerator.prepare(
    model, policy_optimizer, critic_optimizer
)
```
4. Replace `total_loss.backward()` with `accelerator.backward(total_loss)`.
5. Launch with:
```bash
accelerate launch --num_processes 4 ppo_specs/run_e2_7.py --seed 0
```

For FSDP (model parallelism across GPUs for 8B+):
```python
# accelerate config: use "fsdp" strategy with auto wrapping
accelerate launch --config_file accelerate_fsdp.yaml ppo_specs/run_e2_7.py
```

---

## P8 — `torch.float32` on GPU: 2× slower than bfloat16 [MODERATE]

**Location:** `ppo_specs/ppo_trainer.py:420`

**Problem:** Float32 is forced for all hardware. On A100/H100, bfloat16 halves
memory usage and doubles throughput with negligible accuracy impact.

**Fix:** (See also `safety.md` S14 for the safety angle.)
```python
torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    config.model_name, torch_dtype=torch_dtype
).to(device)
```
Add `use_bfloat16: bool = True` to `PPOConfig` so the local test can force float32:
```python
torch_dtype = (
    torch.bfloat16
    if device.type == "cuda" and config.use_bfloat16
    else torch.float32
)
```

---

## P9 — KV cache disabled in evaluation [MINOR]

**Location:** `ppo_specs/ppo_trainer.py:389–394` (`evaluate`)

**Problem:** `model.generate()` during evaluation does not explicitly enable the KV
cache. For long sequences, the KV cache speeds up autoregressive decoding by avoiding
redundant key-value recomputation.

**Fix:**
```python
out = self.model.generate(
    **enc,
    max_new_tokens=self.config.max_new_tokens,
    do_sample=False,
    use_cache=True,    # add this line
    pad_token_id=self.tokenizer.pad_token_id,
)
```
`use_cache=True` is the default for `generate()`, so explicitly stating it is
mainly for clarity. Confirm that `use_cache=False` is not set elsewhere in the eval path.

---

## P10 — Redundant `mkdir` in `run_e2_8.py` [TRIVIAL]

**Location:** `ppo_specs/run_e2_8.py:221`

**Problem:**
```python
out_path = Path(config.output_dir) / "e2_8_sweep_summary.json"
out_path.parent.mkdir(parents=True, exist_ok=True)   # redundant
```
`ExperimentLogger.__init__` (called earlier in each capacity run) already creates
`config.output_dir` with `mkdir(parents=True, exist_ok=True)`.

**Fix:** Remove line 221. `out_path.parent` is already guaranteed to exist.

---

## Implementation Priority

When adapting for the cluster, apply fixes in this order:

| Priority | Issue | Reason |
|----------|-------|--------|
| 1 | P1 — Batched rollout generation | Biggest throughput gain |
| 2 | P6 — Gradient checkpointing | Required to avoid OOM with 8B model |
| 3 | P8 — bfloat16 | 2× speed + memory |
| 4 | P7 — Multi-GPU (accelerate) | Further parallelism |
| 5 | P2 — Combined forward pass | Eliminates redundant compute |
| 6 | P4 — Batched MC rollouts | Speeds up baseline estimation |
| 7 | P3 — Shared model state_dict | Saves re-load time in E2.8 |
| 8 | P5, P9, P10 | Polish; minimal impact |
