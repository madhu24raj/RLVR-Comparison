# Distributed Multi-GPU Training Spec for PPO Pipeline

This document specifies every change needed to run the PPO training pipeline
across multiple GPUs. The current codebase is entirely single-device.

---

## 1. Current Distributed Training Support

### Current State

**There is zero distributed training support.** None of the following appear
anywhere in the codebase:

- `torch.distributed` / `dist.init_process_group`
- `torch.nn.parallel.DistributedDataParallel` (DDP)
- `torch.distributed.fsdp` (FSDP)
- `accelerate.Accelerator`
- DeepSpeed
- Any multi-GPU launcher (`torchrun`, `accelerate launch`)

Despite `accelerate` being listed in `requirements.txt` (line 4), it is never
imported or used in any source file.

Key single-device patterns:

| File | Line(s) | Pattern |
|------|---------|---------|
| `ppo_trainer.py` | 108 | `self.device = device` (single `torch.device`) |
| `ppo_trainer.py` | 505 | `model.to(device)` — one device |
| `ppo_trainer.py` | 511 | `critic.to(device)` — one device |
| `run_e2_7.py` | 52 | `device = torch.device("cuda" if ... else "cpu")` |
| `run_e2_8.py` | 174 | Same single-device selection |
| `advantage.py` | 106 | `.to(device)` with string device |

---

## 2. Blockers for Multi-GPU Training

### 2.1 Hardcoded Single Device

### Current State

Every tensor movement uses a single `self.device` set in `PPOTrainer.__init__`
(line 108). The following locations all bind to this single device:

- `ppo_trainer.py:147` — `enc.to(self.device)` in `generate_rollouts`
- `ppo_trainer.py:279-280` — `rewards.to(self.device)`, `old_log_probs.to(self.device)` in `ppo_update`
- `ppo_trainer.py:356` — `torch.tensor(..., device=self.device)` in `_policy_log_probs`
- `ppo_trainer.py:389` — `enc.to(self.device)` in `_critic_forward`
- `ppo_trainer.py:466` — `enc.to(self.device)` in `evaluate`
- `ppo_trainer.py:505` — `model.to(device)` in `load_ppo_trainer`
- `ppo_trainer.py:511` — `critic.to(device)` in `load_ppo_trainer`

### Required Changes

Replace `self.device` with `accelerator.device` throughout. Under Accelerate,
the device is assigned per-process automatically.

### 2.2 Global Mutable State

### Current State

- `self.total_rollouts` (line 122): Incremented at line 186. Under data
  parallelism, each GPU increments its own copy. The metric would be N-times
  too low (where N = number of GPUs).
- `self.step` (line 122): Same issue — incremented at line 449.
- `run_e2_7.py:142-143` and `run_e2_8.py:124-128`: `trainer.total_rollouts`
  is saved and restored to exclude MC estimation rollouts. This pattern is
  fragile under multi-process execution.

### Required Changes

After each `generate_rollouts` call, all-reduce `total_rollouts` across ranks:
```python
total = torch.tensor(len(prompts), device=accelerator.device)
total = accelerator.reduce(total, reduction="sum")
self.total_rollouts += total.item()
```

### 2.3 Sequential Per-Sample Processing

### Current State (RESOLVED)

All per-sample loops have been converted to batched operations (P1 fix,
2026-04-08). `generate_rollouts`, `_batched_sequence_log_probs`,
`_critic_forward`, `_eval_critic_on_prompts`, and `evaluate` now process
full batches in single forward passes with padded sequences and attention masks.

This prerequisite for effective multi-GPU utilization is complete.

### 2.4 Optimizer Setup

### Current State

Two separate optimizers are created (lines 110-118):
```python
self.policy_optimizer = torch.optim.AdamW(model.parameters(), lr=...)
self.critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=...)
```

Both use vanilla PyTorch AdamW. This is compatible with DDP and Accelerate
wrapping. For DeepSpeed ZeRO, the optimizer must be replaced with
`deepspeed.ops.adam.DeepSpeedCPUAdam` or `FusedAdam`.

### Required Changes

Under Accelerate:
```python
self.policy_optimizer = torch.optim.AdamW(model.parameters(), lr=...)
self.model, self.policy_optimizer = accelerator.prepare(model, self.policy_optimizer)
```

### 2.5 Gradient Synchronization

### Current State

Gradients are computed via `total_loss.backward()` (line 323) on a combined
policy + critic loss. `clip_grad_norm_` (lines 325-327) is called on raw
`model.parameters()`. Under DDP, `clip_grad_norm_` must use
`accelerator.clip_grad_norm_` to handle gradient unscaling correctly.

### Required Changes

Replace:
```python
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```
with:
```python
accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```

---

## 3. HuggingFace Accelerate Integration

### Current State

`accelerate` is a dependency (requirements.txt line 4) but is never imported.

### Required Changes

#### 3.1 Initialization

**File: `ppo_trainer.py`, function `load_ppo_trainer` (line 489)**

Replace the current device selection and model loading:

```python
# BEFORE (lines 489-520)
def load_ppo_trainer(config: PPOConfig, device: torch.device) -> PPOTrainer:
    model = AutoModelForCausalLM.from_pretrained(...).to(device)
    critic = build_critic(...).to(device)
    return PPOTrainer(config, model, tokenizer, critic, gsm8k_reward, device)

# AFTER
def load_ppo_trainer(config: PPOConfig, accelerator: Accelerator) -> PPOTrainer:
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.bfloat16,
    )
    critic = build_critic(config.critic_capacity, model.config.hidden_size)

    return PPOTrainer(config, model, tokenizer, critic, gsm8k_reward, accelerator)
```

**File: `ppo_trainer.py`, class `PPOTrainer.__init__` (line 94)**

```python
# BEFORE
def __init__(self, config, model, tokenizer, critic, reward_fn, device):
    self.device = device
    self.policy_optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

# AFTER
def __init__(self, config, model, tokenizer, critic, reward_fn, accelerator):
    self.accelerator = accelerator
    self.device = accelerator.device

    self.policy_optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    self.model, self.policy_optimizer = accelerator.prepare(model, self.policy_optimizer)

    if critic.is_trainable():
        self.critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=config.critic_lr)
        self.critic, self.critic_optimizer = accelerator.prepare(critic, self.critic_optimizer)
    else:
        self.critic = critic.to(self.device)
        self.critic_optimizer = None
```

**File: `run_e2_7.py`, function `run_e2_7` (line 51)**

```python
# BEFORE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AFTER
from accelerate import Accelerator
accelerator = Accelerator()
device = accelerator.device
```

Same change in `run_e2_8.py` line 174.

#### 3.2 Rollout Generation Under DDP

**File: `ppo_trainer.py`, function `generate_rollouts` (line 127)**

`model.generate()` does not work under DDP wrapping because DDP expects
synchronized forward passes but generation is autoregressive and
non-deterministic. The solution is to unwrap the model for generation:

```python
@torch.no_grad()
def generate_rollouts(self, prompts, ground_truths):
    self.model.eval()
    # Unwrap DDP for generation — generation is inherently local
    unwrapped_model = self.accelerator.unwrap_model(self.model)
    rollouts = []

    # Split prompts across GPUs
    local_prompts, local_gts = self._shard_data(prompts, ground_truths)

    for prompt, gt in zip(local_prompts, local_gts):
        enc = self.tokenizer(prompt, return_tensors="pt", ...).to(self.device)
        out = unwrapped_model.generate(**enc, ...)
        # ... same as current code ...

    # Gather rollouts from all ranks
    all_rollouts = self._gather_rollouts(rollouts)
    self.total_rollouts += len(all_rollouts)
    return RolloutBatch(all_rollouts)
```

#### 3.3 Critic Training Under Accelerate

The critic is a separate `nn.Module` with its own optimizer. Accelerate handles
this naturally — `accelerator.prepare(critic, critic_optimizer)` wraps both.

The key subtlety is in `_critic_forward` (line 367): the policy model is run
with `torch.no_grad()` to get hidden states, then the critic runs WITH grad.
Under DDP, the policy's forward pass inside `_critic_forward` must also use
the unwrapped model to avoid DDP synchronization issues:

```python
def _critic_forward(self, batch, rewards):
    unwrapped_policy = self.accelerator.unwrap_model(self.model)
    for rollout in batch.rollouts:
        with torch.no_grad():
            outputs = unwrapped_policy(
                input_ids=enc["input_ids"],
                output_hidden_states=True, use_cache=False,
            )
        last_hidden = outputs.hidden_states[-1][:, -1, :].detach()
        v = self.critic(last_hidden)  # critic IS wrapped in DDP
        values.append(v)
```

#### 3.4 Backward Pass and Gradient Sync

**File: `ppo_trainer.py`, function `ppo_update` (line 258)**

Replace `total_loss.backward()` with:
```python
self.accelerator.backward(total_loss)
```

Replace `torch.nn.utils.clip_grad_norm_` with:
```python
self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
if self.critic.is_trainable():
    self.accelerator.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
```

### Implementation Plan

1. Add `from accelerate import Accelerator` to `ppo_trainer.py`, `run_e2_7.py`,
   `run_e2_8.py`.
2. Replace `device` parameter with `accelerator` in `PPOTrainer.__init__` and
   `load_ppo_trainer`.
3. Wrap model + optimizer with `accelerator.prepare()`.
4. Use `accelerator.unwrap_model()` for generation and no-grad forward passes.
5. Replace `loss.backward()` with `accelerator.backward(loss)`.
6. Replace `clip_grad_norm_` with `accelerator.clip_grad_norm_`.
7. Update run scripts to use `accelerate launch` instead of `python`.
8. Add data sharding helpers (see Section 5).
9. Guard print statements with `accelerator.is_main_process`.

---

## 4. DeepSpeed ZeRO Integration

### Current State

DeepSpeed is not listed in `requirements.txt` and is not used anywhere.

### 4.1 ZeRO-2 vs ZeRO-3

### Required Changes

**ZeRO-2 is recommended** for this architecture. Reasons:

- **ZeRO-2** shards optimizer states and gradients across GPUs. The model
  weights remain replicated. This works well because:
  - `model.generate()` requires full model weights on each GPU.
  - The critic uses `output_hidden_states=True` from the policy, which
    requires a full forward pass with all weights present.
  - ZeRO-2 already provides significant memory savings (optimizer states
    are the dominant cost for AdamW — 2x model size).

- **ZeRO-3** shards model weights too. This causes problems:
  - `model.generate()` requires weight gathering at every layer, which is
    slow and requires special `deepspeed.zero.GatheredParameters` context
    managers.
  - Hidden state extraction for the critic requires a full forward pass,
    which under ZeRO-3 triggers all-gather at every layer.
  - The added communication overhead may outweigh memory savings for a
    0.5B–8B model that fits in GPU memory with ZeRO-2.

### 4.2 Critic and ZeRO Sharding

The critic modules are very small (the largest, `LargeCriticMLP`, has
`2 * hidden_size * 2 * hidden_size + 2 * hidden_size + 1` parameters —
roughly 3.2M parameters for hidden_size=896). This is negligible compared
to the policy model.

**Recommendation:** Do NOT include the critic in the DeepSpeed engine.
Instead, keep the critic as a regular DDP-wrapped module:

```python
# DeepSpeed handles the policy
model_engine, policy_optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=policy_optimizer,
    config=ds_config,
)

# Critic stays as regular DDP
critic = critic.to(device)
critic = torch.nn.parallel.DistributedDataParallel(critic, device_ids=[local_rank])
critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=config.critic_lr)
```

Alternatively, with Accelerate + DeepSpeed plugin, this is handled by only
passing the policy to DeepSpeed and manually managing the critic.

### 4.3 RolloutBatch and Gathering

**File: `ppo_trainer.py`, class `RolloutBatch` (line 66)**

The `RolloutBatch` stores Python lists of `Rollout` dataclasses. Under data
parallelism, each GPU generates rollouts for its shard. Before `ppo_update`,
rollouts must be gathered:

```python
def gather_rollout_batch(batch: RolloutBatch, accelerator) -> RolloutBatch:
    """Gather rollouts from all ranks onto all ranks."""
    import torch.distributed as dist

    # Serialize rollouts to a list of dicts
    local_data = [dataclasses.asdict(r) for r in batch.rollouts]

    # Use all_gather_object for arbitrary Python objects
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_data)

    # Flatten and reconstruct
    all_rollouts = []
    for rank_data in gathered:
        for d in rank_data:
            all_rollouts.append(Rollout(**d))

    return RolloutBatch(all_rollouts)
```

### 4.4 DeepSpeed Config File

Create `RLVR-Comparison/ppo_specs/ds_config_zero2.json`:

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device": "none"},
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "overlap_comm": true
  },
  "gradient_accumulation_steps": 1,
  "gradient_clipping": 1.0,
  "train_micro_batch_size_per_gpu": 8,
  "wall_clock_breakdown": false
}
```

### Implementation Plan

1. Add `deepspeed` to `requirements.txt`.
2. Create `ds_config_zero2.json` as shown above.
3. In `load_ppo_trainer`, initialize DeepSpeed engine for the policy only.
4. Keep the critic as a separate DDP module.
5. Replace `loss.backward()` with `model_engine.backward(loss)`.
6. Replace `optimizer.step()` with `model_engine.step()`.
7. Use `model_engine.module` to access unwrapped model for `generate()`.
8. Launch with `deepspeed --num_gpus=N run_e2_7.py`.

---

## 5. Data Parallelism Specifics

### 5.1 Training Data Splitting

### Current State

**File: `ppo_specs/utils.py`, function `cycle_batch` (line 22)**

`cycle_batch` selects a contiguous slice of `batch_size` prompts from the
training data based on the step index. Every GPU gets the same slice.

### Required Changes

Each GPU should process a disjoint subset. Two approaches:

**Approach A — Shard at the batch level (recommended):**

```python
def cycle_batch_distributed(items, step, batch_size, rank, world_size):
    """Each rank gets batch_size // world_size items from the global batch."""
    global_batch = cycle_batch(items, step, batch_size)
    per_rank = batch_size // world_size
    start = rank * per_rank
    return global_batch[start : start + per_rank]
```

This requires `batch_size` to be divisible by `world_size`. Add a check:

```python
# In run_e2_7.py / run_e2_8.py, after creating config
assert config.batch_size % accelerator.num_processes == 0, \
    f"batch_size ({config.batch_size}) must be divisible by num_processes ({accelerator.num_processes})"
```

**File: `ppo_specs/config.py`**

The `batch_size` field (line 28) represents the GLOBAL batch size. Document
this and ensure it is always a multiple of the GPU count:

```python
batch_size: int = 16  # GLOBAL batch size; must be divisible by num_gpus
```

**Approach B — Use Accelerate's DataLoader:**

This is more complex because the current code does not use a `DataLoader`.
It would require wrapping prompts in a `Dataset` and using
`accelerator.prepare(dataloader)` for automatic sharding.

### 5.2 Rollout Generation Per GPU

### Current State

Each prompt is processed sequentially in a `for` loop (line 140). All prompts
go to the same GPU.

### Required Changes

Each GPU should generate rollouts for its shard of prompts. After generation,
rollouts are gathered so that the PPO update sees the full batch:

```python
@torch.no_grad()
def generate_rollouts(self, prompts, ground_truths):
    # Shard prompts across ranks
    rank = self.accelerator.process_index
    world_size = self.accelerator.num_processes
    per_rank = len(prompts) // world_size
    local_prompts = prompts[rank * per_rank : (rank + 1) * per_rank]
    local_gts = ground_truths[rank * per_rank : (rank + 1) * per_rank]

    # Generate locally (unwrapped model)
    unwrapped = self.accelerator.unwrap_model(self.model)
    local_rollouts = []
    for prompt, gt in zip(local_prompts, local_gts):
        # ... existing generation code using unwrapped model ...
        local_rollouts.append(rollout)

    # Gather across all ranks
    all_rollouts = gather_rollout_batch(
        RolloutBatch(local_rollouts), self.accelerator
    )
    self.total_rollouts += len(prompts)  # global count
    return all_rollouts
```

### 5.3 Metrics Aggregation

### Current State

Metrics are computed locally in `ppo_update` (lines 334-343) from tensors
that exist on one device:

```python
return {
    "policy_loss": policy_loss.item(),
    "mean_reward": rewards.mean().item(),
    ...
}
```

### Required Changes

After gathering rollouts, all ranks have the same `RolloutBatch`, so the
policy forward pass in `ppo_update` produces gradients that DDP auto-syncs.
Metrics are computed from the full gathered batch, so they are already global.

However, if we do NOT gather rollouts (i.e., each rank does PPO on its own
shard), then metrics must be reduced:

```python
def _reduce_metrics(self, metrics: dict) -> dict:
    """All-reduce scalar metrics across ranks."""
    reduced = {}
    for k, v in metrics.items():
        t = torch.tensor(v, device=self.device)
        t = self.accelerator.reduce(t, reduction="mean")
        reduced[k] = t.item()
    return reduced
```

Only the main process should print and log:

```python
if self.accelerator.is_main_process:
    print(f"step {step} | reward={metrics['mean_reward']:.3f}")
    logger.log_step(step, **log_entry)
    logger.save()
```

### 5.4 MC Baseline Parallelization

### Current State

**File: `advantage.py`, function `estimate_mc_advantages` (line 62)**

MC estimation generates `n_samples` completions per prompt, sequentially.
For 5 prompts x 50 samples = 250 sequential generations. This is the slowest
part of the pipeline.

### Required Changes

Parallelize across GPUs by splitting the samples:

```python
def estimate_mc_advantages_distributed(
    policy, tokenizer, prompts, ground_truths, reward_fn,
    n_samples, max_new_tokens, temperature, accelerator,
):
    """Distributed MC baseline estimation."""
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    # Each rank generates n_samples // world_size samples per prompt
    local_n = n_samples // world_size
    remainder = n_samples % world_size
    if rank < remainder:
        local_n += 1

    unwrapped = accelerator.unwrap_model(policy)
    unwrapped.eval()

    mc_baselines = {}
    with torch.no_grad():
        for prompt, gt in zip(prompts, ground_truths):
            enc = tokenizer(prompt, return_tensors="pt", ...).to(accelerator.device)
            prompt_len = enc["input_ids"].shape[1]
            local_rewards = []

            for _ in range(local_n):
                out = unwrapped.generate(**enc, ...)
                completion = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
                local_rewards.append(reward_fn(completion, gt))

            # All-reduce the sum and count
            local_sum = torch.tensor(sum(local_rewards), device=accelerator.device)
            local_count = torch.tensor(len(local_rewards), device=accelerator.device)
            global_sum = accelerator.reduce(local_sum, reduction="sum")
            global_count = accelerator.reduce(local_count, reduction="sum")

            mc_baselines[prompt] = (global_sum / global_count).item()

    return mc_baselines
```

This gives a near-linear speedup for MC estimation (the current bottleneck).

---

## 6. Launch Configuration

### Current State

Scripts are launched with:
```bash
python ppo_specs/run_e2_7.py --seed 0
```

### Required Changes

#### With Accelerate:

Create `RLVR-Comparison/ppo_specs/accelerate_config.yaml`:
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_machines: 1
num_processes: 4            # number of GPUs
mixed_precision: bf16
downcast_bf16: "no"
```

Launch with:
```bash
accelerate launch --config_file ppo_specs/accelerate_config.yaml \
    ppo_specs/run_e2_7.py --seed 0
```

#### With DeepSpeed:

```bash
deepspeed --num_gpus=4 ppo_specs/run_e2_7.py \
    --deepspeed ppo_specs/ds_config_zero2.json --seed 0
```

#### With torchrun (bare DDP):

```bash
torchrun --nproc_per_node=4 ppo_specs/run_e2_7.py --seed 0
```

---

## 7. Summary of All Required File Changes

| File | Change |
|------|--------|
| `ppo_trainer.py:94-122` | Replace `device` param with `accelerator`; wrap model/optimizer with `accelerator.prepare()` |
| `ppo_trainer.py:127-187` | Shard prompts, unwrap model for generation, gather rollouts |
| `ppo_trainer.py:189-213` | Use unwrapped model for `_sequence_log_prob` during generation |
| `ppo_trainer.py:258-343` | Replace `backward()` with `accelerator.backward()`; use `accelerator.clip_grad_norm_()` |
| `ppo_trainer.py:345-361` | No change needed if rollouts are gathered before `ppo_update` |
| `ppo_trainer.py:367-405` | Use unwrapped model for hidden state extraction |
| `ppo_trainer.py:452-484` | Shard eval prompts, gather results |
| `ppo_trainer.py:489-520` | Accept `accelerator` instead of `device`; remove `.to(device)` calls |
| `config.py:28` | Document that `batch_size` is global; add divisibility constraint |
| `run_e2_7.py:51-57` | Create `Accelerator`; replace device logic; guard prints with `is_main_process` |
| `run_e2_8.py:173-179` | Same as run_e2_7 |
| `advantage.py:62-125` | Add `estimate_mc_advantages_distributed` variant |
| `utils.py:22-40` | Add `cycle_batch_distributed` with rank/world_size params |
| `requirements.txt` | Add `deepspeed` if using ZeRO |

---

## 8. Recommended Migration Order

1. ~~**Batch the inner loops**~~ — **DONE** (2026-04-08). All per-sample loops
   converted to batched operations with padded sequences and attention masks.

2. **Integrate Accelerate** — This is the lowest-effort path. Replace device
   handling, wrap model/optimizer, add `accelerator.backward()`.

3. **Add data sharding** — Split prompts across ranks in `generate_rollouts`
   and `evaluate`. Gather rollouts before `ppo_update`.

4. **Add DeepSpeed ZeRO-2** — Only needed if model size exceeds single-GPU
   memory (e.g., Llama-3-8B). Use Accelerate's DeepSpeed plugin for minimal
   code changes.

5. **Parallelize MC estimation** — Split samples across GPUs, all-reduce sums.

6. **Guard I/O** — Wrap all `print`, `logger.save()`, and file writes with
   `accelerator.is_main_process`.
