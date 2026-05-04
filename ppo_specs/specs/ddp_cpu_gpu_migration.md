# PPO Cluster DDP Enablement: CPU Smoke → Multi-GPU

Status: PARTIALLY LANDED (trainer + checkpoint + run-script Accelerate refactor complete; SLURM `DEVICE_MODE` branch and `setup_env.sh --cpu-only` flag still pending)
Author: planning agents
Date: 2026-04-24 (last refresh: 2026-05-04)

## Baseline invariant (added 2026-05-04)

> The PPO trainer is, at its most baseline, a PPO trainer with custom reward
> functions. The DDP refactor MUST NOT break the existing reward path:
> the `reward_fn` callable, `reward_mode` knob, `_RewardFnWrapper`,
> `make_reward_fn`, and `SelfJudgeRewardModel` all keep working under
> single-process and multi-process Accelerate. The
> [reward_model_integration.md §Scope](reward_model_integration.md#scope)
> baseline invariant applies symmetrically here: any DDP change that
> requires a structural shift in the reward layer (e.g., moving
> `set_questions` semantics) is in scope only if it preserves bit-identical
> behavior in the single-process path.

## Current implementation status (2026-05-04)

| Item | Status | Notes |
|------|--------|-------|
| `PPOTrainer.__init__` Accelerate signature | DONE | [ppo_trainer.py:154-244](../ppo_trainer.py#L154-L244). Accepts both `device` and `accelerator` (exactly one required). |
| `generate_rollouts` shard + unwrap + gather | DONE | [ppo_trainer.py:267-389](../ppo_trainer.py#L267-L389). Shard at L284-L297, gather at L365-L369, global counter at L388. |
| `_extract_last_hidden` `no_grad` envelope | DONE | [ppo_trainer.py:542-567](../ppo_trainer.py#L542-L567) — the §7.6.1 hazard fix is in. |
| `ppo_update` Accelerate backward + clip | DONE | [ppo_trainer.py:766-791](../ppo_trainer.py#L766-L791) — `accelerator.backward` + `accelerator.clip_grad_norm_` under `_is_ddp` branch. |
| `evaluate` shard + gather_for_metrics | DONE | [ppo_trainer.py:954-1035](../ppo_trainer.py#L954-L1035). Includes the divisibility-pad fix at L972-L984. |
| `load_ppo_trainer` Accelerate signature | DONE | [ppo_trainer.py:1040-1197](../ppo_trainer.py#L1040-L1197). `device_or_accelerator: Union[torch.device, Accelerator]`. |
| `set_questions(local_shard)` (§7.6.2 fix) | DONE | [ppo_trainer.py:990-991](../ppo_trainer.py#L990-L991) inside `evaluate`; [run_e2_7.py:216-224](../run_e2_7.py#L216-L224); [run_e2_8.py:127-135](../run_e2_8.py#L127-L135). |
| Resume re-prepare (§7.6.3 fix) | DONE | [run_e2_7.py:153-156](../run_e2_7.py#L153-L156) re-runs `accelerator.prepare` after the model reassignment. |
| Exit-signal all-reduce (§7.6.4 fix) | DONE | [run_e2_7.py:331-348](../run_e2_7.py#L331-L348). |
| MC-baseline rank-0 + broadcast (§7.6.5 fix) | DONE | [run_e2_7.py:174-196](../run_e2_7.py#L174-L196); [run_e2_8.py:275-293](../run_e2_8.py#L275-L293). |
| `checkpoint.save_checkpoint` unwrap + caller-gated | DONE | [checkpoint.py:28-130](../checkpoint.py#L28-L130). The function does NOT internally rank-0-gate; callers must precede it with `wait_for_everyone()` and `is_main_process` (documented in the docstring at L50-L63). |
| `_config_hash` global-batch-size invariant | DONE | [checkpoint.py:210-230](../checkpoint.py#L210-L230). 13 fields hashed; `batch_size` is the global value. |
| `configs/accelerate_cpu.yaml` | DONE | Present at the repo root. |
| `setup_env.sh --cpu-only` flag | **NOT DONE** | `scripts/setup_env.sh` defaults at L19-L26 do not include CPU_ONLY; parser at L29-L39 has no `--cpu-only` case. Agent A still owes this. |
| SLURM `DEVICE_MODE` branch in `slurm_e2_7.sh` / `slurm_e2_8.sh` | **NOT DONE** | `module load cuda` runs unconditionally at [slurm_e2_7.sh:72](../../scripts/slurm_e2_7.sh#L72) and [slurm_e2_8.sh:75](../../scripts/slurm_e2_8.sh#L75); NCCL block at slurm_e2_7.sh:86-93 / slurm_e2_8.sh:88-93 is not gated. Single-GPU launcher at slurm_e2_7.sh:137-142 still uses bare `python` instead of `accelerate launch`. Agent E still owes this. |
| Barriers around saves | PARTIAL | `run_e2_7.py` has 4 effective `_wait()` barriers around the 3 `save_checkpoint` calls (L321→323, L339→342, L347, L352→354). `run_e2_8.py` has 5 `_wait()` barriers but **no `save_checkpoint` calls** (E2.8 doesn't yet support resume). When E2.8 adds checkpointing, mirror the run_e2_7 pattern. |

## Context

The PPO pipeline is single-device today. `PPOTrainer.__init__` accepts a
`torch.device`; every tensor lives on it; `model.generate()`, `.backward()`,
and `clip_grad_norm_` all assume one process. The end goal is multi-node
multi-GPU DDP — the codebase's third-largest blocker to running E2.7 / E2.8
at 8B scale.

Jumping straight to GPU-cluster DDP is expensive: distributed bugs (rank-0
gating, `.generate()` hangs under DDP, shard divisibility) surface on the
first multi-process run and cost GPU-hours per iteration. This spec defines
the **intermediate step**: a single-node multi-process **CPU DDP** smoke
(gloo backend) that exercises the same code paths the GPU run will use.
The GPU migration is then purely a launcher/config swap.

Scope-wise this is a narrower, actionable companion to two existing docs:

- [distributed.md](distributed.md) — the broader Accelerate + DeepSpeed + DDP
  roadmap. This spec implements only the Accelerate subset (§3 of that doc)
  plus the sharding/gathering from §5. DeepSpeed ZeRO (§4) is explicitly out
  of scope.
- [cluster_deployment.md](cluster_deployment.md) — broader cluster ops
  (NCCL, shared filesystems, checkpoint layout, W&B, preemption). This spec
  links to it rather than re-explaining any of that.

The CPU-smoke approach mirrors the "smoke tier before GPU tier" pattern
already used in [reward_model_integration.md](reward_model_integration.md)
for the RM work, so the two tracks compose cleanly: the CPU DDP smoke
exercises both the Accelerate refactor and the (pending) learned-RM
integration in one run.

## Scope

**In scope**
- `PPOTrainer.__init__` signature change from `device` to `accelerator: Accelerator`.
- Accelerate `prepare()` for policy + optimizer; conditional prepare for
  trainable critic; `.to()` for frozen reference.
- `generate_rollouts` sharded across ranks, `unwrap_model` for `.generate()`,
  `all_gather_object` to reassemble the full `RolloutBatch` before PPO update.
- `evaluate` sharded, rewards gathered via `gather_for_metrics`.
- `ppo_update`: `accelerator.backward`, `accelerator.clip_grad_norm_`.
- Rank-0 gating on every `print`, `logger.save`, `save_checkpoint` in both
  run scripts, plus `wait_for_everyone` barriers around saves.
- `checkpoint.py`: unwrap before `state_dict()`, gate writes on rank 0,
  ensure `_config_hash` stays stable across world sizes.
- New `configs/accelerate_cpu.yaml` (MULTI_CPU / gloo, 4 processes).
- `scripts/setup_env.sh --cpu-only` flag.
- `scripts/slurm_e2_7.sh` and `slurm_e2_8.sh` `DEVICE_MODE` env-var branch.
- A 5-step smoke-test recipe (single-proc CPU → 4-proc CPU DDP → CPU
  sbatch → GPU sbatch).

**Out of scope** (follow-up tickets — see §9)
- DeepSpeed ZeRO-2 (see [distributed.md §4](distributed.md#4-deepspeed-zero-integration)).
- Multi-node rendezvous (MASTER_ADDR, MASTER_PORT, machine_rank).
- W&B wiring (`use_wandb` flag is defined but never consumed).
- 8B-model path (gradient checkpointing × DDP × bf16 interactions).
- Parallelizing MC-baseline estimation (see
  [distributed.md §5.4](distributed.md#54-mc-baseline-parallelization));
  the CPU smoke uses `--no-mc`.
- SIGUSR1 preemption handler bug at
  [scripts/slurm_e2_7.sh:103-104](../../scripts/slurm_e2_7.sh#L103-L104).
- FSDP.
- Renaming `batch_size` → `global_batch_size`. Kept as-is to avoid a cascade
  through [config.py](../config.py), [utils.py](../utils.py) `cycle_batch`,
  `_config_hash`, and result-file names.

## Locked decisions

- **Topology:** single-node, 4-process CPU DDP via
  `accelerate launch --config_file configs/accelerate_cpu.yaml`, gloo backend.
  GPU migration is a launcher/config swap only.
- **Distributed library:** HuggingFace Accelerate (`accelerate.Accelerator`).
  DeepSpeed is deferred.
- **SLURM shape:** parameterize existing `scripts/slurm_e2_7.sh` and
  `scripts/slurm_e2_8.sh` via a `DEVICE_MODE` env var. No script duplication.
- **setup_env:** add `--cpu-only` flag; default stays GPU.
- **Rollout gathering:** object-level `dist.all_gather_object` on the
  `Rollout` dataclass list. Batch is tiny (≤16 items).
- **Shard divisibility:** runtime assertion immediately after `Accelerator()`
  init; `config.batch_size` must be a multiple of `accelerator.num_processes`.
- **Mixed precision on CPU:** `mixed_precision: 'no'`. bf16-on-CPU via
  Accelerate is flaky and unnecessary for a smoke.
- **Reference model placement:** stays unwrapped. Frozen, `no_grad`, no
  DDP sync needed — just `.to(accelerator.device)`.
- **Determinism invariant:** `accelerate.utils.set_seed` + identical gathered
  batches → every rank computes the same PPO loss on the same tokens; only
  rank 0 logs. 1-proc vs 4-proc `final_acc` will differ slightly because
  per-rank RNG streams diverge during generation (temperature sampling); the
  PPO loss itself is deterministic on the gathered batch. Correct, not a bug.

## Device placement strategy

The PPO step touches up to **four** transformer-sized models: policy (trains),
critic (trains, head on policy hidden states), reference (frozen, KL anchor),
and reward model (frozen, scoring). Naive replication puts all four on every
GPU and OOMs at 8B scale. The strategy below is what each rank should do
to maximize utilization without OOM.

### Default placement: data-parallel replication on every rank

Each DDP rank holds:

| Module | Placement | Memory share | Sync? |
|--------|-----------|--------------|-------|
| `policy` (trains) | rank-local GPU, wrapped by Accelerate | full bf16 weights + grads + optim | grad sync via DDP all-reduce |
| `critic` (head, trains) | rank-local GPU, wrapped by Accelerate | small (~MB) | grad sync via DDP all-reduce |
| `reference_model` (frozen) | rank-local GPU, **unwrapped** | full bf16 weights, no grads | none |
| `reward_model` (frozen, optional) | rank-local GPU, **unwrapped** | full bf16 weights, no grads | none |

This is the only placement supported by this spec. The reasoning:

1. **Frozen models replicated, not sharded.** Sharding the reference or reward
   model with FSDP/ZeRO-3 saves memory but adds an all-gather per forward
   pass. For frozen modules the all-gather is wasted bandwidth — replication
   is faster on every node that has the VRAM.

2. **Sharding for the policy is FSDP, not DDP.** This spec uses DDP. If the
   policy doesn't fit per-rank in bf16 (8B + AdamW = ~136 GB), the
   right migration is FSDP — explicitly **out of scope** here. See
   [distributed.md](distributed.md) §4.

3. **Same device for all four.** `policy`, `critic`, `reference`, and
   `reward_model` all live on `accelerator.device`. Cross-device transfers
   (e.g., reward model on a separate GPU) trigger PCIe traffic per token
   batch — slower than just running them sequentially on the same GPU.

### Per-step compute breakdown (one rank, one PPO update with K epochs)

| Phase | Module(s) used | Grad? | Sync after? |
|-------|---------------|-------|-------------|
| Generate rollouts | policy (unwrapped via `unwrap_model`) | no | `all_gather_object` on Rollout list |
| Score rewards | reward_model (or `gsm8k_reward` CPU function) | no | none (rewards are part of gathered Rollout) |
| Old log-probs | policy | no | none (per-rank cache; gathered with rollouts) |
| Critic init values | policy (hidden states) + critic | no | none |
| Reference log-probs | reference_model (frozen, unwrapped) | no | none |
| Per-rank PPO update × K | policy (grad), critic (grad) | yes | DDP all-reduce in `accelerator.backward()` |

Every rank runs the **full step** on its shard of the gathered batch. The
only cross-rank traffic is (a) the rollout gather after generation and
(b) gradient all-reduce inside `accelerator.backward()`. No model weights
move between ranks, no activations are exchanged. This is the entire
DDP contract.

### Why not put the reward model on CPU?

A common temptation: if the reward model is a separate LLM, run it on CPU
to free GPU VRAM. **Don't.** Reasons:

1. CPU inference on an 8B LLM is ~50× slower than GPU. Per-step reward
   scoring would dominate wall time.
2. The reward score has to flow back to the GPU for advantage computation
   regardless — the PCIe round trip eats most of any "savings."
3. If GPU VRAM is tight, the right answer is to use a smaller reward
   model tier (`reward_model_capacity="small"` in
   [reward_model_integration.md](reward_model_integration.md)), not
   to move it off-device.

The exception: `reward_model_capacity="none"` calls `gsm8k_reward()` which
is a deterministic CPU regex match. That is genuinely CPU work and stays
on CPU, but it's not a "model" — it's a verifier function and runs in
microseconds.

### Per-rank memory budget at 8B

With `policy=8B`, `reference=8B`, `reward_model=8B-small`, all in bf16:

```
Per-GPU bf16 weights:  3 × 16 GB = 48 GB
Policy gradients:                  16 GB
Policy AdamW states (fp32):        96 GB
Activations (B=16/rank, GC on):     ~6 GB
Critic head:                        <1 GB
Total:                            ~167 GB → does NOT fit 80 GB A100
```

The 8B + 8B-reference + 8B-RM scenario is **not feasible** under DDP on
any current single-GPU SKU. Three paths to fit:

1. **Drop the learned RM** (`reward_model_capacity="none"`), keep
   `gsm8k_reward`. Saves 16 GB. Still doesn't fit (151 GB).
2. **Drop the reference model** (`reference_kl_coeff=0`). Saves 16 GB.
   Still doesn't fit (135 GB).
3. **Switch to FSDP** to shard the policy + optimizer states across
   multiple ranks. Out of scope for this spec; see distributed.md §4.

For the CPU smoke and 0.5B GPU runs, the per-rank budget is well under 24 GB
and the default placement above is correct.

## 1. Trainer Accelerate integration

Minimum set of changes in [ppo_specs/ppo_trainer.py](../ppo_trainer.py).

### 1.1 `PPOTrainer.__init__` ([L154-L244](../ppo_trainer.py#L154-L244)) — LANDED

The signature change has been implemented as a **dual-path** rather than a
hard replacement. Today's `__init__` accepts BOTH `device: torch.device`
(legacy) and `accelerator: Accelerator` (new), and asserts exactly one is
provided ([L166-L171](../ppo_trainer.py#L166-L171)). The class derives
`self.device = accelerator.device` when an accelerator is passed.

`policy_optimizer` is built at [L212-L215](../ppo_trainer.py#L212-L215) and
the conditional `critic_optimizer` at
[L216-L223](../ppo_trainer.py#L216-L223). The prepare block lives at
[L229-L240](../ppo_trainer.py#L229-L240):

```python
self.model, self.policy_optimizer = self.accelerator.prepare(
    self.model, self.policy_optimizer,
)
if critic.is_trainable():
    self.critic, self.critic_optimizer = self.accelerator.prepare(
        self.critic, self.critic_optimizer,
    )
else:
    self.critic = self.critic.to(self.device)
```

`self.reference_model` ([L194-L210](../ppo_trainer.py#L194-L210)) stays unwrapped. Moved with
`reference_model.to(self.device)` at [L240](../ppo_trainer.py#L240); no `accelerator.prepare()`. Frozen +
no-grad → no DDP sync path. The defensive freeze assertion at [L199-L200](../ppo_trainer.py#L199-L200)
is the §7.6 invariant.

**Caveat (DDP wrapper masks methods).** After `accelerator.prepare(critic,
...)`, `critic.is_trainable()` on the wrapped module raises `AttributeError`
(DDP proxies `forward` only). The fix is in: `self._critic_trainable` is
cached at [ppo_trainer.py:187](../ppo_trainer.py#L187) before prepare. The
class also includes a `__getattr__` backstop at
[L253-L263](../ppo_trainer.py#L253-L263) so tests that bypass `__init__`
still get a working derivation. Confirm by `grep -n
'self.critic.is_trainable()' ppo_specs/ppo_trainer.py` returning zero hits
— current grep returns zero.

### 1.2 `generate_rollouts` ([L267-L389](../ppo_trainer.py#L267-L389)) — LANDED

The module-level helper `_shard_list` lives at [L99-L105](../ppo_trainer.py#L99-L105).
The shard-and-rebind block is at [L284-L297](../ppo_trainer.py#L284-L297).
The unwrap-for-`.generate()` is at [L309](../ppo_trainer.py#L309) (single-shot)
and [L424](../ppo_trainer.py#L424) (length-bucketed). The all-gather of the
Rollout list is at [L365-L369](../ppo_trainer.py#L365-L369). The global
counter increment is at [L388](../ppo_trainer.py#L388).

The three pieces below are kept for design history; they are now reference
documentation, not action items.

**(1) Shard prompts and ground truths** at function entry.

Inside `generate_rollouts`, after `global_B = len(prompts)`:

```python
global_B = len(prompts)
rank = self.accelerator.process_index
ws = self.accelerator.num_processes
local_prompts = _shard(prompts, rank, ws)
local_gts = _shard(ground_truths, rank, ws)
B = len(local_prompts)  # CRITICAL: rebind B so the inner `for i in range(B)` loop
                        # iterates over the shard, not the global batch. Forgetting
                        # this is a silent IndexError on ranks > 0.
```

Rebind the local loop to the shard (replace every `prompts[i]` / `ground_truths[i]`
in the body with the shard variables).

**(2) Unwrap for generation.** `model.generate()` hangs under DDP because
generation is autoregressive and non-deterministic; DDP expects sync forward
passes. Implemented at [ppo_trainer.py:309](../ppo_trainer.py#L309) and
[ppo_trainer.py:318-325](../ppo_trainer.py#L318-L325):

```python
unwrapped = self.accelerator.unwrap_model(self.model)
out = unwrapped.generate(
    input_ids=enc["input_ids"],
    attention_mask=enc["attention_mask"],
    ...
)
```

`_batched_per_token_log_probs`
([L473-L487](../ppo_trainer.py#L473-L487)) and `_batched_critic_values`
([L503-L510](../ppo_trainer.py#L503-L510)) keep using `self.model` — they run
`no_grad` forward under DDP and are safe. The §7.6.1 explicit `no_grad`
envelope around `_extract_last_hidden` is in at
[L542-L567](../ppo_trainer.py#L542-L567).

**(3) Gather rollouts back to every rank** before returning. Using
object-level all-gather (batches are small, pickle cost is negligible):

```python
import torch.distributed as dist

if ws > 1:
    gathered = [None] * ws
    dist.all_gather_object(gathered, rollouts)
    rollouts = [r for shard in gathered for r in shard]  # rank-ordered
```

**Import:** `all_gather_object` lives on `torch.distributed`, not on
`Accelerator`. Add `import torch.distributed as dist` to the imports at
the top of `ppo_trainer.py`. The `accelerator.gather_for_metrics()` API
only handles tensors, not Python objects.

`all_gather_object` returns shards in rank order → identically ordered
`RolloutBatch` on every rank. Subsequent log-prob and advantage computation
runs on the same gathered batch → PPO loss is deterministic across ranks.
The actual rollouts (completions, rewards) DO differ per rank because
generation uses temperature sampling under per-rank RNG; this is correct
(not a bug) — see the determinism note in §8.

**(4) Fix the rollout counter.** Done. The global increment lives at
[ppo_trainer.py:388](../ppo_trainer.py#L388):

```python
self.total_rollouts += global_B
```

Earlier the line was `+= B` (local shard size), which would have made
`total_rollouts` N-times too low. See
[distributed.md §2.2](distributed.md#22-global-mutable-state).

### 1.3 `ppo_update` ([L628-L815](../ppo_trainer.py#L628-L815)) — LANDED

Both substitutions are in. The `_is_ddp` branch chooses the Accelerate
path; the legacy single-process path is preserved verbatim.

- [ppo_trainer.py:765-769](../ppo_trainer.py#L765-L769): `total_loss.backward()` is conditional on `self._is_ddp`:
  ```python
  if total_loss.requires_grad:
      if self._is_ddp:
          self.accelerator.backward(total_loss)
      else:
          total_loss.backward()
  ```
- [ppo_trainer.py:776-791](../ppo_trainer.py#L776-L791): clip-grad-norm
  is similarly branched between `self.accelerator.clip_grad_norm_` and
  `torch.nn.utils.clip_grad_norm_` for both policy and critic.

After the gathered-rollout design (§1.2), every rank computes the same loss
on the same batch. The `.item()` values in the returned metrics dict
([L805-L815](../ppo_trainer.py#L805-L815)) are therefore identical across
ranks — no `accelerator.reduce` needed. Only rank 0 logs (gated in run
scripts, §2).

### 1.3.1 Optional: shard the PPO update for actual data-parallel speedup

**Status:** Out of scope for the smoke phase; in scope for the GPU
production phase. Track separately.

The default §1.3 design has every rank run the full B-sample PPO update
on the gathered batch — pure replication, not parallelization. On 4× A100
8B at K=4 this wastes ~16.8 s/train_step (3.5× the real work needed).
Throughput per GPU actually DECREASES with more ranks because generation
is sharded but everything else is replicated.

**Cost of the current design (every rank does full batch):**
- 4× A100 8B, K=4: ~12.0 s/step (0.27 s gen + 11.7 s replicated update)
- Speedup over 1 GPU: 1.3× (almost all from generate sharding)

**Sharded-update alternative:**
- Each rank runs `ppo_update` on rollouts `[r * shard : (r+1) * shard]`
  of the gathered batch.
- DDP all-reduce on `accelerator.backward()` reconstructs the full-batch
  gradient (each rank's gradient is for a different shard, summed).
- Advantage normalization MUST be cross-rank: replace local
  `advantages.std()` with `accelerator.reduce(local_var, "mean").sqrt()`
  before z-scoring. See `compute_advantages` in
  [ppo_specs/advantage.py](../advantage.py).

```python
# In ppo_update, after gathering rollouts:
shard_lo = rank * (B_global // ws)
shard_hi = (rank + 1) * (B_global // ws)
local_rollouts = batch.rollouts[shard_lo:shard_hi]

# Compute local mean and std for advantage normalization
local_adv = ...
local_mean = local_adv.mean()
local_var = local_adv.var(unbiased=False)
global_mean = accelerator.reduce(local_mean, reduction="mean")
global_var = accelerator.reduce(local_var, reduction="mean")
global_std = global_var.sqrt()
advantages_normalized = (local_adv - global_mean) / (global_std + eps)
```

**Speedup:** at 4× A100 8B K=4, ~12.0 s/step → ~5.2 s/step (2.3× speedup,
or 3.0× over 1-GPU). Adds ~5 ms/step in cross-rank reductions.

**Add a config flag** `ppo_update_sharded: bool = False` (default
preserves §1.3 behavior). Add a parity test: 1-proc result vs
2-proc-sharded result, gradient cosine similarity > 0.999 over fixed
seed.

Track in §10 follow-ups; do not block smoke on this.

### 1.4 `evaluate` ([L954-L1035](../ppo_trainer.py#L954-L1035)) — LANDED

The shard-with-padding-for-divisibility logic lives at
[L972-L984](../ppo_trainer.py#L972-L984). Note this implementation goes a
step beyond the original spec by **padding `eval_prompts` up to a multiple
of `world_size`** (using duplicates from the head of the eval set) before
sharding. Without that, when `n_eval % ws != 0` the trailing samples
silently disappeared — the bug is fixed and the duplicates are trimmed
after `gather_for_metrics` at [L1032-L1033](../ppo_trainer.py#L1032-L1033).

The `set_questions(eval_prompts)` call at
[L990-L991](../ppo_trainer.py#L990-L991) covers the §7.6.2 self-judge
sharding hazard inside `evaluate`. Unwrap-for-generate is at
[L994](../ppo_trainer.py#L994). The tensor-level reward gather is at
[L1026-L1034](../ppo_trainer.py#L1026-L1034).

### 1.5 `load_ppo_trainer` ([L1040-L1197](../ppo_trainer.py#L1040-L1197)) — LANDED (with caveats)

Signature: `device_or_accelerator: Union[torch.device, Accelerator]`. Detection is
isinstance-based at [L1054-L1062](../ppo_trainer.py#L1054-L1062). The
function works in both legacy (device) and DDP (accelerator) modes.

- Policy `.to(device)` is gated by `not is_ddp` at [L1100](../ppo_trainer.py#L1100); under DDP, Accelerate's `prepare()` handles placement.
- Critic `.to(device)` similarly gated at [L1115-L1116](../ppo_trainer.py#L1115-L1116).
- `reference_model.to(device)` at [L1168](../ppo_trainer.py#L1168) — stays unwrapped on every rank.
- `accelerator.main_process_first()` HF-cache warming at
  [L1079-L1099](../ppo_trainer.py#L1079-L1099) and
  [L1152-L1158](../ppo_trainer.py#L1152-L1158). Avoids cold-cache thrash
  when N ranks hit HuggingFace simultaneously.
- `torch_dtype` resolution at [L1064-L1070](../ppo_trainer.py#L1064-L1070).

**Caveat — print gates not applied inside `load_ppo_trainer`.** The startup
`print` at [L1072](../ppo_trainer.py#L1072), the gradient-checkpointing
print at [L1107](../ppo_trainer.py#L1107), the hidden-size print at
[L1110-L1111](../ppo_trainer.py#L1110-L1111), and the reference-model print
at [L1125-L1126](../ppo_trainer.py#L1125-L1126) are NOT gated on
`accelerator.is_main_process`. Under multi-process Accelerate, each line
prints `world_size` times. This is non-fatal (Accelerate runs serial loads
through `main_process_first`, so the duplicates are temporally serialized,
not garbled), but it clutters logs. Phase 2 cleanup task: thread an
`accelerator` argument into the prints, or switch to a module-level
`logger.info` with `set_main_process` rank-0 filtering. Track in
**DDP-CLEAN** bead (low priority).

## 2. Run-script integration

### 2.1 `run_e2_7.py` — LANDED

The Accelerate boilerplate at the top of `run_e2_7()`
([run_e2_7.py:55-96](../run_e2_7.py#L55-L96)) implements an opt-in `USE_DDP =
"LOCAL_RANK" in os.environ` pattern: legacy single-process behavior is
preserved exactly when launched as plain `python run_e2_7.py`, and DDP
engages only under `accelerate launch` / `torchrun`. The
`set_seed`, divisibility-assert, and helpers `_is_main`/`_print`/`_wait`
all live in this block.

Pass to `load_ppo_trainer` at [run_e2_7.py:109-111](../run_e2_7.py#L109-L111):

```python
trainer, diagnostic_fn = load_ppo_trainer(
    config, accelerator if accelerator is not None else device,
)
```

**Rank-0 gates — current state.** All side-effecting lines below are
already guarded via the `_print()` / `_is_main()` helpers (functionally
equivalent to `if accelerator.is_main_process:`):

| File:line | Kind | Status |
|-----------|------|--------|
| [run_e2_7.py:97](../run_e2_7.py#L97) | print "Device" | gated via `_print` |
| [run_e2_7.py:100](../run_e2_7.py#L100) | print "Loading GSM8K" | gated |
| [run_e2_7.py:128](../run_e2_7.py#L128) | print "Resuming from" | gated |
| [run_e2_7.py:169](../run_e2_7.py#L169) | print "Resumed at step" | gated |
| [run_e2_7.py:182](../run_e2_7.py#L182) | print "Estimating MC baselines" | gated AND rank-0-only computed (broadcast) |
| [run_e2_7.py:191](../run_e2_7.py#L191) | print "MC baselines" | gated |
| [run_e2_7.py:230-238](../run_e2_7.py#L230-L238) | per-step training print | gated |
| [run_e2_7.py:310-317](../run_e2_7.py#L310-L317) | eval summary print + `logger.log_step` + `logger.save` | gated under `if _is_main():` |
| [run_e2_7.py:320-326](../run_e2_7.py#L320-L326) | periodic `save_checkpoint` | `_wait()` then `_is_main()` then save |
| [run_e2_7.py:338-348](../run_e2_7.py#L338-L348) | graceful-exit save | `_wait()` then `_is_main()` then save + `logger.save` then `_wait()` |
| [run_e2_7.py:351-360](../run_e2_7.py#L351-L360) | final `save_checkpoint` + `logger.save` | `_wait()` then `_is_main()` |
| [run_e2_7.py:366-367](../run_e2_7.py#L366-L367) | final print | gated |

**Barriers before every checkpoint save — LANDED.** Each of the three
`save_checkpoint` calls in `run_e2_7.py` is preceded by `_wait()` (which
calls `accelerator.wait_for_everyone()` when an accelerator exists):

| save_checkpoint call | barrier site |
|-----|-----|
| [L323-L326](../run_e2_7.py#L323-L326) (periodic) | `_wait()` at [L321](../run_e2_7.py#L321) |
| [L342-L345](../run_e2_7.py#L342-L345) (graceful exit) | `_wait()` at [L339](../run_e2_7.py#L339) and tail `_wait()` at [L347](../run_e2_7.py#L347) |
| [L354-L357](../run_e2_7.py#L354-L357) (final) | `_wait()` at [L352](../run_e2_7.py#L352) |

**Note on the audit's "barrier count: 2" finding.** A `grep -n
'wait_for_everyone' run_e2_7.py` returns only 1 hit (inside the `_wait`
helper definition). Counting effective barriers reaching the
`accelerator.wait_for_everyone()` call requires counting `_wait()`
invocations: there are **4 of those** (L321, L339, L347, L352). The
audit's literal-grep methodology underreports; the runtime barrier
count is correct.

**CRITICAL barrier rule (kept for design history):** every call to
`save_checkpoint(...)` MUST be preceded by `accelerator.wait_for_everyone()`
in the CALLER (run script). `save_checkpoint` itself returns early on
non-rank-0 processes (Agent D contract — see
[checkpoint.py:50-63](../checkpoint.py#L50-L63) docstring). If the barrier
is placed *inside* `save_checkpoint` instead of in the caller, non-rank-0
ranks return before reaching it and the program deadlocks. Pattern:

```python
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    save_checkpoint(trainer, step, config, logger, ckpt_dir, ...)
```

### 2.2 `run_e2_8.py` — LANDED (note: no save_checkpoint calls today)

Same pattern as run_e2_7. Implementation in `run_e2_8()`
([run_e2_8.py:215-357](../run_e2_8.py#L215-L357)):

- [run_e2_8.py:216-235](../run_e2_8.py#L216-L235): USE_DDP branch with
  `Accelerator()` + `set_seed`.
- [run_e2_8.py:222-225](../run_e2_8.py#L222-L225): `batch_size %
  num_processes` divisibility assert.
- [run_e2_8.py:312-321](../run_e2_8.py#L312-L321): in-loop per-capacity
  re-seed using `accelerate_set_seed`.
- [run_e2_8.py:237-246](../run_e2_8.py#L237-L246): `_is_main`/`_print`/`_wait` helpers.

Rank-0 gates on `run_e2_8.py` — current state:

| File:line | Kind | Status |
|-----------|------|--------|
| [run_e2_8.py:99-101](../run_e2_8.py#L99-L101) (in `run_one_capacity`) | capacity banner print | gated via `_print` |
| [run_e2_8.py:189-193](../run_e2_8.py#L189-L193) | per-step training print | gated |
| [run_e2_8.py:195-196](../run_e2_8.py#L195-L196) | `logger.save` | `if _is_main():` |
| [run_e2_8.py:202](../run_e2_8.py#L202) | per-capacity final-accuracy print | gated |
| [run_e2_8.py:248](../run_e2_8.py#L248) | "Device" print | gated |
| [run_e2_8.py:251](../run_e2_8.py#L251) | "Loading GSM8K" print | gated |
| [run_e2_8.py:278](../run_e2_8.py#L278) | "Estimating MC baselines" | gated, rank-0 compute only at L280-L288, broadcast at L289-L293 |
| [run_e2_8.py:306](../run_e2_8.py#L306) | "MC baselines" print | gated |
| [run_e2_8.py:341-345](../run_e2_8.py#L341-L345) | summary file write | `_wait()` at L340, `if _is_main():` then write |
| [run_e2_8.py:347-357](../run_e2_8.py#L347-L357) | summary print | gated |

`run_e2_8.py` has no `save_checkpoint` calls (E2.8 does not yet support
resume; see the `--resume-from` warning at
[run_e2_8.py:436-440](../run_e2_8.py#L436-L440)). When checkpoint support
is added later, add `_wait()` before each save and gate the call on
`_is_main()`, mirroring `run_e2_7.py:321-326`.

### 2.3 Saved-rollout-count dance

Both run scripts save and restore `trainer.total_rollouts` around the
per-capacity reference-rollout eval:
[run_e2_7.py:280-282](../run_e2_7.py#L280-L282) and
[run_e2_8.py:158-162](../run_e2_8.py#L158-L162). Because
`generate_rollouts` increments by the global (pre-shard) count
([ppo_trainer.py:388](../ppo_trainer.py#L388)), these sites still work
unchanged — `saved_rollouts` captures the pre-call global count;
`trainer.total_rollouts = saved_rollouts` restores the global count.
No edits needed here, but this is easy to break in review — call it out.

## 3. Config + divisibility

[ppo_specs/config.py](../config.py). No rename of `batch_size`. The field
is at [config.py:65](../config.py#L65) (currently `batch_size: int = 8`
on the dataclass default; `e2_7_config` overrides to 16 at
[config.py:174](../config.py#L174)). A clarifying comment is welcome:

```python
batch_size: int = 8  # GLOBAL batch size; must be divisible by Accelerator.num_processes
```

Existing values already satisfy the constraint:

| Config | `batch_size` | Divisibility |
|--------|-------------|--------------|
| `local_test_config` | 4 | ok for 1 / 2 / 4 procs |
| `e2_7_config` | 16 | ok for 1 / 2 / 4 / 8 procs |

The runtime assertion (§2.1, §2.2) catches anything else at startup.

### 3.1 `cycle_batch` under DDP (no change required, but worth documenting)

`cycle_batch` in [ppo_specs/utils.py](../utils.py) is reached from the training
loops at [run_e2_7.py:208-209](../run_e2_7.py#L208-L209) and
[run_e2_8.py:119-120](../run_e2_8.py#L119-L120). It is deterministic given
`(step, batch_size, seed)`, so every rank produces an identical global batch;
the sharding happens *inside* `generate_rollouts` (§1.2). No DDP-aware variant
of `cycle_batch` is needed, and no DataLoader wrapping. If you later switch to
a randomized sampler here, you will need to reseed per-rank identically or
shard the sampler — out of scope for this spec.

## 4. New file: `configs/accelerate_cpu.yaml`

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_CPU
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
use_cpu: true
```

`MULTI_CPU` selects gloo as the distributed backend (no NCCL dependency).
`num_processes: 4` is the default; `accelerate launch --num_processes N`
overrides at submission time.

Place alongside the existing
[configs/accelerate_single_gpu.yaml](../../configs/accelerate_single_gpu.yaml)
and [configs/accelerate_multi_gpu.yaml](../../configs/accelerate_multi_gpu.yaml)
(described in
[cluster_deployment.md §3](cluster_deployment.md#3-accelerate-configurations)).

## 5. SLURM parameterization

`#SBATCH` directives cannot reference runtime env vars at submission time
— the scheduler parses them before the shell runs. Strategy: leave the
`#SBATCH` block as the GPU default; branch in the script body on
`DEVICE_MODE`; users override scheduler-side resources via `sbatch` flags
(`--gres=none --partition=cpu --export=...`).

Add near the top of
[scripts/slurm_e2_7.sh](../../scripts/slurm_e2_7.sh) and
[scripts/slurm_e2_8.sh](../../scripts/slurm_e2_8.sh), just after the
existing parameter block (slurm_e2_7.sh:40-50, slurm_e2_8.sh:40-51):

```bash
DEVICE_MODE="${DEVICE_MODE:-gpu}"                  # gpu | cpu
NUM_PROCESSES="${NUM_PROCESSES:-4}"                # accelerate --num_processes

if [ "$DEVICE_MODE" = "cpu" ]; then
    ACCEL_CONFIG="configs/accelerate_cpu.yaml"
    # skip cuda module load, skip NCCL_* exports
else
    ACCEL_CONFIG="${ACCEL_CONFIG:-configs/accelerate_multi_gpu.yaml}"
    module load cuda/12.1 2>/dev/null || true
    # existing NCCL_* exports below
fi
```

Gate the existing `module load cuda/12.1` at
[slurm_e2_7.sh:72](../../scripts/slurm_e2_7.sh#L72) and the NCCL block at
[slurm_e2_7.sh:86-93](../../scripts/slurm_e2_7.sh#L86-L93) on
`DEVICE_MODE != cpu`. The equivalents in `slurm_e2_8.sh` are
[slurm_e2_8.sh:75](../../scripts/slurm_e2_8.sh#L75) (`module load cuda`)
and [slurm_e2_8.sh:88-93](../../scripts/slurm_e2_8.sh#L88-L93) (NCCL block).

Replace the existing command-build branches at
[slurm_e2_7.sh:137-142](../../scripts/slurm_e2_7.sh#L137-L142) (single-GPU,
direct python) and at
[slurm_e2_8.sh:152-153](../../scripts/slurm_e2_8.sh#L152-L153) (per-capacity
direct python in `run_capacity()`) with a unified accelerate launch:

```bash
accelerate launch \
    --config_file "$ACCEL_CONFIG" \
    --num_processes "$NUM_PROCESSES" \
    ppo_specs/run_e2_7.py ${ARGS} &
CHILD_PID=$!
wait $CHILD_PID
```

Single-GPU is handled by `accelerate_single_gpu.yaml`
(`distributed_type: 'NO'`, `num_processes: 1`) — no code difference from
multi-GPU.

### 5.1 Submission recipes

CPU sbatch (smoke on the cluster):

```bash
sbatch --gres=none --partition=cpu \
    --export=ALL,DEVICE_MODE=cpu,LOCAL_TEST=true,NUM_PROCESSES=4 \
    scripts/slurm_e2_7.sh
```

GPU sbatch (existing recipe, unchanged):

```bash
sbatch --gres=gpu:4 \
    --export=ALL,SLURM_MODE=multigpu,NUM_GPUS=4 \
    scripts/slurm_e2_7.sh
```

### 5.2 Runtime sanity warning

If `DEVICE_MODE=cpu` but `SLURM_JOB_GRES` mentions `gpu`, the user has
allocated an unused GPU. Print a warning near the top of the script body
after the `DEVICE_MODE` branch:

```bash
if [ "$DEVICE_MODE" = "cpu" ] && [[ "${SLURM_JOB_GRES:-}" == *gpu* ]]; then
    echo "[WARN] DEVICE_MODE=cpu but SLURM_JOB_GRES=${SLURM_JOB_GRES} — wasting a GPU."
fi
```

### 5.3 Unrelated known bug

The SIGUSR1 handler at
[slurm_e2_7.sh:109-118](../../scripts/slurm_e2_7.sh#L109-L118) actually
DOES forward the signal to `$CHILD_PID` via `kill -SIGUSR1` at L111. The
older "doesn't forward" finding from `cluster_deployment.md §9` predates
the fix. The remaining concern is that under multi-GPU (`accelerate launch`),
`$CHILD_PID` is the launcher PID, not the per-rank Python PID, so the
signal may need to walk a process tree — verify behavior on the cluster
before declaring this fully resolved.

## 6. `setup_env.sh --cpu-only` — NOT DONE (Agent A still owes this)

Changes to [scripts/setup_env.sh](../../scripts/setup_env.sh):

- Initialize near [L26](../../scripts/setup_env.sh#L26) (after the
  `MODEL_CACHE` default): `CPU_ONLY=false`.
- Parser at [L29-L39](../../scripts/setup_env.sh#L29-L39): add
  `--cpu-only) CPU_ONLY=true; shift ;;`.
- Step 2 at [L71-L95](../../scripts/setup_env.sh#L71-L95): if
  `CPU_ONLY=true`, set `TORCH_INDEX="https://download.pytorch.org/whl/cpu"`
  unconditionally (bypass the case statement at L76-L82); skip the CUDA
  verification block at
  [L87-L95](../../scripts/setup_env.sh#L87-L95); print only
  `torch.__version__`.
- Step 4 (model pre-download): unchanged. `from_pretrained` is
  device-agnostic.
- Footer banner at [L174-L191](../../scripts/setup_env.sh#L174-L191): when
  `CPU_ONLY=true`, print the smoke-verification command
  `accelerate launch --config_file configs/accelerate_cpu.yaml ppo_specs/run_e2_7.py --local-test --no-mc`
  so the user knows how to verify.

## 7. Smoke-test recipe (runbook)

Run these in order. Each step gates the next.

### 7.1 Single-process CPU baseline

```bash
PYTHONUTF8=1 python ppo_specs/run_e2_7.py --local-test --no-mc
```

Purpose: verify the refactor did not break the solo path.
`Accelerator()` with no launcher reports `num_processes=1`. `assert
batch_size % 1 == 0` passes; `_shard` returns the full list; no
`all_gather_object` call is taken (gated on `ws > 1`).

**Pass criteria:** 5 steps complete; `final_acc` printed; no hang;
`total_rollouts` reaches `batch_size * n_steps`.

### 7.2 4-process CPU DDP (primary target)

```bash
PYTHONUTF8=1 accelerate launch --config_file configs/accelerate_cpu.yaml \
    ppo_specs/run_e2_7.py --local-test --no-mc
```

Purpose: exercise every distributed code path — shard, unwrap, generate,
gather, backward, clip, unwrap-for-eval, gather-for-metrics, rank-0 gating,
wait_for_everyone.

**Pass criteria:**
- Only rank 0 prints (no duplicated lines).
- No hang on `.generate()` (the classic DDP-hang failure mode).
- No `AttributeError` on `critic.is_trainable()` (the DDP-wrapper-masks-methods
  failure mode).
- `final_acc` within numerical noise of 7.1's value. Exact equality is not
  expected (per-rank RNG streams advance independently).
- `results/*.json` written once by rank 0 and well-formed.

### 7.3 E2.8 4-process CPU DDP

```bash
PYTHONUTF8=1 accelerate launch --config_file configs/accelerate_cpu.yaml \
    ppo_specs/run_e2_8.py --local-test
```

Purpose: sweep path works across capacity swaps (the `gc.collect()` +
`empty_cache` dance at
[run_e2_8.py:222-228](../run_e2_8.py#L222-L228) holds up under DDP).

**Pass criteria:** sweep completes; summary file written once; per-capacity
metrics consistent across ranks.

### 7.4 CPU cluster sbatch

```bash
sbatch --gres=none --partition=cpu \
    --export=ALL,DEVICE_MODE=cpu,LOCAL_TEST=true,NUM_PROCESSES=4 \
    scripts/slurm_e2_7.sh
```

Purpose: validate the SLURM wrapper branches correctly on `DEVICE_MODE=cpu`
and avoids CUDA module loads / NCCL exports.

**Pass criteria:** job runs to completion; stdout shows
`[RUN] accelerate launch --config_file configs/accelerate_cpu.yaml ...`;
no `module load cuda` in the log; no NCCL_* warnings.

### 7.5 GPU cluster sbatch (final)

```bash
sbatch --export=ALL,SLURM_MODE=multigpu,NUM_GPUS=4 scripts/slurm_e2_7.sh
```

Purpose: prove the CPU smoke was sufficient — no code change between
7.4 and 7.5, only launcher config.

**Pass criteria:** multi-GPU run completes 5 steps (or 200 for full); NCCL
initializes cleanly; GPU utilization ~uniform across ranks (see
`nvidia-smi`). If this fails with a bug that did not surface in 7.2, that
is a gap in the smoke coverage — report back for follow-up.

## 7.6 Critical correctness hazards (deep review 2026-04-30; LANDING STATUS 2026-05-04)

These hazards were identified by a PhD-level deep review and are NOT covered
by the initial four critical pitfalls (B-rebind, barrier, dist import,
frozen-ref assertion). Each is a silent correctness bug or a deadlock
risk that the smoke-test recipe in §7 may miss. **Status: 5 of 5
mandatory hazards are LANDED in the current code; 2 additional advisory
items remain.** Each subsection below has an updated status marker.

### 7.6.1 `_extract_last_hidden` LM forward NOT in `torch.no_grad()` — LANDED

**Location:** [ppo_trainer.py:514-571](../ppo_trainer.py#L514-L571) inside
`_extract_last_hidden`. The method has `@torch.no_grad()` decorator, but
when called from `_critic_forward` (`ppo_trainer.py:850`) DURING `ppo_update`,
the wrapped `self.model(...)` runs through DDP's reducer-tracked forward.
DDP's reducer expects every model forward in a "training step window" to
either be inside `torch.no_grad()` (skipped by the reducer) or be followed
by a backward (registered for gradient sync). Three forwards per epoch
(critic-extract, optional self-judge ref, policy-with-grad) without explicit
`no_grad` envelopes can trip `RuntimeError: Expected to mark a variable
ready only once`.

**Fix:** wrap every non-grad call site that uses `self.model` in
`with torch.no_grad():` explicitly, even if the called method already has
the decorator. The decorator is bypassed when DDP's hooks fire on the
forward call itself.

**Status:** the explicit `with torch.no_grad():` envelope is in at
[ppo_trainer.py:542-567](../ppo_trainer.py#L542-L567) inside
`_extract_last_hidden`. The call sites that route through it
(`_critic_forward` at [L850](../ppo_trainer.py#L850), `_batched_critic_values`
at [L509](../ppo_trainer.py#L509), `_eval_critic_on_prompts` at
[L622](../ppo_trainer.py#L622)) inherit the no_grad envelope. The other
no_grad-wrapped sites in `train_step` (e.g., reference log probs at
[L904-L909](../ppo_trainer.py#L904-L909), old log probs at
[L890-L894](../ppo_trainer.py#L890-L894), critic init at
[L879-L880](../ppo_trainer.py#L879-L880)) all have explicit `with torch.no_grad():`.

### 7.6.2 Self-judge reward indexing under sharding — LANDED

**Location:** [src/rewards.py:201-239](../../src/rewards.py#L201-L239)
`_RewardFnWrapper` uses an internal `self._idx` that increments per call
to pair each completion with the corresponding question. Under DDP, each
rank generates a SHARD of completions but `set_questions(batch_p)` was
historically called with the GLOBAL batch. On rank > 0, the local
completion at index `i` corresponds to the global question at index
`rank * shard_size + i`, but `_idx` starts at 0 and reads the wrong
question.

**Status:** fixed. `set_questions(local_p)` is called with the LOCAL shard
in three places: [run_e2_7.py:216-224](../run_e2_7.py#L216-L224) (training
loop), [run_e2_8.py:127-135](../run_e2_8.py#L127-L135) (E2.8 training),
and [ppo_trainer.py:990-991](../ppo_trainer.py#L990-L991) (inside
`evaluate`).

**Symptom (now fixed):** `reward_mode = "self_judge"` or `"combined"` runs
produce silently wrong rewards on every rank > 0. Deterministic mode does
not trigger the bug (no question lookup).

**Fix that landed:** call `set_questions(local_prompts)` AFTER sharding,
not before. See LANDED status above.

### 7.6.3 Resume-across-world-sizes silently bypasses DDP wrapping — LANDED

**Location:** [run_e2_7.py:136-156](../run_e2_7.py#L136-L156)
```python
trainer.model = AutoModelForCausalLM.from_pretrained(
    state["model_path"], torch_dtype=...,
).to(device)
```
This reassigns `trainer.model` to a fresh, unwrapped `AutoModelForCausalLM`,
overwriting the DDP-wrapped module set up by `accelerator.prepare()` in
`PPOTrainer.__init__`. Subsequent `accelerator.backward()` calls then run
through the unwrapped model — no all-reduce, ranks diverge silently.

**Fix (LANDED):** the re-prepare block is at
[run_e2_7.py:153-156](../run_e2_7.py#L153-L156):

```python
if accelerator is not None:
    trainer.model, trainer.policy_optimizer = accelerator.prepare(
        trainer.model, trainer.policy_optimizer,
    )
```

`run_e2_8.py` does not yet support resume (warning at
[run_e2_8.py:436-440](../run_e2_8.py#L436-L440)). When resume is added
there, mirror the run_e2_7 pattern.

### 7.6.4 GracefulExitHandler can deadlock on collective-mismatch — LANDED

**Location:** [checkpoint.py GracefulExitHandler](../checkpoint.py#L246-L282) +
[run_e2_7.py:328-348](../run_e2_7.py#L328-L348). The signal handler sets
`should_exit = True` on the rank that received SIGTERM/SIGINT. If only
rank 0 gets the signal (common with terminal Ctrl-C → only the foreground
process), rank 0 jumps to `save_checkpoint` while ranks 1..N continue
into the next training step's collective ops (`generate`, `all_gather_object`).
Rank 0 reaches `accelerator.wait_for_everyone()` (per §2.1 fix); ranks 1..N
are inside `dist.all_gather_object` waiting for rank 0 to join. Deadlock
until gloo timeout (~30 min default).

**Fix (LANDED):** the all-reduce + barrier is at
[run_e2_7.py:330-348](../run_e2_7.py#L330-L348). Snippet:

```python
if accelerator is not None:
    exit_flag = torch.tensor(int(exit_handler.should_exit), device=device)
    exit_flag = accelerator.reduce(exit_flag, reduction="max")
    should_exit_all = bool(exit_flag.item())
else:
    should_exit_all = exit_handler.should_exit

if should_exit_all:
    _wait()
    if _is_main():
        save_checkpoint(...)
        logger.save()
    _wait()
    return
```

### 7.6.5 MC baseline duplicated on every rank in production GPU run — LANDED

**Locations (LANDED):** [run_e2_7.py:174-196](../run_e2_7.py#L174-L196) and
[run_e2_8.py:275-293](../run_e2_8.py#L275-L293). Both gate the MC compute
on `_is_main()` and use `accelerate.utils.broadcast_object_list` to
distribute the dict to all ranks.

### 7.6.6 (additional) bf16 weights × `mixed_precision: bf16` has no fp32 master — ADVISORY

**Location:** [load_ppo_trainer:1064-1099](../ppo_trainer.py#L1064-L1099)
loads model directly as bf16 when `torch_dtype=auto` and device is CUDA.
The `accelerate_multi_gpu.yaml` config sets `mixed_precision: bf16`. The
combination: model weights are bf16, no fp32 master copy is kept, AdamW
applies bf16 updates to bf16 weights — accumulating roundoff over many
steps. Acceptable for 0.5B/200 steps; problematic for 8B/long runs.

**Fix (smoke phase, leave as-is):** the 0.5B GPU smoke is unaffected.

**Fix (8B follow-up):** either set `mixed_precision: 'no'` in
`accelerate_multi_gpu.yaml` (model already bf16, no autocast needed), OR
load model as fp32 and rely on Accelerate's bf16 autocast (TRL convention,
fp32 master, doubles weight memory). Track in §10.

Add to §10: `bf16 master-weight policy for 8B+ runs — explicit choice
between "load bf16 + mixed_precision=no" and "load fp32 + mixed_precision=bf16".`

### 7.6.7 (additional) DDP reducer state across multiple forward passes per step — ADVISORY

`ppo_update` runs three forward passes through `self.model` in one step:
critic-extract (no_grad), old-log-probs (no_grad), new-log-probs (with grad).
DDP's reducer expects exactly ONE registration of parameters per step. The
explicit `no_grad` envelopes (per §7.6.1) ensure passes 1 and 2 do not
register; only pass 3 does. Verify with a unit test:

```python
# After one ppo_update on the 4-proc CPU smoke:
trainer.policy_optimizer.zero_grad()
metrics = trainer.ppo_update(batch)
for name, p in trainer.model.named_parameters():
    assert p.grad is not None, f"param {name} got no gradient"
```

Add to §9: `[ ] After one ppo_update on a CPU 4-proc smoke, every policy
parameter has a non-None .grad (no DDP-find_unused-error). Set
find_unused_parameters=False explicitly via DistributedDataParallelKwargs
to fail fast if violated.`

## 8. Risks and gotchas

| Risk | Mitigation |
|------|-----------|
| gloo CPU `.generate()` is slow. `local_test_config` at 4 procs × batch 4 × 256 tokens × 5 steps ≈ 5–15 min wall time. | Expected. Do not conflate slowness with a hang. Set `NCCL_TIMEOUT` equivalent for gloo: `TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800`. |
| `all_gather_object` pickles full `Rollout` dataclasses including `full_ids`. At batch 16 × 256 tokens → a few KB per rank. | Fine for current scale. Revisit at thousands of rollouts; switch to tensor-level gather with fixed max length. |
| `total_rollouts` off-by-N regression. (LANDED at [L388](../ppo_trainer.py#L388): `+= global_B`.) | Fix landed; add an assertion in a unit test if not already present. |
| Saved-rollout-count dance at [run_e2_7.py:280-282](../run_e2_7.py#L280-L282) and [run_e2_8.py:158-162](../run_e2_8.py#L158-L162) | With the §1.2 fix, `saved_rollouts` is a global count; restoration is symmetric. Flag in code review. |
| 1-proc vs 4-proc `final_acc` differ slightly. | Correct, with scope: per-rank RNG streams diverge during generation (temperature sampling), so each rank produces different completions → different rewards → different `train_accuracy`/`test_accuracy`. The PPO loss itself IS deterministic on the gathered batch (every rank computes log-probs and advantages on the same tokens). So gradient updates are synchronized, but the trajectories sampled differ. Not a bug. |
| 4-process CPU memory footprint. | 4 procs × (model + critic + reference_model) all in CPU RAM. For 0.5B fp32 + reference: ~16-20 GB. If OOM on a laptop, set `reference_kl_coeff=0` to avoid loading the reference model, or run with `num_processes=2` instead of 4. |
| `critic.is_trainable()` after `accelerator.prepare(critic, ...)` raises `AttributeError` because the DDP wrapper proxies only `forward`. | LANDED. `self._critic_trainable` cached at [L187](../ppo_trainer.py#L187) BEFORE prepare; backstop at [L253-L263](../ppo_trainer.py#L253-L263). |
| Logger divergence on non-rank-0 processes. | Keep `ExperimentLogger` on every rank (code simpler); only rank 0 calls `.save()`. Non-rank-0 in-memory state is garbage but never written — dead code under the gate. |
| Reference model inside `_batched_per_token_log_probs(..., model_override=self.reference_model)` | Runs under `no_grad()` on the unwrapped frozen module. No DDP sync issue, since no gradients flow. Leave as-is. |
| `_config_hash` helper at [checkpoint.py:210-230](../checkpoint.py#L210-L230) (call site at [checkpoint.py:97](../checkpoint.py#L97), validation at [L147](../checkpoint.py#L147)) | Hashes the full serialized config; `batch_size` field is the global value → hash stable across world sizes. Resuming a 4-proc run from a 1-proc checkpoint hashes identically. |
| `debug: true` in `accelerate_cpu.yaml` slows startup by seconds and litters stdout. | Keep `debug: false` (§4). Flip only when actively debugging. |

## 9. Verification checklist

- [ ] `configs/accelerate_cpu.yaml` exists, `distributed_type: MULTI_CPU`.
- [ ] `scripts/setup_env.sh --cpu-only` installs CPU-only torch and skips CUDA verification.
- [ ] `PPOTrainer.__init__` accepts `accelerator: Accelerator`; no `device` kwarg.
- [ ] `self._critic_trainable` is set before `prepare()`; no downstream call to `critic.is_trainable()` remains in `ppo_trainer.py`.
- [ ] `generate_rollouts` shards inputs, uses `accelerator.unwrap_model(...).generate(...)`, and all-gathers the `Rollout` list.
- [ ] `generate_rollouts` increments `total_rollouts` by the pre-shard global count, not the local shard size.
- [ ] `ppo_update` uses `accelerator.backward` and `accelerator.clip_grad_norm_`.
- [ ] `evaluate` shards, unwraps for generate, and uses `gather_for_metrics` on the reward tensor.
- [ ] `load_ppo_trainer` accepts `accelerator`; no `.to(accelerator.device)` on `model` or `critic` (Accelerate places them).
- [ ] `reference_model` still gets `.to(accelerator.device)` explicitly — unwrapped.
- [ ] `reference_model` parameters all have `requires_grad=False`; assert this in `__init__` to catch accidental wrapping.
- [ ] `run_e2_7.py` and `run_e2_8.py` construct `Accelerator()` once, call `set_seed`, assert divisibility, and gate every print / logger.save / save_checkpoint on `accelerator.is_main_process`.
- [ ] `accelerator.wait_for_everyone()` precedes every `save_checkpoint` and summary-write.
- [ ] `checkpoint.py` unwraps policy and critic before `state_dict()`; only rank 0 writes.
- [ ] `scripts/slurm_e2_7.sh` / `slurm_e2_8.sh` branch on `DEVICE_MODE`; gloo path skips cuda module load and NCCL_* exports.
- [ ] Smoke 7.1 passes (single-proc CPU baseline).
- [ ] Smoke 7.2 passes (4-proc CPU DDP on `run_e2_7`).
- [ ] Smoke 7.3 passes (4-proc CPU DDP on `run_e2_8`).
- [ ] Smoke 7.4 passes (CPU sbatch).
- [ ] Smoke 7.5 passes (GPU sbatch) — no code change from 7.4.
- [ ] `results/*.json` written exactly once per run (by rank 0), well-formed.
- [ ] No duplicated stdout lines on multi-proc runs.

## 10. Out of scope (follow-ups)

- DeepSpeed ZeRO-2 integration — see [distributed.md §4](distributed.md#4-deepspeed-zero-integration).
- Multi-node rendezvous — see [cluster_deployment.md §6](cluster_deployment.md#6-nccl-and-multi-node-networking).
- W&B wiring (`use_wandb` config flag is defined but unused; logger wrapper described in [cluster_deployment.md §8](cluster_deployment.md#8-logging-and-wandb)).
- 8B-model path: gradient checkpointing under Accelerate + bf16 + DDP interactions. Depends on the RM work in [reward_model_integration.md](reward_model_integration.md) reaching the `large` tier.
- MC-baseline parallelization — see [distributed.md §5.4](distributed.md#54-mc-baseline-parallelization). The CPU smoke uses `--no-mc` so this does not block.
- SIGUSR1 preemption handler at [slurm_e2_7.sh:109-118](../../scripts/slurm_e2_7.sh#L109-L118): forwards to `$CHILD_PID` (the bare-python OR `accelerate launch` PID). Under `accelerate launch`, verify the signal walks the process tree to the actual rank-0 worker. Track separately.
- FSDP config for 8B+ — see [cluster_deployment.md §3](cluster_deployment.md#3-accelerate-configurations) (FSDP template) and [distributed.md §4.1](distributed.md#41-zero-2-vs-zero-3).
