# Cluster Runbook

Step-by-step guide for verifying the DDP integration locally and then
deploying to the GPU cluster. **Read once before launching.**

---

## 0. Prerequisites

```bash
# Verify rlvr conda env has accelerate
conda activate rlvr
python -c "import accelerate; print(accelerate.__version__)"  # expect ≥ 1.0
```

The codebase has three execution modes; the script auto-detects via the
`LOCAL_RANK` environment variable (set by `accelerate launch` and `torchrun`):

| Mode | Trigger | Code path |
|------|---------|-----------|
| **Single-process legacy** | `python ppo_specs/run_e2_7.py` (no LOCAL_RANK) | Uses `device=cpu/cuda`; identical to pre-2026-04 behavior |
| **Single-process Accelerator** | `accelerate launch --num_processes=1 ...` | Uses `Accelerator()`; ws=1 → no all-reduce traffic |
| **Multi-process DDP** | `accelerate launch --num_processes=N ...` | Sharded rollouts + gather, all-reduce on backward |

All three paths share the same Python entry point. Backward compatibility
is preserved: existing scripts that run `python ppo_specs/run_e2_*.py`
continue to work bit-for-bit.

---

## 1. Phase A — Local 1-process baseline (5 min)

Confirms the refactor didn't break the legacy path.

```bash
cd RLVR-Comparison
python ppo_specs/run_e2_7.py --local-test --no-mc
```

**Expected:** prints `[E2.7] Final test accuracy ...`. Saves
`results/ppo_local_test.json`. No accelerate imports, no DDP code paths
engaged.

```bash
python ppo_specs/run_e2_8.py --local-test --capacity small
python ppo_specs/run_e2_8.py --local-test --capacity none   # tests REINFORCE epoch-0 fix
```

**Expected:** both complete; capacity sweep summary printed.

---

## 2. Phase B — Local multi-process DDP smoke (10–15 min)

This is the **primary** integration test. Runs DDP code paths so
distributed bugs surface without burning GPU hours.

### Bash / Linux

```bash
cd RLVR-Comparison
accelerate launch \
    --config_file configs/accelerate_cpu.yaml \
    ppo_specs/run_e2_7.py --local-test --no-mc
```

### PowerShell / Windows

PowerShell uses backtick (`` ` ``) for line continuation, not backslash.
Easiest is to put the command on one line:

```powershell
accelerate launch --config_file .\configs\accelerate_cpu.yaml .\ppo_specs\run_e2_7.py --local-test --no-mc
```

### Forcing the gloo (true CPU) path

If your machine has a CUDA GPU, Accelerate will pick it even with
`use_cpu: true` in the YAML config. To exercise the gloo backend (the
backend the SLURM CPU smoke uses), hide the GPU first:

```powershell
# PowerShell
$env:CUDA_VISIBLE_DEVICES = ""
accelerate launch --config_file .\configs\accelerate_cpu.yaml .\ppo_specs\run_e2_7.py --local-test --no-mc
$env:CUDA_VISIBLE_DEVICES = $null   # restore
```

```bash
# bash
CUDA_VISIBLE_DEVICES="" accelerate launch \
    --config_file configs/accelerate_cpu.yaml \
    ppo_specs/run_e2_7.py --local-test --no-mc
```

Without that, the run still validates the DDP code paths (sharding,
gathering, rank-0 gating, accelerator.backward) — just over NCCL
instead of gloo. Both backends use the same Python code; the value
of forcing gloo locally is to catch bugs that only manifest in
gloo-specific behavior (rare).

**Expected:**
- All 4 processes start; only rank 0 prints (no log spam).
- Periodic `step N | reward=... | acc=... | ...` lines from rank 0.
- Eval lines from rank 0.
- Final accuracy printed once (not 4×).
- `results/ppo_local_test.json` written once (not 4×).
- No NCCL/gloo timeouts (default timeout is 30 min via `TORCH_DISTRIBUTED_DEFAULT_TIMEOUT`).

**Possible failure modes:**

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `IndexError` on rank > 0 inside `generate_rollouts` | `B = len(local_prompts)` not rebound (regression) | Verify `ppo_trainer.py:_shard_list` is called and `B` is reassigned in the DDP branch |
| Hang at `wait_for_everyone` indefinitely | One rank crashed earlier silently; check logs of all ranks | Look at process per-rank stdout/stderr |
| `AssertionError: batch_size not divisible by num_processes` | local_test_config has `batch_size=4`, `num_processes=4` → ok. Other configs need: `batch_size % num_processes == 0` | Adjust `batch_size` or `num_processes` |
| `AttributeError: 'DistributedDataParallel' object has no attribute 'is_trainable'` | `_critic_trainable` cache not used at a call site | grep for `critic.is_trainable()` and replace with `self._critic_trainable` |
| `RuntimeError: element 0 of tensors does not require grad` | P12 epoch-0 skip + capacity="none" + reference_kl_coeff=0 | Already guarded via `if total_loss.requires_grad`; if it returns, check `ppo_trainer.py:ppo_update` |

---

## 3. Phase C — E2.8 4-process CPU DDP smoke (15 min)

```bash
accelerate launch \
    --config_file configs/accelerate_cpu.yaml \
    ppo_specs/run_e2_8.py --local-test --capacity small
```

**Expected:** capacity sweep completes; `e2_8_sweep_summary.json` written
once on rank 0.

---

## 4. Phase D — GPU cluster deployment

> **Requires actual cluster access** with `sbatch` (Slurm) installed.
> `sbatch` is Linux-only and only available on real HPC clusters; you
> cannot complete this phase on a Windows or macOS workstation. SSH
> into the cluster login node first.

Once Phases A–C pass locally, the GPU migration is a launcher swap:

### 4.1 Single-GPU smoke on cluster (sanity check)

```bash
# Interactive GPU session
srun --gres=gpu:1 --mem=64G --time=1:00:00 --pty bash
conda activate rlvr
cd /path/to/RLVR-Comparison
python ppo_specs/run_e2_7.py --local-test --no-mc
```

Should complete in ~2-5 min; same output as Phase A but on CUDA.

### 4.2 Multi-GPU full E2.7

```bash
sbatch scripts/slurm_e2_7.sh
```

The SLURM script already exports the right env vars (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
NCCL tuning) and invokes `accelerate launch` when `SLURM_MODE=multigpu`.

For 4× A100:
```bash
sbatch --export=ALL,SLURM_MODE=multigpu,NUM_GPUS=4 \
       --gres=gpu:4 \
       scripts/slurm_e2_7.sh
```

### 4.3 8B model with all memory mitigations

Edit your run script call to pass through PPOConfig overrides
(or set in `e2_7_config()` before sbatch):

```python
cfg.model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
cfg.gradient_checkpointing = True   # required: ~99 GB activation savings
cfg.optimizer_8bit = True           # bnb AdamW8bit: saves ~48 GB
cfg.reference_quant = "int8"        # bnb LLM.int8 on frozen ref: saves ~8 GB
cfg.optimizer_fused = True          # CUDA fused kernel: saves ~16 GB transient
cfg.length_bucketed_generation = True
cfg.generation_bucket_size = 4
```

With this stack, peak per-GPU memory at 8B drops from **~167 GB → ~25 GB**
(see `ppo_specs/specs/memory_optimization.md` §11.10), fitting comfortably
on a single A100 80GB or 4× A100 40GB DDP.

### 4.4 Resume from checkpoint after preemption

```bash
python ppo_specs/run_e2_7.py --seed 0 --resume-from auto
```

The DDP-aware resume re-prepares the model through Accelerator after
reassignment (§7.6.3 fix), so the wrapping is preserved across resumes.

---

## 5. What's NOT covered by this runbook

These are out of scope; track separately as needed:

- **FSDP / ZeRO-2/3** — required to fit 8B with global batch_size > 16
  on memory-constrained GPUs. The current spec is pure DDP (full-replica).
  See [distributed.md](distributed.md) §4.
- **Multi-node rendezvous** (`MASTER_ADDR`, `MASTER_PORT`, `machine_rank`) —
  the configs/accelerate_*.yaml files are single-node only. For multi-node,
  generate a fresh accelerate config with `accelerate config`.
- **MC-baseline parallelization** — currently MC is computed on rank 0
  and broadcast (§7.6.5). Sharding the MC samples across ranks is a
  follow-up; see distributed.md §5.4.
- **Optional DDP update sharding** — the current implementation has every
  rank run the full PPO update on the gathered batch (~replication, not
  parallelization). Real data-parallel speedup requires sharding the
  update; see ddp_cpu_gpu_migration.md §1.3.1.
- **`--ultrareview` of cluster runs** — multi-agent code review on a
  cluster branch. User-triggered.

---

## 6. Quick-reference cheat sheet

### Local (workstation, any OS)

```powershell
# PowerShell — single-proc baseline
python ppo_specs\run_e2_7.py --local-test --no-mc

# PowerShell — DDP smoke (one line; PowerShell uses ` not \ for continuation)
accelerate launch --config_file .\configs\accelerate_cpu.yaml .\ppo_specs\run_e2_7.py --local-test --no-mc
```

```bash
# bash — single-proc baseline
python ppo_specs/run_e2_7.py --local-test --no-mc

# bash — DDP smoke
accelerate launch --config_file configs/accelerate_cpu.yaml \
    ppo_specs/run_e2_7.py --local-test --no-mc
```

### Cluster (Slurm/Linux only — requires `sbatch`)

```bash
# Single-GPU cluster smoke
sbatch scripts/slurm_e2_7.sh

# 4-GPU full E2.7
sbatch --export=ALL,SLURM_MODE=multigpu,NUM_GPUS=4 \
       --gres=gpu:4 scripts/slurm_e2_7.sh

# Resume after preemption
python ppo_specs/run_e2_7.py --seed 0 --resume-from auto
```
