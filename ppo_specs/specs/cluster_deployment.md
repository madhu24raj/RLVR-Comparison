# Cluster Deployment Guide: PPO Experiments (E2.7 / E2.8)

This document covers everything needed to run the PPO RLVR experiments on a
SLURM-managed GPU cluster. It addresses job scripts, accelerate configs,
environment setup, shared filesystems, checkpointing gaps, logging, and
preemption handling.

---

## Table of Contents

1. [Resource Requirements](#1-resource-requirements)
2. [SLURM Job Scripts](#2-slurm-job-scripts)
3. [Accelerate Configurations](#3-accelerate-configurations)
4. [Environment Setup](#4-environment-setup)
5. [Shared Filesystem Layout](#5-shared-filesystem-layout)
6. [NCCL and Multi-Node Networking](#6-nccl-and-multi-node-networking)
7. [Checkpointing (Gap Analysis)](#7-checkpointing-gap-analysis)
8. [Logging and W&B](#8-logging-and-wandb)
9. [Preemption Handling](#9-preemption-handling)
10. [Pre-Flight Checklist](#10-pre-flight-checklist)

---

## 1. Resource Requirements

### Model sizes and GPU memory

| Model | Params | dtype | Approx VRAM (inference) | Approx VRAM (training) |
|-------|--------|-------|------------------------|------------------------|
| Qwen2.5-0.5B-Instruct | 500M | float32 | ~2 GB | ~6 GB |
| Qwen2.5-0.5B-Instruct | 500M | bfloat16 | ~1 GB | ~3 GB |
| Llama-3-8B-Instruct | 8B | bfloat16 | ~16 GB | ~50 GB |

PPO requires the policy model in training mode plus the critic head. For the
0.5B model, a single GPU (even a consumer 24 GB card) is sufficient. For the
8B model, you need either:
- 4x A100-40GB with FSDP or DeepSpeed ZeRO-2
- 2x A100-80GB with DDP
- 1x A100-80GB with gradient checkpointing (tight)

### Walltime estimates

Based on **batched generation** (P1 fixed). Previous per-sample estimates were
10-15x longer:

| Experiment | Model | GPUs | Steps | Est. Walltime |
|-----------|-------|------|-------|---------------|
| E2.7 local-test | 0.5B | 1x any | 5 | ~5 min |
| E2.7 full | 0.5B | 1x A100 | 200 | ~2-4 hours |
| E2.7 full | 8B | 4x A100 | 200 | ~8-12 hours |
| E2.8 sweep (4 caps) | 0.5B | 1x A100 | 4x150 | ~6-10 hours |
| E2.8 single capacity | 8B | 4x A100 | 150 | ~6-8 hours |

**Note:** P1 (batched generation) and P7 (Accelerate integration) are both
now fixed. Multi-GPU DDP runs end-to-end via `accelerate launch`. See
[ddp_cpu_gpu_migration.md](ddp_cpu_gpu_migration.md) for the design and
[integration_beads.md](integration_beads.md) for the Phase 2 work that landed
the trainer/run-script/checkpoint refactor. FSDP and DeepSpeed remain TBD;
see [distributed.md §4](distributed.md#4-deepspeed-zero-integration).

### CPU and memory

- `--cpus-per-task=8`: tokenizer and data loading benefit from workers
- `--mem=64G`: safe default; 32G is enough for 0.5B, 8B may spike higher

---

## 2. SLURM Job Scripts

### Created files

| File | Purpose |
|------|---------|
| `scripts/slurm_e2_7.sh` | E2.7 head-to-head (single/multi-GPU/array modes) |
| `scripts/slurm_e2_8.sh` | E2.8 critic sweep (sweep/parallel/multi-GPU modes) |

### E2.7: `scripts/slurm_e2_7.sh`

Three modes controlled by `SLURM_MODE`:

**Single-GPU (default):**
```bash
sbatch scripts/slurm_e2_7.sh
```

**Multi-GPU (4x A100):**
```bash
sbatch --gres=gpu:4 \
       --export=ALL,SLURM_MODE=multigpu,NUM_GPUS=4 \
       scripts/slurm_e2_7.sh
```

**Job array (3 seeds in parallel):**
```bash
sbatch --array=0-2 \
       --export=ALL,SLURM_MODE=array \
       scripts/slurm_e2_7.sh
```
Seeds are mapped from the array: `SEEDS=(42 123 456 789 1337)`.

**Cluster customization (all scripts):**
```bash
sbatch -p your_partition -A your_account scripts/slurm_e2_7.sh
```

### E2.8: `scripts/slurm_e2_8.sh`

**Sequential sweep (all 4 capacities, 1 GPU):**
```bash
sbatch scripts/slurm_e2_8.sh
```

**Parallel sweep (each capacity as separate job):**
```bash
sbatch --array=0-3 \
       --export=ALL,SLURM_MODE=parallel \
       scripts/slurm_e2_8.sh
```
Maps: 0=none, 1=small, 2=medium, 3=large.

**Multi-seed for one capacity:**
```bash
sbatch --array=0-2 \
       --export=ALL,SLURM_MODE=parallel,CAPACITY=medium \
       scripts/slurm_e2_8.sh
```

**Large model on multiple GPUs:**
```bash
sbatch --gres=gpu:4 \
       --export=ALL,SLURM_MODE=multigpu,CAPACITY=large,NUM_GPUS=4 \
       scripts/slurm_e2_8.sh
```

### Parameterized values

All scripts use environment variable defaults that work out-of-the-box but
can be overridden:

| Variable | Default | Description |
|----------|---------|-------------|
| `PARTITION` | `gpu` | SLURM partition name |
| `ACCOUNT` | `default` | SLURM account/allocation |
| `CONDA_ENV` | `rlvr` | Conda environment name |
| `MODEL_NAME` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace model ID |
| `MODEL_CACHE` | `$PROJECT_DIR/.model_cache` | HF cache directory |
| `WANDB_MODE` | `offline` | W&B mode (offline/online/disabled) |
| `WANDB_PROJECT` | `rlvr-comparison` | W&B project name |
| `LOCAL_TEST` | `false` | Set `true` for smoke test |
| `DEVICE_MODE` | `gpu` | `cpu` or `gpu`. CPU mode picks `accelerate_cpu.yaml` (gloo MULTI_CPU), skips `module load cuda`, skips NCCL exports. |
| `NUM_PROCESSES` | `4` | accelerate `--num_processes` (only consulted under CPU mode and as the gloo worker count) |
| `CLUSTER_8B` | `0` | Set `1` to auto-engage the 8B mitigation stack (`--gradient-checkpointing --optimizer-8bit --optimizer-fused --reference-quant int8 --length-bucketed-generation`). Auto-engaged when `MODEL_NAME` matches `*8B*`. |
| `REWARD_MODEL_CAPACITY` | unset (= `none`) | Learned RM tier: `none` / `small` / `large`. Default leaves the deterministic verifier (gsm8k_reward) as the sole reward signal — the project baseline. |
| `REWARD_MODEL_NAME` | unset | HF hub id or local path of the learned RM checkpoint (required when `REWARD_MODEL_CAPACITY` ≠ `none`). |
| `REWARD_BLEND_ALPHA` | `1.0` | Convex blend `alpha * RM + (1-alpha) * gsm8k_reward`. `1.0` = RM only; `0.0` = verifier only. |
| `REWARD_MODEL_REUSE_REFERENCE` | unset (= `false`) | Set `true` to reuse the KL-anchor reference model as the RM base. Saves ~16 GB at 8B; requires `reference_kl_coeff > 0`. |

### Single-node multi-process CPU smoke (DEVICE_MODE=cpu)

For a CPU-only smoke at the cluster — useful before booking GPU time, since
distributed bugs (rank-0 gating, `.generate()` hangs under DDP, shard
divisibility) all surface on the CPU run:

```bash
sbatch --gres=none --partition=cpu \
       --export=ALL,DEVICE_MODE=cpu,LOCAL_TEST=true,NUM_PROCESSES=4 \
       scripts/slurm_e2_7.sh
```

What to look for in the SLURM stdout:
- `[RUN] accelerate launch --config_file configs/accelerate_cpu.yaml --num_processes 4 ppo_specs/run_e2_7.py ...`
- **No** `module load cuda/12.1` invocation.
- **No** `NCCL_*` env vars in the dump.
- 5 steps complete; only rank 0 prints; `results/ppo_local_test.json` written exactly once.

This is recipe 7.4 in [ddp_cpu_gpu_migration.md §7](ddp_cpu_gpu_migration.md#7-smoke-test-recipe-runbook).

---

## 3. Accelerate Configurations

### Created files

| File | Use case |
|------|----------|
| `configs/accelerate_single_gpu.yaml` | 1x GPU with bf16 mixed precision |
| `configs/accelerate_multi_gpu.yaml` | 4x GPU DDP with bf16 |

### Single GPU (`configs/accelerate_single_gpu.yaml`)

- `distributed_type: 'NO'`
- `mixed_precision: bf16`
- Suitable for 0.5B model or any model that fits on one GPU

Usage:
```bash
accelerate launch --config_file configs/accelerate_single_gpu.yaml \
    ppo_specs/run_e2_7.py --seed 42
```

### Multi-GPU DDP (`configs/accelerate_multi_gpu.yaml`)

- `distributed_type: MULTI_GPU`
- `num_processes: 4` (override with `--num_processes N`)
- `mixed_precision: bf16`

Usage:
```bash
accelerate launch --config_file configs/accelerate_multi_gpu.yaml \
    --num_processes 8 \
    ppo_specs/run_e2_7.py --seed 42
```

### FSDP Config for 8B+ Models (not yet created)

For models that do not fit on a single GPU even in bf16, FSDP shards model
parameters, gradients, and optimizer states across GPUs. A recommended config:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
mixed_precision: bf16
num_machines: 1
num_processes: 4
```

**Status:** Multi-GPU DDP (via `accelerate_multi_gpu.yaml`) is now wired
end-to-end (Phase 2, 2026-05-04). FSDP and DeepSpeed remain TBD — the FSDP
yaml below is a starting point but needs trainer-side work for
`fsdp_use_orig_params` interactions with the no-grad hidden-state extraction
in `_extract_last_hidden`. Track the FSDP/DeepSpeed work in
[distributed.md §4](distributed.md#4-deepspeed-zero-integration); not in
this doc's scope.

### DeepSpeed ZeRO-2 Config (not yet created)

An alternative to FSDP for memory-efficient training. A recommended config:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_multinode_launcher: standard
  gradient_accumulation_steps: 1
  gradient_clipping: 1.0
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero_stage: 2
mixed_precision: bf16
num_machines: 1
num_processes: 4
```

ZeRO-2 shards optimizer states and gradients (not parameters), so model
forward/backward code needs fewer changes than FSDP. However, the same
single-sample generation loop limitation applies.

---

## 4. Environment Setup

### Script: `scripts/setup_env.sh`

Run on a login node with internet access:

```bash
# Basic setup (0.5B model only)
bash scripts/setup_env.sh

# Full setup including 8B model
bash scripts/setup_env.sh --large-model

# Custom environment name and CUDA version
bash scripts/setup_env.sh --env-name myenv --cuda 12.4

# CPU-only install (useful on CPU-only login or dev nodes — installs the CPU
# torch wheel, skips CUDA verification). Pairs with DEVICE_MODE=cpu sbatch.
bash scripts/setup_env.sh --cpu-only
```

The script performs five steps:
1. Creates conda environment with Python 3.11
2. Installs PyTorch with the specified CUDA version
3. Installs all dependencies from `requirements.txt`
4. Pre-downloads model weights and GSM8K dataset to `$PROJECT_DIR/.model_cache`
5. Creates output directories (`results/`, `logs/`, `checkpoints/`)

### Gated models (Llama-3)

Llama-3-8B-Instruct is a gated model. Before downloading:

```bash
# Set your HuggingFace token
export HF_TOKEN="hf_..."
# Or login interactively
huggingface-cli login
```

Accept the model license at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct.

### Why pre-download matters

- SLURM compute nodes often have no internet access
- Downloading 16 GB of model weights wastes expensive GPU-hours
- Multiple concurrent jobs downloading the same model cause filesystem thrashing
- The cache is stored on the shared filesystem so all nodes can access it

---

## 5. Shared Filesystem Layout

### Recommended structure

```
/shared/projects/rlvr-comparison/          # or $SCRATCH, $WORK, etc.
├── .model_cache/                          # HF_HOME: shared model weights
│   ├── hub/models--Qwen--Qwen2.5-0.5B-Instruct/
│   ├── hub/models--meta-llama--Meta-Llama-3-8B-Instruct/
│   └── datasets/
├── results/                               # experiment outputs (JSON logs)
│   ├── ppo_e2_7_seed42.json
│   ├── ppo_e2_8_none_seed42.json
│   └── e2_8_sweep_summary.json
├── logs/                                  # SLURM stdout/stderr
│   ├── e2_7_12345_0.out
│   └── e2_8_12346_1.err
├── checkpoints/                           # (future) training checkpoints
├── configs/
├── scripts/
├── ppo_specs/
└── src/
```

### Key environment variables

Set these in your `~/.bashrc` or in the SLURM scripts:

```bash
export HF_HOME="/shared/projects/rlvr-comparison/.model_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
```

### Filesystem considerations

- **Model cache:** Place on a fast parallel filesystem (Lustre, GPFS). Models
  are read-heavy at startup, then stay in GPU memory.
- **Results:** Small JSON files. Any filesystem works. Consider periodic
  rsync to a more durable location.
- **Checkpoints (future):** Will be large (2-16 GB per checkpoint). Use
  scratch/work, not home directory. Set a retention policy.

---

## 6. NCCL and Multi-Node Networking

The SLURM scripts set these NCCL variables by default:

```bash
export NCCL_DEBUG=WARN               # INFO for debugging, WARN for production
export NCCL_IB_DISABLE=0             # Enable InfiniBand (set 1 if no IB)
export NCCL_NET_GDR_LEVEL=5          # GPU Direct RDMA level
export NCCL_SOCKET_IFNAME=eth0       # Network interface (adjust per cluster)
export NCCL_P2P_LEVEL=NVL            # NVLink P2P (for multi-GPU on one node)
export NCCL_TIMEOUT=1800             # 30-minute timeout for collectives
```

### Cluster-specific adjustments

| Cluster type | Change |
|-------------|--------|
| No InfiniBand | `NCCL_IB_DISABLE=1` |
| Different NIC name | `NCCL_SOCKET_IFNAME=ib0` or `NCCL_SOCKET_IFNAME=eno1` |
| Multi-node (2+ nodes) | Add `MASTER_ADDR` and `MASTER_PORT` from SLURM env |
| AWS/cloud | `NCCL_SOCKET_IFNAME=ens5`, `NCCL_NET_GDR_LEVEL=0` |

### Multi-node setup

For jobs spanning multiple nodes (not needed for current experiments but
relevant for 8B+ models), add to the SLURM script:

```bash
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
```

And use `srun` instead of direct `python`:
```bash
srun accelerate launch --multi_gpu \
    --num_machines $SLURM_NNODES \
    --num_processes $((SLURM_NNODES * NUM_GPUS)) \
    --machine_rank $SLURM_PROCID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    ppo_specs/run_e2_7.py --seed 42
```

---

## 7. Checkpointing (Gap Analysis)

### Current state: IMPLEMENTED

Checkpointing is now fully implemented in `ppo_specs/checkpoint.py`. Key features:

- Atomic saves (write to temp dir, then rename)
- Checkpoint rotation (keep last K)
- Full state: model, critic, optimizers, RNG states, logger, training counters
- Config compatibility verification on resume via hash
- Signal handling (`GracefulExitHandler`) for SIGTERM/SIGINT
- `run_e2_7.py` integrates checkpoint save/load with graceful exit
- Config fields: `checkpoint_every`, `keep_checkpoints`, `checkpoint_dir`, `resume_from`

The gap analysis below is retained for reference on what is saved.

### What needs to be saved

A complete checkpoint for PPO training includes:

```python
checkpoint = {
    # Core model state
    "model_state_dict":           model.state_dict(),
    "policy_optimizer_state":     policy_optimizer.state_dict(),

    # Critic state
    "critic_state_dict":          critic.state_dict(),
    "critic_optimizer_state":     critic_optimizer.state_dict(),

    # Training progress
    "step":                       trainer.step,
    "total_rollouts":             trainer.total_rollouts,
    "seed":                       config.seed,

    # RNG states for reproducibility
    "torch_rng_state":            torch.get_rng_state(),
    "cuda_rng_state":             torch.cuda.get_rng_state_all(),
    "numpy_rng_state":            np.random.get_state(),
    "python_rng_state":           random.getstate(),

    # Config for verification
    "config":                     dataclasses.asdict(config),
}
```

### Recommended implementation

Add to `PPOTrainer`:

```python
def save_checkpoint(self, path: str):
    """Save training state to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": self.model.state_dict(),
        "policy_optimizer_state": self.policy_optimizer.state_dict(),
        "critic_state_dict": self.critic.state_dict(),
        "step": self.step,
        "total_rollouts": self.total_rollouts,
    }
    if self.critic_optimizer:
        checkpoint["critic_optimizer_state"] = self.critic_optimizer.state_dict()
    torch.save(checkpoint, path)

def load_checkpoint(self, path: str):
    """Resume from a saved checkpoint."""
    checkpoint = torch.load(path, map_location=self.device)
    self.model.load_state_dict(checkpoint["model_state_dict"])
    self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state"])
    self.critic.load_state_dict(checkpoint["critic_state_dict"])
    if self.critic_optimizer and "critic_optimizer_state" in checkpoint:
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state"])
    self.step = checkpoint["step"]
    self.total_rollouts = checkpoint["total_rollouts"]
```

Add to training loops (`run_e2_7.py`, `run_e2_8.py`):

```python
# Save every N steps
CKPT_DIR = f"checkpoints/{config.experiment_name}"
if step % checkpoint_every == 0:
    trainer.save_checkpoint(f"{CKPT_DIR}/step_{step}.pt")

# Resume logic
ckpt_path = find_latest_checkpoint(CKPT_DIR)
if ckpt_path:
    trainer.load_checkpoint(ckpt_path)
    start_step = trainer.step
```

### Checkpoint frequency

- **0.5B model:** Checkpoint every 20 steps (~10 MB per checkpoint, negligible)
- **8B model:** Checkpoint every 50 steps (~16 GB per checkpoint; keep last 3)
- **On preemption signal:** Always save immediately (see Section 9)

---

## 8. Logging and W&B

### Should W&B be used?

**Yes.** The project already lists `wandb` in `requirements.txt`. Benefits:
- Real-time monitoring of training curves across seeds and capacities
- Automatic system metrics (GPU utilization, memory, temperature)
- Easy comparison of E2.8 critic capacities in a single dashboard
- Accessible from outside the cluster (vs. TensorBoard which requires tunneling)

### Configuration for cluster jobs

The SLURM scripts set W&B to **offline mode** by default because many cluster
compute nodes lack internet access:

```bash
export WANDB_MODE=offline    # logs saved locally, sync later
export WANDB_PROJECT=rlvr-comparison
export WANDB_RUN_GROUP=e2.7  # groups runs in the dashboard
export WANDB_TAGS="ppo,e2.7,seed42"
```

To sync offline runs after the job completes:

```bash
wandb sync logs/wandb/offline-run-*
```

If the cluster has internet on compute nodes:
```bash
export WANDB_MODE=online
export WANDB_API_KEY="your_key_here"  # or use wandb login on login node
```

### Integration with current code

The current code does NOT use W&B -- it logs to JSON via `ExperimentLogger`.
To add W&B without changing the experiment scripts, wrap the logger:

```python
import wandb

class WandbExperimentLogger(ExperimentLogger):
    def __init__(self, name, output_dir, config=None):
        super().__init__(name, output_dir)
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "rlvr-comparison"),
            name=name,
            config=config,
        )

    def log_step(self, step, **metrics):
        super().log_step(step, **metrics)
        wandb.log(metrics, step=step)

    def save(self):
        super().save()
        wandb.finish()
```

### Recommended dashboard layout

Create a W&B workspace with these panels:

| Panel | X-axis | Y-axis | Group by |
|-------|--------|--------|----------|
| Convergence (E2.7) | `total_rollouts` | `test_accuracy` | seed |
| Stability (E2.7) | step | `reward_variance` | seed |
| Critic sweep (E2.8) | `total_rollouts` | `test_accuracy` | `critic_capacity` |
| Critic error (E2.8) | step | `critic_error_ev` | `critic_capacity` |
| Advantage bias (E2.8) | step | `advantage_bias` | `critic_capacity` |

---

## 9. Preemption Handling

### The problem

On shared clusters, jobs in preemptible/scavenger partitions can be killed
at any time. Without checkpointing, all progress is lost.

### Current handling

Both SLURM scripts include a preemption trap:

```bash
#SBATCH --signal=B:SIGUSR1@120    # send SIGUSR1 120s before kill

handle_preempt() {
    echo "[PREEMPT] Requeueing job ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
trap handle_preempt SIGUSR1
```

This requeues the job. The body of the training code now does have
checkpoint support (see Section 7 — `ppo_specs/checkpoint.py` is wired with
atomic save/load and a `GracefulExitHandler`), but the SIGUSR1 path in this
SLURM trap does **not yet forward** the signal cleanly under
`accelerate launch` (the SIGUSR1 reaches the launcher, not necessarily the
Python child processes). The handler below is the recommended target shape;
landing it under Accelerate's process tree is tracked as a follow-up bead
([ddp_cpu_gpu_migration.md §10](ddp_cpu_gpu_migration.md#10-out-of-scope-follow-ups)).

```bash
handle_preempt() {
    echo "[PREEMPT] Saving checkpoint..."
    # Forward signal to Python process to trigger checkpoint save
    kill -SIGUSR1 $CHILD_PID 2>/dev/null || true
    sleep 10  # give Python time to save
    echo "[PREEMPT] Requeueing job ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
```

And in Python:

```python
import signal

def handle_preempt(signum, frame):
    print("[PREEMPT] Saving emergency checkpoint...")
    trainer.save_checkpoint(f"checkpoints/{config.experiment_name}/preempt.pt")
    print("[PREEMPT] Checkpoint saved.")

signal.signal(signal.SIGUSR1, handle_preempt)
```

### Best practices for preemptible jobs

1. **Always checkpoint:** Save at least every 20-50 steps
2. **Auto-resume:** Start training loops by checking for existing checkpoints
3. **Idempotent seeds:** Re-seeding from checkpoint RNG state ensures exact
   reproducibility across preemption events
4. **Keep last N checkpoints:** Avoid filling scratch with 16 GB files
5. **Use `--requeue`:** Add `#SBATCH --requeue` to auto-requeue on node failure

---

## 10. Pre-Flight Checklist

Before submitting jobs to the cluster, verify these items:

### One-time setup

- [ ] Run `bash scripts/setup_env.sh` on a login node
- [ ] Verify: `conda activate rlvr && python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Pre-download models: check `$HF_HOME/hub/` contains model directories
- [ ] For Llama-3: `huggingface-cli login` and accept model license
- [ ] Create output directories: `mkdir -p results logs checkpoints`
- [ ] If using W&B online: `wandb login` on the login node

### Per-experiment checks

- [ ] Verify `model_name` in `config.py` matches the intended model
- [ ] For 8B model: set `cfg.gradient_checkpointing = True` (bfloat16 is auto)
- [ ] Adjust SLURM `--partition` and `--account` for your cluster
- [ ] Set `NCCL_SOCKET_IFNAME` to match your cluster's network interface
- [ ] Run a smoke test: `sbatch --export=ALL,LOCAL_TEST=true scripts/slurm_e2_7.sh`

### Previously known blockers (all resolved)

These issues from `performance.md` and `distributed.md` have been **fixed**:

| ID | Issue | Status |
|----|-------|--------|
| P1 | Per-sample generation (no batching) | **Fixed** -- batched generation with left-padding |
| P6 | No gradient checkpointing | **Fixed** -- `config.gradient_checkpointing = True` |
| P7 | No Accelerate / multi-GPU integration | **Fixed** (Phase 2, 2026-05-04) -- `PPOTrainer` runs under `Accelerator`; gloo CPU smoke + multi-GPU DDP both wired. See [ddp_cpu_gpu_migration.md](ddp_cpu_gpu_migration.md). |
| P8 | float32 on GPU | **Fixed** -- `config.torch_dtype = "auto"` (bf16 on CUDA) |
| RM | No tier-based learned reward model | **Fixed** (Phase 2, 2026-05-04) -- `ppo_specs/reward_model.py` with `none`/`small`/`large` tiers; preserves the "PPO trainer with custom reward functions" baseline. See [reward_model_integration.md](reward_model_integration.md). |

Remaining out-of-scope work tracked in
[distributed.md §4](distributed.md#4-deepspeed-zero-integration) (DeepSpeed
ZeRO-2/3) and [ddp_cpu_gpu_migration.md §10](ddp_cpu_gpu_migration.md#10-out-of-scope-follow-ups)
(multi-node rendezvous, FSDP, SIGUSR1-under-accelerate, W&B wiring).

---

## Appendix A: Complete Submission Recipes

### Recipe 1: Quick validation (5 min)

```bash
sbatch --export=ALL,LOCAL_TEST=true scripts/slurm_e2_7.sh
```

### Recipe 2: E2.7 with 3 seeds (parallel)

```bash
sbatch --array=0-2 --export=ALL,SLURM_MODE=array scripts/slurm_e2_7.sh
```

### Recipe 3: E2.8 full sweep (parallel, fastest)

```bash
sbatch --array=0-3 --export=ALL,SLURM_MODE=parallel scripts/slurm_e2_8.sh
```

### Recipe 4: E2.8 full sweep with 3 seeds per capacity (12 jobs)

```bash
for cap in none small medium large; do
    sbatch --array=0-2 \
           --export=ALL,SLURM_MODE=parallel,CAPACITY=${cap} \
           scripts/slurm_e2_8.sh
done
```

### Recipe 5: 8B model on 4x A100 (deterministic verifier reward)

```bash
# E2.7
sbatch --gres=gpu:4 --mem=128G --time=24:00:00 \
       --export=ALL,SLURM_MODE=multigpu,NUM_GPUS=4,MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct \
       scripts/slurm_e2_7.sh

# E2.8 (one capacity at a time)
sbatch --gres=gpu:4 --mem=128G --time=24:00:00 \
       --export=ALL,SLURM_MODE=multigpu,NUM_GPUS=4,CAPACITY=large,MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct \
       scripts/slurm_e2_8.sh
```

The 8B mitigation stack (`--gradient-checkpointing --optimizer-8bit
--optimizer-fused --reference-quant int8 --length-bucketed-generation`)
auto-engages whenever `MODEL_NAME` matches `*8B*`, so no extra flags needed
for the OOM-prevention story. Set `CLUSTER_8B=1` to force it on for non-8B
model names.

### Recipe 6: 8B model with a learned reward model

```bash
sbatch --gres=gpu:4 --mem=128G --time=24:00:00 \
       --export=ALL,SLURM_MODE=multigpu,NUM_GPUS=4,\
MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct,\
REWARD_MODEL_CAPACITY=large,\
REWARD_MODEL_NAME=<your-RM-checkpoint>,\
REWARD_MODEL_REUSE_REFERENCE=true \
       scripts/slurm_e2_7.sh
```

`REWARD_MODEL_REUSE_REFERENCE=true` shares weights with the KL-anchor
reference model — saves ~16 GB of GPU memory at 8B. Requires
`reference_kl_coeff > 0` (default for non-deterministic reward modes). For
"verifier + RM blend" set `REWARD_BLEND_ALPHA=0.7` (or whatever blend you
want); leave unset for RM-only.

The conflict-resolution rule from
[reward_model_integration.md "Interaction with reward_mode"](reward_model_integration.md#interaction-with-reward_mode-orthogonality-contract)
applies: `REWARD_MODEL_CAPACITY != none` × `reward_mode != deterministic`
is rejected at load time. Stick with the default reward_mode for learned-RM
runs and use `REWARD_BLEND_ALPHA` to control the verifier weight.

### Recipe 7: CPU cluster smoke (DDP-correctness gate)

```bash
sbatch --gres=none --partition=cpu \
       --export=ALL,DEVICE_MODE=cpu,LOCAL_TEST=true,NUM_PROCESSES=4 \
       scripts/slurm_e2_7.sh
```

5-step smoke that exercises the same Accelerate code paths as the GPU run
(shard / unwrap / gather / wait_for_everyone) but with gloo on CPU.
Cheaper to schedule than GPU time; catches distributed bugs before they
cost paid GPU-hours. See [ddp_cpu_gpu_migration.md §7](ddp_cpu_gpu_migration.md#7-smoke-test-recipe-runbook).

---

## Appendix B: Accelerate Config Reference

### FSDP (for 8B+ models, 4x A100-40GB)

Save as `configs/accelerate_fsdp.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
mixed_precision: bf16
num_machines: 1
num_processes: 4
main_training_function: main
```

### DeepSpeed ZeRO-2

Save as `configs/accelerate_deepspeed.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_multinode_launcher: standard
  gradient_accumulation_steps: 1
  gradient_clipping: 1.0
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero_stage: 2
mixed_precision: bf16
num_machines: 1
num_processes: 4
main_training_function: main
```

Usage:
```bash
accelerate launch --config_file configs/accelerate_deepspeed.yaml \
    --num_processes 4 \
    ppo_specs/run_e2_7.py --seed 42
```

**Note:** Multi-GPU DDP (via `accelerate_multi_gpu.yaml`) is wired and
working as of Phase 2 (2026-05-04). `ppo_trainer.py` calls
`accelerator.prepare(model, optimizer)` and uses the unwrap pattern for
`.generate()`. FSDP and DeepSpeed remain TBD — both require additional
trainer-side work for the no-grad hidden-state extraction and the
manual-optimizer split between policy and critic; track in
[distributed.md §4](distributed.md#4-deepspeed-zero-integration).
