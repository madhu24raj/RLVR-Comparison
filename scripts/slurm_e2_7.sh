#!/bin/bash
###############################################################################
# SLURM Job Script: E2.7 PPO Head-to-Head on GSM8K
#
# Supports three modes controlled by SLURM_MODE:
#   single   - 1x GPU, local-test or small model (default)
#   multigpu - 4x or 8x GPUs on one node, full-scale run
#   array    - Job array for multiple seeds in parallel
#
# Usage examples:
#   # Single-GPU smoke test
#   sbatch scripts/slurm_e2_7.sh
#
#   # Full-scale 4xA100
#   sbatch --export=ALL,SLURM_MODE=multigpu,NUM_GPUS=4 scripts/slurm_e2_7.sh
#
#   # 3-seed sweep as a job array
#   sbatch --export=ALL,SLURM_MODE=array --array=0-2 scripts/slurm_e2_7.sh
#
#   # Override partition/account
#   sbatch -p my_partition -A my_account scripts/slurm_e2_7.sh
###############################################################################

#SBATCH --job-name=ppo-e2.7
#SBATCH --output=logs/e2_7_%j_%a.out
#SBATCH --error=logs/e2_7_%j_%a.err
# NOTE: --partition and --account are intentionally NOT specified as
# #SBATCH directives because Slurm does not perform shell expansion on
# directive values (so any "default with fallback" syntax would be passed
# literally to the scheduler). Override at submit time:
#   sbatch -p mypart -A myacct slurm_e2_7.sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --signal=B:SIGUSR1@120

# ── Configurable parameters (override via --export or environment) ───────────
SLURM_MODE="${SLURM_MODE:-single}"       # single | multigpu | array
NUM_GPUS="${NUM_GPUS:-1}"                 # GPUs per node (for multigpu mode)
DEVICE_MODE="${DEVICE_MODE:-gpu}"          # gpu | cpu
NUM_PROCESSES="${NUM_PROCESSES:-4}"        # accelerate --num_processes
SEED="${SEED:-42}"                        # default seed (overridden in array mode)
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
LOCAL_TEST="${LOCAL_TEST:-false}"          # set "true" for smoke test
CONDA_ENV="${CONDA_ENV:-rlvr}"
PROJECT_DIR="${PROJECT_DIR:-$(dirname $(dirname $(realpath $0)))}"
MODEL_CACHE="${MODEL_CACHE:-${PROJECT_DIR}/.model_cache}"
WANDB_PROJECT="${WANDB_PROJECT:-rlvr-comparison}"
WANDB_MODE="${WANDB_MODE:-offline}"       # offline by default; set "online" if cluster has internet

# ── Seed array for job arrays ────────────────────────────────────────────────
SEEDS=(42 123 456 789 1337)

# ── Derived settings ─────────────────────────────────────────────────────────
if [ "${SLURM_MODE}" = "array" ]; then
    SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
    echo "[ARRAY] Task ${SLURM_ARRAY_TASK_ID} -> seed ${SEED}"
fi

if [ "${SLURM_MODE}" = "multigpu" ]; then
    # Override SBATCH gpu count at submission time: --gres=gpu:${NUM_GPUS}
    echo "[MULTI-GPU] Using ${NUM_GPUS} GPUs"
fi

# Pick the accelerate config based on DEVICE_MODE x SLURM_MODE.
#   DEVICE_MODE=cpu   -> configs/accelerate_cpu.yaml          (gloo MULTI_CPU)
#   DEVICE_MODE=gpu + SLURM_MODE=single|array  -> configs/accelerate_single_gpu.yaml
#   DEVICE_MODE=gpu + SLURM_MODE=multigpu      -> configs/accelerate_multi_gpu.yaml
if [ "${DEVICE_MODE}" = "cpu" ]; then
    ACCEL_CONFIG="configs/accelerate_cpu.yaml"
else
    if [ "${SLURM_MODE}" = "multigpu" ]; then
        ACCEL_CONFIG="${ACCEL_CONFIG:-configs/accelerate_multi_gpu.yaml}"
    else
        ACCEL_CONFIG="${ACCEL_CONFIG:-configs/accelerate_single_gpu.yaml}"
    fi
fi

# Runtime sanity warning: CPU mode but a GPU was actually allocated.
if [ "${DEVICE_MODE}" = "cpu" ] && [[ "${SLURM_JOB_GRES:-}" == *gpu* ]]; then
    echo "[WARN] DEVICE_MODE=cpu but SLURM_JOB_GRES=${SLURM_JOB_GRES} — wasting a GPU."
fi

# ── Environment setup ────────────────────────────────────────────────────────
set -euo pipefail
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/results"

# Load modules (adjust to your cluster)
module purge 2>/dev/null || true
if [ "${DEVICE_MODE}" != "cpu" ]; then
    module load cuda/12.1 2>/dev/null || true
fi
module load anaconda3 2>/dev/null || true

# Activate conda environment
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate "${CONDA_ENV}" 2>/dev/null || true

cd "${PROJECT_DIR}"

# ── HuggingFace / model cache ───────────────────────────────────────────────
export HF_HOME="${MODEL_CACHE}"
export TRANSFORMERS_CACHE="${MODEL_CACHE}"
export HF_DATASETS_CACHE="${MODEL_CACHE}/datasets"

# ── NCCL tuning (multi-GPU) ─────────────────────────────────────────────────
if [ "${DEVICE_MODE}" != "cpu" ]; then
    export NCCL_DEBUG=WARN
    export NCCL_IB_DISABLE=0
    export NCCL_NET_GDR_LEVEL=5
    export NCCL_SOCKET_IFNAME=eth0
    export NCCL_P2P_LEVEL=NVL
    # Timeout for NCCL collectives (seconds) - increase for large models
    export NCCL_TIMEOUT=1800

    # Reduce CUDA allocator fragmentation when generate() (large transient KV cache)
    # is interleaved with ppo_update (large optimizer moments). Saves 5-10% headroom
    # on A100-80GB at 8B. See ppo_specs/specs/memory_optimization.md §11.8.
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

# ── W&B ──────────────────────────────────────────────────────────────────────
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_MODE="${WANDB_MODE}"
export WANDB_RUN_GROUP="e2.7"
export WANDB_TAGS="ppo,e2.7,seed${SEED}"

# ── Preemption handler ───────────────────────────────────────────────────────
# SLURM sends SIGUSR1 120s before killing the job. Forward to the Python
# child so GracefulExitHandler saves a checkpoint, then requeue.
handle_preempt() {
    echo "[PREEMPT] Received SIGUSR1 — forwarding to Python child..."
    kill -SIGUSR1 $CHILD_PID 2>/dev/null || true
    # Wait briefly for the GracefulExitHandler to save a checkpoint
    sleep 30
    echo "[PREEMPT] Requeueing job ${SLURM_JOB_ID}..."
    scontrol requeue "${SLURM_JOB_ID}" 2>/dev/null || true
    exit 0
}
trap handle_preempt SIGUSR1

# ── Build command ────────────────────────────────────────────────────────────
# Default checkpoint cadence and resume mode. Override via env to disable.
ARGS="--seed ${SEED} --checkpoint-every ${CHECKPOINT_EVERY:-50} --resume-from ${RESUME_FROM:-auto}"
if [ "${LOCAL_TEST}" = "true" ]; then
    ARGS="${ARGS} --local-test --no-mc"
fi
# Pass MODEL_NAME through to Python — the env var alone never reached the
# script, which silently ran the default Qwen-0.5B model regardless.
if [ -n "${MODEL_NAME:-}" ]; then
    ARGS="${ARGS} --model-name ${MODEL_NAME}"
fi
# 8B mitigation stack — engage when CLUSTER_8B=1 OR when MODEL_NAME contains
# '8B'. Without these flags an 8B model will OOM during AdamW step.
if [ "${CLUSTER_8B:-0}" = "1" ] || echo "${MODEL_NAME:-}" | grep -qi "8B"; then
    ARGS="${ARGS} --gradient-checkpointing --optimizer-8bit --optimizer-fused --reference-quant int8 --length-bucketed-generation"
fi
# Learned reward model (optional). Defaults to capacity="none" → deterministic
# gsm8k_reward only (the baseline). Setting REWARD_MODEL_CAPACITY=small|large
# loads a learned RM from REWARD_MODEL_NAME.
if [ -n "${REWARD_MODEL_CAPACITY:-}" ]; then
    ARGS="${ARGS} --reward-model-capacity ${REWARD_MODEL_CAPACITY}"
fi
if [ -n "${REWARD_MODEL_NAME:-}" ]; then
    ARGS="${ARGS} --reward-model-name ${REWARD_MODEL_NAME}"
fi
if [ -n "${REWARD_BLEND_ALPHA:-}" ]; then
    ARGS="${ARGS} --reward-blend-alpha ${REWARD_BLEND_ALPHA}"
fi
if [ "${REWARD_MODEL_REUSE_REFERENCE:-}" = "true" ]; then
    ARGS="${ARGS} --reward-model-reuse-reference"
fi

if [ "${SLURM_MODE}" = "single" ] || [ "${SLURM_MODE}" = "array" ]; then
    # Single-GPU (or CPU) launch via accelerate so the Accelerator code path
    # is always exercised. Under DEVICE_MODE=cpu this uses MULTI_CPU/gloo;
    # under DEVICE_MODE=gpu it uses accelerate_single_gpu.yaml (distributed_type: NO).
    LAUNCH_NUM_PROC="${NUM_PROCESSES}"
    if [ "${DEVICE_MODE}" != "cpu" ]; then
        LAUNCH_NUM_PROC=1
    fi
    echo "[RUN] accelerate launch --config_file ${ACCEL_CONFIG} --num_processes ${LAUNCH_NUM_PROC} ppo_specs/run_e2_7.py ${ARGS}"
    accelerate launch \
        --config_file "${ACCEL_CONFIG}" \
        --num_processes "${LAUNCH_NUM_PROC}" \
        ppo_specs/run_e2_7.py ${ARGS} &
    CHILD_PID=$!
    wait $CHILD_PID

elif [ "${SLURM_MODE}" = "multigpu" ]; then
    # Multi-GPU via accelerate
    echo "[RUN] accelerate launch --config_file ${ACCEL_CONFIG} --num_processes ${NUM_GPUS} ppo_specs/run_e2_7.py ${ARGS}"
    accelerate launch \
        --config_file "${ACCEL_CONFIG}" \
        --num_processes "${NUM_GPUS}" \
        ppo_specs/run_e2_7.py ${ARGS} &
    CHILD_PID=$!
    wait $CHILD_PID
fi

echo "[DONE] E2.7 seed=${SEED} completed. Exit code: $?"
echo "[DONE] Results: results/ppo_e2_7_seed${SEED}.json"
