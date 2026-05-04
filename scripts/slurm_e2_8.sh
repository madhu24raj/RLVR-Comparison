#!/bin/bash
###############################################################################
# SLURM Job Script: E2.8 PPO Critic Quality Sweep
#
# Supports three modes:
#   sweep     - Run all 4 critic capacities sequentially on one GPU (default)
#   parallel  - Job array: one capacity per task (4 tasks)
#   multigpu  - One capacity on multiple GPUs (for large models)
#
# Usage examples:
#   # Full sweep on 1 GPU (sequential, ~4x walltime of single capacity)
#   sbatch scripts/slurm_e2_8.sh
#
#   # Parallel: each capacity as a separate job (fastest)
#   sbatch --export=ALL,SLURM_MODE=parallel --array=0-3 scripts/slurm_e2_8.sh
#
#   # Multi-seed array for one capacity
#   sbatch --export=ALL,SLURM_MODE=parallel,CAPACITY=medium --array=0-2 scripts/slurm_e2_8.sh
#
#   # Single capacity on 4 GPUs (for 8B+ models)
#   sbatch --export=ALL,SLURM_MODE=multigpu,CAPACITY=large,NUM_GPUS=4 \
#          --gres=gpu:4 scripts/slurm_e2_8.sh
###############################################################################

#SBATCH --job-name=ppo-e2.8
#SBATCH --output=logs/e2_8_%j_%a.out
#SBATCH --error=logs/e2_8_%j_%a.err
# NOTE: --partition and --account are intentionally NOT specified as
# #SBATCH directives because Slurm does not perform shell expansion on
# directive values. Override at submit time:
#   sbatch -p mypart -A myacct slurm_e2_8.sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --signal=B:SIGUSR1@120

# ── Configurable parameters ─────────────────────────────────────────────────
SLURM_MODE="${SLURM_MODE:-sweep}"          # sweep | parallel | multigpu
NUM_GPUS="${NUM_GPUS:-1}"
DEVICE_MODE="${DEVICE_MODE:-gpu}"          # gpu | cpu
NUM_PROCESSES="${NUM_PROCESSES:-4}"        # accelerate --num_processes
SEED="${SEED:-42}"
CAPACITY="${CAPACITY:-}"                   # empty = all capacities
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
LOCAL_TEST="${LOCAL_TEST:-false}"
CONDA_ENV="${CONDA_ENV:-rlvr}"
PROJECT_DIR="${PROJECT_DIR:-$(dirname $(dirname $(realpath $0)))}"
MODEL_CACHE="${MODEL_CACHE:-${PROJECT_DIR}/.model_cache}"
WANDB_PROJECT="${WANDB_PROJECT:-rlvr-comparison}"
WANDB_MODE="${WANDB_MODE:-offline}"

# ── Capacity arrays ─────────────────────────────────────────────────────────
ALL_CAPACITIES=(none small medium large)
SEEDS=(42 123 456)

# ── Derived settings ────────────────────────────────────────────────────────
if [ "${SLURM_MODE}" = "parallel" ]; then
    if [ -n "${CAPACITY}" ]; then
        # Multi-seed for one capacity
        SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
        echo "[ARRAY] Capacity=${CAPACITY}, seed=${SEED}"
    else
        # One capacity per task
        CAPACITY=${ALL_CAPACITIES[$SLURM_ARRAY_TASK_ID]}
        echo "[ARRAY] Task ${SLURM_ARRAY_TASK_ID} -> capacity=${CAPACITY}"
    fi
fi

# Pick the accelerate config based on DEVICE_MODE x SLURM_MODE.
#   DEVICE_MODE=cpu                  -> configs/accelerate_cpu.yaml          (gloo MULTI_CPU)
#   DEVICE_MODE=gpu + sweep|parallel -> configs/accelerate_single_gpu.yaml   (one capacity per GPU)
#   DEVICE_MODE=gpu + multigpu       -> configs/accelerate_multi_gpu.yaml
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

# ── Environment setup ───────────────────────────────────────────────────────
set -euo pipefail
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/results"

module purge 2>/dev/null || true
if [ "${DEVICE_MODE}" != "cpu" ]; then
    module load cuda/12.1 2>/dev/null || true
fi
module load anaconda3 2>/dev/null || true

eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate "${CONDA_ENV}" 2>/dev/null || true

cd "${PROJECT_DIR}"

# ── HuggingFace / model cache ───────────────────────────────────────────────
export HF_HOME="${MODEL_CACHE}"
export TRANSFORMERS_CACHE="${MODEL_CACHE}"
export HF_DATASETS_CACHE="${MODEL_CACHE}/datasets"

# ── NCCL tuning ──────────────────────────────────────────────────────────────
if [ "${DEVICE_MODE}" != "cpu" ]; then
    export NCCL_DEBUG=WARN
    export NCCL_IB_DISABLE=0
    export NCCL_NET_GDR_LEVEL=5
    export NCCL_SOCKET_IFNAME=eth0
    export NCCL_TIMEOUT=1800

    # Reduce CUDA allocator fragmentation when generate() (large transient KV cache)
    # is interleaved with ppo_update (large optimizer moments). Saves 5-10% headroom
    # on A100-80GB at 8B. See ppo_specs/specs/memory_optimization.md §11.8.
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

# ── W&B ──────────────────────────────────────────────────────────────────────
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_MODE="${WANDB_MODE}"
export WANDB_RUN_GROUP="e2.8"

# ── Preemption handler ──────────────────────────────────────────────────────
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

# ── Build and run command ────────────────────────────────────────────────────
ARGS="--seed ${SEED}"
if [ "${LOCAL_TEST}" = "true" ]; then
    ARGS="${ARGS} --local-test"
fi
# Pass MODEL_NAME through to Python — env var alone never reached the script.
if [ -n "${MODEL_NAME:-}" ]; then
    ARGS="${ARGS} --model-name ${MODEL_NAME}"
fi
# 8B mitigation stack — engage when CLUSTER_8B=1 OR when MODEL_NAME contains
# '8B'. Without these flags an 8B model will OOM during AdamW step.
# (E2.8 doesn't yet support resume so --checkpoint-every / --resume-from
# are not auto-injected here; pass via CLI if the SLURM script user wants.)
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

run_capacity() {
    local cap=$1
    local extra_args="${ARGS}"
    if [ -n "${cap}" ]; then
        extra_args="${extra_args} --capacity ${cap}"
    fi

    export WANDB_TAGS="ppo,e2.8,${cap},seed${SEED}"

    if [ "${SLURM_MODE}" = "multigpu" ]; then
        echo "[RUN] accelerate launch --config_file ${ACCEL_CONFIG} --num_processes ${NUM_GPUS} ppo_specs/run_e2_8.py ${extra_args}"
        accelerate launch \
            --config_file "${ACCEL_CONFIG}" \
            --num_processes "${NUM_GPUS}" \
            ppo_specs/run_e2_8.py ${extra_args}
    else
        # Per-capacity launch via accelerate so the Accelerator code path is
        # always exercised. Under DEVICE_MODE=cpu this uses MULTI_CPU/gloo;
        # under DEVICE_MODE=gpu it uses accelerate_single_gpu.yaml (distributed_type: NO).
        local launch_num_proc="${NUM_PROCESSES}"
        if [ "${DEVICE_MODE}" != "cpu" ]; then
            launch_num_proc=1
        fi
        echo "[RUN] accelerate launch --config_file ${ACCEL_CONFIG} --num_processes ${launch_num_proc} ppo_specs/run_e2_8.py ${extra_args}"
        accelerate launch \
            --config_file "${ACCEL_CONFIG}" \
            --num_processes "${launch_num_proc}" \
            ppo_specs/run_e2_8.py ${extra_args}
    fi
}

if [ "${SLURM_MODE}" = "sweep" ]; then
    # Sequential sweep: all capacities in one job
    echo "[SWEEP] Running all 4 critic capacities sequentially"
    for cap in "${ALL_CAPACITIES[@]}"; do
        echo ""
        echo "========================================="
        echo "  Critic capacity: ${cap}"
        echo "========================================="
        run_capacity "${cap}" &
        CHILD_PID=$!
        wait $CHILD_PID
    done
else
    # parallel or multigpu: run the selected capacity
    run_capacity "${CAPACITY}" &
    CHILD_PID=$!
    wait $CHILD_PID
fi

echo "[DONE] E2.8 completed. Exit code: $?"
echo "[DONE] Results in results/ppo_e2_8_*_seed${SEED}.json"
