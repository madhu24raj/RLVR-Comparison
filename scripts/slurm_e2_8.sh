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
#SBATCH --partition=${PARTITION:-gpu}
#SBATCH --account=${ACCOUNT:-default}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --signal=B:SIGUSR1@120

# ── Configurable parameters ─────────────────────────────────────────────────
SLURM_MODE="${SLURM_MODE:-sweep}"          # sweep | parallel | multigpu
NUM_GPUS="${NUM_GPUS:-1}"
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

# ── Environment setup ───────────────────────────────────────────────────────
set -euo pipefail
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/results"

module purge 2>/dev/null || true
module load cuda/12.1 2>/dev/null || true
module load anaconda3 2>/dev/null || true

eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate "${CONDA_ENV}" 2>/dev/null || true

cd "${PROJECT_DIR}"

# ── HuggingFace / model cache ───────────────────────────────────────────────
export HF_HOME="${MODEL_CACHE}"
export TRANSFORMERS_CACHE="${MODEL_CACHE}"
export HF_DATASETS_CACHE="${MODEL_CACHE}/datasets"

# ── NCCL tuning ──────────────────────────────────────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_SOCKET_IFNAME=eth0
export NCCL_TIMEOUT=1800

# ── W&B ──────────────────────────────────────────────────────────────────────
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_MODE="${WANDB_MODE}"
export WANDB_RUN_GROUP="e2.8"

# ── Preemption handler ──────────────────────────────────────────────────────
handle_preempt() {
    echo "[PREEMPT] Received SIGUSR1 — requeueing job ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}" 2>/dev/null || true
    exit 0
}
trap handle_preempt SIGUSR1

# ── Build and run command ────────────────────────────────────────────────────
ARGS="--seed ${SEED}"
if [ "${LOCAL_TEST}" = "true" ]; then
    ARGS="${ARGS} --local-test"
fi

run_capacity() {
    local cap=$1
    local extra_args="${ARGS}"
    if [ -n "${cap}" ]; then
        extra_args="${extra_args} --capacity ${cap}"
    fi

    export WANDB_TAGS="ppo,e2.8,${cap},seed${SEED}"

    if [ "${SLURM_MODE}" = "multigpu" ]; then
        echo "[RUN] accelerate launch --num_processes ${NUM_GPUS} -- capacity=${cap}"
        accelerate launch \
            --config_file configs/accelerate_multi_gpu.yaml \
            --num_processes "${NUM_GPUS}" \
            ppo_specs/run_e2_8.py ${extra_args}
    else
        echo "[RUN] python ppo_specs/run_e2_8.py ${extra_args}"
        python ppo_specs/run_e2_8.py ${extra_args}
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
