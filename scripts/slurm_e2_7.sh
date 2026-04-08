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
#SBATCH --partition=${PARTITION:-gpu}
#SBATCH --account=${ACCOUNT:-default}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --signal=B:SIGUSR1@120

# ── Configurable parameters (override via --export or environment) ───────────
SLURM_MODE="${SLURM_MODE:-single}"       # single | multigpu | array
NUM_GPUS="${NUM_GPUS:-1}"                 # GPUs per node (for multigpu mode)
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

# ── Environment setup ────────────────────────────────────────────────────────
set -euo pipefail
mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/results"

# Load modules (adjust to your cluster)
module purge 2>/dev/null || true
module load cuda/12.1 2>/dev/null || true
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
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_SOCKET_IFNAME=eth0
export NCCL_P2P_LEVEL=NVL
# Timeout for NCCL collectives (seconds) - increase for large models
export NCCL_TIMEOUT=1800

# ── W&B ──────────────────────────────────────────────────────────────────────
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_MODE="${WANDB_MODE}"
export WANDB_RUN_GROUP="e2.7"
export WANDB_TAGS="ppo,e2.7,seed${SEED}"

# ── Preemption handler ───────────────────────────────────────────────────────
# SLURM sends SIGUSR1 120s before killing the job. We trap it to save state.
handle_preempt() {
    echo "[PREEMPT] Received SIGUSR1 — job will be killed soon."
    echo "[PREEMPT] Requeueing job ${SLURM_JOB_ID}..."
    # If checkpoint saving were implemented, trigger it here:
    # kill -SIGUSR1 $CHILD_PID  # forward to Python process
    scontrol requeue "${SLURM_JOB_ID}" 2>/dev/null || true
    exit 0
}
trap handle_preempt SIGUSR1

# ── Build command ────────────────────────────────────────────────────────────
ARGS="--seed ${SEED}"
if [ "${LOCAL_TEST}" = "true" ]; then
    ARGS="${ARGS} --local-test --no-mc"
fi

if [ "${SLURM_MODE}" = "single" ] || [ "${SLURM_MODE}" = "array" ]; then
    # Single-GPU: run directly with python
    echo "[RUN] python ppo_specs/run_e2_7.py ${ARGS}"
    python ppo_specs/run_e2_7.py ${ARGS} &
    CHILD_PID=$!
    wait $CHILD_PID

elif [ "${SLURM_MODE}" = "multigpu" ]; then
    # Multi-GPU via accelerate
    ACCEL_CONFIG="configs/accelerate_multi_gpu.yaml"
    echo "[RUN] accelerate launch --config_file ${ACCEL_CONFIG} --num_processes ${NUM_GPUS}"
    accelerate launch \
        --config_file "${ACCEL_CONFIG}" \
        --num_processes "${NUM_GPUS}" \
        ppo_specs/run_e2_7.py ${ARGS} &
    CHILD_PID=$!
    wait $CHILD_PID
fi

echo "[DONE] E2.7 seed=${SEED} completed. Exit code: $?"
echo "[DONE] Results: results/ppo_e2_7_seed${SEED}.json"
