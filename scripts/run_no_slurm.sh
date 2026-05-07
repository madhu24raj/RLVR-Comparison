#!/bin/bash
###############################################################################
# Run full E2.7 PPO experiment on Qwen-0.5B without SLURM.
#
# Configuration (from e2_7_config defaults — no overrides needed):
#   - Model:           Qwen/Qwen2.5-0.5B-Instruct
#   - Reward:          deterministic gsm8k_reward (reward_mode="deterministic")
#   - Reference KL:    enabled (reference_kl_coeff=0.01)
#   - Steps:           200
#   - Batch size:      16 (divisible by NUM_PROCESSES=8 → 2 per rank)
#   - log_every:       5  → JSON row every 5 steps (~40 rows total)
#   - eval_every:      20 → richer eval rows every 20 steps
#   - checkpoint:      every 20 steps + final, last 3 rotated + unrotated final
#
# Resource layout (48 CPUs):
#   - NUM_PROCESSES=8, OMP_NUM_THREADS=6  → 8 × 6 = 48 cores, no contention.
#
# Usage:
#   tmux new -s rlvr         # so SSH disconnects don't kill the run
#   bash scripts/run_no_slurm.sh
#   # Ctrl-b d to detach; tmux attach -t rlvr to come back.
#
# Or background with nohup (no live output):
#   nohup bash scripts/run_no_slurm.sh > /dev/null 2>&1 &
###############################################################################

set -euo pipefail

# ── Resource configuration ────────────────────────────────────────────────────
# 48 cores: 8 procs × 6 threads each. e2_7_config.batch_size=16 is divisible
# by 8 (each rank handles 2 samples per global batch). 6 BLAS threads/rank
# is plenty for Qwen-0.5B matmul.
export NUM_PROCESSES="${NUM_PROCESSES:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"
# UTF-8 stdout for the spec's →/… characters (cp1252 default crashes on them).
export PYTHONUTF8=1
# Force CPU even on a node with a visible GPU. The trainer's USE_DDP branch
# checks LOCAL_RANK and then derives device from accelerator.device; the
# legacy single-process branch falls through to `torch.cuda.is_available()`.
# Hiding CUDA at the env layer makes both paths land on CPU regardless of
# which launcher path actually fires.
export CUDA_VISIBLE_DEVICES=""
# Auto-resume can silently restart a finished run with zero training. Set
# RESUME_FROM=auto explicitly to opt in; default to a fresh run.
RESUME_FROM="${RESUME_FROM:-}"
# Launcher choice: torchrun is more reliable than accelerate launch for
# CPU multi-process — it ALWAYS sets LOCAL_RANK, whereas some accelerate
# versions collapse to single-process despite --num_processes overrides.
# Override with LAUNCHER=accelerate to use accelerate launch instead.
LAUNCHER="${LAUNCHER:-torchrun}"

# ── Conda env activation ──────────────────────────────────────────────────────
CONDA_ENV="${CONDA_ENV:-rlvr}"
if [ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]; then
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "${CONDA_ENV}"
    else
        echo "[WARN] conda not on PATH — assuming the right env is already active."
    fi
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p logs results results/checkpoints

SEED="${SEED:-42}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/e2_7_qwen05b_seed${SEED}_${TS}.log"

# ── Sanity: accelerate must be importable ─────────────────────────────────────
if ! command -v accelerate >/dev/null 2>&1; then
    echo "[ERROR] 'accelerate' command not found. Activate the conda env first:"
    echo "        conda activate ${CONDA_ENV}"
    exit 1
fi

# ── Banner ────────────────────────────────────────────────────────────────────
{
    echo "============================================="
    echo "  E2.7 PPO — Qwen-0.5B — no-SLURM run"
    echo "============================================="
    echo "  Started:         $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Project dir:     ${PROJECT_DIR}"
    echo "  Conda env:       ${CONDA_DEFAULT_ENV:-(none)}"
    echo "  NUM_PROCESSES:   ${NUM_PROCESSES}"
    echo "  OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
    echo "  Seed:            ${SEED}"
    echo "  Log file:        ${LOG_FILE}"
    echo "  Driver PID:      $$"
    echo "============================================="
    echo
} | tee -a "${LOG_FILE}"

# ── Launch ────────────────────────────────────────────────────────────────────
# Defaults engaged (no env overrides): reward_mode=deterministic,
# reward_model_capacity=none, log_every=5, eval_every=20, checkpoint_every=20.
PY_ARGS=(
    ppo_specs/run_e2_7.py
    --seed "${SEED}"
    --checkpoint-every 20
)
# Only inject --resume-from when the user opted in. Empty default means
# fresh run (no auto-resume from a prior finished checkpoint).
if [ -n "${RESUME_FROM}" ]; then
    PY_ARGS+=(--resume-from "${RESUME_FROM}")
fi

if [ "${LAUNCHER}" = "torchrun" ]; then
    echo "[run_no_slurm] launcher=torchrun nproc_per_node=${NUM_PROCESSES}" | tee -a "${LOG_FILE}"
    torchrun --nproc_per_node="${NUM_PROCESSES}" --standalone \
        "${PY_ARGS[@]}" \
        2>&1 | tee -a "${LOG_FILE}"
elif [ "${LAUNCHER}" = "accelerate" ]; then
    echo "[run_no_slurm] launcher=accelerate num_processes=${NUM_PROCESSES}" | tee -a "${LOG_FILE}"
    accelerate launch \
        --config_file configs/accelerate_cpu.yaml \
        --num_processes "${NUM_PROCESSES}" \
        "${PY_ARGS[@]}" \
        2>&1 | tee -a "${LOG_FILE}"
else
    echo "[ERROR] Unknown LAUNCHER='${LAUNCHER}'. Use 'torchrun' or 'accelerate'."
    exit 2
fi

EXIT=${PIPESTATUS[0]}

{
    echo
    echo "============================================="
    echo "  Finished:        $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Exit code:       ${EXIT}"
    echo "  Final model:     results/checkpoints/ppo_e2_7_seed${SEED}/checkpoint_step_000199/model/"
    echo "  Metrics JSON:    results/ppo_e2_7_seed${SEED}.json"
    echo "============================================="
} | tee -a "${LOG_FILE}"

exit "${EXIT}"
