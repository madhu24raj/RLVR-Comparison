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
accelerate launch \
    --config_file configs/accelerate_cpu.yaml \
    --num_processes "${NUM_PROCESSES}" \
    ppo_specs/run_e2_7.py \
    --seed "${SEED}" \
    --checkpoint-every 20 \
    --resume-from auto \
    2>&1 | tee -a "${LOG_FILE}"

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
