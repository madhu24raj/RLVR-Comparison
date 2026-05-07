#!/bin/bash
###############################################################################
# Sequential single-GPU run on Qwen-0.5B:
#   Stage 1: E2.7 head-to-head (200 steps)
#   Stage 2: E2.8 critic sweep (4 capacities × 150 steps each)
#
# Configuration (locked in by this script):
#   - Single-process single-GPU (NUM_PROCESSES=1) — no DDP overhead
#   - Reward:          deterministic gsm8k_reward (config default)
#   - Reference KL:    enabled (reference_kl_coeff=0.01 from configs)
#   - Checkpoint:      every 20 steps + final
#   - Logging cadence: every 5 steps (JSON metrics row)
#   - Resume:          fresh start (no auto-resume)
#   - Memory flags:    gradient_checkpointing + optimizer-fused +
#                      length-bucketed-generation (all bf16-friendly,
#                      no extra deps required for Qwen-0.5B)
#   - Allocator:       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#                      (helps avoid OOM from PPO's generate→backward
#                      memory-pattern fragmentation)
#
# Designed to survive SSH disconnect — wrap in tmux:
#
#   tmux new -s rlvr_gpu
#   bash scripts/run_full_gpu.sh
#   # Ctrl-b d to detach. Safe to log out / shut down laptop.
#   # Reconnect later with: tmux attach -t rlvr_gpu
#
# Outputs (after both stages finish, ~3-6 hours total on a V100/A100):
#   results/ppo_e2_7_seed42.json                      ← E2.7 metrics
#   results/checkpoints/ppo_e2_7_seed42/.../model/    ← E2.7 final HF-loadable
#   results/ppo_e2_8_{capacity}_seed42.json           ← E2.8 per-capacity metrics
#   results/e2_8_sweep_summary.json                   ← E2.8 cross-capacity table
#   results/checkpoints/ppo_e2_8_{capacity}_seed42/   ← E2.8 final per capacity
###############################################################################

set -euo pipefail

# ── Conda env ─────────────────────────────────────────────────────────────────
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
LOG_FILE="logs/sequential_e2_7_e2_8_seed${SEED}_${TS}.log"

# ── Common env ────────────────────────────────────────────────────────────────
export PYTHONUTF8=1
# Help avoid CUDA OOM from fragmentation. PPO does generate (KV cache spike)
# then backward (activation spike) repeatedly — expandable_segments lets the
# allocator return memory to the pool between phases instead of caching it
# in fixed-size chunks.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Memory-saving flags (all safe on Qwen-0.5B without bnb) ──────────────────
# NOT included by default: --optimizer-8bit, --reference-quant int8.
# Both require bitsandbytes; for 0.5B the savings are small (~3 GB and
# ~0.5 GB respectively). The trainer tolerates missing bnb gracefully but
# prints a warning. Add them if your env has bitsandbytes installed.
MEM_FLAGS=(
    --gradient-checkpointing
    --optimizer-fused
    --length-bucketed-generation
)

# ── Move stale checkpoints aside (so this is genuinely a fresh start) ────────
# We don't pass --resume-from, so the trainer starts fresh either way; but
# leaving an old checkpoint dir means new periodic saves land alongside it
# and the directory becomes confusing. Move existing dirs to *_BAK_<ts>.
for stale in \
    "results/checkpoints/ppo_e2_7_seed${SEED}" \
    "results/checkpoints/ppo_e2_8_none_seed${SEED}" \
    "results/checkpoints/ppo_e2_8_small_seed${SEED}" \
    "results/checkpoints/ppo_e2_8_medium_seed${SEED}" \
    "results/checkpoints/ppo_e2_8_large_seed${SEED}"; do
    if [ -d "${stale}" ]; then
        mv "${stale}" "${stale}_BAK_${TS}"
        echo "[fresh-start] moved ${stale} -> ${stale}_BAK_${TS}"
    fi
done

# ── Sanity ────────────────────────────────────────────────────────────────────
if ! command -v accelerate >/dev/null 2>&1; then
    echo "[ERROR] 'accelerate' not on PATH. conda activate ${CONDA_ENV}"
    exit 1
fi

# ── Banner ────────────────────────────────────────────────────────────────────
{
    echo "============================================="
    echo "  Sequential GPU run: E2.7 → E2.8"
    echo "  Qwen2.5-0.5B-Instruct, deterministic reward"
    echo "============================================="
    echo "  Started:         $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Project dir:     ${PROJECT_DIR}"
    echo "  Conda env:       ${CONDA_DEFAULT_ENV:-(none)}"
    echo "  Seed:            ${SEED}"
    echo "  Memory flags:    ${MEM_FLAGS[*]}"
    echo "  Allocator:       PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
    echo "  Log file:        ${LOG_FILE}"
    echo "  Driver PID:      $$"
    echo "============================================="
} | tee -a "${LOG_FILE}"

# ── Stage 1: E2.7 (head-to-head, 200 steps) ───────────────────────────────────
{
    echo
    echo "============================================="
    echo "  STAGE 1: E2.7 — head-to-head (200 steps)"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================="
} | tee -a "${LOG_FILE}"

accelerate launch \
    --config_file configs/accelerate_single_gpu.yaml \
    --num_processes 1 \
    ppo_specs/run_e2_7.py \
    --seed "${SEED}" \
    --checkpoint-every 20 \
    --log-every 5 \
    "${MEM_FLAGS[@]}" \
    2>&1 | tee -a "${LOG_FILE}"

E27_EXIT=${PIPESTATUS[0]}
{
    echo
    echo "  E2.7 finished at $(date '+%Y-%m-%d %H:%M:%S') exit=${E27_EXIT}"
} | tee -a "${LOG_FILE}"

if [ "${E27_EXIT}" -ne 0 ]; then
    echo "[ABORT] E2.7 failed (exit ${E27_EXIT}); skipping E2.8." | tee -a "${LOG_FILE}"
    exit "${E27_EXIT}"
fi

# ── Stage 2: E2.8 (critic sweep, 4 capacities × 150 steps) ────────────────────
{
    echo
    echo "============================================="
    echo "  STAGE 2: E2.8 — critic sweep (4 caps × 150 steps)"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================="
} | tee -a "${LOG_FILE}"

accelerate launch \
    --config_file configs/accelerate_single_gpu.yaml \
    --num_processes 1 \
    ppo_specs/run_e2_8.py \
    --seed "${SEED}" \
    --checkpoint-every 20 \
    --log-every 5 \
    "${MEM_FLAGS[@]}" \
    2>&1 | tee -a "${LOG_FILE}"

E28_EXIT=${PIPESTATUS[0]}

# ── Done ──────────────────────────────────────────────────────────────────────
{
    echo
    echo "============================================="
    echo "  Sequential run COMPLETE"
    echo "  Finished:        $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  E2.7 exit:       ${E27_EXIT}"
    echo "  E2.8 exit:       ${E28_EXIT}"
    echo "============================================="
    echo "  E2.7 metrics:    results/ppo_e2_7_seed${SEED}.json"
    echo "  E2.7 model:      results/checkpoints/ppo_e2_7_seed${SEED}/checkpoint_step_000199/model/"
    echo "  E2.8 sweep:      results/e2_8_sweep_summary.json"
    echo "  E2.8 per-cap:    results/ppo_e2_8_{none,small,medium,large}_seed${SEED}.json"
    echo "  E2.8 models:     results/checkpoints/ppo_e2_8_{capacity}_seed${SEED}/checkpoint_step_000149/model/"
    echo "============================================="
} | tee -a "${LOG_FILE}"

exit "${E28_EXIT}"
