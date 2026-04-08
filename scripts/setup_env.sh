#!/bin/bash
###############################################################################
# Environment Setup Script for RLVR-Comparison PPO Experiments
#
# Creates a conda environment, installs all dependencies, and pre-downloads
# model weights so that SLURM jobs do not download during training.
#
# Usage:
#   bash scripts/setup_env.sh                     # defaults
#   bash scripts/setup_env.sh --env-name my_env   # custom env name
#   bash scripts/setup_env.sh --skip-models        # skip model download
#   bash scripts/setup_env.sh --large-model        # also download Llama-3-8B
#
# Run this on a login node or interactive session with internet access.
###############################################################################

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
ENV_NAME="rlvr"
PYTHON_VERSION="3.11"
CUDA_VERSION="12.1"
SKIP_MODELS=false
DOWNLOAD_LARGE=false
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_CACHE="${PROJECT_DIR}/.model_cache"

# ── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --env-name)     ENV_NAME="$2"; shift 2 ;;
        --python)       PYTHON_VERSION="$2"; shift 2 ;;
        --cuda)         CUDA_VERSION="$2"; shift 2 ;;
        --skip-models)  SKIP_MODELS=true; shift ;;
        --large-model)  DOWNLOAD_LARGE=true; shift ;;
        --cache-dir)    MODEL_CACHE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================="
echo "  RLVR Environment Setup"
echo "============================================="
echo "  Env name:       ${ENV_NAME}"
echo "  Python:         ${PYTHON_VERSION}"
echo "  CUDA:           ${CUDA_VERSION}"
echo "  Project dir:    ${PROJECT_DIR}"
echo "  Model cache:    ${MODEL_CACHE}"
echo "============================================="

# ── Step 1: Create conda environment ────────────────────────────────────────
echo ""
echo "[1/5] Creating conda environment '${ENV_NAME}'..."

# Load modules (cluster-specific, adjust as needed)
module purge 2>/dev/null || true
module load anaconda3 2>/dev/null || true

eval "$(conda shell.bash hook)" 2>/dev/null || true

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Environment '${ENV_NAME}' already exists. Updating..."
    conda activate "${ENV_NAME}"
else
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    conda activate "${ENV_NAME}"
fi

echo "  Python: $(python --version)"

# ── Step 2: Install PyTorch with CUDA ────────────────────────────────────────
echo ""
echo "[2/5] Installing PyTorch with CUDA ${CUDA_VERSION}..."

# Map CUDA version to PyTorch index URL
case "${CUDA_VERSION}" in
    11.8) TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
    12.1) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
    12.4) TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
    *)    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
          echo "  Warning: CUDA ${CUDA_VERSION} not recognized, defaulting to cu121" ;;
esac

pip install torch torchvision torchaudio --index-url "${TORCH_INDEX}"

# Verify CUDA
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU count: {torch.cuda.device_count()}')
    print(f'  GPU 0: {torch.cuda.get_device_name(0)}')
"

# ── Step 3: Install project dependencies ─────────────────────────────────────
echo ""
echo "[3/5] Installing project dependencies from requirements.txt..."

cd "${PROJECT_DIR}"
pip install -r requirements.txt

# Additional cluster-useful packages
pip install tensorboard 2>/dev/null || true

# Verify key imports
python -c "
import transformers, datasets, accelerate, trl, wandb
print(f'  transformers: {transformers.__version__}')
print(f'  datasets:     {datasets.__version__}')
print(f'  accelerate:   {accelerate.__version__}')
print(f'  trl:          {trl.__version__}')
print(f'  wandb:        {wandb.__version__}')
"

# ── Step 4: Pre-download model weights ───────────────────────────────────────
echo ""
echo "[4/5] Pre-downloading model weights..."

mkdir -p "${MODEL_CACHE}"
export HF_HOME="${MODEL_CACHE}"
export TRANSFORMERS_CACHE="${MODEL_CACHE}"
export HF_DATASETS_CACHE="${MODEL_CACHE}/datasets"

if [ "${SKIP_MODELS}" = true ]; then
    echo "  Skipping model download (--skip-models)"
else
    # Small model (used for local tests and default config)
    echo "  Downloading Qwen/Qwen2.5-0.5B-Instruct..."
    python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
print(f'    Downloading tokenizer for {model_name}...')
AutoTokenizer.from_pretrained(model_name)
print(f'    Downloading model weights for {model_name}...')
AutoModelForCausalLM.from_pretrained(model_name)
print(f'    Done: {model_name}')
"

    if [ "${DOWNLOAD_LARGE}" = true ]; then
        echo "  Downloading meta-llama/Meta-Llama-3-8B-Instruct..."
        echo "  (Requires HF_TOKEN set for gated models)"
        python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
print(f'    Downloading tokenizer for {model_name}...')
AutoTokenizer.from_pretrained(model_name)
print(f'    Downloading model weights for {model_name}...')
AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
print(f'    Done: {model_name}')
"
    fi

    # Pre-download GSM8K dataset
    echo "  Downloading GSM8K dataset..."
    python -c "
from datasets import load_dataset
ds = load_dataset('openai/gsm8k', 'main')
print(f'    GSM8K train: {len(ds[\"train\"])} examples')
print(f'    GSM8K test:  {len(ds[\"test\"])} examples')
"
fi

# ── Step 5: Create directory structure ───────────────────────────────────────
echo ""
echo "[5/5] Creating directory structure..."

mkdir -p "${PROJECT_DIR}/results"
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/checkpoints"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo "  Setup complete!"
echo "============================================="
echo ""
echo "  To activate:  conda activate ${ENV_NAME}"
echo "  Model cache:  ${MODEL_CACHE}"
echo ""
echo "  Quick verification:"
echo "    conda activate ${ENV_NAME}"
echo "    cd ${PROJECT_DIR}"
echo "    python ppo_specs/run_e2_7.py --local-test --no-mc"
echo ""
echo "  Submit to SLURM:"
echo "    sbatch scripts/slurm_e2_7.sh"
echo "    sbatch --export=ALL,SLURM_MODE=parallel --array=0-3 scripts/slurm_e2_8.sh"
echo ""
