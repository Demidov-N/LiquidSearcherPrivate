#!/bin/bash
#SBATCH --job-name=liq-xattn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=47:59:00
#SBATCH --output=logs/slurm-train-%j.out
#SBATCH --error=logs/slurm-train-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/users/mvillal4/scratch/liquid/LiquidSearcherPrivate}"
PYTHON_BIN="${PYTHON_BIN:-/users/mvillal4/.pyenv/versions/3.11.9/bin/python}"

cd "$PROJECT_ROOT"
mkdir -p logs checkpoints_cross

export PYTHONUNBUFFERED=1

echo "Project root: $PROJECT_ROOT"
echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" --version
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

unset SLURM_NTASKS SLURM_PROCID SLURM_LOCALID

"$PYTHON_BIN" -m scripts.train_cross_attention_rtx5000 \
    --feature-dir data/processed/all_features.parquet \
    --epochs 50 \
    --batch-size 64 \
    --num-workers 4 \
    --checkpoint-dir checkpoints_cross \
    --resume-from-last
