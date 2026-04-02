#!/bin/bash
#SBATCH --job-name=liq-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-smoke-%j.out
#SBATCH --error=logs/slurm-smoke-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/users/mvillal4/scratch/liquid/LiquidSearcherPrivate}"
PYTHON_BIN="${PYTHON_BIN:-/users/mvillal4/.pyenv/versions/3.11.9/bin/python}"

cd "$PROJECT_ROOT"
mkdir -p logs checkpoints_cross_smoke

export PYTHONUNBUFFERED=1

echo "Project root: $PROJECT_ROOT"
echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" --version

# Avoid Lightning picking up partial interactive Slurm state.
unset SLURM_NTASKS SLURM_PROCID SLURM_LOCALID

"$PYTHON_BIN" -m scripts.train_cross_attention_rtx5000 \
    --feature-dir data/processed/all_features.parquet \
    --epochs 1 \
    --batch-size 16 \
    --num-workers 0 \
    --samples-per-epoch 200 \
    --checkpoint-dir checkpoints_cross_smoke
