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

cd ~/LiquidSearcherPrivate
mkdir -p logs checkpoints_cross_smoke

# Prefer project venv, otherwise use the active python (for pyenv-based setups).
if [[ -d ".venv" ]]; then
    source .venv/bin/activate
fi

echo "Python: $(which python)"
python --version

# Avoid Lightning picking up partial interactive Slurm state.
unset SLURM_NTASKS SLURM_PROCID SLURM_LOCALID

python -m scripts.train_cross_attention_rtx5000 \
    --feature-dir data/processed/all_features.parquet \
    --epochs 1 \
    --batch-size 16 \
    --num-workers 0 \
    --samples-per-epoch 200 \
    --checkpoint-dir checkpoints_cross_smoke
