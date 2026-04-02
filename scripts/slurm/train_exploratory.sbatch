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

cd ~/LiquidSearcherPrivate
mkdir -p logs checkpoints_cross

# Prefer project venv, otherwise use the active python (for pyenv-based setups).
if [[ -d ".venv" ]]; then
    source .venv/bin/activate
fi

echo "Python: $(which python)"
python --version
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Avoid Lightning picking up partial interactive Slurm state when launched from OOD/VSCode.
unset SLURM_NTASKS SLURM_PROCID SLURM_LOCALID

python -m scripts.train_cross_attention_rtx5000 \
    --feature-dir data/processed/all_features.parquet \
    --epochs 50 \
    --batch-size 64 \
    --num-workers 4 \
    --checkpoint-dir checkpoints_cross \
    --resume-from-last
