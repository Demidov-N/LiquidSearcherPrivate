#!/bin/bash
#SBATCH --job-name=liq-shard
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-shard-%j.out
#SBATCH --error=logs/slurm-shard-%j.err

set -euo pipefail

cd ~/LiquidSearcherPrivate
mkdir -p logs data/processed/features

# Prefer project venv, otherwise use the active python (for pyenv-based setups).
if [[ -d ".venv" ]]; then
    source .venv/bin/activate
fi

echo "Python: $(which python)"
python --version

POLARS_MAX_THREADS=1 PYARROW_NUM_THREADS=1 python -m scripts.shard_by_symbol \
    --src data/processed/all_features.parquet \
    --out data/processed/features \
    --resume \
    --workers 1
