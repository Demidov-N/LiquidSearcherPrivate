#!/bin/bash
#SBATCH --job-name=liq-features
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-preprocess-%j.out
#SBATCH --error=logs/slurm-preprocess-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/users/mvillal4/scratch/liquid/LiquidSearcherPrivate}"
PYTHON_BIN="${PYTHON_BIN:-/users/mvillal4/.pyenv/versions/3.11.9/bin/python}"

cd "$PROJECT_ROOT"
mkdir -p logs data/processed

export PYTHONUNBUFFERED=1

echo "Project root: $PROJECT_ROOT"
echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" --version

"$PYTHON_BIN" -m scripts.preprocess_features \
    --universe all_crsp \
    --start-date 2010-01-01 \
    --end-date 2024-12-31 \
    --batch-size 500 \
    --output data/processed/all_features.parquet
