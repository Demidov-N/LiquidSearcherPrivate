#!/usr/bin/env bash
# Convenience wrapper for training with default parameters
#
# Usage:
#   ./scripts/train.sh                              # Fresh training with defaults
#   ./scripts/train.sh --resume                     # Resume from last.ckpt
#   ./scripts/train.sh --resume-from path/to/ckpt   # Resume from specific checkpoint

set -euo pipefail

# Activate virtual environment if it exists
if [[ -d ".venv" ]]; then
    source .venv/bin/activate
elif [[ -d "venv" ]]; then
    source venv/bin/activate
fi

# Default training parameters
FEATURE_DIR="data/processed/features"
TRAIN_START="2010-01-01"
TRAIN_END="2018-12-31"
VAL_START="2020-01-01"
VAL_END="2020-12-31"
BATCH_SIZE=32
EPOCHS=100
LR="1e-4"
NUM_WORKERS=4
CHECKPOINT_DIR="checkpoints"
SEED=42
WARMUP_EPOCHS=5

# Check for resume flags
RESUME_FLAG=""
if [[ "${1:-}" == "--resume" ]]; then
    RESUME_FLAG="--resume-from-last"
elif [[ "${1:-}" == "--resume-from" ]]; then
    if [[ -z "${2:-}" ]]; then
        echo "Error: --resume-from requires a checkpoint path"
        exit 1
    fi
    RESUME_FLAG="--resume-from $2"
fi

echo "🚀 Starting LiquidSearcher Training"
echo "=================================="
echo "Feature Dir:    $FEATURE_DIR"
echo "Train Period:   $TRAIN_START to $TRAIN_END"
echo "Val Period:     $VAL_START to $VAL_END"
echo "Batch Size:     $BATCH_SIZE"
echo "Workers:        $NUM_WORKERS"
echo "Epochs:         $EPOCHS"
echo "Learning Rate:  $LR"
if [[ -n "$RESUME_FLAG" ]]; then
    echo "Resume:         $RESUME_FLAG"
fi
echo "=================================="

# Run training with defaults
python -m scripts.train \
    --feature-dir "$FEATURE_DIR" \
    --train-start "$TRAIN_START" \
    --train-end "$TRAIN_END" \
    --val-start "$VAL_START" \
    --val-end "$VAL_END" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --num-workers "$NUM_WORKERS" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --seed "$SEED" \
    --warmup-epochs "$WARMUP_EPOCHS" \
    $RESUME_FLAG

echo "✅ Training completed successfully!"