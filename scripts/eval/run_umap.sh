#!/bin/bash
# UMAP Visualization for LiquidSearcher
#
# Usage:
#   ./run_umap.sh [OPTIONS]
#
# Quick test (1000 samples):
#   ./run_umap.sh --quick
#
# Production run (all samples, all periods):
#   ./run_umap.sh

set -e

# Default values
PERIODS=("covid_pre" "covid_crisis" "covid_recovery")
MAX_SAMPLES=0  # 0 = all samples
OUTPUT_DIR="results/figures/umap"
CHECKPOINT="checkpoints/last.ckpt"
FEATURES="data/processed/all_features.parquet"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            MAX_SAMPLES=1000
            PERIODS=("covid_pre")
            OUTPUT_DIR="results/figures/umap_test"
            shift
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --period)
            PERIODS=("$2")
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "LIQUIDSEARCHER UMAP VISUALIZATION"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Periods:      ${PERIODS[*]}"
echo "  Max samples:  $MAX_SAMPLES (0 = all)"
echo "  Output:       $OUTPUT_DIR"
echo ""

# Check dependencies
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found. Install from: https://docs.astral.sh/uv/"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$FEATURES" ]; then
    echo "ERROR: Features file not found: $FEATURES"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run UMAP for each period
for PERIOD in "${PERIODS[@]}"; do
    echo "Processing period: $PERIOD"
    
    SAMPLES_ARG=""
    if [ "$MAX_SAMPLES" -gt 0 ]; then
        SAMPLES_ARG="--max-samples $MAX_SAMPLES"
    fi
    
    uv run python -m scripts.visualization.umap_plots \
        --checkpoint "$CHECKPOINT" \
        --features "$FEATURES" \
        --period "$PERIOD" \
        --output-dir "$OUTPUT_DIR" \
        $SAMPLES_ARG
    
    echo ""
done

echo "========================================"
echo "VISUALIZATION COMPLETE"
echo "========================================"
echo ""
echo "Results:"
echo "  Figures: $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR" 2>/dev/null || echo "  (no files generated)"
echo ""