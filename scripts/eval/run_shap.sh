#!/bin/bash
# SHAP Feature Importance Analysis for LiquidSearcher
# 
# Usage:
#   ./run_shap.sh [OPTIONS]
#
# Quick test (3 queries, 5 candidates):
#   ./run_shap.sh --quick
#
# Production run (50 queries, 50 candidates):
#   ./run_shap.sh
#
# Custom run:
#   ./run_shap.sh --n-queries 20 --n-candidates 30

set -e

# Default values for production run
N_QUERIES=50
N_CANDIDATES=50
BACKGROUND_SIZE=100
PERIOD_START="2019-01-01"
PERIOD_END="2019-12-31"
OUTPUT_DIR="results/shap"
CHECKPOINT="checkpoints/last.ckpt"
FEATURES="data/processed/all_features.parquet"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            N_QUERIES=5
            N_CANDIDATES=10
            BACKGROUND_SIZE=50
            OUTPUT_DIR="results/shap_test"
            shift
            ;;
        --n-queries)
            N_QUERIES="$2"
            shift 2
            ;;
        --n-candidates)
            N_CANDIDATES="$2"
            shift 2
            ;;
        --background-size)
            BACKGROUND_SIZE="$2"
            shift 2
            ;;
        --period-start)
            PERIOD_START="$2"
            shift 2
            ;;
        --period-end)
            PERIOD_END="$2"
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
echo "LIQUIDSEARCHER SHAP ANALYSIS"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Queries:        $N_QUERIES"
echo "  Candidates:     $N_CANDIDATES (per query)"
echo "  Background:     $BACKGROUND_SIZE samples"
echo "  Period:         $PERIOD_START to $PERIOD_END"
echo "  Output:         $OUTPUT_DIR"
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
mkdir -p "$OUTPUT_DIR/per_query"
mkdir -p "$OUTPUT_DIR/figures"

# Run SHAP analysis
echo "Starting SHAP analysis..."
echo ""

uv run python -m scripts.evaluation.run_shap_analysis \
    --checkpoint "$CHECKPOINT" \
    --features "$FEATURES" \
    --period-start "$PERIOD_START" \
    --period-end "$PERIOD_END" \
    --n-queries "$N_QUERIES" \
    --n-candidates "$N_CANDIDATES" \
    --background-size "$BACKGROUND_SIZE" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "ANALYSIS COMPLETE"
echo "========================================"
echo ""
echo "Results:"
echo "  Global importance: $OUTPUT_DIR/global_importance.csv"
echo "  Per-query results: $OUTPUT_DIR/per_query/"
echo "  Figures:           $OUTPUT_DIR/figures/"
echo ""