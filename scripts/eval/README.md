# Evaluation Scripts

Standalone scripts for running LiquidSearcher evaluations remotely.

## Prerequisites

1. **uv installed**: https://docs.astral.sh/uv/
2. **Model checkpoint**: `checkpoints/last.ckpt`
3. **Feature data**: `data/processed/all_features.parquet`

## Quick Start

### SHAP Feature Importance (Block 2)

```bash
# Quick test (~2 min)
./scripts/eval/run_shap.sh --quick

# Production run (~30-40 min)
./scripts/eval/run_shap.sh

# Custom run
./scripts/eval/run_shap.sh --n-queries 20 --n-candidates 30
```

**Output:**
- `results/shap/global_importance.csv` - Feature importance rankings
- `results/shap/per_query/SHAP_*.parquet` - Per-query results
- `results/shap/figures/` - Visualizations

### UMAP Visualization (Block 1)

```bash
# Quick test (~1 min)
./scripts/eval/run_umap.sh --quick

# Production run (all periods)
./scripts/eval/run_umap.sh
```

**Output:**
- `results/figures/umap/umap_*.png` - Static plots
- `results/figures/umap/umap_*.html` - Interactive plots
- `results/figures/umap/clustering_metrics.csv` - Metrics

## Available Options

### run_shap.sh

| Option | Default | Description |
|--------|---------|-------------|
| `--quick` | - | Fast test with reduced parameters |
| `--n-queries` | 50 | Number of query stocks |
| `--n-candidates` | 50 | Candidates per query |
| `--background-size` | 100 | SHAP background samples |
| `--period-start` | 2019-01-01 | Start date |
| `--period-end` | 2019-12-31 | End date |
| `--output-dir` | results/shap | Output directory |

### run_umap.sh

| Option | Default | Description |
|--------|---------|-------------|
| `--quick` | - | Fast test with reduced parameters |
| `--max-samples` | 0 | Max samples (0 = all) |
| `--period` | covid_pre | Specific period |
| `--output-dir` | results/figures/umap | Output directory |

## Period Definitions

| Period | Start | End | Description |
|--------|-------|-----|-------------|
| `covid_pre` | 2019-01-01 | 2020-01-31 | Pre-COVID normal |
| `covid_crisis` | 2020-02-01 | 2020-05-31 | COVID crash |
| `covid_recovery` | 2020-06-01 | 2020-12-31 | V-shaped recovery |

## Running Remotely (GPU Server)

```bash
# SSH into server
ssh user@gpu-server

# Clone and setup
git clone <repo>
cd LiquidSearcher
uv sync

# Run evaluation (no GPU needed for SHAP)
nohup ./scripts/eval/run_shap.sh > shap.log 2>&1 &

# Check progress
tail -f shap.log
```

## Troubleshooting

### "Module not found: shap"
```bash
uv pip install shap matplotlib seaborn
```

### "Checkpoint not found"
Ensure model is trained:
```bash
python -m scripts.train --epochs 5
```

### "Out of memory"
Reduce parameters:
```bash
./scripts/eval/run_shap.sh --n-queries 10 --n-candidates 10 --background-size 20
```

## Evaluation Blocks Status

| Block | Name | Script | Status |
|-------|------|--------|--------|
| 1 | UMAP Visualization | `run_umap.sh` | ✅ Complete |
| 2 | SHAP Feature Importance | `run_shap.sh` | ✅ Complete |
| 3 | Manual Inspection | - | ❌ Not started |
| 4 | Recall@10 + Spearman | - | ❌ Not started |
| 5 | Crisis Period Spearman | - | ❌ Not started |
| 6 | Pairs Trading Backtest | - | ❌ Not started |