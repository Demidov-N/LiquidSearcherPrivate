# LiquidSearcher Project Summary

**Status**: Active Development
**Last Updated**: 2026-03-31

## Project Overview

Dual-encoder stock representation model using InfoNCE contrastive learning to align temporal (price) and tabular (fundamental) views for stock similarity search.

## Key Components

### Model Architecture
- **Temporal Encoder**: BiMT-TCN (60-day OHLCV → 128-dim)
- **Tabular Encoder**: TabMixer (15 continuous + 2 categorical → 128-dim)
- **Training**: InfoNCE loss aligns temporal and tabular views
- **Output**: 256-dim joint embedding for similarity search

### Data
- **Source**: `data/processed/all_features.parquet` (12M rows, 5971 symbols, 2010-2024)
- **Checkpoint**: `checkpoints/last.ckpt` (epoch 58)

### Evaluation Framework

| Block | Name | Status | Location |
|-------|------|--------|----------|
| 1 | UMAP Visualization | ✅ Complete | `src/evaluation/visualizations/umap_visualizer.py` |
| 2 | SHAP Feature Importance | ✅ Complete | `src/evaluation/feature_importance/shap_analyzer.py` |
| 3 | Manual Inspection | ❌ Not started | - |
| 4 | Recall@10 + Spearman | ❌ Not started | - |
| 5 | Crisis Period Spearman | ❌ Not started | - |
| 6 | Pairs Trading Backtest | ❌ Not started | - |

## Key Findings

1. **Model Behavior**: Embeddings cluster by volatility profile, not sector (UMAP analysis)
2. **Local Structure**: 2.1x lift in same-sector nearest neighbors despite weak global clustering
3. **SHAP Results**: `beta` is most important feature for similarity (Mean |SHAP|: 0.4952)

## Commands

```bash
#Run evaluation scripts
python -m scripts.visualization.umap_plots --checkpoint checkpoints/last.ckpt
python -m scripts.evaluation.run_shap_analysis --n-queries 50 --n-candidates 50

# Type check
python -m mypy src/evaluation/feature_importance/shap_analyzer.py
```