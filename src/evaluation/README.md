# Evaluation Framework

This directory contains the evaluation framework for the LiquidSearcher dual-encoder contrastive learning model.

## Structure

- **`metrics/`** - Evaluation metric implementations (silhouette, clustering, financial metrics)
- **`visualizations/`** - Plotting functions (t-SNE, heatmaps, training curves)  
- **`benchmarks/`** - Baseline comparison implementations (random, correlation, PCA)
- **`utils/`** - Evaluation utilities (data loaders, model utils, result savers)

## Status

This framework is **prepared for later implementation** as part of Phase 2 of the DVC MLOps refactor.

The structure is ready to support extensive multi-dimensional evaluation including:
- Embedding quality analysis
- Similarity analysis  
- Sector clustering validation
- Temporal consistency testing
- Baseline comparisons
- Ablation studies
- Financial metrics evaluation
- Robustness testing

## Implementation Notes

When implementing, this framework will integrate with the DVC pipeline stages defined in the root `dvc.yaml` file, reading all configuration from the central `params.yaml` file.