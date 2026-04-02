# Evaluation Additionals

This document tracks useful follow-up work that should not block the core evaluation framework.

## Retrieval Follow-Ups

### Alpha Sensitivity For Hybrid Reranking

Block 4 will use a fixed hybrid score:

```text
HybridScore = 0.7 * EmbeddingSim_norm + 0.3 * LiquiditySim_norm
```

Nice-to-have extension:
- run an alpha sweep over `alpha in {0.5, 0.7, 0.9}`
- report `Recall@10` and `nDCG@10` for each alpha
- confirm whether the chosen `0.7` weight is stable across sectors and liquidity tiers

This should be reported as a sensitivity analysis, not the main Block 4 result.

## Full Evaluation Reruns

### Full SHAP Run

Current SHAP work validated the pipeline on a small sample.

Need to run the full version:
- more queries
- more candidates per query
- final output directory under `results/shap/`
- update summary tables after the full run completes

### Full UMAP Run

UMAP implementation is complete, but the final evaluation pass should re-run the full visualization pipeline and refresh outputs in `results/figures/umap/`.

## UMAP Follow-Up: Beta Interpretation

SHAP suggests `beta` is a dominant driver of similarity. This raises a useful follow-up question for UMAP interpretation:

- are apparent local clusters partly explained by beta rather than sector?

Nice-to-have additions:
- color UMAP points by beta quantile or continuous beta value
- compare cluster structure under sector coloring vs beta coloring
- compute a simple association summary between local neighborhoods and beta similarity

This is useful for interpretation, but should be treated as an explanatory extension rather than a required deliverable for Block 4.

## Priority Order

1. Implement Block 4 core retrieval evaluation
2. Run full SHAP evaluation
3. Re-run full UMAP outputs
4. Add hybrid alpha sensitivity analysis
5. Add UMAP beta-oriented interpretation plots
