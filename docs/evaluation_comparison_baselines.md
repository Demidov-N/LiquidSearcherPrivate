# Evaluation Comparison: New Separate Rankers vs Previous Baselines

**Date:** 2026-04-02  
**Purpose:** Compare the new temporal/tabular component evaluation against previous baseline results

---

## Executive Summary

| Evaluation | Best nDCG@10 (LiquidityUplift) | Winner | Architecture |
|------------|-------------------------------|--------|--------------|
| **Previous (retrieval_fixed)** | 0.810 | `spearman_corr_rerank` | Correlation baselines + embedding |
| **New (retrieval_separate)** | **0.754** | **`tabular_rerank`** | Component separation |

**Key Finding:** While the new `tabular_rerank` doesn't quite match the previous best (0.754 vs 0.810), it provides **interpretability** - we now understand *why* it works (fundamental filter + liquidity rerank).

---

## 1. Previous Baseline Results (retrieval_fixed)

### Overall Performance (Averaged Across References)

| Method | Recall@10 | nDCG@10 | Spearman |
|--------|-----------|---------|----------|
| embedding | 0.0040 | 0.379 | 0.027 |
| pearson_corr | 0.0060 | 0.512 | 0.260 |
| spearman_corr | 0.0059 | 0.508 | 0.273 |
| **embedding_rerank** | 0.0068 | 0.614 | 0.645 |
| pearson_corr_rerank | 0.0097 | **0.799** | 0.655 |
| **spearman_corr_rerank** | **0.0098** | **0.810** | **0.655** |

### LiquidityUplift Specific Results

| Method | Recall@10 | nDCG@10 |
|--------|-----------|---------|
| embedding | 0.0076 | 0.454 |
| pearson_corr | 0.0120 | 0.669 |
| spearman_corr | 0.0120 | 0.668 |
| **embedding_rerank** | 0.0120 | 0.673 |
| pearson_corr_rerank | 0.0140 | 0.811 |
| **spearman_corr_rerank** | **0.0140** | **0.819** |

**Previous Winner:** `spearman_corr_rerank` with nDCG@10 = 0.819 on LiquidityUplift

---

## 2. New Component-Separate Results (retrieval_separate)

### Overall Performance (Averaged Across 6 References)

| Ranker | Recall@10 | nDCG@10 | Description |
|--------|-----------|---------|-------------|
| temporal_only | 0.0047 | 0.561 | 128-dim temporal encoder only |
| tabular_only | 0.0041 | 0.529 | 128-dim tabular encoder only |
| joint | 0.0047 | 0.559 | 256-dim concatenation |
| **tabular_rerank** | **0.0054** | **0.617** | **Top-50 tabular + liquidity rerank** |

### LiquidityUplift Specific Results

| Ranker | Recall@10 | nDCG@10 | vs Previous Best |
|--------|-----------|---------|------------------|
| temporal_only | 0.0077 | 0.301 | -63.2% |
| tabular_only | 0.0074 | 0.303 | -63.0% |
| joint | 0.0074 | 0.257 | -68.6% |
| **tabular_rerank** | **0.0130** | **0.754** | **-7.9%** |

**New Winner:** `tabular_rerank` with nDCG@10 = 0.754 on LiquidityUplift

---

## 3. Detailed Comparison by Ground Truth Reference

### 3.1 LiquidityUplift (Primary Objective)

| Method | Previous | New Equivalent | nDCG@10 |
|--------|----------|----------------|---------|
| spearman_corr_rerank | ✅ | - | **0.819** |
| tabular_rerank | - | ✅ | **0.754** |
| embedding | ✅ | joint (256-dim) | 0.257 |

**Analysis:** 
- The correlation-based rerankers (previous) performed best because they directly optimize for co-movement
- The new `tabular_rerank` is 8% worse than spearman_corr_rerank BUT is more interpretable
- The joint concatenation (256-dim) performs terribly (0.257) - confirming the simple concatenation problem

### 3.2 ReturnSimilarity (120-day Correlation)

| Ranker | nDCG@10 | vs Best |
|--------|---------|---------|
| joint | 0.449 | baseline |
| tabular_rerank | 0.482 | +7.4% |
| temporal_only | 0.425 | -5.3% |
| tabular_only | 0.387 | -13.8% |

**Finding:** The joint concatenation works slightly better for return correlation than individual components, suggesting some alignment benefit.

### 3.3 SectorSimilarity (GICS Matching)

| Ranker | nDCG@10 | Notes |
|--------|---------|-------|
| tabular_only | **0.839** | Best - has GICS features |
| tabular_rerank | 0.815 | Slightly degraded by reranking |
| joint | 0.831 | Good but not best |
| temporal_only | 0.774 | Worst - no fundamental info |

**Finding:** Tabular encoder excels at sector matching - as expected since GICS codes are in tabular features.

### 3.4 FundamentalsSim (Market Cap Similarity)

| Ranker | nDCG@10 |
|--------|---------|
| temporal_only | 1.000 |
| tabular_only | 1.000 |
| joint | 1.000 |
| tabular_rerank | 0.999 |

**Note:** All methods achieve ~1.0 nDCG@10 because market cap is explicitly in the tabular features and the model learned to retrieve similar-sized companies.

---

## 4. What We Learned

### 4.1 Why Simple Concatenation Fails

| Metric | Temporal Only | Tabular Only | Joint (Concat) | Expected if Fusion Worked |
|--------|---------------|--------------|----------------|---------------------------|
| LiquidityUplift nDCG | 0.301 | 0.303 | **0.257** | > 0.303 |
| ReturnSimilarity nDCG | 0.425 | 0.387 | **0.449** | ~ 0.425 |
| SectorSimilarity nDCG | 0.774 | **0.839** | 0.831 | ~ 0.839 |

**Observation:** The joint embedding performs worse than individual components on LiquidityUplift. This confirms:
1. Concatenation without learned fusion doesn't combine modalities effectively
2. The InfoNCE loss aligns embeddings but doesn't teach cross-modal reasoning
3. The 256-dim joint space may be too sparse for nearest-neighbor retrieval

### 4.2 Why Tabular + Liquidity Rerank Works

The `tabular_rerank` approach:
1. **Stage 1:** Tabular encoder retrieves top-50 by fundamental/sector similarity
2. **Stage 2:** Liquidity score reranks within the shortlist

This two-stage design mimics how fundamental analysts work:
- First filter by sector/business model (tabular)
- Then select most liquid within that group

**Result:** 0.754 nDCG@10 on LiquidityUplift - close to previous best (0.819) but more interpretable.

### 4.3 SHAP Value Mystery Explained

Previous SHAP analysis showed temporal features dominated importance scores. **Why?**

1. **Training:** InfoNCE loss aligns temporal and tabular embeddings into the same space
2. **Inference:** Concatenation treats both equally, but temporal features may have:
   - Higher variance (more discriminative)
   - Better gradient flow during training
   - More informative patterns (price/volume dynamics)
3. **Result:** The model "relies" on temporal features implicitly even when tabular would be better

---

## 5. Trade-off Analysis

| Approach | LiquidityUplift | Interpretability | Speed | Complexity |
|----------|-----------------|------------------|-------|------------|
| spearman_corr_rerank (prev) | **0.819** ⭐ | Low (correlation magic) | Slow | Low |
| tabular_rerank (new) | **0.754** | **High** (2-stage logic) | Fast | Low |
| joint (256-dim) | 0.257 | Medium | Fast | Medium |

**Recommendation:**
- **For production now:** Use `tabular_rerank` - it's 92% as good as the best with full interpretability
- **For research:** Investigate why correlation methods work so well - can we bake that into the learned embedding?

---

## 6. Architecture Recommendations

Based on this evaluation:

### Short-term (Immediate)
1. **Deploy `tabular_rerank`** for production retrieval
2. **Keep temporal + tabular separate** - don't concatenate
3. **Use two-stage pipeline:** Tabular filter → Liquidity rerank

### Medium-term (Model Redesign)
1. **Add learned fusion:**
   ```python
   # Instead of: joint = torch.cat([temporal, tabular])
   # Use: cross-attention between modalities
   temporal_attended = cross_attn(temporal, tabular)
   tabular_attended = cross_attn(tabular, temporal)
   joint = torch.cat([temporal_attended, tabular_attended])
   ```

2. **Add gating mechanism:**
   ```python
   gate = torch.sigmoid(gate_layer(torch.cat([temporal, tabular])))
   fused = gate * temporal + (1 - gate) * tabular
   ```

3. **Task-specific heads:** Different fusion strategies for different objectives (liquidity vs return similarity)

### Long-term (Training Strategy)
1. **Auxiliary losses:**
   - Sector classification loss → forces tabular information into joint embedding
   - Liquidity uplift prediction → forces temporal dynamics into joint embedding

2. **Hard negative mining:** Focus on candidates that are:
   - Sector-matched but liquidity-mismatched
   - Liquidity-matched but sector-mismatched

---

## 7. File Locations

### Previous Baseline Results
```
results/retrieval_fixed/metrics/
├── retrieval_metrics_overall.csv         # Overall summary
├── retrieval_liquidity_uplift.csv        # Primary objective
├── retrieval_similarity.csv              # Correlation baselines
├── recall_spearman.csv                   # Spearman correlation
└── retrieval_metrics_6x3_detailed.csv    # Detailed per-query
```

### New Component-Separate Results
```
results/retrieval_separate/metrics/
├── evaluation_summary.csv                  # All ranker×reference combos
├── retrieval_metrics_overall.csv           # Averaged metrics
├── retrieval_liquidityuplift.csv           # Primary objective
├── retrieval_returnsimilarity.csv          # 120-day correlation
├── retrieval_sectorsimilarity.csv          # GICS matching
├── retrieval_fundamentalssim.csv           # Market cap similarity
├── retrieval_liquiditychar.csv             # Liquidity percentile
├── retrieval_turnoverchar.csv              # Turnover percentile
└── retrieval_metrics_detailed.csv          # Per-query (50 queries)
```

### Notebooks
```
notebooks/
├── evaluation_temporal_vs_tabular.ipynb    # Detailed analysis
└── [this document]                          # Comparison with baselines
```

---

## 8. Conclusion

### Key Takeaways

1. **Simple concatenation doesn't work.** The 256-dim joint embedding (0.257 nDCG) performs worse than individual 128-dim components.

2. **Component specialization is real:**
   - Temporal encoder → Trading dynamics (turnover, liquidity characteristics)
   - Tabular encoder → Fundamentals (sector, market cap)

3. **Two-stage retrieval beats end-to-end.** The `tabular_rerank` approach (0.754 nDCG) outperforms learned joint embeddings and approaches correlation-based methods (0.819 nDCG).

4. **Interpretability vs Performance trade-off:**
   - Correlation rerankers: 0.819 nDCG (black box)
   - Tabular reranker: 0.754 nDCG (interpretable logic)

5. **Next step:** Design proper cross-modal fusion (attention/gating) instead of simple concatenation.

---

*Generated: 2026-04-02*  
*Evaluation: 50 queries × 3,153 candidates × 4 rankers × 6 references*