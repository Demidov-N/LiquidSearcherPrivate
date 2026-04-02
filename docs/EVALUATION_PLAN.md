# LiquidSearcher Evaluation Plan

**Status**: 3/6 Blocks Complete (Phase 1&2 In Progress)
**Priority**: HIGH - Final coding task to complete the project
**Last Updated**: April 1, 2026

---

## Progress Tracker

### Completed Blocks

| Block | Name | Status | Completed | Key Output |
|-------|------|--------|-----------|------------|
| ✅ 1 | UMAP Visualization | **Complete** | 2026-03-30 | `results/figures/umap/` |
| ✅ 2 | SHAP Feature Importance | **Complete** | 2026-03-31 | `results/shap/` |
| ✅ 4 | Retrieval Evaluation | **Complete** | 2026-04-01 | `results/metrics/retrieval_*.csv` |

### Remaining Blocks

| Block | Name | Status | Priority | Estimated Effort |
|-------|------|--------|----------|------------------|
| ❌ 3 | Manual Inspection | Not started | Phase 2 | 1 day (manual work) |
| ✅ 4 | Retrieval Evaluation | **Complete** | Phase 1 | Completed |
| ❌ 5 | Crisis Period Spearman | Not started | Phase 1 | 2-3 days |
| ❌ 6 | Pairs Trading Backtest | Not started | Phase 3 | 3-4 days |

### Key Findings So Far

**Block 1 (UMAP):**
- Global clustering by sector is weak (silhouette = -0.07)
- Local structure exists (2.1x lift in same-sector neighbors)
- Volatility profile drives similarity more than sector

**Block 2 (SHAP):**
- `beta` is the dominant feature (44% of importance)
- Model learned: similar beta → similar temporal dynamics → similar embedding
- Need larger sample (50 queries) for systematic conclusions

### Next Steps

1. **Priority 1**: Run full SHAP analysis (50 queries × 50 candidates)
2. **Priority 2**: Implement Block 4 (Recall@10) - basic retrieval quality
3. **Priority 3**: Implement Block 5 (Crisis Spearman) - regime robustness
4. **Priority 4**: Block 3 (Manual Inspection) - domain expert validation
5. **Extended**: Block 6 (Pairs Trading) - economic value validation

---

## Research Questions

The evaluation framework answers these core questions:

1. **Representation Quality**: Do embeddings capture meaningful stock relationships beyond raw correlation?
2. **Interpretability**: What features drive the model's similarity judgments?
3. **Regime Robustness**: Do embeddings remain valid during market stress when correlations break down?
4. **Economic Value**: Does embedding-based similarity produce better trading outcomes than traditional methods?

---

## Evaluation Blocks

### QUALITATIVE

| Block | Method | Research Question |
|-------|--------|-------------------|
| 1 | PCA + UMAP Visualization | Do embeddings cluster by economically meaningful categories? |
| 2 | SHAP Analysis | Which features drive similarity judgments? |
| 3 | Manual Inspection | Do retrieved neighbors make domain sense? |

### QUANTITATIVE

| Block | Method | Research Question |
|-------|--------|-------------------|
| 4 | Recall@10 + Spearman ρ + nDCG@10 | Does the model retrieve correct liquidity-tier peers? |
| 5 | Crisis Period Spearman | Are embeddings robust across regime shifts? |
| 6 | Pairs Trading Backtest | Do embeddings produce superior trading signals? |

---

## Qualitative Evaluations

### ✅ Block 1: PCA + UMAP Visualization

**Status**: **COMPLETE** (2026-03-30)

**Implementation**: `src/evaluation/visualizations/umap_visualizer.py`, `scripts/visualization/umap_plots.py`

**Run Command**: `./scripts/eval/run_umap.sh`

**Theoretical Foundation**

Dimensionality reduction serves two purposes:
1. **Validation**: If embeddings cluster by sector/market-cap/liquidity without explicit supervision, the model learned structural relationships
2. **Diagnostics**: Outliers and boundary cases reveal model failures

**Methodology**:
- Apply PCA (50 components) for noise reduction → UMAP for 2D projection
- UMAP preserves both local and global structure better than t-SNE
- Color points by: GICS sector, market-cap tier, liquidity quartile

**Crisis Snapshot Analysis**:
- Compute embeddings on pre-crisis data (Jan 2019 – Jan 2020)
- Compute embeddings on crisis data (Feb 2020 – May 2020)
- Plot side-by-side with identical point positions where possible
- Measure displacement vectors: stocks that moved furthest in embedding space represent relationship breakdown

**Financial Interpretation**:
- Tight sector clusters → model captures industry co-movement
- Liquidity stratification → model separates liquid/illiquid names
- Crisis migration → model adapts to regime-dependent relationships (e.g., travel stocks moving toward distressed clusters)

---

### ✅ Block 2: SHAP Feature Importance

**Status**: **COMPLETE** (2026-03-31)

**Implementation**: `src/evaluation/feature_importance/shap_analyzer.py`, `scripts/evaluation/run_shap_analysis.py`

**Run Command**: `./scripts/eval/run_shap.sh`

**Key Finding**: Beta dominates (44% importance), market_cap and idiosyncratic_vol secondary

**Theoretical Foundation**

SHAP (SHapley Additive exPlanations) decomposes similarity predictions into feature contributions using game-theoretic Shapley values. For a dual-encoder architecture:
- Applied to the joint embedding space
- Measures how each input feature shifts a stock's position relative to query

**Methodology**:
- Use DeepExplainer or GradientExplainer for neural network compatibility
- Compute SHAP values for similarity scores between query and candidate stocks
- Aggregate across queries for global importance
- Show individual waterfall plots for case studies

**Expected Feature Hierarchy** (if model works correctly):
1. Bid-ask spread percentage
2. Average daily volume
3. Market capitalization
4. Amihud illiquidity ratio
5. Turnover rate

**Financial Interpretation**:
- If liquidity features dominate → model learned its intended objective
- If fundamental features dominate → model may be capturing value/growth similarity instead
- Sector-specific patterns → different features matter for different industries

---

### ❌ Block 3: Manual Inspection

**Status**: **NOT STARTED**

**Location**: `results/reports/manual_inspection.txt` (to be created)

**Theoretical Foundation**

Quantitative metrics can miss systematic failures that are obvious to domain experts. Manual inspection provides:
- **Face validity**: Do results pass the "smell test"?
- **Failure mode discovery**: Identify patterns not captured by aggregate metrics
- **Cross-sector insights**: Discover non-obvious but economically sensible connections

**Methodology**:
- Select 10-15 query stocks stratified by:
  - 5 large-cap liquid (e.g., AAPL, MSFT, JNJ)
  - 5 small-cap illiquid (low volume, high spread)
  - 5 mixed-sector (to test cross-industry retrieval)
- For each query, examine top-10 retrieved neighbors
- Annotate: intuitive sense, interesting connections, obvious failures

**Financial Interpretation**:
- High face validity → model is production-ready
- Systematic failures by sector → training data imbalance or architecture limitation
- Interesting cross-sector links → potential pairs trading opportunities

---

## Quantitative Evaluations

- [x] **Block 4: Retrieval Evaluation**
  - Implementation: `scripts/evaluation/run_retrieval_metrics.py`
  - Spec: `docs/superpowers/specs/2026-04-01-revised-retrieval-scoring-design.md`
  - Status: ✅ Revised - 6 rankers × 3 references (Similarity, LiquidityUplift, Utility)
  - Output: 
    - `results/metrics/retrieval_similarity.csv`
    - `results/metrics/retrieval_liquidity_uplift.csv`
    - `results/metrics/retrieval_utility.csv`
    - `results/metrics/retrieval_metrics_overall.csv`

---

### ❌ Block 5: Crisis Period Spearman (Regime Robustness)

**Status**: **NOT STARTED** (Phase 1 Priority)

**Required Implementation**:
- `src/evaluation/metrics/regime_robustness.py`
- `scripts/evaluation/run_regime_analysis.py`

**Theoretical Foundation**

Correlations are notoriously regime-dependent. During crises:
- Correlations converge to 1 (everything falls together)
- Traditional diversification fails
- Relationships based on fundamentals break down

If embeddings learn deeper structural relationships, they should remain more stable across regimes than raw correlation.

**Evaluation Periods**:

| Regime | Window | Characteristics |
|--------|--------|-----------------|
| Normal | Jan 2019 – Jan 2020 | Low volatility, steady growth |
| Crisis | Feb 2020 – May 2020 | COVID crash, extreme volatility |
| Recovery | Jun 2020 – Dec 2020 | V-shaped rebound, stimulus-driven |
| Stress | Jan 2022 – Oct 2022 | Rate hikes, inflation shock |

**Methodology**:
- Fix a query set (e.g., 50 representative stocks)
- For each period, compute similarity rankings using:
  - Model embeddings (trained on that period or fine-tuned)
  - 60-day historical correlation
- Compute ground truth rankings based on liquidity features in each period
- Calculate Spearman ρ for model vs. ground truth and correlation vs. ground truth
- Compare ρ across periods

**Financial Interpretation**:
- ρ(model) stable across periods → embeddings are regime-robust
- ρ(correlation) drops during crisis → confirms correlation breakdown
- ρ(model) > ρ(correlation) during crisis → embeddings provide diversification when needed most

---

### ❌ Block 6: Pairs Trading Strategy Comparison

**Status**: **NOT STARTED** (Phase 3 - Extended)

**Required Implementation**:
- `src/evaluation/backtest/pairs_trading.py`
- `scripts/evaluation/run_backtest.py`

**Theoretical Foundation**

Pairs trading (Gatev, Goetzmann & Rouwenhorst, 2006) is a market-neutral strategy:
1. Identify historically correlated stock pairs
2. When prices diverge, short the outperformer, buy the underperformer
3. Profit from convergence

The critical design choice is **pair selection**. GGR (2006) uses minimum sum of squared deviations (equivalent to correlation-based selection).

**Hypothesis**: Embedding-based similarity produces pairs with:
- More stable relationships (less prone to breakdown)
- Faster convergence (stronger mean reversion)
- Better risk-adjusted returns

**Methodology**:

**Baseline (GGR 2006)**:
- Rank all pairs by historical correlation (or SSD)
- Select top-N pairs
- Entry: when price ratio exceeds 2σ from mean
- Exit: when ratio converges or after max holding period

**Embedding Variant**:
- Rank all pairs by embedding cosine similarity
- Select top-N pairs
- Identical entry/exit rules

**Backtest Metrics**:
- Annualized Sharpe ratio
- Annualized volatility
- Alpha vs. SPY (CAPM regression)
- Maximum drawdown
- Win rate (% of trades profitable)

**Financial Interpretation**:
- Higher Sharpe → better risk-adjusted returns
- Lower volatility → more stable pairs
- Positive alpha → skill-based returns (not market beta)
- Lower drawdown → better risk management
- Higher win rate → more reliable signals

---

## Implementation Priority

### Phase 1: Core Validation (Required for Paper)
1. ✅ ~~**Block 4 (Recall@10)** - Basic retrieval quality~~ → **COMPLETE**
2. **Block 5 (Crisis Spearman)** - Regime robustness (key differentiator) → **NEXT TO IMPLEMENT**
3. ✅ **Block 1 (UMAP)** - Visual validation → **COMPLETE**

### Phase 2: Interpretability (Required for Paper)
4. ✅ **Block 2 (SHAP)** - Feature attribution → **COMPLETE**
5. ~~**Block 3 (Manual)** - Face validity~~ → **AFTER QUANTITATIVE**

### Phase 3: Economic Value (Extended Analysis)
6. ~~**Block 6 (Pairs Trading)** - Real-world utility~~ → **OPTIONAL EXTENSION**

---

## Expected Outputs

| Block | Output Type | Location | Status |
|-------|-------------|----------|--------|
| ✅ 1 | 6-8 figures | `results/figures/umap/` | Complete |
| ✅ 2 | 3-5 figures + CSV | `results/shap/` | Complete |
| ❌ 3 | Text report | `results/reports/manual_inspection.txt` | Pending |
| ✅ 4 | Metrics tables (3 refs) | `results/metrics/retrieval_*.csv` | Complete |
| ❌ 5 | Metrics table | `results/metrics/regime_robustness.csv` | Pending |
| ❌ 6 | Metrics + equity curves | `results/metrics/pairs_trading.csv`, `results/figures/equity_curves.png` | Pending |

---

## Dependencies

```toml
[project.optional-dependencies]
evaluation = [
    "umap-learn>=0.5.0",      # UMAP visualization
    "shap>=0.44.0",           # SHAP analysis
    "matplotlib>=3.7.0",      # Plotting
    "seaborn>=0.12.0",        # Statistical visualization
]
```

---

## Summary Table

| Block | Name | Type | Key Metric | Baseline | Status |
|-------|------|------|------------|----------|--------|
| ✅ 1 | UMAP Visualization | Qualitative | Visual cluster quality | N/A | Complete |
| ✅ 2 | SHAP Analysis | Qualitative | Feature importance ranking | N/A | Complete |
| ❌ 3 | Manual Inspection | Qualitative | Pass/fail per query | N/A | Pending |
| ✅ 4 | Retrieval Evaluation | Quantitative | Recall, Spearman ρ, nDCG@10 | 6 Rankers × 3 References | Complete |
| ❌ 5 | Crisis Spearman | Quantitative | Spearman ρ by regime | Correlation | **Next** |
| ❌ 6 | Pairs Trading | Quantitative | Sharpe, alpha, drawdown | GGR (2006) | Future |

---

**Last Updated**: April 1, 2026
