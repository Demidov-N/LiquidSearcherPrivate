# Liquidity Retrieval Evaluation Design

**Date**: 2026-03-31
**Block**: 4
**Status**: Drafted for review

## Goal

Evaluate whether LiquidSearcher embeddings retrieve stocks with similar liquidity characteristics better than standard finance baselines.

This block should answer two different questions:

1. **Representation question**: does embedding similarity recover liquidity peers better than return correlation?
2. **System question**: does a light liquidity-aware reranking improve final top-10 retrieval quality?

These questions must remain separate in reporting.

## Evaluation Period

Use a fixed holdout-style evaluation period of `2019-01-01` to `2019-12-31`.

Reasoning:
- the training script defaults to training through `2018-12-31`
- this gives a clean post-train, pre-crisis evaluation slice
- it aligns with the existing `normal_2019` period already used elsewhere in evaluation code

For each stock, the query snapshot date should be the final available trading day within this period.

## Scope

This block covers:
- `Recall@10`
- `Spearman`
- `nDCG@10`
- embedding-only retrieval
- return-correlation baseline
- direct liquidity-distance baseline
- a separately reported hybrid embedding-plus-liquidity reranking extension

This block does **not** cover:
- crisis-only labels
- inverse-correlation hedge labels
- domain expert manual labels
- pairs trading backtests

Those belong in later evaluation blocks or explicit extensions.

## Financial Ground Truth

### Why Liquidity Labels

The target for this block is liquidity similarity, not generic co-movement. Return correlation is a valid baseline ranker, but it is not the ground truth for liquidity.

The label construction should therefore use low-frequency, financially interpretable liquidity proxies derived from fields already available in `data/processed/all_features.parquet`.

### Supported Inputs

Available processed columns include:
- `prc`
- `vol`
- `ret`
- `shrout`
- `bidlo`
- `askhi`
- `market_cap`

### Chosen Liquidity Proxies

Use a simple composite built from three proxies:

1. **Quoted spread percentage**

```text
spread_pct = (askhi - bidlo) / ((askhi + bidlo) / 2)
```

Interpretation:
- lower spread means more liquid
- directly reflects transaction cost / immediacy

2. **Amihud illiquidity**

```text
amihud = abs(ret) / (abs(prc) * vol)
```

Interpretation:
- lower Amihud means more liquid
- captures price impact per dollar traded

3. **Turnover**

```text
turnover = vol / shrout
```

Interpretation:
- higher turnover means more liquid
- captures trading activity relative to shares outstanding

### Liquidity Aggregation

For each stock within the selected evaluation window:
- compute daily proxy values
- aggregate to a single stock-level value using a robust summary statistic

Recommended aggregation:
- median `spread_pct`
- median `amihud`
- median `turnover`

Median is preferred over mean because these quantities are noisy and heavy-tailed.

### Liquidity Score

After stock-level aggregation, convert each proxy into a cross-sectional percentile rank within the evaluation universe.

Define:

```text
LiquidityScore =
0.4 * (1 - spread_rank)
+ 0.4 * (1 - amihud_rank)
+ 0.2 * turnover_rank
```

Where:
- `spread_rank` is the percentile rank of aggregated spread
- `amihud_rank` is the percentile rank of aggregated Amihud
- `turnover_rank` is the percentile rank of aggregated turnover

Higher `LiquidityScore` means more liquid.

## Relevance Definition

The same liquidity score should drive both binary and graded evaluation.

### Binary Relevance For Recall@10

Assign each stock to a liquidity quartile using `LiquidityScore`.

For a query stock:
- a candidate is **relevant** if it falls in the same liquidity quartile

This produces binary relevance labels for `Recall@10`.

### Graded Relevance For nDCG@10

For a query stock, define graded relevance from distance in `LiquidityScore`.

Recommended scheme:
- relevance `3`: candidate is in the closest 10% by liquidity-score distance
- relevance `2`: candidate is in the next 15% by liquidity-score distance
- relevance `1`: candidate is in the same quartile but outside the top graded bands
- relevance `0`: otherwise

This preserves a liquidity-based interpretation while rewarding rank order quality.

## Rankers To Compare

All primary rankers must be evaluated against the same liquidity ground truth.

### 1. Embedding-Only

Rank candidates by cosine similarity in embedding space.

Purpose:
- tests whether the learned representation captures liquidity similarity without help

### 2. Correlation Baseline

Rank candidates by trailing 60-trading-day Pearson correlation of daily returns.

Definition:
- for each query stock, compute correlation between the query and every candidate using the 60 trading days ending on the query snapshot date
- require at least 40 overlapping daily returns to produce a valid score
- rank candidates by descending correlation

Purpose:
- standard finance co-movement baseline
- answers whether embeddings beat a familiar stock-similarity heuristic

### 3. Liquidity-Distance Baseline

Rank candidates directly by closeness in `LiquidityScore`.

Purpose:
- handcrafted baseline aligned with the evaluation label family
- provides an upper benchmark for a simple feature-engineered liquidity retrieval system

### 4. Hybrid Reranker Extension

Use a light reranking score that combines embedding similarity with a separate point-in-time liquidity estimate.

This is **not** a primary representation-quality benchmark because it injects explicit liquidity information into ranking. It should be reported in a separate table as an operational retrieval extension.

To avoid directly ranking with the exact evaluation label, the hybrid should use a point-in-time liquidity estimate built from a short trailing window rather than the period-level `LiquidityScore` used for ground truth.

Recommended operational liquidity estimate:
- trailing 20-trading-day median spread
- trailing 20-trading-day median Amihud
- trailing 20-trading-day median turnover
- convert these to a point-in-time `LiquidityScore_20d` using the same proxy weights

Use the hybrid only as a reranker over a shortlist, not as a full-universe first-stage retriever.

Recommended shortlist:
- top 50 candidates from embedding-only retrieval

Main formula:

```text
HybridScore = 0.7 * EmbeddingSim_norm + 0.3 * LiquiditySim20d_norm
```

Where:
- `EmbeddingSim_norm` is min-max normalized embedding cosine similarity
- `LiquiditySim20d_norm` is a normalized point-in-time liquidity closeness measure such as:

```text
LiquiditySim20d_norm = 1 - abs(LiquidityScore20d_i - LiquidityScore20d_q)
```

Assuming `LiquidityScore20d` is normalized to `[0, 1]`.

Purpose:
- practical retrieval improvement
- should be reported separately from representation-quality claims
- should not be used to claim that the embedding itself learned liquidity

### 5. Correlation Rerank (Additional Benchmark)

Use a two-stage approach: correlation-based shortlist + liquidity-distance rerank.

Definition:
- **Stage 1**: Rank all candidates by 60-trading-day correlation (same as Correlation Baseline)
- **Stage 2**: Rerank the top-N candidates (e.g., top 50) by liquidity-distance

This differs from Hybrid in that it does not use embeddings at all. It provides a clean test of whether a traditional correlation shortlist, enhanced only with a handcrafted liquidity feature, can approach the embedding-based systems.

Recommended shortlist:
- top 50 candidates from correlation ranking

Main formula:

```text
CorrRerankScore = LiquiditySim20d_norm
```

Where `LiquiditySim20d_norm` uses the same point-in-time 20-day liquidity estimate as the Hybrid reranker.

Purpose:
- additional baseline that combines correlation with explicit liquidity information
- helps isolate whether embedding's advantage comes from learned representation or from liquidity awareness
- reported in the same extension table as Hybrid

## Metrics

### Headline Metrics

1. **Recall@10**
- binary relevance from liquidity quartiles
- answers whether relevant liquidity peers appear in the top-10 set

2. **nDCG@10**
- graded relevance from liquidity-score distance
- answers whether more relevant liquidity peers are ranked earlier

3. **Spearman**
- rank correlation between each system's ranking and the liquidity-distance ranking
- retained for consistency with the project master evaluation plan

### Secondary Metrics

Optional but useful:
- `Precision@10`
- Spearman correlation between predicted rank and liquidity-distance rank
- mean liquidity-score distance of top-10 retrievals

The main summary table should include `Recall@10`, `Spearman`, and `nDCG@10`.

Recommended emphasis in discussion:
- `Recall@10` for binary retrieval quality
- `nDCG@10` for ranked retrieval quality
- `Spearman` for continuity with the original evaluation plan

## Evaluation Universe

For each query stock:
- candidate universe is all stocks in the `2019-01-01` to `2019-12-31` evaluation period
- exclude the query stock itself
- compute all rankers over the same candidate set

This ensures a fair comparison across rankers.

### Candidate Eligibility Rule

For a candidate to be included for a given query in the **primary benchmark comparison**, it must have:
- a valid embedding
- a valid period-level liquidity label
- at least 40 overlapping daily returns with the query in the trailing 60-trading-day correlation window

If a candidate fails any of these checks, exclude it from the candidate universe for **all three primary rankers**:
- embedding-only
- correlation baseline
- liquidity-distance baseline

This query-specific intersection rule keeps the benchmark universe identical across primary systems.

### Hybrid Extension Eligibility

The hybrid reranker is an extension and may require additional point-in-time liquidity inputs.

For the hybrid extension:
- start from the embedding top-50 shortlist produced on the primary candidate universe
- if a shortlisted name lacks valid trailing-20-day liquidity inputs, keep it in the shortlist but assign the lowest possible `LiquiditySim20d_norm` for reranking

This avoids changing the shortlist composition purely because the reranker has missing auxiliary features.

## Query Sampling

Use a stratified sample rather than only large-cap or high-visibility names.

Fixed sample size:
- 50 query stocks

Stratify across:
- sector
- market-cap tier
- liquidity quartile

### Deterministic Sampling Procedure

1. Build an eligible query snapshot using the final available trading day in 2019 for each stock.
2. Keep only stocks with:
- valid embedding inputs
- valid period-level liquidity labels
- at least 60 trailing daily returns ending on the query snapshot date
3. Assign each eligible stock:
- `gsector`
- `market_cap_tier` from cross-sectional terciles
- `liquidity_quartile` from the period-level `LiquidityScore`
4. Form sampling strata as `(gsector, market_cap_tier)`.
5. Sample 50 query stocks using a fixed random seed of `42`.
6. Use proportional stratified sampling by stratum size, with a minimum allocation of 1 query for any stratum containing at least 10 eligible stocks.
7. If rounding leaves fewer than 50 names, fill the remaining slots from the unsampled eligible pool using the same seed and a deterministic random order.
8. If rounding produces more than 50 names, trim excess names using the same seed and deterministic random order.

This keeps the query set reproducible while still preserving broad sector and size coverage.

### Reporting Query Coverage

The evaluation output should also write a query manifest file:
- `results/retrieval/query_manifest.csv`

This file should contain:
- query ticker
- query date
- sector
- market-cap tier
- liquidity quartile
- stratum label
- sampling seed

## Output Artifacts

### Metrics

Write:
- `results/metrics/recall_spearman.csv`
- `results/metrics/retrieval_metrics_overall.csv`
- `results/metrics/retrieval_metrics_by_sector.csv`
- `results/metrics/retrieval_metrics_by_market_cap.csv`
- `results/metrics/retrieval_metrics_by_liquidity_quartile.csv`

`recall_spearman.csv` exists to stay aligned with the project master plan. The richer `retrieval_metrics_*.csv` outputs can include `nDCG@10` and breakdowns.

### Per-Query Diagnostics

Write:
- `results/retrieval/per_query/*.parquet`

Each per-query file should include:
- query identifier
- candidate identifier
- embedding score
- correlation score
- liquidity-distance score
- hybrid score if the reranker extension is run
- binary relevance
- graded relevance
- final ranks under each system

### Figures

Recommended figures:
- bar chart of overall `Recall@10` by ranker
- bar chart of overall `Spearman` by ranker
- bar chart of overall `nDCG@10` by ranker
- grouped bar chart by sector
- grouped bar chart by liquidity quartile

## Interpretation Framework

Expected interpretations:

- **Embedding > Correlation**
  means the learned representation captures liquidity better than simple co-movement

- **Embedding < Liquidity-Distance**
  means handcrafted liquidity features remain stronger for this exact retrieval target

- **Hybrid > Embedding**
  means liquidity-aware reranking is a useful practical enhancement

- **Hybrid > all**
  means the best system-level retrieval combines representation learning with explicit liquidity information

The write-up must clearly distinguish between:
- representation quality claims
- retrieval system engineering claims

The primary benchmark table should compare only:
- embedding-only
- correlation baseline
- liquidity-distance baseline

The hybrid reranker should appear in a separate extension table.

## Validation And Failure Modes

### Data Validation

Before metric computation:
- drop rows with invalid or non-positive denominators for spread, Amihud, or turnover
- check enough stocks remain per period for quartiles to be meaningful
- verify candidate universe sizes are stable enough for cross-query comparison

### Metric Validation

Check:
- quartile counts are not severely degenerate
- graded relevance labels are not concentrated near zero
- normalized score ranges are well behaved for hybrid reranking

### Failure Modes

Potential issues:
- spread fields may be noisy at daily frequency
- Amihud may be unstable for very low-volume stocks
- query sample may be dominated by large liquid names if stratification is weak
- hybrid reranking can become label leakage if it uses the exact same period-level liquidity score as the ground truth

Mitigations:
- robust median aggregation
- filtering impossible values
- explicit query stratification
- fixed light rerank weight with embedding dominant at `0.7`
- use separate trailing-20-day liquidity features for reranking instead of the period-level ground-truth label

## Implementation Outline

Recommended modules:
- `src/evaluation/metrics/retrieval.py`
- `src/evaluation/utils/liquidity_labels.py`
- `src/evaluation/visualizations/retrieval_plots.py`
- `scripts/evaluation/run_retrieval_metrics.py`

Core responsibilities:
- compute liquidity proxies and stock-level labels
- build ranking outputs for all four rankers
- compute `Recall@10` and `nDCG@10`
- emit summary tables and figures

## References And Methodological Basis

This design follows standard IR ranking evaluation practice:
- `Recall@k` for binary relevance
- `nDCG@k` for graded relevance and rank-aware evaluation

Financially, the liquidity label design is based on established low-frequency liquidity proxies:
- quoted spread as a transaction-cost proxy
- Amihud illiquidity as a price-impact proxy
- turnover as a trading-activity proxy

Return correlation is included because it is a standard and intuitive finance baseline for stock similarity, but it is treated as a competitor ranker rather than the target label.

## Out Of Scope Additions

The following are intentionally deferred to `docs/ADDITIONALS.md`:
- hybrid alpha sensitivity analysis
- full SHAP rerun
- full UMAP rerun
- beta-oriented UMAP interpretation follow-up
