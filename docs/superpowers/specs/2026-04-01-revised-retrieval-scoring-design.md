# Revised Retrieval Scoring Design

**Date**: 2026-04-01
**Block**: 4 (revision)
**Status**: Draft
**Supersedes**: Ground-truth definitions in `2026-03-31-liquidity-retrieval-evaluation-design.md`

## Why This Revision

The original Block 4 spec evaluated all rankers against a **liquidity-ordering ground truth**. That was a methodology mismatch with the actual product objective:

- The model should retrieve **similar stocks** (liquidity-agnostic).
- Reranking should then improve **liquidity** among those similar candidates.

Evaluating the model directly on liquidity ordering penalizes it for being liquidity-agnostic, which is the desired behavior.

## Revised Goal

Block 4 should now answer:

1. Does the retriever return stocks that are genuinely similar to the query?
2. Among retrieved candidates, does reranking improve liquidity relative to the query?
3. Which end-to-end system best balances similarity and liquidity improvement?

## What Changes

- Ground truth switches from liquidity ordering to three reference targets.
- Spearman return correlation added as a retriever baseline alongside Pearson.
- Metrics stay the same: `Recall@10`, `nDCG@10`, `Spearman`.
- Output structure stays the same shape.

## What Stays The Same

- Evaluation period: `2019-01-01` to `2019-12-31`
- Query sampling: 50 queries, seed 42, stratified
- Candidate universe intersection rules, updated as follows:
  - a candidate must have a valid embedding, a valid period-level liquidity label, at least 40 overlapping returns in the 60-day correlation window, **and** at least 80 overlapping returns in the 120-day similarity-label window
  - this is a stricter intersection than before; candidates failing the 120-day check are excluded from **all** rankers for that query
- Liquidity proxy computation (spread, Amihud, turnover)
- `LiquidityScore` formula and quartile assignment
- Reranking mechanics (top-50 shortlist, alpha=0.7)
- Per-query parquet artifacts, figures, CSV outputs

## Rankers

Six rankers, evaluated identically against all three ground truths:

### First-Stage Retrievers

1. **embedding**
   - cosine similarity in embedding space
   - from `checkpoints/last.ckpt`

2. **pearson_corr**
   - trailing 60-day Pearson correlation of daily returns
   - existing implementation

3. **spearman_corr** (new)
   - trailing 60-day Spearman rank correlation of daily returns
   - primary traditional baseline

### Reranked Systems

4. **embedding_rerank**
   - shortlist: top-50 by embedding cosine similarity
   - rerank formula: `0.7 * EmbeddingSim_norm + 0.3 * LiquiditySim20d_norm`
   - final ordering: reranked shortlist first, then remaining candidates in original embedding order
   - for Spearman metric: convert the final ordering into integer ranks (1 = best) and correlate those ranks against the reference

5. **pearson_corr_rerank**
   - shortlist: top-50 by Pearson correlation
   - rerank formula: `0.7 * PearsonCorr_norm + 0.3 * LiquiditySim20d_norm`
   - final ordering: reranked shortlist first, then remaining in original Pearson order
   - for Spearman metric: convert final ordering into integer ranks and correlate against reference
   - existing `correlation_rerank` renamed for clarity

6. **spearman_corr_rerank** (new)
   - shortlist: top-50 by Spearman rank correlation
   - rerank formula: `0.7 * SpearmanCorr_norm + 0.3 * LiquiditySim20d_norm`
   - final ordering: reranked shortlist first, then remaining in original Spearman order
   - for Spearman metric: convert final ordering into integer ranks and correlate against reference

**Spearman metric for all reranked systems**: The final candidate ordering (shortlist reranked, then remainder appended) is converted to dense integer ranks `1..N`. These ranks are then correlated against the reference scores using Spearman. This guarantees a single unambiguous rank vector per system.

## Three Ground Truth References

For each query, build three separate relevance definitions. All use the same candidate universe.

### Reference 1: Similarity

**Purpose**: evaluate whether retrieved candidates are genuinely similar to the query.

**Label construction**: balanced composite from three components.

#### Return similarity component

For each candidate, compute trailing return correlation with the query.
- Use **120-day Pearson** correlation of daily returns for the similarity label.
- This differs from both correlation rankers in window length (120d vs 60d), which reduces circularity: the label captures a longer structural relationship while the rankers use a shorter recent window.
- Normalize to `[0, 1]` cross-sectionally by `(corr + 1) / 2`.
- Require at least 80 overlapping daily returns for a valid label.

#### Sector similarity component

- Same GICS industry group (`ggroup`) as query: 1.0
- Same GICS sector (`gsector`) but different industry group: 0.5
- Different sector: 0.0

#### Size similarity component

- Compute cross-sectional percentile rank of `market_cap`.
- `size_similarity = 1 - abs(query_mcap_rank - candidate_mcap_rank)`

#### Composite similarity score

```text
SimilarityScore =
  0.34 * return_similarity
+ 0.33 * sector_similarity
+ 0.33 * size_similarity
```

Balanced weighting. No single component dominates.

#### Binary relevance for Recall@10

- `relevant` = candidate is in the top quartile of `SimilarityScore` relative to the query.

#### Graded relevance for nDCG@10

- relevance `3`: top 10% closest by `SimilarityScore`
- relevance `2`: next 15%
- relevance `1`: same quartile but farther
- relevance `0`: otherwise

#### Spearman reference

- Spearman metric compares ranker scores against `SimilarityScore` directly.

### Reference 2: Liquidity Uplift

**Purpose**: evaluate whether retrieved candidates offer better liquidity than the query.

**Label construction**:

For each candidate, compute:

```text
LiquidityUplift = LiquidityScore_candidate - LiquidityScore_query
```

Where `LiquidityScore` uses the existing composite (spread, Amihud, turnover).

Positive uplift means the candidate is more liquid than the query.

#### Binary relevance for Recall@10

- `relevant` = candidate has `LiquidityUplift > 0` (strictly more liquid than query).

#### Graded relevance for nDCG@10

- relevance `3`: top 10% by `LiquidityUplift` (most improved liquidity)
- relevance `2`: next 15%
- relevance `1`: positive uplift but outside top bands
- relevance `0`: zero or negative uplift

#### Spearman reference

- Spearman metric compares ranker scores against `LiquidityUplift` directly.

### Reference 3: Utility

**Purpose**: evaluate whether the system retrieves candidates that are both similar and more liquid.

**Label construction**:

```text
UtilityScore = SimilarityScore * max(0, LiquidityUplift)
```

This rewards candidates that are:
- high similarity (from reference 1)
- positive liquidity improvement (from reference 2)

Candidates with zero or negative uplift get utility 0 regardless of similarity.

#### Binary relevance for Recall@10

- `relevant` = candidate has `SimilarityScore` in top quartile **AND** `LiquidityUplift > 0`.
- This requires both conditions, so it does not collapse to either reference alone.

#### Graded relevance for nDCG@10

- relevance `3`: top 10% by `UtilityScore`
- relevance `2`: next 15%
- relevance `1`: positive utility but outside top bands
- relevance `0`: zero utility

#### Spearman reference

- Spearman metric compares ranker scores against `UtilityScore` directly.

## Metrics

For each of the 6 rankers, against each of the 3 references, report:

- `Recall@10`
- `nDCG@10`
- `Spearman`

Total: 6 rankers x 3 references x 3 metrics = 54 numbers.

## Output Artifacts

### Metrics CSVs

Structure: one CSV per reference target.

- `results/metrics/retrieval_similarity.csv`
- `results/metrics/retrieval_liquidity_uplift.csv`
- `results/metrics/retrieval_utility.csv`

Each CSV has columns:

```
metric_name, embedding, pearson_corr, spearman_corr, embedding_rerank, pearson_corr_rerank, spearman_corr_rerank
```

Rows: `Recall@10`, `nDCG@10`, `Spearman`

### Combined Summary

- `results/metrics/retrieval_metrics_overall.csv`

Wide-format table with all 54 numbers for quick comparison.

### Backward Compatibility

Keep generating:
- `results/metrics/recall_spearman.csv` (similarity reference only, for master plan alignment)
- by-sector, by-market-cap, by-liquidity-quartile breakdowns (similarity reference)

### Per-Query Artifacts

- `results/retrieval/per_query/*.parquet`

Each parquet includes:
- all ranker scores and ranks
- `SimilarityScore`, `LiquidityUplift`, `UtilityScore` per candidate
- binary and graded relevance under all three references

### Figures

- `results/retrieval/figures/metrics_similarity_comparison.png`
- `results/retrieval/figures/metrics_liquidity_uplift_comparison.png`
- `results/retrieval/figures/metrics_utility_comparison.png`
- grouped breakdowns by sector and liquidity quartile

### Query Manifest

- `results/retrieval/query_manifest.csv` (unchanged)

## Interpretation Framework

Expected outcomes:

- **embedding > spearman_corr on similarity**
  means the model captures general stock similarity better than rank correlation alone

- **embedding_rerank > spearman_corr_rerank on liquidity uplift**
  means embedding shortlists contain better candidates for liquidity optimization

- **embedding_rerank > spearman_corr_rerank on utility**
  means the full system (model + rerank) outperforms the traditional pipeline

- **reranked > non-reranked on liquidity uplift**
  expected and confirms reranking works as intended

- **non-reranked methods weak on liquidity uplift**
  expected and confirms the model is liquidity-agnostic (desired behavior)

## Extensibility

The three-reference structure is designed for extension:

- swap similarity label to a different definition (e.g., factor-model residual similarity)
- swap liquidity uplift to a different reranking objective (e.g., volatility reduction, diversification)
- add new utility functions combining different objectives

Only the ground-truth construction changes; metrics, rankers, and output format stay the same.

## Implementation Notes

- Spearman return correlation retriever: use `scipy.stats.spearmanr` on trailing 60-day returns per pair
- Reuse existing `LiquidityScore` computation unchanged
- Reuse existing reranking mechanics unchanged
- Add `SimilarityScore`, `LiquidityUplift`, `UtilityScore` as new columns in pipeline
- Keep existing per-query evaluation loop structure; add three reference passes per query
