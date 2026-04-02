# Utility Breakdown Analysis + Critical Bug Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add utility breakdown analysis (return/sector/liquidity components) to per-query outputs and fix three critical bugs affecting result correctness.

**Architecture:** Extend `compute_similarity_score()` to return component scores alongside composite, store components in per-query parquets, generate breakdown CSVs showing top-10 composition by dimension. Fix column reference bug, ggroup encoding bug, and uplift relevance semantics.

**Tech Stack:** Python, pandas, torch, existing evaluation pipeline

---

## Background

Two categories of work:

### Category A: Utility Breakdown Analysis (Feature Request)
The user wants to understand how UtilityScore decomposes into:
- Return similarity (behavioral match)
- Sector similarity (industry match)  
- Liquidity improvement (uplift > 0)

Currently only the composite `SimilarityScore` and `UtilityScore` are stored. Need to store components and create breakdown analysis.

### Category B: Critical Bug Fixes (Verified Issues)
Three bugs confirmed affecting result correctness:

1. **pearson_corr_rerank Spearman column bug** - references wrong data column
2. **ggroup encoding bug** - raw GICS code incorrectly clamped, all stocks map to same embedding
3. **LiquidityUplift relevance semantics** - negative uplift can be marked "relevant" due to top-quartile logic

---

## File Structure

**Files to Modify:**
- `src/evaluation/ground_truth.py` - Add component storage to similarity computation
- `scripts/evaluation/run_retrieval_metrics.py` - Fix 3 bugs, store components in parquets, generate breakdown
- `src/evaluation/feature_importance/shap_analyzer.py` - Fix ggroup encoding bug
- `tests/test_evaluation_retrieval_metrics.py` - Add tests for breakdown analysis

**Files to Create:**
- None (all modifications to existing files)

---

## Task 1: Fix pearson_corr_rerank Spearman Column Reference

**Files:**
- Modify: `scripts/evaluation/run_retrieval_metrics.py:1076-1080`

- [ ] **Step 1: Identify the bug**

Current code at line ~1076:
```python
"pearson_corr_rerank": [
    ...,
    ...,
    np.mean([results_df[f"spearman_corr_rerank_{ref}_Spearman"].mean() ...]),  # ← WRONG
],
```

Should reference `pearson_corr_rerank` not `spearman_corr_rerank`.

- [ ] **Step 2: Fix the column reference**

```python
"pearson_corr_rerank": [
    np.mean([results_df[f"pearson_corr_rerank_{ref}_Recall@10"].mean() for ref in REFERENCES]),
    np.mean([results_df[f"pearson_corr_rerank_{ref}_nDCG@10"].mean() for ref in REFERENCES]),
    np.mean([results_df[f"pearson_corr_rerank_{ref}_Spearman"].mean() for ref in REFERENCES]),  # ← FIXED
],
```

- [ ] **Step 3: Verify syntax**

Run: `python -m py_compile scripts/evaluation/run_retrieval_metrics.py`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add scripts/evaluation/run_retrieval_metrics.py
git commit -m "fix: correct pearson_corr_rerank Spearman column reference"
```

---

## Task 2: Fix ggroup Encoding Bug in run_retrieval_metrics.py

**Files:**
- Modify: `scripts/evaluation/run_retrieval_metrics.py:243-245`

- [ ] **Step 1: Identify the bug**

Current code:
```python
if "ggroup" in row and not pd.isna(row["ggroup"]):
    ggroup = int(float(row["ggroup"]))
    categorical[0, 1] = max(min(ggroup, 24), 0)  # ← WRONG: 1010→24, 1020→24, etc.
```

GICS group codes are 4-digit (1010, 1020, ..., 6020). Need to map to 0-24 index.

- [ ] **Step 2: Fix the encoding**

```python
if "ggroup" in row and not pd.isna(row["ggroup"]):
    ggroup_code = int(float(row["ggroup"]))
    # Map 4-digit GICS code to 0-24 index
    # GICS groups: 1010, 1020, 1030, ..., 6020 (roughly 24 unique values)
    group_idx = (ggroup_code - 1010) // 100
    categorical[0, 1] = max(0, min(24, group_idx))
```

- [ ] **Step 3: Verify syntax**

Run: `python -m py_compile scripts/evaluation/run_retrieval_metrics.py`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add scripts/evaluation/run_retrieval_metrics.py
git commit -m "fix: correct ggroup encoding to use GICS code mapping"
```

---

## Task 3: Fix ggroup Encoding Bug in shap_analyzer.py

**Files:**
- Modify: `src/evaluation/feature_importance/shap_analyzer.py:142-146`

- [ ] **Step 1: Identify the bug**

Current code:
```python
if "ggroup" in row:
    val = row["ggroup"]
    if not pd.isna(val):
        ggroup_val = int(float(val))
        categorical[0, 1] = min(ggroup_val, 24)  # ← WRONG: same issue
```

- [ ] **Step 2: Fix the encoding**

```python
if "ggroup" in row:
    val = row["ggroup"]
    if not pd.isna(val):
        ggroup_code = int(float(val))
        # Map 4-digit GICS code to 0-24 index
        group_idx = (ggroup_code - 1010) // 100
        categorical[0, 1] = max(0, min(24, group_idx))
```

- [ ] **Step 3: Verify syntax**

Run: `python -m py_compile src/evaluation/feature_importance/shap_analyzer.py`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add src/evaluation/feature_importance/shap_analyzer.py
git commit -m "fix: correct ggroup encoding in SHAP analyzer"
```

---

## Task 4: Fix LiquidityUplift Relevance Semantics

**Files:**
- Modify: `scripts/evaluation/run_retrieval_metrics.py:748-754` (relevance building section)

- [ ] **Step 1: Identify the issue**

Current code builds binary relevance the same way for all references:
```python
relevance[f"{ref_name}_binary"] = build_binary_relevance(ref_scores, top_percentile=0.25)
```

For `LiquidityUplift`, this means "top quartile of uplift values" — but if most uplift values are negative, the "relevant" stocks are just the least-negative, not actually more liquid than query.

- [ ] **Step 2: Add special handling for LiquidityUplift reference**

```python
# Build relevance labels for each reference
relevance = {}
for ref_name, ref_scores in references.items():
    if ref_name == "LiquidityUplift":
        # For uplift: relevance = positive uplift (actually more liquid than query)
        # NOT top quartile of all uplift values
        relevance[f"{ref_name}_binary"] = (ref_scores > 0).astype(int)
        # For graded: use quantiles of only positive uplift values
        positive_uplift = ref_scores[ref_scores > 0]
        if len(positive_uplift) > 0:
            graded = pd.Series(0, index=ref_scores.index)
            p75_pos = positive_uplift.quantile(0.75)
            p50_pos = positive_uplift.quantile(0.50)
            p25_pos = positive_uplift.quantile(0.25)
            graded[ref_scores > p75_pos] = 3
            graded[(ref_scores > p50_pos) & (ref_scores <= p75_pos)] = 2
            graded[(ref_scores > p25_pos) & (ref_scores <= p50_pos)] = 1
            graded[(ref_scores > 0) & (ref_scores <= p25_pos)] = 1  # All positive get at least 1
            relevance[f"{ref_name}_graded"] = graded
        else:
            relevance[f"{ref_name}_graded"] = pd.Series(0, index=ref_scores.index)
    else:
        relevance[f"{ref_name}_binary"] = build_binary_relevance(ref_scores, top_percentile=0.25)
        relevance[f"{ref_name}_graded"] = build_graded_relevance(ref_scores)
```

- [ ] **Step 3: Verify syntax and logic**

Run: `python -c "import pandas as pd; import numpy as np; uplift = pd.Series([-0.5, -0.3, 0.1, 0.2, 0.4, 0.6], index=['A','B','C','D','E','F']); relevance = (uplift > 0).astype(int); print('Positive uplift relevance:', relevance.tolist())"`
Expected: Shows only C, D, E, F (positive values) marked as 1

- [ ] **Step 4: Commit**

```bash
git add scripts/evaluation/run_retrieval_metrics.py
git commit -m "fix: LiquidityUplift relevance now requires positive uplift"
```

---

## Task 5: Extend compute_similarity_score to Return Components

**Files:**
- Modify: `src/evaluation/ground_truth.py:131-193`

- [ ] **Step 1: Add component tracking to compute_similarity_score**

Modify the function to return components alongside composite:

```python
def compute_similarity_score(
    query_symbol: str,
    returns_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    min_overlap: int = 80,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute composite SimilarityScore and return components.

    SimilarityScore = 0.34 * return_similarity + 0.33 * sector_similarity + 0.33 * size_similarity
    If size_similarity is unavailable, falls back to:
    SimilarityScore = 0.5 * return_similarity + 0.5 * sector_similarity

    Returns:
        Tuple of (composite_score, return_sim, sector_sim, size_sim)
        Each is a Series indexed by symbol
    """
    # Return similarity
    return_sim = compute_return_similarity_120d(returns_df, query_symbol, min_overlap)

    # Sector similarity
    query_row = snapshot_df[snapshot_df["symbol"] == query_symbol]
    if len(query_row) == 0:
        empty = pd.Series(dtype=float)
        return empty, empty, empty, empty

    query_gsector = query_row["gsector"].iloc[0]
    query_ggroup = query_row["ggroup"].iloc[0]

    sector_sim = compute_sector_similarity(query_gsector, query_ggroup, snapshot_df)

    # Size similarity
    size_sim = compute_size_similarity(query_symbol, snapshot_df)

    # Combine - align indices
    has_size = len(size_sim) > 0
    
    if has_size:
        common_symbols = return_sim.index.intersection(sector_sim.index).intersection(size_sim.index)
        if len(common_symbols) == 0:
            empty = pd.Series(dtype=float)
            return empty, empty, empty, empty

        composite = (
            0.34 * return_sim.loc[common_symbols]
            + 0.33 * sector_sim.loc[common_symbols]
            + 0.33 * size_sim.loc[common_symbols]
        )
    else:
        common_symbols = return_sim.index.intersection(sector_sim.index)
        if len(common_symbols) == 0:
            empty = pd.Series(dtype=float)
            return empty, empty, empty, empty

        composite = 0.50 * return_sim.loc[common_symbols] + 0.50 * sector_sim.loc[common_symbols]
        # Empty size_sim for consistent return type
        size_sim = pd.Series(dtype=float)

    return composite, return_sim, sector_sim, size_sim
```

- [ ] **Step 2: Verify return signature change doesn't break existing calls**

Search for existing calls: `grep -n "compute_similarity_score" scripts/evaluation/run_retrieval_metrics.py`

Only one call exists. Will need to update it to handle tuple return.

- [ ] **Step 3: Commit**

```bash
git add src/evaluation/ground_truth.py
git commit -m "feat: compute_similarity_score returns components tuple"
```

---

## Task 6: Update Evaluation Script to Store Components in Parquets

**Files:**
- Modify: `scripts/evaluation/run_retrieval_metrics.py:721-740` (similarity computation section)
- Modify: `scripts/evaluation/run_retrieval_metrics.py:911-943` (per-query parquet saving)

- [ ] **Step 1: Update similarity computation to capture components**

Change:
```python
# Reference 1: SimilarityScore (composite of return/sector/size)
similarity_scores = compute_similarity_score(
    query_symbol=query,
    returns_df=returns_120d,
    snapshot_df=snapshot_df,
    min_overlap=80,
)
similarity_scores = similarity_scores.reindex(candidates).fillna(0.0)
```

To:
```python
# Reference 1: SimilarityScore with component breakdown
similarity_scores, return_sim, sector_sim, size_sim = compute_similarity_score(
    query_symbol=query,
    returns_df=returns_120d,
    snapshot_df=snapshot_df,
    min_overlap=80,
)
similarity_scores = similarity_scores.reindex(candidates).fillna(0.0)
return_sim = return_sim.reindex(candidates).fillna(0.0)
sector_sim = sector_sim.reindex(candidates).fillna(0.0)
size_sim = size_sim.reindex(candidates).fillna(0.0) if len(size_sim) > 0 else pd.Series(0.0, index=candidates)
```

- [ ] **Step 2: Update per-query parquet saving to include components**

Add to per_query_data dict (around line 911):
```python
per_query_data = {
    "query_symbol": query,
    "candidate_symbol": candidates,
    # ... existing columns ...
    # Component scores for breakdown analysis
    "return_similarity": return_sim.values,
    "sector_similarity": sector_sim.values,
    "size_similarity": size_sim.values,
    # ... rest of existing columns ...
}
```

- [ ] **Step 3: Verify syntax**

Run: `python -m py_compile scripts/evaluation/run_retrieval_metrics.py`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add scripts/evaluation/run_retrieval_metrics.py
git commit -m "feat: store similarity components in per-query parquets"
```

---

## Task 7: Generate Utility Breakdown Analysis CSV

**Files:**
- Modify: `scripts/evaluation/run_retrieval_metrics.py:1095-1130` (after detailed metrics saving)

- [ ] **Step 1: Add breakdown analysis generation**

After saving `retrieval_metrics_6x3_detailed.csv`, add:

```python
# Generate utility breakdown analysis
print("Generating utility breakdown analysis...")
breakdown_results = []

for query in query_symbols:
    try:
        query_df = pd.read_parquet(output_dir / "retrieval" / "per_query" / f"{query}.parquet")
        
        # For each ranker, analyze top-10 composition
        for ranker_col, ranker_name in [
            ("emb_score", "embedding"),
            ("pearson_corr_score", "pearson_corr"),
            ("spearman_corr_score", "spearman_corr"),
            ("emb_rerank_score", "embedding_rerank"),
            ("pearson_rerank_score", "pearson_corr_rerank"),
            ("spearman_rerank_score", "spearman_corr_rerank"),
        ]:
            if ranker_col not in query_df.columns:
                continue
                
            # Get top 10 by this ranker
            top10 = query_df.nlargest(10, ranker_col)
            
            # Count by similarity type
            high_return_sim = (top10["return_similarity"] > top10["return_similarity"].quantile(0.75)).sum()
            high_sector_sim = (top10["sector_similarity"] >= 0.5).sum()  # Same sector or group
            positive_liq_uplift = (top10["LiquidityUplift"] > 0).sum()
            
            breakdown_results.append({
                "query": query,
                "ranker": ranker_name,
                "top10_size": len(top10),
                "high_return_sim": high_return_sim,
                "high_sector_sim": high_sector_sim,
                "positive_liq_uplift": positive_liq_uplift,
                "pct_return_sim": high_return_sim / len(top10) * 100,
                "pct_sector_sim": high_sector_sim / len(top10) * 100,
                "pct_liq_improve": positive_liq_uplift / len(top10) * 100,
            })
    except Exception as e:
        print(f"  Warning: Could not process breakdown for {query}: {e}")

if breakdown_results:
    breakdown_df = pd.DataFrame(breakdown_results)
    breakdown_path = output_dir / "metrics" / "utility_breakdown_analysis.csv"
    breakdown_df.to_csv(breakdown_path, index=False)
    print(f"Saved: {breakdown_path}")
    
    # Also save average breakdown per ranker
    avg_breakdown = breakdown_df.groupby("ranker")[
        ["pct_return_sim", "pct_sector_sim", "pct_liq_improve"]
    ].mean().round(2)
    avg_path = output_dir / "metrics" / "utility_breakdown_averaged.csv"
    avg_breakdown.to_csv(avg_path)
    print(f"Saved: {avg_path}")
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile scripts/evaluation/run_retrieval_metrics.py`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add scripts/evaluation/run_retrieval_metrics.py
git commit -m "feat: generate utility breakdown analysis CSVs"
```

---

## Task 8: Add Integration Test for Breakdown Analysis

**Files:**
- Modify: `tests/test_evaluation_retrieval_metrics.py`

- [ ] **Step 1: Add test for component storage**

```python
def test_per_query_parquet_includes_components(self):
    """Test that per-query parquets include similarity components."""
    # Assuming a test creates per-query parquets
    query_file = self.test_output_dir / "retrieval" / "per_query" / "AAPL.parquet"
    if query_file.exists():
        df = pd.read_parquet(query_file)
        assert "return_similarity" in df.columns
        assert "sector_similarity" in df.columns
        assert "size_similarity" in df.columns
```

- [ ] **Step 2: Add test for breakdown CSV generation**

```python
def test_utility_breakdown_csv_generated(self):
    """Test that utility breakdown analysis CSV is created."""
    breakdown_file = self.test_output_dir / "metrics" / "utility_breakdown_analysis.csv"
    assert breakdown_file.exists()
    
    df = pd.read_csv(breakdown_file)
    assert "query" in df.columns
    assert "ranker" in df.columns
    assert "pct_return_sim" in df.columns
    assert "pct_sector_sim" in df.columns
    assert "pct_liq_improve" in df.columns
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_evaluation_retrieval_metrics.py::TestUtilityBreakdown -v
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_evaluation_retrieval_metrics.py
git commit -m "test: add utility breakdown analysis tests"
```

---

## Task 9: Run Full Evaluation to Verify Fixes

- [ ] **Step 1: Run evaluation**

```bash
uv run python -m scripts.evaluation.run_retrieval_metrics \
    --features data/processed/all_features.parquet \
    --checkpoint checkpoints/last.ckpt \
    --output-dir results/retrieval_v2 \
    2>&1 | tail -50
```

- [ ] **Step 2: Verify pearson_corr_rerank Spearman is different from spearman_corr_rerank**

Check: `cat results/retrieval_v2/metrics/retrieval_metrics_overall.csv | grep -A1 "Spearman"`
Expected: Different values for pearson_corr_rerank and spearman_corr_rerank

- [ ] **Step 3: Verify breakdown CSVs exist**

```bash
ls results/retrieval_v2/metrics/utility_breakdown*.csv
```
Expected: Both files exist

- [ ] **Step 4: Spot-check breakdown numbers**

```python
import pandas as pd
df = pd.read_csv('results/retrieval_v2/metrics/utility_breakdown_averaged.csv')
print(df)
```
Expected: Shows percentages for each ranker across return/sector/liquidity dimensions

- [ ] **Step 5: Commit (if all good)**

```bash
git commit --allow-empty -m "chore: verify all fixes and breakdown analysis working"
```

---

## Summary of Changes

| Task | Issue Type | Fix Description |
|------|------------|-----------------|
| 1 | Bug | Correct pearson_corr_rerank column reference |
| 2 | Bug | Fix ggroup encoding in run_retrieval_metrics.py |
| 3 | Bug | Fix ggroup encoding in shap_analyzer.py |
| 4 | Bug | LiquidityUplift relevance requires positive uplift |
| 5 | Feature | compute_similarity_score returns components tuple |
| 6 | Feature | Store components in per-query parquets |
| 7 | Feature | Generate utility_breakdown_analysis.csv |
| 8 | Test | Add integration tests for breakdown |
| 9 | Verify | Full evaluation run confirms fixes |

## Expected Outcomes

1. **Bug fixes** will correct result numbers in summary CSVs
2. **ggroup fix** may change embedding scores (now correctly using industry groups)
3. **Breakdown analysis** will show what % of each ranker's top-10 are:
   - Similar in returns (high return_similarity)
   - Similar in sector (sector_similarity >= 0.5)
   - Better in liquidity (positive LiquidityUplift)

Co-Authored-By: Claude Code <noreply@anthropic.com>
