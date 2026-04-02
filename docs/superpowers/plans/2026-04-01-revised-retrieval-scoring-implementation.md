# Revised Block 4 Retrieval Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revise Block 4 retrieval evaluation to evaluate against three ground truth references (Similarity, Liquidity Uplift, Utility) instead of the original single liquidity-ordering ground truth. Add spearman_corr ranker and extend to 6 total rankers (embedding, pearson_corr, spearman_corr, embedding_rerank, pearson_corr_rerank, spearman_corr_rerank).

**Architecture:** Add new ground truth computation module `src/evaluation/ground_truth.py` to encapsulate the three reference types (SimilarityScore, LiquidityUplift, UtilityScore). Extend existing retrieval metrics utilities in `src/evaluation/metrics/retrieval.py` with spearman correlation computation. Refactor `scripts/evaluation/run_retrieval_metrics.py` to use new ground truth module and compute 6 rankers × 3 references × 3 metrics = 54 numbers. Update visualization outputs to produce three reference-family plots.

**Tech Stack:** Python, pandas, polars (for large data), scipy.stats.spearmanr, torch (for model inference). Follow existing patterns in `src/evaluation/`.

---

## Background

The original Block 4 implementation evaluated all rankers against a **liquidity-ordering ground truth**. This was a methodology mismatch with the actual product objective:

1. The model should retrieve **similar stocks** (liquidity-agnostic).
2. Reranking should then improve **liquidity** among those similar candidates.

Evaluating the model directly on liquidity ordering penalizes it for being liquidity-agnostic—which is the desired behavior.

## Revised Goal

Block 4 should now answer:

1. Does the retriever return stocks that are genuinely similar to the query?
2. Among retrieved candidates, does reranking improve liquidity relative to the query?
3. Which end-to-end system best balances similarity and liquidity improvement?

## What Stays The Same

- Evaluation period: `2019-01-01` to `2019-12-31`
- Query sampling: 50 queries, seed 42, stratified
- Liquidity proxy computation (spread, Amihud, turnover)
- `LiquidityScore` formula and quartile assignment
- Reranking mechanics (top-50 shortlist, alpha=0.7)
- Per-query parquet artifacts, CSV outputs

## What Changes

- Ground truth switches from liquidity ordering to three reference targets (Similarity, Liquidity Uplift, Utility)
- Spearman return correlation added as retriever baseline alongside Pearson
- Candidate universe intersection rules: require 40 overlapping returns in 60-day correlation window AND 80 overlapping returns in 120-day similarity-label window
- Metrics stay the same: `Recall@10`, `nDCG@10`, `Spearman`
- Output structure: three CSVs (one per reference) plus combined summary

---

## File Structure

**Files to Create:**
- `src/evaluation/ground_truth.py` - New module for computing three ground truth references (SimilarityScore, LiquidityUplift, UtilityScore)
- `tests/test_evaluation_ground_truth.py` - Unit tests for ground truth computation

**Files to Modify:**
- `src/evaluation/metrics/retrieval.py:18` - Add `scipy.stats.spearmanr` import (already present), add `build_spearman_scores_for_query()` function
- `src/evaluation/metrics/retrieval.py:309-399` - Add `build_correlation_scores_120d()` for similarity label computation
- `scripts/evaluation/run_retrieval_metrics.py:44-50` - Import new ground truth functions
- `scripts/evaluation/run_retrieval_metrics.py:340-720` - Refactor evaluation loop for 6 rankers × 3 references
- `scripts/evaluation/run_retrieval_metrics.py:817-930` - Update CSV output structure for 3 reference families
- `src/evaluation/visualizations/retrieval_plots.py` - Add parameter to plot by reference type
- `tests/test_evaluation_retrieval_metrics.py` - Add tests for new functions

---

## Task 1: Create Ground Truth Computation Module

### Task 1.1: Create `src/evaluation/ground_truth.py` structure

**Files:**
- Create: `src/evaluation/ground_truth.py`
- Test: `tests/test_evaluation_ground_truth.py`

- [ ] **Step 1: Write the failing test for SimilarityScore computation**

```python
def test_compute_return_similarity():
    """Test 120-day Pearson return similarity computation."""
    returns_df = pd.DataFrame({
        'date': pd.date_range('2019-01-01', periods=120, freq='D'),
        'AAPL': np.random.randn(120) * 0.02,
        'MSFT': np.random.randn(120) * 0.02,
    }).set_index('date')
    
    from src.evaluation.ground_truth import compute_return_similarity_120d
    result = compute_return_similarity_120d(returns_df, 'AAPL', min_overlap=80)
    
    assert isinstance(result, pd.Series)
    assert 'MSFT' in result.index
    assert 0.0 <= result.loc['MSFT'] <= 1.0  # Normalized to [0,1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evaluation_ground_truth.py::test_compute_return_similarity -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.evaluation.ground_truth'"

- [ ] **Step 3: Write minimal SimilarityScore module**

```python
# src/evaluation/ground_truth.py
"""Ground truth reference computations for retrieval evaluation.

Provides three ground truth references:
- SimilarityScore: Balanced composite of return, sector, and size similarity
- LiquidityUplift: LiquidityScore difference (candidate - query)
- UtilityScore: SimilarityScore * max(0, LiquidityUplift)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_return_similarity_120d(
    returns_df: pd.DataFrame,
    query_symbol: str,
    min_overlap: int = 80,
) -> pd.Series:
    """Compute 120-day Pearson return similarity for similarity label.
    
    Args:
        returns_df: DataFrame with date index and symbol columns (daily returns)
        query_symbol: Query symbol
        min_overlap: Minimum overlapping observations required
        
    Returns:
        Series indexed by symbol with normalized similarity scores [0,1]
    """
    if query_symbol not in returns_df.columns:
        return pd.Series(dtype=float)
    
    query_returns = returns_df[query_symbol].dropna()
    results = {}
    
    for symbol in returns_df.columns:
        if symbol == query_symbol:
            continue
            
        symbol_returns = returns_df[symbol].dropna()
        common_idx = query_returns.index.intersection(symbol_returns.index)
        
        if len(common_idx) < min_overlap:
            continue
            
        q_ret = query_returns.loc[common_idx].values[-120:]
        s_ret = symbol_returns.loc[common_idx].values[-120:]
        
        if len(q_ret) < min_overlap or len(s_ret) < min_overlap:
            continue
            
        if np.std(q_ret) == 0 or np.std(s_ret) == 0:
            results[symbol] = 0.5  # Neutral for zero variance
        else:
            corr = np.corrcoef(q_ret, s_ret)[0, 1]
            # Normalize to [0, 1]
            results[symbol] = (corr + 1) / 2
            
    return pd.Series(results, name=f'return_similarity_{query_symbol}')
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_evaluation_ground_truth.py::test_compute_return_similarity -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_evaluation_ground_truth.py src/evaluation/ground_truth.py
git commit -m "feat: add ground truth module with 120d return similarity"
```

### Task 1.2: Add sector similarity component

- [ ] **Step 6: Write test for sector similarity**

```python
def test_compute_sector_similarity():
    """Test sector similarity computation."""
    from src.evaluation.ground_truth import compute_sector_similarity
    
    # Query: ggroup=10, gsector=1
    candidates = pd.DataFrame({
        'symbol': ['SAME_GGROUP', 'SAME_SECTOR', 'DIFF_SECTOR'],
        'gsector': [1, 1, 2],
        'ggroup': [10, 11, 20],
    })
    
    result = compute_sector_similarity(
        query_gsector=1,
        query_ggroup=10,
        candidates_df=candidates,
    )
    
    assert result['SAME_GGROUP'] == 1.0
    assert result['SAME_SECTOR'] == 0.5
    assert result['DIFF_SECTOR'] == 0.0
```

- [ ] **Step 7: Implement sector similarity**

```python
def compute_sector_similarity(
    query_gsector: int,
    query_ggroup: int,
    candidates_df: pd.DataFrame,
) -> pd.Series:
    """Compute sector similarity scores.
    
    Same ggroup = 1.0
    Same sector, different ggroup = 0.5
    Different sector = 0.0
    
    Args:
        query_gsector: Query's GICS sector code
        query_ggroup: Query's GICS industry group code
        candidates_df: DataFrame with symbol, gsector, ggroup columns
        
    Returns:
        Series indexed by symbol with similarity scores
    """
    def _sector_sim(row):
        if row['ggroup'] == query_ggroup:
            return 1.0
        elif row['gsector'] == query_gsector:
            return 0.5
        else:
            return 0.0
    
    result = candidates_df.apply(_sector_sim, axis=1)
    result.index = candidates_df['symbol']
    return result
```

- [ ] **Step 8: Run test and commit**

Run: `python -m pytest tests/test_evaluation_ground_truth.py::test_compute_sector_similarity -v`
Expected: PASS

```bash
git add tests/test_evaluation_ground_truth.py src/evaluation/ground_truth.py
git commit -m "feat: add sector similarity component"
```

### Task 1.3: Add size similarity component

- [ ] **Step 9: Write test for size similarity**

```python
def test_compute_size_similarity():
    """Test size similarity using market_cap percentiles."""
    from src.evaluation.ground_truth import compute_size_similarity
    
    snapshot_df = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'TSLA'],
        'market_cap': [1e12, 9e11, 1e10],  # AAPL largest, TSLA smallest
    })
    
    result = compute_size_similarity(
        query_symbol='AAPL',
        snapshot_df=snapshot_df,
    )
    
    # AAPL vs itself should be 1.0 (perfect match)
    # AAPL vs MSFT should be high but not 1.0
    # AAPL vs TSLA should be low
    assert result['AAPL'] == 1.0
    assert 0.0 <= result['MSFT'] <= 1.0
    assert 0.0 <= result['TSLA'] <= 1.0
    assert result['TSLA'] < result['MSFT']
```

- [ ] **Step 10: Implement size similarity**

```python
def compute_size_similarity(
    query_symbol: str,
    snapshot_df: pd.DataFrame,
) -> pd.Series:
    """Compute size similarity using market_cap percentile ranks.
    
    size_similarity = 1 - abs(query_mcap_rank - candidate_mcap_rank)
    
    Args:
        query_symbol: Query symbol
        snapshot_df: DataFrame with symbol and market_cap columns
        
    Returns:
        Series indexed by symbol with similarity scores [0, 1]
    """
    if query_symbol not in snapshot_df['symbol'].values:
        return pd.Series(dtype=float)
        
    # Compute percentile ranks
    mcap_ranks = snapshot_df['market_cap'].rank(pct=True)
    
    query_rank = mcap_ranks[snapshot_df['symbol'] == query_symbol].iloc[0]
    
    similarities = 1.0 - (mcap_ranks - query_rank).abs()
    similarities.index = snapshot_df['symbol']
    
    return similarities
```

- [ ] **Step 11: Run test and commit**

Run: `python -m pytest tests/test_evaluation_ground_truth.py::test_compute_size_similarity -v`
Expected: PASS

```bash
git add tests/test_evaluation_ground_truth.py src/evaluation/ground_truth.py
git commit -m "feat: add size similarity component"
```

### Task 1.4: Add composite SimilarityScore

- [ ] **Step 12: Write test for composite SimilarityScore**

```python
def test_compute_similarity_score():
    """Test composite SimilarityScore with balanced weights."""
    from src.evaluation.ground_truth import compute_similarity_score
    
    returns_df = pd.DataFrame({
        'date': pd.date_range('2019-01-01', periods=120, freq='D'),
        'AAPL': np.random.randn(120) * 0.02,
        'MSFT': np.random.randn(120) * 0.02,
    }).set_index('date')
    
    snapshot_df = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT'],
        'gsector': [1, 1],
        'ggroup': [10, 10],
        'market_cap': [1e12, 9e11],
    })
    
    result = compute_similarity_score(
        query_symbol='AAPL',
        returns_df=returns_df,
        snapshot_df=snapshot_df,
        min_overlap=80,
    )
    
    assert isinstance(result, pd.Series)
    assert 'MSFT' in result.index
    assert 0.0 <= result['MSFT'] <= 1.0
```

- [ ] **Step 13: Implement composite SimilarityScore**

```python
def compute_similarity_score(
    query_symbol: str,
    returns_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    min_overlap: int = 80,
) -> pd.Series:
    """Compute composite SimilarityScore.
    
    SimilarityScore = 0.34 * return_similarity + 0.33 * sector_similarity + 0.33 * size_similarity
    
    Args:
        query_symbol: Query symbol
        returns_df: DataFrame with date index and symbol columns (daily returns)
        snapshot_df: DataFrame with symbol, gsector, ggroup, market_cap columns
        min_overlap: Minimum overlapping observations for return similarity
        
    Returns:
        Series indexed by symbol with composite similarity scores
    """
    # Return similarity
    return_sim = compute_return_similarity_120d(returns_df, query_symbol, min_overlap)
    
    # Sector similarity
    query_row = snapshot_df[snapshot_df['symbol'] == query_symbol]
    if len(query_row) == 0:
        return pd.Series(dtype=float)
    
    query_gsector = query_row['gsector'].iloc[0]
    query_ggroup = query_row['ggroup'].iloc[0]
    
    sector_sim = compute_sector_similarity(
        query_gsector, query_ggroup, snapshot_df
    )
    
    # Size similarity
    size_sim = compute_size_similarity(query_symbol, snapshot_df)
    
    # Combine - align indices
    common_symbols = return_sim.index.intersection(sector_sim.index).intersection(size_sim.index)
    if len(common_symbols) == 0:
        return pd.Series(dtype=float)
    
    composite = (
        0.34 * return_sim.loc[common_symbols] +
        0.33 * sector_sim.loc[common_symbols] +
        0.33 * size_sim.loc[common_symbols]
    )
    
    return composite
```

- [ ] **Step 14: Run test and commit**

Run: `python -m pytest tests/test_evaluation_ground_truth.py::test_compute_similarity_score -v`
Expected: PASS

```bash
git add tests/test_evaluation_ground_truth.py src/evaluation/ground_truth.py
git commit -m "feat: add composite SimilarityScore computation"
```

### Task 1.5: Add binary and graded relevance functions

- [ ] **Step 15: Write tests for relevance functions**

```python
def test_build_binary_relevance():
    """Test binary relevance: top quartile of SimilarityScore."""
    from src.evaluation.ground_truth import build_binary_relevance
    
    similarity_scores = pd.Series([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2], 
                                   index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
    
    binary = build_binary_relevance(similarity_scores, top_percentile=0.25)
    
    # Top 25% (2 items) should be 1, rest 0
    assert binary['A'] == 1
    assert binary['B'] == 1
    assert binary['H'] == 0
    
def test_build_graded_relevance():
    """Test graded relevance with 3/2/1/0 bands."""
    from src.evaluation.ground_truth import build_graded_relevance
    
    similarity_scores = pd.Series([0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5], 
                                   index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
    
    graded = build_graded_relevance(similarity_scores)
    
    # Top 10% (0.8 items -> 1 item) = grade 3
    assert graded['A'] == 3
    
    # Next 15% = grade 2
    assert graded['B'] == 2
    
    # Same quartile but farther = grade 1
    # Bottom = grade 0
```

- [ ] **Step 16: Implement relevance functions**

```python
def build_binary_relevance(
    scores: pd.Series,
    top_percentile: float = 0.25,
) -> pd.Series:
    """Build binary relevance: 1 if in top percentile, 0 otherwise.
    
    Args:
        scores: Series of scores indexed by symbol
        top_percentile: Percentile threshold (0.25 = top quartile)
        
    Returns:
        Series with 1 for relevant symbols, 0 otherwise
    """
    threshold = scores.quantile(1 - top_percentile)
    return (scores >= threshold).astype(int)


def build_graded_relevance(
    scores: pd.Series,
) -> pd.Series:
    """Build graded relevance with 3/2/1/0 bands.
    
    relevance 3: top 10%
    relevance 2: next 15%  
    relevance 1: same quartile (25%) but outside top bands
    relevance 0: otherwise
    
    Args:
        scores: Series of scores indexed by symbol
        
    Returns:
        Series with graded relevance (3, 2, 1, 0)
    """
    n = len(scores)
    if n == 0:
        return pd.Series(dtype=int)
    
    # Compute thresholds
    sorted_scores = scores.sort_values(ascending=False)
    
    top_10_idx = max(1, int(np.ceil(0.10 * n))) - 1
    top_25_idx = max(1, int(np.ceil(0.25 * n))) - 1
    
    top_10_threshold = sorted_scores.iloc[top_10_idx]
    top_25_threshold = sorted_scores.iloc[top_25_idx] if top_25_idx < n else sorted_scores.iloc[-1]
    
    # Assign grades
    graded = pd.Series(0, index=scores.index)
    graded[scores >= top_10_threshold] = 3
    graded[(scores >= top_25_threshold) & (graded == 0)] = 2
    
    # Grade 1 for remaining in top quartile
    quartile_75 = scores.quantile(0.75)
    graded[(scores >= quartile_75) & (graded == 0)] = 1
    
    return graded
```

- [ ] **Step 17: Run tests and commit**

Run: `python -m pytest tests/test_evaluation_ground_truth.py::test_build_binary_relevance tests/test_evaluation_ground_truth.py::test_build_graded_relevance -v`
Expected: PASS

```bash
git add tests/test_evaluation_ground_truth.py src/evaluation/ground_truth.py
git commit -m "feat: add binary and graded relevance builders"
```

### Task 1.6: Add LiquidityUplift and UtilityScore

- [ ] **Step 18: Write tests for LiquidityUplift and UtilityScore**

```python
def test_compute_liquidity_uplift():
    """Test LiquidityUplift = LiquidityScore_candidate - LiquidityScore_query."""
    from src.evaluation.ground_truth import compute_liquidity_uplift
    
    liquidity_scores = pd.Series([0.8, 0.6, 0.4, 0.2], index=['A', 'B', 'C', 'D'])
    
    uplift = compute_liquidity_uplift(query_symbol='A', liquidity_scores=liquidity_scores)
    
    assert uplift['B'] == 0.6 - 0.8  # -0.2
    assert uplift['C'] == 0.4 - 0.8  # -0.4
    assert uplift['A'] not in uplift.index  # Exclude self

def test_compute_utility_score():
    """Test UtilityScore = SimilarityScore * max(0, LiquidityUplift)."""
    from src.evaluation.ground_truth import compute_utility_score
    
    similarity = pd.Series([0.9, 0.8, 0.7], index=['A', 'B', 'C'])
    uplift = pd.Series([0.2, -0.1, 0.3], index=['A', 'B', 'C'])
    
    utility = compute_utility_score(similarity, uplift)
    
    # A: 0.9 * 0.2 = 0.18
    assert utility['A'] == 0.9 * 0.2
    # B: 0.8 * 0 = 0 (negative uplift clamped)
    assert utility['B'] == 0.0
    # C: 0.7 * 0.3 = 0.21
    assert utility['C'] == 0.7 * 0.3
```

- [ ] **Step 19: Implement LiquidityUplift and UtilityScore**

```python
def compute_liquidity_uplift(
    query_symbol: str,
    liquidity_scores: pd.Series,
) -> pd.Series:
    """Compute LiquidityUplift for all candidates.
    
    LiquidityUplift = LiquidityScore_candidate - LiquidityScore_query
    
    Args:
        query_symbol: Query symbol
        liquidity_scores: Series indexed by symbol with LiquidityScore values
        
    Returns:
        Series indexed by candidate symbols with uplift values
    """
    if query_symbol not in liquidity_scores.index:
        return pd.Series(dtype=float)
    
    query_score = liquidity_scores.loc[query_symbol]
    candidates = liquidity_scores.drop(query_symbol)
    
    return candidates - query_score


def compute_utility_score(
    similarity_scores: pd.Series,
    liquidity_uplift: pd.Series,
) -> pd.Series:
    """Compute UtilityScore combining similarity and liquidity.
    
    UtilityScore = SimilarityScore * max(0, LiquidityUplift)
    
    Candidates with zero or negative uplift get utility 0 regardless of similarity.
    
    Args:
        similarity_scores: Series indexed by symbol with SimilarityScore values
        liquidity_uplift: Series indexed by symbol with LiquidityUplift values
        
    Returns:
        Series indexed by common symbols with utility scores
    """
    common_symbols = similarity_scores.index.intersection(liquidity_uplift.index)
    
    sim = similarity_scores.loc[common_symbols]
    uplift = liquidity_uplift.loc[common_symbols]
    
    # Clamp negative uplift to 0
    uplift_clamped = uplift.clip(lower=0)
    
    utility = sim * uplift_clamped
    
    return utility
```

- [ ] **Step 20: Run tests and commit**

Run: `python -m pytest tests/test_evaluation_ground_truth.py::test_compute_liquidity_uplift tests/test_evaluation_ground_truth.py::test_compute_utility_score -v`
Expected: PASS

```bash
git add tests/test_evaluation_ground_truth.py src/evaluation/ground_truth.py
git commit -m "feat: add LiquidityUplift and UtilityScore computation"
```

---

## Task 2: Add Spearman Correlation Retriever to Retrieval Metrics

### Task 2.1: Add spearman correlation score builder

- [ ] **Step 21: Write test for spearman score builder**

```python
def test_build_spearman_scores_for_query():
    """Test 60-day Spearman rank correlation computation."""
    returns_df = pd.DataFrame({
        'date': pd.date_range('2019-01-01', periods=60, freq='D'),
        'AAPL': np.random.randn(60) * 0.02,
        'MSFT': np.random.randn(60) * 0.02,
    }).set_index('date')
    
    from src.evaluation.metrics.retrieval import build_spearman_scores_for_query
    
    result = build_spearman_scores_for_query(
        returns_df, 'AAPL', snapshot_date=returns_df.index[-1], lookback=60, min_overlap=40
    )
    
    assert isinstance(result, pd.Series)
    assert 'MSFT' in result.index
    assert -1.0 <= result['MSFT'] <= 1.0
```

- [ ] **Step 22: Implement spearman score builder in retrieval.py**

Add to `src/evaluation/metrics/retrieval.py` after the existing `build_correlation_scores_for_query` function:

```python
def build_spearman_scores_for_query(
    returns_wide: pd.DataFrame,
    query_symbol: str,
    snapshot_date: pd.Timestamp,
    lookback: int = 60,
    min_overlap: int = 40,
) -> pd.Series:
    """Build Spearman rank correlation scores for a query symbol.
    
    Uses Spearman rank correlation on trailing lookback window of returns.
    Spearman is more robust to outliers than Pearson.
    
    Args:
        returns_wide: DataFrame with DateTimeIndex and symbol columns (returns)
        query_symbol: Symbol to compute correlations against
        snapshot_date: Reference date for the snapshot
        lookback: Number of trading days for rolling correlation
        min_overlap: Minimum overlapping observations required
        
    Returns:
        Series indexed by symbol with Spearman correlation scores
    """
    from scipy.stats import spearmanr
    
    if returns_wide.empty or query_symbol not in returns_wide.columns:
        return pd.Series(dtype=float)
    
    # Get lookback window
    cutoff_date = snapshot_date
    lookback_returns = returns_wide.loc[returns_wide.index <= cutoff_date].tail(lookback)
    
    if len(lookback_returns) < min_overlap:
        return pd.Series(dtype=float)
    
    query_returns = lookback_returns[query_symbol].dropna()
    
    results = {}
    for other_symbol in returns_wide.columns:
        if other_symbol == query_symbol:
            continue
        
        other_returns = lookback_returns[other_symbol].dropna()
        
        # Align by index
        common_idx = query_returns.index.intersection(other_returns.index)
        if len(common_idx) < min_overlap:
            continue
        
        q_ret = query_returns.loc[common_idx].values
        o_ret = other_returns.loc[common_idx].values
        
        # Compute Spearman correlation
        if np.std(q_ret) == 0 or np.std(o_ret) == 0:
            results[other_symbol] = np.nan
        else:
            corr, _ = spearmanr(q_ret, o_ret)
            results[other_symbol] = corr
    
    return pd.Series(results, name=f'spearman_{query_symbol}')
```

- [ ] **Step 23: Run test and commit**

Run: `python -m pytest tests/test_evaluation_retrieval_metrics.py::test_build_spearman_scores_for_query -v`
Expected: PASS

```bash
git add src/evaluation/metrics/retrieval.py tests/test_evaluation_retrieval_metrics.py
git commit -m "feat: add Spearman correlation retriever baseline"
```

---

## Task 3: Refactor Run Retrieval Metrics Script for 6 Rankers × 3 References

### Task 3.1: Update imports and ground truth integration

- [ ] **Step 24: Update imports in run_retrieval_metrics.py**

Modify `scripts/evaluation/run_retrieval_metrics.py:44-55`:

```python
from src.evaluation.ground_truth import (
    build_binary_relevance,
    build_graded_relevance,
    compute_liquidity_uplift,
    compute_similarity_score,
    compute_utility_score,
)
from src.evaluation.metrics.retrieval import (
    build_correlation_scores_for_query,
    build_hybrid_scores,
    build_spearman_scores_for_query,  # NEW
    build_snapshot_frame,
    ndcg_at_k,
    recall_at_k,
    sample_query_set,
    spearman_against_reference,
)
```

- [ ] **Step 25: Commit import changes**

```bash
git add scripts/evaluation/run_retrieval_metrics.py
git commit -m "refactor: update imports for 3-reference evaluation"
```

### Task 3.2: Refactor ground truth computation for per-query loop

- [ ] **Step 26: Add ground truth computation in evaluation pipeline**

In `scripts/evaluation/run_retrieval_metrics.py`, after loading period_df and snapshot_df (around line 620), add:

```python
# Compute extended returns for 120-day similarity label (separate from 60d ranker window)
print("Computing 120-day returns for similarity label...")
returns_120d = returns_input.pivot(index="date", columns="symbol", values="ret")
returns_120d = returns_120d.loc[returns_120d.index <= snapshot_date].tail(252)

# Compute 20-day trailing liquidity for reranking (if not already done)
if hybrid_20d_scores is None:
    print("Computing 20-day trailing liquidity scores...")
    hybrid_20d_df = aggregate_trailing_20d_liquidity(period_df, snapshot_date=snapshot_date)
    if not hybrid_20d_df.empty:
        hybrid_20d_scores = hybrid_20d_df.set_index("symbol")["LiquidityScore20d"]
```

- [ ] **Step 27: Commit ground truth setup**

```bash
git add scripts/evaluation/run_retrieval_metrics.py
git commit -m "refactor: add 120d returns and 20d liquidity computation"
```

### Task 3.3: Refactor per-query evaluation loop

- [ ] **Step 28: Replace per-query loop with 6 rankers × 3 references**

Replace the existing evaluation loop (lines 673-885) with:

```python
for i, query in enumerate(query_symbols):
    if (i + 1) % 10 == 0 or i == 0:
        print(f"  Processing query {i + 1}/{len(query_symbols)}: {query}")
    
    # Get candidates (all symbols except query)
    candidates = [s for s in snapshot_df["symbol"].tolist() if s != query]
    
    # === COMPUTE GROUND TRUTH REFERENCES ===
    
    # Reference 1: SimilarityScore (composite of return, sector, size)
    similarity_scores = compute_similarity_score(
        query_symbol=query,
        returns_df=returns_120d,
        snapshot_df=snapshot_df,
        min_overlap=80,
    )
    similarity_scores = similarity_scores.reindex(candidates).fillna(0.5)
    
    binary_sim = build_binary_relevance(similarity_scores, top_percentile=0.25)
    graded_sim = build_graded_relevance(similarity_scores)
    
    # Reference 2: LiquidityUplift
    liquidity_scores_series = snapshot_df.set_index("symbol")["LiquidityScore"]
    uplift_scores = compute_liquidity_uplift(query, liquidity_scores_series)
    uplift_scores = uplift_scores.reindex(candidates).fillna(0.0)
    
    binary_uplift = (uplift_scores > 0).astype(int)
    graded_uplift = build_graded_relevance(uplift_scores.clip(lower=0))  # Only positive uplift
    
    # Reference 3: UtilityScore
    utility_scores = compute_utility_score(similarity_scores, uplift_scores)
    utility_scores = utility_scores.reindex(candidates).fillna(0.0)
    
    binary_utility = ((similarity_scores >= similarity_scores.quantile(0.75)) & 
                      (uplift_scores > 0)).astype(int)
    graded_utility = build_graded_relevance(utility_scores)
    
    # === COMPUTE RANKER SCORES ===
    
    # 1. Embedding scores
    emb_scores = embedding_scores_dict.get(query, pd.Series(dtype=float))
    emb_scores = emb_scores.reindex(candidates).fillna(0.5)
    
    # 2. Pearson correlation (60-day)
    pearson_scores = build_correlation_scores_for_query(
        returns, query, snapshot_date, lookback=60, min_overlap=40
    )
    pearson_scores = pearson_scores.reindex(candidates).fillna(0)
    
    # 3. Spearman correlation (60-day) - NEW
    spearman_scores = build_spearman_scores_for_query(
        returns, query, snapshot_date, lookback=60, min_overlap=40
    )
    spearman_scores = spearman_scores.reindex(candidates).fillna(0)
    
    # 4. Embedding rerank: top-50 by embedding, rerank by hybrid(emb+liq20d)
    top50_emb = emb_scores.sort_values(ascending=False).head(50)
    emb_shortlist_scores = top50_emb
    liq20d_shortlist = hybrid_20d_scores.reindex(top50_emb.index).fillna(0.5)
    
    emb_rerank_shortlist = build_hybrid_scores(
        emb_shortlist_scores, liq20d_shortlist, alpha=0.7
    )
    # Final ordering: reranked shortlist first, then remaining in original embedding order
    remaining_emb = [s for s in emb_scores.sort_values(ascending=False).index if s not in top50_emb.index]
    emb_rerank_order = emb_rerank_shortlist.sort_values(ascending=False).index.tolist() + remaining_emb
    
    # 5. Pearson rerank: top-50 by pearson, rerank by hybrid(pearson+liq20d)
    top50_pearson = pearson_scores.sort_values(ascending=False).head(50)
    pearson_shortlist_scores = top50_pearson
    liq20d_pearson_shortlist = hybrid_20d_scores.reindex(top50_pearson.index).fillna(0.5)
    
    pearson_rerank_shortlist = build_hybrid_scores(
        pearson_shortlist_scores, liq20d_pearson_shortlist, alpha=0.7
    )
    remaining_pearson = [s for s in pearson_scores.sort_values(ascending=False).index if s not in top50_pearson.index]
    pearson_rerank_order = pearson_rerank_shortlist.sort_values(ascending=False).index.tolist() + remaining_pearson
    
    # 6. Spearman rerank: top-50 by spearman, rerank by hybrid(spearman+liq20d)
    top50_spearman = spearman_scores.sort_values(ascending=False).head(50)
    spearman_shortlist_scores = top50_spearman
    liq20d_spearman_shortlist = hybrid_20d_scores.reindex(top50_spearman.index).fillna(0.5)
    
    spearman_rerank_shortlist = build_hybrid_scores(
        spearman_shortlist_scores, liq20d_spearman_shortlist, alpha=0.7
    )
    remaining_spearman = [s for s in spearman_scores.sort_values(ascending=False).index if s not in top50_spearman.index]
    spearman_rerank_order = spearman_rerank_shortlist.sort_values(ascending=False).index.tolist() + remaining_spearman
    
    # === COMPUTE METRICS FOR ALL 6 RANKERS × 3 REFERENCES ===
    
    # Helper function to compute metrics for a ranker
    def compute_ranker_metrics(
        ranking_order: list[str],
        ranker_scores: pd.Series,
        binary_ref: pd.Series,
        graded_ref: pd.Series,
        ref_scores: pd.Series,
    ) -> dict:
        """Compute Recall@10, nDCG@10, Spearman for a ranker against a reference."""
        recall = recall_at_k(ranking_order, binary_ref, k=10)
        ndcg = ndcg_at_k(ranking_order, graded_ref, k=10)
        spearman = spearman_against_reference(ranker_scores, ref_scores)
        return {'recall': recall, 'ndcg': ndcg, 'spearman': spearman}
    
    # Reference 1: Similarity
    emb_sim_metrics = compute_ranker_metrics(
        emb_scores.sort_values(ascending=False).index.tolist(),
        emb_scores, binary_sim, graded_sim, similarity_scores
    )
    pearson_sim_metrics = compute_ranker_metrics(
        pearson_scores.sort_values(ascending=False).index.tolist(),
        pearson_scores, binary_sim, graded_sim, similarity_scores
    )
    spearman_sim_metrics = compute_ranker_metrics(
        spearman_scores.sort_values(ascending=False).index.tolist(),
        spearman_scores, binary_sim, graded_sim, similarity_scores
    )
    emb_rerank_sim_metrics = compute_ranker_metrics(
        emb_rerank_order,
        # For reranked: use dense integer ranks
        pd.Series(range(1, len(emb_rerank_order) + 1), index=emb_rerank_order),
        binary_sim, graded_sim, similarity_scores
    )
    pearson_rerank_sim_metrics = compute_ranker_metrics(
        pearson_rerank_order,
        pd.Series(range(1, len(pearson_rerank_order) + 1), index=pearson_rerank_order),
        binary_sim, graded_sim, similarity_scores
    )
    spearman_rerank_sim_metrics = compute_ranker_metrics(
        spearman_rerank_order,
        pd.Series(range(1, len(spearman_rerank_order) + 1), index=spearman_rerank_order),
        binary_sim, graded_sim, similarity_scores
    )
    
    # Reference 2: LiquidityUplift
    emb_uplift_metrics = compute_ranker_metrics(
        emb_scores.sort_values(ascending=False).index.tolist(),
        emb_scores, binary_uplift, graded_uplift, uplift_scores
    )
    pearson_uplift_metrics = compute_ranker_metrics(
        pearson_scores.sort_values(ascending=False).index.tolist(),
        pearson_scores, binary_uplift, graded_uplift, uplift_scores
    )
    spearman_uplift_metrics = compute_ranker_metrics(
        spearman_scores.sort_values(ascending=False).index.tolist(),
        spearman_scores, binary_uplift, graded_uplift, uplift_scores
    )
    emb_rerank_uplift_metrics = compute_ranker_metrics(
        emb_rerank_order,
        pd.Series(range(1, len(emb_rerank_order) + 1), index=emb_rerank_order),
        binary_uplift, graded_uplift, uplift_scores
    )
    pearson_rerank_uplift_metrics = compute_ranker_metrics(
        pearson_rerank_order,
        pd.Series(range(1, len(pearson_rerank_order) + 1), index=pearson_rerank_order),
        binary_uplift, graded_uplift, uplift_scores
    )
    spearman_rerank_uplift_metrics = compute_ranker_metrics(
        spearman_rerank_order,
        pd.Series(range(1, len(spearman_rerank_order) + 1), index=spearman_rerank_order),
        binary_uplift, graded_uplift, uplift_scores
    )
    
    # Reference 3: Utility
    emb_utility_metrics = compute_ranker_metrics(
        emb_scores.sort_values(ascending=False).index.tolist(),
        emb_scores, binary_utility, graded_utility, utility_scores
    )
    pearson_utility_metrics = compute_ranker_metrics(
        pearson_scores.sort_values(ascending=False).index.tolist(),
        pearson_scores, binary_utility, graded_utility, utility_scores
    )
    spearman_utility_metrics = compute_ranker_metrics(
        spearman_scores.sort_values(ascending=False).index.tolist(),
        spearman_scores, binary_utility, graded_utility, utility_scores
    )
    emb_rerank_utility_metrics = compute_ranker_metrics(
        emb_rerank_order,
        pd.Series(range(1, len(emb_rerank_order) + 1), index=emb_rerank_order),
        binary_utility, graded_utility, utility_scores
    )
    pearson_rerank_utility_metrics = compute_ranker_metrics(
        pearson_rerank_order,
        pd.Series(range(1, len(pearson_rerank_order) + 1), index=pearson_rerank_order),
        binary_utility, graded_utility, utility_scores
    )
    spearman_rerank_utility_metrics = compute_ranker_metrics(
        spearman_rerank_order,
        pd.Series(range(1, len(spearman_rerank_order) + 1), index=spearman_rerank_order),
        binary_utility, graded_utility, utility_scores
    )
    
    # Store query-level result
    query_result = {
        'query_symbol': query,
        # Similarity reference metrics
        'emb_sim_recall@10': emb_sim_metrics['recall'],
        'pearson_sim_recall@10': pearson_sim_metrics['recall'],
        'spearman_sim_recall@10': spearman_sim_metrics['recall'],
        'emb_rerank_sim_recall@10': emb_rerank_sim_metrics['recall'],
        'pearson_rerank_sim_recall@10': pearson_rerank_sim_metrics['recall'],
        'spearman_rerank_sim_recall@10': spearman_rerank_sim_metrics['recall'],
        'emb_sim_ndcg@10': emb_sim_metrics['ndcg'],
        'pearson_sim_ndcg@10': pearson_sim_metrics['ndcg'],
        'spearman_sim_ndcg@10': spearman_sim_metrics['ndcg'],
        'emb_rerank_sim_ndcg@10': emb_rerank_sim_metrics['ndcg'],
        'pearson_rerank_sim_ndcg@10': pearson_rerank_sim_metrics['ndcg'],
        'spearman_rerank_sim_ndcg@10': spearman_rerank_sim_metrics['ndcg'],
        'emb_sim_spearman': emb_sim_metrics['spearman'],
        'pearson_sim_spearman': pearson_sim_metrics['spearman'],
        'spearman_sim_spearman': spearman_sim_metrics['spearman'],
        'emb_rerank_sim_spearman': emb_rerank_sim_metrics['spearman'],
        'pearson_rerank_sim_spearman': pearson_rerank_sim_metrics['spearman'],
        'spearman_rerank_sim_spearman': spearman_rerank_sim_metrics['spearman'],
        # LiquidityUplift reference metrics
        'emb_uplift_recall@10': emb_uplift_metrics['recall'],
        'pearson_uplift_recall@10': pearson_uplift_metrics['recall'],
        'spearman_uplift_recall@10': spearman_uplift_metrics['recall'],
        'emb_rerank_uplift_recall@10': emb_rerank_uplift_metrics['recall'],
        'pearson_rerank_uplift_recall@10': pearson_rerank_uplift_metrics['recall'],
        'spearman_rerank_uplift_recall@10': spearman_rerank_uplift_metrics['recall'],
        'emb_uplift_ndcg@10': emb_uplift_metrics['ndcg'],
        'pearson_uplift_ndcg@10': pearson_uplift_metrics['ndcg'],
        'spearman_uplift_ndcg@10': spearman_uplift_metrics['ndcg'],
        'emb_rerank_uplift_ndcg@10': emb_rerank_uplift_metrics['ndcg'],
        'pearson_rerank_uplift_ndcg@10': pearson_rerank_uplift_metrics['ndcg'],
        'spearman_rerank_uplift_ndcg@10': spearman_rerank_uplift_metrics['ndcg'],
        'emb_uplift_spearman': emb_uplift_metrics['spearman'],
        'pearson_uplift_spearman': pearson_uplift_metrics['spearman'],
        'spearman_uplift_spearman': spearman_uplift_metrics['spearman'],
        'emb_rerank_uplift_spearman': emb_rerank_uplift_metrics['spearman'],
        'pearson_rerank_uplift_spearman': pearson_rerank_uplift_metrics['spearman'],
        'spearman_rerank_uplift_spearman': spearman_rerank_uplift_metrics['spearman'],
        # Utility reference metrics
        'emb_utility_recall@10': emb_utility_metrics['recall'],
        'pearson_utility_recall@10': pearson_utility_metrics['recall'],
        'spearman_utility_recall@10': spearman_utility_metrics['recall'],
        'emb_rerank_utility_recall@10': emb_rerank_utility_metrics['recall'],
        'pearson_rerank_utility_recall@10': pearson_rerank_utility_metrics['recall'],
        'spearman_rerank_utility_recall@10': spearman_rerank_utility_metrics['recall'],
        'emb_utility_ndcg@10': emb_utility_metrics['ndcg'],
        'pearson_utility_ndcg@10': pearson_utility_metrics['ndcg'],
        'spearman_utility_ndcg@10': spearman_utility_metrics['ndcg'],
        'emb_rerank_utility_ndcg@10': emb_rerank_utility_metrics['ndcg'],
        'pearson_rerank_utility_ndcg@10': pearson_rerank_utility_metrics['ndcg'],
        'spearman_rerank_utility_ndcg@10': spearman_rerank_utility_metrics['ndcg'],
        'emb_utility_spearman': emb_utility_metrics['spearman'],
        'pearson_utility_spearman': pearson_utility_metrics['spearman'],
        'spearman_utility_spearman': spearman_utility_metrics['spearman'],
        'emb_rerank_utility_spearman': emb_rerank_utility_metrics['spearman'],
        'pearson_rerank_utility_spearman': pearson_rerank_utility_metrics['spearman'],
        'spearman_rerank_utility_spearman': spearman_rerank_utility_metrics['spearman'],
        'n_candidates': len(candidates),
        'snapshot_date': snapshot_date,
    }
    all_results.append(query_result)
    
    # Save per-query parquet with all scores and references
    per_query_data = {
        'query_symbol': query,
        'candidate_symbol': candidates,
        # Ranker scores
        'emb_score': emb_scores.values,
        'pearson_score': pearson_scores.values,
        'spearman_score': spearman_scores.values,
        # Ground truth scores
        'SimilarityScore': similarity_scores.values,
        'LiquidityUplift': uplift_scores.values,
        'UtilityScore': utility_scores.values,
        # Binary relevance
        'binary_relevance_similarity': binary_sim.values,
        'binary_relevance_uplift': binary_uplift.values,
        'binary_relevance_utility': binary_utility.values,
        # Graded relevance
        'graded_relevance_similarity': graded_sim.values,
        'graded_relevance_uplift': graded_uplift.values,
        'graded_relevance_utility': graded_utility.values,
    }
    
    per_query_df = pd.DataFrame(per_query_data)
    per_query_df.to_parquet(
        output_dir / 'retrieval' / 'per_query' / f'{query}.parquet',
        index=False,
    )
```

- [ ] **Step 29: Commit the refactored loop**

```bash
git add scripts/evaluation/run_retrieval_metrics.py
git commit -m "refactor: update per-query loop for 6 rankers x 3 references"
```

---

## Task 4: Update Output CSV Generation

### Task 4.1: Generate three reference-family CSVs

- [ ] **Step 30: Update CSV generation for 3 references**

Replace the results aggregation section (lines 889-1080) with:

```python
# Aggregate results
results_df = pd.DataFrame(all_results)

# === Generate metrics CSVs per reference ===

# Reference 1: Similarity
similarity_metrics = pd.DataFrame({
    'metric_name': ['Recall@10', 'nDCG@10', 'Spearman'],
    'embedding': [
        results_df['emb_sim_recall@10'].mean(),
        results_df['emb_sim_ndcg@10'].mean(),
        results_df['emb_sim_spearman'].mean(),
    ],
    'pearson_corr': [
        results_df['pearson_sim_recall@10'].mean(),
        results_df['pearson_sim_ndcg@10'].mean(),
        results_df['pearson_sim_spearman'].mean(),
    ],
    'spearman_corr': [
        results_df['spearman_sim_recall@10'].mean(),
        results_df['spearman_sim_ndcg@10'].mean(),
        results_df['spearman_sim_spearman'].mean(),
    ],
    'embedding_rerank': [
        results_df['emb_rerank_sim_recall@10'].mean(),
        results_df['emb_rerank_sim_ndcg@10'].mean(),
        results_df['emb_rerank_sim_spearman'].mean(),
    ],
    'pearson_corr_rerank': [
        results_df['pearson_rerank_sim_recall@10'].mean(),
        results_df['pearson_rerank_sim_ndcg@10'].mean(),
        results_df['pearson_rerank_sim_spearman'].mean(),
    ],
    'spearman_corr_rerank': [
        results_df['spearman_rerank_sim_recall@10'].mean(),
        results_df['spearman_rerank_sim_ndcg@10'].mean(),
        results_df['spearman_rerank_sim_spearman'].mean(),
    ],
})

# Reference 2: LiquidityUplift
liquidity_uplift_metrics = pd.DataFrame({
    'metric_name': ['Recall@10', 'nDCG@10', 'Spearman'],
    'embedding': [
        results_df['emb_uplift_recall@10'].mean(),
        results_df['emb_uplift_ndcg@10'].mean(),
        results_df['emb_uplift_spearman'].mean(),
    ],
    'pearson_corr': [
        results_df['pearson_uplift_recall@10'].mean(),
        results_df['pearson_uplift_ndcg@10'].mean(),
        results_df['pearson_uplift_spearman'].mean(),
    ],
    'spearman_corr': [
        results_df['spearman_uplift_recall@10'].mean(),
        results_df['spearman_uplift_ndcg@10'].mean(),
        results_df['spearman_uplift_spearman'].mean(),
    ],
    'embedding_rerank': [
        results_df['emb_rerank_uplift_recall@10'].mean(),
        results_df['emb_rerank_uplift_ndcg@10'].mean(),
        results_df['emb_rerank_uplift_spearman'].mean(),
    ],
    'pearson_corr_rerank': [
        results_df['pearson_rerank_uplift_recall@10'].mean(),
        results_df['pearson_rerank_uplift_ndcg@10'].mean(),
        results_df['pearson_rerank_uplift_spearman'].mean(),
    ],
    'spearman_corr_rerank': [
        results_df['spearman_rerank_uplift_recall@10'].mean(),
        results_df['spearman_rerank_uplift_ndcg@10'].mean(),
        results_df['spearman_rerank_uplift_spearman'].mean(),
    ],
})

# Reference 3: Utility
utility_metrics = pd.DataFrame({
    'metric_name': ['Recall@10', 'nDCG@10', 'Spearman'],
    'embedding': [
        results_df['emb_utility_recall@10'].mean(),
        results_df['emb_utility_ndcg@10'].mean(),
        results_df['emb_utility_spearman'].mean(),
    ],
    'pearson_corr': [
        results_df['pearson_utility_recall@10'].mean(),
        results_df['pearson_utility_ndcg@10'].mean(),
        results_df['pearson_utility_spearman'].mean(),
    ],
    'spearman_corr': [
        results_df['spearman_utility_recall@10'].mean(),
        results_df['spearman_utility_ndcg@10'].mean(),
        results_df['spearman_utility_spearman'].mean(),
    ],
    'embedding_rerank': [
        results_df['emb_rerank_utility_recall@10'].mean(),
        results_df['emb_rerank_utility_ndcg@10'].mean(),
        results_df['emb_rerank_utility_spearman'].mean(),
    ],
    'pearson_corr_rerank': [
        results_df['pearson_rerank_utility_recall@10'].mean(),
        results_df['pearson_rerank_utility_ndcg@10'].mean(),
        results_df['pearson_rerank_utility_spearman'].mean(),
    ],
    'spearman_corr_rerank': [
        results_df['spearman_rerank_utility_recall@10'].mean(),
        results_df['spearman_rerank_utility_ndcg@10'].mean(),
        results_df['spearman_rerank_utility_spearman'].mean(),
    ],
})

# Save per-reference metrics
similarity_metrics.to_csv(output_dir / 'metrics' / 'retrieval_similarity.csv', index=False)
print(f'Saved: {output_dir / "metrics" / "retrieval_similarity.csv"}')

liquidity_uplift_metrics.to_csv(output_dir / 'metrics' / 'retrieval_liquidity_uplift.csv', index=False)
print(f'Saved: {output_dir / "metrics" / "retrieval_liquidity_uplift.csv"}')

utility_metrics.to_csv(output_dir / 'metrics' / 'retrieval_utility.csv', index=False)
print(f'Saved: {output_dir / "metrics" / "retrieval_utility.csv"}')

# Backward compatibility: Save recall_spearman.csv (similarity reference only)
similarity_metrics.to_csv(output_dir / 'metrics' / 'recall_spearman.csv', index=False)
print(f'Saved: {output_dir / "metrics" / "recall_spearman.csv"}')

# Combined summary: wide-format table with all 54 numbers
combined_summary = pd.DataFrame({
    'ranker': ['embedding', 'pearson_corr', 'spearman_corr', 
               'embedding_rerank', 'pearson_corr_rerank', 'spearman_corr_rerank'],
    # Similarity reference
    'similarity_recall@10': similarity_metrics['embedding'][0], similarity_metrics['pearson_corr'][0], similarity_metrics['spearman_corr'][0],
    'similarity_ndcg@10': similarity_metrics['embedding'][1], similarity_metrics['pearson_corr'][1], similarity_metrics['spearman_corr'][1],
    'similarity_spearman': similarity_metrics['embedding'][2], similarity_metrics['pearson_corr'][2], similarity_metrics['spearman_corr'][2],
    # LiquidityUplift reference  
    'uplift_recall@10': liquidity_uplift_metrics['embedding'][0], liquidity_uplift_metrics['pearson_corr'][0], liquidity_uplift_metrics['spearman_corr'][0],
    'uplift_ndcg@10': liquidity_uplift_metrics['embedding'][1], liquidity_uplift_metrics['pearson_corr'][1], liquidity_uplift_metrics['spearman_corr'][1],
    'uplift_spearman': liquidity_uplift_metrics['embedding'][2], liquidity_uplift_metrics['pearson_corr'][2], liquidity_uplift_metrics['spearman_corr'][2],
    # Utility reference
    'utility_recall@10': utility_metrics['embedding'][0], utility_metrics['pearson_corr'][0], utility_metrics['spearman_corr'][0],
    'utility_ndcg@10': utility_metrics['embedding'][1], utility_metrics['pearson_corr'][1], utility_metrics['spearman_corr'][1],
    'utility_spearman': utility_metrics['embedding'][2], utility_metrics['pearson_corr'][2], utility_metrics['spearman_corr'][2],
})

combined_summary.to_csv(output_dir / 'metrics' / 'retrieval_metrics_overall.csv', index=False)
print(f'Saved: {output_dir / "metrics" / "retrieval_metrics_overall.csv"}')
```

- [ ] **Step 31: Commit CSV generation updates**

```bash
git add scripts/evaluation/run_retrieval_metrics.py
git commit -m "feat: update output CSVs for 3-reference evaluation"
```

---

## Task 5: Update Visualization Outputs

### Task 5.1: Generate figures for all three references

- [ ] **Step 32: Update figure generation**

Replace the figure generation section (lines 1088-1140) with:

```python
# Generate plots
print('Generating visualization figures...')

# Similarity reference plots
plot_overall_metrics(
    metrics_df=similarity_metrics,
    output_path=output_dir / 'retrieval' / 'figures' / 'metrics_similarity_comparison.png',
    title='Retrieval Metrics - Similarity Reference',
)

# LiquidityUplift reference plots
plot_overall_metrics(
    metrics_df=liquidity_uplift_metrics,
    output_path=output_dir / 'retrieval' / 'figures' / 'metrics_liquidity_uplift_comparison.png',
    title='Retrieval Metrics - LiquidityUplift Reference',
)

# Utility reference plots
plot_overall_metrics(
    metrics_df=utility_metrics,
    output_path=output_dir / 'retrieval' / 'figures' / 'metrics_utility_comparison.png',
    title='Retrieval Metrics - Utility Reference',
)

# Grouped metrics by sector (using similarity reference as primary)
if sector_metrics:
    sector_fig_df = pd.DataFrame(sector_metrics)
    if len(sector_fig_df) > 0:
        plot_grouped_metrics(
            grouped_metrics_df=sector_fig_df,
            metric_col='Recall@10',
            group_col='sector',
            output_path=output_dir / 'retrieval' / 'figures' / 'metrics_by_sector.png',
        )

# Grouped metrics by liquidity quartile
if liq_metrics:
    liq_fig_df = pd.DataFrame(liq_metrics)
    if len(liq_fig_df) > 0:
        plot_grouped_metrics(
            grouped_metrics_df=liq_fig_df,
            metric_col='Recall@10',
            group_col='liquidity_quartile',
            output_path=output_dir / 'retrieval' / 'figures' / 'metrics_by_liquidity.png',
        )
```

- [ ] **Step 33: Commit figure generation updates**

```bash
git add scripts/evaluation/run_retrieval_metrics.py
git commit -m "feat: update figure generation for 3-reference evaluation"
```

---

## Task 6: Update Tests and Run Full Verification

### Task 6.1: Add integration test for full evaluation pipeline

- [ ] **Step 34: Write integration test**

Add to `tests/test_evaluation_retrieval_metrics.py`:

```python
class TestRevisedEvaluationPipeline:
    """Test the revised 6 rankers x 3 references pipeline."""
    
    def test_ground_truth_computation(self):
        """Test that all three ground truths are computed correctly."""
        from src.evaluation.ground_truth import (
            compute_similarity_score,
            compute_liquidity_uplift,
            compute_utility_score,
        )
        
        # Setup test data
        returns_df = pd.DataFrame({
            'date': pd.date_range('2019-01-01', periods=120, freq='D'),
            'AAPL': np.random.randn(120) * 0.02,
            'MSFT': np.random.randn(120) * 0.02,
        }).set_index('date')
        
        snapshot_df = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'gsector': [1, 1],
            'ggroup': [10, 10],
            'market_cap': [1e12, 9e11],
            'LiquidityScore': [0.8, 0.6],
        })
        
        # Compute all three ground truths
        sim = compute_similarity_score('AAPL', returns_df, snapshot_df, min_overlap=80)
        uplift = compute_liquidity_uplift('AAPL', snapshot_df.set_index('symbol')['LiquidityScore'])
        utility = compute_utility_score(sim, uplift)
        
        # Assertions
        assert len(sim) == 1  # Only MSFT
        assert 'MSFT' in sim.index
        assert uplift['MSFT'] == 0.6 - 0.8
        assert utility['MSFT'] == sim['MSFT'] * max(0, uplift['MSFT'])
```

- [ ] **Step 35: Run integration test**

Run: `python -m pytest tests/test_evaluation_retrieval_metrics.py::TestRevisedEvaluationPipeline::test_ground_truth_computation -v`
Expected: PASS

```bash
git add tests/test_evaluation_retrieval_metrics.py
git commit -m "test: add integration test for revised evaluation pipeline"
```

### Task 6.2: Run full type checking and linting

- [ ] **Step 36: Run type checker**

```bash
python -m mypy src/evaluation/ground_truth.py src/evaluation/metrics/retrieval.py scripts/evaluation/run_retrieval_metrics.py
```

- [ ] **Step 37: Run linter**

```bash
python -m ruff check src/evaluation/ground_truth.py src/evaluation/metrics/retrieval.py scripts/evaluation/run_retrieval_metrics.py
```

- [ ] **Step 38: Run full test suite**

```bash
python -m pytest tests/test_evaluation_retrieval_metrics.py tests/test_evaluation_ground_truth.py -v
```

- [ ] **Step 39: Commit final verification**

```bash
git commit -m "chore: verify types, lint, and all tests pass"
```

---

## Task 7: Update Documentation

### Task 7.1: Update EVALUATION_PLAN.md checklist

- [ ] **Step 40: Mark Block 4 as completed with revision note**

In `docs/EVALUATION_PLAN.md`, update Block 4 checklist item to:

```markdown
- [x] **Block 4: Retrieval Evaluation**
  - Implementation: `scripts/evaluation/run_retrieval_metrics.py`
  - Spec: `docs/superpowers/specs/2026-04-01-revised-retrieval-scoring-design.md`
  - Status: ✅ Revised - 6 rankers × 3 references (Similarity, LiquidityUplift, Utility)
  - Output: `results/metrics/retrieval_similarity.csv`, `results/metrics/retrieval_liquidity_uplift.csv`, `results/metrics/retrieval_utility.csv`
```

- [ ] **Step 41: Commit documentation update**

```bash
git add docs/EVALUATION_PLAN.md
git commit -m "docs: update evaluation plan with revised Block 4"
```

---

## Verification Steps

After implementation, verify the complete system:

- [ ] Run full retrieval evaluation: `python -m scripts.evaluation.run_retrieval_metrics --features data/processed/all_features.parquet --checkpoint checkpoints/last.ckpt --output-dir results/retrieval_revised`
- [ ] Verify 54 numbers are generated (6 rankers × 3 references × 3 metrics)
- [ ] Check that all three CSVs are created:
  - `results/retrieval_revised/metrics/retrieval_similarity.csv`
  - `results/retrieval_revised/metrics/retrieval_liquidity_uplift.csv`
  - `results/retrieval_revised/metrics/retrieval_utility.csv`
- [ ] Verify per-query parquet files contain all new columns
- [ ] Check that figures are generated for all three references
- [ ] Verify backward compatibility file `recall_spearman.csv` matches `retrieval_similarity.csv`

---

## Implementation Complete

Once all tasks are completed and verified, the revised Block 4 retrieval scoring will be fully operational. The evaluation will now correctly answer:

1. Does the retriever return stocks that are genuinely similar? (Similarity reference)
2. Among retrieved candidates, does reranking improve liquidity? (LiquidityUplift reference)
3. Which system best balances similarity and liquidity improvement? (Utility reference)

Co-Authored-By: Claude Code <noreply@anthropic.com>
