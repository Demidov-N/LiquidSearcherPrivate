"""Retrieval quality metrics for embedding evaluation.

Provides metrics for evaluating retrieval quality including Recall@k, nDCG@k,
Spearman rank correlation, and supporting utilities for candidate filtering,
score normalization, and hybrid scoring.

All primary rankers must share the same candidate intersection universe to ensure
fair comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

if TYPE_CHECKING:
    pass

# =============================================================================
# Candidate Eligibility Intersection
# =============================================================================


def compute_candidate_intersection(
    ranker_candidates: dict[str, set[str]],
) -> set[str]:
    """Compute intersection of candidates across multiple rankers.

    Ensures all primary rankers share the same candidate universe for fair
    comparison. A candidate must be eligible for ALL rankers to be included.

    Args:
        ranker_candidates: Dict mapping ranker name to set of eligible candidate symbols.
            Example: {"model": {"AAPL", "MSFT", ...}, "correlation": {"AAPL", "GOOGL", ...}}

    Returns:
        Set of symbols that are candidates for ALL rankers (intersection).

    Example:
        >>> model_candidates = {"AAPL", "MSFT", "GOOGL", "AMZN"}
        >>> corr_candidates = {"AAPL", "MSFT", "IBM", "GE"}
        >>> intersection = compute_candidate_intersection({
        ...     "model": model_candidates,
        ...     "correlation": corr_candidates
        ... })
        >>> intersection == {"AAPL", "MSFT"}
        True
    """
    if not ranker_candidates:
        return set()

    candidate_sets = list(ranker_candidates.values())
    intersection = candidate_sets[0].copy()

    for candidate_set in candidate_sets[1:]:
        intersection &= candidate_set

    return intersection


def filter_candidates_to_intersection(
    scores: pd.DataFrame,
    candidate_set: set[str],
    score_col: str = "score",
    candidate_col: str = "symbol",
) -> pd.DataFrame:
    """Filter candidate scores to only those in the intersection universe.

    Args:
        scores: DataFrame with candidate scores (must contain candidate_col and score_col)
        candidate_set: Set of eligible candidate symbols
        score_col: Name of the score column
        candidate_col: Name of the candidate symbol column

    Returns:
        Filtered DataFrame containing only candidates in the intersection
    """
    return scores[scores[candidate_col].isin(candidate_set)].copy()


# =============================================================================
# Core Retrieval Metrics
# =============================================================================


def compute_recall_at_k(
    predicted_relevant: set[str],
    ground_truth_relevant: set[str],
    k: int,
) -> float:
    """Compute Recall@k metric.

    Recall@k = (relevant items in top-k predicted) / (total relevant items)

    Handles edge cases:
    - Empty ground truth: returns NaN
    - k > number of predictions: uses all predictions
    - No relevant predictions: returns 0.0

    Args:
        predicted_relevant: Set of symbols predicted as relevant (top-k from model)
        ground_truth_relevant: Set of symbols that are truly relevant
        k: Cutoff position

    Returns:
        Recall@k as float, or NaN if ground truth is empty
    """
    if len(ground_truth_relevant) == 0:
        return np.nan

    if k <= 0:
        return np.nan

    # Take top-k predictions
    top_k = list(predicted_relevant)[:k]
    num_relevant_in_top_k = len(set(top_k) & ground_truth_relevant)

    return num_relevant_in_top_k / len(ground_truth_relevant)


def compute_ndcg_at_k(
    rankings: pd.Series,
    relevance_scores: pd.Series,
    k: int,
) -> float:
    """Compute nDCG@k (Normalized Discounted Cumulative Gain) metric.

    nDCG@k = DCG@k / IDCG@k

    Where:
    - DCG@k = sum of (rel_i / log2(i+1)) for i in 1..k
    - IDCG@k = ideal DCG@k (perfect ranking)

    Handles edge cases:
    - Empty rankings or relevance: returns NaN
    - No relevant items: returns NaN (can't normalize)
    - k > len(rankings): uses available rankings

    Args:
        rankings: Series of symbol rankings (rank 0 = best), indexed by symbol
        relevance_scores: Series of relevance scores indexed by symbol
        k: Cutoff position

    Returns:
        nDCG@k as float, or NaN if cannot be computed
    """
    if rankings.empty or relevance_scores.empty:
        return np.nan

    if k <= 0:
        return np.nan

    # Filter to symbols that exist in both
    common_symbols = rankings.index.intersection(relevance_scores.index)
    if len(common_symbols) == 0:
        return np.nan

    rankings = rankings.loc[common_symbols]
    relevance_scores = relevance_scores.loc[common_symbols]

    # Sort by ranking position and take top-k
    ranked_symbols = rankings.sort_values().index[:k]
    rel = relevance_scores.loc[ranked_symbols]

    # Compute DCG
    positions = np.arange(1, len(rel) + 1)
    dcg = np.sum(rel.values / np.log2(positions + 1))

    # Compute IDCG (ideal case: relevance sorted in descending order)
    ideal_rel = relevance_scores.sort_values(ascending=False).values[:k]
    ideal_positions = np.arange(1, len(ideal_rel) + 1)
    idcg = np.sum(ideal_rel / np.log2(ideal_positions + 1))

    if idcg == 0:
        return np.nan

    return dcg / idcg


def compute_spearman_correlation(
    predicted_ranking: pd.Series,
    reference_ranking: pd.Series,
) -> float:
    """Compute Spearman rank correlation between two rankings.

    Spearman ρ = 1 - (6 * sum(d_i^2)) / (n * (n^2 - 1))
    Where d_i = rank_i(predicted) - rank_i(reference)

    Handles edge cases:
    - Different symbols in rankings: uses intersection only
    - Empty intersection or single item: returns NaN
    - All tied ranks: returns NaN (undefined correlation)

    Args:
        predicted_ranking: Series of predicted ranks indexed by symbol
        reference_ranking: Series of reference ranks indexed by symbol

    Returns:
        Spearman correlation coefficient, or NaN if cannot be computed
    """
    # Find common symbols
    common = predicted_ranking.index.intersection(reference_ranking.index)

    if len(common) < 2:
        return np.nan

    pred_ranks = predicted_ranking.loc[common].values
    ref_ranks = reference_ranking.loc[common].values

    # Check for zero variance (all ties)
    if np.std(pred_ranks) == 0 or np.std(ref_ranks) == 0:
        return np.nan

    correlation, _ = spearmanr(pred_ranks, ref_ranks)

    if np.isnan(correlation):
        return np.nan

    return float(correlation)


# =============================================================================
# Deterministic Query Sampling
# =============================================================================


def sample_queries_deterministic(
    symbols: list[str],
    n_queries: int,
    seed: int = 42,
    stratify_by: pd.Series | None = None,
) -> list[str]:
    """Sample queries deterministically with optional stratification.

    Uses a fixed seed (default 42) to ensure reproducible sampling.
    When stratify_by is provided, ensures proportional representation from
    each stratum.

    Args:
        symbols: List of eligible symbols to sample from
        n_queries: Number of queries to sample
        seed: Random seed for reproducibility (default 42)
        stratify_by: Optional Series indexed by symbol with stratum labels
            for stratified sampling

    Returns:
        List of sampled query symbols

    Example:
        >>> symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "IBM", "GE"]
        >>> stratify = pd.Series([0, 0, 1, 1, 2, 2], index=symbols)
        >>> queries = sample_queries_deterministic(symbols, 4, seed=42, stratify_by=stratify)
        >>> len(queries)
        4
    """
    if n_queries <= 0:
        return []

    if n_queries >= len(symbols):
        return list(symbols)

    rng = np.random.RandomState(seed)

    if stratify_by is None:
        # Simple random sampling without stratification
        indices = rng.choice(len(symbols), size=n_queries, replace=False)
        return [symbols[i] for i in indices]

    # Stratified sampling
    strata = stratify_by.loc[symbols]
    unique_strata = strata.unique()
    sampled = []

    for stratum in unique_strata:
        stratum_symbols = [s for s in symbols if stratify_by.get(s) == stratum]
        n_stratum = len(stratum_symbols)

        if n_stratum == 0:
            continue

        # Proportional allocation
        proportion = n_stratum / len(symbols)
        n_to_sample = max(1, int(np.round(proportion * n_queries)))
        n_to_sample = min(n_to_sample, n_stratum)

        if n_to_sample > 0:
            indices = rng.choice(n_stratum, size=n_to_sample, replace=False)
            sampled.extend([stratum_symbols[i] for i in indices])

    # If we have fewer than requested due to rounding, fill randomly
    if len(sampled) < n_queries:
        remaining = [s for s in symbols if s not in sampled]
        n_fill = min(n_queries - len(sampled), len(remaining))
        if n_fill > 0:
            indices = rng.choice(len(remaining), size=n_fill, replace=False)
            sampled.extend([remaining[i] for i in indices])

    return sampled[:n_queries]


# =============================================================================
# Correlation Score Builder
# =============================================================================


def build_correlation_scores(
    prices: pd.DataFrame,
    lookback_days: int = 60,
    min_overlap: int = 40,
) -> pd.DataFrame:
    """Build correlation-based similarity scores using rolling lookback.

    Computes pairwise Pearson correlation between all stocks using a rolling
    window of lookback_days. Only includes pairs with at least min_overlap
    observations.

    Args:
        prices: DataFrame with DateTimeIndex and symbols as columns,
            containing price data (typically closing prices)
        lookback_days: Number of trading days for rolling correlation (default 60)
        min_overlap: Minimum overlapping observations required (default 40)

    Returns:
        DataFrame with multi-index (symbol_i, symbol_j) and columns:
            - correlation: Pearson correlation coefficient
            - overlap: Number of observations used

    Note:
        Returns NaN for pairs that don't meet min_overlap requirement.
        The output is NOT symmetric - both (i,j) and (j,i) pairs are included.
    """
    if prices.empty or len(prices.columns) < 2:
        return pd.DataFrame(columns=["correlation", "overlap"])

    # Compute rolling returns
    returns = prices.pct_change().dropna()

    if len(returns) < lookback_days:
        return pd.DataFrame(columns=["correlation", "overlap"])

    symbols = returns.columns.tolist()
    results = []

    for i, sym_i in enumerate(symbols):
        for sym_j in symbols[i + 1 :]:
            # Get overlapping returns
            common_mask = returns[sym_i].notna() & returns[sym_j].notna()
            ri = returns.loc[common_mask, sym_i].values
            rj = returns.loc[common_mask, sym_j].values

            if len(ri) < min_overlap:
                continue

            # Use most recent lookback_days of overlapping data
            n = min(lookback_days, len(ri))
            ri = ri[-n:]
            rj = rj[-n:]

            overlap = min(n, len(ri))

            if overlap < min_overlap:
                continue

            # Compute correlation
            if np.std(ri) == 0 or np.std(rj) == 0:
                correlation = np.nan
            else:
                correlation = np.corrcoef(ri, rj)[0, 1]

            # Store both directions for flexibility
            results.append(
                {
                    "symbol_i": sym_i,
                    "symbol_j": sym_j,
                    "correlation": correlation,
                    "overlap": overlap,
                }
            )
            results.append(
                {
                    "symbol_i": sym_j,
                    "symbol_j": sym_i,
                    "correlation": correlation,
                    "overlap": overlap,
                }
            )

    if not results:
        return pd.DataFrame(columns=["correlation", "overlap"])

    result_df = pd.DataFrame(results)
    result_df = result_df.set_index(["symbol_i", "symbol_j"])

    return result_df


def correlation_to_ranking(
    correlation_df: pd.DataFrame,
    query_symbol: str,
    candidate_symbols: list[str],
) -> pd.Series:
    """Convert correlation scores to a ranking for a query symbol.

    Args:
        correlation_df: DataFrame from build_correlation_scores with multi-index
        query_symbol: The symbol to get rankings for
        candidate_symbols: List of candidate symbols to rank

    Returns:
        Series of rankings (lower is better) indexed by candidate symbol
    """
    rankings = {}

    for candidate in candidate_symbols:
        try:
            corr_val = correlation_df.loc[(query_symbol, candidate), "correlation"]
            rankings[candidate] = corr_val
        except KeyError:
            rankings[candidate] = np.nan

    # Sort by correlation descending (higher correlation = better rank)
    sorted_symbols = sorted(rankings.keys(), key=lambda x: rankings[x] or -np.inf, reverse=True)

    return pd.Series(
        range(len(sorted_symbols)),
        index=sorted_symbols,
    )


# =============================================================================
# Score Normalization
# =============================================================================


def normalize_scores(
    scores: pd.Series,
    method: str = "minmax",
    eps: float = 1e-8,
) -> pd.Series:
    """Normalize scores to a standard range.

    Args:
        scores: Series of scores to normalize
        method: Normalization method - "minmax", "zscore", or "rank"
        eps: Small constant to avoid division by zero

    Returns:
        Normalized Series with same index

    Raises:
        ValueError: If method is not recognized
    """
    if scores.empty:
        return scores.copy()

    if method == "minmax":
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val < eps:
            return pd.Series(0.5, index=scores.index)
        return (scores - min_val) / (max_val - min_val)

    elif method == "zscore":
        std_val = scores.std()
        if std_val < eps:
            return pd.Series(0.0, index=scores.index)
        return (scores - scores.mean()) / std_val

    elif method == "rank":
        return scores.rank(method="average", ascending=True) / len(scores)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


# =============================================================================
# Hybrid Score Helper
# =============================================================================


def compute_hybrid_score(
    primary_scores: pd.Series,
    secondary_scores: pd.Series,
    primary_weight: float = 0.6,
    normalize: bool = True,
) -> pd.Series:
    """Compute hybrid score as weighted combination of primary and secondary scores.

    Combines two ranking signals with a weighting factor. When normalize=True,
    both scores are rank-normalized before combining to ensure fair weighting.

    Args:
        primary_scores: Primary ranking scores (e.g., embedding similarity)
        secondary_scores: Secondary ranking scores (e.g., correlation)
        primary_weight: Weight for primary scores (0-1). Secondary gets (1 - primary_weight)
        normalize: Whether to normalize scores before combining (default True)

    Returns:
        Hybrid scores Series with same index as primary_scores

    Example:
        >>> primary = pd.Series([0.9, 0.8, 0.7], index=["A", "B", "C"])
        >>> secondary = pd.Series([0.6, 0.5, 0.4], index=["A", "B", "C"])
        >>> hybrid = compute_hybrid_score(primary, secondary, primary_weight=0.7)
    """
    # Ensure alignment
    common_idx = primary_scores.index.intersection(secondary_scores.index)
    if len(common_idx) == 0:
        return pd.Series(dtype=float)

    primary = primary_scores.loc[common_idx]
    secondary = secondary_scores.loc[common_idx]

    if normalize:
        primary = normalize_scores(primary, method="rank")
        secondary = normalize_scores(secondary, method="rank")

    secondary_weight = 1.0 - primary_weight
    hybrid = primary_weight * primary + secondary_weight * secondary

    return hybrid


# =============================================================================
# Task 2: Additional Retrieval Helpers (FAST MODE)
# =============================================================================


def filter_valid_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Filter candidates to those with valid embedding and liquidity data.

    Primary intersection logic: requires has_embedding + has_liquidity_label +
    correlation_overlap>=40 if those columns exist.

    Args:
        candidates: DataFrame with symbol candidates (must have 'symbol' column)

    Returns:
        Filtered DataFrame with only valid candidates
    """
    if candidates.empty:
        return candidates

    result = candidates.copy()

    # Require has_embedding if column exists
    if "has_embedding" in result.columns:
        result = result[result["has_embedding"]]

    # Require has_liquidity_label if column exists
    if "has_liquidity_label" in result.columns:
        result = result[result["has_liquidity_label"]]

    # Require correlation_overlap >= 40 if column exists
    if "correlation_overlap" in result.columns:
        result = result[result["correlation_overlap"] >= 40]

    return result


def recall_at_k(
    ranking: list[str],
    binary_relevance: pd.Series,
    k: int = 10,
) -> float:
    """Compute Recall@k from a ranking list.

    Args:
        ranking: Ordered list of symbols (best first)
        binary_relevance: Series indexed by symbol, 1=relevant else 0
        k: Cutoff position

    Returns:
        Recall@k as float, or NaN if no relevant items exist
    """
    if k <= 0:
        return np.nan

    relevant_symbols = set(binary_relevance.index[binary_relevance == 1])
    if len(relevant_symbols) == 0:
        return np.nan

    top_k = set(ranking[:k])
    num_relevant_in_top_k = len(top_k & relevant_symbols)

    return num_relevant_in_top_k / len(relevant_symbols)


def ndcg_at_k(
    ranking: list[str],
    graded_relevance: pd.Series,
    k: int = 10,
) -> float:
    """Compute nDCG@k from a ranking list.

    Args:
        ranking: Ordered list of symbols (best first)
        graded_relevance: Series indexed by symbol with relevance scores
        k: Cutoff position

    Returns:
        nDCG@k as float, or NaN if cannot be computed
    """
    if k <= 0 or not ranking or graded_relevance.empty:
        return np.nan

    # Filter to symbols in ranking
    ranked_symbols = [s for s in ranking if s in graded_relevance.index][:k]
    if len(ranked_symbols) == 0:
        return np.nan

    rel = graded_relevance.loc[ranked_symbols].values
    positions = np.arange(1, len(rel) + 1)
    dcg = np.sum(rel / np.log2(positions + 1))

    # Ideal DCG
    ideal_rel = graded_relevance.sort_values(ascending=False).values[:k]
    ideal_positions = np.arange(1, len(ideal_rel) + 1)
    idcg = np.sum(ideal_rel / np.log2(ideal_positions + 1))

    if idcg == 0:
        return np.nan

    return dcg / idcg


def spearman_against_reference(
    ranking_scores: pd.Series,
    reference_scores: pd.Series,
) -> float:
    """Compute Spearman correlation between ranking scores and reference scores.

    Args:
        ranking_scores: Series of scores to evaluate (higher = better rank)
        reference_scores: Series of reference/ground truth scores

    Returns:
        Spearman correlation coefficient, or NaN if cannot be computed
    """
    common = ranking_scores.index.intersection(reference_scores.index)
    if len(common) < 2:
        return np.nan

    pred = ranking_scores.loc[common].values
    ref = reference_scores.loc[common].values

    if np.std(pred) == 0 or np.std(ref) == 0:
        return np.nan

    correlation, _ = spearmanr(pred, ref)
    if np.isnan(correlation):
        return np.nan

    return float(correlation)


def sample_query_set(
    snapshot_df: pd.DataFrame,
    n_queries: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample a deterministic query set from snapshot data.

    Stratifies by gsector and market_cap_tier if present;
    falls back to random deterministic sample otherwise.

    Args:
        snapshot_df: DataFrame with snapshot data (must have 'symbol' column)
        n_queries: Number of queries to sample
        seed: Random seed for reproducibility

    Returns:
        DataFrame with sampled query symbols
    """
    if snapshot_df.empty:
        return snapshot_df

    symbols = snapshot_df["symbol"].tolist()
    if n_queries >= len(symbols):
        return snapshot_df.copy()

    rng = np.random.RandomState(seed)

    # Check for stratification columns
    has_gsector = "gsector" in snapshot_df.columns
    has_market_cap_tier = "market_cap_tier" in snapshot_df.columns

    if has_gsector and has_market_cap_tier:
        # Stratified sampling
        snapshot_df = snapshot_df.copy()
        snapshot_df.loc[:, "_stratum"] = (
            snapshot_df["gsector"].astype(str) + "_" + snapshot_df["market_cap_tier"].astype(str)
        )
        strata = snapshot_df.set_index("symbol")["_stratum"]

        sampled_symbols = sample_queries_deterministic(
            symbols, n_queries, seed=seed, stratify_by=strata
        )
    else:
        # Simple random sampling
        indices = rng.choice(len(symbols), size=n_queries, replace=False)
        sampled_symbols = [symbols[i] for i in indices]

    return snapshot_df[snapshot_df["symbol"].isin(sampled_symbols)].copy()


def build_correlation_scores_for_query(
    returns_wide: pd.DataFrame,
    query_symbol: str,
    snapshot_date: pd.Timestamp,
    lookback: int = 60,
    min_overlap: int = 40,
) -> pd.Series:
    """Build correlation scores for a query symbol against all other symbols.

    Uses rolling lookback window on returns data.

    Args:
        returns_wide: DataFrame with DateTimeIndex and symbol columns (returns)
        query_symbol: Symbol to compute correlations against
        snapshot_date: Reference date for the snapshot
        lookback: Number of trading days for rolling correlation
        min_overlap: Minimum overlapping observations required

    Returns:
        Series indexed by symbol with correlation scores (query_symbol excluded)
    """
    if returns_wide.empty or query_symbol not in returns_wide.columns:
        return pd.Series(dtype=float)

    # Get lookback window (most recent lookback days before/at snapshot_date)
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

        # Align by index (dates)
        common_idx = query_returns.index.intersection(other_returns.index)
        if len(common_idx) < min_overlap:
            continue

        q_ret = query_returns.loc[common_idx].values
        o_ret = other_returns.loc[common_idx].values

        if len(q_ret) < min_overlap:
            continue

        # Compute correlation
        if np.std(q_ret) == 0 or np.std(o_ret) == 0:
            results[other_symbol] = np.nan
        else:
            corr = np.corrcoef(q_ret, o_ret)[0, 1]
            results[other_symbol] = corr

    return pd.Series(results, name=f"correlation_{query_symbol}")


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

    return pd.Series(results, name=f"spearman_{query_symbol}")


def normalize_scores_minmax(scores: pd.Series) -> pd.Series:
    """Normalize scores to [0, 1] range using min-max scaling.

    Args:
        scores: Series of scores to normalize

    Returns:
        Normalized Series with same index
    """
    if scores.empty:
        return scores.copy()

    min_val = scores.min()
    max_val = scores.max()
    range_val = max_val - min_val

    if range_val < 1e-8:
        return pd.Series(0.5, index=scores.index)

    return (scores - min_val) / range_val


def build_hybrid_scores(
    embedding_scores: pd.Series,
    liquidity20d_scores: pd.Series,
    alpha: float = 0.7,
) -> pd.Series:
    """Build hybrid scores combining embedding and liquidity signals.

    Args:
        embedding_scores: Series of embedding-based similarity scores
        liquidity20d_scores: Series of 20-day liquidity scores
        alpha: Weight for embedding scores (0-1), liquidity gets (1-alpha)

    Returns:
        Hybrid scores Series with same index as embedding_scores
    """
    common_idx = embedding_scores.index.intersection(liquidity20d_scores.index)
    if len(common_idx) == 0:
        return pd.Series(dtype=float)

    emb = normalize_scores_minmax(embedding_scores.loc[common_idx])
    liq = normalize_scores_minmax(liquidity20d_scores.loc[common_idx])

    hybrid = alpha * emb + (1 - alpha) * liq
    return hybrid


def build_snapshot_frame(period_df: pd.DataFrame) -> pd.DataFrame:
    """Build snapshot frame keeping last row per symbol.

    Args:
        period_df: DataFrame with 'symbol' column and temporal data

    Returns:
        DataFrame with one row per symbol (the last observation)
    """
    if period_df.empty:
        return period_df

    # Keep last row per symbol
    return period_df.groupby("symbol", as_index=False).last()


# =============================================================================
# Task 3: Model Integration Placeholders (FAST MODE)
# =============================================================================


def prepare_model_inputs(row: pd.Series) -> dict:
    """Prepare model inputs from a row of snapshot data.

    Placeholder for next integration step - prepares features for embedding model.

    Args:
        row: Series with snapshot data for a single symbol

    Returns:
        Dict with prepared inputs for the model
    """
    # TODO: Implement with actual feature preparation logic
    # Expected fields: returns_20d, volume_20d, market_cap, etc.
    return {
        "symbol": row.get("symbol"),
        "features": row.to_dict(),
    }


def compute_embedding_scores(
    snapshot_df: pd.DataFrame,
    checkpoint_path: str | Path,
) -> pd.Series:
    """Compute embedding scores for snapshot data using a trained model.

    Placeholder for next integration step - generates embedding-based similarity scores.

    Args:
        snapshot_df: DataFrame with snapshot data
        checkpoint_path: Path to model checkpoint

    Returns:
        Series indexed by symbol with embedding similarity scores
    """
    # TODO: Implement with actual embedding model inference
    # - Load model from checkpoint_path
    # - Encode snapshot_df features
    # - Compute pairwise cosine similarity
    # - Return as Series
    import warnings

    warnings.warn(
        "compute_embedding_scores is a placeholder - implement for Task 5 integration",
        UserWarning,
        stacklevel=2,
    )

    if snapshot_df.empty:
        return pd.Series(dtype=float)

    # Return uniform scores as placeholder
    return pd.Series(0.5, index=snapshot_df["symbol"])
