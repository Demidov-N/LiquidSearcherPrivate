"""Liquidity label utilities for retrieval evaluation.

Computes daily liquidity proxies, aggregates to period-level liquidity scores,
assigns quartiles, and provides graded relevance helpers for evaluation.
"""

import numpy as np
import pandas as pd


def _filter_valid_liquidity_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out rows with invalid (NaN) liquidity proxies at row level.

    A row is considered invalid if ANY of the liquidity proxy columns
    (spread_pct, amihud, turnover) contains NaN. This ensures period
    aggregations are computed over rows where all proxies are valid.

    Args:
        df: DataFrame with columns [symbol, date, spread_pct, amihud, turnover]

    Returns:
        DataFrame containing only rows where all three proxies are non-NaN
    """
    return df.dropna(subset=["spread_pct", "amihud", "turnover"])


def _add_liquidity_ranks_and_score(
    agg_df: pd.DataFrame,
    score_col: str,
) -> pd.DataFrame:
    """Add percentile ranks and weighted liquidity score to aggregated proxies."""
    n = len(agg_df)

    if n == 0:
        return agg_df.assign(
            spread_rank=pd.Series(dtype=float),
            amihud_rank=pd.Series(dtype=float),
            turnover_rank=pd.Series(dtype=float),
            **{score_col: pd.Series(dtype=float)},
        )

    if n > 1:
        spread_ranks = agg_df["spread_pct_median"].rank(method="average") / n
        amihud_ranks = agg_df["amihud_median"].rank(method="average") / n
        turnover_ranks = agg_df["turnover_median"].rank(method="average") / n
    else:
        spread_ranks = pd.Series([0.5], index=agg_df.index)
        amihud_ranks = pd.Series([0.5], index=agg_df.index)
        turnover_ranks = pd.Series([0.5], index=agg_df.index)

    agg_df = agg_df.assign(
        spread_rank=spread_ranks.values,
        amihud_rank=amihud_ranks.values,
        turnover_rank=turnover_ranks.values,
    )

    return agg_df.assign(
        **{
            score_col: (
                0.4 * (1 - agg_df["spread_rank"])
                + 0.4 * (1 - agg_df["amihud_rank"])
                + 0.2 * agg_df["turnover_rank"]
            )
        }
    )


def compute_daily_liquidity_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily liquidity proxies from price and volume data.

    Computes:
    - spread_pct: (askhi - bidlo) / ((askhi + bidlo) / 2)
    - amihud: abs(ret) / (abs(prc) * vol)
    - turnover: vol / shrout

    Args:
        df: DataFrame with columns [askhi, bidlo, prc, vol, ret, shrout]

    Returns:
        DataFrame with additional columns [spread_pct, amihud, turnover]
    """
    result = df.copy()

    # Spread percentage: handle non-positive denominator safely
    denominator = (df["askhi"] + df["bidlo"]) / 2
    result["spread_pct"] = np.where(
        denominator > 0,
        (df["askhi"] - df["bidlo"]) / denominator,
        np.nan,
    )

    # Amihud illiquidity: handle non-positive denominators
    price_volume = abs(df["prc"]) * df["vol"]
    result["amihud"] = np.where(
        price_volume > 0,
        abs(df["ret"]) / price_volume,
        np.nan,
    )

    # Turnover: handle non-positive shrout
    result["turnover"] = np.where(
        df["shrout"] > 0,
        df["vol"] / df["shrout"],
        np.nan,
    )

    return result


def aggregate_period_liquidity(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily liquidity proxies to stock-level period liquidity.

    Uses median aggregation for each proxy, then builds cross-sectional
    percentile ranks and a composite LiquidityScore.

    LiquidityScore = 0.4*(1 - spread_rank) + 0.4*(1 - amihud_rank) + 0.2*turnover_rank

    Higher LiquidityScore means more liquid.

    Invalid rows (those with NaN in any proxy column) are excluded at row
    level before aggregation.

    Args:
        df: DataFrame with columns [symbol, date, spread_pct, amihud, turnover]

    Returns:
        DataFrame with one row per symbol, containing median proxies,
        percentile ranks, and LiquidityScore
    """
    # Filter out invalid rows before aggregation
    valid_df = _filter_valid_liquidity_rows(df)

    # Aggregate daily values using median per symbol
    agg_df = (
        valid_df.groupby("symbol")
        .agg(
            spread_pct_median=("spread_pct", "median"),
            amihud_median=("amihud", "median"),
            turnover_median=("turnover", "median"),
        )
        .reset_index()
    )

    return _add_liquidity_ranks_and_score(agg_df, score_col="LiquidityScore")


def assign_liquidity_quartiles(scores: pd.Series) -> pd.Series:
    """Assign liquidity quartiles (0-3) based on LiquidityScore.

    Uses qcut with duplicates='drop' to handle ties robustly.
    Quartile 0 = least liquid, Quartile 3 = most liquid.

    Args:
        scores: Series of LiquidityScores with symbol index

    Returns:
        Series of quartile assignments (0, 1, 2, 3) with same index
    """
    if len(scores) == 0:
        return scores.copy()

    if len(scores) == 1:
        # Single stock gets quartile 1 (middle)
        return pd.Series([1], index=scores.index)

    try:
        # Use pd.qcut for equal-frequency quartile assignment
        # duplicates='drop' handles ties gracefully
        quartiles = pd.qcut(scores, q=4, labels=[0, 1, 2, 3], duplicates="drop")
        return quartiles.astype(int)
    except ValueError:
        # Fallback: if qcut fails (e.g., too many ties), use rank-based approach
        ranks = scores.rank(method="average")
        n = len(scores)
        quartiles = ((ranks - 1) / n * 4).astype(int).clip(0, 3)
        return quartiles


def compute_graded_relevance(
    query_score: float,
    candidate_scores: pd.Series,
    candidate_quartiles: pd.Series,
    query_quartile: int,
) -> pd.Series:
    """Compute graded relevance for a query against candidates.

    Grading scheme (from spec):
    - relevance 3: candidate is in the closest ceil(10% * n) by liquidity-score distance
    - relevance 2: candidate is in the next ceil(25% * n) - ceil(10% * n) by distance
    - relevance 1: candidate is in the same quartile but outside the top graded bands
    - relevance 0: otherwise

    Uses distance-threshold banding so labels depend on distance values rather than
    candidate ordering. Ties at thresholds are included in the higher-relevance band.

    Args:
        query_score: LiquidityScore of the query stock
        candidate_scores: Series of LiquidityScores for candidates
        candidate_quartiles: Series of quartile assignments for candidates
        query_quartile: Quartile of the query stock (0-3)

    Returns:
        Series of graded relevance scores (0-3) for each candidate
    """
    n = len(candidate_scores)

    if n == 0:
        return pd.Series([], dtype=int)

    # Compute absolute distance in liquidity score
    distances = abs(candidate_scores - query_score)

    # Find band sizes using ceiling.
    top_10_count = int(np.ceil(0.10 * n))
    top_25_count = int(np.ceil(0.25 * n))

    # Distance thresholds from sorted distances.
    sorted_distances = distances.sort_values()
    top_10_threshold = sorted_distances.iloc[top_10_count - 1]
    top_25_threshold = sorted_distances.iloc[top_25_count - 1]

    # Initialize result Series with zeros.
    graded = pd.Series(0, index=candidate_scores.index)

    # Assign relevance 3 to closest-distance band.
    mask_3 = distances <= top_10_threshold
    graded[mask_3] = 3

    # Assign relevance 2 to next distance band.
    mask_2 = (distances > top_10_threshold) & (distances <= top_25_threshold)
    graded[mask_2] = 2

    # Assign relevance 1 to same quartile but outside top 25% distance bands
    same_quartile = candidate_quartiles == query_quartile
    mask_1 = same_quartile & (graded == 0)  # Not already assigned 2 or 3
    graded[mask_1] = 1

    # Relevance 0: otherwise (already set)

    return graded


def aggregate_trailing_20d_liquidity(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    """Aggregate trailing 20-trading-day liquidity for point-in-time scoring.

    Used for hybrid reranking extension. Filters to exact last 20 trading days
    before snapshot_date (not calendar approximation) and computes median
    liquidity proxies per symbol.

    Eligibility requires 20 DISTINCT trading dates per symbol (duplicate rows
    on the same date do not satisfy eligibility).

    Args:
        df: DataFrame with columns [symbol, date, spread_pct, amihud, turnover]
        snapshot_date: The reference date for the trailing window

    Returns:
        DataFrame with one row per symbol that has at least 20 distinct trading
        days of data, containing trailing medians and LiquidityScore20d
    """
    # Filter to dates on or before snapshot_date and keep valid rows only.
    # This ensures symbol-level trailing windows are built from usable liquidity rows.
    eligible_df = df[df["date"] <= snapshot_date].copy()
    eligible_df = _filter_valid_liquidity_rows(eligible_df)

    if eligible_df.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "spread_pct_median",
                "amihud_median",
                "turnover_median",
                "LiquidityScore20d",
            ]
        )

    # Build symbol-specific trailing 20 trading days (distinct dates per symbol),
    # not a global market-wide 20-day window.
    dedup_by_date = eligible_df.sort_values(
        ["symbol", "date", "spread_pct", "amihud", "turnover"], kind="mergesort"
    ).drop_duplicates(subset=["symbol", "date"], keep="last")

    symbol_distinct_dates = dedup_by_date.groupby("symbol")["date"].nunique()
    eligible_symbols = symbol_distinct_dates[symbol_distinct_dates >= 20].index

    if len(eligible_symbols) == 0:
        return pd.DataFrame(
            columns=[
                "symbol",
                "spread_pct_median",
                "amihud_median",
                "turnover_median",
                "LiquidityScore20d",
            ]
        )

    trailing_df = (
        dedup_by_date[dedup_by_date["symbol"].isin(eligible_symbols)]
        .sort_values(["symbol", "date"])
        .groupby("symbol", as_index=False, group_keys=False)
        .tail(20)
    )

    # Aggregate using median
    agg_df = (
        trailing_df.groupby("symbol")
        .agg(
            spread_pct_median=("spread_pct", "median"),
            amihud_median=("amihud", "median"),
            turnover_median=("turnover", "median"),
        )
        .reset_index()
    )

    return _add_liquidity_ranks_and_score(agg_df, score_col="LiquidityScore20d")
