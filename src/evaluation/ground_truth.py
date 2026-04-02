"""Ground truth reference computations for retrieval evaluation.

Provides return-based similarity for similarity score labeling.
Additional components (sector similarity, size similarity, LiquidityUplift, UtilityScore) will be added in subsequent tasks.
"""

import numpy as np
import pandas as pd


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
        # BUG FIX #1: Sort index to ensure chronological order
        common_idx = query_returns.index.intersection(symbol_returns.index).sort_values()

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

    return pd.Series(results, name=f"return_similarity_{query_symbol}")


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
        if row["ggroup"] == query_ggroup:
            return 1.0
        elif row["gsector"] == query_gsector:
            return 0.5
        else:
            return 0.0

    # BUG FIX #3: Use set_index instead of manual index assignment
    candidates_df = candidates_df.set_index("symbol")
    result = candidates_df.apply(_sector_sim, axis=1)
    return result


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
    if query_symbol not in snapshot_df["symbol"].values:
        return pd.Series(dtype=float)

    # Filter out rows with NaN market_cap
    valid_df = snapshot_df.dropna(subset=["market_cap"])

    if len(valid_df) == 0:
        return pd.Series(dtype=float)

    # Check if query has valid market_cap
    if query_symbol not in valid_df["symbol"].values:
        return pd.Series(dtype=float)

    # BUG FIX #3 & #4: Set index to symbol first
    valid_df = valid_df.set_index("symbol")

    # Store query rank BEFORE dropping query
    mcap_ranks_all = valid_df["market_cap"].rank(pct=True)
    query_rank = mcap_ranks_all[query_symbol]

    # BUG FIX #4: Remove query symbol from results
    if query_symbol in valid_df.index:
        valid_df = valid_df.drop(query_symbol)

    if len(valid_df) == 0:
        return pd.Series(dtype=float)

    # Compute percentile ranks on valid data only (excluding query)
    mcap_ranks = valid_df["market_cap"].rank(pct=True)

    similarities = 1.0 - (mcap_ranks - query_rank).abs()

    return similarities


def compute_similarity_score(
    query_symbol: str,
    returns_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    min_overlap: int = 80,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute composite SimilarityScore.

    SimilarityScore = 0.34 * return_similarity + 0.33 * sector_similarity + 0.33 * size_similarity
    If size_similarity is unavailable (e.g., missing market_cap), falls back to:
    SimilarityScore = 0.5 * return_similarity + 0.5 * sector_similarity

    Args:
        query_symbol: Query symbol
        returns_df: DataFrame with date index and symbol columns (daily returns)
        snapshot_df: DataFrame with symbol, gsector, ggroup, market_cap columns
        min_overlap: Minimum overlapping observations for return similarity

    Returns:
        Tuple of (composite, return_sim, sector_sim, size_sim) Series indexed by symbol
    """
    # Return similarity (already excludes query_symbol)
    return_sim = compute_return_similarity_120d(returns_df, query_symbol, min_overlap)

    # Sector similarity
    query_row = snapshot_df[snapshot_df["symbol"] == query_symbol]
    if len(query_row) == 0:
        empty = pd.Series(dtype=float)
        return empty, empty, empty, empty

    query_gsector = query_row["gsector"].iloc[0]
    query_ggroup = query_row["ggroup"].iloc[0]

    sector_sim = compute_sector_similarity(query_gsector, query_ggroup, snapshot_df)

    # Size similarity (may be empty if market_cap data is missing, excludes query)
    size_sim = compute_size_similarity(query_symbol, snapshot_df)

    # Combine - align indices
    # First check if we have all three components
    has_size = len(size_sim) > 0

    if has_size:
        # Full 3-component similarity
        common_symbols = return_sim.index.intersection(sector_sim.index).intersection(
            size_sim.index
        )
        if len(common_symbols) == 0:
            composite = pd.Series(dtype=float)
        else:
            composite = (
                0.34 * return_sim.loc[common_symbols]
                + 0.33 * sector_sim.loc[common_symbols]
                + 0.33 * size_sim.loc[common_symbols]
            )
    else:
        # Fallback: 2-component similarity (return + sector only)
        common_symbols = return_sim.index.intersection(sector_sim.index)
        if len(common_symbols) == 0:
            composite = pd.Series(dtype=float)
        else:
            composite = (
                0.50 * return_sim.loc[common_symbols] + 0.50 * sector_sim.loc[common_symbols]
            )

    return composite, return_sim, sector_sim, size_sim


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
    relevance 2: next 15% (top 10-25%)
    relevance 1: next 25% (top 25-50%)
    relevance 0: bottom 50%

    Args:
        scores: Series of scores indexed by symbol

    Returns:
        Series with graded relevance (3, 2, 1, 0)
    """
    n = len(scores)
    if n == 0:
        return pd.Series(dtype=int)

    # BUG FIX #2: Use proper percentile cutoffs
    # Grade 3: top 10%
    # Grade 2: 10-25%
    # Grade 1: 25-50%
    # Grade 0: bottom 50%

    p90 = scores.quantile(0.90)  # top 10% threshold
    p75 = scores.quantile(0.75)  # top 25% threshold
    p50 = scores.quantile(0.50)  # median threshold

    graded = pd.Series(0, index=scores.index)
    graded[scores >= p90] = 3
    graded[(scores >= p75) & (scores < p90)] = 2
    graded[(scores >= p50) & (scores < p75)] = 1
    # Remainder is 0 (bottom 50%)

    return graded


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


def compute_fundamentals_similarity(
    query_symbol: str,
    snapshot_df: pd.DataFrame,
    fund_columns: list[str] = None,
) -> pd.Series:
    """Compute fundamentals similarity (market cap, etc.)."""
    if fund_columns is None:
        fund_columns = ["market_cap"]

    available_cols = [c for c in fund_columns if c in snapshot_df.columns]
    if not available_cols:
        return pd.Series(dtype=float)

    query_row = snapshot_df[snapshot_df["symbol"] == query_symbol]
    if len(query_row) == 0:
        return pd.Series(dtype=float)

    candidates = snapshot_df[snapshot_df["symbol"] != query_symbol].copy()
    similarities = pd.Series(0.0, index=candidates["symbol"])

    for col in available_cols:
        valid = candidates.dropna(subset=[col])
        if len(valid) == 0:
            continue

        ranks = valid[col].rank(pct=True)
        query_rank = query_row[col].iloc[0]
        col_sim = 1.0 - (ranks - query_rank).abs()
        col_sim.index = valid["symbol"]
        similarities = similarities.add(col_sim, fill_value=0)

    similarities = similarities / len(available_cols)
    return similarities


def compute_all_references(
    query_symbol: str,
    snapshot_df: pd.DataFrame,
    returns_df: pd.DataFrame = None,
) -> dict[str, pd.Series]:
    """Compute all ground truth reference scores.

    Returns dict with:
    - 'LiquidityUplift': candidate_liq - query_liq
    - 'ReturnSimilarity': 120-day Pearson correlation
    - 'SectorSimilarity': GICS match (1.0 ggroup, 0.5 sector)
    - 'FundamentalsSim': market cap similarity
    - 'LiquidityChar': LiquidityScore percentile similarity
    - 'TurnoverChar': Turnover percentile similarity
    """
    results = {}

    # 1. LiquidityUplift
    if "LiquidityScore" in snapshot_df.columns:
        query_liq = snapshot_df[snapshot_df["symbol"] == query_symbol]["LiquidityScore"].iloc[0]
        candidates = snapshot_df[snapshot_df["symbol"] != query_symbol]
        results["LiquidityUplift"] = candidates.set_index("symbol")["LiquidityScore"] - query_liq

    # 2. ReturnSimilarity
    if returns_df is not None and query_symbol in returns_df.columns:
        results["ReturnSimilarity"] = compute_return_similarity_120d(
            returns_df, query_symbol, min_overlap=80
        )

    # 3. SectorSimilarity
    query_row = snapshot_df[snapshot_df["symbol"] == query_symbol]
    if len(query_row) > 0 and "gsector" in query_row.columns:
        query_gsector = query_row["gsector"].iloc[0]
        query_ggroup = query_row["ggroup"].iloc[0]
        results["SectorSimilarity"] = compute_sector_similarity(
            query_gsector, query_ggroup, snapshot_df
        )

    # 4. FundamentalsSim
    results["FundamentalsSim"] = compute_fundamentals_similarity(
        query_symbol, snapshot_df, fund_columns=["market_cap"]
    )

    # 5. LiquidityChar
    if "LiquidityScore" in snapshot_df.columns:
        liq_ranks = snapshot_df["LiquidityScore"].rank(pct=True)
        query_liq_rank = liq_ranks[snapshot_df["symbol"] == query_symbol].iloc[0]
        liq_sim = 1.0 - (liq_ranks - query_liq_rank).abs()
        liq_sim.index = snapshot_df["symbol"]
        results["LiquidityChar"] = liq_sim.drop(query_symbol, errors="ignore")

    # 6. TurnoverChar
    turnover_col = "turnover_rank" if "turnover_rank" in snapshot_df.columns else "turnover"
    if turnover_col in snapshot_df.columns:
        turn_ranks = snapshot_df[turnover_col].rank(pct=True)
        query_turn_rank = turn_ranks[snapshot_df["symbol"] == query_symbol].iloc[0]
        turn_sim = 1.0 - (turn_ranks - query_turn_rank).abs()
        turn_sim.index = snapshot_df["symbol"]
        results["TurnoverChar"] = turn_sim.drop(query_symbol, errors="ignore")

    return results
