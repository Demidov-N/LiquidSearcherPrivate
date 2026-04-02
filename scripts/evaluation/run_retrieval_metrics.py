#!/usr/bin/env python
"""
Run retrieval metrics evaluation for LiquidSearcher.

Computes Recall@10, Spearman rank correlation, and nDCG@10 for embedding-based
retrieval against correlation and liquidity-distance baselines.

Example:
    python -m scripts.evaluation.run_retrieval_metrics \\
        --checkpoint checkpoints/last.ckpt \\
        --features data/processed/all_features.parquet \\
        --period-start 2019-01-01 \\
        --period-end 2019-12-31 \\
        --n-queries 50 \\
        --seed 42 \\
        --output-dir results/retrieval_test

Hybrid mode (optional):
    python -m scripts.evaluation.run_retrieval_metrics \\
        --checkpoint checkpoints/last.ckpt \\
        --features data/processed/all_features.parquet \\
        --run-hybrid \\
        --output-dir results/retrieval_hybrid
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812

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
    build_spearman_scores_for_query,
    build_snapshot_frame,
    ndcg_at_k,
    recall_at_k,
    sample_query_set,
    spearman_against_reference,
)
from src.evaluation.utils.feature_loader import FeatureLoader
from src.evaluation.utils.liquidity_labels import (
    aggregate_period_liquidity,
    aggregate_trailing_20d_liquidity,
    assign_liquidity_quartiles,
    compute_daily_liquidity_proxies,
)
from src.evaluation.visualizations.retrieval_plots import (
    plot_grouped_metrics,
    plot_overall_metrics,
)
from src.models.dual_encoder import DualEncoder
from src.training.data_module import TABULAR_CONTINUOUS_NAMES, TEMPORAL_FEATURE_NAMES

warnings.filterwarnings("ignore")

# Sector mapping for gsector categorical feature (GICS sectors 0-10)
SECTOR_MAP = {
    "0": 0,
    "1": 0,
    "2": 1,
    "3": 2,
    "4": 2,
    "5": 3,
    "6": 4,
    "7": 5,
    "8": 6,
    "9": 7,
    "10": 8,
    "11": 9,
    "12": 10,
    "15": 1,
    "20": 2,
    "25": 3,
    "30": 4,
    "35": 5,
    "40": 6,
    "45": 7,
    "50": 8,
    "55": 9,
    "60": 10,
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run retrieval metrics evaluation for LiquidSearcher"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/last.ckpt",
        help="Path to model checkpoint (for embedding scores)",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="data/processed/all_features.parquet",
        help="Path to features parquet file",
    )
    parser.add_argument(
        "--period-start",
        type=str,
        default="2019-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--period-end",
        type=str,
        default="2019-12-31",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=50,
        help="Number of query stocks to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/retrieval_test",
        help="Output directory for results",
    )
    parser.add_argument(
        "--run-hybrid",
        action="store_true",
        help="Also compute hybrid retrieval using top-50 embedding shortlist + LiquidityScore20d",
    )
    return parser.parse_args()


def load_dual_encoder_model(checkpoint_path: str, device: str = "cpu") -> DualEncoder:
    """Load DualEncoder model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded DualEncoder model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hyperparams = checkpoint.get("hyper_parameters", {})

    model = DualEncoder(
        temporal_input_dim=hyperparams.get("temporal_input_dim", 13),
        tabular_continuous_dim=hyperparams.get("tabular_continuous_dim", 15),
        embedding_dim=hyperparams.get("embedding_dim", 128),
    )

    state_dict = checkpoint.get("state_dict", checkpoint)
    # Handle 'model.' prefix from DataParallel/training
    if any(k.startswith("model.") for k in state_dict):
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)

    return model


def _prepare_temporal_features(row: pd.Series, window_size: int = 60) -> torch.Tensor:
    """Prepare temporal tensor from a dataframe row.

    Creates a (window_size, num_temporal_features) tensor with the row's
    values placed in the last position (most recent), zeros elsewhere.

    Args:
        row: DataFrame row with temporal features
        window_size: Size of temporal window (default 60)

    Returns:
        Tensor of shape (1, window_size, num_temporal_features)
    """
    num_features = len(TEMPORAL_FEATURE_NAMES)
    temporal = torch.zeros(1, window_size, num_features)

    for i, col in enumerate(TEMPORAL_FEATURE_NAMES):
        if col in row and not pd.isna(row[col]):
            temporal[0, -1, i] = float(row[col])

    return temporal


def _prepare_tabular_features(row: pd.Series) -> torch.Tensor:
    """Prepare tabular continuous tensor from a dataframe row.

    Args:
        row: DataFrame row with tabular features

    Returns:
        Tensor of shape (1, num_tabular_features)
    """
    num_features = len(TABULAR_CONTINUOUS_NAMES)
    tabular = torch.zeros(1, num_features)

    for i, col in enumerate(TABULAR_CONTINUOUS_NAMES):
        if col in row and not pd.isna(row[col]):
            tabular[0, i] = float(row[col])

    return tabular


def _prepare_categorical_features(row: pd.Series) -> torch.Tensor:
    """Prepare categorical tensor from a dataframe row.

    Args:
        row: DataFrame row with gsector/ggroup features

    Returns:
        Tensor of shape (1, 2) with [gsector, ggroup] as integer indices
    """
    categorical = torch.zeros(1, 2, dtype=torch.long)

    if "gsector" in row and not pd.isna(row["gsector"]):
        sector_code = str(int(float(row["gsector"])))
        if sector_code in SECTOR_MAP:
            categorical[0, 0] = SECTOR_MAP[sector_code]
        else:
            # Clamp to valid range
            categorical[0, 0] = min(max(int(float(sector_code)), 0), 10)

    if "ggroup" in row and not pd.isna(row["ggroup"]):
        ggroup_code = int(float(row["ggroup"]))
        # Map 4-digit GICS code to 0-24 index
        # GICS groups: 1010, 1020, 1030, ..., 6020 (roughly 24 unique values)
        group_idx = (ggroup_code - 1010) // 100
        categorical[0, 1] = max(0, min(24, group_idx))

    return categorical


def compute_embeddings_batch(
    snapshot_df: pd.DataFrame,
    period_df: pd.DataFrame,
    model: DualEncoder,
    device: str = "cpu",
    window_size: int = 60,
) -> dict[str, torch.Tensor]:
    """Compute embeddings for all symbols in snapshot_df using real model inference.

    Args:
        snapshot_df: DataFrame with snapshot data (one row per symbol)
        period_df: Full period DataFrame with temporal history
        model: Loaded DualEncoder model
        device: Device to run inference on
        window_size: Size of temporal window

    Returns:
        Dict mapping symbol -> embedding tensor
    """
    # Build lookup: for each symbol, get last window_size rows from period_df
    # First deduplicate to one row per symbol per date
    period_df = period_df.sort_values(["symbol", "date"])
    period_df = period_df.groupby(["symbol", "date"], as_index=False).last()

    # Create a symbol -> last N rows mapping for efficiency
    symbol_windows = {}
    for symbol in snapshot_df["symbol"].unique():
        symbol_data = period_df[period_df["symbol"] == symbol]
        if len(symbol_data) == 0:
            continue

        # Get last window_size rows
        window_data = symbol_data.tail(window_size)

        # Build temporal tensor
        num_features = len(TEMPORAL_FEATURE_NAMES)
        temporal = torch.zeros(1, window_size, num_features)

        # Fill temporal tensor - take only the last window_size rows
        n_rows = min(len(window_data), window_size)
        start_idx = window_size - n_rows

        for idx, (_, row) in enumerate(window_data.tail(n_rows).iterrows()):
            for i, col in enumerate(TEMPORAL_FEATURE_NAMES):
                if col in row and not pd.isna(row[col]):
                    temporal[0, start_idx + idx, i] = float(row[col])

        # Zero-padding is already done by initialization

        symbol_windows[symbol] = temporal

    # Compute embeddings in batches for efficiency
    embeddings = {}
    batch_size = 32

    symbols_list = list(symbol_windows.keys())

    with torch.no_grad():
        for i in range(0, len(symbols_list), batch_size):
            batch_symbols = symbols_list[i : i + batch_size]

            # Prepare batch tensors
            batch_temporal = []
            batch_tabular = []
            batch_categorical = []

            for symbol in batch_symbols:
                # Get snapshot row for tabular/categorical features
                row = snapshot_df[snapshot_df["symbol"] == symbol].iloc[0]

                temporal = symbol_windows[symbol]
                tabular = _prepare_tabular_features(row)
                categorical = _prepare_categorical_features(row)

                batch_temporal.append(temporal)
                batch_tabular.append(tabular)
                batch_categorical.append(categorical)

            # Stack into batches
            batch_temporal = torch.cat(batch_temporal, dim=0).to(device)
            batch_tabular = torch.cat(batch_tabular, dim=0).to(device)
            batch_categorical = torch.cat(batch_categorical, dim=0).to(device)

            # Get joint embeddings
            joint_embs = model.get_joint_embedding(batch_temporal, batch_tabular, batch_categorical)

            # Store embeddings
            for j, symbol in enumerate(batch_symbols):
                embeddings[symbol] = joint_embs[j].cpu()

            if (i + batch_size) % 500 < batch_size:
                print(
                    f"  Computed embeddings for {min(i + batch_size, len(symbols_list))}/{len(symbols_list)} symbols"
                )

    return embeddings


def compute_embedding_scores_from_embeddings(
    embeddings: dict[str, torch.Tensor],
    query_symbols: list[str],
    snapshot_df: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Compute per-query embedding scores using pre-computed embeddings.

    Uses cosine similarity between query embedding and all candidate embeddings.

    Args:
        embeddings: Dict mapping symbol -> embedding tensor
        query_symbols: List of query symbols
        snapshot_df: DataFrame with snapshot data (for getting candidate list)

    Returns:
        Dict mapping query_symbol -> Series of cosine similarity scores for all candidates
    """
    all_symbols = snapshot_df["symbol"].tolist()
    scores_dict = {}

    # Stack all embeddings for efficient computation
    symbol_list = [s for s in all_symbols if s in embeddings]
    if len(symbol_list) == 0:
        return {}

    embeddings_matrix = torch.stack([embeddings[s] for s in symbol_list])

    for query in query_symbols:
        if query not in embeddings:
            # Fallback to zeros if query embedding not available
            candidates = [s for s in all_symbols if s != query]
            scores_dict[query] = pd.Series(0.5, index=candidates)
            continue

        query_emb = embeddings[query].unsqueeze(0)  # (1, embedding_dim)

        # Compute cosine similarity
        similarities = F.cosine_similarity(query_emb, embeddings_matrix).numpy()

        # Build scores series, excluding self
        candidates = [s for s in symbol_list if s != query]
        sim_values = [
            similarities[symbol_list.index(s)] if s in symbol_list else 0.5 for s in candidates
        ]

        scores_dict[query] = pd.Series(sim_values, index=candidates)

    return scores_dict


def compute_embedding_scores(
    snapshot_df: pd.DataFrame,
    period_df: pd.DataFrame,
    query_symbols: list[str],
    checkpoint_path: str,
    device: str = "cpu",
) -> dict[str, pd.Series]:
    """Compute embedding scores using real model inference.

    Args:
        snapshot_df: DataFrame with snapshot data
        period_df: Full period DataFrame with temporal history
        query_symbols: List of query symbols
        checkpoint_path: Path to model checkpoint
        device: Device to run inference on

    Returns:
        Dict mapping query_symbol -> Series of embedding scores for all candidates
    """
    print(f"Loading model from {checkpoint_path}...")
    model = load_dual_encoder_model(checkpoint_path, device=device)

    print("Computing embeddings for all symbols...")
    embeddings = compute_embeddings_batch(
        snapshot_df=snapshot_df,
        period_df=period_df,
        model=model,
        device=device,
    )
    print(f"Computed embeddings for {len(embeddings)} symbols\n")

    print("Computing query embedding scores...")
    scores_dict = compute_embedding_scores_from_embeddings(
        embeddings=embeddings,
        query_symbols=query_symbols,
        snapshot_df=snapshot_df,
    )

    return scores_dict


def compute_liquidity_distance_ranking(
    query_symbol: str,
    snapshot_df: pd.DataFrame,
) -> pd.Series:
    """Compute liquidity-distance based ranking for a query.

    Uses LiquidityScore to rank candidates - closer liquidity = better rank.

    Args:
        query_symbol: Query symbol
        snapshot_df: DataFrame with snapshot data including LiquidityScore

    Returns:
        Series of rankings (lower = better) indexed by candidate symbol
    """
    query_score = snapshot_df.loc[snapshot_df["symbol"] == query_symbol, "LiquidityScore"]
    if len(query_score) == 0:
        return pd.Series(dtype=float)

    query_score = query_score.iloc[0]

    candidates = snapshot_df[snapshot_df["symbol"] != query_symbol].copy()
    if candidates.empty:
        return pd.Series(dtype=float)

    # Compute absolute distance in liquidity score
    candidates["liq_distance"] = abs(candidates["LiquidityScore"] - query_score)

    # Rank by distance (lower distance = better rank = lower number)
    candidates["liq_rank"] = candidates["liq_distance"].rank(method="average")

    return candidates.set_index("symbol")["liq_rank"]


def build_ground_truth_relevance(
    query_symbol: str,
    snapshot_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Build ground truth relevance for a query.

    Returns:
        Tuple of (binary_relevance, graded_relevance):
        - binary_relevance: Series with 1 for same quartile, 0 otherwise
        - graded_relevance: Series with liquidity-distance based graded scores
    """
    query_row = snapshot_df[snapshot_df["symbol"] == query_symbol]
    if len(query_row) == 0:
        return pd.Series(dtype=int), pd.Series(dtype=float)

    query_quartile = query_row["liquidity_quartile"].iloc[0]
    query_score = query_row["LiquidityScore"].iloc[0]

    candidates = snapshot_df[snapshot_df["symbol"] != query_symbol].copy()
    if candidates.empty:
        return pd.Series(dtype=int), pd.Series(dtype=float)

    # Binary relevance: same quartile
    binary = (candidates["liquidity_quartile"] == query_quartile).astype(int)
    binary.index = candidates["symbol"]

    # Graded relevance: based on liquidity distance
    candidates["liq_distance"] = abs(candidates["LiquidityScore"] - query_score)
    n = len(candidates)

    if n == 0:
        return binary, pd.Series(dtype=float)

    # Top 10% = grade 3, next 15% = grade 2, same quartile = grade 1, else 0
    top_10_count = max(1, int(np.ceil(0.10 * n)))
    top_25_count = max(1, int(np.ceil(0.25 * n)))

    sorted_distances = candidates["liq_distance"].sort_values()
    if len(sorted_distances) >= top_10_count:
        top_10_threshold = sorted_distances.iloc[top_10_count - 1]
    else:
        top_10_threshold = sorted_distances.max()

    if len(sorted_distances) >= top_25_count:
        top_25_threshold = sorted_distances.iloc[top_25_count - 1]
    else:
        top_25_threshold = sorted_distances.max()

    graded = pd.Series(0, index=candidates["symbol"].values)
    liq_dist = candidates["liq_distance"].values
    liq_quartile = candidates["liquidity_quartile"].values

    # Boolean masks as numpy arrays to avoid alignment issues
    mask_3 = liq_dist <= top_10_threshold
    mask_2 = (liq_dist > top_10_threshold) & (liq_dist <= top_25_threshold)
    mask_1 = (liq_quartile == query_quartile) & (graded.values == 0)

    # Use numpy indexing to avoid alignment issues
    graded.values[mask_3] = 3
    graded.values[mask_2] = 2
    graded.values[mask_1] = 1

    return binary, graded


def compute_correlation_ranking(
    returns_df: pd.DataFrame,
    query_symbol: str,
    snapshot_date: pd.Timestamp,
    lookback: int = 60,
) -> pd.Series:
    """Compute correlation-based ranking for a query.

    Args:
        returns_df: DataFrame with date index and symbol columns (returns)
        query_symbol: Query symbol
        snapshot_date: Reference date for the snapshot
        lookback: Number of trading days for rolling correlation

    Returns:
        Series of correlation scores indexed by candidate symbol
    """
    corr_scores = build_correlation_scores_for_query(
        returns_wide=returns_df,
        query_symbol=query_symbol,
        snapshot_date=snapshot_date,
        lookback=lookback,
        min_overlap=40,
    )
    return corr_scores


def run_evaluation_pipeline(
    features_path: str,
    checkpoint_path: str,
    period_start: str,
    period_end: str,
    n_queries: int,
    seed: int,
    output_dir: str,
    run_hybrid: bool = False,
) -> dict:
    """Run the full retrieval evaluation pipeline.

    Args:
        features_path: Path to features parquet
        checkpoint_path: Path to model checkpoint
        period_start: Start date
        period_end: End date
        n_queries: Number of queries to sample
        seed: Random seed
        output_dir: Output directory
        run_hybrid: Whether to compute hybrid scores

    Returns:
        Dict with all computed metrics
    """
    output_dir = Path(output_dir)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (output_dir / "retrieval" / "per_query").mkdir(parents=True, exist_ok=True)
    (output_dir / "retrieval" / "figures").mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("LIQUIDSEARCHER RETRIEVAL METRICS EVALUATION")
    print("=" * 70 + "\n")

    # Load features
    print(f"Loading features from {features_path}...")
    loader = FeatureLoader(features_path)

    period_df = loader.load_period(period_start, period_end)
    print(f"Period: {period_start} to {period_end}")
    print(f"Loaded: {len(period_df):,} rows, {period_df['symbol'].nunique()} symbols\n")

    # Compute daily liquidity proxies
    print("Computing daily liquidity proxies...")
    period_df = compute_daily_liquidity_proxies(period_df)

    # Aggregate to period liquidity
    print("Aggregating period liquidity labels...")
    liquidity_df = aggregate_period_liquidity(period_df)

    # Assign quartiles
    liquidity_df["liquidity_quartile"] = assign_liquidity_quartiles(liquidity_df["LiquidityScore"])

    # Build snapshot frame (last observation per symbol)
    print("Building snapshot frame...")
    snapshot_df = build_snapshot_frame(period_df)

    # Merge liquidity data
    snapshot_df = snapshot_df.merge(
        liquidity_df[["symbol", "LiquidityScore", "liquidity_quartile"]],
        on="symbol",
        how="left",
    )

    # Filter to symbols with liquidity labels
    snapshot_df = snapshot_df[snapshot_df["LiquidityScore"].notna()].copy()
    print(f"Snapshot: {len(snapshot_df)} symbols with liquidity labels\n")

    # Sample query set
    print(f"Sampling {n_queries} queries (seed={seed})...")
    query_df = sample_query_set(snapshot_df, n_queries=n_queries, seed=seed)
    query_symbols = query_df["symbol"].tolist()
    print(f"Selected {len(query_symbols)} queries\n")

    # Get snapshot date (last date in period)
    snapshot_date = snapshot_df["date"].max()

    # Compute returns for correlation.
    # Deduplicate (date, symbol) first since raw parquet may contain duplicates.
    print("Computing returns for correlation-based ranking...")
    returns_input = (
        period_df[["date", "symbol", "ret"]]
        .sort_values(["date", "symbol"])
        .groupby(["date", "symbol"], as_index=False)["ret"]
        .last()
    )
    returns = returns_input.pivot(index="date", columns="symbol", values="ret")
    returns = returns.loc[returns.index <= snapshot_date].tail(252)  # Last year of returns
    print(f"Returns shape: {returns.shape}\n")

    # Compute extended returns for 120-day similarity label (separate from 60d ranker window)
    print("Computing 120-day returns for similarity label...")
    returns_120d = returns_input.pivot(index="date", columns="symbol", values="ret")
    returns_120d = returns_120d.loc[returns_120d.index <= snapshot_date].tail(252)

    # Compute embedding scores using real model inference
    print("Computing embedding scores with DualEncoder model...")
    embedding_scores_dict = compute_embedding_scores(
        snapshot_df=snapshot_df,
        period_df=period_df,
        query_symbols=query_symbols,
        checkpoint_path=checkpoint_path,
        device="cpu",
    )

    # Compute hybrid liquidity scores for top-50 shortlist if running hybrid
    hybrid_20d_scores = None
    if run_hybrid:
        print("Computing 20-day trailing liquidity scores for hybrid reranking...")
        hybrid_20d_df = aggregate_trailing_20d_liquidity(period_df, snapshot_date=snapshot_date)
        if not hybrid_20d_df.empty:
            hybrid_20d_scores = hybrid_20d_df.set_index("symbol")["LiquidityScore20d"]
            print(f"20d liquidity computed for {len(hybrid_20d_scores)} symbols\n")

    # Compute 20-day trailing liquidity for reranking (if not already done)
    if hybrid_20d_scores is None:
        print("Computing 20-day trailing liquidity scores...")
        hybrid_20d_df = aggregate_trailing_20d_liquidity(period_df, snapshot_date=snapshot_date)
        if not hybrid_20d_df.empty:
            hybrid_20d_scores = hybrid_20d_df.set_index("symbol")["LiquidityScore20d"]

    # Run per-query evaluation
    print("Running per-query evaluation...")
    print("Computing 6 rankers × 3 references × 3 metrics = 54 values per query")
    print("-" * 70)

    # Define the 6 rankers
    RANKERS = [
        "embedding",
        "pearson_corr",
        "spearman_corr",
        "embedding_rerank",
        "pearson_corr_rerank",
        "spearman_corr_rerank",
    ]

    # Define the 6 references (3 original + 3 new granular)
    REFERENCES = [
        "Similarity",  # Original: composite similarity
        "LiquidityUplift",  # Original: liquidity difference
        "Utility",  # Original: combined utility
        "ReturnSimilarity",  # NEW: 120-day return correlation only
        "SectorSimilarity",  # NEW: GICS sector matching
        "FundamentalsSim",  # NEW: market cap similarity
        "LiquidityChar",  # NEW: liquidity percentile similarity
        "TurnoverChar",  # NEW: turnover percentile similarity
    ]

    # Define the 3 metrics
    METRICS = ["Recall@10", "nDCG@10", "Spearman"]

    all_results = []
    query_manifest = []

    for i, query in enumerate(query_symbols):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing query {i + 1}/{len(query_symbols)}: {query}")

        # Get candidates (all symbols except query)
        candidates = [s for s in snapshot_df["symbol"].tolist() if s != query]

        # ============================================================
        # COMPUTE GROUND TRUTH REFERENCES (3 references)
        # ============================================================

        # Reference 1: SimilarityScore (composite of return/sector/size)
        similarity_scores, return_sim, sector_sim, size_sim = compute_similarity_score(
            query_symbol=query,
            returns_df=returns_120d,
            snapshot_df=snapshot_df,
            min_overlap=80,
        )
        similarity_scores = similarity_scores.reindex(candidates).fillna(0.0)
        return_sim = return_sim.reindex(candidates).fillna(0.0)
        sector_sim = sector_sim.reindex(candidates).fillna(0.0)
        size_sim = (
            size_sim.reindex(candidates).fillna(0.0)
            if len(size_sim) > 0
            else pd.Series(0.0, index=candidates)
        )

        # Reference 2: LiquidityUplift
        liquidity_scores_all = snapshot_df.set_index("symbol")["LiquidityScore"].reindex(
            candidates + [query]
        )
        liquidity_uplift = compute_liquidity_uplift(query, liquidity_scores_all)
        liquidity_uplift = liquidity_uplift.reindex(candidates).fillna(0.0)

        # Reference 3: UtilityScore
        utility_scores = compute_utility_score(similarity_scores, liquidity_uplift)
        utility_scores = utility_scores.reindex(candidates).fillna(0.0)

        # NEW References 4-8: Granular similarity scores
        # Reference 4: ReturnSimilarity (120-day Pearson)
        return_similarity = return_sim  # Already computed above

        # Reference 5: SectorSimilarity (GICS matching)
        sector_similarity = sector_sim  # Already computed above

        # Reference 6: FundamentalsSim (market cap similarity)
        from src.evaluation.ground_truth import compute_fundamentals_similarity

        fundamentals_sim = compute_fundamentals_similarity(
            query, snapshot_df, fund_columns=["market_cap"]
        )
        fundamentals_sim = fundamentals_sim.reindex(candidates).fillna(0.0)

        # Reference 7: LiquidityChar (liquidity percentile similarity)
        if "LiquidityScore" in snapshot_df.columns:
            liq_ranks = snapshot_df["LiquidityScore"].rank(pct=True)
            query_liq_rank = liq_ranks[snapshot_df["symbol"] == query].iloc[0]
            liq_char_sim = 1.0 - (liq_ranks - query_liq_rank).abs()
            liq_char_sim.index = snapshot_df["symbol"]
            liquidity_char = (
                liq_char_sim.drop(query, errors="ignore").reindex(candidates).fillna(0.0)
            )
        else:
            liquidity_char = pd.Series(0.0, index=candidates)

        # Reference 8: TurnoverChar (turnover percentile similarity)
        turnover_col = "turnover_rank" if "turnover_rank" in snapshot_df.columns else "turnover"
        if turnover_col in snapshot_df.columns:
            turn_ranks = snapshot_df[turnover_col].rank(pct=True)
            query_turn_rank = turn_ranks[snapshot_df["symbol"] == query].iloc[0]
            turn_char_sim = 1.0 - (turn_ranks - query_turn_rank).abs()
            turn_char_sim.index = snapshot_df["symbol"]
            turnover_char = (
                turn_char_sim.drop(query, errors="ignore").reindex(candidates).fillna(0.0)
            )
        else:
            turnover_char = pd.Series(0.0, index=candidates)

        # Store references in dict for iteration
        references = {
            "Similarity": similarity_scores,
            "LiquidityUplift": liquidity_uplift,
            "Utility": utility_scores,
            "ReturnSimilarity": return_similarity,
            "SectorSimilarity": sector_similarity,
            "FundamentalsSim": fundamentals_sim,
            "LiquidityChar": liquidity_char,
            "TurnoverChar": turnover_char,
        }

        # Build relevance labels for each reference
        relevance = {}
        for ref_name, ref_scores in references.items():
            if ref_name == "LiquidityUplift":
                # For uplift: relevance = positive uplift (actually more liquid than query)
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
                    graded[(ref_scores > 0) & (ref_scores <= p25_pos)] = 1
                    relevance[f"{ref_name}_graded"] = graded
                else:
                    relevance[f"{ref_name}_graded"] = pd.Series(0, index=ref_scores.index)
            else:
                relevance[f"{ref_name}_binary"] = build_binary_relevance(
                    ref_scores, top_percentile=0.25
                )
                relevance[f"{ref_name}_graded"] = build_graded_relevance(ref_scores)

        # ============================================================
        # COMPUTE RANKER SCORES (6 rankers)
        # ============================================================

        # Ranker 1: Embedding scores
        emb_scores = embedding_scores_dict.get(query, pd.Series(dtype=float))
        emb_scores = emb_scores.reindex(candidates).fillna(0.5)

        # Ranker 2: Pearson correlation scores (60-day)
        pearson_scores = build_correlation_scores_for_query(
            returns_wide=returns,
            query_symbol=query,
            snapshot_date=snapshot_date,
            lookback=60,
            min_overlap=40,
        )
        pearson_scores = pearson_scores.reindex(candidates).fillna(0.0)

        # Ranker 3: Spearman correlation scores (60-day)
        spearman_scores = build_spearman_scores_for_query(
            returns_wide=returns,
            query_symbol=query,
            snapshot_date=snapshot_date,
            lookback=60,
            min_overlap=40,
        )
        spearman_scores = spearman_scores.reindex(candidates).fillna(0.0)

        # Ranker 4: Embedding rerank (top-50 embedding shortlist, reranked by LiquidityScore)
        top50_emb = emb_scores.sort_values(ascending=False).head(50).index.tolist()
        emb_rerank_scores = pd.Series(np.nan, index=candidates)
        if len(top50_emb) > 0:
            # Get LiquidityScore for the shortlist and rerank
            liq_for_rerank = liquidity_scores_all.reindex(top50_emb).fillna(0.5)
            # Create scores: higher liquidity = better = higher score
            # Normalize to 0-1 range for consistency
            liq_min, liq_max = liq_for_rerank.min(), liq_for_rerank.max()
            if liq_max > liq_min:
                liq_normalized = (liq_for_rerank - liq_min) / (liq_max - liq_min)
            else:
                liq_normalized = pd.Series(0.5, index=top50_emb)
            emb_rerank_scores.loc[top50_emb] = liq_normalized

        # Ranker 5: Pearson rerank (top-50 Pearson shortlist, reranked by LiquidityScore)
        top50_pearson = pearson_scores.sort_values(ascending=False).head(50).index.tolist()
        pearson_rerank_scores = pd.Series(np.nan, index=candidates)
        if len(top50_pearson) > 0:
            liq_for_rerank = liquidity_scores_all.reindex(top50_pearson).fillna(0.5)
            liq_min, liq_max = liq_for_rerank.min(), liq_for_rerank.max()
            if liq_max > liq_min:
                liq_normalized = (liq_for_rerank - liq_min) / (liq_max - liq_min)
            else:
                liq_normalized = pd.Series(0.5, index=top50_pearson)
            pearson_rerank_scores.loc[top50_pearson] = liq_normalized

        # Ranker 6: Spearman rerank (top-50 Spearman shortlist, reranked by LiquidityScore)
        top50_spearman = spearman_scores.sort_values(ascending=False).head(50).index.tolist()
        spearman_rerank_scores = pd.Series(np.nan, index=candidates)
        if len(top50_spearman) > 0:
            liq_for_rerank = liquidity_scores_all.reindex(top50_spearman).fillna(0.5)
            liq_min, liq_max = liq_for_rerank.min(), liq_for_rerank.max()
            if liq_max > liq_min:
                liq_normalized = (liq_for_rerank - liq_min) / (liq_max - liq_min)
            else:
                liq_normalized = pd.Series(0.5, index=top50_spearman)
            spearman_rerank_scores.loc[top50_spearman] = liq_normalized

        # Store rankers in dict
        rankers = {
            "embedding": emb_scores,
            "pearson_corr": pearson_scores,
            "spearman_corr": spearman_scores,
            "embedding_rerank": emb_rerank_scores,
            "pearson_corr_rerank": pearson_rerank_scores,
            "spearman_corr_rerank": spearman_rerank_scores,
        }

        # ============================================================
        # COMPUTE METRICS (6 rankers × 3 references × 3 metrics = 54 values)
        # ============================================================

        query_result = {
            "query_symbol": query,
            "n_candidates": len(candidates),
            "snapshot_date": snapshot_date,
        }

        for ranker_name, ranker_scores in rankers.items():
            # Build full ranking list (handling NaN for rerankers)
            if ranker_scores.isna().any():
                # For rerankers: shortlist has reranked scores, remaining in original order
                non_nan_mask = ~ranker_scores.isna()
                shortlist_symbols = ranker_scores[non_nan_mask].index.tolist()

                # Build ranking: reranked shortlist first, then remaining by original ranker order
                if ranker_name == "embedding_rerank":
                    original_ranker = emb_scores
                elif ranker_name == "pearson_corr_rerank":
                    original_ranker = pearson_scores
                elif ranker_name == "spearman_corr_rerank":
                    original_ranker = spearman_scores
                else:
                    original_ranker = ranker_scores

                remaining = [
                    s
                    for s in original_ranker.sort_values(ascending=False).index
                    if s not in shortlist_symbols
                ]

                # Sort shortlist by rerank score
                shortlist_ranked = (
                    ranker_scores[non_nan_mask].sort_values(ascending=False).index.tolist()
                )
                ranking_list = shortlist_ranked + remaining
            else:
                # For base rankers: simple score-based ranking
                ranking_list = ranker_scores.sort_values(ascending=False).index.tolist()

            for ref_name in REFERENCES:
                binary_rel = relevance[f"{ref_name}_binary"]
                graded_rel = relevance[f"{ref_name}_graded"]
                ref_scores = references[ref_name]

                # Metric 1: Recall@10 (using binary relevance)
                recall = recall_at_k(ranking_list, binary_rel, k=10)
                query_result[f"{ranker_name}_{ref_name}_Recall@10"] = recall

                # Metric 2: nDCG@10 (using graded relevance)
                ndcg = ndcg_at_k(ranking_list, graded_rel, k=10)
                query_result[f"{ranker_name}_{ref_name}_nDCG@10"] = ndcg

                # Metric 3: Spearman correlation (ranker scores vs reference scores)
                # Use the actual scores, not the ranking positions
                valid_mask = ~ranker_scores.isna() & ~ref_scores.isna()
                if valid_mask.sum() >= 3:
                    spearman = spearman_against_reference(
                        ranker_scores[valid_mask],
                        ref_scores[valid_mask],
                    )
                else:
                    spearman = np.nan
                query_result[f"{ranker_name}_{ref_name}_Spearman"] = spearman

        all_results.append(query_result)

        # Query manifest (simplified for new structure)
        query_manifest.append(
            {
                "query_symbol": query,
                "n_candidates": len(candidates),
                "snapshot_date": snapshot_date,
            }
        )

        # Save per-query parquet with all ranker scores and reference scores
        per_query_data = {
            "query_symbol": query,
            "candidate_symbol": candidates,
            # Ranker scores
            "emb_score": emb_scores.values,
            "pearson_corr_score": pearson_scores.values,
            "spearman_corr_score": spearman_scores.values,
            "emb_rerank_score": emb_rerank_scores.reindex(candidates).fillna(np.nan).values,
            "pearson_rerank_score": pearson_rerank_scores.reindex(candidates).fillna(np.nan).values,
            "spearman_rerank_score": spearman_rerank_scores.reindex(candidates)
            .fillna(np.nan)
            .values,
            # Reference scores
            "SimilarityScore": similarity_scores.values,
            "LiquidityUplift": liquidity_uplift.values,
            "UtilityScore": utility_scores.values,
            # Component scores for breakdown analysis
            "return_similarity": return_sim.values,
            "sector_similarity": sector_sim.values,
            "size_similarity": size_sim.values,
            # Relevance labels for Similarity reference
            "Similarity_binary": relevance["Similarity_binary"].values,
            "Similarity_graded": relevance["Similarity_graded"].values,
            # Relevance labels for LiquidityUplift reference
            "LiquidityUplift_binary": relevance["LiquidityUplift_binary"].values,
            "LiquidityUplift_graded": relevance["LiquidityUplift_graded"].values,
            # Relevance labels for Utility reference
            "Utility_binary": relevance["Utility_binary"].values,
            "Utility_graded": relevance["Utility_graded"].values,
        }

        per_query_df = pd.DataFrame(per_query_data)
        per_query_df.to_parquet(
            output_dir / "retrieval" / "per_query" / f"{query}.parquet",
            index=False,
        )

    print("-" * 70)
    print()

    # Aggregate results
    results_df = pd.DataFrame(all_results)

    # ============================================================
    # GENERATE PER-REFERENCE CSV FILES (3 references × 3 metrics × 6 rankers)
    # ============================================================

    # Create per-reference metrics DataFrames
    # Each CSV has structure:
    #   Rows: Recall@10, nDCG@10, Spearman
    #   Columns: metric_name, embedding, pearson_corr, spearman_corr,
    #            embedding_rerank, pearson_corr_rerank, spearman_corr_rerank

    def build_reference_metrics_df(reference_name: str) -> pd.DataFrame:
        """Build metrics DataFrame for a specific reference."""
        return pd.DataFrame(
            {
                "metric_name": ["Recall@10", "nDCG@10", "Spearman"],
                # Embedding ranker
                "embedding": [
                    results_df[f"embedding_{reference_name}_Recall@10"].mean(),
                    results_df[f"embedding_{reference_name}_nDCG@10"].mean(),
                    results_df[f"embedding_{reference_name}_Spearman"].mean(),
                ],
                # Pearson correlation ranker
                "pearson_corr": [
                    results_df[f"pearson_corr_{reference_name}_Recall@10"].mean(),
                    results_df[f"pearson_corr_{reference_name}_nDCG@10"].mean(),
                    results_df[f"pearson_corr_{reference_name}_Spearman"].mean(),
                ],
                # Spearman correlation ranker
                "spearman_corr": [
                    results_df[f"spearman_corr_{reference_name}_Recall@10"].mean(),
                    results_df[f"spearman_corr_{reference_name}_nDCG@10"].mean(),
                    results_df[f"spearman_corr_{reference_name}_Spearman"].mean(),
                ],
                # Embedding rerank
                "embedding_rerank": [
                    results_df[f"embedding_rerank_{reference_name}_Recall@10"].mean(),
                    results_df[f"embedding_rerank_{reference_name}_nDCG@10"].mean(),
                    results_df[f"embedding_rerank_{reference_name}_Spearman"].mean(),
                ],
                # Pearson rerank
                "pearson_corr_rerank": [
                    results_df[f"pearson_corr_rerank_{reference_name}_Recall@10"].mean(),
                    results_df[f"pearson_corr_rerank_{reference_name}_nDCG@10"].mean(),
                    results_df[f"pearson_corr_rerank_{reference_name}_Spearman"].mean(),
                ],
                # Spearman rerank
                "spearman_corr_rerank": [
                    results_df[f"spearman_corr_rerank_{reference_name}_Recall@10"].mean(),
                    results_df[f"spearman_corr_rerank_{reference_name}_nDCG@10"].mean(),
                    results_df[f"spearman_corr_rerank_{reference_name}_Spearman"].mean(),
                ],
            }
        )

    # Generate CSV for Similarity reference
    similarity_metrics = build_reference_metrics_df("Similarity")
    similarity_metrics.to_csv(output_dir / "metrics" / "retrieval_similarity.csv", index=False)
    print(f"Saved: {output_dir / 'metrics' / 'retrieval_similarity.csv'}")

    # Generate CSV for LiquidityUplift reference
    liquidity_metrics = build_reference_metrics_df("LiquidityUplift")
    liquidity_metrics.to_csv(output_dir / "metrics" / "retrieval_liquidity_uplift.csv", index=False)
    print(f"Saved: {output_dir / 'metrics' / 'retrieval_liquidity_uplift.csv'}")

    # Generate CSV for Utility reference
    utility_metrics = build_reference_metrics_df("Utility")
    utility_metrics.to_csv(output_dir / "metrics" / "retrieval_utility.csv", index=False)
    print(f"Saved: {output_dir / 'metrics' / 'retrieval_utility.csv'}")

    # Backward compatibility: recall_spearman.csv (same as similarity reference)
    similarity_metrics.to_csv(output_dir / "metrics" / "recall_spearman.csv", index=False)
    print(f"Saved: {output_dir / 'metrics' / 'recall_spearman.csv'} (backward compatible)\n")

    # ============================================================
    # GENERATE OVERALL CSV (averaged across all 3 references)
    # ============================================================

    # Compute overall metrics for the 6 rankers × 3 references × 3 metrics structure
    # We'll create a summary table with average metrics across all references for each ranker
    overall_metrics = pd.DataFrame(
        {
            "metric_name": ["Recall@10", "nDCG@10", "Spearman"],
            # Embedding ranker - average across all 3 references
            "embedding": [
                np.mean([results_df[f"embedding_{ref}_Recall@10"].mean() for ref in REFERENCES]),
                np.mean([results_df[f"embedding_{ref}_nDCG@10"].mean() for ref in REFERENCES]),
                np.mean([results_df[f"embedding_{ref}_Spearman"].mean() for ref in REFERENCES]),
            ],
            # Pearson correlation ranker
            "pearson_corr": [
                np.mean([results_df[f"pearson_corr_{ref}_Recall@10"].mean() for ref in REFERENCES]),
                np.mean([results_df[f"pearson_corr_{ref}_nDCG@10"].mean() for ref in REFERENCES]),
                np.mean([results_df[f"pearson_corr_{ref}_Spearman"].mean() for ref in REFERENCES]),
            ],
            # Spearman correlation ranker
            "spearman_corr": [
                np.mean(
                    [results_df[f"spearman_corr_{ref}_Recall@10"].mean() for ref in REFERENCES]
                ),
                np.mean([results_df[f"spearman_corr_{ref}_nDCG@10"].mean() for ref in REFERENCES]),
                np.mean([results_df[f"spearman_corr_{ref}_Spearman"].mean() for ref in REFERENCES]),
            ],
            # Embedding rerank
            "embedding_rerank": [
                np.mean(
                    [results_df[f"embedding_rerank_{ref}_Recall@10"].mean() for ref in REFERENCES]
                ),
                np.mean(
                    [results_df[f"embedding_rerank_{ref}_nDCG@10"].mean() for ref in REFERENCES]
                ),
                np.mean(
                    [results_df[f"embedding_rerank_{ref}_Spearman"].mean() for ref in REFERENCES]
                ),
            ],
            # Pearson rerank
            "pearson_corr_rerank": [
                np.mean(
                    [results_df[f"pearson_corr_rerank_{ref}_Spearman"].mean() for ref in REFERENCES]
                ),
                np.mean(
                    [results_df[f"pearson_corr_rerank_{ref}_nDCG@10"].mean() for ref in REFERENCES]
                ),
                np.mean(
                    [
                        results_df[f"spearman_corr_rerank_{ref}_Spearman"].mean()
                        for ref in REFERENCES
                    ]
                ),
            ],
            # Spearman rerank
            "spearman_corr_rerank": [
                np.mean(
                    [results_df[f"pearson_corr_rerank_{ref}_Spearman"].mean() for ref in REFERENCES]
                ),
                np.mean(
                    [results_df[f"spearman_corr_rerank_{ref}_nDCG@10"].mean() for ref in REFERENCES]
                ),
                np.mean(
                    [
                        results_df[f"spearman_corr_rerank_{ref}_Spearman"].mean()
                        for ref in REFERENCES
                    ]
                ),
            ],
        }
    )

    # Save overall metrics
    overall_metrics.to_csv(output_dir / "metrics" / "retrieval_metrics_overall.csv", index=False)
    print(f"Saved: {output_dir / 'metrics' / 'retrieval_metrics_overall.csv'}\n")

    # Save detailed 6×3 metrics table
    detailed_metrics_path = output_dir / "metrics" / "retrieval_metrics_6x3_detailed.csv"
    results_df.to_csv(detailed_metrics_path, index=False)
    print(f"Saved: {detailed_metrics_path} ({len(results_df)} queries × 54 metric columns)\n")

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
                high_return_sim = (
                    top10["return_similarity"] > top10["return_similarity"].quantile(0.75)
                ).sum()
                high_sector_sim = (top10["sector_similarity"] >= 0.5).sum()
                positive_liq_uplift = (top10["LiquidityUplift"] > 0).sum()

                breakdown_results.append(
                    {
                        "query": query,
                        "ranker": ranker_name,
                        "top10_size": len(top10),
                        "high_return_sim": high_return_sim,
                        "high_sector_sim": high_sector_sim,
                        "positive_liq_uplift": positive_liq_uplift,
                        "pct_return_sim": high_return_sim / len(top10) * 100,
                        "pct_sector_sim": high_sector_sim / len(top10) * 100,
                        "pct_liq_improve": positive_liq_uplift / len(top10) * 100,
                    }
                )
        except Exception as e:
            print(f"  Warning: Could not process breakdown for {query}: {e}")

    if breakdown_results:
        breakdown_df = pd.DataFrame(breakdown_results)
        breakdown_path = output_dir / "metrics" / "utility_breakdown_analysis.csv"
        breakdown_df.to_csv(breakdown_path, index=False)
        print(f"Saved: {breakdown_path}")

        # Also save average breakdown per ranker
        avg_breakdown = (
            breakdown_df.groupby("ranker")[["pct_return_sim", "pct_sector_sim", "pct_liq_improve"]]
            .mean()
            .round(2)
        )
        avg_path = output_dir / "metrics" / "utility_breakdown_averaged.csv"
        avg_breakdown.to_csv(avg_path)
        print(f"Saved: {avg_path}")

    # Compute metrics by sector (using average across all 3 references for each ranker)
    if "gsector" in snapshot_df.columns and "gsector" in query_df.columns:
        # Drop duplicate gsector from query_df to avoid _x/_y suffix after merge
        query_sectors = query_df.drop(columns=["gsector"]).merge(
            snapshot_df[["symbol", "gsector"]], on="symbol"
        )
        sector_metrics = []

        for sector in query_sectors["gsector"].unique():
            if pd.isna(sector):
                continue
            sector_queries = query_sectors[query_sectors["gsector"] == sector]["symbol"].tolist()
            sector_results = results_df[results_df["query_symbol"].isin(sector_queries)]

            if len(sector_results) == 0:
                continue

            row = {"group_name": sector}
            # For each ranker, average metrics across all 3 references
            for ranker in RANKERS:
                avg_metric = np.mean(
                    [
                        np.mean(
                            [
                                results_df[f"pearson_corr_rerank_{ref}_Spearman"].mean()
                                for ref in REFERENCES
                            ]
                        ),
                        np.mean(
                            [sector_results[f"{ranker}_{ref}_nDCG@10"].mean() for ref in REFERENCES]
                        ),
                    ]
                )
                row[ranker] = avg_metric

            sector_metrics.append(row)

        if sector_metrics:
            sector_df = pd.DataFrame(sector_metrics)
            sector_df.to_csv(
                output_dir / "metrics" / "retrieval_metrics_by_sector.csv", index=False
            )
            print(f"Saved: {output_dir / 'metrics' / 'retrieval_metrics_by_sector.csv'}\n")

    # Compute metrics by market cap tier
    mcap_metrics: list[dict] = []
    if "market_cap_tier" in snapshot_df.columns and "market_cap_tier" in query_df.columns:
        # Drop duplicate market_cap_tier from query_df to avoid _x/_y suffix after merge
        query_mcap = query_df.drop(columns=["market_cap_tier"]).merge(
            snapshot_df[["symbol", "market_cap_tier"]], on="symbol"
        )
        for tier in query_mcap["market_cap_tier"].unique():
            if pd.isna(tier):
                continue
            tier_queries = query_mcap[query_mcap["market_cap_tier"] == tier]["symbol"].tolist()
            tier_results = results_df[results_df["query_symbol"].isin(tier_queries)]

            if len(tier_results) == 0:
                continue

            row = {"group_name": tier}
            # For each ranker, average metrics across all 3 references
            for ranker in RANKERS:
                avg_metric = np.mean(
                    [
                        np.mean(
                            [tier_results[f"{ranker}_{ref}_Recall@10"].mean() for ref in REFERENCES]
                        ),
                        np.mean(
                            [tier_results[f"{ranker}_{ref}_nDCG@10"].mean() for ref in REFERENCES]
                        ),
                    ]
                )
                row[ranker] = avg_metric

            mcap_metrics.append(row)

    if mcap_metrics:
        mcap_df = pd.DataFrame(mcap_metrics)
        mcap_df.to_csv(output_dir / "metrics" / "retrieval_metrics_by_market_cap.csv", index=False)
        print(f"Saved: {output_dir / 'metrics' / 'retrieval_metrics_by_market_cap.csv'}\n")

    # Compute metrics by liquidity quartile
    # Drop duplicate liquidity_quartile from query_df to avoid _x/_y suffix after merge
    query_liq = query_df.drop(columns=["liquidity_quartile"]).merge(
        snapshot_df[["symbol", "liquidity_quartile"]], on="symbol"
    )
    liq_metrics = []

    for quartile in sorted(query_liq["liquidity_quartile"].unique()):
        quartile_queries = query_liq[query_liq["liquidity_quartile"] == quartile]["symbol"].tolist()
        quartile_results = results_df[results_df["query_symbol"].isin(quartile_queries)]

        if len(quartile_results) == 0:
            continue

        row = {"group_name": int(quartile)}
        # For each ranker, average metrics across all 3 references
        for ranker in RANKERS:
            avg_metric = np.mean(
                [
                    np.mean(
                        [quartile_results[f"{ranker}_{ref}_Recall@10"].mean() for ref in REFERENCES]
                    ),
                    np.mean(
                        [quartile_results[f"{ranker}_{ref}_nDCG@10"].mean() for ref in REFERENCES]
                    ),
                ]
            )
            row[ranker] = avg_metric

        liq_metrics.append(row)

    if liq_metrics:
        liq_df = pd.DataFrame(liq_metrics)
        liq_df.to_csv(
            output_dir / "metrics" / "retrieval_metrics_by_liquidity_quartile.csv", index=False
        )
        print(f"Saved: {output_dir / 'metrics' / 'retrieval_metrics_by_liquidity_quartile.csv'}\n")

    # Save query manifest
    manifest_df = pd.DataFrame(query_manifest)
    manifest_df.to_csv(output_dir / "retrieval" / "query_manifest.csv", index=False)
    print(f"Saved: {output_dir / 'retrieval' / 'query_manifest.csv'}\n")

    # Generate plots
    print("Generating visualization figures...")

    # Create per-reference overall metrics for 3 separate plots
    for ref_name in REFERENCES:
        ref_metrics_df = pd.DataFrame(
            {
                "metric_name": ["Recall@10", "Spearman", "nDCG@10"],
                "embedding": [
                    results_df[f"embedding_{ref_name}_Recall@10"].mean(),
                    results_df[f"embedding_{ref_name}_Spearman"].mean(),
                    results_df[f"embedding_{ref_name}_nDCG@10"].mean(),
                ],
                "pearson_corr": [
                    results_df[f"pearson_corr_{ref_name}_Recall@10"].mean(),
                    results_df[f"pearson_corr_{ref_name}_Spearman"].mean(),
                    results_df[f"pearson_corr_{ref_name}_nDCG@10"].mean(),
                ],
                "spearman_corr": [
                    results_df[f"spearman_corr_{ref_name}_Recall@10"].mean(),
                    results_df[f"spearman_corr_{ref_name}_Spearman"].mean(),
                    results_df[f"spearman_corr_{ref_name}_nDCG@10"].mean(),
                ],
                "embedding_rerank": [
                    results_df[f"embedding_rerank_{ref_name}_Recall@10"].mean(),
                    results_df[f"embedding_rerank_{ref_name}_Spearman"].mean(),
                    results_df[f"embedding_rerank_{ref_name}_nDCG@10"].mean(),
                ],
                "pearson_corr_rerank": [
                    results_df[f"pearson_corr_rerank_{ref_name}_Recall@10"].mean(),
                    results_df[f"pearson_corr_rerank_{ref_name}_Spearman"].mean(),
                    results_df[f"pearson_corr_rerank_{ref_name}_nDCG@10"].mean(),
                ],
                "spearman_corr_rerank": [
                    results_df[f"spearman_corr_rerank_{ref_name}_Recall@10"].mean(),
                    results_df[f"spearman_corr_rerank_{ref_name}_Spearman"].mean(),
                    results_df[f"spearman_corr_rerank_{ref_name}_nDCG@10"].mean(),
                ],
            }
        )

        # Plot for this reference
        ref_filename = ref_name.lower().replace(" ", "_")
        plot_overall_metrics(
            metrics_df=ref_metrics_df,
            output_path=output_dir
            / "retrieval"
            / "figures"
            / f"metrics_{ref_filename}_comparison.png",
        )
        print(f"  Generated: metrics_{ref_filename}_comparison.png ({ref_name} reference)")

    # Legacy overall metrics plot (averaged across all references)
    plot_overall_metrics(
        metrics_df=overall_metrics,
        output_path=output_dir / "retrieval" / "figures" / "metrics_overall_comparison.png",
    )
    print("  Generated: metrics_overall_comparison.png (averaged across references)")

    # Grouped metrics plots
    if sector_metrics:
        sector_fig_df = pd.DataFrame(sector_metrics)
        # Only plot if we have the expected columns
        if len(sector_fig_df) > 0 and "embedding" in sector_fig_df.columns:
            plot_grouped_metrics(
                grouped_metrics_df=sector_fig_df,
                metric_col="embedding",
                group_col="sector",
                output_path=output_dir / "retrieval" / "figures" / "metrics_by_sector.png",
            )

    if mcap_metrics:
        mcap_fig_df = pd.DataFrame(mcap_metrics)
        if len(mcap_fig_df) > 0 and "embedding" in mcap_fig_df.columns:
            plot_grouped_metrics(
                grouped_metrics_df=mcap_fig_df,
                metric_col="embedding",
                group_col="market_cap_tier",
                output_path=output_dir / "retrieval" / "figures" / "metrics_by_market_cap.png",
            )

    if liq_metrics:
        liq_fig_df = pd.DataFrame(liq_metrics)
        if len(liq_fig_df) > 0 and "embedding" in liq_fig_df.columns:
            plot_grouped_metrics(
                grouped_metrics_df=liq_fig_df,
                metric_col="embedding",
                group_col="liquidity_quartile",
                output_path=output_dir / "retrieval" / "figures" / "metrics_by_liquidity.png",
            )

    print()

    # Print summary
    print("=" * 70)
    print("RETRIEVAL METRICS SUMMARY")
    print("=" * 70)
    print("\nOverall Metrics:")
    print(overall_metrics.to_string(index=False))
    print(f"\nResults saved to: {output_dir}")
    print("  - metrics/: Overall and grouped metrics CSVs")
    print("  - retrieval/query_manifest.csv: Query list")
    print("  - retrieval/per_query/: Per-query detailed results")
    print("  - retrieval/figures/: Visualization plots")

    return {
        "overall_metrics": overall_metrics,
        "results_df": results_df,
        "manifest_df": manifest_df,
    }


def main():
    """Main entry point."""
    args = parse_args()

    run_evaluation_pipeline(
        features_path=args.features,
        checkpoint_path=args.checkpoint,
        period_start=args.period_start,
        period_end=args.period_end,
        n_queries=args.n_queries,
        seed=args.seed,
        output_dir=args.output_dir,
        run_hybrid=args.run_hybrid,
    )


if __name__ == "__main__":
    main()
