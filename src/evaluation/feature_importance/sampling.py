"""Query and background sampling strategies for SHAP analysis."""

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class QuerySampler:
    """Select representative query stocks for SHAP analysis."""

    @staticmethod
    def stratified_by_sector_and_liquidity(
        df: pd.DataFrame,
        n_queries: int = 50,
    ) -> pd.DataFrame:
        """
        Sample queries stratified by sector and liquidity tier.

        Strategy:
        - 20 large-cap liquid (top 2 by market cap per sector)
        - 20 mid-cap mixed liquidity (middle 2 per sector)
        - 10 small-cap illiquid (bottom 1 per sector)

        Args:
            df: DataFrame with sector and market_cap columns
            n_queries: Total number of queries

        Returns:
            DataFrame of query stocks
        """
        df = df.copy()

        if "gsector" not in df.columns:
            raise ValueError("DataFrame must have 'gsector' column")

        if "market_cap" not in df.columns:
            raise ValueError("DataFrame must have 'market_cap' column")

        df = df.dropna(subset=["gsector", "market_cap"])

        df["market_cap_tier"] = pd.qcut(
            df["market_cap"].rank(method="first"),
            q=3,
            labels=["small", "mid", "large"],
        )

        selected_indices = []

        sectors = df["gsector"].unique()
        n_sectors = len(sectors)

        large_per_sector = max(1, 20 // n_sectors)
        mid_per_sector = max(1, 20 // n_sectors)
        small_per_sector = max(1, 10 // n_sectors)

        for sector in sectors:
            sector_df = df[df["gsector"] == sector].copy()

            if len(sector_df) == 0:
                continue

            sector_df = sector_df.sort_values("market_cap", ascending=False)

            large_stocks = sector_df[sector_df["market_cap_tier"] == "large"]
            if len(large_stocks) > 0:
                n_select = min(large_per_sector, len(large_stocks))
                selected_indices.extend(large_stocks.head(n_select).index.tolist())

            mid_stocks = sector_df[sector_df["market_cap_tier"] == "mid"]
            if len(mid_stocks) > 0:
                n_select = min(mid_per_sector, len(mid_stocks))
                selected_indices.extend(mid_stocks.head(n_select).index.tolist())

            small_stocks = sector_df[sector_df["market_cap_tier"] == "small"]
            if len(small_stocks) > 0:
                n_select = min(small_per_sector, len(small_stocks))
                selected_indices.extend(small_stocks.head(n_select).index.tolist())

        selected_indices = list(set(selected_indices))

        if len(selected_indices) < n_queries:
            remaining = n_queries - len(selected_indices)
            available = df[~df.index.isin(selected_indices)]
            if len(available) > 0:
                additional = available.sample(n=remaining, random_state=42)
                selected_indices.extend(additional.index.tolist())
        elif len(selected_indices) > n_queries:
            selected_indices = list(
                pd.Index(selected_indices).to_series().sample(n=n_queries, random_state=42).index
            )

        return df.loc[selected_indices].reset_index(drop=True)

    @staticmethod
    def random(df: pd.DataFrame, n_queries: int = 50) -> pd.DataFrame:
        """
        Simple random sampling of queries.

        Args:
            df: DataFrame with stock features
            n_queries: Number of queries to sample

        Returns:
            DataFrame of query stocks
        """
        return df.sample(n=n_queries, random_state=42).reset_index(drop=True)

    @staticmethod
    def by_market_cap_tiers(
        df: pd.DataFrame,
        n_queries: int = 50,
    ) -> pd.DataFrame:
        """
        Sample queries evenly across market cap tiers.

        Args:
            df: DataFrame with market_cap column
            n_queries: Total number of queries

        Returns:
            DataFrame of query stocks
        """
        df = df.copy()

        if "market_cap" not in df.columns:
            raise ValueError("DataFrame must have 'market_cap' column")

        df = df.dropna(subset=["market_cap"])

        df["market_cap_tier"] = pd.qcut(
            df["market_cap"].rank(method="first"),
            q=4,
            labels=["Q1", "Q2", "Q3", "Q4"],
        )

        selected_indices = []
        n_per_tier = n_queries // 4

        for tier in ["Q1", "Q2", "Q3", "Q4"]:
            tier_df = df[df["market_cap_tier"] == tier]
            n_select = min(n_per_tier, len(tier_df))
            if n_select > 0:
                selected_indices.extend(tier_df.sample(n=n_select, random_state=42).index.tolist())

        if len(selected_indices) < n_queries:
            remaining = n_queries - len(selected_indices)
            available = df[~df.index.isin(selected_indices)]
            if len(available) > 0:
                additional = available.sample(n=remaining, random_state=42)
                selected_indices.extend(additional.index.tolist())

        return df.loc[selected_indices].reset_index(drop=True)


def select_background_samples(
    df: pd.DataFrame,
    n_samples: int = 100,
    method: Literal["random", "kmeans"] = "kmeans",
) -> pd.DataFrame:
    """
    Select background samples for SHAP baseline.

    Args:
        df: Full dataset
        n_samples: Number of background samples
        method: 'random' for random sampling, 'kmeans' for k-means centroids

    Returns:
        DataFrame with background samples

    Notes:
        - K-means method selects samples closest to cluster centroids
        - This provides better coverage of feature space than random
    """
    feature_cols = [
        "market_cap",
        "beta",
        "idiosyncratic_vol",
        "roe",
        "roa",
        "debt_to_equity",
        "price_to_book",
        "price_to_earnings",
        "operating_margin",
        "profit_margin",
    ]

    available_cols = [c for c in feature_cols if c in df.columns]

    if not available_cols:
        return df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    features = df[available_cols].copy()
    features = features.fillna(0)

    if method == "random":
        return df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    elif method == "kmeans":
        n_clusters = min(n_samples, len(df))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        features_scaled = (features - features.mean()) / (features.std() + 1e-8)

        kmeans.fit(features_scaled)

        distances = []
        for i in range(len(df)):
            dist = np.linalg.norm(
                features_scaled.iloc[i].values - kmeans.cluster_centers_[kmeans.labels_[i]]
            )
            distances.append((i, dist))

        distances.sort(key=lambda x: x[1])

        selected_indices = [idx for idx, _ in distances[:n_samples]]

        return df.iloc[selected_indices].reset_index(drop=True)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'random' or 'kmeans'.")
