"""Hybrid caching system for computed embeddings.

Caches embeddings to parquet files for standard periods.
Custom periods are computed on-demand without caching.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

import pandas as pd


class EmbeddingCache:
    """Cache manager for computed embeddings.

    Standard periods (defined in config) are cached to parquet files.
    Custom periods are computed on-demand without caching.

    Cache key: {period_name}_{model_hash}.parquet
    """

    # Standard periods that should be cached
    STANDARD_PERIODS = {
        "covid_pre",
        "covid_crisis",
        "ratehike_pre",
        "ratehike_crisis",
        "normal_2019",
        "normal_2021",
    }

    def __init__(self, cache_dir: str | Path):
        """Initialize cache with directory path.

        Args:
            cache_dir: Directory to store cached embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, period: str, model_id: str) -> Path:
        """Generate cache file path for period and model.

        Args:
            period: Period identifier (e.g., "covid_pre")
            model_id: Model identifier or hash

        Returns:
            Path to cache file
        """
        filename = f"{period}_{model_id}.parquet"
        return self.cache_dir / filename

    def save(
        self,
        embeddings_df: pd.DataFrame,
        period: str,
        model_id: str,
    ) -> Path:
        """Save embeddings to cache.

        Args:
            embeddings_df: DataFrame with embeddings and metadata
            period: Period identifier
            model_id: Model identifier

        Returns:
            Path to saved cache file
        """
        cache_path = self._get_cache_path(period, model_id)
        embeddings_df.to_parquet(cache_path, index=False)
        print(f"Cached embeddings to {cache_path}")
        return cache_path

    def load(
        self,
        period: str,
        model_id: str,
    ) -> Optional[pd.DataFrame]:
        """Load embeddings from cache if available.

        Args:
            period: Period identifier
            model_id: Model identifier

        Returns:
            DataFrame if cache hit, None if cache miss
        """
        cache_path = self._get_cache_path(period, model_id)

        if cache_path.exists():
            print(f"Loading cached embeddings from {cache_path}")
            return pd.read_parquet(cache_path)

        return None

    def is_cached(self, period: str, model_id: str) -> bool:
        """Check if embeddings are cached.

        Args:
            period: Period identifier
            model_id: Model identifier

        Returns:
            True if cache exists
        """
        cache_path = self._get_cache_path(period, model_id)
        return cache_path.exists()

    def should_cache(self, period: str) -> bool:
        """Determine if this period should be cached.

        Only standard periods defined in config are cached.
        Custom periods are computed on-demand.

        Args:
            period: Period identifier

        Returns:
            True if period should be cached
        """
        return period in self.STANDARD_PERIODS

    def clear(self, period: Optional[str] = None) -> None:
        """Clear cache for period or all cache.

        Args:
            period: Specific period to clear, or None for all
        """
        if period is None:
            for f in self.cache_dir.glob("*.parquet"):
                f.unlink()
            print("Cleared all cache files")
        else:
            for f in self.cache_dir.glob(f"{period}_*.parquet"):
                f.unlink()
            print(f"Cleared cache for period: {period}")

    def list_cached(self) -> list[str]:
        """List all cached periods.

        Returns:
            List of cached period identifiers
        """
        periods = set()
        for f in self.cache_dir.glob("*.parquet"):
            period = f.stem.rsplit("_", 1)[0]
            periods.add(period)
        return sorted(list(periods))
