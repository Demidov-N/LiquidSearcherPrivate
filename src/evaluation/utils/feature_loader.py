"""Feature loader for evaluation pipeline.

Reads from data/processed/all_features.parquet which contains
all symbols and dates in a single file.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


class FeatureLoader:
    """Load features from all_features.parquet for specified periods."""

    def __init__(self, feature_path: str | Path):
        """Initialize with path to all_features.parquet.

        Args:
            feature_path: Path to all_features.parquet file
        """
        self.feature_path = Path(feature_path)

        if not self.feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {self.feature_path}")

        self._data: Optional[pd.DataFrame] = None
        self.feature_dir = self.feature_path.parent

    @property
    def data(self) -> pd.DataFrame:
        """Lazy-load the full dataset."""
        if self._data is None:
            print(f"Loading features from {self.feature_path}...")
            self._data = pd.read_parquet(self.feature_path)
            self._data["date"] = pd.to_datetime(self._data["date"])
            print(f"Loaded {len(self._data):,} rows, {self._data['symbol'].nunique()} symbols")
        return self._data

    def load_period(
        self,
        period_start: str,
        period_end: str,
        symbols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Load features for specified period.

        Args:
            period_start: Start date (YYYY-MM-DD)
            period_end: End date (YYYY-MM-DD)
            symbols: Optional list of symbols to filter (default: all)

        Returns:
            DataFrame with features for period
        """
        df = self.data

        period_min = pd.Timestamp(period_start)
        period_max = pd.Timestamp(period_end)

        mask = (df["date"] >= period_min) & (df["date"] <= period_max)
        df = df[mask].copy()

        if symbols is not None:
            df = df[df["symbol"].isin(symbols)]

        print(
            f"Period {period_start} to {period_end}: {len(df):,} rows, {df['symbol'].nunique()} symbols"
        )

        return df

    def get_available_date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Get the available date range in the dataset.

        Returns:
            Tuple of (min_date, max_date)
        """
        df = self.data
        return df["date"].min(), df["date"].max()

    def get_available_symbols(self) -> list[str]:
        """Get list of available symbols.

        Returns:
            List of symbol names
        """
        return sorted(self.data["symbol"].unique().tolist())
