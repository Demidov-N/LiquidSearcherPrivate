# WRDS Data Loading System Implementation Plan (Updated)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Build a single unified WRDS data loader that fetches CRSP daily prices and Compustat fundamentals (with GICS codes), with intelligent parquet-based caching and a simple API.

**Architecture:** Single unified `WRDSDataLoader` class handles both CRSP and Compustat data internally. Uses parquet-based caching (readable in notebooks via `pd.read_parquet()`). Internal CCM (CRSP/Compustat Merged) linking handles the PERMNO/GVKEY mapping automatically when loading merged data.

**Tech Stack:** Python 3.12, wrds library, pandas (primary), parquet (via pyarrow), pytest, pathlib

**Key Design Decisions:**
- ✅ **Single unified loader** (not separate CRSP/Compustat loaders) - simpler API
- ✅ **Parquet caching** (not pickle or CSV) - preserves types, faster, smaller
- ✅ **Notebook-friendly** - cache files readable via `pd.read_parquet()`
- ✅ **Internal CCM linking** - users just call `load_merged()`, linking happens automatically

**Note:** This is a **PROTOTYPE** - prioritize functionality over perfection. Clean up later.

---

## Task 1: Update Cache Manager to Use Parquet

**Goal:** Refactor existing `CacheManager` to use parquet format instead of pickle.

**Files:**
- Modify: `src/data/cache_manager.py`
- Test: `tests/test_cache_manager.py`

**Step 1: Update the existing cache manager implementation**

Replace the current pickle-based `CacheManager` with a parquet-based one:

```python
# src/data/cache_manager.py
"""Data caching with parquet format for fast persistence and notebook compatibility."""

import hashlib
import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import polars as pl

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Manage local data cache in parquet format (notebook-friendly)."""
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        compression: str = "snappy",
    ):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache files. Uses settings default if None.
            compression: Compression algorithm (snappy, zstd, lz4, gzip)
        """
        if cache_dir is None:
            settings = get_settings()
            cache_dir = settings.cache_dir
        
        self.cache_dir = Path(cache_dir)
        self.compression = compression
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> Path:
        """Generate safe file path for cache key."""
        safe_key = key.replace("/", "_").replace(":", "_").replace(" ", "_").replace("\\", "_")
        if len(safe_key) > 100:
            safe_key = safe_key[:50] + "_" + hashlib.md5(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{safe_key}.parquet"
    
    def get(self, key: str) -> Optional[Union[pd.DataFrame, pl.DataFrame]]:
        """Load DataFrame from cache if exists.
        
        Args:
            key: Cache key
            
        Returns:
            Cached DataFrame or None if not found
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        try:
            # Always return pandas DataFrame (primary format per AGENTS.md)
            return pd.read_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Failed to read cache file {cache_path}: {e}")
            return None
    
    def set(
        self,
        key: str,
        df: Union[pd.DataFrame, pl.DataFrame],
    ) -> None:
        """Save DataFrame to cache.
        
        Args:
            key: Cache key
            df: DataFrame to cache (pandas or polars)
        """
        cache_path = self._get_cache_path(key)
        
        try:
            if isinstance(df, pd.DataFrame):
                df.to_parquet(cache_path, compression=self.compression)
            elif isinstance(df, pl.DataFrame):
                # Convert to pandas for consistent format
                df.to_pandas().to_parquet(cache_path, compression=self.compression)
            else:
                raise TypeError(f"Unsupported DataFrame type: {type(df)}")
            logger.info(f"Cached data to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to write cache file {cache_path}: {e}")
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self._get_cache_path(key).exists()
    
    def delete(self, key: str) -> None:
        """Delete cached item."""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
    
    def clear(self) -> None:
        """Clear all cached data."""
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink(missing_ok=True)
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern.
        
        Args:
            pattern: Glob pattern for matching keys
            
        Returns:
            Number of keys invalidated
        """
        count = 0
        # Validate pattern to prevent path traversal
        if ".." in pattern or "/" in pattern:
            logger.warning(f"Invalid pattern contains path traversal characters: {pattern}")
            return 0
        for path in self.cache_dir.glob(f"{pattern}*.parquet"):
            path.unlink()
            count += 1
        return count
```

**Step 2: Update tests for parquet cache**

```python
# tests/test_cache_manager.py
"""Test cache manager with parquet format."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.cache_manager import CacheManager


def test_cache_save_and_load_parquet():
    """Test saving and loading DataFrame to cache as parquet."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=Path(tmpdir))
        
        # Create test DataFrame with various types
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=10),
            "value": range(10),
            "float_val": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
            "string_val": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        })
        
        # Save to cache
        cache_key = "test_data_2020"
        cache.set(cache_key, df)
        
        # Load from cache
        loaded_df = cache.get(cache_key)
        
        assert loaded_df is not None
        assert len(loaded_df) == 10
        assert list(loaded_df.columns) == ["date", "value", "float_val", "string_val"]
        # Verify parquet preserved types
        assert loaded_df["date"].dtype == "datetime64[ns]"
        assert loaded_df["value"].dtype == "int64"


def test_cache_miss_returns_none():
    """Test that missing cache key returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=Path(tmpdir))
        
        result = cache.get("nonexistent_key")
        assert result is None


def test_cache_exists_check():
    """Test cache exists method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=Path(tmpdir))
        
        df = pd.DataFrame({"a": [1, 2, 3]})
        cache.set("my_key", df)
        
        assert cache.exists("my_key") is True
        assert cache.exists("other_key") is False


def test_cache_notebook_compatibility():
    """Test that cache files can be read directly in notebooks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=Path(tmpdir))
        
        df = pd.DataFrame({
            "symbol": ["AAPL", "MSFT"],
            "price": [150.0, 250.0],
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        })
        
        cache.set("notebook_test", df)
        
        # Simulate notebook reading the file directly
        cache_file = Path(tmpdir) / "notebook_test.parquet"
        notebook_df = pd.read_parquet(cache_file)
        
        assert len(notebook_df) == 2
        assert list(notebook_df.columns) == ["symbol", "price", "date"]


def test_cache_polars_to_pandas():
    """Test that polars DataFrames are converted to pandas on save."""
    import polars as pl
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=Path(tmpdir))
        
        # Create polars DataFrame
        df = pl.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        })
        
        cache.set("polars_test", df)
        
        # Load back - should be pandas
        loaded = cache.get("polars_test")
        assert isinstance(loaded, pd.DataFrame)
        assert list(loaded.columns) == ["a", "b"]
```

**Step 3: Run tests to verify**

```bash
python -m pytest tests/test_cache_manager.py -v
```

Expected: 5 passing tests

**Step 4: Commit**

```bash
git add tests/test_cache_manager.py src/data/cache_manager.py
git commit -m "refactor: convert cache manager to parquet format"
```

---

## Task 2: Create Unified WRDS Data Loader

**Goal:** Create a single unified loader that handles both CRSP and Compustat data.

**Files:**
- Create: `src/data/wrds_loader.py` (or update if exists)
- Test: `tests/test_wrds_loader.py`

**Step 1: Write failing test**

```python
# tests/test_wrds_loader.py
"""Test unified WRDS data loader."""

import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.data.wrds_loader import WRDSDataLoader


def test_loader_initialization():
    """Test loader initialization with mock mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = WRDSDataLoader(
            wrds_username="mock",
            wrds_password="mock",
            cache_dir=Path(tmpdir),
            mock_mode=True,
        )
        assert loader._mock_mode is True
        assert loader.cache is not None


def test_load_prices_mock():
    """Test loading mock price data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = WRDSDataLoader(
            wrds_username="mock",
            wrds_password="mock",
            cache_dir=Path(tmpdir),
            mock_mode=True,
        )
        
        df = loader.load_prices(
            symbols=["AAPL", "MSFT"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 1, 10),
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "symbol" in df.columns
        assert "date" in df.columns
        assert "close" in df.columns or "prc" in df.columns


def test_load_fundamentals_mock():
    """Test loading mock fundamental data with GICS."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = WRDSDataLoader(
            wrds_username="mock",
            wrds_password="mock",
            cache_dir=Path(tmpdir),
            mock_mode=True,
        )
        
        df = loader.load_fundamentals(
            symbols=["AAPL", "MSFT"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "gsector" in df.columns
        assert "ggroup" in df.columns
        assert "symbol" in df.columns


def test_load_merged_mock():
    """Test loading merged data with automatic CCM linking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = WRDSDataLoader(
            wrds_username="mock",
            wrds_password="mock",
            cache_dir=Path(tmpdir),
            mock_mode=True,
        )
        
        df = loader.load_merged(
            symbols=["AAPL", "MSFT"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        # Should have both price and fundamental columns
        assert "close" in df.columns or "prc" in df.columns
        assert "gsector" in df.columns
        assert "ggroup" in df.columns


def test_caching_works():
    """Test that data is cached and retrievable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = WRDSDataLoader(
            wrds_username="mock",
            wrds_password="mock",
            cache_dir=Path(tmpdir),
            mock_mode=True,
        )
        
        # Load data (should cache)
        df1 = loader.load_prices(
            symbols=["AAPL"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 1, 5),
        )
        
        # Load again (should use cache)
        df2 = loader.load_prices(
            symbols=["AAPL"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 1, 5),
        )
        
        # Data should be identical
        pd.testing.assert_frame_equal(df1, df2)
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_wrds_loader.py::test_loader_initialization -v
```

Expected: FAIL with "WRDSDataLoader not defined"

**Step 3: Write unified loader implementation**

```python
# src/data/wrds_loader.py
"""Unified WRDS data loader for CRSP prices and Compustat fundamentals."""

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import polars as pl

from src.data.cache_manager import CacheManager

logger = logging.getLogger(__name__)


@dataclass
class WRDSConfig:
    """Configuration for WRDS connection."""
    username: Optional[str] = None
    password: Optional[str] = None
    host: str = "wrds-cloud.wharton.upenn.edu"
    port: int = 9737
    mock_mode: bool = False
    
    def get_username(self) -> str:
        return self.username or os.getenv("WRDS_USERNAME", "")
    
    def get_password(self) -> str:
        return self.password or os.getenv("WRDS_PASSWORD", "")
    
    def has_credentials(self) -> bool:
        return bool(self.get_username() and self.get_password())


class WRDSConnection:
    """Wrapper for WRDS database connection with context manager support."""
    
    def __init__(self, config: Optional[WRDSConfig] = None) -> None:
        self.config = config or WRDSConfig()
        self._connection = None
        self._mock_mode = self.config.mock_mode
        
    def connect(self) -> None:
        if not self.config.has_credentials():
            self._mock_mode = True
            return
        
        try:
            import wrds
            self._connection = wrds.Connection(
                wrds_username=self.config.get_username(),
                wrds_password=self.config.get_password(),
            )
        except ImportError as err:
            raise ImportError("wrds library not installed. Run: pip install wrds") from err
        except Exception as err:
            raise ConnectionError(f"Failed to connect to WRDS: {err}") from err
    
    def disconnect(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._mock_mode = False
    
    def is_connected(self) -> bool:
        return self._connection is not None or self._mock_mode
    
    def is_mock_mode(self) -> bool:
        return self._mock_mode
    
    def get_connection(self):
        if self._mock_mode:
            return None
        if not self.is_connected():
            raise RuntimeError("Not connected to WRDS")
        return self._connection
    
    def __enter__(self) -> "WRDSConnection":
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()


class WRDSDataLoader:
    """Unified loader for WRDS data (CRSP prices + Compustat fundamentals)."""
    
    # CRSP fields for daily prices
    CRSP_FIELDS = [
        "permno", "date", "prc", "ret", "vol", "shrout",
        "bidlo", "askhi", "cfacpr", "cfacshr"
    ]
    
    # Compustat fields for fundamentals + GICS
    COMPUSTAT_FIELDS = [
        "gvkey", "datadate", "at", "seq", "ni", "csho", "prcc_f",
        "gsector", "ggroup", "gind", "gsubind"
    ]
    
    def __init__(
        self,
        wrds_username: Optional[str] = None,
        wrds_password: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        mock_mode: bool = False,
    ):
        """Initialize unified WRDS loader.
        
        Args:
            wrds_username: WRDS username (or env var WRDS_USERNAME)
            wrds_password: WRDS password (or env var WRDS_PASSWORD)
            cache_dir: Directory for caching (default: data/cache/)
            mock_mode: Use mock data instead of fetching from WRDS
        """
        self.config = WRDSConfig(
            username=wrds_username,
            password=wrds_password,
            mock_mode=mock_mode,
        )
        self._mock_mode = mock_mode or not self.config.has_credentials()
        
        if cache_dir is None:
            from src.config.settings import get_settings
            settings = get_settings()
            cache_dir = settings.cache_dir
        
        self.cache_dir = Path(cache_dir)
        self.cache = CacheManager(cache_dir=self.cache_dir)
    
    def _generate_cache_key(
        self,
        prefix: str,
        symbols: list,
        start_date: datetime,
        end_date: datetime,
    ) -> str:
        """Generate cache key for query."""
        symbols_str = "_".join(sorted(symbols)[:5])
        return f"{prefix}_{symbols_str}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    
    def load_prices(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Load CRSP daily price data.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with columns: symbol, date, open, high, low, close, volume, etc.
        """
        cache_key = self._generate_cache_key("prices", symbols, start_date, end_date)
        
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.info(f"Using cached price data: {cache_key}")
                return cached
        
        if self._mock_mode:
            df = self._generate_mock_prices(symbols, start_date, end_date)
        else:
            df = self._fetch_prices_from_wrds(symbols, start_date, end_date)
        
        if use_cache:
            self.cache.set(cache_key, df)
        
        return df
    
    def load_fundamentals(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        frequency: str = "annual",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Load Compustat fundamental data with GICS codes.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date
            end_date: End date
            frequency: "annual" or "quarterly"
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with columns: symbol, datadate, gsector, ggroup, at, seq, etc.
        """
        cache_key = self._generate_cache_key(
            f"fundamentals_{frequency}", symbols, start_date, end_date
        )
        
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.info(f"Using cached fundamental data: {cache_key}")
                return cached
        
        if self._mock_mode:
            df = self._generate_mock_fundamentals(symbols, start_date, end_date)
        else:
            df = self._fetch_fundamentals_from_wrds(symbols, start_date, end_date, frequency)
        
        if use_cache:
            self.cache.set(cache_key, df)
        
        return df
    
    def load_merged(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Load merged CRSP + Compustat data with automatic CCM linking.
        
        This loads daily prices and merges them with the most recent fundamental
        data (including GICS codes) for each date.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with both price and fundamental columns
        """
        cache_key = self._generate_cache_key("merged", symbols, start_date, end_date)
        
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.info(f"Using cached merged data: {cache_key}")
                return cached
        
        # Load both datasets
        prices = self.load_prices(symbols, start_date, end_date, use_cache=use_cache)
        fundamentals = self.load_fundamentals(symbols, start_date, end_date, use_cache=use_cache)
        
        # Merge them
        merged = self._merge_prices_and_fundamentals(prices, fundamentals)
        
        if use_cache:
            self.cache.set(cache_key, merged)
        
        return merged
    
    def _generate_mock_prices(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Generate mock price data for testing."""
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        
        data_rows = []
        for symbol in symbols:
            base_price = np.random.uniform(10, 500)
            for date in dates:
                ret = np.random.normal(0, 0.02)
                price = base_price * (1 + ret)
                
                data_rows.append({
                    "symbol": symbol,
                    "date": date,
                    "open": price * (1 + np.random.uniform(-0.01, 0.01)),
                    "high": price * (1 + np.random.uniform(0, 0.03)),
                    "low": price * (1 + np.random.uniform(-0.03, 0)),
                    "close": price,
                    "volume": int(np.random.lognormal(10, 1.5)),
                    "return": ret,
                })
                base_price = price
        
        return pd.DataFrame(data_rows)
    
    def _generate_mock_fundamentals(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Generate mock fundamental data with GICS codes."""
        years = range(start_date.year, end_date.year + 1)
        
        gics_sectors = ["10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60"]
        gics_groups = ["1010", "1510", "2010", "2510", "3010", "3510"]
        
        data_rows = []
        for symbol in symbols:
            for year in years:
                data_rows.append({
                    "symbol": symbol,
                    "datadate": pd.Timestamp(f"{year}-12-31"),
                    "gsector": np.random.choice(gics_sectors),
                    "ggroup": np.random.choice(gics_groups),
                    "at": np.random.lognormal(5, 1),  # Total assets
                    "seq": np.random.lognormal(4, 1),  # Book equity
                    "ni": np.random.lognormal(2, 1),  # Net income
                })
        
        return pd.DataFrame(data_rows)
    
    def _fetch_prices_from_wrds(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Fetch CRSP price data from WRDS."""
        with WRDSConnection(self.config) as conn:
            wrds_conn = conn.get_connection()
            if wrds_conn is None:
                raise RuntimeError("Cannot fetch from WRDS: connection not available")
            
            tickers_str = ", ".join([f"'{s.upper()}'" for s in symbols])
            
            query = f"""
                SELECT 
                    b.ticker as symbol,
                    a.date,
                    a.prc as close,
                    a.ret as return,
                    a.vol as volume,
                    a.shrout,
                    a.bidlo as low,
                    a.askhi as high,
                    a.prc * (1 + (a.bidlo - a.askhi) / a.prc * 0.5) as open
                FROM crsp.dsf AS a
                INNER JOIN crsp.dsenames AS b ON a.permno = b.permno
                WHERE b.ticker IN ({tickers_str})
                AND a.date BETWEEN '{start_date.strftime('%Y-%m-%d')}' AND '{end_date.strftime('%Y-%m-%d')}'
                AND a.date BETWEEN b.namedt AND b.nameenddt
                ORDER BY b.ticker, a.date
            """
            
            assert wrds_conn is not None  # type: ignore
            return wrds_conn.raw_sql(query)
    
    def _fetch_fundamentals_from_wrds(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        frequency: str,
    ) -> pd.DataFrame:
        """Fetch Compustat fundamental data from WRDS."""
        with WRDSConnection(self.config) as conn:
            wrds_conn = conn.get_connection()
            if wrds_conn is None:
                raise RuntimeError("Cannot fetch from WRDS: connection not available")
            
            table = "comp.funda" if frequency == "annual" else "comp.fundq"
            tickers_str = ", ".join([f"'{s.upper()}'" for s in symbols])
            fields_str = ", ".join(self.COMPUSTAT_FIELDS)
            
            query = f"""
                SELECT {fields_str}, a.tic as symbol
                FROM {table} AS a
                WHERE a.tic IN ({tickers_str})
                AND a.datadate BETWEEN '{start_date.strftime('%Y-%m-%d')}' AND '{end_date.strftime('%Y-%m-%d')}'
                AND a.indfmt = 'INDL'
                AND a.datafmt = 'STD'
                AND a.popsrc = 'D'
                AND a.consol = 'C'
                ORDER BY a.tic, a.datadate
            """
            
            assert wrds_conn is not None  # type: ignore
            return wrds_conn.raw_sql(query)
    
    def _merge_prices_and_fundamentals(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge price and fundamental data using CCM linking logic."""
        if prices.empty or fundamentals.empty:
            return prices if fundamentals.empty else pd.concat([prices, fundamentals], axis=1)
        
        # Ensure datetime columns
        prices = prices.copy()
        fundamentals = fundamentals.copy()
        
        if "date" in prices.columns:
            prices["date"] = pd.to_datetime(prices["date"])
        if "datadate" in fundamentals.columns:
            fundamentals["datadate"] = pd.to_datetime(fundamentals["datadate"])
        
        # Merge using forward-fill logic
        merged_rows = []
        for symbol in prices["symbol"].unique():
            price_sym = prices[prices["symbol"] == symbol].copy()
            fund_sym = fundamentals[fundamentals["symbol"] == symbol].copy()
            
            if fund_sym.empty:
                merged_rows.append(price_sym)
                continue
            
            # For each price date, find the most recent fundamental record
            for _, price_row in price_sym.iterrows():
                row_dict = price_row.to_dict()
                
                if not fund_sym.empty and "datadate" in fund_sym.columns:
                    mask = fund_sym["datadate"] <= price_row["date"]
                    if mask.any():
                        latest_fund = fund_sym[mask].iloc[-1]
                        for col in ["gsector", "ggroup", "at", "seq", "ni"]:
                            if col in latest_fund:
                                row_dict[col] = latest_fund[col]
                
                merged_rows.append(row_dict)
        
        return pd.DataFrame(merged_rows)
```

**Step 4: Run tests to verify**

```bash
python -m pytest tests/test_wrds_loader.py -v
```

Expected: 5 passing tests

**Step 5: Commit**

```bash
git add tests/test_wrds_loader.py src/data/wrds_loader.py
git commit -m "feat: add unified WRDS data loader with parquet caching"
```

---

## Task 3: Update Data Module Exports

**Files:**
- Modify: `src/data/__init__.py`

**Step 1: Update exports**

```python
# src/data/__init__.py
"""Data loading and WRDS integration."""

from src.data.cache_manager import CacheManager
from src.data.wrds_loader import WRDSConfig, WRDSConnection, WRDSDataLoader

__all__ = [
    "CacheManager",
    "WRDSConfig",
    "WRDSConnection",
    "WRDSDataLoader",
]
```

**Step 2: Test imports work**

```bash
python -c "from src.data import WRDSDataLoader, CacheManager; print('All imports successful')"
```

Expected: "All imports successful"

**Step 3: Commit**

```bash
git add src/data/__init__.py
git commit -m "chore: update data module exports for unified loader"
```

---

## Task 4: Run Full Test Suite and Verify

**Step 1: Run all data-related tests**

```bash
python -m pytest tests/test_cache_manager.py tests/test_wrds_loader.py -v
```

Expected: 10+ passing tests

**Step 2: Run linting**

```bash
python -m ruff check src/data/
```

Expected: No errors

**Step 3: Verify notebook compatibility**

Create a quick test script:

```python
# test_notebook_compat.py
from datetime import datetime
from src.data import WRDSDataLoader

# Load mock data
loader = WRDSDataLoader(mock_mode=True)
df = loader.load_merged(["AAPL", "MSFT"], datetime(2020, 1, 1), datetime(2020, 1, 10))

# Show first few rows
print(df.head())
print(f"\nColumns: {list(df.columns)}")
print(f"\nCache file location: data/cache/")

# Verify you can read cache in notebook
import pandas as pd
# In your notebook, you can do:
# df = pd.read_parquet("data/cache/merged_AAPL_20200101_20200110.parquet")
```

Run it:

```bash
python test_notebook_compat.py
```

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete unified WRDS data loading system

- Single unified WRDSDataLoader for CRSP + Compustat
- Parquet-based caching (notebook-compatible)
- Internal CCM linking for merged data
- Mock mode for development without credentials
- Full test coverage"
```

---

## Summary

This updated plan implements a simplified, unified WRDS data loading system:

**Key Changes from Original:**
1. ✅ **Single unified loader** (`WRDSDataLoader`) - no separate CRSP/Compustat loaders
2. ✅ **Parquet caching** - readable in notebooks via `pd.read_parquet()`
3. ✅ **Simplified API** - just `load_prices()`, `load_fundamentals()`, `load_merged()`
4. ✅ **Internal CCM linking** - happens automatically in `load_merged()`

**API Usage:**

```python
from src.data import WRDSDataLoader
from datetime import datetime

# Initialize loader
loader = WRDSDataLoader(mock_mode=True)  # or use real credentials

# Load just prices
prices = loader.load_prices(["AAPL", "MSFT"], datetime(2020, 1, 1), datetime(2020, 12, 31))

# Load just fundamentals with GICS
fundamentals = loader.load_fundamentals(["AAPL", "MSFT"], datetime(2020, 1, 1), datetime(2020, 12, 31))

# Load merged (prices + GICS via CCM linking)
merged = loader.load_merged(["AAPL", "MSFT"], datetime(2020, 1, 1), datetime(2020, 12, 31))

# Access cached data in notebook
import pandas as pd
cached_df = pd.read_parquet("data/cache/merged_AAPL_MSFT_20200101_20201231.parquet")
```

**Next Steps:**
After completing this plan, move to feature engineering pipeline (G1-G6 features).

---

**Plan saved to:** `docs/plans/2026-01-20-wrds-data-loading.md`

**Execution options:**

1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks
2. **Parallel Session (separate)** - Open new session with executing-plans

Which approach would you prefer?
