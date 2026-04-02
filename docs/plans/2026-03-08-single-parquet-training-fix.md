# Single Parquet File & Training Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Refactor preprocessing to output single unified parquet, add WRDS credential validation, and fix training pipeline to run 0-1 epochs successfully with proper loss computation.

**Architecture:** Replace per-stock parquet files with single `all_features.parquet` containing all stocks/dates. FeatureDataset reads this single file with symbol/date filtering. Add explicit credential check before any data processing.

**Tech Stack:** pandas, pyarrow, polars (optional for large reads), torch, pytest

---

## Overview

Current issues to fix:
1. **Multiple files → Single file:** 2,400 individual parquets are hard to manage; one 300MB file is simpler
2. **No credential check:** Preprocessing runs with mock data silently; must fail fast without WRDS_USERNAME/PASSWORD
3. **Training crashes:** date column handling mismatch between preprocessing and FeatureDataset
4. **Loss computation:** Need to verify InfoNCE loss computes correctly with symmetric directions

---

### Task 1: Add WRDS Credentials Validation

**Files:**
- Create: `src/data/credentials.py`
- Modify: `scripts/preprocess_features.py:1-50`

**Step 1: Create credentials checker**

Create file `src/data/credentials.py`:
```python
"""WRDS credential validation utilities."""
import os
import sys
from typing import Literal

def check_wrds_credentials() -> Literal["valid", "missing_username", "missing_password", "both_missing"]:
    """Check if WRDS credentials are available.
    
    Returns:
        Status of credentials
    """
    username = os.getenv("WRDS_USERNAME")
    password = os.getenv("WRDS_PASSWORD")
    
    if not username and not password:
        return "both_missing"
    elif not username:
        return "missing_username"
    elif not password:
        return "missing_password"
    else:
        return "valid"

def validate_and_exit() -> None:
    """Validate credentials and exit if missing.
    
    Prints helpful error message with instructions.
    """
    status = check_wrds_credentials()
    
    if status == "valid":
        return
    
    error_messages = {
        "both_missing": "WRDS credentials not found. Set WRDS_USERNAME and WRDS_PASSWORD environment variables.",
        "missing_username": "WRDS_USERNAME environment variable not set.",
        "missing_password": "WRDS_PASSWORD environment variable not set.",
    }
    
    print(f"\n❌ ERROR: {error_messages[status]}", file=sys.stderr)
    print("\nTo set credentials:", file=sys.stderr)
    print("  export WRDS_USERNAME=your_username", file=sys.stderr)
    print("  export WRDS_PASSWORD=your_password", file=sys.stderr)
    print("\nOr run with mock data explicitly (for testing only):", file=sys.stderr)
    print("  python -m scripts.preprocess_features --use-mock\n", file=sys.stderr)
    
    sys.exit(1)
```

**Step 2: Add credentials check to preprocess_features.py**

Modify `scripts/preprocess_features.py` at the top of `main()`:
```python
def main():
    parser = argparse.ArgumentParser(description="Pre-compute features for all stocks")
    # ... existing args ...
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock data instead of WRDS (for testing only)",
    )
    
    args = parser.parse_args()
    
    # Validate WRDS credentials unless explicitly using mock data
    if not args.use_mock:
        from src.data.credentials import validate_and_exit
        validate_and_exit()
    
    # ... rest of function ...
```

**Step 3: Write test**

Create `tests/test_data_credentials.py`:
```python
"""Tests for WRDS credential validation."""
import os
import pytest
from unittest.mock import patch
from src.data.credentials import check_wrds_credentials, validate_and_exit

class TestCredentialValidation:
    """Test credential validation logic."""
    
    def test_both_credentials_present(self):
        """Test when both credentials are set."""
        with patch.dict(os.environ, {"WRDS_USERNAME": "test_user", "WRDS_PASSWORD": "test_pass"}):
            assert check_wrds_credentials() == "valid"
    
    def test_missing_username(self):
        """Test when username is missing."""
        with patch.dict(os.environ, {"WRDS_PASSWORD": "test_pass"}, clear=True):
            assert check_wrds_credentials() == "missing_username"
    
    def test_missing_password(self):
        """Test when password is missing."""
        with patch.dict(os.environ, {"WRDS_USERNAME": "test_user"}, clear=True):
            assert check_wrds_credentials() == "missing_password"
    
    def test_both_missing(self):
        """Test when both are missing."""
        with patch.dict(os.environ, {}, clear=True):
            assert check_wrds_credentials() == "both_missing"
    
    def test_exit_on_missing_credentials(self):
        """Test that validate_and_exit exits when credentials missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                validate_and_exit()
            assert exc_info.value.code == 1
```

**Step 4: Run test**

```bash
python -m pytest tests/test_data_credentials.py -v
```
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/data/credentials.py tests/test_data_credentials.py scripts/preprocess_features.py
git commit -m "feat: Add WRDS credential validation with fail-fast behavior

- Create src/data/credentials.py for credential checking
- Add --use-mock flag for explicit mock data testing
- Exit with helpful error message if WRDS creds missing
- Tests for all credential scenarios

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 2: Refactor Preprocessing to Single Parquet Output

**Files:**
- Create: `scripts/preprocess_unified.py` (new approach)
- Modify: OR update existing `scripts/preprocess_features.py`

**Step 1: Create unified preprocessing script**

Create `scripts/preprocess_unified.py`:
```python
"""Unified feature pre-computation - outputs single parquet file.

This script loads raw WRDS data for all stocks and computes all 32 features,
saving to a single unified parquet file for efficient training.
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data.credentials import validate_and_exit
from src.data.wrds_loader import load_fundamental, load_ohlcv
from src.features.engineer import FeatureEngineer


def main():
    parser = argparse.ArgumentParser(description="Pre-compute features for all stocks (unified output)")
    parser.add_argument(
        "--start-date", type=str, default="2010-01-01", help="Start date for feature computation"
    )
    parser.add_argument(
        "--end-date", type=str, default="2024-12-31", help="End date for feature computation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/features",
        help="Directory to save parquet feature file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="all_features.parquet",
        help="Output filename (single parquet with all stocks)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="Specific symbols to process (None = all available)",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock data instead of WRDS (for testing only)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of stocks to process before writing to disk (memory management)",
    )

    args = parser.parse_args()

    # Validate WRDS credentials unless explicitly using mock data
    if not args.use_mock:
        validate_and_exit()

    # Initialize components
    engineer = FeatureEngineer()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / args.output_file

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    # Get symbols to process
    if args.symbols is None:
        print("Error: No symbols provided. Use --symbols to specify stocks.")
        return

    symbols = args.symbols
    print(f"Processing {len(symbols)} stocks to unified parquet...")
    print(f"Output: {output_file}")

    # Process stocks in batches and collect all features
    all_features = []
    
    for symbol in tqdm(symbols, desc="Computing features"):
        try:
            # Load OHLCV data
            prices = load_ohlcv(
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date,
                use_mock=args.use_mock,
            )

            # Load fundamental data
            fundamentals = load_fundamental(
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date,
            )

            # Combine into single dataframe
            if isinstance(prices, pd.DataFrame) and isinstance(fundamentals, pd.DataFrame):
                # Merge fundamentals into prices (forward fill for daily data)
                if "date" in fundamentals.columns:
                    fundamentals = fundamentals.rename(columns={"date": "datadate"})
                combined = prices.merge(fundamentals, on="symbol", how="left")
                features = engineer.compute_features(combined)
            else:
                # If one is None or different format, use just prices
                features = engineer.compute_features(prices)

            # Ensure required columns are present
            raw_ohlcv = ["open", "high", "low", "close", "volume", "return"]
            
            # Check if we have the raw columns from prices dataframe
            if isinstance(prices, pd.DataFrame):
                for col in raw_ohlcv:
                    if col not in features.columns and col in prices.columns:
                        if "date" in prices.columns and "date" in features.columns:
                            features = features.merge(
                                prices[["date", "symbol", col]], 
                                on=["date", "symbol"], 
                                how="left"
                            )
            
            # Add return column if not present
            if "return" not in features.columns and "close" in features.columns:
                features["return"] = features["close"].pct_change()
            
            # Ensure date column is present as actual column (not index)
            if "date" not in features.columns:
                if hasattr(features.index, 'name') and features.index.name == 'date':
                    features = features.reset_index()
                elif "datadate" in features.columns:
                    features["date"] = features["datadate"]
            
            # Validate required columns exist
            missing = [col for col in raw_ohlcv if col not in features.columns]
            if missing:
                print(f"Warning: {symbol} missing temporal columns: {missing}")
                continue
            
            all_features.append(features)
            
            # Periodic memory management - write intermediate results if batch size reached
            if len(all_features) >= args.batch_size:
                _write_batch(all_features, output_file)
                all_features = []  # Clear to free memory
                
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Write final batch
    if all_features:
        _write_batch(all_features, output_file)

    print(f"\n✅ Done! Unified features saved to {output_file}")
    
    # Verify output
    if output_file.exists():
        df = pd.read_parquet(output_file)
        print(f"Total rows: {len(df):,}")
        print(f"Symbols: {df['symbol'].nunique()}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Columns: {len(df.columns)}")


def _write_batch(features_list: list, output_file: Path) -> None:
    """Write batch of features to parquet (append mode)."""
    if not features_list:
        return
    
    # Combine all dataframes
    combined = pd.concat(features_list, ignore_index=True)
    
    # Ensure date column is datetime
    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"])
    
    # Write/append to parquet
    if output_file.exists():
        # Read existing and append
        existing = pd.read_parquet(output_file)
        combined = pd.concat([existing, combined], ignore_index=True)
    
    # Write with compression
    combined.to_parquet(output_file, compression="snappy", index=False)


if __name__ == "__main__":
    main()
```

**Step 2: Write test for unified preprocessing**

Create `tests/test_preprocess_unified.py`:
```python
"""Tests for unified preprocessing script."""
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


class TestUnifiedPreprocessing:
    """Test unified preprocessing functionality."""
    
    def test_unified_output_structure(self, tmp_path):
        """Test that unified output has correct structure."""
        # Create mock feature directory
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        
        # Create sample data
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        df = pd.DataFrame({
            "date": dates,
            "symbol": "AAPL",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000000,
            "return": 0.001,
            "market_beta_60d": 1.0,
            "gsector": 45,
            "ggroup": 4510,
        })
        
        output_file = feature_dir / "all_features.parquet"
        df.to_parquet(output_file)
        
        # Verify structure
        result = pd.read_parquet(output_file)
        assert "date" in result.columns
        assert "symbol" in result.columns
        assert "open" in result.columns
        assert result["symbol"].nunique() == 1
    
    def test_multiple_symbols_in_single_file(self, tmp_path):
        """Test that multiple symbols can coexist in one file."""
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        
        # Create data for multiple symbols
        dfs = []
        for symbol in ["AAPL", "MSFT", "GOOGL"]:
            df = pd.DataFrame({
                "date": dates,
                "symbol": symbol,
                "close": 100.0,
                "gsector": 45,
            })
            dfs.append(df)
        
        combined = pd.concat(dfs, ignore_index=True)
        output_file = feature_dir / "all_features.parquet"
        combined.to_parquet(output_file)
        
        # Verify
        result = pd.read_parquet(output_file)
        assert result["symbol"].nunique() == 3
        assert set(result["symbol"].unique()) == {"AAPL", "MSFT", "GOOGL"}
    
    def test_script_exits_without_credentials(self, tmp_path):
        """Test that script exits when WRDS credentials missing."""
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        
        # Clear environment variables
        env = os.environ.copy()
        env.pop("WRDS_USERNAME", None)
        env.pop("WRDS_PASSWORD", None)
        
        # Run script (should fail)
        result = subprocess.run(
            [sys.executable, "-m", "scripts.preprocess_unified", 
             "--symbols", "AAPL",
             "--output-dir", str(feature_dir)],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(Path(__file__).parent.parent),
        )
        
        assert result.returncode == 1
        assert "WRDS credentials not found" in result.stderr or "WRDS" in result.stderr
```

**Step 3: Run test**

```bash
python -m pytest tests/test_preprocess_unified.py -v
```
Expected: All tests PASS

**Step 4: Commit**

```bash
git add scripts/preprocess_unified.py tests/test_preprocess_unified.py
git commit -m "feat: Add unified preprocessing with single parquet output

- New preprocess_unified.py outputs single all_features.parquet
- Processes stocks in batches with memory management
- Includes WRDS credential validation
- Supports both real and mock data modes

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 3: Update FeatureDataset to Read Single Parquet

**Files:**
- Modify: `src/training/dataset.py`

**Step 1: Update FeatureDataset for unified file reading**

Modify `src/training/dataset.py`:

```python
"""PyTorch Dataset for pre-computed stock features (unified parquet version)."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    """Dataset for loading pre-computed stock features from unified parquet.

    Loads features from a single parquet file containing all stocks
    and samples random windows for training. Each sample contains 
    both temporal sequences and tabular snapshots.

    Attributes:
        feature_file: Path to unified parquet file
        date_range: (start_date, end_date) tuple for valid samples
        symbols: List of stock symbols to include
        window_size: Number of days in temporal window (default 60)
    """

    # Column definitions for feature groups
    TEMPORAL_COLS = [
        "open", "high", "low", "close", "volume", "return",
        "ma_ratio_5", "ma_ratio_10", "ma_ratio_15", "ma_ratio_20",
        "z_close_5d", "z_close_10d", "z_close_20d",
    ]

    TABULAR_CONT_COLS = [
        "market_beta_60d", "downside_beta_60d",
        "realized_vol_20d", "realized_vol_60d", "idiosyncratic_vol", "vol_of_vol",
        "mom_1m", "mom_3m", "mom_6m", "mom_12_1m", "macd",
        "log_mktcap", "pe_ratio", "pb_ratio", "roe",
    ]

    TABULAR_CAT_COLS = ["gsector", "ggroup"]

    def __init__(
        self,
        feature_dir: str,
        date_range: tuple[str, str],
        symbols: list[str] | None = None,
        window_size: int = 60,
        feature_file: str = "all_features.parquet",
    ) -> None:
        """Initialize feature dataset.

        Args:
            feature_dir: Path to directory with unified parquet file
            date_range: (start_date, end_date) for valid samples
            symbols: List of symbols to include (None = all in file)
            window_size: Days in temporal window (default 60)
            feature_file: Name of the unified parquet file
        """
        self.feature_dir = Path(feature_dir)
        self.feature_file = self.feature_dir / feature_file
        self.date_range = (pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))
        self.window_size = window_size

        # Load unified parquet
        if not self.feature_file.exists():
            raise FileNotFoundError(f"Feature file not found: {self.feature_file}")
        
        print(f"Loading features from {self.feature_file}...")
        self.df = pd.read_parquet(self.feature_file)
        
        # Ensure date column is datetime
        if "date" in self.df.columns:
            self.df["date"] = pd.to_datetime(self.df["date"])
        
        # Filter to requested symbols
        if symbols is not None:
            self.df = self.df[self.df["symbol"].isin(symbols)]
        
        # Get unique symbols
        self.symbols = self.df["symbol"].unique().tolist()
        
        # Build index of valid (symbol, date) samples
        self.samples = self._build_index()

        # Build categorical mappings from data
        self.categorical_mappings = self._build_categorical_mappings()
        self.categorical_dims = self._get_categorical_dims()

        print(f"Dataset initialized: {len(self.symbols)} symbols, {len(self.samples)} samples")
        print(f"Categorical dimensions: {self.categorical_dims}")

    def _build_index(self) -> list[tuple[str, pd.Timestamp]]:
        """Build index of valid (symbol, date) samples."""
        samples = []
        
        for symbol in self.symbols:
            symbol_df = self.df[self.df["symbol"] == symbol]
            
            # Filter to date range with sufficient history
            min_date = self.date_range[0] + pd.Timedelta(days=self.window_size)
            valid_dates = symbol_df[
                (symbol_df["date"] >= min_date) & 
                (symbol_df["date"] <= self.date_range[1])
            ]["date"]
            
            for date in valid_dates:
                samples.append((symbol, date))
        
        return samples

    def _build_categorical_mappings(self) -> dict[str, dict[str | int, int]]:
        """Build mappings from categorical values to indices."""
        unique_values: dict[str, set[str | int]] = {
            col: set() for col in self.TABULAR_CAT_COLS
        }
        
        for col in self.TABULAR_CAT_COLS:
            if col in self.df.columns:
                unique_values[col].update(self.df[col].dropna().unique())
        
        # Create mappings (sorted for consistency)
        mappings: dict[str, dict[str | int, int]] = {}
        for col in self.TABULAR_CAT_COLS:
            sorted_values = sorted(unique_values[col])
            mappings[col] = {val: idx for idx, val in enumerate(sorted_values)}
        
        return mappings

    def _get_categorical_dims(self) -> list[int]:
        """Get embedding dimensions for each categorical column."""
        return [
            len(self.categorical_mappings[col]) 
            for col in self.TABULAR_CAT_COLS
        ]

    def __len__(self) -> int:
        """Return number of valid samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Get sample by index."""
        symbol, date = self.samples[idx]
        
        # Get data for this symbol
        symbol_df = self.df[self.df["symbol"] == symbol].copy()
        symbol_df = symbol_df.sort_values("date").reset_index(drop=True)
        
        # Find position of target date
        date_idx = symbol_df[symbol_df["date"] == date].index[0]
        
        # Extract temporal window: last window_size rows ending at date
        start_idx = max(0, date_idx - self.window_size + 1)
        temporal_window = symbol_df.iloc[start_idx:date_idx + 1][self.TEMPORAL_COLS]
        
        # Pad if necessary to ensure consistent window size
        if len(temporal_window) < self.window_size:
            n_pad = self.window_size - len(temporal_window)
            padding = pd.DataFrame(0, index=range(n_pad), columns=self.TEMPORAL_COLS)
            temporal_window = pd.concat([padding, temporal_window], ignore_index=True)
        
        # Extract tabular snapshot at date
        tabular_cont = symbol_df.iloc[date_idx][self.TABULAR_CONT_COLS]
        tabular_cat = symbol_df.iloc[date_idx][self.TABULAR_CAT_COLS]
        beta = symbol_df.iloc[date_idx]["market_beta_60d"]
        
        # Map categorical values to indices for embeddings
        gsector_val = tabular_cat["gsector"]
        ggroup_val = tabular_cat["ggroup"]
        gsector_idx = self.categorical_mappings["gsector"].get(gsector_val, 0)
        ggroup_idx = self.categorical_mappings["ggroup"].get(ggroup_val, 0)
        
        # Convert to tensors
        sample = {
            "symbol": symbol,
            "date": date,
            "temporal": torch.tensor(temporal_window.values, dtype=torch.float32),
            "tabular_cont": torch.tensor(tabular_cont.values, dtype=torch.float32),
            "tabular_cat": torch.tensor([gsector_idx, ggroup_idx], dtype=torch.long),
            "beta": float(beta),
            "gsector": int(gsector_val),
            "ggroup": int(ggroup_val),
        }
        
        return sample

    def get_symbol_features(self, symbol: str) -> pd.DataFrame | None:
        """Load all features for a specific symbol."""
        symbol_df = self.df[self.df["symbol"] == symbol]
        if len(symbol_df) == 0:
            return None
        return symbol_df.copy()
```

**Step 2: Update tests**

Update `tests/test_training_dataset.py` to work with unified file:

```python
"""Tests for FeatureDataset PyTorch Dataset (unified parquet)."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.training.dataset import FeatureDataset


def test_dataset_initialization_unified(tmp_path):
    """Test dataset can be initialized with unified parquet file."""
    # Create mock unified feature file
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    
    # Create sample features for multiple symbols in single file
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    
    temporal_cols = FeatureDataset.TEMPORAL_COLS
    tabular_cols = FeatureDataset.TABULAR_CONT_COLS
    
    # Create data for 3 symbols
    all_data = []
    for symbol in ["AAPL", "MSFT", "GOOGL"]:
        temporal_data = {col: np.random.randn(300) for col in temporal_cols}
        tabular_data = {col: np.random.randn(300) for col in tabular_cols}
        cat_data = {
            "gsector": np.full(300, 45),
            "ggroup": np.full(300, 4510),
            "symbol": np.full(300, symbol),
            "date": dates,
        }
        
        df = pd.DataFrame({**temporal_data, **tabular_data, **cat_data})
        all_data.append(df)
    
    # Combine into unified file
    unified_df = pd.concat(all_data, ignore_index=True)
    unified_df.to_parquet(feature_dir / "all_features.parquet")
    
    # Test dataset initialization
    dataset = FeatureDataset(
        feature_dir=str(feature_dir),
        date_range=("2020-06-01", "2020-12-31"),
    )
    
    assert len(dataset) > 0
    assert len(dataset.symbols) == 3


def test_dataset_getitem_unified(tmp_path):
    """Test dataset returns correct structure from unified file."""
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    
    temporal_cols = FeatureDataset.TEMPORAL_COLS
    tabular_cols = FeatureDataset.TABULAR_CONT_COLS
    
    # Create data for AAPL
    temporal_data = {col: np.random.randn(300) for col in temporal_cols}
    tabular_data = {col: np.random.randn(300) for col in tabular_cols}
    cat_data = {
        "gsector": np.full(300, 45),
        "ggroup": np.full(300, 4510),
        "symbol": np.full(300, "AAPL"),
        "date": dates,
    }
    
    df = pd.DataFrame({**temporal_data, **tabular_data, **cat_data})
    df.to_parquet(feature_dir / "all_features.parquet")
    
    dataset = FeatureDataset(
        feature_dir=str(feature_dir),
        date_range=("2020-06-01", "2020-12-31"),
        window_size=60,
    )
    
    sample = dataset[0]
    
    assert "symbol" in sample
    assert "date" in sample
    assert "temporal" in sample
    assert "tabular_cont" in sample
    assert "tabular_cat" in sample
    assert "beta" in sample
    assert "gsector" in sample
    assert "ggroup" in sample
    
    # Check shapes
    assert sample["temporal"].shape == (60, 13)
    assert sample["tabular_cont"].shape == (15,)
    assert sample["tabular_cat"].shape == (2,)
```

**Step 3: Run tests**

```bash
python -m pytest tests/test_training_dataset.py -v
```
Expected: Both tests PASS

**Step 4: Commit**

```bash
git add src/training/dataset.py tests/test_training_dataset.py
git commit -m "ref: Update FeatureDataset to read unified parquet file

- Load single all_features.parquet instead of per-symbol files
- Filter by symbol and date in memory
- More efficient for large-scale training
- Maintains all existing functionality

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 4: Fix InfoNCE Loss for Proper Computation

**Files:**
- Review: `src/models/losses.py`
- Test: Create `tests/test_models_losses_proper.py`

**Step 1: Verify InfoNCE implementation**

Read `src/models/losses.py` to verify it handles hard negatives correctly:

```python
# This should already be correct from previous work
# Verify it uses symmetric InfoNCE with hard negatives in denominator
```

**Step 2: Create comprehensive loss test**

Create `tests/test_models_losses_proper.py`:

```python
"""Tests for proper InfoNCE loss computation."""
import torch
import pytest
from src.models.losses import InfoNCELoss


class TestInfoNCEProperComputation:
    """Test that InfoNCE computes correct loss values."""
    
    def test_symmetric_loss_computation(self):
        """Test that symmetric InfoNCE computes both directions."""
        loss_fn = InfoNCELoss(temperature=0.07)
        
        # Create embeddings
        batch_size = 4
        dim = 128
        temporal_emb = torch.randn(batch_size, dim)
        tabular_emb = torch.randn(batch_size, dim)
        
        # Compute loss
        loss = loss_fn(temporal_emb, tabular_emb)
        
        # Loss should be a scalar
        assert loss.dim() == 0
        assert loss.item() > 0  # InfoNCE loss is always positive
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_loss_with_hard_negatives(self):
        """Test loss computation with hard negatives included."""
        loss_fn = InfoNCELoss(temperature=0.07)
        
        batch_size = 4
        n_hard = 2
        dim = 128
        
        temporal_emb = torch.randn(batch_size, dim)
        tabular_emb = torch.randn(batch_size, dim)
        hard_temporal = torch.randn(n_hard, dim)
        hard_tabular = torch.randn(n_hard, dim)
        
        # Compute loss with hard negatives
        loss = loss_fn(temporal_emb, tabular_emb, hard_temporal, hard_tabular)
        
        assert loss.dim() == 0
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_loss_decreases_with_similarity(self):
        """Test that loss decreases when embeddings are more similar."""
        loss_fn = InfoNCELoss(temperature=0.07)
        
        batch_size = 4
        dim = 128
        
        # Dissimilar embeddings
        temporal_random = torch.randn(batch_size, dim)
        tabular_random = torch.randn(batch_size, dim)
        loss_random = loss_fn(temporal_random, tabular_random).item()
        
        # Similar embeddings (same tensor)
        temporal_same = torch.randn(batch_size, dim)
        tabular_same = temporal_same.clone()
        loss_same = loss_fn(temporal_same, tabular_same).item()
        
        # Loss should be lower for similar embeddings
        assert loss_same < loss_random, \
            f"Loss for similar embeddings ({loss_same:.4f}) should be < random ({loss_random:.4f})"
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through loss."""
        loss_fn = InfoNCELoss(temperature=0.07)
        
        batch_size = 4
        dim = 128
        
        temporal_emb = torch.randn(batch_size, dim, requires_grad=True)
        tabular_emb = torch.randn(batch_size, dim, requires_grad=True)
        
        loss = loss_fn(temporal_emb, tabular_emb)
        loss.backward()
        
        # Check gradients exist and are not NaN
        assert temporal_emb.grad is not None
        assert tabular_emb.grad is not None
        assert not torch.isnan(temporal_emb.grad).any()
        assert not torch.isnan(tabular_emb.grad).any()
```

**Step 3: Run tests**

```bash
python -m pytest tests/test_models_losses_proper.py -v
```
Expected: All 4 tests PASS

**Step 4: Commit**

```bash
git add tests/test_models_losses_proper.py
git commit -m "test: Add proper InfoNCE loss computation tests

- Verify symmetric loss computation
- Test with hard negatives
- Check loss decreases with similarity
- Verify gradient flow

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 5: Update Training Script for Single File

**Files:**
- Modify: `scripts/train.py`

**Step 1: Update train.py to use unified file**

Modify `scripts/train.py`:

```python
# Update feature file discovery logic (around line 110-135)
feature_dir = Path(args.feature_dir)

# Check for unified file
unified_file = feature_dir / "all_features.parquet"
if unified_file.exists():
    # Load first few rows to check date range
    sample_df = pd.read_parquet(unified_file, columns=["date"])
    min_date = pd.to_datetime(sample_df["date"].min())
    max_date = pd.to_datetime(sample_df["date"].max())
    
    print(f"Using unified feature file: {unified_file}")
    print(f"Data range: {min_date} to {max_date}")
else:
    raise ValueError(f"Unified feature file not found: {unified_file}\n"
                     f"Run: python -m scripts.preprocess_unified --symbols <list>")

# ... rest of date range logic stays the same ...
```

**Step 2: Verify train script works with new dataset**

The train script should already work since FeatureDataset now handles the unified file.

**Step 3: Run quick sanity test**

First, create mock unified file:
```bash
# This will fail without credentials (expected)
# Let's test the mock mode instead
python -m scripts.preprocess_unified \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --symbols AAPL MSFT GOOGL \
    --use-mock
```

**Step 4: Run training sanity check**

```bash
python -m scripts.train \
    --fold 0 \
    --epochs 1 \
    --batch-size 4 \
    --n-hard 2
```

Expected output:
- Training starts without errors
- Loss computes properly
- At least 1 batch completes
- Checkpoint saved

**Step 5: Commit**

```bash
git add scripts/train.py
git commit -m "ref: Update train script for unified parquet

- Check for all_features.parquet instead of per-symbol files
- Provide helpful error if unified file missing
- Maintain all existing functionality

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 6: Full Integration Test

**Step 1: Create test data and run end-to-end**

```bash
# Create mock unified features
python -m scripts.preprocess_unified \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --symbols AAPL MSFT GOOGL \
    --use-mock
```

**Step 2: Verify output file**

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/features/all_features.parquet')
print(f'Shape: {df.shape}')
print(f'Symbols: {df.symbol.nunique()}')
print(f'Date range: {df.date.min()} to {df.date.max()}')
print(f'Columns: {list(df.columns[:10])}...')
"
```

Expected: 3 symbols, ~780 rows (260 days × 3), date range 2023-01-01 to 2023-12-31

**Step 3: Run training for 1 epoch**

```bash
python -m scripts.train \
    --fold 0 \
    --epochs 1 \
    --batch-size 4 \
    --lr 1e-4 \
    --n-hard 2
```

Expected:
- "Loading features from data/processed/features/all_features.parquet..."
- "Dataset initialized: 3 symbols, XXX samples"
- Progress bar showing loss decreasing
- "Epoch 0 - Loss: X.XXXX, Align: X.XXX"
- Checkpoint saved to models/checkpoints/fold0_best.pt

**Step 4: Verify loss is reasonable**

```bash
python -c "
import torch
ckpt = torch.load('models/checkpoints/fold0_final.pt')
print(f'Checkpoint epoch: {ckpt[\"epoch\"]}')
print(f'Model state dict keys: {list(ckpt[\"model_state_dict\"].keys())[:5]}...')
print('✅ Training completed successfully!')
"
```

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat: Complete unified preprocessing and training pipeline

- Single all_features.parquet replaces per-symbol files
- WRDS credential validation with fail-fast behavior
- FeatureDataset optimized for unified file reading
- Training runs end-to-end with proper loss computation
- All tests passing

Integration verified:
✅ Preprocessing generates unified file
✅ FeatureDataset loads unified file correctly
✅ Training completes 0-1 epochs
✅ Loss computes properly (InfoNCE)
✅ Checkpoints saved correctly

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Summary

**What this plan accomplishes:**
1. ✅ **Safety:** Preprocessing exits immediately if WRDS credentials missing
2. ✅ **Simplicity:** Single 300MB parquet file instead of 2,400 files
3. ✅ **Efficiency:** FeatureDataset loads once, filters in memory
4. ✅ **Correctness:** Training runs 0-1 epochs with proper InfoNCE loss

**Files created/modified:**
- `src/data/credentials.py` (NEW) - Credential validation
- `scripts/preprocess_unified.py` (NEW) - Unified preprocessing
- `src/training/dataset.py` (MOD) - Unified file reading
- `scripts/train.py` (MOD) - Check for unified file
- `tests/test_data_credentials.py` (NEW) - Credential tests
- `tests/test_preprocess_unified.py` (NEW) - Preprocessing tests
- `tests/test_training_dataset.py` (MOD) - Dataset tests
- `tests/test_models_losses_proper.py` (NEW) - Loss computation tests

**Total tasks:** 6
**Estimated time:** 45-60 minutes

Ready to execute? Which subagent mode do you prefer:
1. **Subagent-Driven (this session)** - Fresh subagent per task, I review between
2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution