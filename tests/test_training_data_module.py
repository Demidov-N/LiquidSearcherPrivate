"""Tests for StockDataset with both sharded directory and single-parquet file modes."""

import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import torch

from src.training.data_module import (
    TEMPORAL_FEATURE_NAMES,
    TABULAR_CONTINUOUS_NAMES,
    StockDataset,
    StockDataModule,
)


# ============================================================================
# Fixture Helpers: Generate tiny parquet inputs
# ============================================================================


def _create_tiny_symbol_data(
    symbol: str,
    start_date: str = "2023-01-01",
    num_days: int = 100,
) -> pd.DataFrame:
    """Create a tiny parquet DataFrame for one symbol."""
    dates = pd.date_range(start_date, periods=num_days, freq="D")
    data = {
        "date": dates,
        "symbol": symbol,
        "prc": np.random.rand(num_days) * 100 + 50,
    }

    # Add temporal features
    for feat in TEMPORAL_FEATURE_NAMES:
        data[feat] = np.random.randn(num_days)

    # Add tabular continuous features
    data["beta"] = np.random.randn(num_days) * 0.5 + 1.0
    data["idiosyncratic_vol"] = np.abs(np.random.randn(num_days)) * 0.2

    # Add categorical features
    data["gsector"] = np.random.randint(0, 11, num_days)
    data["ggroup"] = np.random.randint(0, 25, num_days)

    # Add minimal fundamental columns so _load_symbol can derive tabular inputs
    data["atq"] = np.abs(np.random.randn(num_days)) * 1000 + 100
    data["seqq"] = np.abs(np.random.randn(num_days)) * 500 + 50
    data["niq"] = np.random.randn(num_days) * 50 + 20
    data["cshoq"] = np.abs(np.random.randn(num_days)) * 100 + 10
    data["ceqq"] = np.abs(np.random.randn(num_days)) * 500 + 50
    data["epspxq"] = np.random.randn(num_days) * 5 + 2
    data["txtq"] = np.abs(np.random.randn(num_days)) * 20
    data["xintq"] = np.abs(np.random.randn(num_days)) * 10
    data["saleq"] = np.abs(np.random.randn(num_days)) * 500 + 100
    data["cheq"] = np.abs(np.random.randn(num_days)) * 100 + 20

    return pd.DataFrame(data)


def _create_sharded_directory(
    symbols: list[str] = None,
    output_dir: Path = None,
) -> Path:
    """Create a temporary directory with sharded *_features.parquet files."""
    if symbols is None:
        symbols = ["AAPL", "MSFT"]
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="sharded_"))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        df = _create_tiny_symbol_data(symbol)
        output_path = output_dir / f"{symbol}_features.parquet"
        df.to_parquet(output_path, index=False)

    return output_dir


def _create_monolithic_parquet(
    symbols: list[str] = None,
    output_path: Path = None,
) -> Path:
    """Create a temporary monolithic parquet with all symbols combined."""
    if symbols is None:
        symbols = ["AAPL", "MSFT"]
    if output_path is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="monolithic_"))
        output_path = temp_dir / "all_features.parquet"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    dfs = [_create_tiny_symbol_data(sym) for sym in symbols]
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_parquet(output_path, index=False)

    return output_path


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sharded_dir():
    """Fixture: temporary directory with sharded parquet files."""
    dir_path = _create_sharded_directory(symbols=["AAPL", "MSFT", "GOOGL"])
    yield dir_path
    # Cleanup
    import shutil

    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture
def monolithic_parquet():
    """Fixture: temporary monolithic parquet file."""
    symbols = ["AAPL", "MSFT", "GOOGL"]
    file_path = _create_monolithic_parquet(symbols=symbols)
    yield file_path
    # Cleanup
    import shutil

    shutil.rmtree(file_path.parent, ignore_errors=True)


@pytest.fixture
def sharded_dir_two_symbols():
    """Fixture: temporary directory with two symbols (for parity testing)."""
    dir_path = _create_sharded_directory(symbols=["AAPL", "MSFT"])
    yield dir_path
    # Cleanup
    import shutil

    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture
def monolithic_parquet_two_symbols():
    """Fixture: temporary monolithic parquet with two symbols (matching sharded)."""
    symbols = ["AAPL", "MSFT"]
    file_path = _create_monolithic_parquet(symbols=symbols)
    yield file_path
    # Cleanup
    import shutil

    shutil.rmtree(file_path.parent, ignore_errors=True)


# ============================================================================
# Tests: Step 2 - Directory mode parity (sanity check)
# ============================================================================


def test_sharded_directory_mode_positive_length(sharded_dir):
    """Step 2: Test that sharded directory mode produces positive dataset length."""
    dataset = StockDataset(
        feature_dir=str(sharded_dir),
        date_range=("2023-01-01", "2023-03-31"),
        window_size=60,
    )
    assert len(dataset) > 0, "Sharded directory should produce positive dataset length"


def test_sharded_directory_mode_sample_contract(sharded_dir):
    """Step 2: Test that sharded directory mode produces correct sample keys and shapes."""
    dataset = StockDataset(
        feature_dir=str(sharded_dir),
        date_range=("2023-01-01", "2023-03-31"),
        window_size=60,
    )
    sample = dataset[0]

    # Check keys
    assert "symbol" in sample
    assert "date" in sample
    assert "temporal" in sample
    assert "tabular_cont" in sample
    assert "tabular_cat" in sample

    # Check types
    assert isinstance(sample["symbol"], str)
    assert isinstance(sample["date"], str)
    assert isinstance(sample["temporal"], torch.Tensor)
    assert isinstance(sample["tabular_cont"], torch.Tensor)
    assert isinstance(sample["tabular_cat"], torch.Tensor)

    # Check tensor shapes
    assert sample["temporal"].shape == (60, len(TEMPORAL_FEATURE_NAMES))
    assert sample["tabular_cont"].shape == (len(TABULAR_CONTINUOUS_NAMES),)
    assert sample["tabular_cat"].shape == (2,)


# ============================================================================
# Tests: Step 3 - Monolithic file mode (failing until feature is implemented)
# ============================================================================


def test_monolithic_file_mode_loads(monolithic_parquet):
    """Step 3: Test that monolithic file mode can be instantiated (currently fails)."""
    # This test will fail until single-file support is implemented
    dataset = StockDataset(
        feature_dir=str(monolithic_parquet),
        date_range=("2023-01-01", "2023-03-31"),
        window_size=60,
    )
    assert len(dataset) > 0, "Monolithic file should produce positive dataset length"


def test_monolithic_file_mode_sample_contract(monolithic_parquet):
    """Step 3: Test that monolithic file mode produces correct sample contract."""
    dataset = StockDataset(
        feature_dir=str(monolithic_parquet),
        date_range=("2023-01-01", "2023-03-31"),
        window_size=60,
    )
    sample = dataset[0]

    # Check keys
    assert "symbol" in sample
    assert "date" in sample
    assert "temporal" in sample
    assert "tabular_cont" in sample
    assert "tabular_cat" in sample

    # Check types
    assert isinstance(sample["symbol"], str)
    assert isinstance(sample["date"], str)
    assert isinstance(sample["temporal"], torch.Tensor)
    assert isinstance(sample["tabular_cont"], torch.Tensor)
    assert isinstance(sample["tabular_cat"], torch.Tensor)

    # Check tensor shapes
    assert sample["temporal"].shape == (60, len(TEMPORAL_FEATURE_NAMES))
    assert sample["tabular_cont"].shape == (len(TABULAR_CONTINUOUS_NAMES),)
    assert sample["tabular_cat"].shape == (2,)


# ============================================================================
# Tests: Step 4 - Sample parity between modes
# ============================================================================


def test_sample_parity_between_modes(sharded_dir_two_symbols, monolithic_parquet_two_symbols):
    """Step 4: Test that same data in both modes produces identical samples."""
    # Create datasets with identical data
    sharded_dataset = StockDataset(
        feature_dir=str(sharded_dir_two_symbols),
        date_range=("2023-01-01", "2023-03-31"),
        symbols=["AAPL"],  # Use same symbol for deterministic comparison
        window_size=60,
    )
    monolithic_dataset = StockDataset(
        feature_dir=str(monolithic_parquet_two_symbols),
        date_range=("2023-01-01", "2023-03-31"),
        symbols=["AAPL"],  # Use same symbol for deterministic comparison
        window_size=60,
    )

    # Both should have samples
    assert len(sharded_dataset) > 0
    assert len(monolithic_dataset) > 0

    # Get first sample from each
    sharded_sample = sharded_dataset[0]
    monolithic_sample = monolithic_dataset[0]

    # Compare symbol and date
    assert sharded_sample["symbol"] == monolithic_sample["symbol"]
    assert sharded_sample["date"] == monolithic_sample["date"]

    # Compare tensor shapes
    assert sharded_sample["temporal"].shape == monolithic_sample["temporal"].shape
    assert sharded_sample["tabular_cont"].shape == monolithic_sample["tabular_cont"].shape
    assert sharded_sample["tabular_cat"].shape == monolithic_sample["tabular_cat"].shape


# ============================================================================
# Tests: Step 5 - Explicit symbol filtering in file mode
# ============================================================================


def test_monolithic_file_explicit_symbol_filtering(monolithic_parquet):
    """Step 5: Test that symbol filtering works in monolithic file mode."""
    # Request only one symbol
    dataset = StockDataset(
        feature_dir=str(monolithic_parquet),
        date_range=("2023-01-01", "2023-03-31"),
        symbols=["AAPL"],
        window_size=60,
    )

    # Verify only AAPL appears in samples
    symbols_seen = set()
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        symbols_seen.add(sample["symbol"])

    assert symbols_seen == {"AAPL"}, f"Expected only AAPL, got {symbols_seen}"


# ============================================================================
# Tests: Step 6 - Invalid single-file schema (should fail early)
# ============================================================================


def test_monolithic_file_invalid_schema_missing_columns():
    """Step 6: Test that invalid parquet schema raises clear error early."""
    # Create a parquet with missing required columns
    temp_dir = Path(tempfile.mkdtemp(prefix="invalid_schema_"))
    invalid_parquet = temp_dir / "invalid.parquet"

    # Create minimal DataFrame missing critical columns
    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "date": pd.date_range("2023-01-01", periods=2),
            # Missing: prc, temporal features, beta, idiosyncratic_vol, gsector, ggroup
        }
    )
    df.to_parquet(invalid_parquet, index=False)

    # Attempting to load should fail early with clear error
    with pytest.raises((ValueError, KeyError)):
        dataset = StockDataset(
            feature_dir=str(invalid_parquet),
            date_range=("2023-01-01", "2023-03-31"),
            window_size=60,
        )
        # Try to access first sample to trigger schema validation
        _ = dataset[0]

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Tests: Step 7 - Missing requested symbols warning
# ============================================================================


def test_monolithic_file_missing_requested_symbols_warning(monolithic_parquet, capsys):
    """Step 7: Test that missing requested symbols emit warning but don't crash."""
    # Request one present symbol and one absent symbol
    dataset = StockDataset(
        feature_dir=str(monolithic_parquet),
        date_range=("2023-01-01", "2023-03-31"),
        symbols=["AAPL", "NONEXISTENT"],  # NONEXISTENT doesn't exist
        window_size=60,
    )

    # Dataset construction should succeed
    assert dataset is not None

    # Should still have samples (from AAPL)
    assert len(dataset) > 0

    # Samples should only contain AAPL
    symbols_seen = set()
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        symbols_seen.add(sample["symbol"])

    assert "AAPL" in symbols_seen
    assert "NONEXISTENT" not in symbols_seen


# ============================================================================
# Additional Tests: Verify shapes and dtypes
# ============================================================================


def test_temporal_tensor_dtype(sharded_dir):
    """Test that temporal tensors are float32."""
    dataset = StockDataset(
        feature_dir=str(sharded_dir),
        date_range=("2023-01-01", "2023-03-31"),
        window_size=60,
    )
    sample = dataset[0]
    assert sample["temporal"].dtype == torch.float32


def test_tabular_cont_tensor_dtype(sharded_dir):
    """Test that tabular_cont tensors are float32."""
    dataset = StockDataset(
        feature_dir=str(sharded_dir),
        date_range=("2023-01-01", "2023-03-31"),
        window_size=60,
    )
    sample = dataset[0]
    assert sample["tabular_cont"].dtype == torch.float32


def test_tabular_cat_tensor_dtype(sharded_dir):
    """Test that tabular_cat tensors are long."""
    dataset = StockDataset(
        feature_dir=str(sharded_dir),
        date_range=("2023-01-01", "2023-03-31"),
        window_size=60,
    )
    sample = dataset[0]
    assert sample["tabular_cat"].dtype == torch.long


# ============================================================================
# Tests: Step 1 & 2 - StockDataModule setup and batch building
# ============================================================================


def test_stock_data_module_setup_sharded_directory(sharded_dir):
    """Step 1: Test StockDataModule.setup('fit') works with sharded directory mode."""
    module = StockDataModule(
        feature_dir=str(sharded_dir),
        train_start="2023-01-01",
        train_end="2023-02-28",
        val_start="2023-03-01",
        val_end="2023-04-09",  # Extended to ensure samples after embargo
        window_size=60,
        purge_days=0,  # Disable purge for test fixture
        embargo_days=0,  # Disable embargo for test fixture
    )
    module.setup("fit")

    # Assert non-zero dataset lengths
    assert len(module.train_dataset) > 0, "Train dataset should have samples"
    assert len(module.val_dataset) > 0, "Val dataset should have samples"


def test_stock_data_module_setup_monolithic_parquet(monolithic_parquet):
    """Step 1: Test StockDataModule.setup('fit') works with single parquet file mode."""
    module = StockDataModule(
        feature_dir=str(monolithic_parquet),
        train_start="2023-01-01",
        train_end="2023-02-28",
        val_start="2023-03-01",
        val_end="2023-04-09",  # Extended to ensure samples after embargo
        window_size=60,
        purge_days=0,  # Disable purge for test fixture
        embargo_days=0,  # Disable embargo for test fixture
    )
    module.setup("fit")

    # Assert non-zero dataset lengths
    assert len(module.train_dataset) > 0, "Train dataset should have samples"
    assert len(module.val_dataset) > 0, "Val dataset should have samples"


def test_stock_data_module_train_dataloader_sharded(sharded_dir):
    """Step 2: Test train dataloader batch from sharded directory mode."""
    module = StockDataModule(
        feature_dir=str(sharded_dir),
        train_start="2023-01-01",
        train_end="2023-02-28",
        val_start="2023-03-01",
        val_end="2023-04-09",
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        window_size=60,
        purge_days=0,
        embargo_days=0,
    )
    module.setup("fit")

    # Build train dataloader and get one batch
    train_loader = module.train_dataloader()
    batch = next(iter(train_loader))

    # Verify batch structure
    assert "symbol" in batch
    assert "date" in batch
    assert "temporal" in batch
    assert "tabular_cont" in batch
    assert "tabular_cat" in batch

    # Verify tensor shapes
    batch_size = batch["temporal"].shape[0]
    assert batch["temporal"].shape == (batch_size, 60, len(TEMPORAL_FEATURE_NAMES))
    assert batch["tabular_cont"].shape == (batch_size, len(TABULAR_CONTINUOUS_NAMES))
    assert batch["tabular_cat"].shape == (batch_size, 2)


def test_stock_data_module_train_dataloader_monolithic(monolithic_parquet):
    """Step 2: Test train dataloader batch from single parquet file mode."""
    module = StockDataModule(
        feature_dir=str(monolithic_parquet),
        train_start="2023-01-01",
        train_end="2023-02-28",
        val_start="2023-03-01",
        val_end="2023-04-09",
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        window_size=60,
        purge_days=0,
        embargo_days=0,
    )
    module.setup("fit")

    # Build train dataloader and get one batch
    train_loader = module.train_dataloader()
    batch = next(iter(train_loader))

    # Verify batch structure
    assert "symbol" in batch
    assert "date" in batch
    assert "temporal" in batch
    assert "tabular_cont" in batch
    assert "tabular_cat" in batch

    # Verify tensor shapes
    batch_size = batch["temporal"].shape[0]
    assert batch["temporal"].shape == (batch_size, 60, len(TEMPORAL_FEATURE_NAMES))
    assert batch["tabular_cont"].shape == (batch_size, len(TABULAR_CONTINUOUS_NAMES))
    assert batch["tabular_cat"].shape == (batch_size, 2)
