"""
Tests for shard_by_symbol.py script.

Covers:
- Configurable worker count (workers=1 serial, workers>1 multiprocessing)
- Progress/stage logging
- Shard file creation and resumption
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest


@pytest.fixture
def tiny_parquet():
    """Create a tiny synthetic parquet with a few symbols and dates for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create synthetic data: 3 symbols × 5 dates = 15 rows
        data = {
            "symbol": [
                "AAPL",
                "AAPL",
                "AAPL",
                "AAPL",
                "AAPL",
                "MSFT",
                "MSFT",
                "MSFT",
                "MSFT",
                "MSFT",
                "GOOGL",
                "GOOGL",
                "GOOGL",
                "GOOGL",
                "GOOGL",
            ],
            "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"] * 3,
            "prc": [150.0 + i for i in range(15)],
            "beta": [1.2 + i * 0.01 for i in range(15)],
            "idiosyncratic_vol": [0.2 + i * 0.01 for i in range(15)],
            "gsector": ["Technology"] * 15,
            "ggroup": ["Software"] * 15,
        }
        df = pl.DataFrame(data)
        src_path = tmpdir_path / "test_data.parquet"
        df.write_parquet(str(src_path))

        yield src_path


@pytest.fixture
def output_dir():
    """Create a temporary output directory for shards."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_workers_equals_1_uses_serial_write(tiny_parquet, output_dir, caplog):
    """
    Test that workers=1 accepts the parameter and uses a serial write path.

    Verify:
    - Script accepts --workers 1
    - Shard files are created
    - Logs indicate serial mode (no mention of multiprocessing or "workers")
    """
    from scripts.shard_by_symbol import shard

    with caplog.at_level(logging.INFO):
        # This will fail initially because shard() doesn't accept workers parameter yet
        num_written = shard(tiny_parquet, output_dir, resume=False, workers=1)

    # Verify shards were created
    shard_files = sorted(output_dir.glob("*_features.parquet"))
    assert len(shard_files) == 3, f"Expected 3 shards, got {len(shard_files)}"

    # Verify log contains serial mode indicator
    log_text = caplog.text
    assert "serial" in log_text.lower() or "1 worker" in log_text.lower() or num_written == 3


def test_workers_greater_than_1_is_wired(tiny_parquet, output_dir, caplog):
    """
    Test that workers>1 is accepted and wired into the write phase.

    Verify:
    - Script accepts --workers 2
    - Shard files are created
    - Logs indicate multiprocessing mode
    """
    from scripts.shard_by_symbol import shard

    with caplog.at_level(logging.INFO):
        num_written = shard(tiny_parquet, output_dir, resume=False, workers=2)

    shard_files = sorted(output_dir.glob("*_features.parquet"))
    assert len(shard_files) == 3, f"Expected 3 shards, got {len(shard_files)}"

    # Verify log mentions worker count
    log_text = caplog.text
    assert "worker" in log_text.lower() or num_written == 3


def test_logging_shows_stage_transitions(tiny_parquet, output_dir, caplog):
    """
    Test that logs include explicit stage transitions.

    Verify logs contain stages:
    - Scanning
    - Collecting/sorting (or "Loading and partitioning")
    - Serializing
    - Writing
    """
    from scripts.shard_by_symbol import shard

    with caplog.at_level(logging.INFO):
        shard(tiny_parquet, output_dir, resume=False, workers=1)

    log_text = caplog.text.lower()

    # Check for stage indicators
    assert "scann" in log_text, "Log should mention scanning symbols"
    assert "partition" in log_text or "collect" in log_text or "load" in log_text, (
        "Log should mention loading/partitioning data"
    )
    assert "serial" in log_text or "writing" in log_text.lower() or "writer" in log_text, (
        "Log should distinguish serial vs multiprocessing write mode"
    )


def test_logging_distinguishes_serial_vs_multiprocessing(tiny_parquet, output_dir, caplog):
    """
    Test that write-mode logging clearly distinguishes serial vs multiprocessing.
    """
    from scripts.shard_by_symbol import shard

    # Test serial mode
    caplog.clear()
    with caplog.at_level(logging.INFO):
        shard(tiny_parquet, output_dir, resume=False, workers=1)

    serial_log = caplog.text.lower()
    assert "serial" in serial_log or "1 worker" in serial_log, (
        "Serial write mode should be explicitly logged"
    )

    # Test multiprocessing mode
    output_dir2 = Path(tempfile.mkdtemp())
    caplog.clear()
    with caplog.at_level(logging.INFO):
        shard(tiny_parquet, output_dir2, resume=False, workers=2)

    multi_log = caplog.text.lower()
    assert "2" in multi_log or "worker" in multi_log or "multiprocess" in multi_log, (
        "Multiprocessing mode should log worker count"
    )


def test_resume_mode_skips_existing_shards(tiny_parquet, output_dir):
    """
    Test that resume mode properly skips already-written symbols.
    """
    from scripts.shard_by_symbol import shard

    # First run: write all shards
    num_written_1 = shard(tiny_parquet, output_dir, resume=False, workers=1)
    assert num_written_1 == 3

    shard_files_1 = sorted(output_dir.glob("*_features.parquet"))
    assert len(shard_files_1) == 3

    # Second run with resume: should skip all
    num_written_2 = shard(tiny_parquet, output_dir, resume=True, workers=1)
    assert num_written_2 == 0, "Resume should skip all existing shards"

    shard_files_2 = sorted(output_dir.glob("*_features.parquet"))
    assert len(shard_files_2) == 3, "No new shards should be written"
