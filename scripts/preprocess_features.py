"""Preprocessing script for data collection and feature computation.

This script:
1. Loads symbols in batches from WRDS
2. Fetches CRSP market returns once (needed for local beta computation)
3. Fetches prices, betas, fundamentals, GICS per batch
4. Writes each batch as a separate parquet file (no read-concat-write)
5. Merges all batch files into one final parquet at the end
6. Applies two-pass cross-sectional normalization on the full dataset

Usage:
    python -m scripts.preprocess_features \
        --start-date 2010-01-01 --end-date 2024-12-31 \
        --universe all_crsp --output data/processed/all_features.parquet
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import polars as pl

from src.config.settings import get_settings
from src.data.credentials import validate_and_exit
from src.data.universe import SymbolUniverse
from src.data.wrds_loader import WRDSDataLoader
from src.features.processor import FeatureProcessor
from src.utils.memory import get_recommended_batch_size, print_memory_status

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TEMPORAL_FEATURE_NAMES = [
    "z_close", "z_volume", "ma_ratio_5d", "ma_ratio_10d", "ma_ratio_20d",
    "realized_vol_20d", "realized_vol_60d", "mom_1m", "mom_3m", "mom_6m",
    "mom_12m", "mom_12_1m", "log_ret_cum",
]
TABULAR_CONTINUOUS_NAMES = [
    "beta", "idiosyncratic_vol", "roe", "roa", "debt_to_equity",
    "price_to_book", "price_to_earnings", "market_cap", "dividend_yield",
    "revenue", "net_income", "total_assets", "cash", "operating_margin",
    "profit_margin",
]


# ---------------------------------------------------------------------------
# Market returns — fetched once, shared across all batches
# ---------------------------------------------------------------------------

def fetch_market_returns(loader: WRDSDataLoader, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch CRSP value-weighted market return (vwretd) from crsp.dsi.

    This is fetched ONCE before the symbol batch loop and passed into
    every call to processor.process_batch().  It must not be fetched
    per-batch because (a) it's the same series for every symbol and
    (b) re-fetching 15 years of daily data 200 times is wasteful.
    """
    logger.info("Fetching CRSP market returns (vwretd) from crsp.dsi...")
    df = loader.conn.raw_sql(f"""
        SELECT date, vwretd
        FROM crsp.dsi
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
    """)
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"  Got {len(df)} market return rows")
    return df


# ---------------------------------------------------------------------------
# Per-batch processing
# ---------------------------------------------------------------------------

def process_symbol_batch(
    symbols: List[str],
    start_date: str,
    end_date: str,
    loader: WRDSDataLoader,
    processor: FeatureProcessor,
    market_returns_df: pd.DataFrame,
    skip_betas: bool = False,
) -> pd.DataFrame:
    """Fetch and compute features for one batch of symbols."""
    logger.info(f"Fetching prices for {len(symbols)} symbols...")
    prices_df = loader.fetch_prices(symbols, start_date, end_date)  # updated API name

    if prices_df.empty:
        logger.warning(f"No price data returned for batch: {symbols[:5]}...")
        return pd.DataFrame()

    logger.info(f"  {len(prices_df):,} price rows")

    # Betas — WRDS Beta Suite if available, else computed locally from vwretd
    betas_df = None
    if not skip_betas:
        try:
            betas_df = loader.fetch_betas(symbols, start_date, end_date, estper=60)  # updated API name
            if not betas_df.empty:
                logger.info(f"  {len(betas_df):,} pre-computed beta rows")
        except Exception as exc:
            logger.warning(f"  Beta Suite fetch failed ({exc}), will compute locally")

    # WRDS pre-computed ratios (replaces raw fundq fetch + manual derivation)
    fundamentals_df = None
    try:
        fundamentals_df = loader.fetch_wrds_ratios(symbols, start_date, end_date)
        if fundamentals_df is not None and not fundamentals_df.empty:
            logger.info(f"  {len(fundamentals_df):,} wrds_ratios rows")
    except Exception as exc:
        logger.warning(f"  wrds_ratios fetch failed: {exc}")

    # GICS codes
    gics_df = None
    try:
        gics_df = loader.fetch_gics(symbols)  # updated API name
        if gics_df is not None and not gics_df.empty:
            logger.info(f"  {len(gics_df):,} GICS rows")
    except Exception as exc:
        logger.warning(f"  GICS fetch failed: {exc}")

    # Compute features — pass market_returns_df so local betas use vwretd
    features_df = processor.process_batch(
        prices_df=prices_df,
        market_returns_df=market_returns_df,   # <-- was missing entirely before
        betas_df=betas_df,
        fundamentals_df=fundamentals_df,
        gics_df=gics_df,
    )

    logger.info(f"  → {len(features_df):,} feature rows, {len(features_df.columns)} columns")
    return features_df


# ---------------------------------------------------------------------------
# Incremental parquet writer (one file per batch — no read-concat-write)
# ---------------------------------------------------------------------------

def write_batch(df: pd.DataFrame, batch_dir: Path, batch_num: int) -> Path:
    """Write a single batch to its own numbered parquet file."""
    batch_dir.mkdir(parents=True, exist_ok=True)
    path = batch_dir / f"batch_{batch_num:04d}.parquet"
    df.to_parquet(path, index=False, compression="snappy")
    return path


def merge_batches(batch_dir: Path, output_path: Path) -> None:
    """
    Concatenate all batch files into one final parquet using Polars.

    Polars scan_parquet + sink_parquet streams the merge without loading
    everything into RAM at once.
    """
    batch_files = sorted(batch_dir.glob("batch_*.parquet"))
    if not batch_files:
        logger.error("No batch files found to merge")
        return

    logger.info(f"Merging {len(batch_files)} batch files → {output_path}")

    # Read all batches and find the union of all columns
    frames = []
    for f in batch_files:
        df = pl.read_parquet(str(f))
        if df["date"].dtype != pl.Datetime("us"):
            df = df.with_columns(pl.col("date").cast(pl.Datetime("us")))
        frames.append(df)

    # Build union schema — batches missing a column get it filled with null
    all_cols = []
    seen = set()
    for df in frames:
        for col in df.columns:
            if col not in seen:
                all_cols.append((col, df[col].dtype))
                seen.add(col)

    aligned = []
    for df in frames:
        missing = [(c, t) for c, t in all_cols if c not in df.columns]
        if missing:
            df = df.with_columns([
                pl.lit(None).cast(t).alias(c) for c, t in missing
            ])
        # Reorder to consistent column order
        df = df.select([c for c, _ in all_cols])
        aligned.append(df)

    (
        pl.concat(aligned)
          .sort(["symbol", "date"])
          .write_parquet(str(output_path), compression="snappy")
    )
    logger.info("Merge complete")


# ---------------------------------------------------------------------------
# Universe helpers
# ---------------------------------------------------------------------------

def get_universe_symbols(
    universe_type: str,
    start_date: str,
    end_date: str,
    loader: Optional[WRDSDataLoader] = None,
) -> List[str]:
    if universe_type == "hardcoded":
        symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM",
            "JNJ", "V", "PG", "UNH", "HD", "MA", "BAC", "ABBV", "PFE",
            "KO", "AVGO", "PEP", "TMO", "COST", "DIS", "ABT", "ADBE",
            "WMT", "MRK", "CSCO", "ACN", "VZ", "NKE", "TXN", "CMCSA",
        ]
        logger.info(f"Using {len(symbols)} hardcoded symbols")
        return symbols

    if loader is None:
        logger.warning("WRDS loader required for index universe; falling back to hardcoded")
        return get_universe_symbols("hardcoded", start_date, end_date)

    # Date-scoped query — avoids returning delisted/recycled tickers outside
    # the processing window (same bug that affected the original WRDS loader)
    logger.info(f"Fetching {universe_type} universe from CRSP (date-scoped)...")
    df = loader.conn.raw_sql(f"""
        SELECT DISTINCT ticker
        FROM crsp.dsenames
        WHERE shrcd IN (10, 11)
          AND namedt     <= '{end_date}'
          AND (nameendt >= '{start_date}' OR nameendt IS NULL)
          AND ticker IS NOT NULL
          AND ticker != ''
        ORDER BY ticker
    """)
    symbols = df["ticker"].dropna().unique().tolist()
    logger.info(f"Universe size: {len(symbols)} symbols")
    return symbols


# ---------------------------------------------------------------------------
# Normalization (runs on the full merged dataset)
# ---------------------------------------------------------------------------

def apply_normalization(output_path: Path, processor: FeatureProcessor) -> None:
    """
    Apply two-pass cross-sectional normalization to the full feature file.

    This MUST run after all batches are merged.  Per-batch normalization
    would produce different z-score scales for each batch and make the
    features non-comparable across time.
    """
    logger.info("Loading merged features for normalization...")
    df = pd.read_parquet(output_path)

    feature_groups = {
        "temporal":  TEMPORAL_FEATURE_NAMES,
        "tabular":   TABULAR_CONTINUOUS_NAMES,
    }

    logger.info("Applying two-pass cross-sectional normalization...")
    df = processor.apply_normalization(df, feature_groups)

    logger.info(f"Writing normalized features → {output_path}")
    df.to_parquet(output_path, index=False, compression="snappy")
    logger.info("Normalization complete")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess features for stock universe")
    parser.add_argument("--start-date",   default="2010-01-01")
    parser.add_argument("--end-date",     default="2024-12-31")
    parser.add_argument("--batch-size",   type=int, default=None)
    parser.add_argument("--output",       default="data/processed/all_features.parquet")
    parser.add_argument("--skip-betas",   action="store_true")
    parser.add_argument("--skip-normalize", action="store_true",
                        help="Skip normalization step (useful for debugging)")
    parser.add_argument("--universe",     default="hardcoded",
                        choices=["hardcoded", "all_crsp"])
    parser.add_argument("--resume",       action="store_true",
                        help="Skip batches whose output file already exists")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    print_memory_status()

    batch_size = args.batch_size or get_recommended_batch_size(safety_factor=0.5)
    logger.info(f"Batch size: {batch_size}")

    validate_and_exit()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch_dir   = output_path.parent / "batches"

    processor = FeatureProcessor()

    with WRDSDataLoader() as loader:
        symbols = get_universe_symbols(
            args.universe, args.start_date, args.end_date, loader
        )
        universe     = SymbolUniverse(symbols, batch_size=batch_size)
        total_batches = len(universe)

        # Fetch market returns ONCE — same series for every symbol batch
        market_returns_df = fetch_market_returns(loader, args.start_date, args.end_date)

        logger.info(f"Processing {len(symbols)} symbols in {total_batches} batches")

        for batch_num, symbol_batch in enumerate(universe.batches(desc="Processing"), 1):

            batch_path = batch_dir / f"batch_{batch_num:04d}.parquet"

            # Resume support — skip batches already written
            if args.resume and batch_path.exists():
                logger.info(f"Batch {batch_num}/{total_batches}: already exists, skipping")
                continue

            logger.info(f"\nBatch {batch_num}/{total_batches}: {len(symbol_batch)} symbols")
            try:
                features_df = process_symbol_batch(
                    symbols          = symbol_batch,
                    start_date       = args.start_date,
                    end_date         = args.end_date,
                    loader           = loader,
                    processor        = processor,
                    market_returns_df= market_returns_df,
                    skip_betas       = args.skip_betas,
                )
                if not features_df.empty:
                    write_batch(features_df, batch_dir, batch_num)
            except Exception as exc:
                logger.error(f"Batch {batch_num} failed: {exc}", exc_info=True)
                # Continue — don't abort the entire pipeline for one bad batch

    # --- Post-processing (outside WRDS context — connection no longer needed) ---

    merge_batches(batch_dir, output_path)

    if not args.skip_normalize:
        apply_normalization(output_path, processor)

    # Final stats
    final = pl.scan_parquet(str(output_path))
    n_rows = final.select(pl.len()).collect().item()
    logger.info(f"\nDone. {n_rows:,} rows → {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()