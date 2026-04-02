"""
Shard a monolithic features parquet into one file per symbol.

Strategy: Polars sink_parquet with partition_by — single sequential read,
streaming writes, never loads the full dataset into RAM.

Usage:
    python -m scripts.shard_by_symbol \
        --src data/processed/all_features.parquet \
        --out data/processed/by_symbol

    # Resume-safe (skip already-written symbols):
    python -m scripts.shard_by_symbol \
        --src data/processed/all_features.parquet \
        --out data/processed/by_symbol \
        --resume

    # Verify output after sharding:
    python -m scripts.shard_by_symbol --verify \
        --src data/processed/all_features.parquet \
        --out data/processed/by_symbol
"""

import argparse
import logging
import time
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Module-level worker function (must be picklable for multiprocessing)
# ---------------------------------------------------------------------------

def _write_arrow(args: tuple) -> None:
    """Write a PyArrow table to parquet. Called in worker processes."""
    import pyarrow.parquet as pq
    arrow_table, path = args
    pq.write_table(arrow_table, path, compression="snappy")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core sharding
# ---------------------------------------------------------------------------

def shard(src: Path, out: Path, resume: bool = False) -> int:
    """
    Shard src into one {symbol}_features.parquet per symbol in out/.

    Returns the number of symbols written.
    """
    out.mkdir(parents=True, exist_ok=True)

    if resume:
        already_done = {f.stem.replace("_features", "")
                        for f in out.glob("*_features.parquet")}
        logger.info(f"Resume mode: {len(already_done)} symbols already sharded, skipping them")
    else:
        already_done = set()

    # Discover symbols first (cheap — only reads unique values)
    logger.info(f"Scanning {src} for symbol list...")
    lf = pl.scan_parquet(str(src))
    all_symbols = (
        lf.select(pl.col("symbol").unique())
          .collect()["symbol"]
          .to_list()
    )
    symbols_to_write = [s for s in all_symbols if s not in already_done]
    logger.info(f"Total symbols: {len(all_symbols)} — to write: {len(symbols_to_write)}")

    if not symbols_to_write:
        logger.info("Nothing to do.")
        return 0

    if resume and already_done:
        # Filter to only unwritten symbols before sinking
        logger.info("Filtering to unwritten symbols...")
        lf = lf.filter(pl.col("symbol").is_in(symbols_to_write))

    import multiprocessing as mp
    import os

    t0 = time.time()
    logger.info("Loading and partitioning (one pass)...")

    df = lf.sort(["symbol", "date"]).collect()
    partitions = df.partition_by("symbol", as_dict=True)
    n = len(partitions)

    # Build write tasks — list of (bytes, path) tuples
    # Serialize to bytes in main process (Polars DataFrames aren't picklable)
    logger.info(f"Serializing {n} partitions...")
    tasks = []
    for sym, part in partitions.items():
        sym_str = sym[0] if isinstance(sym, tuple) else sym
        path = str(out / f"{sym_str}_features.parquet")
        tasks.append((part.to_arrow(), path))

    n_workers = 4 #min(os.cpu_count() or 4, 8)
    logger.info(f"Writing {n} files with {n_workers} workers...")

    with mp.Pool(processes=n_workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_write_arrow, tasks, chunksize=50)):
            if (i + 1) % 500 == 0:
                logger.info(f"  {i+1}/{n}")

    elapsed = time.time() - t0
    logger.info(f"Done — {n} symbol files written in {elapsed:.1f}s "
                f"({elapsed/max(n,1)*1000:.0f}ms/symbol)")
    return n


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(src: Path, out: Path) -> None:
    """
    Sanity-check: total row counts and date ranges match between
    the source parquet and the sum of all shards.
    """
    logger.info("Verifying shard integrity...")

    src_stats = (
        pl.scan_parquet(str(src))
          .select([
              pl.len().alias("rows"),
              pl.col("symbol").n_unique().alias("symbols"),
              pl.col("date").min().alias("date_min"),
              pl.col("date").max().alias("date_max"),
          ])
          .collect()
    )

    shard_files = sorted(out.glob("*_features.parquet"))
    shard_stats = (
        pl.scan_parquet([str(f) for f in shard_files])
          .select([
              pl.len().alias("rows"),
              pl.col("symbol").n_unique().alias("symbols"),
              pl.col("date").min().alias("date_min"),
              pl.col("date").max().alias("date_max"),
          ])
          .collect()
    )

    print("\n── Source ──────────────────────────────────")
    print(f"  Rows:        {src_stats['rows'][0]:,}")
    print(f"  Symbols:     {src_stats['symbols'][0]:,}")
    print(f"  Date range:  {src_stats['date_min'][0]} → {src_stats['date_max'][0]}")

    print("\n── Shards ──────────────────────────────────")
    print(f"  Files:       {len(shard_files):,}")
    print(f"  Rows:        {shard_stats['rows'][0]:,}")
    print(f"  Symbols:     {shard_stats['symbols'][0]:,}")
    print(f"  Date range:  {shard_stats['date_min'][0]} → {shard_stats['date_max'][0]}")

    row_match = src_stats['rows'][0] == shard_stats['rows'][0]
    sym_match = src_stats['symbols'][0] == shard_stats['symbols'][0]
    print("\n── Check ───────────────────────────────────")
    print(f"  Row count match:    {'✓' if row_match else '✗ MISMATCH'}")
    print(f"  Symbol count match: {'✓' if sym_match else '✗ MISMATCH'}")

    if row_match and sym_match:
        print("\n✓ Shards are consistent with source.\n")
    else:
        print("\n✗ Integrity check failed — re-run sharding without --resume.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Shard features parquet by symbol")
    parser.add_argument("--src",    default="data/processed/all_features.parquet",
                        help="Source monolithic parquet file")
    parser.add_argument("--out",    default="data/processed/by_symbol",
                        help="Output directory for per-symbol shards")
    parser.add_argument("--resume", action="store_true",
                        help="Skip symbols whose shard file already exists")
    parser.add_argument("--verify", action="store_true",
                        help="Verify shard integrity instead of sharding")
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    if not src.exists():
        logger.error(f"Source file not found: {src}")
        return

    if args.verify:
        verify(src, out)
    else:
        shard(src, out, resume=args.resume)


if __name__ == "__main__":
    main()