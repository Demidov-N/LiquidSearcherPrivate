"""Two-pass cross-sectional + time-series normalization for financial features.

Standard approach used in Barra-style factor models:

  Pass 1 — Cross-sectional (per date):
      For each feature on each date, z-score across all stocks.
      Removes market-wide shocks and scale differences between dates.

  Pass 2 — Time-series (per symbol):
      For each feature for each stock, z-score across time.
      Removes persistent stock-level biases (e.g. a growth stock always
      having high momentum) so the model sees deviations from the stock's
      own history.

Winsorization is applied before each pass to prevent a handful of extreme
observations from distorting the mean/std used to normalize everything else.
"""

import logging
from typing import Optional

import numpy as np
import polars as pl
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point (called from FeatureProcessor.apply_normalization)
# ---------------------------------------------------------------------------

def two_pass_normalization(
    df: pd.DataFrame,
    feature_cols: list[str],
    date_col: str = "date",
    symbol_col: str = "symbol",
    winsor_std: float = 4.0,      # clip at ±N std before each pass
    min_stocks_per_date: int = 10, # skip cross-sectional pass if fewer stocks
    min_dates_per_symbol: int = 20,# skip time-series pass if fewer dates
) -> pd.DataFrame:
    """
    Apply two-pass normalization to a panel DataFrame.

    Args:
        df:                   Full panel (all symbols, all dates).
        feature_cols:         Columns to normalize — must all exist in df.
        date_col:             Name of the date column.
        symbol_col:           Name of the symbol column.
        winsor_std:           Winsorization threshold in standard deviations.
        min_stocks_per_date:  Minimum stocks required to compute cross-sectional stats.
        min_dates_per_symbol: Minimum dates required to compute time-series stats.

    Returns:
        DataFrame with feature_cols replaced by their normalized values.
        All other columns are unchanged.
    """
    existing = [c for c in feature_cols if c in df.columns]
    missing  = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning(f"Skipping missing columns: {missing}")
    if not existing:
        return df

    logger.info(f"Two-pass normalization on {len(existing)} features, "
                f"{df[symbol_col].nunique()} symbols, "
                f"{df[date_col].nunique()} dates")

    # Work in Polars for speed — pandas groupby z-score on 33M rows is slow
    pl_df = pl.from_pandas(df)

    # Pass 1: cross-sectional z-score per date
    logger.info("  Pass 1: cross-sectional normalization...")
    pl_df = _cross_sectional_normalize(
        pl_df, existing, date_col, min_stocks_per_date, winsor_std
    )

    # Pass 2: time-series z-score per symbol
    logger.info("  Pass 2: time-series normalization...")
    pl_df = _timeseries_normalize(
        pl_df, existing, symbol_col, min_dates_per_symbol, winsor_std
    )

    logger.info("  Normalization complete")
    return pl_df.to_pandas()


# ---------------------------------------------------------------------------
# Pass 1 — Cross-sectional
# ---------------------------------------------------------------------------

def _cross_sectional_normalize(
    df: pl.DataFrame,
    cols: list[str],
    date_col: str,
    min_stocks: int,
    winsor_std: float,
) -> pl.DataFrame:
    """
    For each (date, feature): winsorize → subtract cross-sectional mean → divide by std.

    Stocks with fewer than `min_stocks` observations on a date are left as NaN
    rather than being normalized against a meaningless single-stock mean.
    """
    exprs = []
    for col in cols:
        # Step 1: winsorize relative to cross-sectional distribution on each date
        mu_cs  = pl.col(col).mean().over(date_col)
        sd_cs  = pl.col(col).std().over(date_col)
        lo     = mu_cs - winsor_std * sd_cs
        hi     = mu_cs + winsor_std * sd_cs
        winsor = pl.col(col).clip(lo, hi)

        # Step 2: z-score using winsorized distribution
        mu_w  = winsor.mean().over(date_col)
        sd_w  = winsor.std().over(date_col)
        count = pl.col(col).count().over(date_col)

        z = pl.when(count >= min_stocks)  \
              .then((winsor - mu_w) / sd_w) \
              .otherwise(pl.lit(None))      \
              .alias(col)
        exprs.append(z)

    return df.with_columns(exprs)


# ---------------------------------------------------------------------------
# Pass 2 — Time-series
# ---------------------------------------------------------------------------

def _timeseries_normalize(
    df: pl.DataFrame,
    cols: list[str],
    symbol_col: str,
    min_dates: int,
    winsor_std: float,
) -> pl.DataFrame:
    """
    For each (symbol, feature): winsorize → subtract time-series mean → divide by std.

    Symbols with fewer than `min_dates` observations are left as-is rather
    than being normalized against a near-empty history.
    """
    exprs = []
    for col in cols:
        # Step 1: winsorize relative to this stock's own time-series distribution
        mu_ts  = pl.col(col).mean().over(symbol_col)
        sd_ts  = pl.col(col).std().over(symbol_col)
        lo     = mu_ts - winsor_std * sd_ts
        hi     = mu_ts + winsor_std * sd_ts
        winsor = pl.col(col).clip(lo, hi)

        # Step 2: z-score using winsorized distribution
        mu_w  = winsor.mean().over(symbol_col)
        sd_w  = winsor.std().over(symbol_col)
        count = pl.col(col).count().over(symbol_col)

        z = pl.when(count >= min_dates)   \
              .then((winsor - mu_w) / sd_w) \
              .otherwise(pl.col(col))       \
              .alias(col)
        exprs.append(z)

    return df.with_columns(exprs)


# ---------------------------------------------------------------------------
# Diagnostic utility
# ---------------------------------------------------------------------------

def normalization_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    feature_cols: list[str],
    date_col: str = "date",
    n_dates: int = 5,
) -> pd.DataFrame:
    """
    Compare cross-sectional mean/std/skew before and after normalization
    for a random sample of dates.  Useful for sanity-checking.

    Returns a DataFrame with one row per (date, feature, pass).
    """
    sample_dates = (
        df_before[date_col].drop_duplicates()
                           .sample(min(n_dates, df_before[date_col].nunique()),
                                   random_state=42)
                           .tolist()
    )

    records = []
    for d in sample_dates:
        for col in feature_cols:
            if col not in df_before.columns:
                continue
            before_slice = df_before.loc[df_before[date_col] == d, col].dropna()
            after_slice  = df_after.loc[df_after[date_col]  == d, col].dropna()
            records.append({
                "date":    d,
                "feature": col,
                "pass":    "before",
                "mean":    before_slice.mean(),
                "std":     before_slice.std(),
                "skew":    before_slice.skew(),
                "p1":      before_slice.quantile(0.01),
                "p99":     before_slice.quantile(0.99),
            })
            records.append({
                "date":    d,
                "feature": col,
                "pass":    "after",
                "mean":    after_slice.mean(),
                "std":     after_slice.std(),
                "skew":    after_slice.skew(),
                "p1":      after_slice.quantile(0.01),
                "p99":     after_slice.quantile(0.99),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Standalone winsorize — kept as a public export so any existing
# imports in src/features/__init__.py or elsewhere continue to work.
# ---------------------------------------------------------------------------

def winsorize(
    series: pd.Series,
    std_threshold: float = 4.0,
) -> pd.Series:
    """
    Clip a pandas Series at ±std_threshold standard deviations from its mean.

    This is the same operation used internally by two_pass_normalization.
    Exposed here so other modules can import it directly if needed.
    """
    mu = series.mean()
    sd = series.std()
    if sd == 0 or np.isnan(sd):
        return series
    return series.clip(lower=mu - std_threshold * sd, upper=mu + std_threshold * sd)


def cross_sectional_zscore(
    df: pd.DataFrame,
    feature_cols: list[str],
    date_col: str = "date",
    winsor_std: float = 4.0,
    min_stocks: int = 10,
) -> pd.DataFrame:
    """
    Single cross-sectional z-score pass (no time-series pass).

    Useful when you only want to normalize within each date without
    removing persistent stock-level biases.
    """
    pl_df = pl.from_pandas(df)
    existing = [c for c in feature_cols if c in df.columns]
    pl_df = _cross_sectional_normalize(pl_df, existing, date_col, min_stocks, winsor_std)
    return pl_df.to_pandas()


def rank_normalize(
    df: pd.DataFrame,
    feature_cols: list[str],
    date_col: str = "date",
    min_stocks: int = 10,
) -> pd.DataFrame:
    """
    Cross-sectional rank normalization: on each date, replace raw values
    with uniform scores in [-0.5, +0.5] based on rank.

    More robust than z-score for heavily skewed features like market_cap
    or volume, where winsorization still leaves fat tails.

    Formula: rank / (n + 1) - 0.5  →  maps to (-0.5, +0.5)
    """
    pl_df = pl.from_pandas(df)
    existing = [c for c in feature_cols if c in df.columns]

    exprs = []
    for col in existing:
        count = pl.col(col).count().over(date_col)
        ranked = (
            pl.when(count >= min_stocks)
              .then(
                  pl.col(col).rank(method="average").over(date_col)
                  / (count + 1)
                  - 0.5
              )
              .otherwise(pl.lit(None))
              .alias(col)
        )
        exprs.append(ranked)

    return pl_df.with_columns(exprs).to_pandas()