"""Feature processor with Polars for efficient computation."""

import logging
from typing import Optional

import numpy as np
import polars as pl
import pandas as pd

from src.features.normalization import two_pass_normalization

logger = logging.getLogger(__name__)


class FeatureProcessor:
    """Process raw data into features using Polars for efficiency.

    Features computed:
    - G1: Market risk (pre-computed betas when available, else local OLS)
    - G2: Volatility (realized vol, idiosyncratic vol)
    - G3: Momentum (1m, 3m, 6m, 12m, 12_1m, cumulative log return)
    - G4: Valuation (P/E, P/B, ROE from fundamentals)
    - G5: OHLCV technicals (z-scores, MA ratios)
    - G6: Sector (GICS codes)
    """

    def process_batch(
        self,
        prices_df: pd.DataFrame,
        market_returns_df: Optional[pd.DataFrame] = None,
        betas_df: Optional[pd.DataFrame] = None,
        fundamentals_df: Optional[pd.DataFrame] = None,
        gics_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Process a batch of raw data into features.

        Args:
            prices_df:          CRSP daily prices (symbol, date, prc, vol, ret)
            market_returns_df:  CRSP market returns (date, vwretd) — required for
                                local beta computation if betas_df is not supplied
            betas_df:           Pre-computed betas from WRDS Beta Suite (optional)
            fundamentals_df:    Compustat quarterly fundamentals (optional)
            gics_df:            GICS sector codes (optional)
        """
        logger.info(f"Processing batch: {len(prices_df):,} price rows")

        pl_df = pl.from_pandas(prices_df)
        pl_df = pl_df.sort(["symbol", "date"])

        pl_df = self._compute_ohlcv_features(pl_df)
        pl_df = self._compute_momentum_features(pl_df)
        pl_df = self._compute_volatility_features(pl_df)

        # Betas
        if betas_df is not None and not betas_df.empty:
            logger.info("Merging pre-computed betas...")
            pl_df = self._merge_betas_polars(pl_df, pl.from_pandas(betas_df))
        elif market_returns_df is not None and not market_returns_df.empty:
            logger.info("Computing betas locally from CRSP vwretd...")
            pl_mkt = pl.from_pandas(market_returns_df)
            pl_df = self._compute_betas_polars(pl_df, pl_mkt, window=60)
        else:
            logger.warning(
                "No beta source provided. Pass either betas_df (WRDS Beta Suite) "
                "or market_returns_df (CRSP vwretd) to process_batch()."
            )

        # Fundamentals and GICS need merge_asof — drop to pandas once
        features_df = pl_df.to_pandas()

        if fundamentals_df is not None and not fundamentals_df.empty:
            logger.info("Merging WRDS ratios (merge_asof)...")
            features_df = self._merge_wrds_ratios(features_df, fundamentals_df)

        if gics_df is not None and not gics_df.empty:
            logger.info("Merging GICS codes...")
            features_df = self._merge_gics(features_df, gics_df)

        return features_df

    # ------------------------------------------------------------------
    # G5: OHLCV features
    # ------------------------------------------------------------------

    def _compute_ohlcv_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """z_close, z_volume, ma_ratio_5d/10d/20d."""

        # Use CRSP 'ret' (total return incl. dividends) if available,
        # otherwise fall back to price-derived simple return.
        if "ret" not in df.columns:
            df = df.with_columns(
                (pl.col("prc") / pl.col("prc").shift(1).over("symbol") - 1)
                .alias("ret_1d")
            )
        else:
            df = df.with_columns(pl.col("ret").alias("ret_1d"))

        # Rolling z-score of return (252-day window)
        df = df.with_columns([
            pl.col("ret_1d")
              .rolling_mean(window_size=252, min_samples=60)
              .over("symbol")
              .alias("_ret_mu"),
            pl.col("ret_1d")
              .rolling_std(window_size=252, min_samples=60)
              .over("symbol")
              .alias("_ret_sd"),
        ]).with_columns(
            ((pl.col("ret_1d") - pl.col("_ret_mu")) / pl.col("_ret_sd"))
            .alias("z_close")
        ).drop(["_ret_mu", "_ret_sd"])

        # Rolling z-score of volume change
        df = df.with_columns(
            (pl.col("vol") / pl.col("vol").shift(1).over("symbol") - 1)
            .alias("_vol_chg")
        ).with_columns([
            pl.col("_vol_chg")
              .rolling_mean(window_size=252, min_samples=60)
              .over("symbol")
              .alias("_vol_mu"),
            pl.col("_vol_chg")
              .rolling_std(window_size=252, min_samples=60)
              .over("symbol")
              .alias("_vol_sd"),
        ]).with_columns(
            ((pl.col("_vol_chg") - pl.col("_vol_mu")) / pl.col("_vol_sd"))
            .alias("z_volume")
        ).drop(["_vol_chg", "_vol_mu", "_vol_sd"])

        # Moving average ratios
        for w in [5, 10, 20]:
            df = df.with_columns(
                pl.col("prc")
                  .rolling_mean(window_size=w, min_samples=max(1, w // 2))
                  .over("symbol")
                  .alias(f"_ma{w}")
            ).with_columns(
                (pl.col("prc") / pl.col(f"_ma{w}") - 1).alias(f"ma_ratio_{w}d")
            ).drop(f"_ma{w}")

        return df

    # ------------------------------------------------------------------
    # G3: Momentum features
    # ------------------------------------------------------------------

    def _compute_momentum_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """mom_1m/3m/6m/12m, mom_12_1m, log_ret_cum."""

        df = df.with_columns(
            (1 + pl.col("ret_1d")).log().alias("_log_ret")
        )

        windows = {"mom_1m": 21, "mom_3m": 63, "mom_6m": 126, "mom_12m": 252}

        for name, w in windows.items():
            df = df.with_columns(
                (
                    pl.col("_log_ret")
                      .rolling_sum(window_size=w, min_samples=w // 2)
                      .over("symbol")
                      .exp() - 1
                ).alias(name)
            )

        # 12-1 momentum: skip most recent month to avoid short-term reversal
        df = df.with_columns(
            ((1 + pl.col("mom_12m")) / (1 + pl.col("mom_1m")) - 1)
            .alias("mom_12_1m")
        )

        # Cumulative log return from start of history (feature index 12)
        df = df.with_columns(
            pl.col("_log_ret")
              .cum_sum()
              .over("symbol")
              .alias("log_ret_cum")
        )

        return df.drop("_log_ret")

    # ------------------------------------------------------------------
    # G2: Volatility features
    # ------------------------------------------------------------------

    def _compute_volatility_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """realized_vol_20d, realized_vol_60d (annualised)."""
        for w, label in [(20, "20d"), (60, "60d")]:
            df = df.with_columns(
                (
                    pl.col("ret_1d")
                      .rolling_std(window_size=w, min_samples=w // 2)
                      .over("symbol")
                    * np.sqrt(252)
                ).alias(f"realized_vol_{label}")
            )
        return df

    # ------------------------------------------------------------------
    # G1 (local): Beta from CRSP value-weighted market return
    # ------------------------------------------------------------------

    def _compute_betas_polars(
        self,
        df: pl.DataFrame,
        market_df: pl.DataFrame,   # columns: date, vwretd
        window: int = 60,
    ) -> pl.DataFrame:
        """
        Rolling OLS beta = cov(r_i, r_m) / var(r_m).

        Uses CRSP vwretd — NOT cross-sectional mean which is equal-weighted
        and contaminated by your universe composition.

        cov(x,y) = E[xy] - E[x]*E[y]  (numerically stable, one Polars pass)
        """
        # Cast both date columns to pl.Date to ensure type match.
        # pl_df.date may be Utf8/str (from CRSP via pandas str),
        # market_df.date may be datetime[ns] (from pd.to_datetime).
        df         = df.with_columns(pl.col("date").cast(pl.Date))
        market_df  = market_df.with_columns(pl.col("date").cast(pl.Date))

        df = df.join(
            market_df.select(["date", "vwretd"]).rename({"vwretd": "mkt_ret"}),
            on="date",
            how="left",
        )

        df = df.with_columns([
            pl.col("ret_1d")
              .rolling_mean(window_size=window, min_samples=window // 2)
              .over("symbol")
              .alias("_ri_mean"),
            pl.col("mkt_ret")
              .rolling_mean(window_size=window, min_samples=window // 2)
              .over("symbol")
              .alias("_rm_mean"),
        ])

        df = df.with_columns([
            (pl.col("ret_1d") * pl.col("mkt_ret"))
              .rolling_mean(window_size=window, min_samples=window // 2)
              .over("symbol")
              .alias("_cov_num"),
            (pl.col("mkt_ret") ** 2)
              .rolling_mean(window_size=window, min_samples=window // 2)
              .over("symbol")
              .alias("_var_num"),
        ])

        df = df.with_columns([
            (pl.col("_cov_num") - pl.col("_ri_mean") * pl.col("_rm_mean"))
              .alias("_cov"),
            (pl.col("_var_num") - pl.col("_rm_mean") ** 2)
              .alias("_var"),
        ]).with_columns(
            (pl.col("_cov") / pl.col("_var")).alias("beta")
        )

        df = df.with_columns(
            (pl.col("ret_1d") - pl.col("beta") * pl.col("mkt_ret"))
            .alias("_resid")
        ).with_columns(
            (
                pl.col("_resid")
                  .rolling_std(window_size=window, min_samples=window // 2)
                  .over("symbol")
                * np.sqrt(252)
            ).alias("idiosyncratic_vol")
        )

        return df.drop([
            "mkt_ret", "_ri_mean", "_rm_mean",
            "_cov_num", "_var_num", "_cov", "_var", "_resid",
        ])

    # ------------------------------------------------------------------
    # G1 (external): Merge WRDS Beta Suite betas
    # ------------------------------------------------------------------

    def _merge_betas_polars(
        self,
        df: pl.DataFrame,
        betas_pl: pl.DataFrame,
    ) -> pl.DataFrame:
        cols = [c for c in ["symbol", "date", "beta", "idiosyncratic_vol"]
                if c in betas_pl.columns]
        return df.join(betas_pl.select(cols), on=["symbol", "date"], how="left")

    # ------------------------------------------------------------------
    # G4: Fundamentals
    # ------------------------------------------------------------------

    def _merge_fundamentals(
        self,
        features_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Forward-fill quarterly fundamentals using rdq (no look-ahead bias)."""
        features_df     = features_df.copy()
        fundamentals_df = fundamentals_df.copy()

        # After Polars→pandas round-trip, date may be Python date objects,
        # datetime[ms], or datetime[ns] depending on Polars version.
        # Coerce everything to datetime64[us] (no tz) so merge_asof is happy.
        features_df["date"] = pd.to_datetime(
            features_df["date"], utc=False
        ).astype("datetime64[us]")

        fundamentals_df["rdq"] = pd.to_datetime(
            fundamentals_df["rdq"], utc=False
        ).astype("datetime64[us]")

        # merge_asof requires GLOBAL sort by the key column only.
        # `by="symbol"` does per-symbol matching but still expects rows
        # to be in ascending date order across the whole dataframe.
        features_df     = features_df.sort_values("date").reset_index(drop=True)
        fundamentals_df = fundamentals_df.sort_values("rdq").reset_index(drop=True)

        merged = pd.merge_asof(
            features_df,
            fundamentals_df,
            left_on="date",
            right_on="rdq",
            by="symbol",
            direction="backward",
        )
        return merged.sort_values(["symbol", "date"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # G4: Merge WRDS pre-computed ratios (comp.wrds_ratios)
    # ------------------------------------------------------------------

    def _merge_wrds_ratios(
        self,
        features_df: pd.DataFrame,
        ratios_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Forward-fill WRDS ratios onto the daily price panel using public_date.
        public_date is when the ratio becomes public — no look-ahead bias.

        wrds_ratios columns → TABULAR_CONTINUOUS_NAMES mapping:
            roe       → roe
            roa       → roa
            de_ratio  → debt_to_equity
            ptb       → price_to_book
            pe_op_dil → price_to_earnings
            mktcap    → market_cap
            divyield  → dividend_yield
            npm       → profit_margin
            opmad     → operating_margin
            at_turn   → (asset turnover — used as proxy for revenue signal)
        """
        features_df = features_df.copy()
        ratios_df   = ratios_df.copy()

        features_df["date"]          = pd.to_datetime(features_df["date"]).astype("datetime64[us]")
        ratios_df["public_date"]     = pd.to_datetime(ratios_df["public_date"]).astype("datetime64[us]")

        # Rename to match TABULAR_CONTINUOUS_NAMES expected by the model
        ratios_df = ratios_df.rename(columns={
            "de_ratio":  "debt_to_equity",
            "ptb":       "price_to_book",
            "pe_op_dil": "price_to_earnings",
            "mktcap":    "market_cap",
            "divyield":  "dividend_yield",
            "npm":       "profit_margin",
            "opmad":     "operating_margin",
            "at_turn":   "revenue",          # asset turnover as revenue proxy
        })

        # Fill columns not in wrds_ratios with NaN
        for col in ["net_income", "total_assets", "cash"]:
            ratios_df[col] = float("nan")

        features_df = features_df.sort_values("date")
        ratios_df   = ratios_df.sort_values("public_date")

        merged = pd.merge_asof(
            features_df,
            ratios_df,
            left_on="date",
            right_on="public_date",
            by="symbol",
            direction="backward",
        )
        return merged.sort_values(["symbol", "date"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # G6: GICS codes
    # ------------------------------------------------------------------
    def _merge_gics(
        self,
        features_df: pd.DataFrame,
        gics_df: pd.DataFrame,
    ) -> pd.DataFrame:
        cols = [c for c in ["symbol", "gsector", "ggroup"] if c in gics_df.columns]
        return features_df.merge(gics_df[cols], on="symbol", how="left")

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def apply_normalization(
        self,
        df: pd.DataFrame,
        feature_groups: dict[str, list[str]],
    ) -> pd.DataFrame:
        result = df.copy()
        for group_name, cols in feature_groups.items():
            existing = [c for c in cols if c in result.columns]
            if existing:
                logger.info(f"Normalising {group_name}: {existing}")
                result = two_pass_normalization(
                    result, feature_cols=existing, date_col="date"
                )
        return result