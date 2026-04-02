"""PyTorch Lightning DataModule with temporal splits for financial data."""

import torch
torch.set_float32_matmul_precision("medium")  # utilise Tensor Cores on RTX/A-series GPUs

from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

pd.set_option('future.no_silent_downcasting', True)

TEMPORAL_FEATURE_NAMES = [
    "z_close", "z_volume", "ma_ratio_5d", "ma_ratio_10d", "ma_ratio_20d",
    "realized_vol_20d", "realized_vol_60d", "mom_1m", "mom_3m", "mom_6m",
    "mom_12m", "mom_12_1m", "log_ret_cum",
]  # 13 features

TABULAR_CONTINUOUS_NAMES = [
    "beta",              # 0:  from CRSP vwretd local OLS
    "idiosyncratic_vol", # 1:  from CRSP vwretd local OLS
    "roe",               # 2:  wrds_ratios.roe
    "roa",               # 3:  wrds_ratios.roa
    "debt_to_equity",    # 4:  wrds_ratios.de_ratio
    "price_to_book",     # 5:  wrds_ratios.ptb
    "price_to_earnings", # 6:  wrds_ratios.pe_op_dil
    "market_cap",        # 7:  wrds_ratios.mktcap
    "dividend_yield",    # 8:  wrds_ratios.divyield
    "revenue",           # 9:  wrds_ratios.at_turn (asset turnover proxy)
    "net_income",        # 10: NaN placeholder (not in wrds_ratios)
    "total_assets",      # 11: NaN placeholder (not in wrds_ratios)
    "cash",              # 12: NaN placeholder (not in wrds_ratios)
    "operating_margin",  # 13: wrds_ratios.opmad
    "profit_margin",     # 14: wrds_ratios.npm
]  # 15 features

# Raw Compustat columns — may or may not exist in the parquet depending
# on whether fundamentals were successfully fetched during preprocessing.
# _load_symbol reads only what is present and fills the rest with NaN.
_FUNDAMENTAL_RAW_COLS = [
    "atq", "seqq", "niq", "cshoq", "ceqq",
    "epspxq", "txtq", "xintq", "saleq", "cheq",
]

# Always-present columns (price features + technicals + beta + sector)
_BASE_COLS = (
    ["date", "symbol", "prc"]
    + TEMPORAL_FEATURE_NAMES
    + ["beta", "idiosyncratic_vol"]
    + ["gsector", "ggroup"]
)

# Will be filtered to only columns that actually exist in the file
_NEEDED_COLS = _BASE_COLS + _FUNDAMENTAL_RAW_COLS


# --------------------------------------------------------f-------------------
# Worker-local per-symbol DataFrame cache (LRU-bounded)
# ---------------------------------------------------------------------------
# LRU_CACHE_SIZE caps how many symbol DataFrames one worker holds at once.
# Each symbol ~= 15yr x 252 days x 30 cols x 8 bytes ~ 1-2 MB.
# 4 workers x 128 symbols x 1.5 MB ~ 768 MB peak. Lower if OOM persists.

from collections import OrderedDict

LRU_CACHE_SIZE = 128
_DF_CACHE: OrderedDict = OrderedDict()


def _load_symbol(file_path: str) -> pd.DataFrame:
    if file_path in _DF_CACHE:
        _DF_CACHE.move_to_end(file_path)
        return _DF_CACHE[file_path]

    # Read only columns that actually exist in this parquet file
    import pyarrow.parquet as pq
    available = set(pq.read_schema(file_path).names)
    cols_to_read = [c for c in _NEEDED_COLS if c in available]
    df = pd.read_parquet(file_path, columns=cols_to_read)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    # gsector/ggroup exist now — fill before ratio derivation
    df["gsector"] = df["gsector"].fillna(0).astype(int) if "gsector" in df.columns else 0
    df["ggroup"]  = df["ggroup"].fillna(0).astype(int)  if "ggroup"  in df.columns else 0
    # TABULAR_CONTINUOUS_NAMES filled AFTER ratio derivation below

    # Derive fundamental ratios if raw Compustat columns are present.
    # If fundamentals were not fetched during preprocessing, all ratio
    # columns are filled with NaN — the model sees zeros after NaN-fill.
    has_fundamentals = "atq" in df.columns

    def _safe_div(num, den):
        return (num / den.replace(0, float("nan"))) if has_fundamentals else float("nan")

    nan_col = pd.Series(float("nan"), index=df.index)

    df["roe"]               = (_safe_div(df["niq"],  df["ceqq"]).clip(-100, 100)
                                if has_fundamentals else nan_col)
    df["roa"]               = (_safe_div(df["niq"],  df["atq"]).clip(-100, 100)
                                if has_fundamentals else nan_col)
    df["debt_to_equity"]    = (_safe_div(df["atq"] - df["seqq"], df["seqq"]).clip(-100, 100)
                                if has_fundamentals else nan_col)
    df["price_to_book"]     = (_safe_div(df["prc"], _safe_div(df["ceqq"], df["cshoq"])).clip(-100, 100)
                                if has_fundamentals else nan_col)
    df["price_to_earnings"] = (_safe_div(df["prc"], df["epspxq"]).clip(-100, 100)
                                if has_fundamentals else nan_col)
    df["market_cap"]        = (df["prc"] * df["cshoq"] if has_fundamentals else nan_col)
    df["net_income"]        = (df["niq"]  if has_fundamentals else nan_col)
    df["total_assets"]      = (df["atq"]  if has_fundamentals else nan_col)
    df["revenue"]           = (df["saleq"] if "saleq" in df.columns else nan_col)
    df["cash"]              = (df["cheq"]  if "cheq"  in df.columns else nan_col)

    if has_fundamentals:
        ebit = (df["niq"]
                + df.get("txtq",  nan_col).fillna(0)
                + df.get("xintq", nan_col).fillna(0))
        sale = df.get("saleq", nan_col)
        df["operating_margin"] = _safe_div(ebit, sale)
        df["profit_margin"]    = _safe_div(df["niq"], sale)
    else:
        df["operating_margin"] = nan_col
        df["profit_margin"]    = nan_col

    df["dividend_yield"] = 0.0

    # Drop raw columns — not needed downstream
    df.drop(columns=[c for c in _FUNDAMENTAL_RAW_COLS if c in df.columns],
            inplace=True, errors="ignore")

    # Now all TABULAR_CONTINUOUS_NAMES columns exist (either derived or NaN)
    # Fill NaN → 0 so __getitem__ never receives NaN into the model
    for col in TABULAR_CONTINUOUS_NAMES:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    _DF_CACHE[file_path] = df
    if len(_DF_CACHE) > LRU_CACHE_SIZE:
        _DF_CACHE.popitem(last=False)

    return _DF_CACHE[file_path]


# ---------------------------------------------------------------------------
# Compact index
# ---------------------------------------------------------------------------

class _SymbolEntry:
    """Holds all valid end-row indices for one symbol. ~8 bytes × N rows."""
    __slots__ = ("symbol", "file_path", "row_indices")

    def __init__(self, symbol: str, file_path: str, row_indices: np.ndarray):
        self.symbol      = symbol
        self.file_path   = file_path
        self.row_indices = row_indices


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StockDataset(Dataset):

    GSECTOR_MAX = 10
    GGROUP_MAX  = 24

    def __init__(
        self,
        feature_dir: str,
        date_range:  tuple[str, str],
        symbols:     Optional[List[str]] = None,
        window_size: int = 60,
    ):
        self.feature_dir = Path(feature_dir)
        self.window_size = window_size
        self.start_date  = pd.Timestamp(date_range[0])
        self.end_date    = pd.Timestamp(date_range[1])
        self._symbols    = symbols or self._discover_symbols()
        self._entries, self._offsets = self._build_index()

    def _discover_symbols(self) -> List[str]:
        return sorted(
            f.stem.replace("_features", "")
            for f in self.feature_dir.glob("*_features.parquet")
        )

    def _build_index(self):
        """
        Read only the 'date' column per symbol to find valid rows.
        RAM: O(symbols × trading_days × 8 bytes) — typically <100 MB total.
        """
        entries: List[_SymbolEntry] = []
        offsets = [0]

        for sym in self._symbols:
            fp = self.feature_dir / f"{sym}_features.parquet"
            if not fp.exists():
                continue
            try:
                dates = (
                    pd.read_parquet(fp, columns=["date"])
                      .assign(date=lambda d: pd.to_datetime(d["date"]))
                      .sort_values("date")
                      .reset_index(drop=True)
                )
                mask = (
                    (dates["date"] >= self.start_date) &
                    (dates["date"] <= self.end_date)
                )
                rows = np.where(mask.values)[0]
                if len(rows) == 0:
                    continue
                entries.append(_SymbolEntry(sym, str(fp), rows))
                offsets.append(offsets[-1] + len(rows))
            except Exception as exc:
                print(f"Warning: skipping {sym}: {exc}")

        return entries, np.array(offsets, dtype=np.int64)

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def _locate(self, idx: int):
        """O(log N) flat-index → (entry, end_row)."""
        ei      = int(np.searchsorted(self._offsets, idx, side="right")) - 1
        entry   = self._entries[ei]
        end_row = int(entry.row_indices[idx - int(self._offsets[ei])])
        return entry, end_row

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry, end_row = self._locate(idx)

        df = _load_symbol(entry.file_path)   # cache hit after first access

        # Temporal window
        start_row    = max(0, end_row - self.window_size + 1)
        window       = df.iloc[start_row : end_row + 1]
        temporal_np  = window[TEMPORAL_FEATURE_NAMES].to_numpy(dtype=np.float32)
        temporal_np  = np.nan_to_num(temporal_np, nan=0.0, posinf=0.0, neginf=0.0)

        if len(window) < self.window_size:
            pad = self.window_size - len(window)
            buf = np.zeros((self.window_size, len(TEMPORAL_FEATURE_NAMES)), dtype=np.float32)
            buf[pad:] = temporal_np
            temporal_np = buf

        # Tabular snapshot
        row          = df.iloc[end_row]
        tabular_cont = row[TABULAR_CONTINUOUS_NAMES].to_numpy(dtype=np.float32)
        tabular_cont = np.nan_to_num(tabular_cont, nan=0.0, posinf=0.0, neginf=0.0)
        gsector      = int(np.clip(row["gsector"], 0, self.GSECTOR_MAX))
        ggroup       = int(np.clip(row["ggroup"],  0, self.GGROUP_MAX))

        return {
            "symbol":       entry.symbol,
            "date":         str(row["date"].date()),
            "temporal":     torch.from_numpy(temporal_np),
            "tabular_cont": torch.from_numpy(tabular_cont),
            "tabular_cat":  torch.tensor([gsector, ggroup], dtype=torch.long),
            "beta":         float(tabular_cont[0]),
            "gsector":      gsector,
            "ggroup":       ggroup,
        }


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class StockDataModule(pl.LightningDataModule):

    def __init__(
        self,
        feature_dir:   str,
        train_start:   str,
        train_end:     str,
        val_start:     str,
        val_end:       str,
        test_start:    Optional[str] = None,
        test_end:      Optional[str] = None,
        symbols:       Optional[List[str]] = None,
        batch_size:    int = 32,
        num_workers:   int = 4,
        purge_days:    int = 252,
        embargo_days:  int = 63,
        window_size:   int = 60,
        samples_per_epoch: Optional[bool] = None

    ):
        super().__init__()
        self.save_hyperparameters()
        self.feature_dir = feature_dir
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.window_size = window_size
        self.symbols     = symbols

        self.train_start = pd.Timestamp(train_start)
        self.train_end   = pd.Timestamp(train_end)
        self.val_start   = pd.Timestamp(val_start)
        self.val_end     = pd.Timestamp(val_end)

        self.effective_train_end  = self.train_end  - timedelta(days=purge_days)
        self.effective_val_start  = self.val_start  + timedelta(days=embargo_days)

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", None):
            self.train_dataset = StockDataset(
                feature_dir = self.feature_dir,
                date_range  = (str(self.train_start.date()),
                               str(self.effective_train_end.date())),
                symbols     = self.symbols,
                window_size = self.window_size,
            )
            self.val_dataset = StockDataset(
                feature_dir = self.feature_dir,
                date_range  = (str(self.effective_val_start.date()),
                               str(self.val_end.date())),
                symbols     = self.symbols,
                window_size = self.window_size,
            )
            print(f"Train: {len(self.train_dataset):,} samples")
            print(f"Val:   {len(self.val_dataset):,} samples")

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        # multiprocessing_context="fork" is faster on Linux and avoids the
        # worker re-import overhead that "spawn" causes on every epoch.
        # prefetch_factor=2 means each worker pre-loads 2*batch_size samples
        # ahead — keeps the GPU fed without excessive RAM pressure.
        return DataLoader(
            dataset,
            batch_size             = self.batch_size,
            shuffle                = shuffle,
            num_workers            = self.num_workers,
            pin_memory             = True,
            drop_last              = shuffle,
            persistent_workers     = self.num_workers > 0,
            prefetch_factor        = 2 if self.num_workers > 0 else None,
            multiprocessing_context= "fork" if self.num_workers > 0 else None,
        )

    def train_dataloader(self) -> DataLoader:
        dataset = self.train_dataset
        if self.hparams.get("samples_per_epoch"):
            n = self.hparams.samples_per_epoch
            indices = torch.randperm(len(dataset))[:n]
            dataset = torch.utils.data.Subset(dataset, indices)
        return self._loader(dataset, shuffle=True)


    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)