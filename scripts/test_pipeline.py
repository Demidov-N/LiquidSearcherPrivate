"""
Pipeline validation script for LiquidSearcher.

Runs a series of checks from data loading through a forward pass,
without doing any actual training. Safe to run before committing
to a full training run on an expensive machine.

Usage:
    python -m scripts.test_pipeline \
        --feature-dir data/processed/by_symbol \
        --train-start 2010-01-01 \
        --train-end 2018-12-31 \
        --val-start 2020-01-01 \
        --val-end 2020-12-31

All checks print ✓ or ✗. Exit code 0 = all passed, 1 = any failed.
"""

import argparse
import sys
import traceback
import time
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "✓"
FAIL = "✗"
results = []


def check(name: str, fn):
    """Run fn(), print result, collect pass/fail."""
    try:
        t0 = time.time()
        info = fn()
        elapsed = time.time() - t0
        msg = f"  {info}" if info else ""
        print(f"{PASS}  {name}{msg}  ({elapsed:.2f}s)")
        results.append((name, True, None))
    except Exception as exc:
        print(f"{FAIL}  {name}")
        print(f"     {type(exc).__name__}: {exc}")
        traceback.print_exc()
        results.append((name, False, exc))


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_shard_dir(feature_dir: str):
    p = Path(feature_dir)
    assert p.exists(), f"Directory not found: {p}"
    files = sorted(p.glob("*_features.parquet"))
    assert len(files) > 0, "No *_features.parquet files found"
    return f"{len(files)} symbol files found"


def check_parquet_schema(feature_dir: str):
    import pyarrow.parquet as pq
    from src.training.data_module import TEMPORAL_FEATURE_NAMES, TABULAR_CONTINUOUS_NAMES
    files = sorted(Path(feature_dir).glob("*_features.parquet"))
    schema = pq.read_schema(str(files[0]))
    cols = set(schema.names)
    # Check temporal features exist
    missing_temporal = [c for c in TEMPORAL_FEATURE_NAMES if c not in cols]
    # Check either derived tabular OR raw Compustat cols exist
    has_beta = "beta" in cols
    assert not missing_temporal, f"Missing temporal cols: {missing_temporal}"
    assert has_beta, "Missing 'beta' column"
    return f"Schema OK — {len(cols)} columns in {files[0].name}"


def check_date_range(feature_dir: str, train_start: str, train_end: str,
                     val_start: str, val_end: str):
    import pandas as pd
    files = sorted(Path(feature_dir).glob("*_features.parquet"))
    # Sample 5 files
    for f in files[:5]:
        dates = pd.read_parquet(f, columns=["date"])["date"]
        dates = pd.to_datetime(dates)
        assert dates.min() <= pd.Timestamp(train_end), \
            f"{f.name}: no data before train_end {train_end} (min={dates.min().date()})"
    return f"Date coverage OK (checked {min(5, len(files))} files)"


def check_dataset_build(feature_dir: str, train_start: str, train_end: str,
                        val_start: str, val_end: str):
    from src.training.data_module import StockDataset
    train_ds = StockDataset(
        feature_dir=feature_dir,
        date_range=(train_start, train_end),
        window_size=60,
    )
    val_ds = StockDataset(
        feature_dir=feature_dir,
        date_range=(val_start, val_end),
        window_size=60,
    )
    assert len(train_ds) > 0, f"Train dataset empty (check date range {train_start}→{train_end})"
    assert len(val_ds) > 0,   f"Val dataset empty (check date range {val_start}→{val_end})"
    return f"train={len(train_ds):,} samples  val={len(val_ds):,} samples"


def check_single_sample(feature_dir: str, train_start: str, train_end: str):
    from src.training.data_module import StockDataset, TEMPORAL_FEATURE_NAMES, TABULAR_CONTINUOUS_NAMES
    ds = StockDataset(feature_dir=feature_dir, date_range=(train_start, train_end))
    sample = ds[0]

    # Shape checks
    assert sample["temporal"].shape    == (60, len(TEMPORAL_FEATURE_NAMES)), \
        f"temporal shape {sample['temporal'].shape}, expected (60, {len(TEMPORAL_FEATURE_NAMES)})"
    assert sample["tabular_cont"].shape == (len(TABULAR_CONTINUOUS_NAMES),), \
        f"tabular_cont shape {sample['tabular_cont'].shape}, expected ({len(TABULAR_CONTINUOUS_NAMES)},)"
    assert sample["tabular_cat"].shape  == (2,), \
        f"tabular_cat shape {sample['tabular_cat'].shape}, expected (2,)"

    # NaN check
    assert not torch.isnan(sample["temporal"]).any(),    "NaN in temporal features"
    assert not torch.isnan(sample["tabular_cont"]).any(), "NaN in tabular features"

    # Reasonable value check
    assert not torch.isinf(sample["temporal"]).any(),    "Inf in temporal features"
    assert not torch.isinf(sample["tabular_cont"]).any(), "Inf in tabular features"

    return (f"symbol={sample['symbol']}  date={sample['date']}  "
            f"gsector={sample['gsector']}  beta={sample['beta']:.3f}")


def check_dataloader(feature_dir: str, train_start: str, train_end: str,
                     val_start: str, val_end: str):
    from src.training.data_module import StockDataModule
    dm = StockDataModule(
        feature_dir=feature_dir,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        batch_size=32,
        num_workers=2,           # low for test
        purge_days=252,
        embargo_days=63,
        window_size=60,
    )
    dm.setup("fit")

    # Fetch 3 batches from train and val
    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()

    for split, loader in [("train", train_loader), ("val", val_loader)]:
        for i, batch in enumerate(loader):
            assert not torch.isnan(batch["temporal"]).any(),    f"{split} batch {i}: NaN in temporal"
            assert not torch.isnan(batch["tabular_cont"]).any(), f"{split} batch {i}: NaN in tabular"
            assert batch["temporal"].shape[0] > 0,              f"{split} batch {i}: empty batch"
            if i >= 2:
                break

    return "3 train batches + 3 val batches loaded cleanly"


def check_forward_pass(feature_dir: str, train_start: str, train_end: str,
                       val_start: str, val_end: str):
    from src.training.data_module import StockDataModule
    from src.training.module import DualEncoderModule

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dm = StockDataModule(
        feature_dir=feature_dir,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        batch_size=32,
        num_workers=2,
        purge_days=252,
        embargo_days=63,
    )
    dm.setup("fit")

    model = DualEncoderModule(
        temporal_input_dim=13,
        tabular_continuous_dim=15,
        embedding_dim=128,
    ).to(device)
    model.eval()

    loader = dm.val_dataloader()
    batch  = next(iter(loader))
    batch  = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in batch.items()}

    with torch.no_grad():
        t_emb, tab_emb = model.encode(batch)

    assert t_emb.shape   == (batch["temporal"].shape[0], 128), \
        f"temporal emb shape {t_emb.shape}"
    assert tab_emb.shape == (batch["temporal"].shape[0], 128), \
        f"tabular emb shape {tab_emb.shape}"
    assert not torch.isnan(t_emb).any(),   "NaN in temporal embeddings"
    assert not torch.isnan(tab_emb).any(), "NaN in tabular embeddings"

    # Check embeddings are L2-normalised (dot product with self ≈ 1)
    norms = t_emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        f"Temporal embeddings not normalised (norms: {norms.min():.4f}–{norms.max():.4f})"

    return f"Forward pass OK on {device} — emb shape {t_emb.shape}"


def check_loss(feature_dir: str, train_start: str, train_end: str,
               val_start: str, val_end: str):
    from src.training.data_module import StockDataModule
    from src.training.module import DualEncoderModule

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dm = StockDataModule(
        feature_dir=feature_dir,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        batch_size=32,
        num_workers=2,
        purge_days=252,
        embargo_days=63,
    )
    dm.setup("fit")

    model = DualEncoderModule(
        temporal_input_dim=13,
        tabular_continuous_dim=15,
        embedding_dim=128,
        temperature=0.1,
    ).to(device)

    loader = dm.val_dataloader()
    batch  = next(iter(loader))
    batch  = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in batch.items()}

    t_emb, tab_emb = model.encode(batch)
    loss, metrics  = model._info_nce(t_emb, tab_emb, batch["symbol"])

    assert not torch.isnan(loss),  f"Loss is NaN"
    assert not torch.isinf(loss),  f"Loss is Inf"
    assert loss.item() > 0,        f"Loss is zero or negative: {loss.item()}"

    fn_frac = metrics["fn_frac"]
    if fn_frac > 0.15:
        print(f"     ⚠  fn_frac={fn_frac:.2%} — high false-negative rate, "
              f"consider SymbolGroupSampler")

    return (f"loss={loss.item():.4f}  "
            f"alignment={metrics['alignment']:.4f}  "
            f"neg_sim={metrics['neg_sim']:.4f}  "
            f"fn_frac={fn_frac:.2%}")


def check_gpu():
    if not torch.cuda.is_available():
        return "No GPU — running on CPU (training will be slow)"
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    return (f"{props.name}  "
            f"VRAM={props.total_memory/1e9:.0f}GB  "
            f"free={free/1e9:.1f}GB")


def check_gradient_flow(feature_dir: str, train_start: str, train_end: str,
                        val_start: str, val_end: str):
    """Verify gradients flow through both towers."""
    from src.training.data_module import StockDataModule
    from src.training.module import DualEncoderModule

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dm = StockDataModule(
        feature_dir=feature_dir,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        batch_size=32,
        num_workers=2,
        purge_days=252,
        embargo_days=63,
    )
    dm.setup("fit")

    model = DualEncoderModule(
        temporal_input_dim=13,
        tabular_continuous_dim=15,
        embedding_dim=128,
    ).to(device)

    loader = dm.val_dataloader()
    batch  = next(iter(loader))
    batch  = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in batch.items()}

    t_emb, tab_emb = model.encode(batch)
    loss, _        = model._info_nce(t_emb, tab_emb, batch["symbol"])
    loss.backward()

    # Check all parameters received gradients
    no_grad = [n for n, p in model.named_parameters()
               if p.grad is None and p.requires_grad]
    zero_grad = [n for n, p in model.named_parameters()
                 if p.grad is not None and p.grad.abs().max() == 0]

    assert not no_grad,   f"Parameters with no gradient: {no_grad[:3]}"

    if zero_grad:
        return f"OK (⚠ {len(zero_grad)} params have zero grad — may be normal)"
    return f"All {sum(1 for p in model.parameters() if p.requires_grad)} param groups have gradients"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate LiquidSearcher pipeline")
    parser.add_argument("--feature-dir",  default="data/processed/by_symbol")
    parser.add_argument("--train-start",  default="2010-01-01")
    parser.add_argument("--train-end",    default="2018-12-31")
    parser.add_argument("--val-start",    default="2020-01-01")
    parser.add_argument("--val-end",      default="2020-12-31")
    args = parser.parse_args()

    fd = args.feature_dir
    ts, te = args.train_start, args.train_end
    vs, ve = args.val_start,   args.val_end

    print("\n" + "="*60)
    print("  LiquidSearcher Pipeline Validation")
    print("="*60 + "\n")

    check("GPU available",          lambda: check_gpu())
    check("Shard directory",        lambda: check_shard_dir(fd))
    check("Parquet schema",         lambda: check_parquet_schema(fd))
    check("Date range coverage",    lambda: check_date_range(fd, ts, te, vs, ve))
    check("Dataset index build",    lambda: check_dataset_build(fd, ts, te, vs, ve))
    check("Single sample __getitem__", lambda: check_single_sample(fd, ts, te))
    check("DataLoader (2 workers)", lambda: check_dataloader(fd, ts, te, vs, ve))
    check("Forward pass",           lambda: check_forward_pass(fd, ts, te, vs, ve))
    check("Loss computation",       lambda: check_loss(fd, ts, te, vs, ve))
    check("Gradient flow",          lambda: check_gradient_flow(fd, ts, te, vs, ve))

    # Summary
    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    print("\n" + "="*60)
    print(f"  {passed}/{total} checks passed")
    print("="*60 + "\n")

    if passed < total:
        failed = [(n, e) for n, ok, e in results if not ok]
        print("Failed checks:")
        for name, exc in failed:
            print(f"  {FAIL} {name}: {exc}")
        print()
        sys.exit(1)
    else:
        print("All checks passed — safe to start training.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()