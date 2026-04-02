# Features Implemented: Dual-Encoder Stock Substitute Model

**Date:** 2026-03-09  
**Branch:** feature/data-collection  
**Status:** Complete - 33 tests passing, training pipeline working

---

## Summary

Implemented a production-ready dual-encoder contrastive learning system for stock substitute recommendation using PyTorch Lightning. The system learns joint embeddings from temporal (60-day OHLCV) and tabular (fundamentals + GICS) data using InfoNCE loss.

---

## Architecture

### Model Components

| Component | Description | Parameters |
|-----------|-------------|------------|
| **Temporal Encoder (BiMT-TCN)** | TCN (dilations 1,2,4,8) + Transformer (2 layers, 4 heads) + GAP | ~300K |
| **Tabular Encoder (TabMixer)** | MLP-Mixer with GICS embeddings (gsector→8-dim, ggroup→16-dim) | ~380K |
| **InfoNCE Loss** | Contrastive loss aligning temporal/tabular views | - |
| **Joint Embedding** | Concatenate [temporal\|\|tabular] → 256-dim for similarity search | - |

**Total Parameters:** 681K (small, fast training)

---

## Training Pipeline

### PyTorch Lightning Infrastructure

1. **StockDataModule** (`src/training/data_module.py`)
   - Temporal train/val/test splits with purge (252 days) and embargo (63 days)
   - Financial ML best practices to prevent look-ahead bias
   - Window-based sampling from parquet files
   - Supports real and mock data

2. **DualEncoderModule** (`src/training/module.py`)
   - LightningModule with InfoNCE loss
   - AdamW optimizer with cosine annealing
   - Training/validation metrics logging
   - Joint embedding extraction for inference

3. **Training Script** (`scripts/train.py`)
   - Full CLI with argparse
   - ModelCheckpoint (saves top 3 based on val/loss)
   - EarlyStopping (patience=15)
   - TQDM progress bar for training/validation
   - Checkpoint saved to `checkpoints/`

4. **Validation Script** (`scripts/validate.py`)
   - Load trained model from checkpoint
   - Compute validation metrics (loss, alignment, hard negative similarity)
   - Optional sector silhouette score
   - Sanity checks with pass/fail indicators
   - JSON output for results

---

## Features Implemented

### 1. Neural Network Models (`src/models/`)

| File | Description | Status |
|------|-------------|--------|
| `base.py` | Abstract BaseEncoder class | ✅ |
| `tcn.py` | CausalConv1d + TemporalConvNet with dilations | ✅ |
| `positional_encoding.py` | Sinusoidal PE (Vaswani et al. 2017) | ✅ |
| `temporal_encoder.py` | BiMT-TCN: TCN → PE → Transformer → GAP | ✅ |
| `mixer.py` | MixerBlock + TabMixer with categorical embeddings | ✅ |
| `tabular_encoder.py` | TabularEncoder with GICS config | ✅ |
| `dual_encoder.py` | DualEncoder with InfoNCE loss | ✅ |

### 2. Training Infrastructure (`src/training/`)

| File | Description | Status |
|------|-------------|--------|
| `data_module.py` | StockDataModule with temporal splits | ✅ |
| `module.py` | DualEncoderModule (LightningModule) | ✅ |
| `__init__.py` | Package exports | ✅ |

### 3. Scripts (`scripts/`)

| File | Description | Status |
|------|-------------|--------|
| `train.py` | Main training script with CLI | ✅ |
| `validate.py` | Validation/sanity check script | ✅ |
| `preprocess_features.py` | Feature pre-processing | ✅ (existing) |

### 4. Testing (`tests/`)

| File | Description | Tests |
|------|-------------|-------|
| `test_models_*.py` | Model component tests | 25 |
| `test_training_integration.py` | Training infra tests | 2 |
| `test_data_module.py` | DataModule tests | 2 |
| `test_training_module.py` | LightningModule tests | 3 |
| `test_training_end_to_end.py` | Full pipeline tests | 3 |

**Total Tests:** 33 - all passing ✅

### 5. Documentation

- `README_MODEL.md` - Model architecture and usage guide
- Feature name mappings in `data_module.py` (TEMPORAL_FEATURE_NAMES, TABULAR_CONTINUOUS_NAMES)
- Inline docstrings throughout codebase

---

## Training Results

### Test Run (Real Data)

```bash
python -m scripts.train \
    --feature-dir /home/redbear/projects/liquidity-new/data/processed/features \
    --train-start 2010-01-01 --train-end 2018-12-31 \
    --val-start 2020-01-01 --val-end 2020-12-31 \
    --symbols AAPL MSFT --epochs 1 --batch-size 16
```

**Results:**
- Train dataset: 6,070 samples
- Val dataset: 606 samples
- 1 epoch: train/loss=2.81, val/loss=2.77
- Checkpoint saved successfully
- Training time: ~2 minutes (CPU)

---

## Technical Details

### Feature Mappings

**Temporal Features (13):**
```python
TEMPORAL_FEATURE_NAMES = [
    "z_close", "z_volume",           # Price/volume z-scores
    "ma_ratio_5d", "ma_ratio_10d", "ma_ratio_20d",  # MA ratios
    "realized_vol_20d", "realized_vol_60d",  # Volatility
    "mom_1m", "mom_3m", "mom_6m", "mom_12m", "mom_12_1m",  # Momentum
    "log_ret_cum",  # Cumulative returns
]
```

**Tabular Continuous (15):**
```python
TABULAR_CONTINUOUS_NAMES = [
    "beta", "idiosyncratic_vol",     # Risk
    "roe", "roa", "debt_to_equity",  # Financial health
    "price_to_book", "price_to_earnings", "market_cap", "dividend_yield",  # Valuation
    "revenue", "net_income", "total_assets", "cash",  # Fundamentals
    "operating_margin", "profit_margin",  # Profitability
]
```

### Temporal Splits

Implements financial ML best practices:
- **Purge (252 days):** Remove last year from training to prevent feature overlap
- **Embargo (63 days):** Gap between train/val to prevent return autocorrelation

```
Train: 2010-2018 (purge removes last 252 days)
     ↓
Embargo: 2019 (63 days gap)
     ↓
Val: 2020
```

---

## Dependencies

```
pytorch-lightning
torch
info-nce-pytorch
tqdm
pandas
polars
numpy
```

---

## Next Steps / Future Work

1. **Hyperparameter Tuning:** Grid search on temperature, lr, batch size
2. **Hard Negative Mining:** Implement GICS-structured hard negatives
3. **Sector Silhouette:** Regular validation metric for clustering quality
4. **Inference Pipeline:** Fast similarity search with FAISS
5. **W&B Integration:** Add experiment tracking
6. **More Symbols:** Scale to full universe

---

## Commands

```bash
# Run training
python -m scripts.train --epochs 10 --batch-size 32 --symbols AAPL MSFT GOOGL

# Run validation
python -m scripts.validate --checkpoint checkpoints/best.ckpt --compute-silhouette

# Run tests
python -m pytest tests/ -v

# Generate mock data
python tests/fixtures/create_mock_data.py
```

---

## Git History

```
6445ee4 feat: add feature name mappings + TQDM progress bar
8d2e3d5 test: add end-to-end training pipeline tests
df42c8f feat: add training script + end-to-end tests working
98ca36f feat: add validation script with sanity checks
1ddb4c6 chore: add mock data generator for testing
a1a33a3 feat: add DualEncoder LightningModule with InfoNCE loss
73cde32 feat: add StockDataModule with temporal splits
f9aedad chore: setup training infrastructure package
```
