# Training Workflow Guide

**How to get from raw data to trained model**

---

## Quick Start (TL;DR)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set WRDS credentials
export WRDS_USERNAME="your_username"
export WRDS_PASSWORD="your_password"

# 3. Pre-compute features (one-time, ~2-4 hours)
python scripts/preprocess_features.py \
    --start-date 2010-01-01 \
    --end-date 2024-12-31 \
    --output-dir data/processed/features

# 4. Train on first validation fold
python scripts/train.py \
    --fold 0 \
    --batch-size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --n-hard 8 \
    --feature-dir data/processed/features

# 5. Monitor metrics
# - Loss: should decrease
# - Alignment: should increase toward 1.0
# - Hard neg sim: should decrease (model learning discrimination)
```

---

## Table of Contents

1. [Data Collection](#data-collection)
2. [Preprocessing Features](#preprocessing-features)
3. [Understanding Data Splits](#understanding-data-splits)
4. [Training](#training)
5. [Validation Metrics](#validation-metrics)
6. [Inference on New Stocks](#inference-on-new-stocks)
7. [Troubleshooting](#troubleshooting)

---

## Data Collection

### What You Need

**WRDS Access Required:**
- CRSP (daily prices, returns, volume)
- Compustat (fundamentals: P/E, P/B, ROE, market cap, GICS codes)
- Ken French (market factors for beta computation)

**Universe:** Russell 2000 + S&P 400 (~2,400 small/mid-cap stocks)

**Time Period:** 2010-01-01 to 2024-12-31 (14 years)

### Setup

```bash
# Option 1: Environment variables
export WRDS_USERNAME="your_wrds_username"
export WRDS_PASSWORD="your_wrds_password"

# Option 2: .env file (if you use python-dotenv)
echo "WRDS_USERNAME=your_username" > .env
echo "WRDS_PASSWORD=your_password" >> .env
```

### Test Connection

```python
from src.data.wrds_loader import WRDSLoader

loader = WRDSLoader()
print("✓ WRDS connection successful")

# Test with one stock
prices = loader.load_prices(
    symbols=['AAPL'],
    start_date='2020-01-01',
    end_date='2020-12-31'
)
print(f"Loaded {len(prices)} days of AAPL data")
```

---

## Preprocessing Features

### What Happens Here

The preprocessing script:
1. Loads raw price/fundamental data from WRDS
2. Computes all 32 features for each stock
3. Saves to parquet: `data/processed/features/{SYMBOL}_features.parquet`

**Why pre-compute?** Feature computation is expensive (60-day windows, beta regressions). Do once, train efficiently.

### Run Preprocessing

```bash
# All stocks (Russell 2000 + S&P 400) - takes 2-4 hours
python scripts/preprocess_features.py \
    --start-date 2010-01-01 \
    --end-date 2024-12-31 \
    --output-dir data/processed/features

# Or just a few symbols for testing
python scripts/preprocess_features.py \
    --start-date 2020-01-01 \
    --end-date 2023-12-31 \
    --output-dir data/processed/features \
    --symbols AAPL MSFT GOOGL AMZN TSLA
```

### Output Structure

```
data/processed/features/
├── AAPL_features.parquet     # (trading_days, 32_features)
├── MSFT_features.parquet
├── GOOGL_features.parquet
└── ... (~2,400 files)
```

Each parquet file contains:
- **Index:** Date (datetime64)
- **Columns:** 32 engineered features
  - Technical: z-scores, MA ratios, volume trends, momentum (13 features)
  - Market risk: Beta 60d, downside beta (2 features)
  - Volatility: Realized vol, GARCH, Parkinson (4 features)
  - Momentum: Returns 20d/60d/252d, ratios (5 features)
  - Valuation: P/E, P/B, ROE, market cap (4 features)
  - Sector: gsector, ggroup, gind, gsubind (4 categorical)

---

## Understanding Data Splits

### Why This Structure?

Financial data requires **temporal splits** (not random) to prevent look-ahead bias:

```
┌─────────────────────────────────────────────────────────────────┐
│  TRAINING (2010-2018) → Learn patterns from past                  │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│  EMBARGO GAP (2019) → No data (autocorrelation decay)           │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│  VALIDATION FOLD 1 (2020) → COVID crash (test robustness)       │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│  EMBARGO GAP (2021 Q1)                                          │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│  VALIDATION FOLD 2 (2021-2022) → Meme stocks + rate shock       │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│  EMBARGO GAP (2023 Q1)                                          │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│  VALIDATION FOLD 3 (2023) → AI boom (test generalization)      │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│  EMBARGO GAP (2024 Jan)                                         │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│  TEST SET (2024 Feb-Dec) → NEVER TOUCH UNTIL DONE               │
└─────────────────────────────────────────────────────────────────┘
```

### Key Rules

**Purge (252 days):** Remove last year of training data. Why? Features have 60-day lookback windows. Training on late 2019 would see data overlapping with early 2020.

**Embargo (252 days):** Gap between splits. Why? Returns are autocorrelated (today predicts tomorrow). Without embargo, validation performance is inflated.

**Test Set:** Never use until all training, validation, hyperparameter tuning complete.

### Date Boundaries

```python
from src.training.validator import CrossRegimeValidator

validator = CrossRegimeValidator()
validator.print_summary()
```

Output:
```
======================================================================
CROSS-REGIME VALIDATION STRUCTURE
======================================================================

TRAINING (clean, after purge)
  2010-01-01 → 2018-12-31
  (Purged last 252 days to prevent look-ahead)

EMBARGO GAP 1 (no data)
  2019-01-01 → 2019-12-31
  (252 trading days autocorrelation decay)

VALIDATION FOLD 1: COVID Crash + Recovery
  2020-01-01 → 2020-12-31
  Regime: High volatility, market crash, recovery

VALIDATION FOLD 2: Meme Stocks + Rate Shock
  2021-04-01 → 2022-12-31
  Regime: Retail trading boom, inflation, rate hikes

VALIDATION FOLD 3: AI Boom / Soft Landing
  2023-04-01 → 2023-12-31
  Regime: AI enthusiasm, soft landing narrative

======================================================================
FINAL TEST SET — NEVER TOUCH UNTIL COMPLETELY DONE
  2024-02-01 → 2024-12-31
  (Contains August 2024 carry unwind)
======================================================================
```

---

## Training

### Basic Training Command

```bash
python scripts/train.py \
    --fold 0 \              # Which validation fold (0=COVID, 1=Meme, 2=AI)
    --batch-size 32 \      # Number of positive samples per batch
    --epochs 50 \          # Training epochs
    --lr 1e-4 \            # Learning rate (AdamW)
    --n-hard 8 \           # Hard negatives per batch
    --feature-dir data/processed/features \
    --output-dir models/checkpoints
```

### What Happens During Training

**Per Batch:**
1. Sample 32 stocks with random dates from training period
2. Fetch hard negatives via GICS sampler (same ggroup, different beta)
3. Forward pass: temporal (BiMT-TCN) + tabular (TabMixer) → embeddings
4. Compute symmetric InfoNCE loss (both temporal→tabular and tabular→temporal)
5. Backward pass with gradient clipping
6. Log: loss, alignment score, hard negative similarity

**Per Epoch:**
- Average loss across all batches
- Average alignment score (should increase toward 1.0)
- Average hard negative similarity (should decrease, model learning discrimination)

**Every 5 Epochs:**
- Compute sector silhouette score on validation fold
- Save checkpoint if silhouette improves
- Log silhouette (should be > 0.1 for meaningful embeddings)

### Monitoring Training

**Console Output:**
```
Epoch 0: 100%|████████| 450/450 [02:15<00:00,  3.32it/s, loss=2.3421, align=0.234, hard_neg=0.456]
Epoch 0 - Loss: 2.3421, Align: 0.234
Val Silhouette: 0.089

Epoch 5: 100%|████████| 450/450 [02:14<00:00,  3.35it/s, loss=1.8765, align=0.456, hard_neg=0.234]
Epoch 5 - Loss: 1.8765, Align: 0.456
Val Silhouette: 0.123
Saved best model (silhouette=0.123)
```

**Expected Behavior:**
- Loss: Decreases over time (target < 1.0)
- Alignment: Increases toward 1.0 (target > 0.5)
- Hard neg sim: Decreases (target < 0.3, model learning to discriminate)
- Silhouette: Increases then stabilizes (target > 0.1)

### Training Across Folds

```bash
# Fold 0: COVID (2020) - High volatility
python scripts/train.py --fold 0 --epochs 50

# Fold 1: Meme stocks + Rate shock (2021-2022)
python scripts/train.py --fold 1 --epochs 50

# Fold 2: AI boom (2023)
python scripts/train.py --fold 2 --epochs 50
```

Each fold produces a checkpoint: `models/checkpoints/fold{N}_best.pt`

### Hyperparameter Tuning

Tune on **Fold 0** (COVID), validate generalization on **Folds 1-2**:

| Hyperparameter | Start With | Range |
|---------------|-------------|-------|
| Learning rate | 1e-4 | [5e-5, 2e-4] |
| Batch size | 32 | [16, 64] |
| Hard negatives | 8 | [4, 16] |
| Temperature (InfoNCE) | 0.07 | [0.05, 0.1] |

---

## Validation Metrics

### What We Track

**Every Batch:**
- **Loss**: InfoNCE loss (lower is better)
- **Hard negative similarity**: Cosine sim between positives and their hard negatives (lower = better discrimination)

**Every Epoch:**
- **Alignment score**: Cosine sim of positive pairs (higher = temporal and tabular views align better)

**Every 5 Epochs:**
- **Sector silhouette score**: Clustering quality by GICS sector (higher = embeddings have financial meaning)

### Interpreting Metrics

**Good Training:**
```
Loss:       2.5 → 1.2 ✓ (decreasing)
Alignment:  0.2 → 0.7 ✓ (increasing)
Hard neg:   0.5 → 0.2 ✓ (decreasing)
Silhouette: 0.05 → 0.15 ✓ (increasing)
```

**Bad Training (Collapse):**
```
Loss:       2.5 → 0.01 ✗ (too low = collapse)
Alignment:  0.2 → 0.99 ✗ (too fast = collapse)
Hard neg:   0.5 → 0.01 ✗ (model not discriminating)
Silhouette: 0.05 → -0.2 ✗ (inverted clustering)
```

**Collapse** = all embeddings identical. Fix by:
- Lower learning rate
- Increase temperature
- Add gradient clipping (already in code)

---

## Inference on New Stocks

### For Real-Time Recommendations

```python
from src.data.wrds_loader import WRDSLoader
from src.training.preprocessor import InferencePreprocessor
from src.models import DualEncoder
import torch

# 1. Load raw data for target stock
loader = WRDSLoader()
preprocessor = InferencePreprocessor()

prices = loader.load_prices(
    symbols=['TSLA'],
    start_date='2023-01-01',  # Need 252+ days
    end_date='2024-01-01',
)

# 2. Compute features
features = preprocessor.compute_features(prices)

# 3. Extract latest window
window_dict = preprocessor.extract_latest_window(features)

# 4. Convert to tensors (with training statistics)
stats = torch.load('models/training_stats.pt')  # Mean/std from training
tensors = preprocessor.to_tensors(window_dict, normalize=True, stats=stats)

# 5. Load trained model
model = DualEncoder(
    temporal_input_dim=13,
    tabular_continuous_dim=15,
    tabular_categorical_dims=[11, 25],
)
model.load_state_dict(torch.load('models/checkpoints/fold0_best.pt')['model_state_dict'])
model.eval()

# 6. Encode
with torch.no_grad():
    embedding = model.get_joint_embedding(
        tensors['temporal'].unsqueeze(0),
        tensors['tabular_cont'].unsqueeze(0),
        tensors['tabular_cat'].unsqueeze(0)
    )  # (1, 256)

# 7. Search nearest neighbors in pre-computed universe
# (You need to encode all universe stocks once, then search)
print(f"TSLA embedding shape: {embedding.shape}")
# Now use FAISS or sklearn.neighbors to find similar stocks
```

### Requirements for Inference

1. **Minimum history:** 252 days (for beta computation)
2. **Data completeness:** All OHLCV fields present
3. **GICS codes:** Must have gsector and ggroup from Compustat
4. **Training stats:** Must use same mean/std as training data

---

## Troubleshooting

### "Insufficient history: got X days, minimum 252 required"

**Cause:** Stock has less than 1 year of trading history.

**Fix:** Exclude the stock or use a shorter lookback (not recommended, beta will be noisy).

### "No hard negatives found for batch"

**Cause:** Not enough stocks in batch from same ggroup, or all have similar beta.

**Fix:** Increase batch size, or the training will use in-batch negatives only (still works, less effective).

### Loss not decreasing

**Check:**
- Learning rate too high? Try 5e-5
- Batch size too small? Try 64
- Hard negatives overwhelming? Reduce n_hard to 4

### Silhouette score negative

**Cause:** Embeddings clustering inversely (tech stocks with energy, etc.).

**Fix:** Usually fixes itself after more epochs. If persists, check GICS codes are loaded correctly.

### CUDA out of memory

**Fix:** Reduce batch size or use gradient accumulation (not yet implemented).

---

## Next Steps

After training:
1. **Evaluate on test set** (2024 data) — only after all training complete
2. **Compute NDCG@10** — do top-10 similar stocks share sector?
3. **Track real portfolio** — substitute recommendations vs actual tracking error
4. **Deploy inference pipeline** — encode target, search universe, filter by liquidity

---

## File Reference

| File | Purpose |
|------|---------|
| `scripts/preprocess_features.py` | One-time feature computation |
| `scripts/train.py` | Main training script |
| `src/training/preprocessor.py` | Inference preprocessing |
| `src/training/dataset.py` | PyTorch Dataset for features |
| `src/training/dataloader.py` | DataLoader with hard negatives |
| `src/training/metrics.py` | Alignment, silhouette, hard neg sim |
| `src/training/validator.py` | Date splits with embargo/purge |
| `src/training/trainer.py` | Training loop with gradient clipping |

---

**Questions?** Check the design doc: `docs/plans/2026-03-08-training-pipeline-design.md`
