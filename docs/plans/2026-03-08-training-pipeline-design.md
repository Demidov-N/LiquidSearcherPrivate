# Training Pipeline Design: Dual-Encoder Stock Substitute Model

**Date:** 2026-03-08  
**Status:** Design Ready for Implementation  
**Scope:** Training infrastructure, validation metrics, data preprocessing, and inference documentation

---

## Table of Contents

1. [Overview](#overview)
2. [Data Splits & Temporal Structure](#data-splits--temporal-structure)
   - [Why Temporal Splits Matter in Finance](#why-temporal-splits-matter-in-finance)
   - [The Embargo/Purge Pattern Explained](#the-embargopurge-pattern-explained)
   - [Exact Date Boundaries](#exact-date-boundaries)
3. [Data Pipeline Architecture](#data-pipeline-architecture)
   - [Pre-computation Strategy](#pre-computation-strategy)
   - [Feature Storage Format](#feature-storage-format)
   - [Window Sampling Logic](#window-sampling-logic)
4. [Training Components](#training-components)
   - [Dataset Class](#dataset-class)
   - [DataLoader with Hard Negatives](#dataloader-with-hard-negatives)
   - [Training Loop Structure](#training-loop-structure)
5. [Validation Metrics](#validation-metrics)
   - [Batch-Level: Hard Negative Similarity](#batch-level-hard-negative-similarity)
   - [Epoch-Level: Alignment Score](#epoch-level-alignment-score)
   - [Periodic: Sector Silhouette Score](#periodic-sector-silhouette-score)
6. [Inference Documentation](#inference-documentation)
   - [Standalone Preprocessing Pipeline](#standalone-preprocessing-pipeline)
   - [Feature Computation for New Stocks](#feature-computation-for-new-stocks)
7. [Implementation Files](#implementation-files)

---

## Overview

This design implements a production-ready training pipeline for the dual-encoder stock substitute recommendation system. The pipeline handles:

- **Temporal data splitting** with purge and embargo periods to prevent look-ahead bias
- **Feature pre-computation** for efficient training
- **GICS-structured hard negative sampling** during batch construction
- **Multi-level validation metrics** (batch, epoch, fold) to monitor learning quality
- **Comprehensive documentation** for inference on new stocks

---

## Data Splits & Temporal Structure

### Why Temporal Splits Matter in Finance

In financial machine learning, **random splitting destroys validity** because:

1. **Look-ahead bias**: Random split puts "future" stocks in training, "past" stocks in validation
2. **Non-stationarity**: Markets in 2020 (COVID) ≠ markets in 2024 (AI boom)
3. **Autocorrelation**: Stock returns today predict returns tomorrow

**Solution**: Train on past, validate on future. Only temporal splits preserve causality.

### The Embargo/Purge Pattern Explained

**Purge (252 trading days ≈ 1 year)**:
- Remove the last 252 days of training data
- **Why?** Features have 60-day lookback windows. Training on late 2019 uses data that overlaps with early 2020.
- **Without purge**: Model sees "future" information during feature computation

**Embargo (252 trading days)**:
- Gap between training end and validation start
- **Why?** Returns are autocorrelated. Yesterday's return predicts today's.
- **Without embargo**: Validation performance is artificially inflated

### Exact Date Boundaries

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TEMPORAL DATA STRUCTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRAINING (what model learns from)                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2010-01-01 → 2019-12-31                                            │   │
│  │                                                                      │   │
│  │ PURGE: Remove last 252 days                                         │   │
│  │ (2019-01-01 → 2019-12-31 removed from training features)            │   │
│  │                                                                      │   │
│  │ Effective clean train end: 2018-12-31                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  EMBARGO GAP 1 (252 trading days)                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2019-01-01 → 2019-12-31                                            │   │
│  │ DO NOT USE - contamination buffer                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  VALIDATION FOLD 1: COVID Crash + Recovery                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2020-01-01 → 2020-12-31                                            │   │
│  │ Tune hyperparameters here (temperature, lr, batch size, depth)      │   │
│  │ Regime: High volatility, market crash, recovery                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  EMBARGO GAP 2                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2021-01-01 → 2021-03-31 (3 months)                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  VALIDATION FOLD 2: Meme Stocks + Rate Shock                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2021-04-01 → 2022-12-31                                            │   │
│  │ Confirm hyperparameters generalize across regimes                   │   │
│  │ Regime: Retail trading boom, inflation, rate hikes                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  EMBARGO GAP 3                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2023-01-01 → 2023-03-31 (3 months)                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  VALIDATION FOLD 3: AI Boom / Soft Landing                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2023-04-01 → 2023-12-31                                            │   │
│  │ Final architecture decisions before test                            │   │
│  │ Regime: AI enthusiasm, "soft landing" narrative                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  EMBARGO GAP 4                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2024-01-01 → 2024-01-31 (1 month)                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  FINAL TEST SET — NEVER TOUCH UNTIL DONE                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2024-02-01 → 2024-12-31                                            │   │
│  │ Contains: August 2024 carry unwind                                  │   │
│  │ Report all business metrics here (NDCG@10, tracking error)          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Source-Specific Rules

| Source | Extra Rule |
|--------|-----------|
| CRSP | 252-day purge as above |
| Compustat | Use `rdq` (report date), not `datadate` — adds ~60-day natural lag |
| Ken French FF5 | Same 252-day purge (rolling regressions) |
| IBES | Use `actdats` (activation date), not fiscal period end |

---

## Data Pipeline Architecture

### Pre-computation Strategy

**Why pre-compute?**
- Feature computation (32 features × 60-day windows × 2,400 stocks) = expensive
- Do once, cache to disk, load fast during training
- Enables efficient mini-batch sampling

**Process:**
1. For each stock, load raw price/fundamental data from WRDS
2. Compute all 32 features using FeatureEngineer
3. Save as parquet: `data/processed/features/{symbol}_features.parquet`
4. During training, load slices from these files

### Feature Storage Format

Each stock's features stored as:

```python
# data/processed/features/AAPL_features.parquet
Schema:
- date: datetime (index)
- market_beta_60d: float64
- downside_beta_60d: float64
- realized_vol_20d: float64
- ... (all 32 features)
- gsector: int64 (categorical)
- ggroup: int64 (categorical)
```

### Window Sampling Logic

**Training batch construction:**

```python
# For each batch:
1. Sample N stocks uniformly from training universe
2. For each stock, sample a random date from training period
3. Extract:
   - Temporal: [date-60d : date] of technical features (13 dims)
   - Tabular: fundamental features at date (15 dims + 2 categorical)
4. Query GICSHardNegativeSampler for hard negatives
5. Fetch embeddings for hard negative stocks
6. Concatenate to batch: [batch_positives, hard_negatives]
```

---

## Training Components

### Dataset Class

**FeatureDataset** (`src/training/dataset.py`):

```python
class FeatureDataset(Dataset):
    """PyTorch Dataset for pre-computed stock features."""
    
    def __init__(
        self,
        feature_dir: str,
        date_range: tuple[str, str],  # ('2010-01-01', '2018-12-31')
        symbols: list[str] | None = None,  # None = all available
        window_size: int = 60,
    )
    
    def __getitem__(self, idx) -> dict:
        """Returns:
        {
            'symbol': str,
            'date': datetime,
            'temporal': torch.Tensor,  # (60, 13)
            'tabular_cont': torch.Tensor,  # (15,)
            'tabular_cat': torch.Tensor,   # (2,) - gsector, ggroup
            'beta': float,
            'gsector': int,
            'ggroup': int,
        }
        """
```

### DataLoader with Hard Negatives

**StockDataLoader** (`src/training/dataloader.py`):

```python
class StockDataLoader(DataLoader):
    """Custom DataLoader with GICS-structured hard negative sampling."""
    
    def __init__(
        self,
        dataset: FeatureDataset,
        batch_size: int,
        sampler: GICSHardNegativeSampler,
        n_hard: int = 8,  # hard negatives per batch
        feature_dir: str,  # to fetch hard neg features
    )
    
    def _get_hard_negatives(
        self, 
        batch_samples: list[dict]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch temporal and tabular features for hard negative symbols."""
        # Use GICSHardNegativeSampler to find symbols
        # Load their features from disk
        # Return concatenated tensors
```

**Hard negative batch construction:**

```python
# Batch structure after hard negative injection:
# Temporal: (batch + n_hard, 60, 13)
# Tabular: (batch + n_hard, 15 continuous + 2 categorical)

# InfoNCE loss uses:
# - Positives: first 'batch' entries (diagonal similarity)
# - Negatives: all other entries (in-batch) + hard negatives
```

### Training Loop Structure

**Training flow per epoch:**

```python
for batch_idx, batch in enumerate(train_loader):
    # 1. Get batch with hard negatives
    temporal, tabular_cont, tabular_cat = batch
    
    # 2. Forward pass
    temporal_emb, tabular_emb = model(temporal, tabular_cont, tabular_cat)
    
    # 3. Compute loss
    # temporal_emb[:batch_size] vs all tabular_emb (with hard negatives)
    loss = loss_fn(temporal_emb, tabular_emb)
    
    # 4. Backward + optimize (with gradient clipping)
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # 5. Log batch metrics
    if batch_idx % log_interval == 0:
        hard_neg_sim = compute_hard_negative_similarity(temporal_emb, tabular_emb)
        logger.log({
            'train/loss': loss.item(),
            'train/hard_neg_similarity': hard_neg_sim,
        })

# End of epoch: compute alignment score
alignment = compute_alignment_score(model, val_samples)
logger.log({'train/alignment': alignment})

# Every 5 epochs: compute sector silhouette
if epoch % 5 == 0:
    silhouette = compute_sector_silhouette(model, val_dataset)
    logger.log({'val/sector_silhouette': silhouette})
```

---

## Validation Metrics

### Batch-Level: Hard Negative Similarity

**What it measures:** Are hard negatives actually different from positives?

**Formula:**
```python
# For positive pair (temporal[i], tabular[i])
# Compare against its hard negatives
pos_sim = cosine_similarity(temporal[i], tabular[i])
hard_neg_sims = [cosine_similarity(temporal[i], tabular[j]) 
                 for j in hard_negative_indices]

hard_neg_similarity = mean(hard_neg_sims)  # Should be << pos_sim
```

**Interpretation:**
- If hard_neg_similarity ≈ pos_sim: Model not learning discrimination
- If hard_neg_similarity << pos_sim: Model learning what matters

**Target:** Hard negative similarity should decrease over training.

### Epoch-Level: Alignment Score

**What it measures:** Are positive pairs (same stock, different views) aligning?

**Formula:**
```python
# Average cosine similarity of positive pairs
alignment = mean([cosine_similarity(temporal_emb[i], tabular_emb[i]) 
                  for i in range(batch_size)])
```

**Interpretation:**
- Random embeddings: alignment ≈ 0
- Perfect alignment: alignment = 1
- Collapse (all same): alignment = 1 but high loss

**Target:** Alignment should increase, but not too fast (avoid collapse).

### Periodic: Sector Silhouette Score

**What it measures:** Do embeddings cluster by GICS sector?

**Formula:**
```python
from sklearn.metrics import silhouette_score

# Compute embeddings for validation set
embeddings = get_joint_embeddings(model, val_dataset)  # (N, 256)
labels = [sample['gsector'] for sample in val_dataset]  # 11 sectors

silhouette = silhouette_score(embeddings, labels)
```

**Interpretation:**
- Range: [-1, 1]
- +1: Perfect sector clustering
- 0: Random clustering
- -1: Inverted clustering

**Target:** Silhouette > 0.1 indicates geometry makes financial sense.

---

## Inference Documentation

### Standalone Preprocessing Pipeline

**Purpose:** Compute features for ANY stock (not just training set) for real-time inference.

**Location:** `src/training/preprocessor.py`

**Usage:**

```python
from src.training.preprocessor import InferencePreprocessor
from src.data.wrds_loader import WRDSLoader

# Initialize
loader = WRDSLoader()
preprocessor = InferencePreprocessor()

# For a new stock you want recommendations for:
raw_data = loader.load_prices(symbols=['TSLA'], 
                                start_date='2023-01-01',
                                end_date='2024-01-01')

# Compute features
features = preprocessor.compute_features(raw_data)
# Returns dict with:
#   'temporal': (60, 13) - ready for model input
#   'tabular_cont': (15,)
#   'tabular_cat': (2,)

# Encode
model.eval()
with torch.no_grad():
    embedding = model.get_joint_embedding(
        features['temporal'].unsqueeze(0),
        features['tabular_cont'].unsqueeze(0),
        features['tabular_cat'].unsqueeze(0)
    )  # (1, 256)

# Search for similar stocks in pre-computed universe
similar_stocks = search_nearest_neighbors(embedding, universe_embeddings, k=10)
```

### Feature Computation for New Stocks

**Step-by-step for inference preprocessing:**

```python
# 1. Load raw data
prices = wrds_loader.load_prices(symbols=['TARGET'], 
                                  start_date='2023-01-01', 
                                  end_date='2024-01-01')
fundamentals = wrds_loader.load_fundamentals(symbols=['TARGET'],
                                               start_date='2023-01-01',
                                               end_date='2024-01-01')

# 2. Engineer features
engineer = FeatureEngineer()
features = engineer.compute_all_features(prices, fundamentals)

# 3. Extract latest window
temporal_window = features['technical'].iloc[-60:]  # Last 60 days
tabular_snapshot = features[['valuation', 'market_risk', 'volatility', 
                          'momentum', 'sector']].iloc[-1]

# 4. Normalize (use training statistics)
temporal_norm = (temporal_window - temporal_mean) / temporal_std
tabular_norm = (tabular_snapshot - tabular_mean) / tabular_std

# 5. Convert to tensors
temporal_tensor = torch.tensor(temporal_norm.values, dtype=torch.float32)
tabular_cont_tensor = torch.tensor(tabular_norm[continuous_cols].values, 
                                   dtype=torch.float32)
tabular_cat_tensor = torch.tensor(tabular_snapshot[categorical_cols].values,
                                  dtype=torch.long)

# Ready for model input!
```

**Key Documentation Points:**

1. **Window requirement:** Need at least 252 days of history for feature computation (60-day features + 252-day beta)

2. **Normalization:** Must use training set statistics, not compute fresh means/stds

3. **Missing data:** TabMixer handles NaNs, but temporal sequences must be complete

4. **GICS codes:** Must fetch from Compustat (gsector, ggroup)

---

## Implementation Files

| File | Purpose |
|------|---------|
| `src/training/dataset.py` | FeatureDataset - load pre-computed features |
| `src/training/dataloader.py` | StockDataLoader with hard negative batching |
| `src/training/metrics.py` | compute_alignment_score, compute_sector_silhouette, compute_hard_negative_similarity |
| `src/training/validator.py` | ValidationFoldEvaluator - run on each val fold |
| `src/training/preprocessor.py` | InferencePreprocessor - standalone feature computation |
| `src/training/checkpoint.py` | ModelCheckpoint - save/load model states |
| `src/training/logger.py` | MetricsLogger - wandb/tensorboard integration |
| `scripts/preprocess_features.py` | One-time feature pre-computation |
| `scripts/train.py` | Main training script with all date splits |
| `scripts/validate.py` | Run validation on a specific fold |

---

## Next Steps

1. Write implementation plan using `writing-plans` skill
2. Implement in order: preprocessor → dataset → dataloader → metrics → training script
3. Test each component individually
4. Run full training pipeline on first validation fold
5. Iterate on hyperparameters

---

**Design approved?** If yes, proceed to implementation plan.
