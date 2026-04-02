# Data Collection and Feature Processing Pipeline

This pipeline fetches data from WRDS in memory-efficient batches and computes features for the stock substitute recommendation system.

## Architecture

```
Symbols (2,400) 
    ↓
[Batch 1: 750 symbols] → WRDS → Features → Parquet
[Batch 2: 750 symbols] → WRDS → Features → Parquet
[Batch 3: 750 symbols] → WRDS → Features → Parquet
[Batch 4: 150 symbols] → WRDS → Features → Parquet
    ↓
Single Output: data/processed/all_features.parquet
```

## Key Features

- **Memory Efficient**: 30GB RAM limit, processes 750 symbols at a time
- **Progress Tracking**: tqdm progress bars for all operations
- **Pre-computed Betas**: Uses WRDS Beta Suite (no expensive computation)
- **Correct Date Handling**: Uses `rdq` (report date) not `datadate`
- **Incremental Writing**: Never holds all data in memory
- **Two-Pass Normalization**: Accurate cross-sectional z-scores

## Usage

### 1. Set WRDS Credentials

```bash
export WRDS_USERNAME=your_username
export WRDS_PASSWORD=your_password
```

### 2. Run Preprocessing

```bash
python -m scripts.preprocess_features \
    --start-date 2010-01-01 \
    --end-date 2024-12-31 \
    --batch-size 750 \
    --output data/processed/all_features.parquet
```

Options:
- `--batch-size`: Number of symbols per batch (default 750 for 30GB RAM)
- `--skip-betas`: Don't fetch pre-computed betas
- `--use-mock`: Use mock data for testing

### 3. Apply Normalization

```bash
python -m scripts.normalize_features \
    --input data/processed/all_features.parquet \
    --output data/processed/all_features_normalized.parquet
```

## Output Schema

The final parquet file contains:

**Identifiers:**
- `symbol`: Ticker symbol
- `date`: Trading date
- `permno`: CRSP permanent identifier

**G1 - Market Risk:**
- `beta`: 60-day rolling market beta (from WRDS Beta Suite)
- `downside_beta`: Beta on negative market days

**G2 - Volatility:**
- `realized_vol_20d`: 20-day realized volatility
- `realized_vol_60d`: 60-day realized volatility
- `idiosyncratic_vol`: Idiosyncratic volatility (from WRDS Beta Suite)

**G3 - Momentum:**
- `mom_1m`: 1-month momentum
- `mom_3m`: 3-month momentum
- `mom_6m`: 6-month momentum
- `mom_12_1m`: 12-month momentum skip last month

**G4 - Valuation:**
- `pe_ratio`: Price-to-earnings ratio
- `pb_ratio`: Price-to-book ratio
- `roe`: Return on equity
- `log_mktcap`: Log market capitalization

**G5 - OHLCV Technicals:**
- `z_close`: Time-series z-score of returns
- `z_volume`: Time-series z-score of volume changes
- `ma_ratio_5d`: Price / 5-day moving average
- `ma_ratio_10d`: Price / 10-day moving average
- `ma_ratio_20d`: Price / 20-day moving average

**G6 - Sector:**
- `gsector`: GICS sector code
- `ggroup`: GICS industry group code

**Normalized Features (suffix `_zscore`):**
All features have z-score normalized versions for model training.

## Memory Usage

With 750 symbols per batch:
- Peak RAM: ~8-12 GB
- Processing time: ~30-60 minutes per batch (depends on WRDS speed)
- Total time: ~2-4 hours for full universe (2,400 symbols)

## Troubleshooting

### WRDS Connection Issues

```bash
# Test credentials
python -c "from src.data.credentials import validate_and_exit; validate_and_exit()"
```

### Out of Memory

Reduce batch size:
```bash
python -m scripts.preprocess_features --batch-size 500
```

### Partial Failure

The script continues on batch failure. Check logs for failed symbols, then re-run with:
```bash
python -m scripts.preprocess_features --start-date 2023-01-01  # Retry specific dates
```
