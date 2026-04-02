# LiquidSearcher - Stock Substitute Recommendation System

A dual-encoder contrastive learning system for stock substitute recommendation using BiMT-TCN + TabMixer with InfoNCE loss.

## Overview

This project implements a CLIP-style contrastive learning system that learns joint embeddings from:
- **Temporal data**: 60-day OHLCV price patterns (13 features)
- **Tabular data**: Fundamentals + GICS sector embeddings (39 features total)

**Key Innovation**: InfoNCE loss aligns temporal and tabular embeddings of the same stock, pushing different stocks apart.

## Quick Start

### Prerequisites

```bash
# Install package with dependencies
pip install -e .

# Set WRDS credentials (for data collection)
export WRDS_USERNAME=your_username
export WRDS_PASSWORD=your_password

# Optional: Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

## Step 1: Data Preparation

### Option A: Use Existing WRDS Data (Recommended)

If you have access to WRDS (Wharton Research Data Services):

```bash
# Process full CRSP universe (~3,000-5,000 stocks)
python -m scripts.preprocess_features \
    --universe all_crsp \
    --start-date 2010-01-01 \
    --end-date 2024-12-31 \
    --batch-size 500 \
    --output data/processed/all_features.parquet
```

**Other universe options:**
```bash
# S&P 500 only
python -m scripts.preprocess_features --universe sp500 --output data/processed/sp500.parquet

# Russell 2000 (small-cap)
python -m scripts.preprocess_features --universe russell2000 --output data/processed/russell2000.parquet

# Combined S&P 500 + Russell 2000
python -m scripts.preprocess_features --universe combined --output data/processed/combined.parquet
```

### Option B: Use Mock Data (For Testing)

If you don't have WRDS access:

```bash
python -m scripts.preprocess_features \
    --use-mock \
    --start-date 2010-01-01 \
    --end-date 2024-12-31 \
    --output data/processed/mock_features.parquet
```

This generates synthetic data for 8 large-cap stocks (AAPL, MSFT, GOOGL, etc.) for testing the pipeline.

### Check Data Coverage

After preprocessing, analyze what data you have:

```bash
python analyze_coverage.py data/processed/all_features.parquet
```

This shows:
- Total symbols processed
- Percentage with beta/fundamentals/GICS
- Row-level coverage statistics
- List of symbols with missing data

## Step 2: Training

### Quick Training (1-2 epochs, for testing)

```bash
python -m scripts.train \
    --feature-dir data/processed \
    --train-start 2010-01-01 \
    --train-end 2018-12-31 \
    --val-start 2020-01-01 \
    --val-end 2020-12-31 \
    --epochs 2 \
    --batch-size 32 \
    --lr 1e-4
```

### Full Training (100 epochs, production)

```bash
python -m scripts.train \
    --feature-dir data/processed \
    --train-start 2010-01-01 \
    --train-end 2018-12-31 \
    --val-start 2020-01-01 \
    --val-end 2020-12-31 \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4 \
    --checkpoint-dir checkpoints
```

**Key parameters:**
- `--batch-size`: Larger = more stable negatives (recommend 32-128)
- `--lr`: Peak learning rate (default 1e-4, try 3e-4 to 1e-5 range)
- `--epochs`: More epochs = better convergence (default 100)

### Training with Custom Settings

```bash
# Smaller batch for limited RAM
python -m scripts.train --batch-size 16 --epochs 50

# Different learning rate
python -m scripts.train --lr 3e-4 --epochs 100

# Specific symbols only
python -m scripts.train --symbols AAPL MSFT GOOGL AMZN TSLA
```

## Step 3: Validation

After training, validate the model:

```bash
python -m scripts.validate \
    --checkpoint checkpoints/best_model.ckpt \
    --val-start 2021-01-01 \
    --val-end 2022-12-31 \
    --compute-silhouette
```

**Sanity checks:**
- ✓ Loss < 5.0
- ✓ Alignment > Hard Neg Similarity
- ✓ Hard Neg Similarity < 0.5

## Step 4: Inference

Use the trained model to find stock substitutes:

```python
from src.training.module import DualEncoderModule
import torch

# Load model
model = DualEncoderModule.load_from_checkpoint("checkpoints/best_model.ckpt")
model.eval()

# Prepare data for target stock
batch = {
    "temporal": torch.randn(1, 60, 13),      # 60-day OHLCV window
    "tabular_cont": torch.randn(1, 15),     # Fundamentals
    "tabular_cat": torch.tensor([[5, 510]]),  # GICS codes
}

# Get joint embedding (256-dim)
joint_emb = model.get_joint_embeddings(batch)

# Compare with universe
similarities = torch.matmul(joint_emb, universe_embeddings.t())
top_10 = torch.topk(similarities, k=10)
```

## Architecture

### Model Components

```
Temporal Data (60-day OHLCV)          Tabular Data (Fundamentals + GICS)
         ↓                                      ↓
    BiMT-TCN Encoder                    TabMixer Encoder
    (TCN + Transformer)                   (MLP-Mixer + Embeddings)
         ↓                                      ↓
    128-dim embedding                   128-dim embedding
         ↓                                      ↓
         └────────── InfoNCE Loss ──────────────┘
         
Inference: Concatenate → 256-dim → Similarity Search
```

### Key Features

- **BiMT-TCN**: Temporal convolution + Transformer (2 layers, 4 heads)
- **TabMixer**: MLP-Mixer with GICS embeddings (8-dim gsector, 16-dim ggroup)
- **InfoNCE Loss**: Contrastive learning aligning temporal/tabular views
- **Total params**: 681K (small, fast training)

## Project Structure

```
.
├── src/
│   ├── data/              # WRDS loading, credentials
│   ├── features/          # Feature engineering (OHLCV, momentum, volatility)
│   ├── models/            # Dual-encoder (BiMT-TCN + TabMixer)
│   ├── training/          # PyTorch Lightning training pipeline
│   └── utils/             # Memory management, helpers
├── scripts/
│   ├── preprocess_features.py   # Data collection from WRDS
│   ├── train.py                 # Training script
│   └── validate.py              # Model validation
├── tests/                 # Test suite (33 tests)
├── docs/                  # Documentation and plans
├── data/
│   ├── processed/         # Computed features (parquet files)
│   └── cache/            # Cache files
└── checkpoints/         # Model checkpoints
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_training_end_to_end.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Troubleshooting

### WRDS Connection Issues

```bash
# Test credentials
python -c "from src.data.credentials import validate_and_exit; validate_and_exit()"

# Check available tables
python -c "
import wrds
conn = wrds.Connection()
print(conn.list_tables(library='crsp'))
"
```

### Import Errors

```bash
# Ensure running from project root
pwd  # Should show: /path/to/LiquidSearcher

# Run as module (required)
python -m scripts.train  # ✓ Correct
python scripts/train.py  # ✗ Wrong
```

### Out of Memory

```bash
# Reduce batch size
python -m scripts.train --batch-size 16

# Reduce workers
python -m scripts.train --num-workers 0
```

### Missing Features

If you have missing fundamentals/GICS:
- This is normal (~5-10% of stocks lack Compustat coverage)
- The model uses masking to handle missing values
- Local betas are computed from price data (always available)

## Evaluation

The model is validated through a comprehensive evaluation framework. See `docs/EVALUATION_PLAN.md` for the complete implementation roadmap.

**Evaluation Blocks**:
- **Qualitative**: UMAP visualization, SHAP feature importance, manual inspection
- **Quantitative**: Recall@10, Crisis Period Spearman, Pairs Trading Strategy

```bash
# Run full evaluation suite (after implementation)
python -m scripts.run_evaluation \
    --checkpoint checkpoints/best_model.ckpt \
    --output-dir results
```

## Evaluation

The model is validated through a comprehensive evaluation framework. See `docs/EVALUATION_PLAN.md` for the complete implementation roadmap.

**Evaluation Blocks**:
- **Qualitative**: UMAP visualization, SHAP feature importance, manual inspection
- **Quantitative**: Recall@10, Crisis Period Spearman, Pairs Trading Strategy

```bash
# Run full evaluation suite (after implementation)
python -m scripts.run_evaluation \
    --checkpoint checkpoints/best_model.ckpt \
    --output-dir results
```

## Citation

```bibtex
@software{liquidsearcher,
  title={LiquidSearcher: Dual-Encoder Contrastive Learning for Stock Substitutes},
  author={Claude Code},
  year={2026},
  note={CLIP-style embeddings for financial similarity search}
}
```

## License

MIT License - See LICENSE file for details.

## Support

For issues and questions:
- Check docs/FEATURES_SUMMARY.md for detailed implementation notes
- Review docs/plans/ for architecture decisions
- Run analyze_coverage.py to debug data issues
