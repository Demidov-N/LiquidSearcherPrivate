# Liquidity Risk Management System — Project Setup & Framework Definition

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Initialize project structure, dependencies, and engineering frameworks for a liquidity-aware stock substitute recommendation system using dual-encoder contrastive learning.

**Architecture:** Python-based quantitative finance pipeline with PyTorch for deep learning (dual-encoder model), polars/pandas for data preprocessing, and modular architecture following MVP.md specifications. WRDS data integration with local caching and feature engineering pipeline.

**Tech Stack:** Python 3.11+, PyTorch, polars (primary), pandas (fallback), numpy, pytest, ruff, mypy, uv/pip

---

## Prerequisites & Research

**Framework Decisions (from online sources):**

1. **Deep Learning:** PyTorch (preferred for research/flexibility vs TensorFlow)
   - Dual-encoder architecture needs custom contrastive loss (InfoNCE, RankSCL)
   - PyTorch better for custom training loops and academic implementations
   - TCN/Transformer temporal encoders readily available

2. **Data Processing:** polars (primary), pandas (compatibility)
   - polars: 10-50x faster than pandas for large datasets (>100k rows)
   - CRSP daily data ~2,400 stocks × 252 days × 15 years = ~9M rows
   - Lazy evaluation for memory efficiency
   - pandas: only for WRDS library compatibility if needed

3. **Feature Engineering:** Custom pipeline with sklearn for scalers
   - Cross-sectional z-score, rank normalization
   - Rolling window computations (252-day)
   - Winsorization at 1%/99%

4. **Testing:** pytest with fixtures for mock data
5. **Linting/Type-checking:** ruff + mypy (AGENTS.md requirements)
6. **Environment:** uv (preferred per AGENTS.md) or pip

---

## Task 1: Initialize Project Structure

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `docs/plans/` (directory)

**Step 1: Create pyproject.toml with project metadata**

```toml
[project]
name = "liquidity-risk-system"
version = "0.1.0"
description = "Liquidity-aware stock substitute recommendation using dual-encoder contrastive learning"
authors = [{name = "Developer"}]
requires-python = ">=3.11"
dependencies = [
    "polars>=1.0.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "pytorch>=2.0.0",
    "scikit-learn>=1.3.0",
    "scipy>=1.11.0",
    "tqdm>=4.65.0",
    "pyarrow>=14.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
    "black>=23.0.0",
    "ipython>=8.0.0",
    "jupyter>=1.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
target-version = "py311"
line-length = 100
select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM"]
ignore = ["E501"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
```

**Step 2: Create .gitignore**

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/
.venv/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Data files (large)
data/raw/*.csv
data/raw/*.parquet
data/processed/*.csv
data/processed/*.parquet
data/*.csv
data/*.parquet
*.csv
*.parquet

# WRDS credentials
.env
config/wrds_credentials.yaml

# Jupyter
.ipynb_checkpoints/
*.ipynb

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Model artifacts
models/*.pt
models/*.pth
models/*.pkl
models/checkpoints/
```

**Step 3: Create README.md with project overview**

```markdown
# Liquidity Risk Management System

Stock substitute recommendation system for liquidity-constrained portfolio management.

## Problem
When small-cap positions become illiquid (spread spikes, volume drops), portfolio managers must find liquid substitutes that preserve risk/return characteristics.

## Solution
Dual-encoder contrastive learning model:
- **Temporal Encoder**: TCN/Transformer on OHLCV price behavior (G5 features)
- **Tabular Encoder**: FT-Transformer on fundamentals + risk factors (G1-G4 + G6)
- **RankSCL**: Ordinal similarity ranking
- **Liquidity Filters**: 7 hard constraints (Amihud, DArLiQ, spread, volume)

## Quick Start
\`\`\`bash
# Install dependencies
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
\`\`\`

## Project Structure
- `src/`: Source code
  - `data/`: Data loading and WRDS integration
  - `features/`: Feature engineering (G1-G6)
  - `models/`: Dual-encoder architecture
  - `liquidity/`: Liquidity metrics and filters
  - `pipeline/`: End-to-end inference pipeline
- `tests/`: Test files
- `notebooks/`: Analysis notebooks
- `data/`: Data storage (not committed)
- `docs/`: Documentation and plans

## Data Requirements
- WRDS access (CRSP, Compustat, IBES, TAQ)
- Universe: Russell 2000 + S&P 400 (~2,400 stocks)
- Period: 2010–2024
- Fallback: ML-estimated spreads if TAQ unavailable

## License
MIT
```

**Step 4: Create docs/plans/ directory**

```bash
mkdir -p docs/plans
```

**Step 5: Commit**

```bash
git add pyproject.toml .gitignore README.md docs/plans/
git commit -m "chore: initialize project structure and dependencies"
```

---

## Task 2: Install Dependencies

**Files:**
- Create: `requirements.txt` (fallback)
- Modify: project environment

**Step 1: Create requirements.txt for compatibility**

```
polars>=1.0.0
pandas>=2.0.0
numpy>=1.24.0
torch>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
tqdm>=4.65.0
pyarrow>=14.0.0
pytest>=7.4.0
pytest-cov>=4.1.0
ruff>=0.1.0
mypy>=1.5.0
ipython>=8.0.0
jupyter>=1.0.0
```

**Step 2: Install with uv (preferred)**

```bash
uv pip install -e ".[dev]"
```

**Step 3: Verify installation**

```bash
python -c "import polars; print('polars:', polars.__version__)"
python -c "import torch; print('torch:', torch.__version__)"
python -c "import pandas; print('pandas:', pandas.__version__)"
```

Expected output:
```
polars: 1.x.x
torch: 2.x.x
pandas: 2.x.x
```

**Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add requirements.txt and verify dependencies"
```

---

## Task 3: Create Source Directory Structure

**Files:**
- Create: `src/__init__.py`
- Create: `src/data/__init__.py`
- Create: `src/features/__init__.py`
- Create: `src/models/__init__.py`
- Create: `src/liquidity/__init__.py`
- Create: `src/pipeline/__init__.py`

**Step 1: Create module structure**

```python
# src/__init__.py
"""Liquidity Risk Management System."""

__version__ = "0.1.0"
```

```python
# src/data/__init__.py
"""Data loading and WRDS integration."""
```

```python
# src/features/__init__.py
"""Feature engineering pipeline (G1-G6)."""
```

```python
# src/models/__init__.py
"""Dual-encoder contrastive learning models."""
```

```python
# src/liquidity/__init__.py
"""Liquidity metrics and hard filters."""
```

```python
# src/pipeline/__init__.py
"""End-to-end inference pipeline."""
```

**Step 2: Commit**

```bash
git add src/
git commit -m "chore: create source module structure"
```

---

## Task 4: Create Test Directory Structure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/data/__init__.py`
- Create: `tests/features/__init__.py`
- Create: `tests/models/__init__.py`
- Create: `tests/liquidity/__init__.py`

**Step 1: Create test configuration**

```python
# tests/__init__.py
"""Test suite for liquidity risk management system."""
```

```python
# tests/conftest.py
"""Pytest configuration and fixtures."""

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    dates = pl.date_range(
        start="2020-01-01",
        end="2020-12-31",
        interval="1d",
        eager=True
    )
    n_days = len(dates)
    
    return pl.DataFrame({
        "date": dates,
        "permno": [1] * n_days,
        "prc": np.random.randn(n_days).cumsum() + 100,
        "vol": np.random.randint(1000, 100000, n_days),
        "bidlo": np.random.randn(n_days).cumsum() + 99,
        "askhi": np.random.randn(n_days).cumsum() + 101,
    })


@pytest.fixture
def sample_fundamental_data():
    """Generate sample fundamental data for testing."""
    return pl.DataFrame({
        "permno": [1, 2, 3],
        "year": [2020, 2020, 2020],
        "at": [1000.0, 2000.0, 1500.0],
        "seq": [500.0, 1000.0, 750.0],
        "ni": [50.0, 100.0, 75.0],
        "csho": [100.0, 200.0, 150.0],
        "prcc_f": [10.0, 20.0, 15.0],
    })
```

**Step 2: Create test module init files**

```python
# tests/data/__init__.py
# tests/features/__init__.py
# tests/models/__init__.py
# tests/liquidity/__init__.py
```

**Step 3: Verify pytest works**

```bash
python -m pytest tests/ -v
```

Expected: No tests found but no errors (exit code 0)

**Step 4: Commit**

```bash
git add tests/
git commit -m "chore: create test structure with fixtures"
```

---

## Task 5: Create Data Directory Structure

**Files:**
- Create: `data/raw/.gitkeep`
- Create: `data/processed/.gitkeep`
- Create: `data/cache/.gitkeep`

**Step 1: Create data directories**

```bash
mkdir -p data/raw data/processed data/cache
touch data/raw/.gitkeep data/processed/.gitkeep data/cache/.gitkeep
```

**Step 2: Commit**

```bash
git add data/
git commit -m "chore: create data directory structure"
```

---

## Task 6: Define Configuration System

**Files:**
- Create: `src/config/__init__.py`
- Create: `src/config/settings.py`
- Create: `config/default.yaml`
- Create: `tests/test_config.py`

**Step 1: Create config module**

```python
# src/config/__init__.py
"""Configuration management."""

from src.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
```

```python
# src/config/settings.py
"""Application settings and constants from MVP.md."""

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DataConfig:
    """Data configuration."""
    
    train_start: str = "2010-01-01"
    train_end: str = "2022-12-31"
    val_start: str = "2023-01-01"
    val_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    test_end: str = "2024-12-31"
    
    universe: List[str] = None
    
    def __post_init__(self):
        if self.universe is None:
            object.__setattr__(
                self, 
                'universe', 
                ['Russell 2000', 'S&P 400']
            )


@dataclass(frozen=True)
class FeatureConfig:
    """Feature engineering configuration."""
    
    # Windows
    beta_window: int = 60
    vol_windows: List[int] = None
    momentum_windows: List[int] = None
    rolling_z_window: int = 252
    
    # Winsorization
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99
    
    def __post_init__(self):
        if self.vol_windows is None:
            object.__setattr__(self, 'vol_windows', [20, 60])
        if self.momentum_windows is None:
            object.__setattr__(self, 'momentum_windows', [21, 63, 126])


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture configuration."""
    
    temporal_dim: int = 128
    fundamental_dim: int = 128
    joint_dim: int = 256
    temperature: float = 0.07
    batch_size: int = 256


@dataclass(frozen=True)
class LiquidityConfig:
    """Liquidity filter thresholds."""
    
    amihud_percentile: float = 0.30
    illiq_zscore_threshold: float = 2.0
    spread_bps_threshold: float = 50.0
    spread_vol_threshold: float = 20.0
    zero_return_threshold: float = 0.05


@dataclass(frozen=True)
class Settings:
    """Application settings."""
    
    data: DataConfig = None
    features: FeatureConfig = None
    model: ModelConfig = None
    liquidity: LiquidityConfig = None
    
    # Paths
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    cache_dir: Path = Path("data/cache")
    
    def __post_init__(self):
        if self.data is None:
            object.__setattr__(self, 'data', DataConfig())
        if self.features is None:
            object.__setattr__(self, 'features', FeatureConfig())
        if self.model is None:
            object.__setattr__(self, 'model', ModelConfig())
        if self.liquidity is None:
            object.__setattr__(self, 'liquidity', LiquidityConfig())


# Singleton instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get application settings (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
```

**Step 2: Create default YAML config**

```yaml
# config/default.yaml
data:
  train_period: ["2010-01-01", "2022-12-31"]
  val_period: ["2023-01-01", "2023-12-31"]
  test_period: ["2024-01-01", "2024-12-31"]
  universe: ["Russell 2000", "S&P 400"]

features:
  beta_window: 60
  vol_windows: [20, 60]
  momentum_windows: [21, 63, 126]
  rolling_z_window: 252
  winsorize_pct: [1, 99]

model:
  temporal_encoder: "TCN"  # or "Transformer"
  temporal_dim: 128
  fundamental_dim: 128
  joint_dim: 256
  temperature: 0.07
  loss: "RankSCL"  # or "InfoNCE"
  
liquidity:
  filters:
    amihud_percentile: 30
    illiq_zscore: 2.0
    spread_bps: 50
    spread_vol_bps: 20
    zero_return_pct: 5
```

**Step 3: Write config test**

```python
# tests/test_config.py
"""Test configuration system."""

from src.config.settings import get_settings, Settings


def test_settings_singleton():
    """Test that settings is a singleton."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_default_values():
    """Test default configuration values."""
    s = Settings()
    
    # Data config
    assert s.data.train_start == "2010-01-01"
    assert s.data.train_end == "2022-12-31"
    
    # Feature config
    assert s.features.beta_window == 60
    assert s.features.winsorize_lower == 0.01
    
    # Model config
    assert s.model.joint_dim == 256
    assert s.model.temperature == 0.07
    
    # Liquidity config
    assert s.liquidity.spread_bps_threshold == 50.0
```

**Step 4: Run test to verify**

```bash
python -m pytest tests/test_config.py -v
```

Expected: 2 passing tests

**Step 5: Commit**

```bash
git add src/config/ config/ tests/test_config.py
git commit -m "feat: add configuration system with MVP.md defaults"
```

---

## Task 7: Create Type Definitions

**Files:**
- Create: `src/types/__init__.py`
- Create: `src/types/common.py`

**Step 1: Create type definitions**

```python
# src/types/__init__.py
"""Type definitions for the system."""

from src.types.common import (
    StockID,
    DateStr,
    FeatureVector,
    EmbeddingVector,
    ShockEvent,
    SubstituteCandidate,
)

__all__ = [
    "StockID",
    "DateStr",
    "FeatureVector",
    "EmbeddingVector",
    "ShockEvent",
    "SubstituteCandidate",
]
```

```python
# src/types/common.py
"""Common type definitions."""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import polars as pl

# Basic types
StockID: TypeAlias = int  # CRSP PERMNO
DateStr: TypeAlias = str  # ISO format: "YYYY-MM-DD"
FeatureVector: TypeAlias = np.ndarray
EmbeddingVector: TypeAlias = np.ndarray  # 256-dim joint embedding


@dataclass(frozen=True)
class ShockEvent:
    """Liquidity shock event on a stock."""
    
    permno: StockID
    date: DateStr
    trigger_type: str  # "spread", "volume", "illiq"
    severity: float  # z-score or percentage
    
    def __repr__(self) -> str:
        return f"ShockEvent({self.permno}, {self.date}, {self.trigger_type})"


@dataclass
class SubstituteCandidate:
    """Recommended substitute stock with metrics."""
    
    permno: StockID
    similarity_score: float  # Cosine similarity
    rank_score: float  # RankSCL ordinal distance
    
    # Liquidity metrics
    amihud_illiq: float
    est_spread_bps: float
    dollar_volume: float
    
    # Risk preservation
    beta_diff: float
    vol_diff: float
    
    passed_filters: bool = False
```

**Step 2: Commit**

```bash
git add src/types/
git commit -m "chore: add type definitions for domain entities"
```

---

## Task 8: Create Utilities Module

**Files:**
- Create: `src/utils/__init__.py`
- Create: `src/utils/logging.py`
- Create: `src/utils/validation.py`

**Step 1: Create logging utilities**

```python
# src/utils/__init__.py
"""Utility functions."""

from src.utils.logging import get_logger, setup_logging
from src.utils.validation import validate_dataframe_schema

__all__ = ["get_logger", "setup_logging", "validate_dataframe_schema"]
```

```python
# src/utils/logging.py
"""Logging configuration."""

import logging
import sys
from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None
) -> None:
    """Configure logging for the application."""
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)
```

```python
# src/utils/validation.py
"""Data validation utilities."""

from typing import Dict, List, Set

import polars as pl

from src.utils.logging import get_logger

logger = get_logger(__name__)


def validate_dataframe_schema(
    df: pl.DataFrame,
    required_columns: Set[str],
    column_types: Dict[str, pl.DataType] | None = None,
    name: str = "DataFrame"
) -> bool:
    """Validate that a DataFrame has required columns and types.
    
    Args:
        df: DataFrame to validate
        required_columns: Set of column names that must exist
        column_types: Optional dict of column name to expected type
        name: Name for error messages
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    actual_cols = set(df.columns)
    missing = required_columns - actual_cols
    
    if missing:
        raise ValueError(
            f"{name} missing required columns: {missing}. "
            f"Have: {actual_cols}"
        )
    
    if column_types:
        for col, expected_type in column_types.items():
            if col in df.columns:
                actual_type = df.schema[col]
                if actual_type != expected_type:
                    logger.warning(
                        f"{name} column {col} has type {actual_type}, "
                        f"expected {expected_type}"
                    )
    
    return True
```

**Step 2: Commit**

```bash
git add src/utils/
git commit -m "chore: add logging and validation utilities"
```

---

## Task 9: Create Constants Module

**Files:**
- Create: `src/constants/__init__.py`
- Create: `src/constants/features.py`
- Create: `src/constants/liquidity.py`

**Step 1: Create feature constants**

```python
# src/constants/__init__.py
"""Constants from MVP.md specification."""

from src.constants.features import (
    FACTOR_GROUPS,
    NORMALIZATION_METHODS,
    SYSTEMATIC_RISK_FEATURES,
    VOLATILITY_FEATURES,
    MOMENTUM_FEATURES,
    FUNDAMENTAL_FEATURES,
    OHLCV_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.constants.liquidity import (
    LIQUIDITY_GATES,
    SHOCK_TRIGGERS,
)

__all__ = [
    "FACTOR_GROUPS",
    "NORMALIZATION_METHODS",
    "SYSTEMATIC_RISK_FEATURES",
    "VOLATILITY_FEATURES",
    "MOMENTUM_FEATURES",
    "FUNDAMENTAL_FEATURES",
    "OHLCV_FEATURES",
    "CATEGORICAL_FEATURES",
    "LIQUIDITY_GATES",
    "SHOCK_TRIGGERS",
]
```

```python
# src/constants/features.py
"""Feature group definitions from MVP.md Stage 1."""

from typing import Dict, List, Tuple

# Factor Groups (G1-G6) from MVP.md
FACTOR_GROUPS = {
    "G1": "Systematic Risk Exposure",  # Beta, FF5 loadings
    "G2": "Volatility Profile",        # Realized vol, idio vol
    "G3": "Return Momentum & Trend",   # Momentum windows
    "G4": "Valuation & Fundamentals", # P/E, P/B, ROE
    "G5": "Multi-Scale OHLCV",        # Temporal price behavior
    "G6": "Sector / Industry",        # GICS codes
}

# Normalization methods from MVP.md table
NORMALIZATION_METHODS = {
    "cs_zscore": "Cross-sectional z-score",
    "log_cs_zscore": "Log then cross-sectional z-score",
    "cs_rank": "Cross-sectional rank [0,1]",
    "rolling_zscore": "Rolling time-series z-score (252-day)",
    "embedding": "Learned trainable embedding",
    "winsorize": "Winsorize [1%, 99%] or [2%, 98%]",
}

# G1: Systematic Risk (MVP.md table)
SYSTEMATIC_RISK_FEATURES: List[Tuple[str, str, List[float], str]] = [
    ("market_beta_60d", "[-0.5, 3.0]", [0.01, 0.99], "cs_zscore"),
    ("downside_beta", "[-0.5, 3.0]", [0.01, 0.99], "cs_zscore"),
    ("smb_loading", "[-2, 2]", None, "cs_zscore"),
    ("hml_loading", "[-2, 2]", None, "cs_zscore"),
    ("mom_loading", "[-2, 2]", None, "cs_zscore"),
    ("rmw_loading", "[-2, 2]", None, "cs_zscore"),
    ("cma_loading", "[-2, 2]", None, "cs_zscore"),
]

# G2: Volatility
VOLATILITY_FEATURES: List[Tuple[str, str, str]] = [
    ("realized_vol_20d", "log(vol) → cs_zscore", "[0.005, 0.15]"),
    ("realized_vol_60d", "log(vol) → cs_zscore", "[0.005, 0.15]"),
    ("idiosyncratic_vol", "log(vol) → cs_zscore", "[0.003, 0.12]"),
    ("vol_of_vol", "Winsorize → cs_zscore", "Right-skewed"),
]

# G3: Momentum (all use rank normalization)
MOMENTUM_FEATURES: List[Tuple[str, str]] = [
    ("mom_1m", "[-30%, +50%]"),
    ("mom_3m", "[-40%, +80%]"),
    ("mom_6m", "[-50%, +150%]"),
    ("mom_12_1m", "[-60%, +200%]"),  # Jegadeesh-Titman
    ("macd", "[-5, +5]"),
]

# G4: Fundamentals
FUNDAMENTAL_FEATURES: List[Tuple[str, str, str]] = [
    ("log_mktcap", "log → cs_zscore", "[$50M, $3T]"),
    ("pe_ratio", "Winsorize [2%,98%] → cs_rank", "[5, 200]+"),
    ("pb_ratio", "log → cs_zscore", "[0.5, 50]+"),
    ("roe", "Winsorize [2%,98%] → cs_zscore", "[-50%, +80%]"),
    ("earnings_quality", "cs_zscore", "[-1, 1]"),
]

# G5: OHLCV (SimStock features)
OHLCV_FEATURES: List[Tuple[str, List[int], str]] = [
    ("z_close", [5, 10, 20], "rolling_zscore"),
    ("z_high", [], "rolling_zscore"),
    ("z_low", [], "rolling_zscore"),
    ("z_volume", [5, 10, 20], "rolling_zscore"),
    ("ma_ratio", [5, 10, 15, 20, 25], "rolling_zscore"),
]

# G6: Categorical
CATEGORICAL_FEATURES: Dict[str, Tuple[int, int]] = {
    "gics_sector": (11, 8),           # 11 classes, 8-dim embed
    "gics_industry_group": (25, 16),  # 25 classes, 16-dim embed
}
```

```python
# src/constants/liquidity.py
"""Liquidity filter and shock trigger constants from MVP.md."""

from typing import Dict, Tuple

# Shock trigger conditions (MVP.md)
SHOCK_TRIGGERS: Dict[str, Tuple[str, float]] = {
    "spread_spike": ("bid_ask_spread_pct", 2.0),  # > 2% of stock price
    "volume_crater": ("volume_drop_pct", 0.50),  # > 50% vs 20-day MA
    "illiq_innovation": ("darliq_zscore", 2.0),  # z-score > 2.0
}

# Liquidity hard filter gates (Stage 4, MVP.md table)
LIQUIDITY_GATES: Dict[str, Dict] = {
    "amihud_illiq": {
        "metric": "|R_daily| / DollarVolume",
        "threshold": "Bottom 30% of sector ILLIQ",
        "source": "BIS WP 1229 (2024)",
    },
    "illiq_innovation": {
        "metric": "DArLiQ autoregressive residual",
        "threshold": "z-score < 2.0 in last 10 days",
        "source": "DArLiQ model (2023)",
    },
    "dollar_volume": {
        "metric": "Price × Volume, 20-day avg",
        "threshold": "≥ 2× target stock's dollar volume",
        "source": "Market impact constraint",
    },
    "est_bid_ask_spread": {
        "metric": "ML-estimated effective spread from OHLCV",
        "threshold": "< 50 bps",
        "source": "ScienceDirect ML est. (2025)",
    },
    "spread_volatility": {
        "metric": "Rolling 5-day std of estimated spread",
        "threshold": "< 20 bps",
        "source": "L-VaR framework (SSRN 2021)",
    },
    "zero_return_ratio": {
        "metric": "Fraction of zero-return days in 60-day",
        "threshold": "< 5%",
        "source": "Structural illiquidity proxy",
    },
    "turnover_ratio": {
        "metric": "Volume / shares_outstanding",
        "threshold": "> target stock's turnover",
        "source": "Float-normalized activity",
    },
}

# Target output (MVP.md)
TARGET_SUBSTITUTES_MIN = 5
TARGET_SUBSTITUTES_MAX = 10
CANDIDATES_PRE_FILTER = 50  # Top 20-50 from similarity
```

**Step 2: Commit**

```bash
git add src/constants/
git commit -m "feat: add constants module with MVP.md specifications"
```

---

## Task 10: Setup Linting and Type Checking

**Files:**
- Create: `Makefile`
- Modify: `pyproject.toml` (add tool configs if not present)

**Step 1: Create Makefile with common commands**

```makefile
.PHONY: help install test lint format typecheck clean

help:
	@echo "Available commands:"
	@echo "  install      - Install dependencies"
	@echo "  test         - Run all tests"
	@echo "  lint         - Run ruff linter"
	@echo "  format       - Format code with ruff"
	@echo "  typecheck    - Run mypy type checker"
	@echo "  all-checks   - Run lint + typecheck + test"
	@echo "  clean        - Remove cache files"

install:
	uv pip install -e ".[dev]"

test:
	python -m pytest tests/ -v --tb=short

lint:
	python -m ruff check src/ tests/

format:
	python -m ruff format src/ tests/

typecheck:
	python -m mypy src/ --ignore-missing-imports

all-checks: lint typecheck test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache
```

**Step 2: Verify ruff configuration works**

```bash
python -m ruff check src/ --select E,W,I
```

Expected: No errors (exit code 0)

**Step 3: Run format check**

```bash
python -m ruff format src/ --check
```

**Step 4: Commit**

```bash
git add Makefile
git commit -m "chore: add Makefile with lint, format, test commands"
```

---

## Task 11: Create Documentation Structure

**Files:**
- Create: `docs/architecture.md`
- Create: `docs/data-sources.md`
- Create: `docs/implementation-notes.md`

**Step 1: Create architecture documentation**

```markdown
# System Architecture

## Overview

Two-stage architecture separating similarity learning from liquidity constraints.

## Stage Breakdown

### Stage 0: Data Collection
- WRDS integration (CRSP, Compustat, IBES)
- Local caching with parquet format
- Universe: Russell 2000 + S&P 400

### Stage 1: Feature Engineering
- 6 feature groups (G1-G6)
- Normalization: z-score, rank, rolling
- Output: 256-dim temporal + fundamental vectors

### Stage 2: Dual-Encoder Model
- Temporal Encoder: TCN/Transformer on OHLCV (G5)
- Tabular Encoder: FT-Transformer on factors (G1-G4, G6)
- Loss: InfoNCE or RankSCL
- Output: 256-dim joint embeddings

### Stage 3: Substitute Ranking
- Cosine similarity search
- RankSCL re-ranking
- Top 20-50 candidates

### Stage 4: Liquidity Filtering
- 7 hard gates (binary pass/fail)
- Output: 5-10 liquid substitutes

### Stage 5: Backtesting
- Stress event validation
- Metrics: tracking error, cost savings
```

**Step 2: Create data sources documentation**

```markdown
# Data Sources

## Primary: WRDS (via Brown University)

### Datasets

| Dataset | Library | Key Fields | Frequency |
|---------|---------|------------|-----------|
| CRSP Daily | crsp | PRC, RET, VOL, SHROUT, BIDLO, ASKHI | Daily |
| Compustat Annual | comp | AT, SEQ, NI, CSHO, PRCC_F, EPSPX | Annual |
| Compustat Quarterly | comp | Same | Quarterly |
| TAQ | taq | BID, ASK, PRICE, SIZE | Intraday |
| IBES Summary | ibes | EPS estimates, actuals | Event |
| OptionMetrics | optionm | Implied vol | Daily |

## Free Sources

- Ken French FF5: mba.tuck.dartmouth.edu/pages/faculty/ken.french
- GICS mappings: S&P Dow Jones Indices

## Fallback Strategy

If TAQ unavailable: Use ML-estimated spreads from OHLCV (ScienceDirect 2025)
```

**Step 3: Create implementation notes**

```markdown
# Implementation Notes

## Tech Stack Decisions

### Why PyTorch?
- Flexibility for custom contrastive losses (RankSCL)
- Better academic paper reproduction
- Native TCN/Transformer support

### Why Polars?
- 10-50x faster than pandas for large datasets
- ~9M rows (2,400 stocks × 252 days × 15 years)
- Lazy evaluation for memory efficiency
- Pandas only for WRDS library compatibility

### Why Not TensorFlow?
- More boilerplate for custom training loops
- PyTorch dominates quantitative finance research

## Data Pipeline Design

1. **Lazy Loading**: polars.LazyFrame for transformations
2. **Caching**: Parquet format with snappy compression
3. **Incremental**: Only fetch missing dates
4. **Validation**: Schema checks at each stage

## Model Architecture Notes

- Dual-encoder: Do NOT share weights between encoders
- Temperature: 0.07 (InfoNCE default)
- Batch size: 256-512 (GPU memory permitting)
- Embedding dim: 256 (128 per encoder)

## Liquidity Constraint Design

**Critical Rule**: Liquidity is a constraint, NOT a similarity dimension.
- Do NOT embed liquidity features
- Hard filters (binary) applied AFTER ranking
- This prevents recommending similarly illiquid stocks
```

**Step 4: Commit**

```bash
git add docs/
git commit -m "docs: add architecture, data sources, and implementation notes"
```

---

## Task 12: Create Empty Module Placeholders

**Files:**
- Create: `src/data/wrds_loader.py` (stub)
- Create: `src/data/cache_manager.py` (stub)
- Create: `src/features/normalization.py` (stub)
- Create: `src/features/feature_pipeline.py` (stub)
- Create: `src/models/dual_encoder.py` (stub)
- Create: `src/models/losses.py` (stub)
- Create: `src/liquidity/metrics.py` (stub)
- Create: `src/liquidity/filters.py` (stub)
- Create: `src/pipeline/recommender.py` (stub)

**Step 1: Create stub files with docstrings**

```python
# src/data/wrds_loader.py
"""WRDS data loading and integration.

TODO: Implement WRDS connection using wrds library
TODO: Add query builders for CRSP, Compustat, IBES
TODO: Implement incremental data fetching
"""


class WRDSLoader:
    """Load data from Wharton Research Data Services."""
    
    def __init__(self, username: str | None = None):
        """Initialize WRDS connection.
        
        Args:
            username: WRDS username (None to use environment)
        """
        raise NotImplementedError("WRDSLoader not yet implemented")
```

```python
# src/data/cache_manager.py
"""Local data caching with parquet format.

TODO: Implement parquet read/write with polars
TODO: Add metadata tracking for cache freshness
TODO: Implement cache invalidation logic
"""

from pathlib import Path

import polars as pl

from src.config.settings import get_settings


class CacheManager:
    """Manage local data cache in parquet format."""
    
    def __init__(self, cache_dir: Path | None = None):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = cache_dir or get_settings().cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str) -> pl.DataFrame | None:
        """Load DataFrame from cache if exists."""
        raise NotImplementedError("CacheManager.get not yet implemented")
    
    def put(self, key: str, df: pl.DataFrame) -> None:
        """Save DataFrame to cache."""
        raise NotImplementedError("CacheManager.put not yet implemented")
```

```python
# src/features/normalization.py
"""Feature normalization methods from MVP.md.

TODO: Implement winsorization
TODO: Implement cross-sectional z-score
TODO: Implement cross-sectional rank
TODO: Implement rolling z-score
"""

import polars as pl


def winsorize(
    df: pl.DataFrame,
    column: str,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99
) -> pl.DataFrame:
    """Winsorize values at percentiles.
    
    Args:
        df: Input DataFrame
        column: Column to winsorize
        lower_pct: Lower percentile (default 1%)
        upper_pct: Upper percentile (default 99%)
        
    Returns:
        DataFrame with winsorized column
    """
    raise NotImplementedError("winsorize not yet implemented")


def cross_sectional_zscore(
    df: pl.DataFrame,
    column: str,
    date_col: str = "date"
) -> pl.DataFrame:
    """Compute cross-sectional z-score.
    
    Args:
        df: Input DataFrame
        column: Column to normalize
        date_col: Date column for grouping
        
    Returns:
        DataFrame with z-scored column
    """
    raise NotImplementedError("cross_sectional_zscore not yet implemented")
```

```python
# src/features/feature_pipeline.py
"""End-to-end feature engineering pipeline.

TODO: Implement G1: Systematic risk features
TODO: Implement G2: Volatility features
TODO: Implement G3: Momentum features
TODO: Implement G4: Fundamental features
TODO: Implement G5: OHLCV features
TODO: Implement G6: Categorical embeddings
"""

from src.config.settings import Settings


class FeaturePipeline:
    """Compute all feature groups for dual-encoder model."""
    
    def __init__(self, config: Settings | None = None):
        """Initialize pipeline with configuration."""
        raise NotImplementedError("FeaturePipeline not yet implemented")
```

```python
# src/models/dual_encoder.py
"""Dual-encoder contrastive learning architecture.

TODO: Implement TemporalEncoder (TCN or Transformer)
TODO: Implement TabularEncoder (FT-Transformer)
TODO: Implement DualEncoder combining both
TODO: Add forward pass for inference
"""

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """Encode OHLCV temporal sequences."""
    
    def __init__(self, input_dim: int, embed_dim: int = 128):
        super().__init__()
        raise NotImplementedError("TemporalEncoder not yet implemented")


class TabularEncoder(nn.Module):
    """Encode tabular fundamental features."""
    
    def __init__(self, input_dim: int, embed_dim: int = 128):
        super().__init__()
        raise NotImplementedError("TabularEncoder not yet implemented")


class DualEncoder(nn.Module):
    """Joint dual-encoder model."""
    
    def __init__(
        self,
        temporal_input_dim: int,
        tabular_input_dim: int,
        temporal_embed_dim: int = 128,
        tabular_embed_dim: int = 128,
        joint_dim: int = 256
    ):
        super().__init__()
        raise NotImplementedError("DualEncoder not yet implemented")
```

```python
# src/models/losses.py
"""Contrastive loss functions.

TODO: Implement InfoNCE loss (baseline)
TODO: Implement RankSCL loss (proposed)
TODO: Implement SoftCLT loss (optional)
"""

import torch
import torch.nn as nn


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        temporal_embed: torch.Tensor,
        tabular_embed: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("InfoNCELoss not yet implemented")


class RankSCLLoss(nn.Module):
    """Rank-supervised contrastive learning loss."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        embeddings: torch.Tensor,
        ranks: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("RankSCLLoss not yet implemented")
```

```python
# src/liquidity/metrics.py
"""Liquidity metrics computation.

TODO: Implement Amihud ILLIQ
TODO: Implement DArLiQ decomposition
TODO: Implement ML spread estimator
TODO: Implement zero-return ratio
TODO: Implement turnover ratio
"""

import polars as pl


def compute_amihud_illiq(
    df: pl.DataFrame,
    return_col: str = "ret",
    volume_col: str = "vol",
    price_col: str = "prc"
) -> pl.DataFrame:
    """Compute Amihud illiquidity measure.
    
    ILLIQ = |R_daily| / DollarVolume
    
    Args:
        df: DataFrame with returns and volume
        return_col: Daily return column
        volume_col: Volume column
        price_col: Price column
        
    Returns:
        DataFrame with illiq column added
    """
    raise NotImplementedError("compute_amihud_illiq not yet implemented")


def compute_darliq_innovation(
    illiq_series: pl.Series,
    window: int = 10
) -> pl.Series:
    """Compute DArLiQ innovation (residual).
    
    Decomposes ILLIQ into trend + AR component.
    Returns the residual (shock signal).
    
    Args:
        illiq_series: Amihud ILLIQ time series
        window: AR window for short-run component
        
    Returns:
        Series of innovation z-scores
    """
    raise NotImplementedError("compute_darliq_innovation not yet implemented")
```

```python
# src/liquidity/filters.py
"""Hard liquidity filters (Stage 4).

TODO: Implement all 7 liquidity gates
TODO: Add batch filtering for candidates
TODO: Add filter result reporting
"""

from typing import List

from src.types.common import SubstituteCandidate


class LiquidityFilter:
    """Apply hard liquidity constraints to candidates."""
    
    def __init__(self):
        """Initialize filter with MVP.md thresholds."""
        raise NotImplementedError("LiquidityFilter not yet implemented")
    
    def filter_candidates(
        self,
        candidates: List[SubstituteCandidate],
        target_liquidity: dict
    ) -> List[SubstituteCandidate]:
        """Filter candidates through all gates.
        
        Args:
            candidates: List of substitute candidates
            target_liquidity: Target stock liquidity metrics
            
        Returns:
            Filtered list passing all gates
        """
        raise NotImplementedError("filter_candidates not yet implemented")
```

```python
# src/pipeline/recommender.py
"""End-to-end substitute recommendation pipeline.

TODO: Implement shock detection
TODO: Implement embedding computation
TODO: Implement similarity search
TODO: Implement re-ranking
TODO: Implement filtering
TODO: Implement result formatting
"""

from typing import List

from src.models.dual_encoder import DualEncoder
from src.liquidity.filters import LiquidityFilter
from src.types.common import ShockEvent, SubstituteCandidate


class SubstituteRecommender:
    """Full pipeline: shock → substitutes."""
    
    def __init__(
        self,
        model: DualEncoder,
        liquidity_filter: LiquidityFilter
    ):
        """Initialize recommender with trained model."""
        raise NotImplementedError("SubstituteRecommender not yet implemented")
    
    def recommend(
        self,
        shock: ShockEvent,
        n_candidates: int = 50,
        n_output: int = 10
    ) -> List[SubstituteCandidate]:
        """Get substitute recommendations for a shock event.
        
        Args:
            shock: Detected liquidity shock
            n_candidates: Number of initial candidates
            n_output: Final number of substitutes
            
        Returns:
            List of recommended substitutes
        """
        raise NotImplementedError("recommend not yet implemented")
```

**Step 2: Commit**

```bash
git add src/data/wrds_loader.py src/data/cache_manager.py
git add src/features/normalization.py src/features/feature_pipeline.py
git add src/models/dual_encoder.py src/models/losses.py
git add src/liquidity/metrics.py src/liquidity/filters.py
git add src/pipeline/recommender.py
git commit -m "chore: add module stubs for all system components"
```

---

## Task 13: Verify Full Setup

**Step 1: Run all checks**

```bash
make all-checks
```

Expected:
- ruff passes with no errors
- mypy passes (may have import errors for torch initially, that's OK)
- pytest passes (tests/test_config.py should work)

**Step 2: Test imports**

```bash
python -c "
from src.config import get_settings
from src.constants import FACTOR_GROUPS
from src.types import StockID, ShockEvent
print('All imports successful')
"
```

Expected: "All imports successful"

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: complete project setup and framework initialization"
```

---

## Summary

This plan sets up:

1. **Dependencies**: PyTorch, polars, pandas, pytest, ruff, mypy
2. **Project Structure**: Modular architecture (data, features, models, liquidity, pipeline)
3. **Configuration**: Settings system with MVP.md defaults
4. **Testing**: pytest with fixtures
5. **Types**: Domain-specific type definitions
6. **Constants**: MVP.md feature groups and liquidity gates
7. **Utilities**: Logging and validation
8. **Documentation**: Architecture, data sources, implementation notes
9. **Stubs**: All module placeholders ready for implementation
10. **Tooling**: Makefile with lint, format, test, typecheck commands

**Next Steps After Setup:**
- Implement WRDS data loading (Task 1 of next plan)
- Implement feature engineering pipeline (G1-G6)
- Build dual-encoder model
- Create backtesting framework

