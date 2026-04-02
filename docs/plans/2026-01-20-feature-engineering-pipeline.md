# Feature Engineering Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a comprehensive feature engineering pipeline that computes G1-G6 feature groups (risk, volatility, momentum, valuation, OHLCV behavior, and sector) with proper normalization for the stock substitute recommendation system.

**Architecture:** Modular feature engineering system with separate classes for each feature group (G1-G6). Each group handles its own computation and normalization. A unified `FeatureEngineer` orchestrates the pipeline, loading data via `WRDSDataLoader` and outputting a wide DataFrame with all features. Follows the normalization methods specified in MVP: cross-sectional z-scores, log transforms, rank normalization, and learned embeddings.

**Tech Stack:** Python 3.12, pandas (primary), numpy, scikit-learn (for rolling statistics), pytorch (for sector embeddings later), pytest

**Note:** This is a **PROTOTYPE** - prioritize functionality over perfection. Clean up, documentation, and edge cases can be handled later. Speed > perfection.

---

## Task 1: Create Feature Engineering Base Infrastructure

**Files:**
- Create: `src/features/base.py`
- Create: `src/features/__init__.py`
- Test: `tests/test_features_base.py`

**Step 1: Write failing test**

```python
# tests/test_features_base.py
"""Test feature engineering base classes."""

import pandas as pd
import pytest

from src.features.base import FeatureGroup, FeatureRegistry, NormalizationMethod


def test_normalization_method_enum():
    """Test normalization method enum."""
    assert NormalizationMethod.Z_SCORE.value == "z_score"
    assert NormalizationMethod.LOG_Z_SCORE.value == "log_z_score"
    assert NormalizationMethod.RANK.value == "rank"
    assert NormalizationMethod.ROLLING_Z_SCORE.value == "rolling_z_score"
    assert NormalizationMethod.NONE.value == "none"


def test_feature_group_base_class():
    """Test FeatureGroup base class interface."""
    
    class TestFeatureGroup(FeatureGroup):
        def compute(self, df: pd.DataFrame) -> pd.DataFrame:
            return df.assign(test_feature=1.0)
        
        def get_feature_names(self) -> list[str]:
            return ["test_feature"]
    
    group = TestFeatureGroup()
    assert group.get_feature_names() == ["test_feature"]
    
    test_df = pd.DataFrame({"symbol": ["AAPL"], "date": pd.to_datetime(["2020-01-01"])})
    result = group.compute(test_df)
    assert "test_feature" in result.columns


def test_feature_registry():
    """Test feature registry for managing feature groups."""
    registry = FeatureRegistry()
    
    class TestGroup(FeatureGroup):
        def compute(self, df: pd.DataFrame) -> pd.DataFrame:
            return df
        def get_feature_names(self) -> list[str]:
            return ["test"]
    
    registry.register("test", TestGroup())
    assert "test" in registry.list_groups()
    assert registry.get("test") is not None
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_features_base.py::test_normalization_method_enum -v
```

Expected: FAIL with "NormalizationMethod not defined"

**Step 3: Write minimal implementation**

```python
# src/features/base.py
"""Base classes and utilities for feature engineering."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class NormalizationMethod(Enum):
    """Normalization methods for features."""
    Z_SCORE = "z_score"
    LOG_Z_SCORE = "log_z_score"
    RANK = "rank"
    ROLLING_Z_SCORE = "rolling_z_score"
    NONE = "none"


class FeatureGroup(ABC):
    """Abstract base class for feature groups (G1-G6)."""
    
    def __init__(self, name: str, normalization: NormalizationMethod = NormalizationMethod.Z_SCORE):
        """Initialize feature group.
        
        Args:
            name: Name of the feature group (e.g., 'G1_risk')
            normalization: Default normalization method for this group
        """
        self.name = name
        self.normalization = normalization
    
    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features for this group.
        
        Args:
            df: Input DataFrame with raw data
            
        Returns:
            DataFrame with computed features added
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Return list of feature names this group produces."""
        pass
    
    def _winsorize(self, series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
        """Winsorize series at given percentiles."""
        lower_val = series.quantile(lower)
        upper_val = series.quantile(upper)
        return series.clip(lower=lower_val, upper=upper_val)
    
    def _cross_sectional_zscore(
        self, 
        df: pd.DataFrame, 
        column: str,
        date_col: str = "date",
    ) -> pd.Series:
        """Compute cross-sectional z-score for a column.
        
        Normalizes values within each date across all stocks.
        """
        def zscore(group):
            mean = group[column].mean()
            std = group[column].std()
            if std == 0 or pd.isna(std):
                return pd.Series(0.0, index=group.index)
            return (group[column] - mean) / std
        
        return df.groupby(date_col, group_keys=False).apply(zscore)
    
    def _cross_sectional_rank(
        self,
        df: pd.DataFrame,
        column: str,
        date_col: str = "date",
    ) -> pd.Series:
        """Compute cross-sectional rank normalized to [0, 1]."""
        def rank_norm(group):
            ranks = group[column].rank(method="average")
            n = len(group)
            if n <= 1:
                return pd.Series(0.5, index=group.index)
            return (ranks - 1) / (n - 1)
        
        return df.groupby(date_col, group_keys=False).apply(rank_norm)
    
    def _rolling_zscore(
        self,
        series: pd.Series,
        window: int = 252,
        min_periods: int = 60,
    ) -> pd.Series:
        """Compute rolling time-series z-score.
        
        Uses stock's own history for normalization.
        """
        rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = series.rolling(window=window, min_periods=min_periods).std()
        
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)
        
        zscore = (series - rolling_mean) / rolling_std
        return zscore.fillna(0)


class FeatureRegistry:
    """Registry for managing feature groups."""
    
    def __init__(self):
        self._groups: dict[str, FeatureGroup] = {}
    
    def register(self, name: str, group: FeatureGroup) -> None:
        """Register a feature group."""
        self._groups[name] = group
    
    def get(self, name: str) -> Optional[FeatureGroup]:
        """Get a registered feature group by name."""
        return self._groups.get(name)
    
    def list_groups(self) -> list[str]:
        """List all registered group names."""
        return list(self._groups.keys())
    
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features from all registered groups."""
        result = df.copy()
        for name, group in self._groups.items():
            result = group.compute(result)
        return result
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_features_base.py -v
```

Expected: 3 passing tests

**Step 5: Commit**

```bash
git add tests/test_features_base.py src/features/base.py src/features/__init__.py
git commit -m "feat: add feature engineering base infrastructure"
```

---

## Task 2: Implement G1 Systematic Risk Features

**Files:**
- Create: `src/features/g1_risk.py`
- Test: `tests/test_features_g1.py`

**Step 1: Write failing test**

```python
# tests/test_features_g1.py
"""Test G1 systematic risk features."""

import numpy as np
import pandas as pd
import pytest

from src.features.g1_risk import G1RiskFeatures


def test_g1_initialization():
    """Test G1 feature group initialization."""
    g1 = G1RiskFeatures()
    assert g1.name == "G1_risk"
    assert "market_beta_60d" in g1.get_feature_names()


def test_beta_computation():
    """Test beta computation."""
    # Create synthetic data with known beta
    np.random.seed(42)
    n_days = 100
    
    # Market returns
    market_ret = np.random.normal(0.001, 0.02, n_days)
    
    # Stock returns with beta = 1.5
    beta_true = 1.5
    stock_ret = beta_true * market_ret + np.random.normal(0, 0.01, n_days)
    
    df = pd.DataFrame({
        "symbol": ["TEST"] * n_days,
        "date": pd.date_range("2020-01-01", periods=n_days, freq="B"),
        "return": stock_ret,
        "market_return": market_ret,
    })
    
    g1 = G1RiskFeatures()
    result = g1.compute(df)
    
    assert "market_beta_60d" in result.columns
    # Beta should be approximately 1.5 for the last rows (after 60-day window)
    last_beta = result["market_beta_60d"].iloc[-1]
    assert not pd.isna(last_beta)
    assert 1.0 < last_beta < 2.0  # Should be close to 1.5


def test_factor_loadings():
    """Test factor loading computations."""
    np.random.seed(42)
    n_days = 100
    
    df = pd.DataFrame({
        "symbol": ["TEST"] * n_days,
        "date": pd.date_range("2020-01-01", periods=n_days, freq="B"),
        "return": np.random.normal(0.001, 0.02, n_days),
        "smb_factor": np.random.normal(0, 0.01, n_days),
        "hml_factor": np.random.normal(0, 0.01, n_days),
        "mom_factor": np.random.normal(0, 0.01, n_days),
        "rmw_factor": np.random.normal(0, 0.01, n_days),
        "cma_factor": np.random.normal(0, 0.01, n_days),
    })
    
    g1 = G1RiskFeatures()
    result = g1.compute(df)
    
    # Check all factor loading columns exist
    assert "smb_loading" in result.columns
    assert "hml_loading" in result.columns
    assert "mom_loading" in result.columns
    assert "rmw_loading" in result.columns
    assert "cma_loading" in result.columns
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_features_g1.py::test_g1_initialization -v
```

Expected: FAIL with "G1RiskFeatures not defined"

**Step 3: Write minimal implementation**

```python
# src/features/g1_risk.py
"""G1 Systematic Risk Features (Beta, Factor Loadings)."""

import numpy as np
import pandas as pd
from scipy import stats

from src.features.base import FeatureGroup, NormalizationMethod


class G1RiskFeatures(FeatureGroup):
    """Compute systematic risk features: beta and factor loadings."""
    
    def __init__(self):
        super().__init__("G1_risk", NormalizationMethod.Z_SCORE)
        self.lookback_60d = 60
        self.lookback_252d = 252
    
    def get_feature_names(self) -> list[str]:
        return [
            "market_beta_60d",
            "downside_beta",
            "smb_loading",
            "hml_loading",
            "mom_loading",
            "rmw_loading",
            "cma_loading",
        ]
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute G1 risk features."""
        result = df.copy()
        
        # Compute beta (60-day rolling)
        result = self._compute_beta(result, window=self.lookback_60d)
        
        # Compute downside beta
        result = self._compute_downside_beta(result, window=self.lookback_60d)
        
        # Compute factor loadings (252-day rolling for stability)
        result = self._compute_factor_loadings(result, window=self.lookback_252d)
        
        # Apply cross-sectional z-score normalization
        for col in self.get_feature_names():
            if col in result.columns:
                # Winsorize first
                result[col] = self._winsorize(result[col], lower=0.01, upper=0.99)
                # Then z-score
                result[col] = self._cross_sectional_zscore(result, col)
        
        return result
    
    def _compute_beta(
        self,
        df: pd.DataFrame,
        window: int = 60,
        stock_return_col: str = "return",
        market_return_col: str = "market_return",
    ) -> pd.DataFrame:
        """Compute rolling market beta via OLS regression."""
        
        def rolling_beta(group):
            if len(group) < window:
                group["market_beta_60d"] = np.nan
                return group
            
            stock_ret = group[stock_return_col].values
            market_ret = group[market_return_col].values
            
            betas = []
            for i in range(len(group)):
                if i < window - 1:
                    betas.append(np.nan)
                else:
                    y = stock_ret[i-window+1:i+1]
                    x = market_ret[i-window+1:i+1]
                    
                    # Add constant for regression
                    x_with_const = np.column_stack([np.ones(len(x)), x])
                    
                    try:
                        beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0][1]
                        betas.append(beta)
                    except:
                        betas.append(np.nan)
            
            group["market_beta_60d"] = betas
            return group
        
        return df.groupby("symbol", group_keys=False).apply(rolling_beta)
    
    def _compute_downside_beta(
        self,
        df: pd.DataFrame,
        window: int = 60,
        stock_return_col: str = "return",
        market_return_col: str = "market_return",
    ) -> pd.DataFrame:
        """Compute downside beta (beta on negative market return days only)."""
        
        def rolling_downside_beta(group):
            if len(group) < window:
                group["downside_beta"] = np.nan
                return group
            
            stock_ret = group[stock_return_col].values
            market_ret = group[market_return_col].values
            
            betas = []
            for i in range(len(group)):
                if i < window - 1:
                    betas.append(np.nan)
                else:
                    y = stock_ret[i-window+1:i+1]
                    x = market_ret[i-window+1:i+1]
                    
                    # Filter to negative market return days
                    mask = x < 0
                    if mask.sum() < 10:  # Need minimum observations
                        betas.append(np.nan)
                    else:
                        y_down = y[mask]
                        x_down = x[mask]
                        x_with_const = np.column_stack([np.ones(len(x_down)), x_down])
                        
                        try:
                            beta = np.linalg.lstsq(x_with_const, y_down, rcond=None)[0][1]
                            betas.append(beta)
                        except:
                            betas.append(np.nan)
            
            group["downside_beta"] = betas
            return group
        
        return df.groupby("symbol", group_keys=False).apply(rolling_downside_beta)
    
    def _compute_factor_loadings(
        self,
        df: pd.DataFrame,
        window: int = 252,
        stock_return_col: str = "return",
    ) -> pd.DataFrame:
        """Compute FF5 factor loadings via rolling regression."""
        
        factor_cols = ["smb_factor", "hml_factor", "mom_factor", "rmw_factor", "cma_factor"]
        
        # Check if factor columns exist
        available_factors = [f for f in factor_cols if f in df.columns]
        
        if not available_factors:
            # If no factor data, create placeholder columns
            for factor in ["smb", "hml", "mom", "rmw", "cma"]:
                df[f"{factor}_loading"] = 0.0
            return df
        
        def rolling_factor_loadings(group):
            if len(group) < window:
                for factor in available_factors:
                    group[f"{factor.replace('_factor', '')}_loading"] = np.nan
                return group
            
            stock_ret = group[stock_return_col].values
            factor_data = {f: group[f].values for f in available_factors}
            
            loadings = {f: [] for f in available_factors}
            
            for i in range(len(group)):
                if i < window - 1:
                    for f in available_factors:
                        loadings[f].append(np.nan)
                else:
                    y = stock_ret[i-window+1:i+1]
                    
                    # Build factor matrix
                    X = np.column_stack([factor_data[f][i-window+1:i+1] for f in available_factors])
                    X = np.column_stack([np.ones(len(X)), X])  # Add constant
                    
                    try:
                        coefs = np.linalg.lstsq(X, y, rcond=None)[0]
                        # coefs[0] is intercept, coefs[1:] are factor loadings
                        for j, f in enumerate(available_factors):
                            loadings[f].append(coefs[j + 1])
                    except:
                        for f in available_factors:
                            loadings[f].append(np.nan)
            
            for f in available_factors:
                group[f"{f.replace('_factor', '')}_loading"] = loadings[f]
            
            return group
        
        return df.groupby("symbol", group_keys=False).apply(rolling_factor_loadings)
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_features_g1.py -v
```

Expected: 3 passing tests

**Step 5: Commit**

```bash
git add tests/test_features_g1.py src/features/g1_risk.py
git commit -m "feat: add G1 systematic risk features (beta, factor loadings)"
```

---

## Task 3: Implement G2 Volatility Features

**Files:**
- Create: `src/features/g2_volatility.py`
- Test: `tests/test_features_g2.py`

**Step 1: Write failing test**

```python
# tests/test_features_g2.py
"""Test G2 volatility features."""

import numpy as np
import pandas as pd
import pytest

from src.features.g2_volatility import G2VolatilityFeatures


def test_g2_initialization():
    """Test G2 feature group initialization."""
    g2 = G2VolatilityFeatures()
    assert g2.name == "G2_volatility"
    assert "realized_vol_20d" in g2.get_feature_names()


def test_volatility_computation():
    """Test volatility feature computation."""
    np.random.seed(42)
    n_days = 100
    
    # Create synthetic returns with known volatility
    target_vol = 0.30  # 30% annualized
    daily_vol = target_vol / np.sqrt(252)
    returns = np.random.normal(0.001, daily_vol, n_days)
    
    df = pd.DataFrame({
        "symbol": ["TEST"] * n_days,
        "date": pd.date_range("2020-01-01", periods=n_days, freq="B"),
        "return": returns,
    })
    
    g2 = G2VolatilityFeatures()
    result = g2.compute(df)
    
    assert "realized_vol_20d" in result.columns
    assert "realized_vol_60d" in result.columns
    
    # Check vol is roughly in expected range (annualized)
    last_vol = result["realized_vol_20d"].iloc[-1]
    assert not pd.isna(last_vol)
    assert 0.10 < last_vol < 0.50  # Should be close to 30%


def test_idiosyncratic_vol():
    """Test idiosyncratic volatility computation."""
    np.random.seed(42)
    n_days = 100
    
    df = pd.DataFrame({
        "symbol": ["TEST"] * n_days,
        "date": pd.date_range("2020-01-01", periods=n_days, freq="B"),
        "return": np.random.normal(0.001, 0.02, n_days),
        "market_return": np.random.normal(0.001, 0.015, n_days),
    })
    
    g2 = G2VolatilityFeatures()
    result = g2.compute(df)
    
    assert "idiosyncratic_vol" in result.columns
    assert result["idiosyncratic_vol"].iloc[-1] >= 0
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_features_g2.py::test_g2_initialization -v
```

Expected: FAIL with "G2VolatilityFeatures not defined"

**Step 3: Write minimal implementation**

```python
# src/features/g2_volatility.py
"""G2 Volatility Profile Features."""

import numpy as np
import pandas as pd

from src.features.base import FeatureGroup, NormalizationMethod


class G2VolatilityFeatures(FeatureGroup):
    """Compute volatility features (realized vol, idiosyncratic vol, vol of vol)."""
    
    def __init__(self):
        super().__init__("G2_volatility", NormalizationMethod.LOG_Z_SCORE)
    
    def get_feature_names(self) -> list[str]:
        return [
            "realized_vol_20d",
            "realized_vol_60d",
            "idiosyncratic_vol",
            "vol_of_vol",
        ]
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute G2 volatility features."""
        result = df.copy()
        
        # Realized volatility (annualized)
        result = self._compute_realized_vol(result, window=20, col_name="realized_vol_20d")
        result = self._compute_realized_vol(result, window=60, col_name="realized_vol_60d")
        
        # Idiosyncratic volatility
        result = self._compute_idiosyncratic_vol(result, window=60)
        
        # Vol of vol
        result = self._compute_vol_of_vol(result, vol_col="realized_vol_20d")
        
        # Apply log transformation then cross-sectional z-score
        for col in ["realized_vol_20d", "realized_vol_60d", "idiosyncratic_vol", "vol_of_vol"]:
            if col in result.columns:
                # Log transform first (vol is log-normal)
                result[col] = np.log(result[col].replace(0, np.nan))
                result[col] = result[col].fillna(0)
                # Then z-score
                result[col] = self._cross_sectional_zscore(result, col)
        
        return result
    
    def _compute_realized_vol(
        self,
        df: pd.DataFrame,
        window: int = 20,
        col_name: str = "realized_vol",
        return_col: str = "return",
    ) -> pd.DataFrame:
        """Compute annualized realized volatility."""
        
        def rolling_vol(group):
            vol = group[return_col].rolling(window=window, min_periods=window//2).std() * np.sqrt(252)
            group[col_name] = vol
            return group
        
        return df.groupby("symbol", group_keys=False).apply(rolling_vol)
    
    def _compute_idiosyncratic_vol(
        self,
        df: pd.DataFrame,
        window: int = 60,
        return_col: str = "return",
        market_return_col: str = "market_return",
    ) -> pd.DataFrame:
        """Compute idiosyncratic volatility (residual vol after removing market)."""
        
        def rolling_idio_vol(group):
            if len(group) < window:
                group["idiosyncratic_vol"] = np.nan
                return group
            
            returns = group[return_col].values
            market_returns = group[market_return_col].values if market_return_col in group.columns else np.zeros(len(group))
            
            residuals = []
            for i in range(len(group)):
                if i < window - 1:
                    residuals.append(np.nan)
                else:
                    y = returns[i-window+1:i+1]
                    x = market_returns[i-window+1:i+1]
                    
                    # Simple regression: r_stock = alpha + beta * r_market + residual
                    x_with_const = np.column_stack([np.ones(len(x)), x])
                    try:
                        coefs = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
                        beta = coefs[1]
                        pred = coefs[0] + beta * x
                        resid = y - pred
                        residuals.append(np.std(resid) * np.sqrt(252))
                    except:
                        residuals.append(np.nan)
            
            group["idiosyncratic_vol"] = residuals
            return group
        
        return df.groupby("symbol", group_keys=False).apply(rolling_idio_vol)
    
    def _compute_vol_of_vol(
        self,
        df: pd.DataFrame,
        vol_col: str = "realized_vol_20d",
        window: int = 20,
    ) -> pd.DataFrame:
        """Compute volatility of volatility."""
        
        def rolling_vov(group):
            vov = group[vol_col].rolling(window=window, min_periods=window//2).std()
            group["vol_of_vol"] = vov
            return group
        
        return df.groupby("symbol", group_keys=False).apply(rolling_vov)
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_features_g2.py -v
```

Expected: 3 passing tests

**Step 5: Commit**

```bash
git add tests/test_features_g2.py src/features/g2_volatility.py
git commit -m "feat: add G2 volatility features (realized vol, idiosyncratic vol)"
```

---

## Task 4: Implement G3 Momentum Features

**Files:**
- Create: `src/features/g3_momentum.py`
- Test: `tests/test_features_g3.py`

**Step 1: Write failing test**

```python
# tests/test_features_g3.py
"""Test G3 momentum features."""

import numpy as np
import pandas as pd
import pytest

from src.features.g3_momentum import G3MomentumFeatures


def test_g3_initialization():
    """Test G3 feature group initialization."""
    g3 = G3MomentumFeatures()
    assert g3.name == "G3_momentum"
    assert "mom_1m" in g3.get_feature_names()


def test_momentum_computation():
    """Test momentum feature computation."""
    # Create price series with known momentum
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    
    # Upward trending price (positive momentum)
    prices = 100 * (1.001 ** np.arange(300))  # ~0.1% daily growth
    
    df = pd.DataFrame({
        "symbol": ["TEST"] * 300,
        "date": dates,
        "close": prices,
    })
    
    g3 = G3MomentumFeatures()
    result = g3.compute(df)
    
    assert "mom_1m" in result.columns
    assert "mom_3m" in result.columns
    assert "mom_6m" in result.columns
    assert "mom_12_1m" in result.columns
    
    # Should have positive momentum
    last_mom = result["mom_1m"].iloc[-1]
    assert not pd.isna(last_mom)
    assert last_mom > 0


def test_macd_computation():
    """Test MACD computation."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    prices = 100 + np.cumsum(np.random.normal(0, 1, 100))
    
    df = pd.DataFrame({
        "symbol": ["TEST"] * 100,
        "date": dates,
        "close": prices,
    })
    
    g3 = G3MomentumFeatures()
    result = g3.compute(df)
    
    assert "macd" in result.columns
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_features_g3.py::test_g3_initialization -v
```

Expected: FAIL with "G3MomentumFeatures not defined"

**Step 3: Write minimal implementation**

```python
# src/features/g3_momentum.py
"""G3 Return Momentum & Trend Features."""

import numpy as np
import pandas as pd

from src.features.base import FeatureGroup, NormalizationMethod


class G3MomentumFeatures(FeatureGroup):
    """Compute momentum features (1M, 3M, 6M, 12-1M, MACD)."""
    
    def __init__(self):
        super().__init__("G3_momentum", NormalizationMethod.RANK)
    
    def get_feature_names(self) -> list[str]:
        return [
            "mom_1m",
            "mom_3m",
            "mom_6m",
            "mom_12_1m",
            "macd",
        ]
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute G3 momentum features."""
        result = df.copy()
        
        # Momentum features
        result = self._compute_momentum(result, days=21, col_name="mom_1m")
        result = self._compute_momentum(result, days=63, col_name="mom_3m")
        result = self._compute_momentum(result, days=126, col_name="mom_6m")
        result = self._compute_jegadeesh_titman(result, days_long=252, days_skip=21, col_name="mom_12_1m")
        
        # MACD
        result = self._compute_macd(result)
        
        # Apply cross-sectional rank normalization [0, 1]
        for col in ["mom_1m", "mom_3m", "mom_6m", "mom_12_1m", "macd"]:
            if col in result.columns:
                result[col] = self._cross_sectional_rank(result, col)
        
        return result
    
    def _compute_momentum(
        self,
        df: pd.DataFrame,
        days: int,
        col_name: str,
        price_col: str = "close",
    ) -> pd.DataFrame:
        """Compute simple momentum: (P_t / P_{t-n}) - 1."""
        
        def momentum(group):
            group[col_name] = group[price_col].pct_change(periods=days)
            return group
        
        return df.groupby("symbol", group_keys=False).apply(momentum)
    
    def _compute_jegadeesh_titman(
        self,
        df: pd.DataFrame,
        days_long: int = 252,
        days_skip: int = 21,
        col_name: str = "mom_12_1m",
        price_col: str = "close",
    ) -> pd.DataFrame:
        """Compute Jegadeesh-Titman momentum: 12M return skip last 1M."""
        
        def jt_momentum(group):
            # Price (days_long + days_skip) ago
            price_long_ago = group[price_col].shift(days_long + days_skip)
            # Price days_skip ago
            price_recent = group[price_col].shift(days_skip)
            
            # Momentum = (price_recent / price_long_ago) - 1
            group[col_name] = (price_recent / price_long_ago) - 1
            return group
        
        return df.groupby("symbol", group_keys=False).apply(jt_momentum)
    
    def _compute_macd(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        fast: int = 12,
        slow: int = 26,
    ) -> pd.DataFrame:
        """Compute MACD: EMA(12) - EMA(26)."""
        
        def macd(group):
            ema_fast = group[price_col].ewm(span=fast, adjust=False).mean()
            ema_slow = group[price_col].ewm(span=slow, adjust=False).mean()
            group["macd"] = ema_fast - ema_slow
            return group
        
        return df.groupby("symbol", group_keys=False).apply(macd)
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_features_g3.py -v
```

Expected: 3 passing tests

**Step 5: Commit**

```bash
git add tests/test_features_g3.py src/features/g3_momentum.py
git commit -m "feat: add G3 momentum features (1M, 3M, 6M, 12-1M, MACD)"
```

---

## Task 5: Implement G4 Valuation Features

**Files:**
- Create: `src/features/g4_valuation.py`
- Test: `tests/test_features_g4.py`

**Step 1: Write failing test**

```python
# tests/test_features_g4.py
"""Test G4 valuation features."""

import numpy as np
import pandas as pd
import pytest

from src.features.g4_valuation import G4ValuationFeatures


def test_g4_initialization():
    """Test G4 feature group initialization."""
    g4 = G4ValuationFeatures()
    assert g4.name == "G4_valuation"
    assert "log_mktcap" in g4.get_feature_names()


def test_valuation_computation():
    """Test valuation feature computation."""
    df = pd.DataFrame({
        "symbol": ["A", "B", "C"],
        "date": pd.to_datetime(["2020-01-01"] * 3),
        "close": [100.0, 200.0, 50.0],
        "shares_outstanding": [1000000, 500000, 2000000],
        "eps": [5.0, 8.0, 2.0],  # Earnings per share
        "book_value_per_share": [50.0, 80.0, 25.0],
        "net_income": [5000000, 4000000, 4000000],
        "shareholders_equity": [50000000, 40000000, 50000000],
    })
    
    g4 = G4ValuationFeatures()
    result = g4.compute(df)
    
    assert "log_mktcap" in result.columns
    assert "pe_ratio" in result.columns
    assert "pb_ratio" in result.columns
    assert "roe" in result.columns
    
    # Check P/E calculation
    expected_pe_a = 100.0 / 5.0  # 20.0
    assert abs(result.loc[result["symbol"] == "A", "pe_ratio"].iloc[0] - expected_pe_a) < 0.01


def test_pe_winsorization():
    """Test that extreme P/E values are winsorized."""
    df = pd.DataFrame({
        "symbol": ["A", "B", "C", "D", "E"],
        "date": pd.to_datetime(["2020-01-01"] * 5),
        "close": [100.0, 200.0, 50.0, 1000.0, 10.0],
        "eps": [5.0, 0.01, 2.0, 0.1, 10.0],  # B has very high P/E (20000), D has 10000
        "book_value_per_share": [50.0, 80.0, 25.0, 100.0, 100.0],
        "net_income": [5000000, 100000, 4000000, 10000000, 10000000],
        "shareholders_equity": [50000000, 40000000, 50000000, 100000000, 100000000],
    })
    
    g4 = G4ValuationFeatures()
    result = g4.compute(df)
    
    # All P/E ratios should be finite and reasonable after winsorization
    assert result["pe_ratio"].notna().all()
    assert result["pe_ratio"].max() < 1000  # Should be winsorized
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_features_g4.py::test_g4_initialization -v
```

Expected: FAIL with "G4ValuationFeatures not defined"

**Step 3: Write minimal implementation**

```python
# src/features/g4_valuation.py
"""G4 Valuation & Fundamentals Features."""

import numpy as np
import pandas as pd

from src.features.base import FeatureGroup, NormalizationMethod


class G4ValuationFeatures(FeatureGroup):
    """Compute valuation features (P/E, P/B, ROE, market cap)."""
    
    def __init__(self):
        super().__init__("G4_valuation", NormalizationMethod.Z_SCORE)
    
    def get_feature_names(self) -> list[str]:
        return [
            "log_mktcap",
            "pe_ratio",
            "pb_ratio",
            "roe",
        ]
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute G4 valuation features."""
        result = df.copy()
        
        # Market cap (in log space)
        result = self._compute_market_cap(result)
        
        # P/E ratio
        result = self._compute_pe_ratio(result)
        
        # P/B ratio
        result = self._compute_pb_ratio(result)
        
        # ROE
        result = self._compute_roe(result)
        
        # Apply transformations and normalization
        # log_mktcap: log then z-score
        result["log_mktcap"] = np.log(result["log_mktcap"].replace(0, np.nan))
        result["log_mktcap"] = result["log_mktcap"].fillna(0)
        result["log_mktcap"] = self._cross_sectional_zscore(result, "log_mktcap")
        
        # pe_ratio: winsorize [2%, 98%] then rank [0, 1]
        result["pe_ratio"] = self._winsorize(result["pe_ratio"], lower=0.02, upper=0.98)
        result["pe_ratio"] = self._cross_sectional_rank(result, "pe_ratio")
        
        # pb_ratio: log then z-score
        result["pb_ratio"] = np.log(result["pb_ratio"].replace(0, np.nan))
        result["pb_ratio"] = result["pb_ratio"].fillna(0)
        result["pb_ratio"] = self._cross_sectional_zscore(result, "pb_ratio")
        
        # roe: winsorize then z-score
        result["roe"] = self._winsorize(result["roe"], lower=0.02, upper=0.98)
        result["roe"] = self._cross_sectional_zscore(result, "roe")
        
        return result
    
    def _compute_market_cap(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        shares_col: str = "shares_outstanding",
    ) -> pd.DataFrame:
        """Compute market cap."""
        if shares_col in df.columns and price_col in df.columns:
            df["log_mktcap"] = df[price_col] * df[shares_col]
        else:
            df["log_mktcap"] = 1e9  # Default placeholder
        return df
    
    def _compute_pe_ratio(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        eps_col: str = "eps",
    ) -> pd.DataFrame:
        """Compute P/E ratio."""
        if eps_col in df.columns and price_col in df.columns:
            # Avoid division by zero and negative earnings
            eps_safe = df[eps_col].replace(0, np.nan)
            df["pe_ratio"] = df[price_col] / eps_safe
        else:
            df["pe_ratio"] = np.nan
        return df
    
    def _compute_pb_ratio(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        book_value_col: str = "book_value_per_share",
    ) -> pd.DataFrame:
        """Compute P/B ratio."""
        if book_value_col in df.columns and price_col in df.columns:
            bv_safe = df[book_value_col].replace(0, np.nan)
            df["pb_ratio"] = df[price_col] / bv_safe
        else:
            df["pb_ratio"] = np.nan
        return df
    
    def _compute_roe(
        self,
        df: pd.DataFrame,
        net_income_col: str = "net_income",
        equity_col: str = "shareholders_equity",
    ) -> pd.DataFrame:
        """Compute ROE (Return on Equity)."""
        if net_income_col in df.columns and equity_col in df.columns:
            equity_safe = df[equity_col].replace(0, np.nan)
            df["roe"] = df[net_income_col] / equity_safe
        else:
            df["roe"] = np.nan
        return df
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_features_g4.py -v
```

Expected: 3 passing tests

**Step 5: Commit**

```bash
git add tests/test_features_g4.py src/features/g4_valuation.py
git commit -m "feat: add G4 valuation features (P/E, P/B, ROE, market cap)"
```

---

## Task 6: Implement G5 OHLCV Behavior Features

**Files:**
- Create: `src/features/g5_ohlcv.py`
- Test: `tests/test_features_g5.py`

**Step 1: Write failing test**

```python
# tests/test_features_g5.py
"""Test G5 OHLCV behavior features."""

import numpy as np
import pandas as pd
import pytest

from src.features.g5_ohlcv import G5OHLCVFeatures


def test_g5_initialization():
    """Test G5 feature group initialization."""
    g5 = G5OHLCVFeatures()
    assert g5.name == "G5_ohlcv"
    assert "z_close_5d" in g5.get_feature_names()


def test_ohlcv_computation():
    """Test OHLCV feature computation."""
    np.random.seed(42)
    n_days = 300
    
    # Generate OHLCV data
    close = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n_days)))
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    volume = np.random.lognormal(15, 0.5, n_days)
    
    df = pd.DataFrame({
        "symbol": ["TEST"] * n_days,
        "date": pd.date_range("2020-01-01", periods=n_days, freq="B"),
        "open": close * (1 + np.random.normal(0, 0.005, n_days)),
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    
    g5 = G5OHLCVFeatures()
    result = g5.compute(df)
    
    # Check feature columns exist
    assert "z_close_5d" in result.columns
    assert "z_close_10d" in result.columns
    assert "z_high" in result.columns
    assert "z_low" in result.columns
    assert "z_volume_5d" in result.columns


def test_rolling_zscore():
    """Test that rolling z-scores are computed per stock."""
    np.random.seed(42)
    n_days = 300
    
    # Two stocks with different volatility regimes
    close_a = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.01, n_days)))  # Low vol
    close_b = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.05, n_days)))  # High vol
    
    df = pd.DataFrame({
        "symbol": ["A"] * n_days + ["B"] * n_days,
        "date": list(pd.date_range("2020-01-01", periods=n_days, freq="B")) * 2,
        "close": np.concatenate([close_a, close_b]),
        "high": np.concatenate([close_a * 1.01, close_b * 1.05]),
        "low": np.concatenate([close_a * 0.99, close_b * 0.95]),
        "volume": np.random.lognormal(15, 0.5, n_days * 2),
    })
    
    g5 = G5OHLCVFeatures()
    result = g5.compute(df)
    
    # Both stocks should have z-scores centered around 0 (after enough history)
    z_scores_a = result[result["symbol"] == "A"]["z_close_5d"].iloc[252:]
    z_scores_b = result[result["symbol"] == "B"]["z_close_5d"].iloc[252:]
    
    assert abs(z_scores_a.mean()) < 0.5  # Should be centered
    assert abs(z_scores_b.mean()) < 0.5
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_features_g5.py::test_g5_initialization -v
```

Expected: FAIL with "G5OHLCVFeatures not defined"

**Step 3: Write minimal implementation**

```python
# src/features/g5_ohlcv.py
"""G5 Multi-Scale OHLCV Price Behavior Features."""

import numpy as np
import pandas as pd

from src.features.base import FeatureGroup, NormalizationMethod


class G5OHLCVFeatures(FeatureGroup):
    """Compute OHLCV behavior features (z-scores of returns, ranges, volumes)."""
    
    def __init__(self):
        super().__init__("G5_ohlcv", NormalizationMethod.ROLLING_Z_SCORE)
    
    def get_feature_names(self) -> list[str]:
        return [
            "z_close_5d",
            "z_close_10d",
            "z_close_20d",
            "z_high",
            "z_low",
            "z_volume_5d",
            "z_volume_10d",
            "z_volume_20d",
            "ma_ratio_5",
            "ma_ratio_10",
            "ma_ratio_15",
            "ma_ratio_20",
            "ma_ratio_25",
        ]
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute G5 OHLCV features."""
        result = df.copy()
        
        # Compute z-scores of returns (moving averages)
        result = self._compute_z_returns(result, window=5, col_name="z_close_5d")
        result = self._compute_z_returns(result, window=10, col_name="z_close_10d")
        result = self._compute_z_returns(result, window=20, col_name="z_close_20d")
        
        # Z-scores of high/low ranges
        result = self._compute_z_high_low(result)
        
        # Z-scores of volume changes
        result = self._compute_z_volume(result, window=5, col_name="z_volume_5d")
        result = self._compute_z_volume(result, window=10, col_name="z_volume_10d")
        result = self._compute_z_volume(result, window=20, col_name="z_volume_20d")
        
        # MA ratios
        for window in [5, 10, 15, 20, 25]:
            result = self._compute_ma_ratio(result, window=window)
        
        # Apply rolling time-series z-score (per stock)
        for col in self.get_feature_names():
            if col in result.columns:
                result[col] = result.groupby("symbol", group_keys=False).apply(
                    lambda x: self._rolling_zscore(x[col], window=252, min_periods=60)
                )
        
        return result
    
    def _compute_z_returns(
        self,
        df: pd.DataFrame,
        window: int,
        col_name: str,
        close_col: str = "close",
    ) -> pd.DataFrame:
        """Compute z-score of returns over moving average window."""
        
        def compute(group):
            returns = group[close_col].pct_change()
            ma_returns = returns.rolling(window=window, min_periods=window//2).mean()
            
            # Store raw MA of returns (will be z-scored later)
            group[col_name] = ma_returns
            return group
        
        return df.groupby("symbol", group_keys=False).apply(compute)
    
    def _compute_z_high_low(
        self,
        df: pd.DataFrame,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
    ) -> pd.DataFrame:
        """Compute z-scores of high and low relative to close."""
        
        if high_col in df.columns:
            df["z_high"] = (df[high_col] / df[close_col]) - 1
        else:
            df["z_high"] = 0.0
        
        if low_col in df.columns:
            df["z_low"] = (df[low_col] / df[close_col]) - 1
        else:
            df["z_low"] = 0.0
        
        return df
    
    def _compute_z_volume(
        self,
        df: pd.DataFrame,
        window: int,
        col_name: str,
        volume_col: str = "volume",
    ) -> pd.DataFrame:
        """Compute z-score of volume changes."""
        
        def compute(group):
            vol_change = group[volume_col].pct_change()
            ma_vol_change = vol_change.rolling(window=window, min_periods=window//2).mean()
            group[col_name] = ma_vol_change
            return group
        
        return df.groupby("symbol", group_keys=False).apply(compute)
    
    def _compute_ma_ratio(
        self,
        df: pd.DataFrame,
        window: int,
        price_col: str = "close",
    ) -> pd.DataFrame:
        """Compute (Price / MA_n) - 1."""
        
        def compute(group):
            ma = group[price_col].rolling(window=window, min_periods=window//2).mean()
            group[f"ma_ratio_{window}"] = (group[price_col] / ma) - 1
            return group
        
        return df.groupby("symbol", group_keys=False).apply(compute)
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_features_g5.py -v
```

Expected: 3 passing tests

**Step 5: Commit**

```bash
git add tests/test_features_g5.py src/features/g5_ohlcv.py
git commit -m "feat: add G5 OHLCV behavior features (z-scores, MA ratios)"
```

---

## Task 7: Implement G6 Sector Features

**Files:**
- Create: `src/features/g6_sector.py`
- Test: `tests/test_features_g6.py`

**Step 1: Write failing test**

```python
# tests/test_features_g6.py
"""Test G6 sector features."""

import pandas as pd
import pytest

from src.features.g6_sector import G6SectorFeatures


def test_g6_initialization():
    """Test G6 feature group initialization."""
    g6 = G6SectorFeatures()
    assert g6.name == "G6_sector"
    assert "gics_sector" in g6.get_feature_names()


def test_sector_encoding():
    """Test sector encoding."""
    df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "XOM", "JPM"],
        "date": pd.to_datetime(["2020-01-01"] * 4),
        "gics_sector": ["Technology", "Technology", "Energy", "Financials"],
        "gics_industry_group": ["Software", "Software", "Oil & Gas", "Banks"],
    })
    
    g6 = G6SectorFeatures()
    result = g6.compute(df)
    
    # Should have encoded sector columns
    assert "gics_sector" in result.columns
    assert "gics_industry_group" in result.columns
    
    # Values should be numeric codes (not strings)
    assert result["gics_sector"].dtype in ["int64", "int32"]
    assert result["gics_industry_group"].dtype in ["int64", "int32"]


def test_sector_consistency():
    """Test that same sectors get same codes."""
    df = pd.DataFrame({
        "symbol": ["A", "B", "C", "D", "E"],
        "date": pd.to_datetime(["2020-01-01"] * 5),
        "gics_sector": ["Tech", "Tech", "Energy", "Tech", "Energy"],
        "gics_industry_group": ["Software", "Software", "Oil", "Hardware", "Oil"],
    })
    
    g6 = G6SectorFeatures()
    result = g6.compute(df)
    
    # Same sectors should have same codes
    tech_codes = result[result["gics_sector_str"] == "Tech"]["gics_sector"].unique()
    energy_codes = result[result["gics_sector_str"] == "Energy"]["gics_sector"].unique()
    
    assert len(tech_codes) == 1
    assert len(energy_codes) == 1
    assert tech_codes[0] != energy_codes[0]
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_features_g6.py::test_g6_initialization -v
```

Expected: FAIL with "G6SectorFeatures not defined"

**Step 3: Write minimal implementation**

```python
# src/features/g6_sector.py
"""G6 Sector / Industry Categorical Features."""

import pandas as pd

from src.features.base import FeatureGroup, NormalizationMethod


class G6SectorFeatures(FeatureGroup):
    """Encode sector and industry as categorical features."""
    
    def __init__(self):
        super().__init__("G6_sector", NormalizationMethod.NONE)
        self.sector_mapping: dict[str, int] = {}
        self.industry_mapping: dict[str, int] = {}
    
    def get_feature_names(self) -> list[str]:
        return [
            "gics_sector",
            "gics_industry_group",
        ]
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode G6 sector features as integer codes.
        
        Note: In the full model, these will be passed to nn.Embedding.
        For now, we just encode them as integers.
        """
        result = df.copy()
        
        # Store original strings for reference
        if "gics_sector" in result.columns:
            result["gics_sector_str"] = result["gics_sector"]
        if "gics_industry_group" in result.columns:
            result["gics_industry_group_str"] = result["gics_industry_group"]
        
        # Encode sectors
        result = self._encode_categorical(
            result, 
            col="gics_sector", 
            mapping=self.sector_mapping,
            default_val=0,
        )
        
        # Encode industry groups
        result = self._encode_categorical(
            result,
            col="gics_industry_group",
            mapping=self.industry_mapping,
            default_val=0,
        )
        
        return result
    
    def _encode_categorical(
        self,
        df: pd.DataFrame,
        col: str,
        mapping: dict[str, int],
        default_val: int = 0,
    ) -> pd.DataFrame:
        """Encode categorical column to integer codes."""
        
        if col not in df.columns:
            df[col] = default_val
            return df
        
        # Get unique values
        unique_vals = df[col].dropna().unique()
        
        # Update mapping with new values
        for val in unique_vals:
            if val not in mapping:
                mapping[val] = len(mapping)
        
        # Map to integers
        df[col] = df[col].map(mapping).fillna(default_val).astype(int)
        
        return df
    
    def get_sector_embedding_dim(self) -> int:
        """Get recommended embedding dimension for sectors."""
        return 8
    
    def get_industry_embedding_dim(self) -> int:
        """Get recommended embedding dimension for industry groups."""
        return 16
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_features_g6.py -v
```

Expected: 3 passing tests

**Step 5: Commit**

```bash
git add tests/test_features_g6.py src/features/g6_sector.py
git commit -m "feat: add G6 sector features (categorical encoding)"
```

---

## Task 8: Create Unified Feature Engineer

**Files:**
- Create: `src/features/engineer.py`
- Create: `src/features/__init__.py` (update)
- Test: `tests/test_feature_engineer.py`

**Step 1: Write failing test**

```python
# tests/test_feature_engineer.py
"""Test unified feature engineer."""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import FeatureEngineer


def test_engineer_initialization():
    """Test feature engineer initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engineer = FeatureEngineer(cache_dir=Path(tmpdir))
        assert engineer.cache_dir == Path(tmpdir)
        assert len(engineer.registry.list_groups()) == 6  # G1-G6


def test_engineer_compute_features_mock():
    """Test computing all features with mock data."""
    np.random.seed(42)
    n_days = 300
    
    # Create synthetic OHLCV data
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    close = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n_days)))
    
    df = pd.DataFrame({
        "symbol": ["TEST"] * n_days,
        "date": dates,
        "open": close * (1 + np.random.normal(0, 0.005, n_days)),
        "high": close * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
        "close": close,
        "volume": np.random.lognormal(15, 0.5, n_days),
        "return": np.random.normal(0.001, 0.02, n_days),
        "market_return": np.random.normal(0.001, 0.015, n_days),
        "eps": [5.0] * n_days,
        "book_value_per_share": [50.0] * n_days,
        "shares_outstanding": [1000000] * n_days,
        "net_income": [5000000] * n_days,
        "shareholders_equity": [50000000] * n_days,
        "gics_sector": ["Technology"] * n_days,
        "gics_industry_group": ["Software"] * n_days,
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        engineer = FeatureEngineer(cache_dir=Path(tmpdir))
        result = engineer.compute_features(df)
        
        # Check that all G1-G6 features are present
        assert "market_beta_60d" in result.columns  # G1
        assert "realized_vol_20d" in result.columns  # G2
        assert "mom_1m" in result.columns  # G3
        assert "pe_ratio" in result.columns  # G4
        assert "z_close_5d" in result.columns  # G5
        assert "gics_sector" in result.columns  # G6
        
        print(f"Total features: {len([c for c in result.columns if c not in ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']])}")


def test_engineer_caching():
    """Test that features are cached."""
    np.random.seed(42)
    n_days = 100
    
    df = pd.DataFrame({
        "symbol": ["AAPL"] * n_days,
        "date": pd.date_range("2020-01-01", periods=n_days, freq="B"),
        "close": 100 + np.cumsum(np.random.normal(0, 1, n_days)),
        "return": np.random.normal(0.001, 0.02, n_days),
        "market_return": np.random.normal(0.001, 0.015, n_days),
        "eps": [5.0] * n_days,
        "book_value_per_share": [50.0] * n_days,
        "shares_outstanding": [1000000] * n_days,
        "net_income": [5000000] * n_days,
        "shareholders_equity": [50000000] * n_days,
        "gics_sector": ["Tech"] * n_days,
        "gics_industry_group": ["Software"] * n_days,
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        engineer = FeatureEngineer(cache_dir=Path(tmpdir))
        
        # First computation
        result1 = engineer.compute_features(df, cache_key="test_features")
        
        # Second computation (should use cache)
        result2 = engineer.compute_features(df, cache_key="test_features")
        
        # Should be identical
        pd.testing.assert_frame_equal(result1, result2)
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_feature_engineer.py::test_engineer_initialization -v
```

Expected: FAIL with "FeatureEngineer not defined"

**Step 3: Write minimal implementation**

```python
# src/features/engineer.py
"""Unified feature engineer that orchestrates G1-G6 feature computation."""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.cache_manager import CacheManager
from src.features.base import FeatureRegistry
from src.features.g1_risk import G1RiskFeatures
from src.features.g2_volatility import G2VolatilityFeatures
from src.features.g3_momentum import G3MomentumFeatures
from src.features.g4_valuation import G4ValuationFeatures
from src.features.g5_ohlcv import G5OHLCVFeatures
from src.features.g6_sector import G6SectorFeatures


class FeatureEngineer:
    """Orchestrate computation of all G1-G6 features."""
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
    ):
        """Initialize feature engineer.
        
        Args:
            cache_dir: Directory for caching computed features
            use_cache: Whether to use caching
        """
        if cache_dir is None:
            cache_dir = Path("data/cache/features")
        
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.cache = CacheManager(cache_dir=self.cache_dir) if use_cache else None
        
        # Initialize feature registry with all groups
        self.registry = FeatureRegistry()
        self._register_feature_groups()
    
    def _register_feature_groups(self) -> None:
        """Register all G1-G6 feature groups."""
        self.registry.register("G1_risk", G1RiskFeatures())
        self.registry.register("G2_volatility", G2VolatilityFeatures())
        self.registry.register("G3_momentum", G3MomentumFeatures())
        self.registry.register("G4_valuation", G4ValuationFeatures())
        self.registry.register("G5_ohlcv", G5OHLCVFeatures())
        self.registry.register("G6_sector", G6SectorFeatures())
    
    def compute_features(
        self,
        df: pd.DataFrame,
        cache_key: Optional[str] = None,
    ) -> pd.DataFrame:
        """Compute all G1-G6 features for input data.
        
        Args:
            df: Input DataFrame with OHLCV and fundamental data
            cache_key: Optional cache key for storing/retrieving results
            
        Returns:
            DataFrame with all features computed
        """
        # Check cache first
        if self.use_cache and cache_key and self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        
        # Compute all features
        result = self.registry.compute_all(df)
        
        # Cache result
        if self.use_cache and cache_key and self.cache:
            self.cache.set(cache_key, result)
        
        return result
    
    def get_feature_names(self) -> list[str]:
        """Get list of all feature names produced."""
        all_features = []
        for group_name in self.registry.list_groups():
            group = self.registry.get(group_name)
            if group:
                all_features.extend(group.get_feature_names())
        return all_features
    
    def compute_single_group(
        self,
        df: pd.DataFrame,
        group_name: str,
    ) -> pd.DataFrame:
        """Compute features for a single group only.
        
        Useful for debugging or ablation studies.
        """
        group = self.registry.get(group_name)
        if group is None:
            raise ValueError(f"Unknown feature group: {group_name}")
        
        return group.compute(df)
```

**Step 4: Update features module init**

```python
# src/features/__init__.py
"""Feature engineering pipeline for G1-G6 feature groups."""

from src.features.base import FeatureGroup, FeatureRegistry, NormalizationMethod
from src.features.engineer import FeatureEngineer
from src.features.g1_risk import G1RiskFeatures
from src.features.g2_volatility import G2VolatilityFeatures
from src.features.g3_momentum import G3MomentumFeatures
from src.features.g4_valuation import G4ValuationFeatures
from src.features.g5_ohlcv import G5OHLCVFeatures
from src.features.g6_sector import G6SectorFeatures

__all__ = [
    "FeatureGroup",
    "FeatureRegistry",
    "NormalizationMethod",
    "FeatureEngineer",
    "G1RiskFeatures",
    "G2VolatilityFeatures",
    "G3MomentumFeatures",
    "G4ValuationFeatures",
    "G5OHLCVFeatures",
    "G6SectorFeatures",
]
```

**Step 5: Run test to verify it passes**

```bash
python -m pytest tests/test_feature_engineer.py -v
```

Expected: 3 passing tests

**Step 6: Commit**

```bash
git add tests/test_feature_engineer.py src/features/engineer.py src/features/__init__.py
git commit -m "feat: add unified feature engineer orchestrating G1-G6 computation"
```

---

## Task 9: Create End-to-End Integration Test

**Files:**
- Create: `tests/test_features_integration.py`
- Create: `examples/compute_features.py` (example script)

**Step 1: Write integration test**

```python
# tests/test_features_integration.py
"""Integration test for complete feature engineering pipeline."""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import WRDSDataLoader
from src.features import FeatureEngineer


def test_end_to_end_feature_pipeline():
    """Test complete pipeline from data loading to feature computation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        
        # Step 1: Load mock data
        data_loader = WRDSDataLoader(
            mock_mode=True,
            cache_dir=cache_dir / "data",
        )
        
        prices = data_loader.load_prices(
            symbols=["AAPL", "MSFT", "XOM"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 6, 30),
        )
        
        fundamentals = data_loader.load_fundamentals(
            symbols=["AAPL", "MSFT", "XOM"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 6, 30),
        )
        
        # Step 2: Merge data (simulate merged data with all required columns)
        merged = pd.merge(
            prices,
            fundamentals,
            on=["symbol", "date"],
            how="left",
        )
        
        # Fill fundamental columns forward
        for col in ["eps", "book_value_per_share", "net_income", "shareholders_equity"]:
            if col in merged.columns:
                merged[col] = merged.groupby("symbol")[col].ffill()
        
        # Add required columns for features
        merged["return"] = merged.groupby("symbol")["close"].pct_change()
        merged["market_return"] = merged["return"].mean()  # Simplified
        
        # Step 3: Compute features
        engineer = FeatureEngineer(cache_dir=cache_dir / "features")
        
        # Only compute for rows with sufficient history
        merged = merged.dropna(subset=["return"])
        
        result = engineer.compute_features(merged, cache_key="integration_test")
        
        # Step 4: Verify output
        assert len(result) > 0
        assert "symbol" in result.columns
        assert "date" in result.columns
        
        # Check all feature groups are represented
        g1_features = ["market_beta_60d", "downside_beta"]
        g2_features = ["realized_vol_20d", "idiosyncratic_vol"]
        g3_features = ["mom_1m", "mom_3m", "macd"]
        g4_features = ["pe_ratio", "pb_ratio", "roe"]
        g5_features = ["z_close_5d", "z_high", "z_volume_5d"]
        g6_features = ["gics_sector"]
        
        all_expected = g1_features + g2_features + g3_features + g4_features + g5_features + g6_features
        
        for feat in all_expected:
            assert feat in result.columns, f"Missing feature: {feat}"
        
        print(f"✅ Successfully computed {len(all_expected)} features for {len(result)} rows")


def test_feature_coverage():
    """Test that we cover all MVP-required features."""
    engineer = FeatureEngineer(use_cache=False)
    all_features = engineer.get_feature_names()
    
    # MVP-required features
    required = {
        # G1
        "market_beta_60d",
        "downside_beta",
        "smb_loading",
        "hml_loading",
        "mom_loading",
        # G2
        "realized_vol_20d",
        "realized_vol_60d",
        "idiosyncratic_vol",
        # G3
        "mom_1m",
        "mom_3m",
        "mom_6m",
        "mom_12_1m",
        # G4
        "pe_ratio",
        "pb_ratio",
        "roe",
        # G5
        "z_close_5d",
        "z_high",
        "z_volume_5d",
        # G6
        "gics_sector",
    }
    
    missing = required - set(all_features)
    if missing:
        pytest.fail(f"Missing required features: {missing}")
    
    print(f"✅ All {len(required)} MVP-required features are implemented")
```

**Step 2: Create example script**

```python
# examples/compute_features.py
"""Example: Compute features for a set of stocks."""

from datetime import datetime
from pathlib import Path

from src.data import WRDSDataLoader
from src.features import FeatureEngineer
import pandas as pd


def main():
    """Compute features for sample stocks."""
    print("=" * 60)
    print("Feature Engineering Pipeline Example")
    print("=" * 60)
    
    # Initialize loaders
    data_loader = WRDSDataLoader(mock_mode=True)
    engineer = FeatureEngineer()
    
    # Load data
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2020, 12, 31)
    
    print(f"\nLoading data for {len(symbols)} stocks...")
    merged = data_loader.load_merged(symbols, start_date, end_date)
    
    print(f"Loaded {len(merged)} rows of data")
    
    # Add required columns
    merged["return"] = merged.groupby("symbol")["close"].pct_change()
    merged["market_return"] = merged.groupby("date")["return"].transform("mean")
    
    # Compute features
    print("\nComputing G1-G6 features...")
    features = engineer.compute_features(merged)
    
    # Report
    all_feature_cols = [c for c in features.columns if c not in ["symbol", "date", "open", "high", "low", "close", "volume"]]
    print(f"\n✅ Computed {len(all_feature_cols)} features:")
    
    for group in ["G1", "G2", "G3", "G4", "G5", "G6"]:
        group_cols = [c for c in all_feature_cols if c.startswith(group.lower()) or 
                      any(g in c for g in ["beta", "vol", "mom", "pe", "pb", "z_", "gics"])]
        if group == "G1":
            group_cols = [c for c in all_feature_cols if any(x in c for x in ["beta", "loading"])]
        elif group == "G2":
            group_cols = [c for c in all_feature_cols if "vol" in c]
        elif group == "G3":
            group_cols = [c for c in all_feature_cols if "mom" in c or "macd" in c]
        elif group == "G4":
            group_cols = [c for c in all_feature_cols if any(x in c for x in ["pe", "pb", "roe", "mktcap"])]
        elif group == "G5":
            group_cols = [c for c in all_feature_cols if c.startswith("z_") or c.startswith("ma_")]
        elif group == "G6":
            group_cols = [c for c in all_feature_cols if "gics" in c]
        
        print(f"  {group}: {len(group_cols)} features")
    
    # Show sample
    print(f"\nSample output:")
    print(features[["symbol", "date", "market_beta_60d", "realized_vol_20d", "mom_1m", "pe_ratio"]].head(10))
    
    # Save example
    output_path = Path("examples/feature_output_sample.parquet")
    features.to_parquet(output_path)
    print(f"\n💾 Saved sample to: {output_path}")
    print("\nYou can read this in a notebook with:")
    print("  import pandas as pd")
    print(f"  df = pd.read_parquet('{output_path}')")


if __name__ == "__main__":
    main()
```

**Step 3: Run integration test**

```bash
python -m pytest tests/test_features_integration.py -v
```

Expected: 2 passing tests

**Step 4: Run example script**

```bash
python examples/compute_features.py
```

Expected: Runs successfully and outputs feature counts

**Step 5: Commit**

```bash
git add tests/test_features_integration.py examples/compute_features.py
git commit -m "test: add integration tests and usage example for feature pipeline"
```

---

## Task 10: Run Full Test Suite and Final Verification

**Step 1: Run all feature-related tests**

```bash
python -m pytest tests/test_features*.py -v
```

Expected: 20+ passing tests

**Step 2: Run linting**

```bash
python -m ruff check src/features/
```

Expected: No errors

**Step 3: Run type checking**

```bash
python -m mypy src/features/
```

Expected: No type errors

**Step 4: Test import from package**

```bash
python -c "from src.features import FeatureEngineer, G1RiskFeatures; print('All imports successful')"
```

Expected: "All imports successful"

**Step 5: Generate test coverage report**

```bash
python -m pytest tests/test_features*.py --cov=src/features --cov-report=term-missing
```

Expected: >80% coverage

**Step 6: Final commit**

```bash
git add -A
git commit -m "feat: complete feature engineering pipeline (G1-G6)

- G1: Systematic risk (beta, factor loadings)
- G2: Volatility profile (realized vol, idiosyncratic vol)
- G3: Momentum (1M, 3M, 6M, 12-1M, MACD)
- G4: Valuation (P/E, P/B, ROE, market cap)
- G5: OHLCV behavior (z-scores, MA ratios)
- G6: Sector encoding (GICS codes)
- Unified FeatureEngineer orchestrating all groups
- Parquet caching for computed features
- Notebook-compatible output
- 20+ tests with >80% coverage"
```

---

## Summary

This plan implements a complete feature engineering pipeline with:

**Architecture:**
- Modular G1-G6 feature groups (each self-contained)
- Unified `FeatureEngineer` orchestrating computation
- Base classes with common normalization methods
- Parquet caching for computed features

**Feature Groups:**
- **G1 Risk**: Market beta, downside beta, FF5 factor loadings
- **G2 Volatility**: Realized vol (20d, 60d), idiosyncratic vol, vol-of-vol
- **G3 Momentum**: 1M, 3M, 6M, 12-1M momentum, MACD
- **G4 Valuation**: P/E, P/B, ROE, market cap (with log/rank transforms)
- **G5 OHLCV**: Z-scores of returns, ranges, volumes (rolling normalization)
- **G6 Sector**: GICS sector/industry encoding for embeddings

**Normalization Methods (per MVP spec):**
- Cross-sectional z-score (G1, G2, G4)
- Log then z-score (G2 volatility)
- Cross-sectional rank [0,1] (G3 momentum)
- Rolling time-series z-score (G5 OHLCV)

**Key Design Decisions:**
- ✅ Compute all features upfront (Option A per user request)
- ✅ Pandas primary (per AGENTS.md)
- ✅ Parquet caching (notebook-compatible)
- ✅ Proper handling of edge cases (missing data, division by zero)
- ✅ TDD approach with comprehensive tests

---

**Plan saved to:** `docs/plans/2026-01-20-feature-engineering-pipeline.md`

**Execution options:**

1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks
2. **Parallel Session (separate)** - Open new session with executing-plans

Which approach would you prefer?
