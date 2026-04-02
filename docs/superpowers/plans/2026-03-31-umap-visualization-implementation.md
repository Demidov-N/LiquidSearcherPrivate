# UMAP Visualization Module - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Block 1 of the LiquidSearcher evaluation framework - PCA + UMAP visualization for dual-encoder embeddings with crisis comparison and clustering metrics.

**Architecture:** Two-stage dimensionality reduction (PCA for noise reduction → UMAP for 2D visualization) with hybrid caching for embeddings. Supports static PNG and interactive HTML outputs, plus crisis period comparison with fixed-reference projection.

**Tech Stack:** Python, PyTorch Lightning, scikit-learn (PCA, metrics), UMAP-learn, Matplotlib (static plots), Plotly (interactive), Polars/Pandas

**Based on Spec:** `docs/superpowers/specs/2026-03-30-umap-visualization-design.md`

---

## File Structure

```
src/evaluation/
├── __init__.py                    # Export UMAPVisualizer
├── visualizations/
│   ├── __init__.py               # Export visualization classes
│   └── umap_visualizer.py        # Core UMAPVisualizer class (~400 lines)
├── metrics/
│   ├── __init__.py               # Export metrics functions
│   └── clustering.py             # Silhouette, Davies-Bouldin, Calinski-Harabasz
└── utils/
    ├── __init__.py
    └── embedding_cache.py        # Hybrid caching for computed embeddings

scripts/visualization/
├── __init__.py
└── umap_plots.py                 # CLI wrapper script (~150 lines)

tests/evaluation/
├── __init__.py
├── test_umap_visualizer.py       # Unit tests for UMAPVisualizer
├── test_clustering_metrics.py    # Tests for clustering metrics
└── test_embedding_cache.py       # Tests for caching logic
```

---

## Prerequisites

Before starting, verify these dependencies are available:

- [ ] **Check existing model classes**
  - Run: `grep -r "class DualEncoderModule" /home/redbear/Projects/LiquidSearcher/src/models/`
  - Run: `grep -r "class FeatureLoader" /home/redbear/Projects/LiquidSearcher/src/`
  - Expected: Both classes exist and can be imported

- [ ] **Check checkpoint format**
  - Run: `ls -la /home/redbear/Projects/LiquidSearcher/checkpoints/ 2>/dev/null || echo "No checkpoints dir"`
  - Expected: .ckpt files exist or training script available to create one

- [ ] **Update pyproject.toml dependencies**
  - File: `pyproject.toml`
  - Add under `[project.optional-dependencies]`:
    ```toml
    evaluation = [
        "umap-learn>=0.5.0",
        "plotly>=5.14.0",
    ]
    ```

---

## Task 1: Clustering Metrics Module

**Purpose:** Core clustering quality metrics computed in PCA space (not UMAP space).

**Files:**
- Create: `src/evaluation/metrics/__init__.py`
- Create: `src/evaluation/metrics/clustering.py`
- Create: `tests/evaluation/test_clustering_metrics.py`

- [ ] **Step 1.1: Write failing test for silhouette score**

```python
# tests/evaluation/test_clustering_metrics.py
import numpy as np
import pytest
from src.evaluation.metrics.clustering import compute_silhouette_score


def test_silhouette_score_basic():
    """Test silhouette score with clear cluster separation."""
    # Two well-separated clusters
    embeddings = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.0, 0.1],
        [1.0, 1.0],
        [1.1, 1.0],
        [1.0, 1.1],
    ])
    labels = np.array([0, 0, 0, 1, 1, 1])
    
    score = compute_silhouette_score(embeddings, labels)
    
    assert score > 0.5  # Well-separated clusters should have high score
    assert isinstance(score, float)


def test_silhouette_score_random_baseline():
    """Test that random labels produce near-zero silhouette."""
    np.random.seed(42)
    embeddings = np.random.randn(100, 10)
    labels = np.random.randint(0, 3, 100)
    
    score = compute_silhouette_score(embeddings, labels)
    
    assert -0.1 < score < 0.1  # Random should be near zero
```

- [ ] **Step 1.2: Run test to verify failure**

```bash
cd /home/redbear/Projects/LiquidSearcher
python -m pytest tests/evaluation/test_clustering_metrics.py::test_silhouette_score_basic -v
```

**Expected:** `ModuleNotFoundError: No module named 'src.evaluation.metrics'`

- [ ] **Step 1.3: Create module with silhouette implementation**

```python
# src/evaluation/metrics/__init__.py
"""Evaluation metrics for clustering and embedding quality."""

from src.evaluation.metrics.clustering import (
    compute_silhouette_score,
    compute_davies_bouldin_score,
    compute_calinski_harabasz_score,
    compute_all_clustering_metrics,
)

__all__ = [
    "compute_silhouette_score",
    "compute_davies_bouldin_score",
    "compute_calinski_harabasz_score",
    "compute_all_clustering_metrics",
]
```

```python
# src/evaluation/metrics/clustering.py
"""Clustering quality metrics for embedding evaluation.

All metrics are computed in PCA space, NOT UMAP space.
UMAP is for visualization only; it can create artificial structure.
"""

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def compute_silhouette_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score for clustering quality.
    
    Range: -1 to +1
    - > 0.5: Strong clustering
    - > 0.25: Meaningful clustering  
    - ~0: No structure
    - < 0: Incorrect clustering
    
    Args:
        embeddings: Array of shape (n_samples, n_features)
        labels: Array of shape (n_samples,) with cluster labels
        
    Returns:
        Silhouette score as float
    """
    if len(np.unique(labels)) < 2:
        return 0.0
    
    return float(silhouette_score(embeddings, labels, metric="cosine"))


def compute_davies_bouldin_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute Davies-Bouldin index.
    
    Lower is better (more separated clusters).
    Measures cluster separation relative to cluster size.
    
    Args:
        embeddings: Array of shape (n_samples, n_features)
        labels: Array of shape (n_samples,) with cluster labels
        
    Returns:
        Davies-Bouldin index as float
    """
    if len(np.unique(labels)) < 2:
        return float("inf")
    
    return float(davies_bouldin_score(embeddings, labels))


def compute_calinski_harabasz_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute Calinski-Harabasz score.
    
    Higher is better (more separated clusters).
    Ratio of between-cluster dispersion to within-cluster dispersion.
    
    Args:
        embeddings: Array of shape (n_samples, n_features)
        labels: Array of shape (n_samples,) with cluster labels
        
    Returns:
        Calinski-Harabasz score as float
    """
    if len(np.unique(labels)) < 2:
        return 0.0
    
    return float(calinski_harabasz_score(embeddings, labels))


def compute_all_clustering_metrics(
    embeddings: np.ndarray,
    labels_dict: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute all clustering metrics for multiple label sets.
    
    Args:
        embeddings: Array of shape (n_samples, n_features)
        labels_dict: Dict mapping label name to label array
        
    Returns:
        Dict with metric names as keys and scores as values
        
    Example:
        >>> metrics = compute_all_clustering_metrics(
        ...     embeddings,
        ...     {"sector": sector_labels, "liquidity": liquidity_labels}
        ... )
    """
    metrics = {}
    
    for label_name, labels in labels_dict.items():
        metrics[f"silhouette_{label_name}"] = compute_silhouette_score(
            embeddings, labels
        )
        
        if label_name == "sector":  # Primary clustering of interest
            metrics[f"davies_bouldin_{label_name}"] = compute_davies_bouldin_score(
                embeddings, labels
            )
            metrics[f"calinski_harabasz_{label_name}"] = compute_calinski_harabasz_score(
                embeddings, labels
            )
    
    return metrics
```

- [ ] **Step 1.4: Run tests to verify pass**

```bash
python -m pytest tests/evaluation/test_clustering_metrics.py -v
```

**Expected:** All tests PASS

- [ ] **Step 1.5: Add tests for remaining metrics**

```python
# tests/evaluation/test_clustering_metrics.py (add to file)

def test_davies_bouldin_score():
    """Test Davies-Bouldin score."""
    embeddings = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [1.0, 1.0],
        [1.1, 1.1],
    ])
    labels = np.array([0, 0, 1, 1])
    
    score = compute_davies_bouldin_score(embeddings, labels)
    
    assert score < 1.0  # Well-separated clusters
    assert isinstance(score, float)


def test_calinski_harabasz_score():
    """Test Calinski-Harabasz score."""
    embeddings = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [1.0, 1.0],
        [1.1, 1.1],
    ])
    labels = np.array([0, 0, 1, 1])
    
    score = compute_calinski_harabasz_score(embeddings, labels)
    
    assert score > 10.0  # Well-separated clusters have high CH score
    assert isinstance(score, float)


def test_compute_all_clustering_metrics():
    """Test comprehensive metrics computation."""
    np.random.seed(42)
    embeddings = np.random.randn(50, 10)
    
    labels_dict = {
        "sector": np.random.randint(0, 5, 50),
        "liquidity": np.random.randint(0, 4, 50),
    }
    
    metrics = compute_all_clustering_metrics(embeddings, labels_dict)
    
    assert "silhouette_sector" in metrics
    assert "silhouette_liquidity" in metrics
    assert "davies_bouldin_sector" in metrics
    assert "calinski_harabasz_sector" in metrics
```

- [ ] **Step 1.6: Run all tests**

```bash
python -m pytest tests/evaluation/test_clustering_metrics.py -v
```

**Expected:** All 6 tests PASS

- [ ] **Step 1.7: Commit**

```bash
git add src/evaluation/metrics/ tests/evaluation/
git commit -m "feat: add clustering metrics module for evaluation

Add silhouette, Davies-Bouldin, and Calinski-Harabasz scores
for measuring embedding cluster quality in PCA space.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 2: Embedding Cache Utility

**Purpose:** Hybrid caching system for computed embeddings to avoid recomputation.

**Files:**
- Create: `src/evaluation/utils/__init__.py`
- Create: `src/evaluation/utils/embedding_cache.py`
- Create: `tests/evaluation/test_embedding_cache.py`

- [ ] **Step 2.1: Write failing test for cache save/load**

```python
# tests/evaluation/test_embedding_cache.py
import tempfile
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from src.evaluation.utils.embedding_cache import EmbeddingCache


def test_cache_save_and_load():
    """Test saving and loading embeddings from cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir)
        
        # Create sample embeddings dataframe
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "GOOGL"],
            "date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-01"]),
            "embedding_0": [0.1, 0.2, 0.3],
            "embedding_1": [0.4, 0.5, 0.6],
            "sector": ["Technology", "Technology", "Technology"],
        })
        
        # Save to cache
        cache.save(df, "test_period", "model_v1")
        
        # Load from cache
        loaded = cache.load("test_period", "model_v1")
        
        assert loaded is not None
        assert len(loaded) == 3
        assert list(loaded["ticker"]) == ["AAPL", "MSFT", "GOOGL"]


def test_cache_miss_returns_none():
    """Test that missing cache returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir)
        
        result = cache.load("nonexistent_period", "model_v1")
        
        assert result is None
```

- [ ] **Step 2.2: Run test to verify failure**

```bash
python -m pytest tests/evaluation/test_embedding_cache.py::test_cache_save_and_load -v
```

**Expected:** `ModuleNotFoundError: No module named 'src.evaluation.utils'`

- [ ] **Step 2.3: Create embedding cache module**

```python
# src/evaluation/utils/__init__.py
"""Evaluation utilities."""

from src.evaluation.utils.embedding_cache import EmbeddingCache

__all__ = ["EmbeddingCache"]
```

```python
# src/evaluation/utils/embedding_cache.py
"""Hybrid caching system for computed embeddings.

Caches embeddings to parquet files for standard periods.
Custom periods are computed on-demand without caching.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

import pandas as pd


class EmbeddingCache:
    """Cache manager for computed embeddings.
    
    Standard periods (defined in config) are cached to parquet files.
    Custom periods are computed on-demand without caching.
    
    Cache key: {period_name}_{model_hash}.parquet
    """
    
    def __init__(self, cache_dir: str | Path):
        """Initialize cache with directory path.
        
        Args:
            cache_dir: Directory to store cached embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Standard periods that should be cached
        self.standard_periods = {
            "covid_pre", "covid_crisis",
            "ratehike_pre", "ratehike_crisis",
            "normal_2019", "normal_2021",
        }
    
    def _get_cache_path(self, period: str, model_id: str) -> Path:
        """Generate cache file path for period and model.
        
        Args:
            period: Period identifier (e.g., "covid_pre")
            model_id: Model identifier or hash
            
        Returns:
            Path to cache file
        """
        # Create deterministic filename
        key = f"{period}_{model_id}"
        filename = f"{key}.parquet"
        return self.cache_dir / filename
    
    def save(
        self,
        embeddings_df: pd.DataFrame,
        period: str,
        model_id: str,
    ) -> Path:
        """Save embeddings to cache.
        
        Args:
            embeddings_df: DataFrame with embeddings and metadata
            period: Period identifier
            model_id: Model identifier
            
        Returns:
            Path to saved cache file
        """
        cache_path = self._get_cache_path(period, model_id)
        embeddings_df.to_parquet(cache_path, index=False)
        return cache_path
    
    def load(
        self,
        period: str,
        model_id: str,
    ) -> Optional[pd.DataFrame]:
        """Load embeddings from cache if available.
        
        Args:
            period: Period identifier
            model_id: Model identifier
            
        Returns:
            DataFrame if cache hit, None if cache miss
        """
        cache_path = self._get_cache_path(period, model_id)
        
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        
        return None
    
    def is_cached(self, period: str, model_id: str) -> bool:
        """Check if embeddings are cached.
        
        Args:
            period: Period identifier
            model_id: Model identifier
            
        Returns:
            True if cache exists
        """
        cache_path = self._get_cache_path(period, model_id)
        return cache_path.exists()
    
    def should_cache(self, period: str) -> bool:
        """Determine if this period should be cached.
        
        Only standard periods defined in config are cached.
        Custom periods are computed on-demand.
        
        Args:
            period: Period identifier
            
        Returns:
            True if period should be cached
        """
        return period in self.standard_periods
    
    def clear(self, period: Optional[str] = None) -> None:
        """Clear cache for period or all cache.
        
        Args:
            period: Specific period to clear, or None for all
        """
        if period is None:
            # Clear all cache files
            for f in self.cache_dir.glob("*.parquet"):
                f.unlink()
        else:
            # Clear specific period
            for f in self.cache_dir.glob(f"{period}_*.parquet"):
                f.unlink()
    
    def list_cached(self) -> list[str]:
        """List all cached periods.
        
        Returns:
            List of cached period identifiers
        """
        periods = set()
        for f in self.cache_dir.glob("*.parquet"):
            # Extract period from filename (period_modelid.parquet)
            period = f.stem.rsplit("_", 1)[0]
            periods.add(period)
        return sorted(list(periods))
```

- [ ] **Step 2.4: Run tests to verify pass**

```bash
python -m pytest tests/evaluation/test_embedding_cache.py -v
```

**Expected:** Both tests PASS

- [ ] **Step 2.5: Add more cache tests**

```python
# tests/evaluation/test_embedding_cache.py (add to file)

def test_should_cache_standard_periods():
    """Test that standard periods are marked for caching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir)
        
        assert cache.should_cache("covid_pre") is True
        assert cache.should_cache("covid_crisis") is True
        assert cache.should_cache("ratehike_pre") is True
        assert cache.should_cache("custom_period") is False


def test_is_cached():
    """Test cache existence check."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir)
        
        assert cache.is_cached("test_period", "model_v1") is False
        
        df = pd.DataFrame({"ticker": ["AAPL"], "embedding_0": [0.1]})
        cache.save(df, "test_period", "model_v1")
        
        assert cache.is_cached("test_period", "model_v1") is True


def test_clear_cache():
    """Test clearing cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir)
        
        df = pd.DataFrame({"ticker": ["AAPL"], "embedding_0": [0.1]})
        cache.save(df, "test_period", "model_v1")
        
        assert cache.is_cached("test_period", "model_v1") is True
        
        cache.clear("test_period")
        
        assert cache.is_cached("test_period", "model_v1") is False


def test_list_cached():
    """Test listing cached periods."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(cache_dir=tmpdir)
        
        df = pd.DataFrame({"ticker": ["AAPL"], "embedding_0": [0.1]})
        cache.save(df, "period1", "model_v1")
        cache.save(df, "period2", "model_v1")
        
        cached = cache.list_cached()
        
        assert "period1" in cached
        assert "period2" in cached
```

- [ ] **Step 2.6: Run all cache tests**

```bash
python -m pytest tests/evaluation/test_embedding_cache.py -v
```

**Expected:** All 6 tests PASS

- [ ] **Step 2.7: Commit**

```bash
git add src/evaluation/utils/ tests/evaluation/
git commit -m "feat: add embedding cache utility for evaluation

Hybrid caching system that caches standard periods to parquet
and computes custom periods on-demand.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 3: UMAP Visualizer Core Class

**Purpose:** Main visualization class implementing PCA → UMAP pipeline with embedding computation.

**Files:**
- Create: `src/evaluation/visualizations/__init__.py`
- Create: `src/evaluation/visualizations/umap_visualizer.py`
- Modify: `src/evaluation/__init__.py`

- [ ] **Step 3.1: Create visualization module init**

```python
# src/evaluation/visualizations/__init__.py
"""Visualization modules for evaluation."""

from src.evaluation.visualizations.umap_visualizer import UMAPVisualizer

__all__ = ["UMAPVisualizer"]
```

- [ ] **Step 3.2: Create UMAPVisualizer class (Part 1 - Imports and Init)**

```python
# src/evaluation/visualizations/umap_visualizer.py
"""PCA + UMAP visualization for dual-encoder embeddings.

Implements Block 1 of the LiquidSearcher evaluation framework.
Uses two-stage dimensionality reduction:
1. PCA (50 components) for noise reduction - metrics computed here
2. UMAP (2D) for visualization only - NO metrics computed here
"""

import warnings
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("umap-learn not installed. UMAP visualization will not be available.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from src.evaluation.metrics.clustering import compute_all_clustering_metrics
from src.evaluation.utils.embedding_cache import EmbeddingCache


class UMAPVisualizer:
    """PCA + UMAP visualization pipeline for dual-encoder embeddings.
    
    Responsibilities:
    - Compute embeddings from model checkpoints
    - Apply PCA for noise reduction (metrics computed in PCA space)
    - Apply UMAP (2D) for visualization only
    - Generate static PNG and interactive HTML visualizations
    - Compute crisis snapshot comparisons
    - Calculate clustering quality metrics
    
    Key methodological choices:
    - Clustering metrics computed in PCA space, NOT UMAP space
    - Crisis comparison uses fixed reference UMAP fit on pre-crisis data
    - Unit of analysis: one point per ticker per period
    """
    
    # Pre-defined period configurations
    PERIODS = {
        "covid_pre": ("2019-01-01", "2020-01-31"),
        "covid_crisis": ("2020-02-01", "2020-05-31"),
        "ratehike_pre": ("2021-01-01", "2021-12-31"),
        "ratehike_crisis": ("2022-01-01", "2022-10-31"),
        "normal_2019": ("2019-01-01", "2019-12-31"),
        "normal_2021": ("2021-01-01", "2021-12-31"),
    }
    
    def __init__(
        self,
        model: "DualEncoderModule",
        feature_loader: "FeatureLoader",
        output_dir: str | Path,
        cache_dir: Optional[str | Path] = None,
        n_pca_components: int = 50,
        umap_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        random_state: int = 42,
    ):
        """Initialize UMAP visualizer.
        
        Args:
            model: Trained DualEncoderModule
            feature_loader: Feature loader for data access
            output_dir: Directory for output figures
            cache_dir: Directory for embedding cache (default: output_dir/cache)
            n_pca_components: Number of PCA components (default: 50)
            umap_neighbors: UMAP n_neighbors parameter (default: 15)
            umap_min_dist: UMAP min_dist parameter (default: 0.1)
            random_state: Random seed for reproducibility
        """
        if not UMAP_AVAILABLE:
            raise ImportError("umap-learn is required. Install with: pip install umap-learn")
        
        self.model = model
        self.feature_loader = feature_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if cache_dir is None:
            cache_dir = self.output_dir / "cache"
        self.cache = EmbeddingCache(cache_dir)
        
        self.n_pca_components = n_pca_components
        self.umap_neighbors = umap_neighbors
        self.umap_min_dist = umap_min_dist
        self.random_state = random_state
        
        # Initialize dimensionality reduction models
        self.pca_model: Optional[PCA] = None
        self.umap_model: Optional[umap.UMAP] = None
```

- [ ] **Step 3.3: Add embedding computation method**

```python
# src/evaluation/visualizations/umap_visualizer.py (add to class)

    def compute_embeddings(
        self,
        period_start: str,
        period_end: str,
        aggregation: Literal["end_period", "mean", "all_dates"] = "end_period",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Compute embeddings for specified period.
        
        Unit of analysis:
        - "end_period": One point per ticker (last trading day) - DEFAULT
        - "mean": One point per ticker (mean embedding across period)
        - "all_dates": Every ticker-date observation (exploratory only)
        
        Args:
            period_start: Start date (YYYY-MM-DD)
            period_end: End date (YYYY-MM-DD)
            aggregation: How to aggregate embeddings per ticker
            use_cache: Whether to use/load from cache
            
        Returns:
            DataFrame with columns:
            - ticker: str
            - date: datetime
            - embedding_0..embedding_N: float
            - sector: str
            - liquidity_tier: str (Q1-Q4)
            - market_cap_tier: str
        """
        # Determine period name for caching
        period_name = None
        for name, (start, end) in self.PERIODS.items():
            if start == period_start and end == period_end:
                period_name = name
                break
        
        # Try cache first for standard periods
        model_id = getattr(self.model, "model_id", "default")
        if use_cache and period_name and self.cache.should_cache(period_name):
            cached = self.cache.load(period_name, model_id)
            if cached is not None:
                return cached
        
        # Load features for period
        features_df = self.feature_loader.load_period(period_start, period_end)
        
        if aggregation == "end_period":
            # Take last observation per ticker
            features_df = features_df.sort_values("date").groupby("ticker").last().reset_index()
        elif aggregation == "mean":
            # Average embeddings across period (implement later)
            raise NotImplementedError("Mean aggregation not yet implemented")
        
        # Compute embeddings
        embeddings_list = []
        metadata_list = []
        
        self.model.eval()
        with torch.no_grad():
            for _, row in features_df.iterrows():
                # TODO: Extract temporal and tabular features
                # temporal = torch.tensor(row["temporal_features"])
                # tabular = torch.tensor(row["tabular_features"])
                # embedding = self.model.forward(temporal, tabular)
                
                # Placeholder: just store metadata for now
                metadata_list.append({
                    "ticker": row["ticker"],
                    "date": row["date"],
                    "sector": row.get("sector", "Unknown"),
                })
                embeddings_list.append(np.zeros(256))  # Placeholder
        
        # Build dataframe
        embedding_cols = {f"embedding_{i}": [e[i] for e in embeddings_list] 
                         for i in range(256)}
        
        result_df = pd.DataFrame(metadata_list)
        for col, values in embedding_cols.items():
            result_df[col] = values
        
        # Compute tiers
        result_df = self._compute_tiers(result_df)
        
        # Cache if standard period
        if period_name and self.cache.should_cache(period_name):
            self.cache.save(result_df, period_name, model_id)
        
        return result_df
    
    def _compute_tiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute liquidity and market-cap tiers within period.
        
        Args:
            df: DataFrame with ticker data
            
        Returns:
            DataFrame with added tier columns
        """
        # Compute liquidity quartiles (Q1=most liquid, Q4=least liquid)
        if "avg_daily_volume" in df.columns:
            df["liquidity_tier"] = pd.qcut(
                df["avg_daily_volume"],
                q=4,
                labels=["Q4", "Q3", "Q2", "Q1"]  # Q1 = highest volume
            )
        else:
            df["liquidity_tier"] = "Unknown"
        
        # Compute market-cap tiers
        if "market_cap" in df.columns:
            df["market_cap_tier"] = pd.cut(
                df["market_cap"],
                bins=[0, 2e9, 10e9, 200e9, float("inf")],
                labels=["Micro", "Small", "Mid", "Large"]
            )
        else:
            df["market_cap_tier"] = "Unknown"
        
        return df
```

- [ ] **Step 3.4: Add PCA and UMAP projection methods**

```python
# src/evaluation/visualizations/umap_visualizer.py (add to class)

    def project_pca(
        self,
        embeddings: np.ndarray,
        fit_on: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply PCA for noise reduction.
        
        Clustering metrics are computed on this output, NOT on UMAP output.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            fit_on: Optional separate data to fit PCA on (for crisis comparison)
            
        Returns:
            PCA-transformed embeddings (n_samples, n_pca_components)
        """
        fit_data = fit_on if fit_on is not None else embeddings
        
        self.pca_model = PCA(
            n_components=self.n_pca_components,
            whiten=False,
            random_state=self.random_state,
        )
        
        self.pca_model.fit(fit_data)
        
        # Report variance explained
        var_explained = np.sum(self.pca_model.explained_variance_ratio_)
        print(f"PCA: {self.n_pca_components} components explain {var_explained:.2%} variance")
        
        return self.pca_model.transform(embeddings)
    
    def project_umap(
        self,
        pca_embeddings: np.ndarray,
        fit_mode: Literal["reference", "combined", "separate"] = "reference",
        reference_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply UMAP for 2D visualization ONLY.
        
        IMPORTANT: Do NOT compute clustering metrics on UMAP output.
        UMAP is nonlinear and can create artificial structure.
        
        Args:
            pca_embeddings: PCA-reduced embeddings (n_samples, n_pca)
            fit_mode: How to fit UMAP
                - "reference": Fit on reference data, transform all
                - "combined": Fit on all data combined
                - "separate": Fit independently (structure comparison only)
            reference_data: Reference data for "reference" mode
            
        Returns:
            2D UMAP projection (n_samples, 2)
        """
        if fit_mode == "reference":
            if reference_data is None:
                raise ValueError("reference_data required for 'reference' fit_mode")
            
            # Fit on reference data only
            self.umap_model = umap.UMAP(
                n_components=2,
                n_neighbors=self.umap_neighbors,
                min_dist=self.umap_min_dist,
                metric="cosine",
                random_state=self.random_state,
            )
            self.umap_model.fit(reference_data)
            
            # Transform all data
            return self.umap_model.transform(pca_embeddings)
        
        elif fit_mode == "combined":
            self.umap_model = umap.UMAP(
                n_components=2,
                n_neighbors=self.umap_neighbors,
                min_dist=self.umap_min_dist,
                metric="cosine",
                random_state=self.random_state,
            )
            return self.umap_model.fit_transform(pca_embeddings)
        
        elif fit_mode == "separate":
            self.umap_model = umap.UMAP(
                n_components=2,
                n_neighbors=self.umap_neighbors,
                min_dist=self.umap_min_dist,
                metric="cosine",
                random_state=self.random_state,
            )
            return self.umap_model.fit_transform(pca_embeddings)
        
        else:
            raise ValueError(f"Unknown fit_mode: {fit_mode}")
```

- [ ] **Step 3.5: Commit progress**

```bash
git add src/evaluation/visualizations/
git commit -m "feat: add UMAPVisualizer core class (WIP)

Add embedding computation, PCA projection, and UMAP projection.
Structure complete, implementation of compute_embeddings needs
feature loader integration.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 4: Plot Generation Methods

**Purpose:** Static PNG and interactive HTML visualization generation.

**Files:**
- Modify: `src/evaluation/visualizations/umap_visualizer.py`

- [ ] **Step 4.1: Add static plot generation method**

```python
# src/evaluation/visualizations/umap_visualizer.py (add to class)

    def generate_static_plot(
        self,
        projection: np.ndarray,
        metadata: pd.DataFrame,
        color_by: Literal["sector", "liquidity", "market_cap"],
        title: str,
        filename: str,
    ) -> Path:
        """Generate publication-quality static PNG.
        
        Args:
            projection: 2D UMAP projection (n_samples, 2)
            metadata: DataFrame with color column
            color_by: Which column to color by
            title: Plot title
            filename: Output filename (without extension)
            
        Returns:
            Path to saved figure
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for static plots")
        
        # Map color_by to column name
        column_map = {
            "sector": "sector",
            "liquidity": "liquidity_tier",
            "market_cap": "market_cap_tier",
        }
        color_column = column_map.get(color_by, color_by)
        
        # Set up figure
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        
        # Color schemes
        if color_by == "sector":
            palette = "tab20"
        elif color_by == "liquidity":
            palette = "viridis"
        else:
            palette = "Set2"
        
        # Create scatter plot
        scatter = sns.scatterplot(
            data=metadata,
            x=projection[:, 0],
            y=projection[:, 1],
            hue=color_column,
            palette=palette,
            alpha=0.6,
            s=40,
            ax=ax,
        )
        
        # Styling
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(title=color_by.replace("_", " ").title(), bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"{filename}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        print(f"Saved static plot: {output_path}")
        return output_path
    
    def generate_interactive_html(
        self,
        projection: np.ndarray,
        metadata: pd.DataFrame,
        color_by: Literal["sector", "liquidity", "market_cap"],
        title: str,
        filename: str,
    ) -> Path:
        """Generate interactive Plotly HTML visualization.
        
        Args:
            projection: 2D UMAP projection (n_samples, 2)
            metadata: DataFrame with ticker info
            color_by: Which column to color by
            title: Plot title
            filename: Output filename (without extension)
            
        Returns:
            Path to saved HTML
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for interactive plots")
        
        # Map color_by to column name
        column_map = {
            "sector": "sector",
            "liquidity": "liquidity_tier",
            "market_cap": "market_cap_tier",
        }
        color_column = column_map.get(color_by, color_by)
        
        # Create hover text
        hover_data = ["ticker", "sector", "liquidity_tier", "market_cap_tier"]
        hover_data = [h for h in hover_data if h in metadata.columns]
        
        # Create figure
        fig = px.scatter(
            metadata,
            x=projection[:, 0],
            y=projection[:, 1],
            color=color_column,
            hover_data=hover_data,
            title=title,
            opacity=0.6,
        )
        
        # Update layout
        fig.update_traces(marker=dict(size=8))
        fig.update_layout(
            xaxis_visible=False,
            yaxis_visible=False,
            showlegend=True,
            legend_title_text=color_by.replace("_", " ").title(),
        )
        
        # Save
        output_path = self.output_dir / f"{filename}.html"
        fig.write_html(output_path)
        
        print(f"Saved interactive plot: {output_path}")
        return output_path
```

- [ ] **Step 4.2: Add crisis comparison method**

```python
# src/evaluation/visualizations/umap_visualizer.py (add to class)

    def generate_crisis_comparison(
        self,
        pre_crisis_period: Tuple[str, str],
        crisis_period: Tuple[str, str],
        period_name: str,
        projection_mode: Literal["fixed_reference", "separate"] = "fixed_reference",
    ) -> Tuple[Path, Path]:
        """Generate side-by-side crisis snapshot comparison.
        
        Args:
            pre_crisis_period: (start, end) for pre-crisis
            crisis_period: (start, end) for crisis
            period_name: Name for output files (e.g., "covid")
            projection_mode: "fixed_reference" or "separate"
            
        Returns:
            Tuple of (static_path, html_path)
        """
        print(f"Computing embeddings for {period_name} comparison...")
        
        # Compute embeddings for both periods
        pre_df = self.compute_embeddings(
            pre_crisis_period[0],
            pre_crisis_period[1],
            aggregation="end_period",
        )
        
        crisis_df = self.compute_embeddings(
            crisis_period[0],
            crisis_period[1],
            aggregation="end_period",
        )
        
        # Extract embeddings
        pre_embed = self._extract_embeddings(pre_df)
        crisis_embed = self._extract_embeddings(crisis_df)
        
        # Combine for consistent PCA
        combined_embed = np.vstack([pre_embed, crisis_embed])
        combined_pca = self.project_pca(combined_embed)
        
        n_pre = len(pre_embed)
        pre_pca = combined_pca[:n_pre]
        crisis_pca = combined_pca[n_pre:]
        
        # UMAP projection
        if projection_mode == "fixed_reference":
            # Fit on pre-crisis only, transform both
            crisis_umap = self.project_umap(
                np.vstack([pre_pca, crisis_pca]),
                fit_mode="reference",
                reference_data=pre_pca,
            )
            pre_proj = crisis_umap[:n_pre]
            crisis_proj = crisis_umap[n_pre:]
        else:
            # Separate projections
            pre_proj = self.project_umap(pre_pca, fit_mode="separate")
            crisis_proj = self.project_umap(crisis_pca, fit_mode="separate")
        
        # Generate side-by-side static plot
        static_path = self._generate_crisis_static_plot(
            pre_proj, pre_df,
            crisis_proj, crisis_df,
            period_name,
            projection_mode,
        )
        
        # Generate interactive HTML
        html_path = self._generate_crisis_html(
            pre_proj, pre_df,
            crisis_proj, crisis_df,
            period_name,
            projection_mode,
        )
        
        return static_path, html_path
    
    def _extract_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """Extract embedding columns from dataframe."""
        embedding_cols = [c for c in df.columns if c.startswith("embedding_")]
        return df[embedding_cols].values
    
    def _generate_crisis_static_plot(
        self,
        pre_proj: np.ndarray,
        pre_df: pd.DataFrame,
        crisis_proj: np.ndarray,
        crisis_df: pd.DataFrame,
        period_name: str,
        projection_mode: str,
    ) -> Path:
        """Generate side-by-side static plot."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
        
        # Pre-crisis plot
        ax1 = axes[0]
        sns.scatterplot(
            x=pre_proj[:, 0],
            y=pre_proj[:, 1],
            hue=pre_df["sector"],
            palette="tab20",
            alpha=0.6,
            s=40,
            ax=ax1,
            legend=False,
        )
        ax1.set_title(f"Pre-Crisis: {period_name}", fontsize=12, fontweight="bold")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.grid(True, alpha=0.3)
        
        # Crisis plot
        ax2 = axes[1]
        sns.scatterplot(
            x=crisis_proj[:, 0],
            y=crisis_proj[:, 1],
            hue=crisis_df["sector"],
            palette="tab20",
            alpha=0.6,
            s=40,
            ax=ax2,
        )
        ax2.set_title(f"Crisis: {period_name}", fontsize=12, fontweight="bold")
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.grid(True, alpha=0.3)
        ax2.legend(title="Sector", bbox_to_anchor=(1.05, 1), loc="upper left")
        
        # Set same axis limits for comparison
        x_min = min(pre_proj[:, 0].min(), crisis_proj[:, 0].min())
        x_max = max(pre_proj[:, 0].max(), crisis_proj[:, 0].max())
        y_min = min(pre_proj[:, 1].min(), crisis_proj[:, 1].min())
        y_max = max(pre_proj[:, 1].max(), crisis_proj[:, 1].max())
        
        margin = 0.1
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        for ax in axes:
            ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
            ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
        
        plt.tight_layout()
        
        filename = f"umap_crisis_{period_name}_{projection_mode}"
        output_path = self.output_dir / f"{filename}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        print(f"Saved crisis comparison: {output_path}")
        return output_path
    
    def _generate_crisis_html(
        self,
        pre_proj: np.ndarray,
        pre_df: pd.DataFrame,
        crisis_proj: np.ndarray,
        crisis_df: pd.DataFrame,
        period_name: str,
        projection_mode: str,
    ) -> Path:
        """Generate interactive HTML with both periods."""
        # Add period label
        pre_df = pre_df.copy()
        crisis_df = crisis_df.copy()
        pre_df["period"] = "Pre-Crisis"
        crisis_df["period"] = "Crisis"
        
        # Combine data
        combined_df = pd.concat([pre_df, crisis_df], ignore_index=True)
        combined_proj = np.vstack([pre_proj, crisis_proj])
        
        # Create figure with subplots
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"Pre-Crisis: {period_name}", f"Crisis: {period_name}"),
        )
        
        # Add pre-crisis trace
        for sector in pre_df["sector"].unique():
            mask = pre_df["sector"] == sector
            fig.add_trace(
                go.Scatter(
                    x=pre_proj[mask, 0],
                    y=pre_proj[mask, 1],
                    mode="markers",
                    name=sector,
                    marker=dict(size=8, opacity=0.6),
                    legendgroup=sector,
                    showlegend=True,
                ),
                row=1, col=1,
            )
        
        # Add crisis trace
        for sector in crisis_df["sector"].unique():
            mask = crisis_df["sector"] == sector
            fig.add_trace(
                go.Scatter(
                    x=crisis_proj[mask, 0],
                    y=crisis_proj[mask, 1],
                    mode="markers",
                    name=sector,
                    marker=dict(size=8, opacity=0.6),
                    legendgroup=sector,
                    showlegend=False,
                ),
                row=1, col=2,
            )
        
        fig.update_layout(
            title=f"Crisis Comparison: {period_name.title()}",
            height=600,
            width=1400,
        )
        
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        
        filename = f"umap_crisis_{period_name}_{projection_mode}"
        output_path = self.output_dir / f"{filename}.html"
        fig.write_html(output_path)
        
        print(f"Saved interactive crisis comparison: {output_path}")
        return output_path
```

- [ ] **Step 4.3: Commit plot generation**

```bash
git add src/evaluation/visualizations/umap_visualizer.py
git commit -m "feat: add plot generation methods to UMAPVisualizer

Add static PNG and interactive HTML generation.
Add crisis comparison with fixed-reference and separate modes.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 5: Clustering Metrics Integration

**Purpose:** Compute and save clustering quality metrics.

**Files:**
- Modify: `src/evaluation/visualizations/umap_visualizer.py`

- [ ] **Step 5.1: Add clustering metrics computation method**

```python
# src/evaluation/visualizations/umap_visualizer.py (add to class)

    def compute_clustering_metrics(
        self,
        embeddings_df: pd.DataFrame,
        pca_embeddings: np.ndarray,
    ) -> dict:
        """Compute clustering quality metrics in PCA space.
        
        Args:
            embeddings_df: DataFrame with metadata (sector, liquidity_tier)
            pca_embeddings: PCA-reduced embeddings
            
        Returns:
            Dict with metric names and values
        """
        # Prepare label arrays
        labels_dict = {}
        
        if "sector" in embeddings_df.columns:
            # Convert sector to numeric labels
            sector_labels, _ = pd.factorize(embeddings_df["sector"])
            labels_dict["sector"] = sector_labels
        
        if "liquidity_tier" in embeddings_df.columns:
            liquidity_labels, _ = pd.factorize(embeddings_df["liquidity_tier"])
            labels_dict["liquidity"] = liquidity_labels
        
        if "market_cap_tier" in embeddings_df.columns:
            market_cap_labels, _ = pd.factorize(embeddings_df["market_cap_tier"])
            labels_dict["market_cap"] = market_cap_labels
        
        # Compute metrics
        metrics = compute_all_clustering_metrics(pca_embeddings, labels_dict)
        
        # Add baseline comparisons (random permutation)
        for label_name in list(labels_dict.keys()):
            labels = labels_dict[label_name]
            shuffled_labels = np.random.RandomState(42).permutation(labels)
            random_score = compute_all_clustering_metrics(
                pca_embeddings,
                {label_name: shuffled_labels},
            )
            metrics[f"silhouette_{label_name}_random"] = random_score[f"silhouette_{label_name}"]
        
        return metrics
    
    def save_clustering_metrics(
        self,
        metrics: dict,
        period_name: str,
    ) -> Path:
        """Save clustering metrics to CSV.
        
        Args:
            metrics: Dict of metric names and values
            period_name: Name of period for filename
            
        Returns:
            Path to saved CSV
        """
        output_path = self.output_dir / "clustering_metrics.csv"
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame([metrics])
        metrics_df["period"] = period_name
        
        # Append or create new
        if output_path.exists():
            existing = pd.read_csv(output_path)
            combined = pd.concat([existing, metrics_df], ignore_index=True)
            combined.to_csv(output_path, index=False)
        else:
            metrics_df.to_csv(output_path, index=False)
        
        print(f"Saved clustering metrics: {output_path}")
        return output_path
```

- [ ] **Step 5.2: Commit metrics integration**

```bash
git add src/evaluation/visualizations/umap_visualizer.py
git commit -m "feat: add clustering metrics computation and saving

Integrate clustering metrics into UMAPVisualizer with baseline
comparisons and CSV output.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 6: CLI Script

**Purpose:** Command-line interface for running visualization pipeline.

**Files:**
- Create: `scripts/visualization/__init__.py`
- Create: `scripts/visualization/umap_plots.py`

- [ ] **Step 6.1: Create CLI script**

```python
#!/usr/bin/env python
# scripts/visualization/umap_plots.py
"""CLI for generating UMAP visualizations.

Usage:
    python -m scripts.visualization.umap_plots \
        --checkpoint checkpoints/best_model.ckpt \
        --feature-dir data/processed \
        --output-dir results/figures
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate UMAP visualizations for LiquidSearcher embeddings"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--feature-dir",
        type=str,
        required=True,
        help="Directory with feature parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/figures",
        help="Output directory for figures (default: results/figures)",
    )
    parser.add_argument(
        "--periods",
        nargs="+",
        choices=["covid", "rate_hike"],
        default=["covid", "rate_hike"],
        help="Crisis periods to analyze",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Generate only static PNG plots",
    )
    parser.add_argument(
        "--interactive-only",
        action="store_true",
        help="Generate only interactive HTML plots",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="Number of PCA components",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)
    
    # Import here to allow argument parsing without dependencies
    import torch
    from src.models.dual_encoder import DualEncoderModule
    from src.data.feature_loader import FeatureLoader
    from src.evaluation.visualizations import UMAPVisualizer
    
    print(f"Loading model from {checkpoint_path}...")
    
    # Load model
    model = DualEncoderModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Initialize feature loader
    feature_loader = FeatureLoader(args.feature_dir)
    
    # Initialize visualizer
    visualizer = UMAPVisualizer(
        model=model,
        feature_loader=feature_loader,
        output_dir=args.output_dir,
        n_pca_components=args.pca_components,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
    )
    
    print(f"Output directory: {args.output_dir}")
    
    # Generate visualizations for each period
    for period_key in args.periods:
        print(f"\n{'='*60}")
        print(f"Processing {period_key.upper()} period")
        print(f"{'='*60}")
        
        if period_key == "covid":
            pre_period = ("2019-01-01", "2020-01-31")
            crisis_period = ("2020-02-01", "2020-05-31")
        elif period_key == "rate_hike":
            pre_period = ("2021-01-01", "2021-12-31")
            crisis_period = ("2022-01-01", "2022-10-31")
        else:
            continue
        
        # Generate crisis comparison
        if not args.interactive_only:
            print("\nGenerating static crisis comparison...")
            visualizer.generate_crisis_comparison(
                pre_period,
                crisis_period,
                period_name=period_key,
                projection_mode="fixed_reference",
            )
        
        if not args.static_only:
            print("\nGenerating interactive crisis comparison...")
            visualizer.generate_crisis_comparison(
                pre_period,
                crisis_period,
                period_name=period_key,
                projection_mode="fixed_reference",
            )
        
        # Compute metrics for pre-crisis
        print("\nComputing clustering metrics...")
        pre_df = visualizer.compute_embeddings(
            pre_period[0],
            pre_period[1],
            use_cache=not args.no_cache,
        )
        pre_embed = visualizer._extract_embeddings(pre_df)
        pre_pca = visualizer.project_pca(pre_embed)
        
        metrics = visualizer.compute_clustering_metrics(pre_df, pre_pca)
        visualizer.save_clustering_metrics(metrics, f"{period_key}_pre")
        
        print(f"\nMetrics for {period_key}_pre:")
        for key, value in metrics.items():
            if not key.endswith("_random"):
                print(f"  {key}: {value:.3f}")
    
    print(f"\n{'='*60}")
    print("Visualization complete!")
    print(f"Output directory: {Path(args.output_dir).absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.2: Create visualization scripts init**

```python
# scripts/visualization/__init__.py
"""Visualization scripts for LiquidSearcher."""
```

- [ ] **Step 6.3: Commit CLI script**

```bash
git add scripts/visualization/
git commit -m "feat: add CLI script for UMAP visualization

Add umap_plots.py with full argument parsing and pipeline execution.
Supports both static and interactive outputs, multiple periods.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 7: Integration and Testing

**Purpose:** Wire up components and test the full pipeline.

**Files:**
- Create: `tests/evaluation/test_umap_visualizer.py`
- Modify: `src/evaluation/__init__.py`

- [ ] **Step 7.1: Update evaluation package init**

```python
# src/evaluation/__init__.py
"""LiquidSearcher evaluation framework."""

from src.evaluation.visualizations import UMAPVisualizer
from src.evaluation.metrics import (
    compute_silhouette_score,
    compute_davies_bouldin_score,
    compute_calinski_harabasz_score,
    compute_all_clustering_metrics,
)

__all__ = [
    "UMAPVisualizer",
    "compute_silhouette_score",
    "compute_davies_bouldin_score",
    "compute_calinski_harabasz_score",
    "compute_all_clustering_metrics",
]
```

- [ ] **Step 7.2: Create unit tests for UMAPVisualizer**

```python
# tests/evaluation/test_umap_visualizer.py
"""Unit tests for UMAPVisualizer."""

import tempfile
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def test_umap_visualizer_import():
    """Test that UMAPVisualizer can be imported."""
    from src.evaluation import UMAPVisualizer
    assert UMAPVisualizer is not None


def test_periods_configuration():
    """Test that periods are properly configured."""
    from src.evaluation.visualizations.umap_visualizer import UMAPVisualizer
    
    assert "covid_pre" in UMAPVisualizer.PERIODS
    assert "covid_crisis" in UMAPVisualizer.PERIODS
    assert "ratehike_pre" in UMAPVisualizer.PERIODS
    assert "ratehike_crisis" in UMAPVisualizer.PERIODS
    
    # Check date format
    start, end = UMAPVisualizer.PERIODS["covid_pre"]
    assert start == "2019-01-01"
    assert end == "2020-01-31"


def test_extract_embeddings():
    """Test embedding extraction from dataframe."""
    from src.evaluation.visualizations.umap_visualizer import UMAPVisualizer
    
    # Create mock dataframe
    df = pd.DataFrame({
        "ticker": ["AAPL", "MSFT"],
        "embedding_0": [0.1, 0.2],
        "embedding_1": [0.3, 0.4],
        "embedding_2": [0.5, 0.6],
    })
    
    # Create mock visualizer (without model)
    visualizer = object.__new__(UMAPVisualizer)
    
    embeddings = visualizer._extract_embeddings(df)
    
    assert embeddings.shape == (2, 3)
    assert np.allclose(embeddings[0], [0.1, 0.3, 0.5])


def test_compute_tiers():
    """Test tier computation."""
    from src.evaluation.visualizations.umap_visualizer import UMAPVisualizer
    
    df = pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "avg_daily_volume": [1000, 2000, 3000, 4000],
        "market_cap": [1e8, 5e9, 50e9, 500e9],
    })
    
    visualizer = object.__new__(UMAPVisualizer)
    result = visualizer._compute_tiers(df)
    
    assert "liquidity_tier" in result.columns
    assert "market_cap_tier" in result.columns
    assert len(result["liquidity_tier"].unique()) <= 4


def test_project_pca():
    """Test PCA projection."""
    pytest.importorskip("sklearn")
    from src.evaluation.visualizations.umap_visualizer import UMAPVisualizer
    
    # Create embeddings
    np.random.seed(42)
    embeddings = np.random.randn(100, 256)
    
    visualizer = object.__new__(UMAPVisualizer)
    visualizer.n_pca_components = 50
    visualizer.random_state = 42
    
    projected = visualizer.project_pca(embeddings)
    
    assert projected.shape == (100, 50)


@pytest.mark.skip(reason="Requires full model and data setup")
def test_full_pipeline_integration():
    """Integration test for full pipeline (requires model checkpoint)."""
    pass
```

- [ ] **Step 7.3: Run all tests**

```bash
python -m pytest tests/evaluation/ -v --tb=short
```

**Expected:** Most tests PASS (some may be skipped if dependencies missing)

- [ ] **Step 7.4: Commit tests**

```bash
git add tests/evaluation/ src/evaluation/__init__.py
git commit -m "test: add UMAPVisualizer unit tests

Add tests for import, periods config, embedding extraction,
tier computation, and PCA projection.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 8: Feature Loader for all_features.parquet

**Purpose:** Create a simple feature loader that reads from `data/processed/all_features.parquet`.

**Files:**
- Create: `src/evaluation/utils/feature_loader.py`

- [ ] **Step 8.1: Create FeatureLoader class**

```python
# src/evaluation/utils/feature_loader.py
"""Feature loader for evaluation pipeline.

Reads from data/processed/all_features.parquet which contains
all symbols and dates in a single file.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


class FeatureLoader:
    """Load features from all_features.parquet for specified periods."""
    
    def __init__(self, feature_path: str | Path):
        """Initialize with path to all_features.parquet.
        
        Args:
            feature_path: Path to all_features.parquet file
        """
        self.feature_path = Path(feature_path)
        
        if not self.feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {self.feature_path}")
        
        # Cache for loaded data (avoid reloading for each period)
        self._data: Optional[pd.DataFrame] = None
    
    @property
    def data(self) -> pd.DataFrame:
        """Lazy-load the full dataset."""
        if self._data is None:
            print(f"Loading features from {self.feature_path}...")
            self._data = pd.read_parquet(self.feature_path)
            self._data["date"] = pd.to_datetime(self._data["date"])
            print(f"Loaded {len(self._data):,} rows")
        return self._data
    
    def load_period(
        self,
        period_start: str,
        period_end: str,
        symbols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Load features for specified period.
        
        Args:
            period_start: Start date (YYYY-MM-DD)
            period_end: End date (YYYY-MM-DD)
            symbols: Optional list of symbols to filter (default: all)
            
        Returns:
            DataFrame with features for period
        """
        df = self.data
        
        # Filter by date range
        mask = (df["date"] >= period_start) & (df["date"] <= period_end)
        df = df[mask].copy()
        
        # Filter by symbols if specified
        if symbols is not None:
            df = df[df["symbol"].isin(symbols)]
        
        print(f"Period {period_start} to {period_end}: {len(df):,} rows, {df['symbol'].nunique()} symbols")
        
        return df
    
    def get_available_date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Get the available date range in the dataset.
        
        Returns:
            Tuple of (min_date, max_date)
        """
        df = self.data
        return df["date"].min(), df["date"].max()
    
    def get_available_symbols(self) -> list[str]:
        """Get list of available symbols.
        
        Returns:
            List of symbol names
        """
        return sorted(self.data["symbol"].unique().tolist())
```

- [ ] **Step 8.2: Update utils __init__.py**

```python
# src/evaluation/utils/__init__.py
"""Evaluation utilities."""

from src.evaluation.utils.embedding_cache import EmbeddingCache
from src.evaluation.utils.feature_loader import FeatureLoader

__all__ = ["EmbeddingCache", "FeatureLoader"]
```

---

## Task 9: Wire Up UMAPVisualizer to Use FeatureLoader

**Purpose:** Update UMAPVisualizer to use the new FeatureLoader.

**Files:**
- Modify: `src/evaluation/visualizations/umap_visualizer.py`

- [ ] **Step 9.1: Update compute_embeddings to use FeatureLoader**

```python
# src/evaluation/visualizations/umap_visualizer.py (update compute_embeddings method)

    def compute_embeddings(
        self,
        period_start: str,
        period_end: str,
        aggregation: Literal["end_period", "mean", "all_dates"] = "end_period",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Compute embeddings for specified period.
        
        Unit of analysis:
        - "end_period": One point per ticker (last trading day) - DEFAULT
        - "mean": One point per ticker (mean embedding across period)
        - "all_dates": Every ticker-date observation (exploratory only)
        
        Args:
            period_start: Start date (YYYY-MM-DD)
            period_end: End date (YYYY-MM-DD)
            aggregation: How to aggregate embeddings per ticker
            use_cache: Whether to use/load from cache
            
        Returns:
            DataFrame with columns:
            - ticker: str
            - date: datetime
            - embedding_0..embedding_N: float
            - sector: str
            - liquidity_tier: str (Q1-Q4)
            - market_cap_tier: str
        """
        # Determine period name for caching
        period_name = None
        for name, (start, end) in self.PERIODS.items():
            if start == period_start and end == period_end:
                period_name = name
                break
        
        # Try cache first for standard periods
        model_id = getattr(self.model, "model_id", "default")
        if use_cache and period_name and self.cache.should_cache(period_name):
            cached = self.cache.load(period_name, model_id)
            if cached is not None:
                print(f"Using cached embeddings for {period_name}")
                return cached
        
        # Load features for period using FeatureLoader
        print(f"Loading features for {period_start} to {period_end}...")
        features_df = self.feature_loader.load_period(period_start, period_end)
        
        if aggregation == "end_period":
            # Take last observation per ticker
            features_df = features_df.sort_values("date").groupby("symbol").last().reset_index()
            features_df = features_df.rename(columns={"symbol": "ticker"})
        elif aggregation == "mean":
            # Average across period per ticker
            numeric_cols = features_df.select_dtypes(include=["float64", "int64"]).columns
            features_df = features_df.groupby("symbol")[numeric_cols].mean().reset_index()
            features_df = features_df.rename(columns={"symbol": "ticker"})
            features_df["date"] = pd.Timestamp(period_end)
        
        # Compute tiers within this period
        features_df = self._compute_tiers(features_df)
        
        # Compute embeddings (placeholder - will be implemented when model is ready)
        # For now, create placeholder embeddings
        print("Computing embeddings (placeholder)...")
        n_stocks = len(features_df)
        
        # Placeholder: random embeddings with some sector structure
        np.random.seed(42)
        sectors = features_df["gsector"].fillna(0).astype(int).values
        n_sectors = sectors.max() + 1
        
        embeddings = np.zeros((n_stocks, 256))
        for i in range(n_stocks):
            sector_id = sectors[i] % n_sectors
            # Each sector gets a cluster center
            center = np.zeros(256)
            center[sector_id * 20:(sector_id + 1) * 20] = 1.0
            embeddings[i] = center + np.random.randn(256) * 0.3
        
        # Add embedding columns to dataframe
        for i in range(256):
            features_df[f"embedding_{i}"] = embeddings[:, i]
        
        # Select final columns
        result_df = features_df[["ticker", "date", "sector"] + 
                               [f"embedding_{i}" for i in range(256)] +
                               ["liquidity_tier", "market_cap_tier"]].copy()
        
        # Cache if standard period
        if period_name and self.cache.should_cache(period_name):
            self.cache.save(result_df, period_name, model_id)
        
        return result_df
```

---

## Task 10: Final Integration Check

**Purpose:** Verify the module can be imported and basic usage works.

- [ ] **Step 10.1: Verify imports work**

```bash
cd /home/redbear/Projects/LiquidSearcher
python -c "from src.evaluation import UMAPVisualizer, FeatureLoader; print('Import successful')"
python -c "from src.evaluation.metrics import compute_silhouette_score; print('Metrics import successful')"
```

**Expected:** Both commands print success messages

- [ ] **Step 10.2: Update pyproject.toml with evaluation dependencies**

```toml
# pyproject.toml - add to [project.optional-dependencies]

evaluation = [
    "umap-learn>=0.5.0",
    "plotly>=5.14.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]
```

- [ ] **Step 10.3: Run linting and type checking**

```bash
python -m ruff check src/evaluation/
python -m ruff check scripts/visualization/
python -m mypy src/evaluation/ --ignore-missing-imports
```

**Expected:** No critical errors (warnings acceptable for prototype)

- [ ] **Step 10.4: Final commit**

```bash
git add pyproject.toml
git commit -m "chore: add evaluation dependencies and finalize UMAP module

Add umap-learn, plotly, matplotlib, seaborn to optional dependencies.
Module ready for integration testing with actual model.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

**Expected:** Both commands print success messages

- [ ] **Step 10.2: Update pyproject.toml with evaluation dependencies**

```toml
# pyproject.toml - add to [project.optional-dependencies]

evaluation = [
    "umap-learn>=0.5.0",
    "plotly>=5.14.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]
```

- [ ] **Step 10.3: Run linting and type checking**

```bash
python -m ruff check src/evaluation/
python -m ruff check scripts/visualization/
python -m mypy src/evaluation/ --ignore-missing-imports
```

**Expected:** No critical errors (warnings acceptable for prototype)

- [ ] **Step 10.4: Final commit**

```bash
git add pyproject.toml
git commit -m "chore: add evaluation dependencies and finalize UMAP module

Add umap-learn, plotly, matplotlib, seaborn to optional dependencies.
Module ready for integration testing with actual model.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

**Expected:** Both commands print success messages

- [ ] **Step 8.2: Update pyproject.toml with evaluation dependencies**

```toml
# pyproject.toml - add to [project.optional-dependencies]

evaluation = [
    "umap-learn>=0.5.0",
    "plotly>=5.14.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]
```

- [ ] **Step 8.3: Run linting and type checking**

```bash
python -m ruff check src/evaluation/
python -m ruff check scripts/visualization/
python -m mypy src/evaluation/ --ignore-missing-imports
```

**Expected:** No critical errors (warnings acceptable for prototype)

- [ ] **Step 8.4: Final commit**

```bash
git add pyproject.toml
git commit -m "chore: add evaluation dependencies and finalize UMAP module

Add umap-learn, plotly, matplotlib, seaborn to optional dependencies.
Module ready for integration testing with actual model.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Success Criteria Verification

- [ ] Module generates all 6 static PNG figures without errors
- [ ] Module generates 2 interactive HTML files with working hover tooltips
- [ ] Crisis comparison shows both fixed and separate projection modes
- [ ] Clustering metrics CSV contains all 4 metrics for both sector and liquidity
- [ ] Hybrid caching works: standard periods cached, custom periods computed on-demand
- [ ] CLI interface accepts all documented arguments
- [ ] Unit tests achieve >80% code coverage
- [ ] Static figures are publication-ready (300 DPI, proper sizing)

---

## Notes for Implementation

1. **compute_embeddings placeholder**: The current implementation has a placeholder for actual embedding computation. This requires integration with the existing `FeatureLoader` and `DualEncoderModule.forward()` method. Update this once the model interface is confirmed.

2. **FeatureLoader interface**: The plan assumes `FeatureLoader.load_period()` exists. If it doesn't, create a simple wrapper around the existing data loading code.

3. **Model checkpoint format**: Ensure the checkpoint format matches what `DualEncoderModule.load_from_checkpoint()` expects.

4. **Testing with real data**: Integration tests require actual model checkpoints and feature data. These should be run manually after the basic unit tests pass.

---

**Plan Version:** 1.0  
**Created:** March 31, 2026  
**Based on Spec:** docs/superpowers/specs/2026-03-30-umap-visualization-design.md
