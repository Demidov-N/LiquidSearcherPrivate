# Block 2: SHAP Feature Importance - Implementation Plan

**Status**: Ready for Implementation  
**Priority**: Phase 2 (Interpretability)  
**Estimated Effort**: 2-3 days

---

## Overview

This plan implements SHAP (SHapley Additive exPlanations) analysis to decompose the dual-encoder's similarity predictions into feature contributions. The analysis answers: **"Which input features drive the model's similarity judgments?"**

---

## Architecture-Specific SHAP Methodology

### Dual-Encoder Challenge

The LiquidSearcher model has a unique architecture:
```
Temporal Input (60, 13) ──→ Temporal Encoder ──→ 128-dim ──┐
                                                            ├─→ Concat → 256-dim → Similarity
Tabular Input (15+2) ─────→ Tabular Encoder ────→ 128-dim ──┘
```

**Key Decision**: SHAP is computed on **tabular features only** because:
1. Temporal features are 60-day sequences (780 values per stock) - too high-dimensional for interpretable SHAP
2. Tabular features (17 values: 15 continuous + 2 categorical) are directly interpretable
3. Liquidity features (bid-ask spread, volume, market cap, Amihud) are in the tabular input
4. SHAP on tabular features answers: "Given two stocks with similar price histories, what fundamental differences make them more/less similar?"

### SHAP Value Computation

For a query stock $q$ and candidate stock $c$:
1. Compute joint embeddings: $e_q, e_c \in \mathbb{R}^{256}$
2. Compute similarity: $s = \cos(e_q, e_c)$
3. SHAP decomposes $s$ into: $s = \phi_0 + \sum_{i=1}^{17} \phi_i$

Where $\phi_i$ is the contribution of feature $i$ to the similarity score.

### Explainer Choice

**GradientExplainer** (recommended):
- Uses expected gradients, faster than DeepExplainer
- Works well with pre-trained models
- No need for background dataset selection bias

**DeepExplainer** (fallback):
- More accurate but slower
- Requires careful background dataset selection

---

## Module Structure

```
src/evaluation/
  shap/
    __init__.py
    explainer.py          # SHAP explainer wrapper
    analyzer.py           # Core analysis logic
    aggregator.py         # SHAP value aggregation
    visualizer.py         # SHAP plotting utilities
  metrics/
    clustering.py         # (existing)
  visualizations/
    umap_visualizer.py    # (existing)
    shap_plots.py         # High-level plot orchestration
  utils/
    feature_loader.py     # (existing)
    embedding_cache.py    # (existing)
  stock_similarity.py     # (existing)
```

---

## Key Classes and Functions

### 1. `src/evaluation/shap/explainer.py`

```python
"""SHAP explainer for dual-encoder similarity model."""

import torch
import shap
from pathlib import Path
from typing import Callable, Optional
import numpy as np


class DualEncoderExplainer:
    """SHAP explainer for dual-encoder stock similarity.
    
    Explains how tabular features affect similarity between
    a query stock and candidate stocks.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        checkpoint_path: str | Path,
        background_samples: int = 100,
        device: str = "cpu",
    ):
        """Initialize explainer.
        
        Args:
            model: DualEncoder model instance
            checkpoint_path: Path to model checkpoint
            background_samples: Number of samples for background dataset
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.background_samples = background_samples
        
    def create_explainer(
        self,
        background_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> shap.GradientExplainer:
        """Create SHAP explainer with background dataset.
        
        Args:
            background_data: Tuple of (temporal, tabular_cont, tabular_cat)
                           tensors for background dataset
            
        Returns:
            Configured SHAP explainer
        """
        # ... implementation
        
    def explain_similarity(
        self,
        query_idx: int,
        candidate_indices: list[int],
        data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> np.ndarray:
        """Compute SHAP values for similarity scores.
        
        Args:
            query_idx: Index of query stock
            candidate_indices: Indices of candidate stocks
            data: Full dataset tensors (temporal, tabular_cont, tabular_cat)
            
        Returns:
            SHAP values array of shape (n_candidates, n_features)
        """
        # ... implementation
```

### 2. `src/evaluation/shap/analyzer.py`

```python
"""Core SHAP analysis logic."""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from src.evaluation.shap.explainer import DualEncoderExplainer


@dataclass
class SHAPResult:
    """Container for SHAP analysis results."""
    shap_values: np.ndarray  # (n_samples, n_features)
    feature_names: list[str]
    sample_ids: list[str]  # "SYMBOL-DATE" format
    query_id: str
    base_values: np.ndarray  # Expected similarity


class SHAPAnalyzer:
    """Orchestrates SHAP analysis for dual-encoder model.
    
    Handles:
    - Data loading and batching (memory efficiency)
    - SHAP computation with progress tracking
    - Result caching and serialization
    """
    
    def __init__(
        self,
        feature_path: str | Path,
        checkpoint_path: str | Path,
        output_dir: str | Path,
        batch_size: int = 256,
    ):
        """Initialize analyzer.
        
        Args:
            feature_path: Path to all_features.parquet
            checkpoint_path: Path to model checkpoint
            output_dir: Directory for results
            batch_size: Batch size for SHAP computation
        """
        self.feature_path = Path(feature_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data_for_period(
        self,
        period_start: str,
        period_end: str,
        symbols: Optional[list[str]] = None,
        aggregation: str = "end_period",
    ) -> tuple[pd.DataFrame, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Load and prepare data for SHAP analysis.
        
        Args:
            period_start: Start date (YYYY-MM-DD)
            period_end: End date (YYYY-MM-DD)
            symbols: Optional symbol filter
            aggregation: "end_period" or "mean"
            
        Returns:
            Tuple of (metadata_df, (temporal, tabular_cont, tabular_cat))
        """
        # ... implementation
        
    def analyze_query(
        self,
        query_ticker: str,
        period_start: str,
        period_end: str,
        n_candidates: int = 100,
        aggregation: str = "end_period",
    ) -> SHAPResult:
        """Compute SHAP values for one query stock.
        
        Args:
            query_ticker: Query stock symbol
            period_start: Start date
            period_end: End date
            n_candidates: Number of candidates to analyze
            aggregation: Feature aggregation method
            
        Returns:
            SHAPResult with values and metadata
        """
        # ... implementation
        
    def save_result(self, result: SHAPResult, prefix: str) -> Path:
        """Save SHAP result to parquet.
        
        Args:
            result: SHAPResult to save
            prefix: Filename prefix
            
        Returns:
            Path to saved file
        """
        # ... implementation
```

### 3. `src/evaluation/shap/aggregator.py`

```python
"""SHAP value aggregation for global importance."""

import numpy as np
import pandas as pd
from typing import Literal
from dataclasses import dataclass


@dataclass
class GlobalImportance:
    """Aggregated feature importance rankings."""
    mean_abs_shap: pd.Series  # Mean |SHAP| per feature
    std_shap: pd.Series       # Std dev of SHAP values
    feature_rankings: pd.DataFrame  # Full ranking table


class SHAPAggregator:
    """Aggregate SHAP values across multiple queries.
    
    Supports:
    - Mean absolute SHAP (standard importance metric)
    - Feature ranking stability across queries
    - Sector-stratified importance
    """
    
    def __init__(self, feature_names: list[str]):
        """Initialize aggregator.
        
        Args:
            feature_names: List of feature names (17 tabular features)
        """
        self.feature_names = feature_names
        
    def aggregate_across_queries(
        self,
        results: list[SHAPResult],
    ) -> GlobalImportance:
        """Aggregate SHAP values across multiple query analyses.
        
        Args:
            results: List of SHAPResult from different queries
            
        Returns:
            GlobalImportance with aggregated metrics
        """
        all_shap = []
        for result in results:
            # Mean absolute SHAP per feature
            mean_abs = np.abs(result.shap_values).mean(axis=0)
            all_shap.append(mean_abs)
        
        mean_shap = np.mean(all_shap, axis=0)
        std_shap = np.std(all_shap, axis=0)
        
        rankings = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_shap,
            'std_shap': std_shap,
            'rank': np.argsort(-mean_shap) + 1,
        })
        
        return GlobalImportance(
            mean_abs_shap=pd.Series(mean_shap, index=self.feature_names),
            std_shap=pd.Series(std_shap, index=self.feature_names),
            feature_rankings=rankings,
        )
        
    def compare_to_expected_hierarchy(
        self,
        importance: GlobalImportance,
    ) -> dict:
        """Compare observed importance to expected liquidity hierarchy.
        
        Expected hierarchy (if model works correctly):
        1. Bid-ask spread percentage
        2. Average daily volume
        3. Market capitalization
        4. Amihud illiquidity ratio
        5. Turnover rate
        
        Args:
            importance: GlobalImportance from aggregation
            
        Returns:
            Dictionary with comparison metrics
        """
        expected_liquidity_features = [
            'market_cap', 'dividend_yield',  # Proxies in current features
        ]
        
        observed_top5 = importance.feature_rankings.head(5)['feature'].tolist()
        
        return {
            'observed_top5': observed_top5,
            'expected_liquidity_features': expected_liquidity_features,
            'liquidity_in_top5': sum(1 for f in expected_liquidity_features if f in observed_top5),
        }
```

### 4. `src/evaluation/shap/visualizer.py`

```python
"""SHAP visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional


class SHAPVisualizer:
    """Generate SHAP visualization plots.
    
    Produces:
    - Summary plots (beeswarm)
    - Waterfall plots (individual predictions)
    - Bar plots (global importance)
    - Heatmaps (feature contributions)
    """
    
    def __init__(self, output_dir: str | Path, style: str = "seaborn-v0_8"):
        """Initialize visualizer.
        
        Args:
            output_dir: Directory for saved figures
            style: Matplotlib style name
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use(style)
        
    def plot_summary_beeswarm(
        self,
        shap_values: np.ndarray,
        features: np.ndarray,
        feature_names: list[str],
        filename: str = "shap_beeswarm.png",
        max_display: int = 15,
    ) -> Path:
        """Create beeswarm summary plot.
        
        Shows distribution of SHAP values for each feature,
        colored by feature value (red=high, blue=low).
        
        Args:
            shap_values: SHAP values (n_samples, n_features)
            features: Original feature values
            feature_names: Feature names
            filename: Output filename
            max_display: Max features to show
            
        Returns:
            Path to saved figure
        """
        # ... implementation
        
    def plot_waterfall(
        self,
        shap_values: np.ndarray,
        feature_values: np.ndarray,
        feature_names: list[str],
        base_value: float,
        query_ticker: str,
        candidate_ticker: str,
        filename: Optional[str] = None,
    ) -> Path:
        """Create waterfall plot for individual prediction.
        
        Shows how features push similarity up/down from base value.
        
        Args:
            shap_values: SHAP values for one sample (n_features,)
            feature_values: Feature values for that sample
            feature_names: Feature names
            base_value: Expected similarity (phi_0)
            query_ticker: Query stock symbol
            candidate_ticker: Candidate stock symbol
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved figure
        """
        # ... implementation
        
    def plot_global_importance(
        self,
        mean_abs_shap: np.ndarray,
        feature_names: list[str],
        filename: str = "shap_global_importance.png",
        top_k: int = 15,
    ) -> Path:
        """Create bar plot of global feature importance.
        
        Args:
            mean_abs_shap: Mean absolute SHAP per feature
            feature_names: Feature names
            filename: Output filename
            top_k: Number of top features to show
            
        Returns:
            Path to saved figure
        """
        # ... implementation
        
    def plot_feature_contributions(
        self,
        shap_values: np.ndarray,
        feature_names: list[str],
        sample_ids: list[str],
        filename: str = "shap_heatmap.png",
        max_samples: int = 50,
    ) -> Path:
        """Create heatmap of feature contributions.
        
        Args:
            shap_values: SHAP values (n_samples, n_features)
            feature_names: Feature names
            sample_ids: Sample identifiers
            filename: Output filename
            max_samples: Max samples to show in heatmap
            
        Returns:
            Path to saved figure
        """
        # ... implementation
```

### 5. `src/evaluation/visualizations/shap_plots.py`

```python
"""High-level SHAP plot orchestration."""

from pathlib import Path
from typing import Optional


def generate_all_shap_plots(
    results_dir: str | Path,
    feature_path: str | Path,
    checkpoint_path: str | Path,
    query_tickers: list[str],
    period_start: str,
    period_end: str,
) -> dict:
    """Generate complete set of SHAP visualizations.
    
    Args:
        results_dir: Directory for outputs
        feature_path: Path to all_features.parquet
        checkpoint_path: Path to model checkpoint
        query_tickers: List of query stocks to analyze
        period_start: Start date
        period_end: End date
        
    Returns:
        Dictionary with paths to generated figures
    """
    # ... implementation
```

---

## CLI Script

### `scripts/shap_analysis.py`

```python
#!/usr/bin/env python
"""SHAP feature importance analysis for LiquidSearcher.

Examples:

    # Analyze single query stock
    python -m scripts.shap_analysis \\
        --query-ticker AAPL \\
        --period-start 2019-01-01 \\
        --period-end 2020-01-31

    # Analyze multiple queries and aggregate
    python -m scripts.shap_analysis \\
        --query-tickers AAPL MSFT JNJ XOM \\
        --period-start 2019-01-01 \\
        --period-end 2020-01-31 \\
        --aggregate

    # Custom candidate pool size
    python -m scripts.shap_analysis \\
        --query-ticker AAPL \\
        --n-candidates 500 \\
        --batch-size 128
"""

import argparse
from pathlib import Path

from src.evaluation.shap.analyzer import SHAPAnalyzer
from src.evaluation.shap.aggregator import SHAPAggregator
from src.evaluation.shap.visualizer import SHAPVisualizer
from src.evaluation.visualizations.shap_plots import generate_all_shap_plots


def main():
    parser = argparse.ArgumentParser(
        description="SHAP feature importance analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--query-ticker",
        type=str,
        help="Single query stock ticker",
    )
    parser.add_argument(
        "--query-tickers",
        type=str,
        nargs="+",
        help="Multiple query tickers (for aggregation)",
    )
    parser.add_argument(
        "--feature-path",
        type=str,
        default="data/processed/all_features.parquet",
        help="Path to features parquet",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/last.ckpt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/shap_analysis",
        help="Output directory",
    )
    parser.add_argument(
        "--period-start",
        type=str,
        default="2019-01-01",
        help="Analysis period start",
    )
    parser.add_argument(
        "--period-end",
        type=str,
        default="2020-01-31",
        help="Analysis period end",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=100,
        help="Number of candidate stocks per query",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for SHAP computation",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate results across multiple queries",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.query_ticker and args.query_tickers:
        parser.error("Cannot specify both --query-ticker and --query-tickers")
    if not args.query_ticker and not args.query_tickers:
        parser.error("Must specify either --query-ticker or --query-tickers")
    
    # Single query analysis
    if args.query_ticker:
        analyzer = SHAPAnalyzer(
            feature_path=args.feature_path,
            checkpoint_path=args.checkpoint_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
        
        result = analyzer.analyze_query(
            query_ticker=args.query_ticker,
            period_start=args.period_start,
            period_end=args.period_end,
            n_candidates=args.n_candidates,
        )
        
        visualizer = SHAPVisualizer(output_dir=args.output_dir)
        # Generate plots...
        
    # Multiple query analysis with aggregation
    elif args.query_tickers:
        all_results = []
        
        for ticker in args.query_tickers:
            analyzer = SHAPAnalyzer(
                feature_path=args.feature_path,
                checkpoint_path=args.checkpoint_path,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
            )
            
            result = analyzer.analyze_query(
                query_ticker=ticker,
                period_start=args.period_start,
                period_end=args.period_end,
                n_candidates=args.n_candidates,
            )
            all_results.append(result)
        
        if args.aggregate:
            aggregator = SHAPAggregator(
                feature_names=TABULAR_CONTINUOUS_NAMES  # 15 features
            )
            importance = aggregator.aggregate_across_queries(all_results)
            
            # Save rankings
            importance.feature_rankings.to_csv(
                Path(args.output_dir) / "feature_importance_rankings.csv",
                index=False,
            )
            
            # Generate global plots
            visualizer = SHAPVisualizer(output_dir=args.output_dir)
            visualizer.plot_global_importance(
                mean_abs_shap=importance.mean_abs_shap.values,
                feature_names=importance.mean_abs_shap.index.tolist(),
            )


if __name__ == "__main__":
    main()
```

---

## Memory-Efficient Computation Strategy

### Problem
- 12M rows cannot fit in memory
- SHAP requires multiple forward passes per sample
- GradientExplainer needs background dataset

### Solution: Streaming + Sampling

```python
def compute_shap_memory_efficient(
    analyzer: SHAPAnalyzer,
    query_idx: int,
    n_candidates: int = 100,
    batch_size: int = 256,
) -> np.ndarray:
    """Compute SHAP values in batches to avoid OOM.
    
    Strategy:
    1. Select top-N candidates by similarity (cheap cosine sim)
    2. Process candidates in batches through SHAP
    3. Clear GPU/CPU cache between batches
    """
    # Step 1: Compute all embeddings (streaming if needed)
    # Step 2: Find top-N similar candidates
    # Step 3: Batch SHAP computation on subset
    # Step 4: Aggregate and return
```

### Batching Strategy

```
┌─────────────────────────────────────────┐
│  Background Dataset (100 samples)       │  ← Loaded once
├─────────────────────────────────────────┤
│  Query Stock (1 sample)                 │  ← Fixed
├─────────────────────────────────────────┤
│  Candidates (100 samples)               │  ← Processed in batches of 25
│    Batch 1: Candidates 0-24             │
│    Batch 2: Candidates 25-49            │
│    Batch 3: Candidates 50-74            │
│    Batch 4: Candidates 75-99            │
└─────────────────────────────────────────┘
```

---

## Output Formats

### 1. CSV Files

**`results/shap_analysis/shap_values_{QUERY}_{PERIOD}.csv`**
```csv
query_ticker,candidate_ticker,date,feature_1,feature_2,...,feature_17,base_value,similarity_score
AAPL,MSFT,2020-01-31,0.023,-0.015,...,0.008,0.15,0.87
AAPL,GOOGL,2020-01-31,0.019,-0.022,...,0.011,0.15,0.82
...
```

**`results/shap_analysis/feature_importance_rankings.csv`**
```csv
feature,mean_abs_shap,std_shap,rank
market_cap,0.045,0.012,1
dividend_yield,0.038,0.015,2
beta,0.032,0.018,3
...
```

### 2. Figures

| Filename | Type | Description |
|----------|------|-------------|
| `shap_beeswarm.png` | Beeswarm | Global feature importance distribution |
| `shap_global_importance.png` | Bar chart | Top 15 features by mean |SHAP| |
| `shap_waterfall_{QUERY}_{CANDIDATE}.png` | Waterfall | Individual prediction breakdown (5-10 case studies) |
| `shap_heatmap.png` | Heatmap | Feature contributions across samples |

### 3. Text Report

**`results/shap_analysis/summary.txt`**
```
SHAP Feature Importance Analysis
================================
Period: 2019-01-01 to 2020-01-31
Queries analyzed: 10
Total candidates: 1000

TOP 10 MOST IMPORTANT FEATURES:
1. market_cap        (mean |SHAP|: 0.045)
2. dividend_yield    (mean |SHAP|: 0.038)
3. beta              (mean |SHAP|: 0.032)
...

EXPECTED LIQUIDITY HIERARCHY CHECK:
- Liquidity features in top 5: 2/5
- market_cap ranked: 1
- dividend_yield ranked: 2

INTERPRETATION:
Model relies heavily on size and yield features, consistent with
liquidity-focused learning objective. Beta's high importance suggests
the model also captures risk-factor similarity.
```

---

## Integration with Existing Framework

### Reusing Existing Components

1. **FeatureLoader** (`src/evaluation/utils/feature_loader.py`):
   - Load data for specified period
   - Filter by symbols

2. **EmbeddingCache** (`src/evaluation/utils/embedding_cache.py`):
   - Cache computed embeddings to avoid recomputation
   - Use model hash as cache key

3. **stock_similarity.py**:
   - Reuse `_prepare_features()` function
   - Reuse model loading logic

### Exported API

```python
# src/evaluation/__init__.py
from src.evaluation.shap import (
    SHAPAnalyzer,
    SHAPAggregator,
    SHAPVisualizer,
    DualEncoderExplainer,
)

__all__ = [
    # Existing
    "UMAPVisualizer",
    "FeatureLoader",
    "EmbeddingCache",
    # New
    "SHAPAnalyzer",
    "SHAPAggregator",
    "SHAPVisualizer",
    "DualEncoderExplainer",
]
```

---

## Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Create `src/evaluation/shap/__init__.py`
- [ ] Implement `DualEncoderExplainer` class
- [ ] Implement `SHAPAnalyzer` with memory-efficient batching
- [ ] Test on single query stock

### Phase 2: Aggregation
- [ ] Implement `SHAPAggregator` class
- [ ] Implement hierarchy comparison logic
- [ ] Test aggregation across 5-10 queries

### Phase 3: Visualization
- [ ] Implement `SHAPVisualizer` with all plot types
- [ ] Create `shap_plots.py` orchestration module
- [ ] Generate all figure types

### Phase 4: CLI & Integration
- [ ] Create `scripts/shap_analysis.py` CLI
- [ ] Update `src/evaluation/__init__.py` exports
- [ ] Add to evaluation plan checklist

### Phase 5: Validation
- [ ] Run on full dataset (10 queries, 100 candidates each)
- [ ] Verify expected liquidity feature hierarchy
- [ ] Generate summary report
- [ ] Document findings

---

## Feature Names

The 15 tabular continuous features (from `data_module.py`):

```python
TABULAR_CONTINUOUS_NAMES = [
    "beta",              # 0
    "idiosyncratic_vol", # 1
    "roe",               # 2
    "roa",               # 3
    "debt_to_equity",    # 4
    "price_to_book",     # 5
    "price_to_earnings", # 6
    "market_cap",        # 7  ← Expected important (liquidity proxy)
    "dividend_yield",    # 8  ← Expected important (liquidity proxy)
    "revenue",           # 9
    "net_income",        # 10
    "total_assets",      # 11
    "cash",              # 12
    "operating_margin",  # 13
    "profit_margin",     # 14
]
```

Note: The evaluation plan mentions "bid-ask spread, avg daily volume, Amihud ratio, turnover" as expected important features. These are **not** in the current tabular features—they may be in the temporal features or need to be added. The current liquidity proxies are `market_cap` and `dividend_yield`.

---

## Testing Strategy

### Unit Tests

```python
# tests/evaluation/shap/test_explainer.py
def test_explainer_initialization():
    """Test explainer loads model correctly."""
    
def test_explain_similarity_shape():
    """Test SHAP output has correct shape."""
    
def test_waterfall_plot_generation():
    """Test waterfall plot is created."""
```

### Integration Test

```bash
# Test full pipeline on small subset
python -m scripts.shap_analysis \
    --query-ticker AAPL \
    --period-start 2019-01-01 \
    --period-end 2019-01-31 \
    --n-candidates 10 \
    --output-dir /tmp/shap_test
```

---

## Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
evaluation = [
    "umap-learn>=0.5.0",
    "shap>=0.44.0",           # SHAP analysis
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]
```

---

## Expected Runtime

| Operation | Time (estimate) |
|-----------|-----------------|
| Load data (1 period) | 30-60 seconds |
| Compute embeddings (1000 stocks) | 1-2 minutes |
| SHAP per query (100 candidates) | 5-10 minutes |
| Full analysis (10 queries) | 1-2 hours |
| Visualization generation | 1-2 minutes |

**Total for full evaluation**: ~2 hours

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| OOM with large datasets | Streaming batches, reduce background samples |
| SHAP too slow | Use GradientExplainer (faster), reduce candidates |
| Features not in tabular input | Document limitation, note in evaluation plan |
| SHAP values unstable | Increase background samples, average multiple runs |

---

## Success Criteria

1. **Technical**: SHAP values computed for 10 queries × 100 candidates each
2. **Interpretability**: Top 5 features identified and ranked
3. **Validation**: Liquidity features (`market_cap`, `dividend_yield`) appear in top 5
4. **Visualization**: All 4 plot types generated and publication-ready
5. **Documentation**: Summary report with financial interpretation

---

**Last Updated**: March 31, 2026  
**Author**: Planning Agent  
**Review Status**: Ready for Implementation
