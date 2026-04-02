# Block 2: SHAP Feature Importance - Implementation Plan

**Status**: Ready for Implementation  
**Priority**: HIGH (Phase 2 - Interpretability)  
**Estimated Effort**: 2-3 days

---

## Overview

This plan implements SHAP (SHapley Additive exPlanations) analysis to decompose LiquidSearcher's similarity predictions into feature contributions. The dual-encoder architecture requires a specialized approach since similarity is computed in the 256-dim joint embedding space, not directly from input features.

---

## Key Design Decisions

### 1. SHAP Methodology for Dual-Encoder Architecture

**Challenge**: Standard SHAP explains model outputs given inputs. Our model outputs embeddings, and similarity is computed post-hoc via cosine similarity.

**Solution**: We explain the **similarity score** as the model output:
- **Input**: Feature vector (temporal + tabular features for a stock)
- **Output**: Cosine similarity between query stock embedding and candidate stock embedding
- **Approach**: Treat one encoder pathway as the "background" and compute how features shift embeddings

### 2. Which Features to Analyze?

**Decision**: Analyze **tabular features only** for SHAP, with temporal features as context.

**Rationale**:
1. Tabular features (15 continuous + 2 categorical) are interpretable: market cap, beta, ROE, etc.
2. Temporal features (13 features × 60 timesteps = 780 values) are too high-dimensional for SHAP
3. Expected important features (bid-ask spread, volume, market cap, Amihud) are tabular
4. Tabular encoder is a separate pathway (TabMixer) - cleaner attribution

**Implementation**: 
- Compute SHAP values for tabular continuous features
- Treat temporal embedding as a "context vector" that conditions the similarity
- For each query-candidate pair, explain: "Given the query's temporal pattern, which tabular features of the candidate make it similar?"

### 3. Aggregation Strategy

**Per-query analysis**:
- For each query stock, compute SHAP values for top-50 retrieved candidates
- SHAP values sum to the similarity score (additivity property)

**Global aggregation**:
- Average absolute SHAP values across all query-candidate pairs
- Rank features by mean |SHAP value|
- Compute statistical significance via bootstrap confidence intervals

### 4. Memory-Efficient Computation

**Strategy**: Process in mini-batches, cache intermediate results

```
Total candidates: ~6000 stocks
Queries to analyze: 50 representative stocks
SHAP background samples: 100 (subset of training data)
Memory per SHAP computation: O(background_size × embedding_dim)
```

**Approach**:
1. Pre-compute embeddings for all stocks (cache to disk)
2. For each query:
   - Load query embedding
   - Compute similarities to all candidates
   - Select top-50 candidates
   - Run SHAP on query + top-50 only
3. Aggregate results across queries

---

## Module Structure

```
src/evaluation/
  feature_importance/
    __init__.py
    shap_analyzer.py          # Core SHAP analysis engine
    feature_groups.py         # Feature groupings and aggregations
    sampling.py               # Query and background set sampling
    
  visualizations/
    shap_plots.py             # SHAP visualization functions
    
  scripts/
    run_shap_analysis.py      # CLI script for SHAP analysis
    
results/
  shap/
    global_importance.csv     # Aggregated feature importance
    per_query/                # Per-query SHAP values
      SHAP_AAPL_2019.parquet
      SHAP_MSFT_2019.parquet
      ...
    figures/
      shap_summary.png        # Global summary plot
      shap_waterfall_*.png    # Case study waterfalls
      shap_beeswarm.png       # Feature impact distribution
```

---

## Key Classes and Functions

### `src/evaluation/feature_importance/shap_analyzer.py`

```python
class DualEncoderExplainer:
    """
    SHAP explainer for dual-encoder similarity model.
    
    Explains similarity scores as:
    similarity(q, c) = base_value + Σ φ_i
    
    where φ_i is the SHAP value for feature i.
    """
    
    def __init__(
        self,
        model: DualEncoder,
        background_data: pd.DataFrame,  # Background samples for SHAP
        background_size: int = 100,
        device: str = "cpu",
    ):
        """
        Args:
            model: Trained DualEncoder model
            background_data: DataFrame with tabular features for background samples
            background_size: Number of background samples (larger = more accurate, slower)
            device: Device for model computation
        """
        
    def explain_similarity(
        self,
        query_row: pd.Series,
        candidate_rows: pd.DataFrame,
        n_candidates: int = 50,
    ) -> pd.DataFrame:
        """
        Compute SHAP values for similarity between query and candidates.
        
        Args:
            query_row: Query stock features
            candidate_rows: Candidate stock features
            n_candidates: Number of top candidates to explain
        
        Returns:
            DataFrame with columns:
            - candidate_ticker
            - similarity_score
            - shap_<feature_name> for each feature
            - residual (should be ~0)
        """
        
    def explain_batch(
        self,
        queries_df: pd.DataFrame,
        all_stocks_df: pd.DataFrame,
        n_candidates_per_query: int = 50,
        output_dir: Path,
    ) -> dict:
        """
        Run SHAP analysis for multiple queries.
        
        Args:
            queries_df: Query stocks to analyze
            all_stocks_df: All candidate stocks
            n_candidates_per_query: How many neighbors to explain
            output_dir: Directory to save per-query results
        
        Returns:
            Dictionary with aggregated statistics
        """


def _create_similarity_wrapper(
    model: DualEncoder,
    query_temporal: torch.Tensor,
    query_tabular: torch.Tensor,
    query_categorical: torch.Tensor,
) -> Callable:
    """
    Create a function that maps candidate tabular features → similarity score.
    
    This wrapper is passed to shap.DeepExplainer.
    
    Args:
        model: DualEncoder model
        query_*: Pre-computed query inputs
    
    Returns:
        Function: candidate_tabular_features → similarity scores
    """
```

### `src/evaluation/feature_importance/sampling.py`

```python
class QuerySampler:
    """Select representative query stocks for SHAP analysis."""
    
    @staticmethod
    def stratified_by_sector_and_liquidity(
        df: pd.DataFrame,
        n_queries: int = 50,
    ) -> pd.DataFrame:
        """
        Sample queries stratified by sector and liquidity tier.
        
        Strategy:
        - 20 large-cap liquid stocks (top 2 by market cap per sector)
        - 20 mid-cap mixed liquidity (middle 2 per sector)
        - 10 small-cap illiquid (bottom 1 per sector)
        
        Args:
            df: DataFrame with sector and market_cap columns
            n_queries: Total number of queries
        
        Returns:
            DataFrame of query stocks
        """


def select_background_samples(
    df: pd.DataFrame,
    n_samples: int = 100,
    method: Literal["random", "kmeans"] = "kmeans",
) -> pd.DataFrame:
    """
    Select background samples for SHAP baseline.
    
    Args:
        df: Full dataset
        n_samples: Number of background samples
        method: 'random' for random sampling, 'kmeans' for k-means centroids
    
    Returns:
        DataFrame with background samples
    
    Notes:
        - K-means method selects samples closest to cluster centroids
        - This provides better coverage of feature space than random
    """
```

### `src/evaluation/feature_importance/feature_groups.py`

```python
# Feature groupings for aggregated importance
LIQUIDITY_FEATURES = [
    "market_cap",
    "beta",
    "idiosyncratic_vol",
    # Note: bid-ask spread, Amihud, turnover should be added if available
]

FUNDAMENTAL_FEATURES = [
    "roe",
    "roa",
    "debt_to_equity",
    "price_to_book",
    "price_to_earnings",
    "operating_margin",
    "profit_margin",
]

SIZE_FEATURES = ["market_cap"]

VOLATILITY_FEATURES = [
    "beta",
    "idiosyncratic_vol",
]


def aggregate_shap_by_group(
    shap_df: pd.DataFrame,
    feature_groups: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Aggregate SHAP values by feature group.
    
    Args:
        shap_df: Per-feature SHAP values
        feature_groups: Mapping of group name → feature names
    
    Returns:
        DataFrame with aggregated importance per group
    """
```

### `src/evaluation/visualizations/shap_plots.py`

```python
def plot_global_summary(
    shap_values: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    max_features: int = 20,
) -> Path:
    """
    Create global SHAP summary plot (beeswarm-style).
    
    Shows feature importance ranking and impact direction.
    
    Args:
        shap_values: (n_samples, n_features) array of SHAP values
        feature_names: Feature names
        output_path: Where to save plot
        max_features: Top N features to show
    
    Returns:
        Path to saved figure
    """


def plot_waterfall(
    shap_values: np.ndarray,
    feature_names: list[str],
    base_value: float,
    similarity_score: float,
    candidate_ticker: str,
    output_path: Path,
) -> Path:
    """
    Create waterfall plot for single prediction.
    
    Shows how features add up from base value to final similarity.
    
    Args:
        shap_values: (n_features,) array for one candidate
        feature_names: Feature names
        base_value: Expected similarity (average over background)
        similarity_score: Actual predicted similarity
        candidate_ticker: Candidate stock ticker
        output_path: Where to save plot
    
    Returns:
        Path to saved figure
    """


def plot_feature_distribution(
    shap_values: np.ndarray,
    feature_names: list[str],
    feature_values: np.ndarray,
    output_path: Path,
) -> Path:
    """
    Create distribution plot showing SHAP value vs feature value.
    
    For each feature, shows whether high/low values increase similarity.
    
    Args:
        shap_values: (n_samples, n_features) SHAP values
        feature_names: Feature names
        feature_values: (n_samples, n_features) original feature values
        output_path: Where to save plot
    
    Returns:
        Path to saved figure
    """
```

### `scripts/run_shap_analysis.py` (CLI Script)

```python
#!/usr/bin/env python
"""
Run SHAP feature importance analysis for LiquidSearcher.

Example:
    python -m scripts.run_shap_analysis \\
        --checkpoint checkpoints/last.ckpt \\
        --features data/processed/all_features.parquet \\
        --period-start 2019-01-01 \\
        --period-end 2019-12-31 \\
        --n-queries 50 \\
        --output-dir results/shap
"""

import argparse
from pathlib import Path

from src.evaluation.feature_importance.shap_analyzer import DualEncoderExplainer
from src.evaluation.feature_importance.sampling import (
    QuerySampler,
    select_background_samples,
)
from src.evaluation.visualizations.shap_plots import (
    plot_global_summary,
    plot_waterfall,
)
from src.evaluation.utils.feature_loader import FeatureLoader
from src.evaluation.utils.embedding_cache import EmbeddingCache
from src.models.dual_encoder import DualEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/last.ckpt")
    parser.add_argument("--features", type=str, default="data/processed/all_features.parquet")
    parser.add_argument("--period-start", type=str, default="2019-01-01")
    parser.add_argument("--period-end", type=str, default="2019-12-31")
    parser.add_argument("--n-queries", type=int, default=50)
    parser.add_argument("--n-candidates", type=int, default=50)
    parser.add_argument("--background-size", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="results/shap")
    args = parser.parse_args()
    
    # Load data
    loader = FeatureLoader(args.features)
    df = loader.load_period(args.period_start, args.period_end)
    
    # Aggregate to period-level (end-of-period snapshot)
    stocks_df = df.sort_values('date').groupby('symbol').last().reset_index()
    
    # Select queries and background
    queries = QuerySampler.stratified_by_sector_and_liquidity(
        stocks_df, n_queries=args.n_queries
    )
    background = select_background_samples(stocks_df, n_samples=args.background_size)
    
    # Load model
    model = load_model(args.checkpoint)
    
    # Initialize explainer
    explainer = DualEncoderExplainer(
        model=model,
        background_data=background,
        background_size=args.background_size,
    )
    
    # Run SHAP analysis
    results = explainer.explain_batch(
        queries_df=queries,
        all_stocks_df=stocks_df,
        n_candidates_per_query=args.n_candidates,
        output_dir=Path(args.output_dir) / "per_query",
    )
    
    # Save global importance
    results["global_importance"].to_csv(
        Path(args.output_dir) / "global_importance.csv"
    )
    
    # Generate visualizations
    plot_global_summary(
        shap_values=results["shap_matrix"],
        feature_names=results["feature_names"],
        output_path=Path(args.output_dir) / "figures" / "shap_summary.png",
    )
    
    # Save case study waterfalls
    for ticker in queries["symbol"].head(5):
        plot_waterfall(
            shap_values=results["waterfall_data"][ticker]["shap_values"],
            feature_names=results["feature_names"],
            base_value=results["waterfall_data"][ticker]["base_value"],
            similarity_score=results["waterfall_data"][ticker]["similarity"],
            candidate_ticker=results["waterfall_data"][ticker]["candidate"],
            output_path=Path(args.output_dir) / "figures" / f"shap_waterfall_{ticker}.png",
        )
    
    print(f"\nSHAP analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
```

---

## SHAP Computation Details

### Algorithm

For each query stock Q and candidate stock C:

1. **Pre-compute query embedding**:
   ```python
   query_emb = model.get_joint_embedding(
       query_temporal, query_tabular, query_categorical
   )  # (1, 256)
   ```

2. **Define similarity function for SHAP**:
   ```python
   def similarity_fn(candidate_tabular_features):
       # candidate_tabular_features: (batch, 15)
       candidate_emb = model.tabular_encoder(
           candidate_tabular_features,
           candidate_categorical  # fixed to candidate's actual sector
       )
       return F.cosine_similarity(query_emb, candidate_emb).cpu().numpy()
   ```

3. **Run DeepExplainer**:
   ```python
   explainer = shap.DeepExplainer(
       model=similarity_fn,
       data=background_tabular_features  # (100, 15)
   )
   
   shap_values = explainer.shap_values(
       candidate_tabular_features  # (50, 15) for top-50 candidates
   )
   ```

### Handling Categorical Features

GICS sector and group are **not explained** directly. Instead:
- They are fixed to the candidate's actual values during SHAP computation
- The tabular encoder's sector embedding becomes part of the "base model"
- This is valid because we're explaining **continuous feature variation**, not sector assignment

### Base Value Interpretation

The SHAP base value represents:
> "Expected similarity between this query and a random stock from the background distribution"

Typical values: 0.0 to 0.3 (near-zero for random pairs, higher for similar pairs)

---

## Output Formats

### `global_importance.csv`

```csv
feature,mean_abs_shap,std_shap,mean_shap,pct_importance
market_cap,0.0234,0.0156,0.0089,18.5
beta,0.0198,0.0134,-0.0045,15.7
idiosyncratic_vol,0.0167,0.0112,0.0123,13.2
roe,0.0145,0.0098,-0.0067,11.5
price_to_book,0.0123,0.0087,0.0034,9.7
...
```

### `per_query/SHAP_{TICKER}_{DATE}.parquet`

```python
# Columns:
{
    "query_ticker": str,
    "query_date": datetime,
    "candidate_ticker": str,
    "similarity_score": float,
    "rank": int,
    "shap_market_cap": float,
    "shap_beta": float,
    "shap_idiosyncratic_vol": float,
    "shap_roe": float,
    # ... one column per feature
    "base_value": float,
    "sum_shap_values": float,  # Should equal similarity - base_value
}
```

### Figures

1. **`shap_summary.png`**: Beeswarm plot showing feature importance ranking
   - X-axis: SHAP value (impact on similarity)
   - Y-axis: Features ranked by importance
   - Color: Feature value (red = high, blue = low)
   - Each point: One query-candidate pair

2. **`shap_waterfall_*.png`**: Waterfall for individual predictions
   - Shows how features add from base value to final similarity
   - Top 10 features by absolute SHAP value
   - Green bars: Increase similarity, Red bars: Decrease similarity

3. **`shap_beeswarm.png`**: Alternative view of feature impacts
   - Similar to summary but emphasizes distribution shape

---

## Integration with Existing Framework

### Reuse Existing Utilities

1. **FeatureLoader** (`src/evaluation/utils/feature_loader.py`):
   - Load data for specified period
   - Already handles date filtering and symbol selection

2. **EmbeddingCache** (`src/evaluation/utils/embedding_cache.py`):
   - Cache pre-computed embeddings to avoid recomputation
   - Standard periods (covid_pre, etc.) automatically cached

3. **Stock similarity analysis** (`src/evaluation/stock_similarity.py`):
   - Reuse `_prepare_features()` function for model input preparation
   - Reuse model loading logic

### Compatibility with Other Evaluation Blocks

SHAP analysis complements:
- **Block 1 (UMAP)**: SHAP explains what features drive the embedding positions
- **Block 3 (Manual)**: SHAP provides quantitative backing for manual inspections
- **Block 4 (Recall)**: SHAP shows which features matter for retrieval quality

---

## Expected Results

Based on the model's training objective (InfoNCE alignment of temporal and tabular views), we expect:

### High Importance (Top 5)
1. **market_cap** - Size is a dominant factor in stock behavior
2. **beta** - Market sensitivity drives co-movement
3. **idiosyncratic_vol** - Stock-specific volatility pattern
4. **price_to_book** - Value/growth dimension
5. **roe** - Profitability indicator

### Medium Importance
- debt_to_equity - Leverage
- operating_margin - Profitability
- momentum features (if included in tabular)

### Low Importance
- price_to_earnings - Noisy metric
- dividend_yield - Many stocks have zero dividends
- revenue - Less informative than margins

---

## Validation Checks

Before considering implementation complete:

1. **Additivity check**: For each prediction, verify:
   ```python
   base_value + sum(shap_values) ≈ similarity_score
   # Tolerance: |error| < 0.01
   ```

2. **Consistency check**: Similar stocks should have similar SHAP profiles:
   - Correlation between SHAP vectors of same-sector pairs > cross-sector pairs

3. **Face validity**: Top features should align with domain knowledge:
   - Liquidity features should dominate if model learned liquidity structure
   - Fundamental features should matter for value/growth similarity

4. **Stability check**: Bootstrap confidence intervals should be narrow:
   - Re-run SHAP on 80% subsample → feature ranking correlation > 0.8

---

## Implementation Checklist

### Phase 2.1: Core SHAP Engine
- [ ] Create `src/evaluation/feature_importance/__init__.py`
- [ ] Implement `DualEncoderExplainer` class
- [ ] Implement similarity wrapper function
- [ ] Test on single query-candidate pair

### Phase 2.2: Sampling and Batching
- [ ] Implement `QuerySampler` strategies
- [ ] Implement `select_background_samples()`
- [ ] Implement batch processing with progress tracking
- [ ] Add memory-efficient embedding caching

### Phase 2.3: Aggregation and Statistics
- [ ] Implement global importance aggregation
- [ ] Compute bootstrap confidence intervals
- [ ] Implement feature group aggregation
- [ ] Save per-query results to parquet

### Phase 2.4: Visualizations
- [ ] Implement `plot_global_summary()`
- [ ] Implement `plot_waterfall()`
- [ ] Implement `plot_feature_distribution()`
- [ ] Generate all figures for paper

### Phase 2.5: CLI and Integration
- [ ] Create `scripts/run_shap_analysis.py`
- [ ] Test end-to-end pipeline
- [ ] Document usage in README
- [ ] Run on full dataset (50 queries × 50 candidates)

---

## Potential Issues and Mitigations

| Issue | Mitigation |
|-------|-----------|
| SHAP computation too slow | Use GradientExplainer instead of DeepExplainer (faster, approximate) |
| Out of memory | Reduce background_size, process in smaller batches |
| SHAP values unstable | Increase background samples, use k-means sampling |
| Temporal features ignored | Create separate analysis for temporal patterns (future work) |
| Categorical features unexplained | Document limitation, focus on continuous features |

---

## Future Extensions

1. **Temporal SHAP**: Explain how temporal patterns (60-day windows) affect similarity
   - Would require aggregating SHAP values across timesteps
   - Could use integrated gradients instead

2. **Counterfactual Analysis**: "What if this stock had different features?"
   - Generate counterfactual candidates
   - Show how similarity changes

3. **Interactive Dashboard**: Web-based SHAP explorer
   - Select query stock → see SHAP breakdown for neighbors
   - Filter by sector, market cap, etc.

---

## References

- **SHAP Original Paper**: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (NeurIPS 2017)
- **DeepExplainer**: Lundberg et al., "From Local Explanations to Global Understanding with Explainable AI for Trees" (Nature MI 2020)
- **Dual-Encoder Interpretability**: Works on CLIP interpretability (similar architecture)

---

**Last Updated**: March 31, 2026
