# UMAP Visualization Module - Design Specification

**Date**: March 30, 2026  
**Version**: 1.0  
**Status**: Approved for Implementation

---

## Overview

This specification defines the PCA + UMAP visualization module for Block 1 of the LiquidSearcher evaluation framework. The module generates publication-quality static figures and interactive HTML visualizations of the dual-encoder embedding space.

### Research Questions Addressed

1. Do embeddings cluster by economically meaningful categories (sector, liquidity, market-cap)?
2. How do embedding relationships shift during crisis periods?
3. What is the clustering quality of the learned representations in the original embedding space?
4. Are observed clusters statistically significant compared to random baselines?

---

## Architecture

### Module Structure

```
src/evaluation/
└── umap_visualizer.py      # Core UMAPVisualizer class

scripts/visualization/
└── umap_plots.py           # CLI wrapper
```

### Class Design: `UMAPVisualizer`

```python
class UMAPVisualizer:
    """
    PCA → UMAP visualization pipeline for dual-encoder embeddings.
    
    Responsibilities:
    - Compute embeddings from model checkpoints (hybrid: pre-compute + on-demand)
    - Apply PCA for noise reduction (metrics computed in PCA space)
    - Apply UMAP (2D) for visualization only
    - Generate static PNG figures (sector/liquidity/market-cap colored)
    - Generate interactive HTML visualizations (Plotly)
    - Compute crisis snapshot comparisons (fixed reference projection)
    - Calculate clustering quality metrics in embedding/PCA space
    - Compute baseline comparisons (random permutation, correlation-based)
    
    Key methodological choices:
    - Clustering metrics computed in PCA space, NOT UMAP space (UMAP is for visualization only)
    - Crisis comparison uses fixed reference UMAP fit on pre-crisis data only
    - Unit of analysis: one point per ticker per period (end-of-period snapshot)
    """
    
    def __init__(
        self,
        model: DualEncoderModule,
        feature_loader: FeatureLoader,
        output_dir: Path,
        n_pca_components: int = 50,
        umap_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        random_state: int = 42,
    )
    
    def compute_embeddings(
        self,
        period_start: str,
        period_end: str,
        aggregation: Literal["end_period", "mean", "all_dates"] = "end_period",
        use_cache: bool = True,
    ) -> pd.DataFrame
    """
    Compute embeddings for specified period.
    
    Unit of analysis (aggregation parameter):
    - "end_period": One point per ticker (last trading day of period) - DEFAULT
    - "mean": One point per ticker (mean embedding across period)
    - "all_dates": Every ticker-date observation (for exploratory analysis only)
    
    Default is "end_period" to avoid temporal autocorrelation inflating cluster metrics.
    
    Returns DataFrame with columns:
    - ticker: str
    - date: datetime (end of period or snapshot date)
    - embedding_0..embedding_255: float
    - sector: str (GICS sector)
    - liquidity_tier: str (Q1-Q4, computed within period)
    - market_cap_tier: str (Large/Mid/Small/Micro, computed within period)
    """
    
    def project_pca(
        self,
        embeddings: np.ndarray,
        fit_period: Optional[str] = None,
    ) -> np.ndarray
    """
    Apply PCA for noise reduction and dimensionality reduction.
    
    PCA settings:
    - n_components=50 (retains >95% variance in typical embeddings)
    - whitening=False (preserves cosine geometry of embedding space)
    - Fit on reference period (default: pre-crisis) or full dataset
    - Transform all points using same fitted PCA
    
    Clustering metrics (silhouette, Davies-Bouldin, Calinski-Harabasz)
    are computed in this PCA space, NOT in UMAP space.
    """
    
    def project_umap(
        self,
        pca_embeddings: np.ndarray,
        metadata: pd.DataFrame,
        fit_mode: Literal["reference", "combined", "separate"] = "reference",
        reference_pca: Optional[np.ndarray] = None,
    ) -> np.ndarray
    """
    Apply UMAP for 2D visualization.
    
    UMAP settings:
    - n_components=2
    - n_neighbors=15 (default, configurable)
    - min_dist=0.1 (default, configurable)
    - metric='cosine' (appropriate for normalized embeddings)
    - random_state=42 (reproducibility)
    
    Fit modes:
    - "reference": Fit on reference period only, transform others (for crisis comparison)
    - "combined": Fit on all data combined (not recommended for crisis comparison - leaks information)
    - "separate": Fit independently per period (structure comparison only, no migration claims)
    
    Returns 2D projection for visualization. Do NOT compute clustering metrics on this output.
    """
    
    def generate_static_plots(
        self,
        projection: np.ndarray,
        metadata: pd.DataFrame,
        color_by: Literal["sector", "liquidity", "market_cap"],
        filename: str,
    ) -> Path
    
    def generate_interactive_html(
        self,
        projection: np.ndarray,
        metadata: pd.DataFrame,
        color_by: Literal["sector", "liquidity", "market_cap"],
        filename: str,
    ) -> Path
    
    def generate_crisis_comparison(
        self,
        pre_crisis_period: Tuple[str, str],
        crisis_period: Tuple[str, str],
        projection_mode: Literal["fixed_reference", "separate"] = "fixed_reference",
    ) -> Tuple[Path, Path]
    """
    Generate side-by-side crisis snapshot comparison.
    
    Two modes:
    
    1. Fixed Reference Projection (recommended, methodologically sound):
       - Fit UMAP on pre-crisis reference data ONLY
       - Transform crisis period points using same fitted UMAP
       - Pre-crisis points are in their natural positions
       - Crisis points are projected into pre-crisis space
       - Displacement vectors show genuine embedding migration
       - Interpretation: "How did crisis stocks' relationships change relative to pre-crisis structure?"
    
    2. Separate Projections (qualitative structure comparison only):
       - Fit UMAP independently per period
       - Each period optimized for its own structure
       - Better local structure preservation
       - CANNOT compare point positions directly
       - Interpretation: "How did internal cluster structure change?"
    
    Crisis windows (exact dates):
    - COVID: Pre (2019-01-01 to 2020-01-31) vs Crisis (2020-02-01 to 2020-05-31)
    - Rate Hike: Pre (2021-01-01 to 2021-12-31) vs Crisis (2022-01-01 to 2022-10-31)
    
    Liquidity and market-cap tiers are recomputed within each period
    to reflect changing market conditions.
    
    Output: Two side-by-side plots with identical axis scales
    """
    
    def compute_clustering_metrics(
        self,
        pca_embeddings: np.ndarray,
        metadata: pd.DataFrame,
    ) -> Dict[str, float]
    """
    Compute clustering quality metrics in PCA space (NOT UMAP space).
    
    Metrics:
    - Silhouette Score (by sector): Cohesion vs separation [-1 to 1, higher=better]
    - Silhouette Score (by liquidity): Liquidity stratification quality
    - Silhouette Score (by market_cap): Market-cap stratification quality
    - Davies-Bouldin Index (by sector): Lower = better clustering
    - Calinski-Harabasz Score (by sector): Higher = better clustering
    
    Baseline comparisons:
    - Random permutation: Shuffle labels, recompute metrics (null distribution)
    - Correlation-based embeddings: Same metrics on 60-day return correlation embeddings
    
    Interpretation criteria:
    - silhouette > 0.25: Meaningful clustering
    - silhouette > 0.5: Strong clustering
    - Model silhouette > baseline silhouette: Embeddings add value beyond correlation
    - Model silhouette >> random: Clusters not due to chance
    
    Returns dict:
    {
        "silhouette_sector": 0.45,
        "silhouette_liquidity": 0.32,
        "silhouette_market_cap": 0.38,
        "davies_bouldin_sector": 1.23,
        "calinski_harabasz_sector": 856.7,
        "silhouette_sector_random": 0.02,  # baseline
        "silhouette_sector_correlation": 0.28,  # baseline
    }
    """
    
    def run_full_evaluation(self) -> EvaluationResults
```

---

## Data Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Model Checkpoint│     │ Feature Parquet  │     │ Period Config   │
└────────┬────────┘     └────────┬─────────┘     └────────┬────────┘
         │                       │                        │
         └───────────────────────┼────────────────────────┘
                                 │
                                 ↓
                    ┌────────────────────────┐
                    │  compute_embeddings()  │
                    │  (hybrid caching)      │
                    └───────────┬────────────┘
                                │
                                ↓
              ┌─────────────────────────────────┐
              │  Embeddings DataFrame           │
              │  - ticker, date                 │
              │  - 256-dim joint embedding      │
              │  - sector, liquidity_tier       │
              │  - market_cap_tier              │
              └─────────────┬───────────────────┘
                            │
                            ↓
              ┌─────────────────────────────────┐
              │  PCA (50 components)            │
              │  - Noise reduction              │
              │  - Dimensionality reduction     │
              └─────────────┬───────────────────┘
                            │
                            ↓
              ┌─────────────────────────────────┐
              │  UMAP (2D projection)           │
              │  - n_neighbors=15               │
              │  - min_dist=0.1                 │
              │  - metric=cosine                │
              └─────────────┬───────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ↓                  ↓                  ↓
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Static PNG      │ │ Interactive HTML│ │ Crisis Compare  │
│ - sector        │ │ - sector        │ │ - fixed proj    │
│ - liquidity     │ │ - liquidity     │ │ - separate proj │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

---

## Implementation Details

### Embedding Computation (Hybrid Approach)

```python
def compute_embeddings(self, period_start, period_end, use_cache=True):
    """
    Compute embeddings for specified period with hybrid caching.
    
    Cache strategy:
    - Standard periods (pre-defined in config) → cache to parquet
    - Custom periods → compute on-demand, no cache
    
    Returns DataFrame with columns:
    - ticker: str
    - date: datetime
    - embedding_0..embedding_255: float
    - sector: str (GICS sector)
    - liquidity_tier: str (Q1-Q4)
    - market_cap_tier: str (Large/Mid/Small/Micro)
    """
```

### PCA → UMAP Pipeline (Two-Stage Reduction)

```python
def project_pca(self, embeddings, fit_period=None):
    """
    Stage 1: PCA for noise reduction.
    
    PCA settings:
    - n_components=50 (retains >95% variance, determined via sensitivity analysis)
    - whitening=False (preserves cosine geometry of embedding space)
    - Fit on reference period or full dataset
    - Clustering metrics computed HERE, not on UMAP output
    
    Sensitivity checks:
    - Report variance explained at n_components=10,25,50,100
    - Default 50 balances noise reduction vs. information loss
    """

def project_umap(self, pca_embeddings, metadata, fit_mode="reference"):
    """
    Stage 2: UMAP for 2D visualization ONLY.
    
    UMAP settings:
    - n_components=2
    - n_neighbors=15 (default, configurable)
    - min_dist=0.1 (default, configurable)
    - metric='cosine' (appropriate for normalized embeddings)
    - random_state=42 (reproducibility)
    
    Fit modes:
    - "reference": Fit on pre-crisis, transform crisis (for migration analysis)
    - "combined": Fit on all data (not recommended - information leakage)
    - "separate": Fit per period (structure comparison only)
    
    IMPORTANT: Do NOT compute clustering metrics on UMAP output.
    UMAP is nonlinear and can create/exaggerate apparent separation.
    """
```

### Static Plot Generation

```python
def generate_static_plots(self, projection, metadata, color_by, filename):
    """
    Generate publication-quality static PNG.
    
    Specifications:
    - Size: 10x8 inches (300 DPI)
    - Font: sans-serif, 10pt
    - Alpha: 0.6 (point transparency for density)
    - Point size: 40
    - Legend: Right side, titled
    - Grid: Light gray, alpha=0.3
    - Axis: Hidden (no numeric labels)
    - Title: Top, bold 12pt
    
    Color schemes:
    - Sector: Tab20 colormap (20 colors, supports 11 sectors without reuse)
    - Liquidity: Viridis (Q1-Q4 sequential)
    - Market Cap: Set2 (4 colors for Large/Mid/Small/Micro)
    - Market Cap: Set2 (4 colors for Large/Mid/Small/Micro)
    """
```

### Interactive HTML Generation

```python
def generate_interactive_html(self, projection, metadata, color_by, filename):
    """
    Generate interactive Plotly HTML visualization.
    
    Features (basic, extensible):
    - Hover tooltips: ticker, sector, liquidity_tier, date
    - Zoom/pan controls
    - Legend click to toggle visibility
    - Download plot button
    
    Architecture for extension:
    - Tooltip fields defined in config dict
    - Easy to add: embedding coords, similarity scores, etc.
    """
```

### Crisis Comparison (Fixed Reference Projection)

```python
def generate_crisis_comparison(
    self,
    pre_crisis_period,
    crisis_period,
    projection_mode="fixed_reference"
):
    """
    Generate side-by-side crisis snapshot comparison.
    
    Methodologically sound approach:
    
    1. Fixed Reference Projection (DEFAULT, recommended):
       - Fit UMAP on pre-crisis reference data ONLY
       - Transform crisis period points using same fitted UMAP
       - Pre-crisis points are in their natural positions
       - Crisis points are projected into pre-crisis space
       - Displacement vectors show genuine embedding migration
       - Interpretation: "How did crisis stocks' relationships change 
         relative to pre-crisis structure?"
       - Statistically valid: no information leakage from crisis to reference
    
    2. Separate Projections (qualitative only, no migration claims):
       - Fit UMAP independently per period
       - Each period optimized for its own structure
       - Better local structure preservation
       - CANNOT compare point positions directly
       - Interpretation: "How did internal cluster structure change?"
    
    Crisis windows (exact dates, business days only):
    - COVID: 
      - Pre: 2019-01-01 to 2020-01-31 (13 months baseline)
      - Crisis: 2020-02-01 to 2020-05-31 (4 months shock)
    - Rate Hike:
      - Pre: 2021-01-01 to 2021-12-31 (12 months baseline)
      - Crisis: 2022-01-01 to 2022-10-31 (10 months shock)
    
    Tier recomputation:
    - Liquidity quartiles computed within each period
    - Market-cap tiers computed within each period
    - Reflects changing market conditions during crisis
    
    Output: Two side-by-side plots with identical axis scales,
    plus displacement statistics (mean distance moved, top 10 movers).
    """
```

### Clustering Metrics

```python
def compute_clustering_metrics(self, projection, metadata):
    """
    Compute clustering quality metrics.
    
    Metrics:
    - Silhouette Score (by sector): Cohesion vs separation
    - Silhouette Score (by liquidity): Liquidity stratification quality
    - Davies-Bouldin Index (by sector): Lower = better clustering
    - Calinski-Harabasz Score (by sector): Higher = better clustering
    
    Returns dict:
    {
        "silhouette_sector": 0.45,
        "silhouette_liquidity": 0.32,
        "davies_bouldin_sector": 1.23,
        "calinski_harabasz_sector": 856.7,
    }
    """
```

---

## CLI Interface

### Script: `scripts/visualization/umap_plots.py`

```bash
# Full evaluation (all plots, both crisis periods)
python -m scripts.visualization.umap_plots \
    --checkpoint checkpoints/best_model.ckpt \
    --feature-dir data/processed \
    --output-dir results/figures \
    --periods covid rate_hike

# Static plots only
python -m scripts.visualization.umap_plots \
    --checkpoint checkpoints/best_model.ckpt \
    --feature-dir data/processed \
    --output-dir results/figures \
    --static-only

# Interactive plots only
python -m scripts.visualization.umap_plots \
    --checkpoint checkpoints/best_model.ckpt \
    --feature-dir data/processed \
    --output-dir results/figures \
    --interactive-only

# Single crisis period
python -m scripts.visualization.umap_plots \
    --checkpoint checkpoints/best_model.ckpt \
    --feature-dir data/processed \
    --output-dir results/figures \
    --periods covid

# Custom UMAP parameters
python -m scripts.visualization.umap_plots \
    --checkpoint checkpoints/best_model.ckpt \
    --feature-dir data/processed \
    --output-dir results/figures \
    --umap-neighbors 30 \
    --umap-min-dist 0.05
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | Path | Required | Model checkpoint path |
| `--feature-dir` | Path | Required | Directory with feature parquet files |
| `--output-dir` | Path | `results/figures` | Output directory |
| `--periods` | List | `["covid", "rate_hike"]` | Crisis periods to analyze |
| `--static-only` | Flag | False | Generate only static PNG |
| `--interactive-only` | Flag | False | Generate only interactive HTML |
| `--umap-neighbors` | int | 15 | UMAP n_neighbors parameter |
| `--umap-min-dist` | float | 0.1 | UMAP min_dist parameter |
| `--pca-components` | int | 50 | PCA n_components |
| `--no-cache` | Flag | False | Disable embedding cache |

---

## Output Files

### Static Figures

| File | Description |
|------|-------------|
| `umap_sector.png` | 2D scatter colored by GICS sector |
| `umap_liquidity.png` | 2D scatter colored by liquidity quartile |
| `umap_crisis_covid_fixed.png` | COVID: pre vs crisis (fixed projection) |
| `umap_crisis_covid_separate.png` | COVID: pre vs crisis (separate projections) |
| `umap_crisis_ratehike_fixed.png` | Rate hike: pre vs crisis (fixed) |
| `umap_crisis_ratehike_separate.png` | Rate hike: pre vs crisis (separate) |

### Interactive HTML

| File | Description |
|------|-------------|
| `umap_sector.html` | Interactive sector-colored plot |
| `umap_liquidity.html` | Interactive liquidity-colored plot |

### Metrics

| File | Description |
|------|-------------|
| `clustering_metrics.csv` | Silhouette, Davies-Bouldin, Calinski-Harabasz by coloring scheme |

---

## Dependencies

### New Dependencies (add to `pyproject.toml`)

```toml
[project.optional-dependencies]
evaluation = [
    "umap-learn>=0.5.0",      # UMAP projection
    "plotly>=5.14.0",         # Interactive HTML visualizations
]
```

### Existing Dependencies Used

- `scikit-learn>=1.3.0` - PCA, clustering metrics
- `matplotlib>=3.7.0` - Static plotting
- `pandas>=2.0.0` - Data handling
- `numpy>=1.24.0` - Array operations
- `torch>=2.0.0` - Model loading
- `pytorch_lightning>=2.6.1` - Lightning module

---

## Error Handling

| Error | Handling |
|-------|----------|
| Checkpoint not found | Raise `FileNotFoundError` with helpful message |
| Feature data missing for period | Log warning, skip period, continue |
| UMAP convergence failure | Retry with increased `n_neighbors`, log warning |
| Empty embedding set | Raise `ValueError` with minimum sample requirement |
| Memory overflow (large datasets) | Process in batches, log progress |

---

## Testing Strategy

### Unit Tests (`tests/evaluation/test_umap_visualizer.py`)

```python
class TestUMAPVisualizer:
    def test_embedding_computation()
    def test_pca_umap_pipeline()
    def test_static_plot_generation()
    def test_interactive_html_generation()
    def test_crisis_comparison_fixed()
    def test_crisis_comparison_separate()
    def test_clustering_metrics()
    def test_hybrid_caching()
```

### Integration Tests (`tests/evaluation/test_visualization_integration.py`)

```python
class TestVisualizationIntegration:
    def test_full_pipeline_with_mock_model()
    def test_cli_interface()
    def test_output_file_generation()
```

---

## Success Criteria

- [ ] Module generates all 6 static PNG figures without errors
- [ ] Module generates 2 interactive HTML files with working hover tooltips
- [ ] Crisis comparison shows both fixed and separate projection modes
- [ ] Clustering metrics CSV contains all 4 metrics for both sector and liquidity
- [ ] Hybrid caching works: standard periods cached, custom periods computed on-demand
- [ ] CLI interface accepts all documented arguments
- [ ] Unit tests achieve >80% code coverage
- [ ] Static figures are publication-ready (300 DPI, proper sizing)

---

## Future Extensions (Not in Scope)

The following are explicitly out of scope for this implementation but the architecture supports them:

- **Separate embedding views**: Visualize temporal-only, tabular-only embeddings
- **Enhanced tooltips**: Add embedding coordinates, similarity scores, neighbor info
- **Dashboard mode**: Period slider, sector filter, search by ticker
- **Multi-crisis comparison**: More than 2 crisis periods in single view
- **Animation**: Time-lapse of embedding evolution
- **3D UMAP**: Three-dimensional projections

---

## References

- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426.
- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
- GICS Sector Classification: https://www.msci.com/our-solutions/indexes/gics

---

**Spec Approved**: March 30, 2026  
**Next Step**: Invoke `writing-plans` skill for implementation plan
