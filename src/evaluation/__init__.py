"""LiquidSearcher evaluation framework."""

from src.evaluation.visualizations import UMAPVisualizer
from src.evaluation.metrics import (
    compute_silhouette_score,
    compute_davies_bouldin_score,
    compute_calinski_harabasz_score,
    compute_all_clustering_metrics,
)
from src.evaluation.utils import FeatureLoader, EmbeddingCache
from src.evaluation.feature_importance import (
    DualEncoderExplainer,
    QuerySampler,
    select_background_samples,
    LIQUIDITY_FEATURES,
    FUNDAMENTAL_FEATURES,
    SIZE_FEATURES,
    VOLATILITY_FEATURES,
    FEATURE_GROUPS,
    aggregate_shap_by_group,
)

__all__ = [
    "UMAPVisualizer",
    "FeatureLoader",
    "EmbeddingCache",
    "compute_silhouette_score",
    "compute_davies_bouldin_score",
    "compute_calinski_harabasz_score",
    "compute_all_clustering_metrics",
    "DualEncoderExplainer",
    "QuerySampler",
    "select_background_samples",
    "LIQUIDITY_FEATURES",
    "FUNDAMENTAL_FEATURES",
    "SIZE_FEATURES",
    "VOLATILITY_FEATURES",
    "FEATURE_GROUPS",
    "aggregate_shap_by_group",
]
