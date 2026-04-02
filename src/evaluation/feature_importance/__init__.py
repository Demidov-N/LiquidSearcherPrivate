"""Feature importance analysis using SHAP values."""

from src.evaluation.feature_importance.shap_analyzer import DualEncoderExplainer
from src.evaluation.feature_importance.sampling import (
    QuerySampler,
    select_background_samples,
)
from src.evaluation.feature_importance.feature_groups import (
    LIQUIDITY_FEATURES,
    FUNDAMENTAL_FEATURES,
    SIZE_FEATURES,
    VOLATILITY_FEATURES,
    FEATURE_GROUPS,
    aggregate_shap_by_group,
)

__all__ = [
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
