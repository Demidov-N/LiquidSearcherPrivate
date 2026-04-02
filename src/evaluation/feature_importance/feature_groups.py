"""Feature groupings for aggregated SHAP importance analysis."""

from typing import Dict, List

import pandas as pd

LIQUIDITY_FEATURES = [
    "market_cap",
    "beta",
    "idiosyncratic_vol",
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

PROFITABILITY_FEATURES = [
    "roe",
    "roa",
    "operating_margin",
    "profit_margin",
]

VALUATION_FEATURES = [
    "price_to_book",
    "price_to_earnings",
]

LEVERAGE_FEATURES = [
    "debt_to_equity",
]

FEATURE_GROUPS: Dict[str, List[str]] = {
    "liquidity": LIQUIDITY_FEATURES,
    "fundamental": FUNDAMENTAL_FEATURES,
    "size": SIZE_FEATURES,
    "volatility": VOLATILITY_FEATURES,
    "profitability": PROFITABILITY_FEATURES,
    "valuation": VALUATION_FEATURES,
    "leverage": LEVERAGE_FEATURES,
}


def aggregate_shap_by_group(
    shap_df: pd.DataFrame,
    feature_groups: Dict[str, List[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate SHAP values by feature group.

    Args:
        shap_df: Per-feature SHAP values with columns like 'shap_market_cap'
        feature_groups: Mapping of group name → feature names

    Returns:
        DataFrame with aggregated importance per group
    """
    if feature_groups is None:
        feature_groups = FEATURE_GROUPS

    group_importance = []

    for group_name, features in feature_groups.items():
        shap_cols = [f"shap_{f}" for f in features if f"shap_{f}" in shap_df.columns]

        if not shap_cols:
            continue

        group_abs_shap = shap_df[shap_cols].abs().mean(axis=1)

        group_importance.append(
            {
                "group": group_name,
                "mean_group_shap": group_abs_shap.mean(),
                "std_group_shap": group_abs_shap.std(),
                "n_features": len(shap_cols),
                "features": features,
            }
        )

    result_df = pd.DataFrame(group_importance)

    if len(result_df) > 0:
        result_df["pct_importance"] = (
            result_df["mean_group_shap"] / result_df["mean_group_shap"].sum() * 100
        )
        result_df = result_df.sort_values("mean_group_shap", ascending=False)

    return result_df


def get_feature_metadata() -> pd.DataFrame:
    """
    Get metadata for all continuous features.

    Returns:
        DataFrame with feature descriptions and categories
    """
    metadata = {
        "market_cap": {
            "description": "Market capitalization",
            "category": "size",
            "expected_importance": "high",
        },
        "beta": {
            "description": "Market beta (systematic risk)",
            "category": "volatility",
            "expected_importance": "high",
        },
        "idiosyncratic_vol": {
            "description": "Idiosyncratic volatility",
            "category": "volatility",
            "expected_importance": "high",
        },
        "roe": {
            "description": "Return on equity",
            "category": "profitability",
            "expected_importance": "medium",
        },
        "roa": {
            "description": "Return on assets",
            "category": "profitability",
            "expected_importance": "medium",
        },
        "debt_to_equity": {
            "description": "Debt-to-equity ratio",
            "category": "leverage",
            "expected_importance": "medium",
        },
        "price_to_book": {
            "description": "Price-to-book ratio",
            "category": "valuation",
            "expected_importance": "medium",
        },
        "price_to_earnings": {
            "description": "Price-to-earnings ratio",
            "category": "valuation",
            "expected_importance": "low",
        },
        "operating_margin": {
            "description": "Operating margin",
            "category": "profitability",
            "expected_importance": "medium",
        },
        "profit_margin": {
            "description": "Net profit margin",
            "category": "profitability",
            "expected_importance": "medium",
        },
        "dividend_yield": {
            "description": "Dividend yield",
            "category": "returns",
            "expected_importance": "low",
        },
        "revenue": {
            "description": "Total revenue",
            "category": "size",
            "expected_importance": "low",
        },
        "net_income": {
            "description": "Net income",
            "category": "profitability",
            "expected_importance": "low",
        },
        "total_assets": {
            "description": "Total assets",
            "category": "size",
            "expected_importance": "low",
        },
        "cash": {
            "description": "Cash and equivalents",
            "category": "liquidity",
            "expected_importance": "low",
        },
    }

    return (
        pd.DataFrame.from_dict(metadata, orient="index")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
