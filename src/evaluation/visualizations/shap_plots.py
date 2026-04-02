"""SHAP visualization functions for feature importance analysis."""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_global_summary(
    shap_values: np.ndarray,
    feature_names: List[str],
    output_path: Path,
    max_features: int = 20,
    figsize: tuple = (10, 8),
) -> Path:
    """
    Create global SHAP summary plot (beeswarm-style).

    Shows feature importance ranking and impact direction.

    Args:
        shap_values: (n_samples, n_features) array of SHAP values
        feature_names: Feature names
        output_path: Where to save plot
        max_features: Top N features to show
        figsize: Figure size

    Returns:
        Path to saved figure
    """
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    n_features = min(max_features, len(feature_names))

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:n_features]

    top_shap = shap_values[:, top_indices]
    top_features = [feature_names[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=figsize)

    colors = sns.color_palette("rocket", as_cmap=True)

    for i, feat_idx in enumerate(reversed(top_indices)):
        shap_vals = shap_values[:, feat_idx]

        y_pos = np.ones(len(shap_vals)) * (n_features - 1 - i)

        scatter = ax.scatter(
            shap_vals,
            y_pos + np.random.uniform(-0.3, 0.3, len(shap_vals)),
            c=np.abs(shap_vals),
            cmap=colors,
            alpha=0.6,
            s=30,
        )

        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)

    ax.set_yticks(range(n_features))
    ax.set_yticklabels([top_features[i] for i in range(n_features - 1, -1, -1)])
    ax.set_xlabel("SHAP Value (impact on similarity)", fontsize=12)
    ax.set_title("Global SHAP Summary Plot", fontsize=14, fontweight="bold")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("|SHAP Value|", fontsize=10)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved summary plot to {output_path}")
    return output_path


def plot_waterfall(
    shap_values: np.ndarray,
    feature_names: List[str],
    base_value: float,
    similarity_score: float,
    candidate_ticker: str,
    output_path: Path,
    max_features: int = 10,
    figsize: tuple = (10, 6),
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
        max_features: Top N features to show
        figsize: Figure size

    Returns:
        Path to saved figure
    """
    if shap_values.ndim > 1:
        shap_values = shap_values.flatten()

    n_features = min(max_features, len(feature_names))

    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[::-1][:n_features]

    top_shap = shap_values[top_indices]
    top_features = [feature_names[i] for i in top_indices]

    sorted_indices = np.argsort(top_shap)
    sorted_shap = top_shap[sorted_indices]
    sorted_features = [top_features[i] for i in sorted_indices]

    cumulative = np.cumsum(sorted_shap)

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["#008B00" if v > 0 else "#B22222" for v in sorted_shap]

    bars = ax.barh(
        range(len(sorted_shap)),
        sorted_shap,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.axvline(x=0, color="gray", linestyle="-", linewidth=1)

    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel("SHAP Value", fontsize=12)
    ax.set_title(
        f"SHAP Waterfall: {candidate_ticker}\n"
        f"Base: {base_value:.4f} → Predicted: {similarity_score:.4f}",
        fontsize=14,
    )

    for i, (bar, val) in enumerate(zip(bars, sorted_shap)):
        width = bar.get_width()
        ax.text(
            width + np.sign(width) * 0.001,
            i,
            f"{val:.4f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved waterfall plot to {output_path}")
    return output_path


def plot_feature_distribution(
    shap_values: np.ndarray,
    feature_names: List[str],
    feature_values: np.ndarray,
    output_path: Path,
    max_features: int = 10,
    figsize: tuple = (12, 8),
) -> Path:
    """
    Create distribution plot showing SHAP value vs feature value.

    For each feature, shows whether high/low values increase similarity.

    Args:
        shap_values: (n_samples, n_features) SHAP values
        feature_names: Feature names
        feature_values: (n_samples, n_features) original feature values
        output_path: Where to save plot
        max_features: Top N features to show
        figsize: Figure size

    Returns:
        Path to saved figure
    """
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)
        feature_values = feature_values.reshape(1, -1)

    n_features = min(max_features, len(feature_names))

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:n_features]

    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]

    for i, feat_idx in enumerate(top_indices):
        ax = axes[i]

        shap_vals = shap_values[:, feat_idx]
        feat_vals = feature_values[:, feat_idx]

        scatter = ax.scatter(
            feat_vals,
            shap_vals,
            c=feat_vals,
            cmap="RdBu_r",
            alpha=0.6,
            s=20,
        )

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlabel(feature_names[feat_idx], fontsize=10)
        ax.set_ylabel("SHAP Value", fontsize=10)
        ax.set_title(feature_names[feat_idx], fontsize=11, fontweight="bold")

        plt.colorbar(scatter, ax=ax, label="Feature Value")

    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(
        "SHAP Values vs Feature Values",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved feature distribution plot to {output_path}")
    return output_path


def plot_importance_ranking(
    importance_df: pd.DataFrame,
    output_path: Path,
    max_features: int = 15,
    figsize: tuple = (8, 10),
) -> Path:
    """
    Create horizontal bar plot of feature importance ranking.

    Args:
        importance_df: DataFrame with 'feature' and 'mean_abs_shap' columns
        output_path: Where to save plot
        max_features: Top N features to show
        figsize: Figure size

    Returns:
        Path to saved figure
    """
    top_features = importance_df.head(max_features).copy()
    top_features = top_features.sort_values("mean_abs_shap", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(
        top_features["feature"],
        top_features["mean_abs_shap"],
        color=sns.color_palette("Blues_d", len(top_features)),
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("Global Feature Importance Ranking", fontsize=14, fontweight="bold")

    for bar, val in zip(bars, top_features["mean_abs_shap"]):
        width = bar.get_width()
        ax.text(
            width + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved importance ranking plot to {output_path}")
    return output_path


def create_all_shap_figures(
    results: dict,
    output_dir: Path,
    per_query_results: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Generate all SHAP visualization figures.

    Args:
        results: Output from DualEncoderExplainer.explain_batch()
        output_dir: Directory to save figures
        per_query_results: Optional combined per-query DataFrame

    Returns:
        Dictionary with paths to generated figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {}

    figures["summary"] = plot_global_summary(
        shap_values=results["shap_matrix"],
        feature_names=results["feature_names"],
        output_path=output_dir / "shap_summary.png",
        max_features=15,
    )

    figures["importance_ranking"] = plot_importance_ranking(
        importance_df=results["global_importance"],
        output_path=output_dir / "shap_importance_ranking.png",
        max_features=15,
    )

    if per_query_results is not None and len(per_query_results) > 0:
        sample_query = per_query_results["query_ticker"].iloc[0]
        sample_candidate = per_query_results["candidate_ticker"].iloc[0]

        sample_row = per_query_results[
            (per_query_results["query_ticker"] == sample_query)
            & (per_query_results["candidate_ticker"] == sample_candidate)
        ].iloc[0]

        shap_vals = np.array([sample_row[f"shap_{f}"] for f in results["feature_names"]])

        figures["waterfall"] = plot_waterfall(
            shap_values=shap_vals,
            feature_names=results["feature_names"],
            base_value=sample_row["base_value"],
            similarity_score=sample_row["similarity_score"],
            candidate_ticker=sample_candidate,
            output_path=output_dir / f"shap_waterfall_{sample_query}.png",
        )

    if per_query_results is not None:
        feature_vals = np.column_stack(
            [per_query_results[f"shap_{f}"] for f in results["feature_names"]]
        )

        figures["distribution"] = plot_feature_distribution(
            shap_values=results["shap_matrix"],
            feature_names=results["feature_names"],
            feature_values=feature_vals,
            output_path=output_dir / "shap_feature_distribution.png",
            max_features=10,
        )

    return figures
