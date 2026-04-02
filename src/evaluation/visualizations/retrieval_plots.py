"""Retrieval visualization functions for evaluation metrics plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_overall_metrics(
    metrics_df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (14, 6),
) -> Path:
    """Create bar chart of overall retrieval metrics comparison.

    Compares the 6 rankers side-by-side:
    - embedding, pearson_corr, spearman_corr (base rankers)
    - embedding_rerank, pearson_corr_rerank, spearman_corr_rerank (rerankers)

    Args:
        metrics_df: DataFrame with columns [metric_name, embedding, pearson_corr,
            spearman_corr, embedding_rerank, pearson_corr_rerank, spearman_corr_rerank]
        output_path: Where to save plot
        figsize: Figure size

    Returns:
        Path to saved figure
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for grouped bar chart
    metrics = metrics_df["metric_name"].tolist()
    n_metrics = len(metrics)

    # Set up bar positions
    x = np.arange(n_metrics)
    width = 0.12  # Width of each bar (6 bars total)

    # Define colors for each ranker
    colors = {
        "embedding": "#2E86AB",
        "pearson_corr": "#A23B72",
        "spearman_corr": "#F18F01",
        "embedding_rerank": "#06A77D",
        "pearson_corr_rerank": "#6B4EE6",
        "spearman_corr_rerank": "#C73E1D",
    }

    # Plot each ranker
    bars_list = []
    for i, ranker in enumerate(
        [
            "embedding",
            "pearson_corr",
            "spearman_corr",
            "embedding_rerank",
            "pearson_corr_rerank",
            "spearman_corr_rerank",
        ]
    ):
        if ranker in metrics_df.columns:
            values = metrics_df[ranker].values
            offset = (i - 2.5) * width  # Center the bars
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=ranker.replace("_", " ").title(),
                color=colors[ranker],
            )
            bars_list.append(bars)

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Overall Retrieval Metrics Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    # Add value labels on bars
    for bars in bars_list:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved overall metrics plot to {output_path}")
    return output_path


def plot_grouped_metrics(
    grouped_metrics_df: pd.DataFrame,
    metric_col: str,
    group_col: str,
    output_path: Path,
    figsize: tuple = (14, 6),
) -> Path:
    """Create grouped bar chart of metrics by category (sector, market_cap, liquidity quartile).

    Shows comparison across groups for embedding vs correlation vs liquidity-distance.

    Args:
        grouped_metrics_df: DataFrame with columns [group_name, embedding, correlation,
            liquidity_distance, hybrid] - hybrid is optional
        metric_col: Name of the metric being plotted (e.g., 'Recall@10')
        group_col: Name of the grouping column (e.g., 'sector')
        output_path: Where to save plot
        figsize: Figure size

    Returns:
        Path to saved figure
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    groups = grouped_metrics_df["group_name"].tolist()
    n_groups = len(groups)

    x = np.arange(n_groups)
    width = 0.25

    ax.bar(x - width, grouped_metrics_df["embedding"], width, label="Embedding", color="#2E86AB")
    ax.bar(x, grouped_metrics_df["correlation"], width, label="Correlation", color="#A23B72")
    ax.bar(
        x + width,
        grouped_metrics_df["liquidity_distance"],
        width,
        label="Liquidity-Distance",
        color="#F18F01",
    )

    if "hybrid" in grouped_metrics_df.columns:
        ax.bar(x + 2 * width, grouped_metrics_df["hybrid"], width, label="Hybrid", color="#C73E1D")

    ax.set_xlabel(group_col.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(metric_col, fontsize=12)
    ax.set_title(
        f"{metric_col} by {group_col.replace('_', ' ').title()}", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved grouped metrics plot to {output_path}")
    return output_path


def create_all_retrieval_figures(
    overall_metrics: pd.DataFrame,
    by_sector: pd.DataFrame,
    by_market_cap: pd.DataFrame,
    by_liquidity: pd.DataFrame,
    output_dir: Path,
    hybrid_metrics: pd.DataFrame | None = None,
) -> dict:
    """Generate all retrieval visualization figures.

    Args:
        overall_metrics: DataFrame with overall metrics
        by_sector: DataFrame with metrics grouped by sector
        by_market_cap: DataFrame with metrics grouped by market cap
        by_liquidity: DataFrame with metrics grouped by liquidity quartile
        output_dir: Directory to save figures
        hybrid_metrics: Optional DataFrame with hybrid metrics

    Returns:
        Dictionary with paths to generated figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {}

    # Overall metrics plot
    figures["overall"] = plot_overall_metrics(
        metrics_df=overall_metrics,
        output_path=output_dir / "metrics_overall_comparison.png",
    )

    # Grouped metrics plots
    if len(by_sector) > 0:
        for metric in ["Recall@10", "Spearman", "nDCG@10"]:
            if metric in by_sector.columns or metric in overall_metrics["metric_name"].values:
                figures[f"sector_{metric.lower().replace('@', '_at_')}"] = plot_grouped_metrics(
                    grouped_metrics_df=by_sector,
                    metric_col=metric,
                    group_col="sector",
                    output_path=output_dir
                    / f"metrics_by_sector_{metric.lower().replace('@', '_at_')}.png",
                )

    if len(by_market_cap) > 0:
        for metric in ["Recall@10", "Spearman", "nDCG@10"]:
            if metric in by_market_cap.columns or metric in overall_metrics["metric_name"].values:
                figures[f"market_cap_{metric.lower().replace('@', '_at_')}"] = plot_grouped_metrics(
                    grouped_metrics_df=by_market_cap,
                    metric_col=metric,
                    group_col="market_cap_tier",
                    output_path=output_dir
                    / f"metrics_by_market_cap_{metric.lower().replace('@', '_at_')}.png",
                )

    if len(by_liquidity) > 0:
        for metric in ["Recall@10", "Spearman", "nDCG@10"]:
            if metric in by_liquidity.columns or metric in overall_metrics["metric_name"].values:
                figures[f"liquidity_{metric.lower().replace('@', '_at_')}"] = plot_grouped_metrics(
                    grouped_metrics_df=by_liquidity,
                    metric_col=metric,
                    group_col="liquidity_quartile",
                    output_path=output_dir
                    / f"metrics_by_liquidity_{metric.lower().replace('@', '_at_')}.png",
                )

    return figures
