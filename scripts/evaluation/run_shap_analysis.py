#!/usr/bin/env python
"""
Run SHAP feature importance analysis for LiquidSearcher.

Example:
    python -m scripts.evaluation.run_shap_analysis \\
        --checkpoint checkpoints/last.ckpt \\
        --features data/processed/all_features.parquet \\
        --period-start 2019-01-01 \\
        --period-end 2019-12-31 \\
        --n-queries 5 \\
        --output-dir results/shap_test
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

from src.evaluation.feature_importance.shap_analyzer import DualEncoderExplainer
from src.evaluation.feature_importance.sampling import (
    QuerySampler,
    select_background_samples,
)
from src.evaluation.visualizations.shap_plots import (
    plot_global_summary,
    plot_waterfall,
    plot_importance_ranking,
)
from src.models.dual_encoder import DualEncoder


def load_model(checkpoint_path: str | Path) -> DualEncoder:
    """Load DualEncoder model from checkpoint."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    hyperparams = checkpoint.get("hyper_parameters", {})

    model = DualEncoder(
        temporal_input_dim=hyperparams.get("temporal_input_dim", 13),
        tabular_continuous_dim=hyperparams.get("tabular_continuous_dim", 15),
        tabular_categorical_dims=hyperparams.get("tabular_categorical_dims", [11, 25]),
        tabular_embedding_dims=hyperparams.get("tabular_embedding_dims", [8, 16]),
        embedding_dim=hyperparams.get("embedding_dim", 128),
        temperature=hyperparams.get("temperature", 0.07),
    )

    state_dict = checkpoint.get("state_dict", checkpoint)
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"Model loaded successfully")
    print(f"  Embedding dim: {model.embedding_dim}")
    print(f"  Device: cpu")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Run SHAP feature importance analysis for LiquidSearcher"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/last.ckpt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="data/processed/all_features.parquet",
        help="Path to features parquet file",
    )
    parser.add_argument(
        "--period-start",
        type=str,
        default="2019-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--period-end",
        type=str,
        default="2019-12-31",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=5,
        help="Number of query stocks to analyze",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=10,
        help="Number of candidates to explain per query",
    )
    parser.add_argument(
        "--background-size",
        type=int,
        default=50,
        help="Number of background samples for SHAP",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/shap_test",
        help="Output directory for results",
    )
    parser.add_argument(
        "--query-method",
        type=str,
        default="stratified",
        choices=["stratified", "random", "market_cap"],
        help="Query sampling method",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("LIQUIDSEARCHER SHAP FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70 + "\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_query").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    print(f"Loading features from {args.features}...")
    df = pd.read_parquet(args.features)
    df["date"] = pd.to_datetime(df["date"])

    mask = (df["date"] >= args.period_start) & (df["date"] <= args.period_end)
    period_df = df[mask].copy()

    stocks_df = period_df.sort_values("date").groupby("symbol").last().reset_index()

    print(f"Period: {args.period_start} to {args.period_end}")
    print(f"Stocks: {len(stocks_df)}")
    print(f"  Sectors: {stocks_df['gsector'].nunique()}")
    print(f"  Date range: {stocks_df['date'].min()} to {stocks_df['date'].max()}\n")

    print("Selecting query stocks...")
    if args.query_method == "stratified":
        queries = QuerySampler.stratified_by_sector_and_liquidity(
            stocks_df, n_queries=args.n_queries
        )
    elif args.query_method == "random":
        queries = QuerySampler.random(stocks_df, n_queries=args.n_queries)
    else:
        queries = QuerySampler.by_market_cap_tiers(stocks_df, n_queries=args.n_queries)

    print(f"Selected {len(queries)} queries:")
    for idx, row in queries.iterrows():
        print(
            f"  {row['symbol']:8} (Sector: {row.get('gsector', 'N/A')}, "
            f"Market Cap: {row.get('market_cap', 0):.2f})"
        )
    print()

    print("Selecting background samples...")
    background = select_background_samples(
        stocks_df, n_samples=args.background_size, method="kmeans"
    )
    print(f"  {len(background)} background samples (k-means method)\n")

    print("Loading model...")
    model = load_model(args.checkpoint)
    print()

    print("Initializing SHAP explainer...")
    explainer = DualEncoderExplainer(
        model=model,
        background_data=background,
        background_size=args.background_size,
        device="cpu",
    )
    print()

    print("Running SHAP analysis...")
    print("-" * 70)

    results = explainer.explain_batch(
        queries_df=queries,
        all_stocks_df=stocks_df,
        n_candidates_per_query=args.n_candidates,
        output_dir=output_dir / "per_query",
    )

    print("-" * 70)
    print()

    print("Saving global importance rankings...")
    importance_path = output_dir / "global_importance.csv"
    results["global_importance"].to_csv(importance_path, index=False)
    print(f"  Saved: {importance_path}\n")

    print("Generating visualizations...")
    figures_dir = output_dir / "figures"

    plot_global_summary(
        shap_values=results["shap_matrix"],
        feature_names=results["feature_names"],
        output_path=figures_dir / "shap_summary.png",
        max_features=15,
    )

    plot_importance_ranking(
        importance_df=results["global_importance"],
        output_path=figures_dir / "shap_importance_ranking.png",
        max_features=15,
    )

    if len(results["per_query_results"]) > 0:
        sample = results["per_query_results"].iloc[0]
        shap_vals = np.array([sample[f"shap_{f}"] for f in results["feature_names"]])

        plot_waterfall(
            shap_values=shap_vals,
            feature_names=results["feature_names"],
            base_value=float(sample["base_value"]),
            similarity_score=float(sample["similarity_score"]),
            candidate_ticker=str(sample["candidate_ticker"]),
            output_path=figures_dir / f"shap_waterfall_{sample['query_ticker']}.png",
        )

    print()
    print("=" * 70)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Global importance: {importance_path}")
    print(f"  - Per-query results: {output_dir / 'per_query'}/")
    print(f"  - Figures: {figures_dir}/")

    print(f"\nValidation Statistics:")
    print(f"  Mean residual: {results['residual_stats']['mean_residual']:.6f}")
    print(f"  Max residual: {results['residual_stats']['max_residual']:.6f}")
    print(f"  Pass rate (< 0.01): {results['residual_stats']['pct_pass_validation']:.1f}%")

    print(f"\nTop 5 Most Important Features:")
    for i, row in results["global_importance"].head(5).iterrows():
        print(f"  {i + 1}. {row['feature']:20} (Mean |SHAP|: {row['mean_abs_shap']:.4f})")

    print(f"\n{'=' * 70}\n")

    return results


if __name__ == "__main__":
    main()
