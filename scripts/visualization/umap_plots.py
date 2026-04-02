#!/usr/bin/env python
# scripts/visualization/umap_plots.py
"""CLI for generating UMAP visualizations.

Usage:
    python -m scripts.visualization.umap_plots \
        --feature-path data/processed/all_features.parquet \
        --output-dir results/figures/umap
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
        "--feature-path",
        type=str,
        required=True,
        help="Path to all_features.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/figures/umap",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--periods",
        nargs="+",
        choices=["covid", "rate_hike"],
        default=["covid"],
        help="Crisis periods to analyze",
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

    feature_path = Path(args.feature_path)
    if not feature_path.exists():
        print(f"Error: Feature file not found: {feature_path}", file=sys.stderr)
        sys.exit(1)

    from src.evaluation import UMAPVisualizer, FeatureLoader

    feature_loader = FeatureLoader(feature_path)

    min_date, max_date = feature_loader.get_available_date_range()
    print(f"Data available: {min_date.date()} to {max_date.date()}")

    visualizer = UMAPVisualizer(
        feature_loader=feature_loader,
        output_dir=args.output_dir,
        n_pca_components=args.pca_components,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
    )

    print(f"Output directory: {args.output_dir}")

    for period_key in args.periods:
        visualizer.run_full_evaluation(period_key)

    print(f"\n{'=' * 60}")
    print("Visualization complete!")
    print(f"Output directory: {Path(args.output_dir).absolute()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
