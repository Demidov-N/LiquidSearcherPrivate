"""Validation script for evaluating trained dual encoder models.

This script:
1. Loads a trained model from checkpoint
2. Runs inference on validation dataset
3. Computes metrics (loss, alignment, hard negative similarity)
4. Performs sanity checks to verify model is learning
5. Outputs results as JSON

Usage:
    python -m scripts.validate \
        --checkpoint checkpoints/best.ckpt \
        --val-start 2020-01-01 \
        --val-end 2020-12-31 \
        --compute-silhouette
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader

from src.training.data_module import StockDataset
from src.training.module import DualEncoderModule


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Validate trained dual encoder model")

    # Checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    # Data arguments
    parser.add_argument(
        "--feature-dir",
        type=str,
        default="data/processed/features",
        help="Directory containing feature parquet files",
    )
    parser.add_argument(
        "--val-start",
        type=str,
        required=True,
        help="Validation start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--val-end",
        type=str,
        required=True,
        help="Validation end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="List of stock symbols to validate on (default: all available)",
    )

    # Inference arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default="validation_results.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--compute-silhouette",
        action="store_true",
        help="Compute sector silhouette score",
    )

    return parser.parse_args()


def compute_alignment_score(temporal_emb: torch.Tensor, tabular_emb: torch.Tensor) -> float:
    """Compute alignment score (mean positive similarity).

    This is the cosine similarity between temporal and tabular embeddings
    for the same stock (positive pairs).

    Args:
        temporal_emb: Temporal embeddings (batch, dim)
        tabular_emb: Tabular embeddings (batch, dim)

    Returns:
        Mean cosine similarity
    """
    similarities = torch.sum(temporal_emb * tabular_emb, dim=1)
    return similarities.mean().item()


def compute_hard_negative_similarity(
    temporal_emb: torch.Tensor, tabular_emb: torch.Tensor
) -> float:
    """Compute hard negative similarity (mean off-diagonal similarity).

    This measures how similar temporal embeddings are to tabular embeddings
    of different stocks (negative pairs).

    Args:
        temporal_emb: Temporal embeddings (batch, dim)
        tabular_emb: Tabular embeddings (batch, dim)

    Returns:
        Mean off-diagonal similarity
    """
    # Normalize embeddings
    temporal_norm = torch.nn.functional.normalize(temporal_emb, dim=1)
    tabular_norm = torch.nn.functional.normalize(tabular_emb, dim=1)

    # Compute all pairwise similarities (batch x batch)
    similarity_matrix = torch.mm(temporal_norm, tabular_norm.t())

    # Get off-diagonal elements (negative pairs)
    batch_size = similarity_matrix.size(0)
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=similarity_matrix.device)
    off_diagonal = similarity_matrix[mask]

    return off_diagonal.mean().item()


def compute_sector_silhouette(embeddings: np.ndarray, sector_labels: np.ndarray) -> float | None:
    """Compute silhouette score based on sector labels.

    Args:
        embeddings: Joint embeddings (num_samples, dim)
        sector_labels: Sector labels (num_samples,)

    Returns:
        Silhouette score or None if computation fails
    """
    if len(np.unique(sector_labels)) < 2:
        return None

    try:
        score = silhouette_score(embeddings, sector_labels)
        return float(score)
    except Exception:
        return None


@torch.no_grad()
def run_validation(
    model: DualEncoderModule,
    dataloader: DataLoader,
    compute_silhouette: bool = False,
) -> tuple[dict[str, Any], int]:
    """Run validation inference and compute metrics.

    Args:
        model: Loaded dual encoder model
        dataloader: Validation data loader
        compute_silhouette: Whether to compute silhouette score

    Returns:
        Tuple of (metrics dict, number of samples)
    """
    model.eval()
    device = next(model.parameters()).device

    all_losses = []
    all_alignments = []
    all_hard_negs = []

    all_embeddings = []
    all_sectors = []

    for _batch_idx, batch in enumerate(dataloader):
        # Move to device
        temporal = batch["temporal"].to(device)
        tabular_cont = batch["tabular_cont"].to(device)
        tabular_cat = batch["tabular_cat"].to(device)

        # Forward pass (train mode to get loss and embeddings)
        loss, temporal_emb, tabular_emb = model(
            temporal=temporal,
            tabular_cont=tabular_cont,
            tabular_cat=tabular_cat,
            mode="train",
        )

        # Collect metrics
        all_losses.append(loss.item())
        all_alignments.append(compute_alignment_score(temporal_emb, tabular_emb))
        all_hard_negs.append(compute_hard_negative_similarity(temporal_emb, tabular_emb))

        # Collect embeddings for silhouette if requested
        if compute_silhouette:
            joint_emb = model.model.get_joint_embedding(temporal, tabular_cont, tabular_cat)
            all_embeddings.append(joint_emb.cpu().numpy())
            all_sectors.append(batch["gsector"].cpu().numpy())

    # Compute aggregate metrics
    metrics = {
        "loss": {
            "mean": float(np.mean(all_losses)),
            "std": float(np.std(all_losses)),
            "min": float(np.min(all_losses)),
            "max": float(np.max(all_losses)),
        },
        "alignment": {
            "mean": float(np.mean(all_alignments)),
            "std": float(np.std(all_alignments)),
        },
        "hard_neg_similarity": {
            "mean": float(np.mean(all_hard_negs)),
            "std": float(np.std(all_hard_negs)),
        },
    }

    # Compute silhouette if requested
    if compute_silhouette and all_embeddings:
        embeddings = np.concatenate(all_embeddings, axis=0)
        sectors = np.concatenate(all_sectors, axis=0)
        silhouette = compute_sector_silhouette(embeddings, sectors)
        metrics["sector_silhouette"] = silhouette

    num_samples = len(dataloader.dataset)

    return metrics, num_samples


def perform_sanity_checks(metrics: dict[str, Any]) -> list[tuple[str, bool]]:
    """Perform sanity checks on validation metrics.

    Args:
        metrics: Computed metrics dictionary

    Returns:
        List of (check_name, passed) tuples
    """
    checks = []

    # Check 1: Loss is reasonable (< 5.0)
    loss_mean = metrics["loss"]["mean"]
    loss_check = loss_mean < 5.0
    checks.append((f"Loss is reasonable ({loss_mean:.3f} < 5.0)", loss_check))

    # Check 2: Alignment > Hard Neg Similarity (model is learning)
    alignment_mean = metrics["alignment"]["mean"]
    hard_neg_mean = metrics["hard_neg_similarity"]["mean"]
    alignment_check = alignment_mean > hard_neg_mean
    checks.append(
        (
            f"Alignment > Hard Neg ({alignment_mean:.3f} > {hard_neg_mean:.3f})",
            alignment_check,
        )
    )

    # Check 3: Hard Neg Similarity is low (< 0.5)
    hard_neg_check = hard_neg_mean < 0.5
    checks.append((f"Hard Neg similarity is low ({hard_neg_mean:.3f} < 0.5)", hard_neg_check))

    return checks


def print_results(metrics: dict[str, Any], checks: list[tuple[str, bool]], num_samples: int):
    """Print validation results and sanity checks.

    Args:
        metrics: Computed metrics
        checks: Sanity check results
        num_samples: Number of validation samples
    """
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)

    print(f"\nSamples processed: {num_samples}")

    print("\nMetrics:")
    print(f"  Loss: {metrics['loss']['mean']:.4f} ± {metrics['loss']['std']:.4f}")
    print(f"        (min: {metrics['loss']['min']:.4f}, max: {metrics['loss']['max']:.4f})")
    print(f"  Alignment: {metrics['alignment']['mean']:.4f} ± {metrics['alignment']['std']:.4f}")
    print(
        f"  Hard Neg Similarity: {metrics['hard_neg_similarity']['mean']:.4f} ± {metrics['hard_neg_similarity']['std']:.4f}"
    )

    if "sector_silhouette" in metrics and metrics["sector_silhouette"] is not None:
        print(f"  Sector Silhouette: {metrics['sector_silhouette']:.4f}")

    print("\nSanity Checks:")
    passed = 0
    failed = 0
    for check_name, check_result in checks:
        status = "✓" if check_result else "✗"
        print(f"  {status} {check_name}")
        if check_result:
            passed += 1
        else:
            failed += 1

    print(f"\nPassed: {passed}/{len(checks)}")
    if failed > 0:
        print(f"Failed: {failed}/{len(checks)}")

    print("=" * 60)


def main():
    """Main validation function."""
    args = parse_args()

    # Validate checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load model
    try:
        model = DualEncoderModule.load_from_checkpoint(str(checkpoint_path))
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Create validation dataset
    feature_dir = Path(args.feature_dir)
    if not feature_dir.exists():
        print(f"Error: Feature directory not found: {feature_dir}")
        sys.exit(1)

    print("\nCreating validation dataset...")
    print(f"  Feature directory: {feature_dir}")
    print(f"  Date range: {args.val_start} to {args.val_end}")

    val_dataset = StockDataset(
        feature_dir=str(feature_dir),
        date_range=(args.val_start, args.val_end),
        symbols=args.symbols,
        window_size=60,
    )

    print(f"  Samples: {len(val_dataset)}")

    if len(val_dataset) == 0:
        print("Error: No validation samples found")
        sys.exit(1)

    # Create dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Run validation
    print("\nRunning inference...")
    metrics, num_samples = run_validation(
        model=model,
        dataloader=val_loader,
        compute_silhouette=args.compute_silhouette,
    )

    # Perform sanity checks
    checks = perform_sanity_checks(metrics)

    # Print results
    print_results(metrics, checks, num_samples)

    # Prepare output
    results = {
        "checkpoint": str(checkpoint_path),
        "val_start": args.val_start,
        "val_end": args.val_end,
        "num_samples": num_samples,
        "metrics": metrics,
        "sanity_checks": {
            "total": len(checks),
            "passed": sum(1 for _, result in checks if result),
            "failed": sum(1 for _, result in checks if not result),
            "checks": [{"name": name, "passed": result} for name, result in checks],
        },
    }

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Exit with appropriate code
    failed_count = sum(1 for _, result in checks if not result)
    if failed_count > 0:
        print(f"\nWarning: {failed_count} sanity check(s) failed")
        sys.exit(0)  # Still exit 0 to not break pipelines, but warn

    print("\nAll sanity checks passed!")


if __name__ == "__main__":
    main()
