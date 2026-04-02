"""Clustering quality metrics for embedding evaluation.

All metrics are computed in PCA space, NOT UMAP space.
UMAP is for visualization only; it can create artificial structure.
"""

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def compute_silhouette_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score for clustering quality.

    Range: -1 to +1
    - > 0.5: Strong clustering
    - > 0.25: Meaningful clustering
    - ~0: No structure
    - < 0: Incorrect clustering

    Args:
        embeddings: Array of shape (n_samples, n_features)
        labels: Array of shape (n_samples,) with cluster labels

    Returns:
        Silhouette score as float
    """
    if len(np.unique(labels)) < 2:
        return 0.0

    return float(silhouette_score(embeddings, labels, metric="cosine"))


def compute_davies_bouldin_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute Davies-Bouldin index.

    Lower is better (more separated clusters).
    Measures cluster separation relative to cluster size.

    Args:
        embeddings: Array of shape (n_samples, n_features)
        labels: Array of shape (n_samples,) with cluster labels

    Returns:
        Davies-Bouldin index as float
    """
    if len(np.unique(labels)) < 2:
        return float("inf")

    return float(davies_bouldin_score(embeddings, labels))


def compute_calinski_harabasz_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute Calinski-Harabasz score.

    Higher is better (more separated clusters).
    Ratio of between-cluster dispersion to within-cluster dispersion.

    Args:
        embeddings: Array of shape (n_samples, n_features)
        labels: Array of shape (n_samples,) with cluster labels

    Returns:
        Calinski-Harabasz score as float
    """
    if len(np.unique(labels)) < 2:
        return 0.0

    return float(calinski_harabasz_score(embeddings, labels))


def compute_all_clustering_metrics(
    embeddings: np.ndarray,
    labels_dict: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute all clustering metrics for multiple label sets.

    Args:
        embeddings: Array of shape (n_samples, n_features)
        labels_dict: Dict mapping label name to label array

    Returns:
        Dict with metric names as keys and scores as values

    Example:
        >>> metrics = compute_all_clustering_metrics(
        ...     embeddings,
        ...     {"sector": sector_labels, "liquidity": liquidity_labels}
        ... )
    """
    metrics = {}

    for label_name, labels in labels_dict.items():
        metrics[f"silhouette_{label_name}"] = compute_silhouette_score(embeddings, labels)

        if label_name == "sector":  # Primary clustering of interest
            metrics[f"davies_bouldin_{label_name}"] = compute_davies_bouldin_score(
                embeddings, labels
            )
            metrics[f"calinski_harabasz_{label_name}"] = compute_calinski_harabasz_score(
                embeddings, labels
            )

    return metrics
