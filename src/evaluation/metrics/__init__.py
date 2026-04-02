"""Evaluation metrics for clustering and embedding quality."""

from src.evaluation.metrics.clustering import (
    compute_silhouette_score,
    compute_davies_bouldin_score,
    compute_calinski_harabasz_score,
    compute_all_clustering_metrics,
)
from src.evaluation.metrics.retrieval import (
    compute_candidate_intersection,
    filter_candidates_to_intersection,
    compute_recall_at_k,
    compute_ndcg_at_k,
    compute_spearman_correlation,
    sample_queries_deterministic,
    build_correlation_scores,
    correlation_to_ranking,
    normalize_scores,
    compute_hybrid_score,
    # Task 2 new functions
    filter_valid_candidates,
    recall_at_k,
    ndcg_at_k,
    spearman_against_reference,
    sample_query_set,
    build_correlation_scores_for_query,
    normalize_scores_minmax,
    build_hybrid_scores,
    build_snapshot_frame,
    # Task 3 placeholders
    prepare_model_inputs,
    compute_embedding_scores,
)

__all__ = [
    # Clustering metrics
    "compute_silhouette_score",
    "compute_davies_bouldin_score",
    "compute_calinski_harabasz_score",
    "compute_all_clustering_metrics",
    # Retrieval metrics
    "compute_candidate_intersection",
    "filter_candidates_to_intersection",
    "compute_recall_at_k",
    "compute_ndcg_at_k",
    "compute_spearman_correlation",
    "sample_queries_deterministic",
    "build_correlation_scores",
    "correlation_to_ranking",
    "normalize_scores",
    "compute_hybrid_score",
    # Task 2 new functions
    "filter_valid_candidates",
    "recall_at_k",
    "ndcg_at_k",
    "spearman_against_reference",
    "sample_query_set",
    "build_correlation_scores_for_query",
    "normalize_scores_minmax",
    "build_hybrid_scores",
    "build_snapshot_frame",
    # Task 3 placeholders
    "prepare_model_inputs",
    "compute_embedding_scores",
]
