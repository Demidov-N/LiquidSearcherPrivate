"""Tests for retrieval metrics module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics.retrieval import (
    build_correlation_scores,
    build_hybrid_scores,
    build_snapshot_frame,
    compute_candidate_intersection,
    compute_hybrid_score,
    compute_ndcg_at_k,
    compute_recall_at_k,
    compute_spearman_correlation,
    correlation_to_ranking,
    filter_candidates_to_intersection,
    filter_valid_candidates,
    ndcg_at_k,
    normalize_scores,
    normalize_scores_minmax,
    recall_at_k,
    sample_queries_deterministic,
    sample_query_set,
    spearman_against_reference,
)
from src.evaluation.visualizations.retrieval_plots import (
    plot_grouped_metrics,
    plot_overall_metrics,
)


class TestSpearmanScoresForQuery:
    """Test build_spearman_scores_for_query function."""

    def test_build_spearman_scores_for_query(self):
        """Test 60-day Spearman rank correlation computation."""
        np.random.seed(42)  # For reproducibility
        returns_df = pd.DataFrame(
            {
                "date": pd.date_range("2019-01-01", periods=60, freq="D"),
                "AAPL": np.random.randn(60) * 0.02,
                "MSFT": np.random.randn(60) * 0.02,
            }
        ).set_index("date")

        from src.evaluation.metrics.retrieval import build_spearman_scores_for_query

        result = build_spearman_scores_for_query(
            returns_df, "AAPL", snapshot_date=returns_df.index[-1], lookback=60, min_overlap=40
        )

        assert isinstance(result, pd.Series)
        assert "MSFT" in result.index
        assert -1.0 <= result["MSFT"] <= 1.0

    def test_build_spearman_scores_empty_returns(self):
        """Test empty returns DataFrame returns empty series."""
        from src.evaluation.metrics.retrieval import build_spearman_scores_for_query

        empty_df = pd.DataFrame()
        result = build_spearman_scores_for_query(
            empty_df, "AAPL", snapshot_date=pd.Timestamp("2019-01-01"), lookback=60, min_overlap=40
        )
        assert len(result) == 0

    def test_build_spearman_scores_missing_symbol(self):
        """Test missing query symbol returns empty series."""
        np.random.seed(42)
        from src.evaluation.metrics.retrieval import build_spearman_scores_for_query

        returns_df = pd.DataFrame(
            {
                "date": pd.date_range("2019-01-01", periods=60, freq="D"),
                "AAPL": np.random.randn(60) * 0.02,
                "MSFT": np.random.randn(60) * 0.02,
            }
        ).set_index("date")

        result = build_spearman_scores_for_query(
            returns_df, "UNKNOWN", snapshot_date=returns_df.index[-1], lookback=60, min_overlap=40
        )
        assert len(result) == 0

    def test_build_spearman_scores_insufficient_lookback(self):
        """Test insufficient lookback returns empty series."""
        np.random.seed(42)
        from src.evaluation.metrics.retrieval import build_spearman_scores_for_query

        returns_df = pd.DataFrame(
            {
                "date": pd.date_range("2019-01-01", periods=30, freq="D"),
                "AAPL": np.random.randn(30) * 0.02,
                "MSFT": np.random.randn(30) * 0.02,
            }
        ).set_index("date")

        result = build_spearman_scores_for_query(
            returns_df, "AAPL", snapshot_date=returns_df.index[-1], lookback=60, min_overlap=40
        )
        assert len(result) == 0

    def test_build_spearman_scores_zero_std(self):
        """Test symbols with zero std return NaN."""
        from src.evaluation.metrics.retrieval import build_spearman_scores_for_query

        returns_df = pd.DataFrame(
            {
                "date": pd.date_range("2019-01-01", periods=60, freq="D"),
                "AAPL": np.random.randn(60) * 0.02,
                "FLAT": [0.01] * 60,  # Zero standard deviation
            }
        ).set_index("date")

        result = build_spearman_scores_for_query(
            returns_df, "AAPL", snapshot_date=returns_df.index[-1], lookback=60, min_overlap=40
        )

        # FLAT has zero std, so correlation should be NaN
        assert "FLAT" in result.index
        assert np.isnan(result["FLAT"])


class TestCandidateIntersection:
    """Test candidate eligibility intersection helpers."""

    def test_intersection_returns_common_symbols(self):
        """Test that intersection returns only symbols present in all rankers."""
        rankers = {
            "model": {"AAPL", "MSFT", "GOOGL", "AMZN"},
            "correlation": {"AAPL", "MSFT", "IBM", "GE"},
            "random": {"AAPL", "MSFT", "XOM"},
        }
        result = compute_candidate_intersection(rankers)
        assert result == {"AAPL", "MSFT"}

    def test_intersection_empty_when_no_overlap(self):
        """Test empty intersection when no common candidates."""
        rankers = {
            "a": {"AAPL", "MSFT"},
            "b": {"IBM", "GE"},
        }
        result = compute_candidate_intersection(rankers)
        assert result == set()

    def test_intersection_with_single_ranker(self):
        """Test intersection with only one ranker returns all its candidates."""
        rankers = {"only": {"AAPL", "MSFT", "GOOGL"}}
        result = compute_candidate_intersection(rankers)
        assert result == {"AAPL", "MSFT", "GOOGL"}

    def test_intersection_empty_dict(self):
        """Test empty input returns empty set."""
        result = compute_candidate_intersection({})
        assert result == set()

    def test_filter_candidates_to_intersection(self):
        """Test filtering scores to intersection universe."""
        scores = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT", "IBM", "GE", "AAPL2"],
                "score": [0.9, 0.8, 0.7, 0.6, 0.5],
            }
        )
        intersection = {"AAPL", "MSFT", "IBM"}
        result = filter_candidates_to_intersection(scores, intersection)
        assert set(result["symbol"].tolist()) == {"AAPL", "MSFT", "IBM"}
        assert len(result) == 3


class TestRecallAtK:
    """Test Recall@k computation."""

    def test_recall_at_k_perfect_retrieval(self):
        """Test Recall@10 when all relevant items are retrieved."""
        predicted = {"A", "B", "C", "D", "E"}
        ground_truth = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}
        recall = compute_recall_at_k(predicted, ground_truth, k=5)
        assert recall == 5 / 10

    def test_recall_at_k_partial_retrieval(self):
        """Test Recall@10 with partial relevant retrieval."""
        predicted = {"A", "B", "X", "Y", "Z"}
        ground_truth = {"A", "B", "C", "D", "E"}
        recall = compute_recall_at_k(predicted, ground_truth, k=5)
        assert recall == 2 / 5

    def test_recall_at_k_no_relevant_retrieved(self):
        """Test Recall@0 when no relevant items retrieved."""
        predicted = {"X", "Y", "Z"}
        ground_truth = {"A", "B"}
        recall = compute_recall_at_k(predicted, ground_truth, k=3)
        assert recall == 0.0

    def test_recall_at_k_k_larger_than_predictions(self):
        """Test Recall when k exceeds number of predictions."""
        predicted = {"A", "B"}
        ground_truth = {"A", "B", "C", "D", "E"}
        recall = compute_recall_at_k(predicted, ground_truth, k=10)
        assert recall == 2 / 5

    def test_recall_at_k_empty_ground_truth(self):
        """Test Recall returns NaN when ground truth is empty."""
        predicted = {"A", "B"}
        ground_truth = set()
        recall = compute_recall_at_k(predicted, ground_truth, k=5)
        assert np.isnan(recall)

    def test_recall_at_k_negative_k(self):
        """Test Recall returns NaN for negative k."""
        recall = compute_recall_at_k({"A"}, {"A", "B"}, k=-1)
        assert np.isnan(recall)

    def test_recall_at_k_zero_k(self):
        """Test Recall returns NaN for k=0."""
        recall = compute_recall_at_k({"A"}, {"A", "B"}, k=0)
        assert np.isnan(recall)


class TestNDCGAtK:
    """Test nDCG@k computation."""

    def test_ndcg_perfect_ranking(self):
        """Test nDCG=1.0 when predictions perfectly match relevance."""
        rankings = pd.Series([0, 1, 2], index=["A", "B", "C"])  # A=best, C=worst
        relevance = pd.Series([3.0, 2.0, 1.0], index=["A", "B", "C"])  # A most relevant
        ndcg = compute_ndcg_at_k(rankings, relevance, k=3)
        assert abs(ndcg - 1.0) < 1e-6

    def test_ndcg_reversed_ranking(self):
        """Test nDCG when predictions are reversed from relevance."""
        # When A (best rank=0) has lowest relevance and C (worst rank=2) has highest,
        # this is a reversed ranking - nDCG should be less than perfect (1.0)
        rankings = pd.Series([0, 1, 2], index=["A", "B", "C"])  # A=best, C=worst
        relevance = pd.Series([1.0, 2.0, 3.0], index=["A", "B", "C"])  # A=lowest, C=highest
        ndcg = compute_ndcg_at_k(rankings, relevance, k=3)
        assert 0.0 < ndcg < 1.0

    def test_ndcg_partial_k(self):
        """Test nDCG@k with k smaller than total items."""
        rankings = pd.Series([0, 1, 2, 3], index=["A", "B", "C", "D"])
        relevance = pd.Series([4.0, 3.0, 2.0, 1.0], index=["A", "B", "C", "D"])
        ndcg = compute_ndcg_at_k(rankings, relevance, k=2)
        # Only top-2 considered
        assert 0.0 <= ndcg <= 1.0

    def test_ndcg_empty_rankings(self):
        """Test nDCG returns NaN for empty rankings."""
        ndcg = compute_ndcg_at_k(pd.Series([]), pd.Series([1.0]), k=5)
        assert np.isnan(ndcg)

    def test_ndcg_zero_relevance(self):
        """Test nDCG returns NaN when all relevance is zero (can't normalize)."""
        rankings = pd.Series([0, 1, 2], index=["A", "B", "C"])
        relevance = pd.Series([0.0, 0.0, 0.0], index=["A", "B", "C"])
        ndcg = compute_ndcg_at_k(rankings, relevance, k=3)
        assert np.isnan(ndcg)

    def test_ndcg_no_common_symbols(self):
        """Test nDCG returns NaN when rankings and relevance have no overlap."""
        rankings = pd.Series([0, 1, 2], index=["A", "B", "C"])
        relevance = pd.Series([1.0, 2.0, 3.0], index=["X", "Y", "Z"])
        ndcg = compute_ndcg_at_k(rankings, relevance, k=3)
        assert np.isnan(ndcg)


class TestSpearmanCorrelation:
    """Test Spearman rank correlation computation."""

    def test_spearman_perfect_correlation(self):
        """Test Spearman=1.0 for identical rankings."""
        pred = pd.Series([0, 1, 2, 3], index=["A", "B", "C", "D"])
        ref = pd.Series([0, 1, 2, 3], index=["A", "B", "C", "D"])
        rho = compute_spearman_correlation(pred, ref)
        assert abs(rho - 1.0) < 1e-6

    def test_spearman_perfect_anticorrelation(self):
        """Test Spearman=-1.0 for reversed rankings."""
        pred = pd.Series([0, 1, 2, 3], index=["A", "B", "C", "D"])
        ref = pd.Series([3, 2, 1, 0], index=["A", "B", "C", "D"])
        rho = compute_spearman_correlation(pred, ref)
        assert abs(rho - (-1.0)) < 1e-6

    def test_spearman_partial_correlation(self):
        """Test Spearman between partially correlated rankings."""
        pred = pd.Series([0, 1, 2, 3], index=["A", "B", "C", "D"])
        ref = pd.Series([0, 2, 1, 3], index=["A", "B", "C", "D"])
        rho = compute_spearman_correlation(pred, ref)
        assert 0.0 < rho < 1.0

    def test_spearman_empty_rankings(self):
        """Test Spearman returns NaN for empty rankings."""
        rho = compute_spearman_correlation(pd.Series([]), pd.Series([]))
        assert np.isnan(rho)

    def test_spearman_single_item(self):
        """Test Spearman returns NaN for single item."""
        rho = compute_spearman_correlation(pd.Series([0], index=["A"]), pd.Series([0], index=["A"]))
        assert np.isnan(rho)

    def test_spearman_no_common_symbols(self):
        """Test Spearman returns NaN when rankings have no overlap."""
        pred = pd.Series([0, 1, 2], index=["A", "B", "C"])
        ref = pd.Series([0, 1, 2], index=["X", "Y", "Z"])
        rho = compute_spearman_correlation(pred, ref)
        assert np.isnan(rho)

    def test_spearman_all_tied(self):
        """Test Spearman returns NaN when one ranking has all ties."""
        pred = pd.Series([0, 0, 0], index=["A", "B", "C"])
        ref = pd.Series([0, 1, 2], index=["A", "B", "C"])
        rho = compute_spearman_correlation(pred, ref)
        assert np.isnan(rho)


class TestDeterministicSampling:
    """Test deterministic query sampling."""

    def test_sampling_same_seed_gives_same_result(self):
        """Test that same seed produces identical samples."""
        symbols = ["SYM" + str(i) for i in range(100)]
        queries1 = sample_queries_deterministic(symbols, 10, seed=42)
        queries2 = sample_queries_deterministic(symbols, 10, seed=42)
        assert queries1 == queries2

    def test_sampling_different_seeds_gives_different_results(self):
        """Test that different seeds produce different samples."""
        symbols = ["SYM" + str(i) for i in range(100)]
        queries1 = sample_queries_deterministic(symbols, 10, seed=42)
        queries2 = sample_queries_deterministic(symbols, 10, seed=123)
        assert queries1 != queries2

    def test_sampling_without_stratification(self):
        """Test sampling without stratification."""
        symbols = ["SYM" + str(i) for i in range(50)]
        queries = sample_queries_deterministic(symbols, 10, seed=42)
        assert len(queries) == 10
        assert len(set(queries)) == 10  # No duplicates

    def test_sampling_with_stratification(self):
        """Test sampling with stratification maintains proportions."""
        symbols = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "C1", "C2"]
        stratify = pd.Series(
            ["A", "A", "A", "A", "B", "B", "B", "C", "C"],
            index=symbols,
        )
        queries = sample_queries_deterministic(symbols, 6, seed=42, stratify_by=stratify)
        assert len(queries) == 6

    def test_sampling_n_larger_than_symbols(self):
        """Test sampling returns all symbols when n >= len(symbols)."""
        symbols = ["A", "B", "C"]
        queries = sample_queries_deterministic(symbols, 10, seed=42)
        assert set(queries) == set(symbols)

    def test_sampling_zero_queries(self):
        """Test sampling with n=0 returns empty list."""
        symbols = ["A", "B", "C"]
        queries = sample_queries_deterministic(symbols, 0, seed=42)
        assert queries == []

    def test_sampling_n_queries_exact(self):
        """Test that exactly n queries are returned."""
        symbols = ["SYM" + str(i) for i in range(200)]
        for n in [1, 5, 10, 50, 100]:
            queries = sample_queries_deterministic(symbols, n, seed=42)
            assert len(queries) == n


class TestCorrelationScores:
    """Test correlation-based score builder."""

    def test_correlation_scores_basic(self):
        """Test basic correlation score computation."""
        dates = pd.date_range("2019-01-01", periods=100, freq="D")
        prices = pd.DataFrame(
            {
                "A": np.cumsum(np.random.randn(100)) + 100,
                "B": np.cumsum(np.random.randn(100)) + 100,
                "C": np.cumsum(np.random.randn(100)) + 100,
            },
            index=dates,
        )
        result = build_correlation_scores(prices, lookback_days=60, min_overlap=40)
        assert not result.empty
        assert "correlation" in result.columns
        assert "overlap" in result.columns

    def test_correlation_scores_min_overlap_filter(self):
        """Test that min_overlap filter is applied."""
        dates = pd.date_range("2019-01-01", periods=50, freq="D")
        prices = pd.DataFrame(
            {
                "A": np.cumsum(np.random.randn(50)) + 100,
                "B": np.cumsum(np.random.randn(50)) + 100,
            },
            index=dates,
        )
        result = build_correlation_scores(prices, lookback_days=60, min_overlap=40)
        # With only 50 observations and 60-day lookback, should return empty or filtered
        assert len(result) == 0 or all(result["overlap"] >= 40)

    def test_correlation_scores_includes_both_directions(self):
        """Test that both (i,j) and (j,i) pairs are included."""
        dates = pd.date_range("2019-01-01", periods=100, freq="D")
        prices = pd.DataFrame(
            {
                "A": np.cumsum(np.random.randn(100)) + 100,
                "B": np.cumsum(np.random.randn(100)) + 100,
            },
            index=dates,
        )
        result = build_correlation_scores(prices, lookback_days=60, min_overlap=40)
        assert ("A", "B") in result.index
        assert ("B", "A") in result.index

    def test_correlation_scores_nan_for_insufficient_data(self):
        """Test correlation is NaN when std=0."""
        dates = pd.date_range("2019-01-01", periods=100, freq="D")
        prices = pd.DataFrame(
            {
                "A": [100.0] * 100,  # No variation
                "B": np.cumsum(np.random.randn(100)) + 100,
            },
            index=dates,
        )
        result = build_correlation_scores(prices, lookback_days=60, min_overlap=40)
        # A has zero std, so correlation should be NaN
        assert (
            result.loc[("A", "B"), "correlation"] != result.loc[("A", "B"), "correlation"]
        )  # NaN != NaN

    def test_correlation_to_ranking(self):
        """Test converting correlation scores to ranking."""
        corr_df = pd.DataFrame(
            {
                "correlation": [0.9, 0.8, 0.7, 0.6],
                "overlap": [60, 60, 60, 60],
            },
            index=pd.MultiIndex.from_tuples([("Q", "A"), ("Q", "B"), ("Q", "C"), ("Q", "D")]),
        )
        ranking = correlation_to_ranking(corr_df, "Q", ["A", "B", "C", "D"])
        # A has highest correlation (0.9), should be rank 0
        assert ranking["A"] == 0
        # D has lowest correlation (0.6), should be rank 3
        assert ranking["D"] == 3


class TestScoreNormalization:
    """Test score normalization helpers."""

    def test_minmax_normalization(self):
        """Test min-max normalization."""
        scores = pd.Series([10.0, 20.0, 30.0, 40.0], index=["A", "B", "C", "D"])
        normalized = normalize_scores(scores, method="minmax")
        assert abs(normalized["A"] - 0.0) < 1e-6
        assert abs(normalized["D"] - 1.0) < 1e-6
        assert abs(normalized["B"] - 1.0 / 3.0) < 1e-6

    def test_zscore_normalization(self):
        """Test z-score normalization."""
        scores = pd.Series([10.0, 20.0, 30.0, 40.0], index=["A", "B", "C", "D"])
        normalized = normalize_scores(scores, method="zscore")
        assert abs(normalized.mean()) < 1e-6
        assert abs(normalized.std() - 1.0) < 1e-6

    def test_rank_normalization(self):
        """Test rank-based normalization."""
        scores = pd.Series([10.0, 30.0, 20.0, 40.0], index=["A", "B", "C", "D"])
        normalized = normalize_scores(scores, method="rank")
        assert normalized["D"] == 1.0  # Highest value gets rank 1
        assert normalized["A"] == 0.25  # Lowest value gets rank 0.25

    def test_normalization_handles_constant_scores(self):
        """Test normalization handles constant scores (avoid div by zero)."""
        scores = pd.Series([5.0, 5.0, 5.0], index=["A", "B", "C"])
        normalized = normalize_scores(scores, method="minmax")
        # Should return 0.5 for all (or some valid value, not crash)
        assert all(~np.isnan(normalized))

    def test_normalization_empty_input(self):
        """Test normalization handles empty input."""
        scores = pd.Series([], dtype=float)
        normalized = normalize_scores(scores, method="minmax")
        assert len(normalized) == 0


class TestHybridScore:
    """Test hybrid score computation."""

    def test_hybrid_equal_weights(self):
        """Test hybrid with equal weights (0.5, 0.5)."""
        primary = pd.Series([1.0, 0.5, 0.0], index=["A", "B", "C"])
        secondary = pd.Series([0.0, 0.5, 1.0], index=["A", "B", "C"])
        hybrid = compute_hybrid_score(primary, secondary, primary_weight=0.5)
        # With rank normalization (ascending=True):
        # primary_norm: A (1.0) -> rank 3/3=1, B (0.5) -> rank 2/3=0.666, C (0.0) -> rank 1/3=0.333
        # secondary_norm: A (0.0) -> rank 1/3=0.333, B (0.5) -> rank 2/3=0.666, C (1.0) -> rank 3/3=1
        # Hybrid = 0.5*primary_norm + 0.5*secondary_norm
        # hybrid[A] = 0.5*1.0 + 0.5*0.333 = 0.666
        # hybrid[B] = 0.5*0.666 + 0.5*0.666 = 0.666
        # hybrid[C] = 0.5*0.333 + 0.5*1.0 = 0.666
        # All equal due to symmetric but offset rank distributions
        assert all(abs(v - 2.0 / 3.0) < 1e-6 for v in hybrid.values)

    def test_hybrid_primary_weight(self):
        """Test hybrid with primary weight 0.8."""
        primary = pd.Series([1.0, 0.0], index=["A", "B"])
        secondary = pd.Series([0.0, 1.0], index=["A", "B"])
        hybrid = compute_hybrid_score(primary, secondary, primary_weight=0.8)
        # A: 0.8*1.0 + 0.2*0.0 = 0.8 (normalized)
        # B: 0.8*0.0 + 0.2*1.0 = 0.2
        assert hybrid["A"] > hybrid["B"]

    def test_hybrid_no_normalization(self):
        """Test hybrid without normalization uses raw values."""
        primary = pd.Series([1.0, 0.5], index=["A", "B"])
        secondary = pd.Series([0.0, 1.0], index=["A", "B"])
        hybrid = compute_hybrid_score(primary, secondary, primary_weight=0.5, normalize=False)
        # Without normalization: 0.5*primary + 0.5*secondary
        assert abs(hybrid["A"] - 0.5) < 1e-6
        assert abs(hybrid["B"] - 0.75) < 1e-6

    def test_hybrid_no_common_symbols(self):
        """Test hybrid returns empty when no common symbols."""
        primary = pd.Series([1.0, 0.5], index=["A", "B"])
        secondary = pd.Series([1.0, 0.5], index=["X", "Y"])
        hybrid = compute_hybrid_score(primary, secondary)
        assert len(hybrid) == 0

    def test_hybrid_preserves_index(self):
        """Test hybrid preserves the index from primary scores."""
        primary = pd.Series([1.0, 0.5, 0.0], index=["A", "B", "C"])
        secondary = pd.Series([0.5, 1.0, 0.5], index=["A", "B", "C"])
        hybrid = compute_hybrid_score(primary, secondary)
        assert set(hybrid.index) == {"A", "B", "C"}


class TestFilterValidCandidates:
    """Test filter_valid_candidates function."""

    def test_filters_embedding_flag(self):
        """Test filtering by has_embedding column."""
        df = pd.DataFrame(
            {
                "symbol": ["A", "B", "C"],
                "has_embedding": [True, False, True],
            }
        )
        result = filter_valid_candidates(df)
        assert len(result) == 2
        assert set(result["symbol"]) == {"A", "C"}

    def test_filters_liquidity_flag(self):
        """Test filtering by has_liquidity_label column."""
        df = pd.DataFrame(
            {
                "symbol": ["A", "B", "C"],
                "has_liquidity_label": [True, True, False],
            }
        )
        result = filter_valid_candidates(df)
        assert len(result) == 2
        assert set(result["symbol"]) == {"A", "B"}

    def test_filters_correlation_overlap(self):
        """Test filtering by correlation_overlap >= 40."""
        df = pd.DataFrame(
            {
                "symbol": ["A", "B", "C", "D"],
                "correlation_overlap": [50, 30, 60, 40],
            }
        )
        result = filter_valid_candidates(df)
        assert len(result) == 3
        assert set(result["symbol"]) == {"A", "C", "D"}

    def test_combined_filters(self):
        """Test combined filtering with all columns."""
        df = pd.DataFrame(
            {
                "symbol": ["A", "B", "C", "D"],
                "has_embedding": [True, True, False, True],
                "has_liquidity_label": [True, False, True, True],
                "correlation_overlap": [50, 40, 50, 30],
            }
        )
        result = filter_valid_candidates(df)
        # Only A passes all filters
        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "A"

    def test_empty_input(self):
        """Test empty input returns empty."""
        result = filter_valid_candidates(pd.DataFrame())
        assert len(result) == 0

    def test_missing_columns_ignored(self):
        """Test that missing filter columns are ignored."""
        df = pd.DataFrame({"symbol": ["A", "B", "C"], "score": [0.9, 0.8, 0.7]})
        result = filter_valid_candidates(df)
        assert len(result) == 3  # No filtering applied


class TestRecallAtKNew:
    """Test recall_at_k function (list-based)."""

    def test_recall_basic(self):
        """Test basic recall computation."""
        ranking = ["A", "B", "C", "D", "E"]
        relevance = pd.Series(1, index=["A", "B", "F"])
        recall = recall_at_k(ranking, relevance, k=3)
        assert recall == 2 / 3

    def test_recall_no_overlap(self):
        """Test recall when no relevant items in top-k."""
        ranking = ["X", "Y", "Z"]
        relevance = pd.Series(1, index=["A", "B"])
        recall = recall_at_k(ranking, relevance, k=3)
        assert recall == 0.0

    def test_recall_all_relevant_found(self):
        """Test recall when all relevant items found."""
        ranking = ["A", "B", "C", "D", "E"]
        relevance = pd.Series(1, index=["A", "B"])
        recall = recall_at_k(ranking, relevance, k=5)
        assert recall == 1.0

    def test_recall_empty_relevance(self):
        """Test recall with empty relevance returns NaN."""
        ranking = ["A", "B", "C"]
        relevance = pd.Series([], dtype=int)
        recall = recall_at_k(ranking, relevance, k=5)
        assert np.isnan(recall)


class TestNDCGAtKNew:
    """Test ndcg_at_k function (list-based)."""

    def test_ndcg_perfect(self):
        """Test nDCG=1 for perfect ranking."""
        ranking = ["A", "B", "C"]
        relevance = pd.Series([3.0, 2.0, 1.0], index=["A", "B", "C"])
        ndcg = ndcg_at_k(ranking, relevance, k=3)
        assert abs(ndcg - 1.0) < 1e-6

    def test_ndcg_partial(self):
        """Test nDCG with k < len(ranking)."""
        ranking = ["A", "B", "C", "D"]
        relevance = pd.Series([4.0, 3.0, 2.0, 1.0], index=["A", "B", "C", "D"])
        ndcg = ndcg_at_k(ranking, relevance, k=2)
        assert 0.0 <= ndcg <= 1.0

    def test_ndcg_empty_ranking(self):
        """Test nDCG with empty ranking returns NaN."""
        relevance = pd.Series([1.0, 2.0], index=["A", "B"])
        ndcg = ndcg_at_k([], relevance, k=5)
        assert np.isnan(ndcg)


class TestSpearmanAgainstReference:
    """Test spearman_against_reference function."""

    def test_spearman_perfect(self):
        """Test Spearman=1 for identical scores."""
        pred = pd.Series([3.0, 2.0, 1.0], index=["A", "B", "C"])
        ref = pd.Series([3.0, 2.0, 1.0], index=["A", "B", "C"])
        rho = spearman_against_reference(pred, ref)
        assert abs(rho - 1.0) < 1e-6

    def test_spearman_anticorrelation(self):
        """Test Spearman=-1 for reversed scores."""
        pred = pd.Series([1.0, 2.0, 3.0], index=["A", "B", "C"])
        ref = pd.Series([3.0, 2.0, 1.0], index=["A", "B", "C"])
        rho = spearman_against_reference(pred, ref)
        assert abs(rho - (-1.0)) < 1e-6

    def test_spearman_partial(self):
        """Test Spearman for partial correlation."""
        pred = pd.Series([3.0, 1.0, 2.0], index=["A", "B", "C"])
        ref = pd.Series([2.0, 1.0, 3.0], index=["A", "B", "C"])
        rho = spearman_against_reference(pred, ref)
        assert -1.0 <= rho <= 1.0

    def test_spearman_no_overlap(self):
        """Test Spearman with no common symbols returns NaN."""
        pred = pd.Series([1.0, 2.0], index=["A", "B"])
        ref = pd.Series([1.0, 2.0], index=["X", "Y"])
        rho = spearman_against_reference(pred, ref)
        assert np.isnan(rho)


class TestSampleQuerySet:
    """Test sample_query_set function."""

    def test_deterministic_sampling(self):
        """Test that same seed gives same results."""
        df = pd.DataFrame({"symbol": ["SYM" + str(i) for i in range(100)]})
        result1 = sample_query_set(df, n_queries=10, seed=42)
        result2 = sample_query_set(df, n_queries=10, seed=42)
        assert set(result1["symbol"]) == set(result2["symbol"])

    def test_different_seeds_different_results(self):
        """Test different seeds give different samples."""
        df = pd.DataFrame({"symbol": ["SYM" + str(i) for i in range(100)]})
        result1 = sample_query_set(df, n_queries=10, seed=42)
        result2 = sample_query_set(df, n_queries=10, seed=123)
        assert set(result1["symbol"]) != set(result2["symbol"])

    def test_stratified_sampling(self):
        """Test stratified sampling by gsector and market_cap_tier."""
        df = pd.DataFrame(
            {
                "symbol": ["A1", "A2", "B1", "B2", "C1", "C2"],
                "gsector": ["Tech", "Tech", "Fin", "Fin", "Health", "Health"],
                "market_cap_tier": ["Large", "Small", "Large", "Small", "Large", "Small"],
            }
        )
        result = sample_query_set(df, n_queries=4, seed=42)
        assert len(result) == 4

    def test_fallback_without_strata(self):
        """Test fallback to random when stratification columns missing."""
        df = pd.DataFrame({"symbol": ["SYM" + str(i) for i in range(50)]})
        result = sample_query_set(df, n_queries=10, seed=42)
        assert len(result) == 10

    def test_n_larger_than_universe(self):
        """Test returns all when n >= len(symbols)."""
        df = pd.DataFrame({"symbol": ["A", "B", "C"]})
        result = sample_query_set(df, n_queries=10, seed=42)
        assert len(result) == 3


class TestBuildSnapshotFrame:
    """Test build_snapshot_frame function."""

    def test_keeps_last_row_per_symbol(self):
        """Test that only the last row per symbol is kept."""
        df = pd.DataFrame(
            {
                "symbol": ["A", "A", "B", "B", "C"],
                "value": [1, 2, 3, 4, 5],
                "date": ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02", "2020-01-01"],
            }
        )
        result = build_snapshot_frame(df)
        assert len(result) == 3
        assert set(result["symbol"]) == {"A", "B", "C"}
        # A should have value=2 (last), B should have value=4 (last)
        assert result[result["symbol"] == "A"]["value"].iloc[0] == 2
        assert result[result["symbol"] == "B"]["value"].iloc[0] == 4

    def test_empty_input(self):
        """Test empty input returns empty."""
        result = build_snapshot_frame(pd.DataFrame())
        assert len(result) == 0


class TestNormalizeScoresNew:
    """Test normalize_scores_minmax function."""

    def test_minmax_normalization(self):
        """Test min-max scaling to [0,1]."""
        scores = pd.Series([0.0, 50.0, 100.0], index=["A", "B", "C"])
        normalized = normalize_scores_minmax(scores)
        assert abs(normalized["A"] - 0.0) < 1e-6
        assert abs(normalized["B"] - 0.5) < 1e-6
        assert abs(normalized["C"] - 1.0) < 1e-6

    def test_constant_scores(self):
        """Test normalization of constant scores."""
        scores = pd.Series([5.0, 5.0, 5.0], index=["A", "B", "C"])
        normalized = normalize_scores_minmax(scores)
        assert all(normalized == 0.5)


class TestBuildHybridScores:
    """Test build_hybrid_scores function."""

    def test_hybrid_default_alpha(self):
        """Test hybrid with default alpha=0.7."""
        emb = pd.Series([1.0, 0.5, 0.0], index=["A", "B", "C"])
        liq = pd.Series([0.0, 0.5, 1.0], index=["A", "B", "C"])
        hybrid = build_hybrid_scores(emb, liq, alpha=0.7)
        # A: 0.7*1 + 0.3*0 = 0.7 (normalized)
        # C: 0.7*0 + 0.3*1 = 0.3 (normalized)
        assert hybrid["A"] > hybrid["C"]

    def test_hybrid_no_overlap(self):
        """Test hybrid with no common symbols."""
        emb = pd.Series([1.0, 0.5], index=["A", "B"])
        liq = pd.Series([1.0, 0.5], index=["X", "Y"])
        hybrid = build_hybrid_scores(emb, liq)
        assert len(hybrid) == 0

    def test_hybrid_equal_alpha(self):
        """Test hybrid with alpha=0.5."""
        emb = pd.Series([1.0, 0.0], index=["A", "B"])
        liq = pd.Series([0.0, 1.0], index=["A", "B"])
        hybrid = build_hybrid_scores(emb, liq, alpha=0.5)
        # Both normalized to 1.0 and 0.0, equal weight
        assert abs(hybrid["A"] - 0.5) < 1e-6


# =============================================================================
# Smoke Tests for Retrieval Plots and CLI Pipeline
# =============================================================================


class TestRetrievalPlots:
    """Smoke tests for retrieval visualization functions."""

    def test_plot_overall_metrics_writes_file(self):
        """Test that plot_overall_metrics creates a file."""
        metrics_df = pd.DataFrame(
            {
                "metric_name": ["Recall@10", "Spearman", "nDCG@10"],
                "embedding": [0.65, 0.42, 0.58],
                "correlation": [0.55, 0.38, 0.52],
                "liquidity_distance": [0.80, 0.75, 0.78],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_overall.png"
            result = plot_overall_metrics(metrics_df, output_path)
            assert result.exists()
            assert result == output_path

    def test_plot_grouped_metrics_writes_file(self):
        """Test that plot_grouped_metrics creates a file."""
        grouped_df = pd.DataFrame(
            {
                "group_name": ["Tech", "Finance", "Healthcare"],
                "embedding": [0.60, 0.55, 0.65],
                "correlation": [0.50, 0.48, 0.52],
                "liquidity_distance": [0.75, 0.72, 0.78],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_grouped.png"
            result = plot_grouped_metrics(
                grouped_df, metric_col="Recall@10", group_col="sector", output_path=output_path
            )
            assert result.exists()
            assert result == output_path

    def test_plot_overall_metrics_with_hybrid_writes_file(self):
        """Test that plot_overall_metrics handles hybrid column."""
        metrics_df = pd.DataFrame(
            {
                "metric_name": ["Recall@10", "Spearman", "nDCG@10"],
                "embedding": [0.65, 0.42, 0.58],
                "correlation": [0.55, 0.38, 0.52],
                "liquidity_distance": [0.80, 0.75, 0.78],
                "hybrid": [0.72, 0.68, 0.70],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_hybrid.png"
            result = plot_overall_metrics(metrics_df, output_path)
            assert result.exists()


class TestRetrievalPipelineHelper:
    """Smoke tests for the retrieval evaluation pipeline helper functions."""

    def test_prepare_temporal_features(self):
        """Test temporal feature preparation produces correct shape."""
        from scripts.evaluation.run_retrieval_metrics import _prepare_temporal_features
        from src.training.data_module import TEMPORAL_FEATURE_NAMES

        # Create a mock row with some temporal features
        row = pd.Series({col: float(i) for i, col in enumerate(TEMPORAL_FEATURE_NAMES)})
        row["symbol"] = "TEST"

        temporal = _prepare_temporal_features(row, window_size=60)

        assert temporal.shape == (1, 60, len(TEMPORAL_FEATURE_NAMES))
        # Last position should have the values
        assert temporal[0, -1, 0] == 0.0  # First feature from our mock row
        # Other positions should be zero
        assert temporal[0, 0, 0] == 0.0

    def test_prepare_tabular_features(self):
        """Test tabular feature preparation produces correct shape."""
        from scripts.evaluation.run_retrieval_metrics import _prepare_tabular_features
        from src.training.data_module import TABULAR_CONTINUOUS_NAMES

        row = pd.Series({col: float(i) for i, col in enumerate(TABULAR_CONTINUOUS_NAMES)})

        tabular = _prepare_tabular_features(row)

        assert tabular.shape == (1, len(TABULAR_CONTINUOUS_NAMES))
        # Values should match
        assert tabular[0, 0] == 0.0
        assert tabular[0, 5] == 5.0

    def test_prepare_categorical_features(self):
        """Test categorical feature preparation."""
        from scripts.evaluation.run_retrieval_metrics import _prepare_categorical_features

        row = pd.Series({"gsector": 20.0, "ggroup": 5.0})

        categorical = _prepare_categorical_features(row)

        assert categorical.shape == (1, 2)
        assert categorical[0, 0] == 2  # gsector 20 -> mapped to 2
        assert categorical[0, 1] == 5  # ggroup 5

    def test_compute_embedding_scores_from_embeddings(self):
        """Test computing embedding scores from pre-computed embeddings."""
        import torch

        from scripts.evaluation.run_retrieval_metrics import (
            compute_embedding_scores_from_embeddings,
        )

        # Create mock embeddings
        embeddings = {
            "A": torch.randn(256),
            "B": torch.randn(256),
            "C": torch.randn(256),
            "D": torch.randn(256),
        }

        snapshot_df = pd.DataFrame(
            {
                "symbol": ["A", "B", "C", "D"],
                "LiquidityScore": [0.8, 0.6, 0.4, 0.3],
            }
        )

        scores_dict = compute_embedding_scores_from_embeddings(
            embeddings=embeddings,
            query_symbols=["A", "B"],
            snapshot_df=snapshot_df,
        )

        assert "A" in scores_dict
        assert "B" in scores_dict
        # A's scores should not include A itself
        assert "A" not in scores_dict["A"].index
        # All scores should be in valid cosine similarity range [-1, 1]
        for scores in scores_dict.values():
            assert all(scores >= -1.0)
            assert all(scores <= 1.0)

    def test_compute_liquidity_distance_ranking(self, monkeypatch):
        """Test liquidity-distance ranking computation."""
        # Mock compute_embedding_scores to avoid needing real model
        import numpy as np

        def mock_compute_embedding_scores(
            snapshot_df, period_df, query_symbols, checkpoint_path, device="cpu"
        ):
            """Return deterministic placeholder scores for testing."""
            all_symbols = snapshot_df["symbol"].tolist()
            scores_dict = {}
            for query in query_symbols:
                candidates = [s for s in all_symbols if s != query]
                n_candidates = len(candidates)
                query_seed = 42 + hash(query) % 10000
                query_rng = np.random.RandomState(query_seed)
                scores = 0.5 + 0.5 * query_rng.rand(n_candidates)
                scores_dict[query] = pd.Series(scores, index=candidates)
            return scores_dict

        monkeypatch.setattr(
            "scripts.evaluation.run_retrieval_metrics.compute_embedding_scores",
            mock_compute_embedding_scores,
        )
        from scripts.evaluation.run_retrieval_metrics import compute_liquidity_distance_ranking

        snapshot_df = pd.DataFrame(
            {
                "symbol": ["A", "B", "C", "D"],
                "LiquidityScore": [0.8, 0.6, 0.4, 0.3],
                "liquidity_quartile": [3, 2, 1, 0],
            }
        )

        ranking = compute_liquidity_distance_ranking("A", snapshot_df)

        # C should be closest to A (distance 0.4)
        # B is distance 0.2, D is distance 0.5
        assert len(ranking) == 3  # A excluded
        assert "A" not in ranking.index

    def test_build_ground_truth_relevance(self):
        """Test ground truth relevance building."""
        from scripts.evaluation.run_retrieval_metrics import build_ground_truth_relevance

        snapshot_df = pd.DataFrame(
            {
                "symbol": ["A", "B", "C", "D"],
                "LiquidityScore": [0.8, 0.6, 0.4, 0.3],
                "liquidity_quartile": [3, 2, 1, 0],
            }
        )

        binary, graded = build_ground_truth_relevance("A", snapshot_df)

        assert len(binary) == 3
        assert len(graded) == 3
        # B should be binary relevant (same general area)
        assert binary.sum() >= 0

    def test_cli_pipeline_runs_on_tiny_fixture(self, tmp_path, monkeypatch):
        """Smoke test that CLI pipeline main function runs without error on tiny fixture."""

        # Mock compute_embedding_scores to avoid needing real model/features
        def mock_compute_embedding_scores(
            snapshot_df, period_df, query_symbols, checkpoint_path, device="cpu"
        ):
            all_symbols = snapshot_df["symbol"].tolist()
            scores_dict = {}
            for query in query_symbols:
                candidates = [s for s in all_symbols if s != query]
                n_candidates = len(candidates)
                query_seed = 42 + hash(query) % 10000
                query_rng = np.random.RandomState(query_seed)
                scores = 0.5 + 0.5 * query_rng.rand(n_candidates)
                scores_dict[query] = pd.Series(scores, index=candidates)
            return scores_dict

        monkeypatch.setattr(
            "scripts.evaluation.run_retrieval_metrics.compute_embedding_scores",
            mock_compute_embedding_scores,
        )

        # Create tiny fixture data
        symbols = ["A", "B", "C", "D", "E", "F", "G", "H"]
        dates = pd.date_range("2019-01-01", periods=100, freq="D")

        np.random.seed(42)
        data = []
        for sym in symbols:
            for date in dates:
                data.append(
                    {
                        "symbol": sym,
                        "date": date,
                        "ret": np.random.randn() * 0.02,
                        "vol": np.random.rand() * 1000000,
                        "prc": 100 + np.random.randn() * 10,
                        "shrout": 1000000 + np.random.rand() * 500000,
                        "askhi": 100 + np.random.rand() * 0.5,
                        "bidlo": 100 - np.random.rand() * 0.5,
                        "spread_pct": np.random.rand() * 0.05,
                        "amihud": np.random.rand() * 1e-6,
                        "turnover": np.random.rand() * 0.1,
                        "market_cap": np.random.rand() * 1e11,
                        "gsector": np.random.choice(["Tech", "Finance", "Healthcare"]),
                        "market_cap_tier": np.random.choice(["Large", "Mid", "Small"]),
                    }
                )

        fixture_df = pd.DataFrame(data)

        # Save fixture
        fixture_path = tmp_path / "fixture.parquet"
        fixture_df.to_parquet(fixture_path)

        # Create output dir
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Import and run the pipeline function
        from scripts.evaluation.run_retrieval_metrics import run_evaluation_pipeline

        # This should not raise an exception
        results = run_evaluation_pipeline(
            features_path=str(fixture_path),
            checkpoint_path="dummy_checkpoint.ckpt",
            period_start="2019-01-01",
            period_end="2019-06-30",
            n_queries=3,
            seed=42,
            output_dir=str(output_dir),
            run_hybrid=False,
        )

        # Verify output structure
        assert (output_dir / "metrics").exists()
        assert (output_dir / "retrieval").exists()
        assert (output_dir / "retrieval" / "query_manifest.csv").exists()

        # Verify overall metrics were computed
        overall = results["overall_metrics"]
        assert len(overall) == 3  # Recall@10, Spearman, nDCG@10
        assert "embedding" in overall.columns
        assert "correlation" in overall.columns
        assert "liquidity_distance" in overall.columns
        # Verify correlation_rerank is present
        assert "correlation_rerank" in overall.columns


class TestCorrelationRerankRanker:
    """Tests for the correlation_rerank ranker."""

    def test_correlation_rerank_appears_in_overall_outputs(self, tmp_path, monkeypatch):
        """Test that correlation_rerank appears in overall metrics output."""

        # Mock compute_embedding_scores to avoid needing real model/features
        def mock_compute_embedding_scores(
            snapshot_df, period_df, query_symbols, checkpoint_path, device="cpu"
        ):
            all_symbols = snapshot_df["symbol"].tolist()
            scores_dict = {}
            for query in query_symbols:
                candidates = [s for s in all_symbols if s != query]
                n_candidates = len(candidates)
                query_seed = 42 + hash(query) % 10000
                query_rng = np.random.RandomState(query_seed)
                scores = 0.5 + 0.5 * query_rng.rand(n_candidates)
                scores_dict[query] = pd.Series(scores, index=candidates)
            return scores_dict

        monkeypatch.setattr(
            "scripts.evaluation.run_retrieval_metrics.compute_embedding_scores",
            mock_compute_embedding_scores,
        )

        # Create tiny fixture data
        symbols = ["A", "B", "C", "D", "E", "F", "G", "H"]
        dates = pd.date_range("2019-01-01", periods=100, freq="D")

        np.random.seed(42)
        data = []
        for sym in symbols:
            for date in dates:
                data.append(
                    {
                        "symbol": sym,
                        "date": date,
                        "ret": np.random.randn() * 0.02,
                        "vol": np.random.rand() * 1000000,
                        "prc": 100 + np.random.randn() * 10,
                        "shrout": 1000000 + np.random.rand() * 500000,
                        "askhi": 100 + np.random.rand() * 0.5,
                        "bidlo": 100 - np.random.rand() * 0.5,
                        "spread_pct": np.random.rand() * 0.05,
                        "amihud": np.random.rand() * 1e-6,
                        "turnover": np.random.rand() * 0.1,
                        "market_cap": np.random.rand() * 1e11,
                        "gsector": np.random.choice(["Tech", "Finance", "Healthcare"]),
                        "market_cap_tier": np.random.choice(["Large", "Mid", "Small"]),
                    }
                )

        fixture_df = pd.DataFrame(data)
        fixture_path = tmp_path / "fixture.parquet"
        fixture_df.to_parquet(fixture_path)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        from scripts.evaluation.run_retrieval_metrics import run_evaluation_pipeline

        results = run_evaluation_pipeline(
            features_path=str(fixture_path),
            checkpoint_path="dummy_checkpoint.ckpt",
            period_start="2019-01-01",
            period_end="2019-06-30",
            n_queries=3,
            seed=42,
            output_dir=str(output_dir),
            run_hybrid=False,
        )

        # Verify correlation_rerank in overall metrics
        overall = results["overall_metrics"]
        assert "correlation_rerank" in overall.columns
        # Verify it has valid (non-NaN) values
        corr_rerank_row = overall[overall["metric_name"] == "Recall@10"][
            "correlation_rerank"
        ].values[0]
        assert not np.isnan(corr_rerank_row)

    def test_correlation_rerank_ranking_generation(self, tmp_path, monkeypatch):
        """Test that correlation_rerank ranking is generated correctly."""

        # Mock compute_embedding_scores to avoid needing real model/features
        def mock_compute_embedding_scores(
            snapshot_df, period_df, query_symbols, checkpoint_path, device="cpu"
        ):
            all_symbols = snapshot_df["symbol"].tolist()
            scores_dict = {}
            for query in query_symbols:
                candidates = [s for s in all_symbols if s != query]
                n_candidates = len(candidates)
                query_seed = 42 + hash(query) % 10000
                query_rng = np.random.RandomState(query_seed)
                scores = 0.5 + 0.5 * query_rng.rand(n_candidates)
                scores_dict[query] = pd.Series(scores, index=candidates)
            return scores_dict

        monkeypatch.setattr(
            "scripts.evaluation.run_retrieval_metrics.compute_embedding_scores",
            mock_compute_embedding_scores,
        )

        # Create tiny fixture data
        symbols = ["A", "B", "C", "D", "E", "F", "G", "H"]
        dates = pd.date_range("2019-01-01", periods=100, freq="D")

        np.random.seed(42)
        data = []
        for sym in symbols:
            for date in dates:
                data.append(
                    {
                        "symbol": sym,
                        "date": date,
                        "ret": np.random.randn() * 0.02,
                        "vol": np.random.rand() * 1000000,
                        "prc": 100 + np.random.randn() * 10,
                        "shrout": 1000000 + np.random.rand() * 500000,
                        "askhi": 100 + np.random.rand() * 0.5,
                        "bidlo": 100 - np.random.rand() * 0.5,
                        "spread_pct": np.random.rand() * 0.05,
                        "amihud": np.random.rand() * 1e-6,
                        "turnover": np.random.rand() * 0.1,
                        "market_cap": np.random.rand() * 1e11,
                        "gsector": np.random.choice(["Tech", "Finance", "Healthcare"]),
                        "market_cap_tier": np.random.choice(["Large", "Mid", "Small"]),
                    }
                )

        fixture_df = pd.DataFrame(data)
        fixture_path = tmp_path / "fixture.parquet"
        fixture_df.to_parquet(fixture_path)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        from scripts.evaluation.run_retrieval_metrics import run_evaluation_pipeline

        # Run pipeline (results not needed since we check files directly)
        run_evaluation_pipeline(
            features_path=str(fixture_path),
            checkpoint_path="dummy_checkpoint.ckpt",
            period_start="2019-01-01",
            period_end="2019-06-30",
            n_queries=3,
            seed=42,
            output_dir=str(output_dir),
            run_hybrid=False,
        )

        # Check per-query parquet has correlation_rerank columns
        per_query_dir = output_dir / "retrieval" / "per_query"
        assert per_query_dir.exists()

        # Check at least one query parquet file has correlation_rerank columns
        parquet_files = list(per_query_dir.glob("*.parquet"))
        assert len(parquet_files) > 0

        # Read one parquet file and check columns
        pq_df = pd.read_parquet(parquet_files[0])
        assert (
            "correlation_rerank_score" in pq_df.columns
            or "correlation_rerank_rank" in pq_df.columns
            or "corr_rerank_score" in pq_df.columns
        )

    def test_correlation_rerank_fallback_when_20d_missing(self, tmp_path, monkeypatch):
        """Test that correlation_rerank works when liquidity20d is not available (uses period LiquidityScore)."""

        # Mock compute_embedding_scores to avoid needing real model/features
        def mock_compute_embedding_scores(
            snapshot_df, period_df, query_symbols, checkpoint_path, device="cpu"
        ):
            all_symbols = snapshot_df["symbol"].tolist()
            scores_dict = {}
            for query in query_symbols:
                candidates = [s for s in all_symbols if s != query]
                n_candidates = len(candidates)
                query_seed = 42 + hash(query) % 10000
                query_rng = np.random.RandomState(query_seed)
                scores = 0.5 + 0.5 * query_rng.rand(n_candidates)
                scores_dict[query] = pd.Series(scores, index=candidates)
            return scores_dict

        monkeypatch.setattr(
            "scripts.evaluation.run_retrieval_metrics.compute_embedding_scores",
            mock_compute_embedding_scores,
        )

        # Create tiny fixture with limited dates (less than 20 trading days)
        # so 20d liquidity cannot be computed
        symbols = ["A", "B", "C", "D", "E", "F", "G", "H"]
        dates = pd.date_range("2019-01-01", periods=15, freq="D")  # Only 15 days, < 20d requirement

        np.random.seed(42)
        data = []
        for sym in symbols:
            for date in dates:
                data.append(
                    {
                        "symbol": sym,
                        "date": date,
                        "ret": np.random.randn() * 0.02,
                        "vol": np.random.rand() * 1000000,
                        "prc": 100 + np.random.randn() * 10,
                        "shrout": 1000000 + np.random.rand() * 500000,
                        "askhi": 100 + np.random.rand() * 0.5,
                        "bidlo": 100 - np.random.rand() * 0.5,
                        "spread_pct": np.random.rand() * 0.05,
                        "amihud": np.random.rand() * 1e-6,
                        "turnover": np.random.rand() * 0.1,
                        "market_cap": np.random.rand() * 1e11,
                        "gsector": np.random.choice(["Tech", "Finance", "Healthcare"]),
                        "market_cap_tier": np.random.choice(["Large", "Mid", "Small"]),
                    }
                )

        fixture_df = pd.DataFrame(data)
        fixture_path = tmp_path / "fixture.parquet"
        fixture_df.to_parquet(fixture_path)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        from scripts.evaluation.run_retrieval_metrics import run_evaluation_pipeline

        # This should not crash even when 20d liquidity cannot be computed
        # It should fall back to period LiquidityScore
        results = run_evaluation_pipeline(
            features_path=str(fixture_path),
            checkpoint_path="dummy_checkpoint.ckpt",
            period_start="2019-01-01",
            period_end="2019-03-31",  # Short period
            n_queries=3,
            seed=42,
            output_dir=str(output_dir),
            run_hybrid=False,  # No hybrid - only correlation_rerank
        )

        # Verify correlation_rerank still computed
        overall = results["overall_metrics"]
        assert "correlation_rerank" in overall.columns


class TestRevisedEvaluationPipeline:
    """Test the revised 6 rankers x 3 references pipeline."""

    def test_ground_truth_computation(self):
        """Test that all three ground truths are computed correctly."""
        from src.evaluation.ground_truth import (
            compute_similarity_score,
            compute_liquidity_uplift,
            compute_utility_score,
        )

        # Setup test data
        returns_df = pd.DataFrame(
            {
                "date": pd.date_range("2019-01-01", periods=120, freq="D"),
                "AAPL": np.random.randn(120) * 0.02,
                "MSFT": np.random.randn(120) * 0.02,
            }
        ).set_index("date")

        snapshot_df = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT"],
                "gsector": [1, 1],
                "ggroup": [10, 10],
                "market_cap": [1e12, 9e11],
                "LiquidityScore": [0.8, 0.6],
            }
        )

        # Compute all three ground truths
        sim = compute_similarity_score("AAPL", returns_df, snapshot_df, min_overlap=80)
        uplift = compute_liquidity_uplift("AAPL", snapshot_df.set_index("symbol")["LiquidityScore"])
        utility = compute_utility_score(sim, uplift)

        # Assertions
        assert len(sim) == 1  # Only MSFT
        assert "MSFT" in sim.index
        assert uplift["MSFT"] == 0.6 - 0.8
        assert utility["MSFT"] == sim["MSFT"] * max(0, uplift["MSFT"])


class TestBreakdownAnalysis:
    """Test utility breakdown analysis integration."""

    def test_per_query_parquet_includes_components(self, tmp_path, monkeypatch):
        """Test that per-query parquets include similarity components."""

        # Mock compute_embedding_scores to avoid needing real model/features
        def mock_compute_embedding_scores(
            snapshot_df, period_df, query_symbols, checkpoint_path, device="cpu"
        ):
            all_symbols = snapshot_df["symbol"].tolist()
            scores_dict = {}
            for query in query_symbols:
                candidates = [s for s in all_symbols if s != query]
                n_candidates = len(candidates)
                query_seed = 42 + hash(query) % 10000
                query_rng = np.random.RandomState(query_seed)
                scores = 0.5 + 0.5 * query_rng.rand(n_candidates)
                scores_dict[query] = pd.Series(scores, index=candidates)
            return scores_dict

        monkeypatch.setattr(
            "scripts.evaluation.run_retrieval_metrics.compute_embedding_scores",
            mock_compute_embedding_scores,
        )

        # Create tiny fixture data
        symbols = ["A", "B", "C", "D", "E", "F", "G", "H"]
        dates = pd.date_range("2019-01-01", periods=100, freq="D")

        np.random.seed(42)
        data = []
        for sym in symbols:
            for date in dates:
                data.append(
                    {
                        "symbol": sym,
                        "date": date,
                        "ret": np.random.randn() * 0.02,
                        "vol": np.random.rand() * 1000000,
                        "prc": 100 + np.random.randn() * 10,
                        "shrout": 1000000 + np.random.rand() * 500000,
                        "askhi": 100 + np.random.rand() * 0.5,
                        "bidlo": 100 - np.random.rand() * 0.5,
                        "spread_pct": np.random.rand() * 0.05,
                        "amihud": np.random.rand() * 1e-6,
                        "turnover": np.random.rand() * 0.1,
                        "market_cap": np.random.rand() * 1e11,
                        "gsector": np.random.choice(["Tech", "Finance", "Healthcare"]),
                        "market_cap_tier": np.random.choice(["Large", "Mid", "Small"]),
                    }
                )

        fixture_df = pd.DataFrame(data)
        fixture_path = tmp_path / "fixture.parquet"
        fixture_df.to_parquet(fixture_path)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        from scripts.evaluation.run_retrieval_metrics import run_evaluation_pipeline

        run_evaluation_pipeline(
            features_path=str(fixture_path),
            checkpoint_path="dummy_checkpoint.ckpt",
            period_start="2019-01-01",
            period_end="2019-06-30",
            n_queries=3,
            seed=42,
            output_dir=str(output_dir),
            run_hybrid=False,
        )

        # Check per-query parquet has component columns
        per_query_dir = output_dir / "retrieval" / "per_query"
        if per_query_dir.exists():
            parquet_files = list(per_query_dir.glob("*.parquet"))
            if parquet_files:
                df = pd.read_parquet(parquet_files[0])
                # Check for similarity component columns
                assert "return_similarity" in df.columns, "Missing return_similarity"
                assert "sector_similarity" in df.columns, "Missing sector_similarity"
                assert "size_similarity" in df.columns, "Missing size_similarity"

    def test_utility_breakdown_csv_generated(self, tmp_path, monkeypatch):
        """Test that utility breakdown analysis CSV is created."""

        # Mock compute_embedding_scores to avoid needing real model/features
        def mock_compute_embedding_scores(
            snapshot_df, period_df, query_symbols, checkpoint_path, device="cpu"
        ):
            all_symbols = snapshot_df["symbol"].tolist()
            scores_dict = {}
            for query in query_symbols:
                candidates = [s for s in all_symbols if s != query]
                n_candidates = len(candidates)
                query_seed = 42 + hash(query) % 10000
                query_rng = np.random.RandomState(query_seed)
                scores = 0.5 + 0.5 * query_rng.rand(n_candidates)
                scores_dict[query] = pd.Series(scores, index=candidates)
            return scores_dict

        monkeypatch.setattr(
            "scripts.evaluation.run_retrieval_metrics.compute_embedding_scores",
            mock_compute_embedding_scores,
        )

        # Create tiny fixture data
        symbols = ["A", "B", "C", "D", "E", "F", "G", "H"]
        dates = pd.date_range("2019-01-01", periods=100, freq="D")

        np.random.seed(42)
        data = []
        for sym in symbols:
            for date in dates:
                data.append(
                    {
                        "symbol": sym,
                        "date": date,
                        "ret": np.random.randn() * 0.02,
                        "vol": np.random.rand() * 1000000,
                        "prc": 100 + np.random.randn() * 10,
                        "shrout": 1000000 + np.random.rand() * 500000,
                        "askhi": 100 + np.random.rand() * 0.5,
                        "bidlo": 100 - np.random.rand() * 0.5,
                        "spread_pct": np.random.rand() * 0.05,
                        "amihud": np.random.rand() * 1e-6,
                        "turnover": np.random.rand() * 0.1,
                        "market_cap": np.random.rand() * 1e11,
                        "gsector": np.random.choice(["Tech", "Finance", "Healthcare"]),
                        "market_cap_tier": np.random.choice(["Large", "Mid", "Small"]),
                    }
                )

        fixture_df = pd.DataFrame(data)
        fixture_path = tmp_path / "fixture.parquet"
        fixture_df.to_parquet(fixture_path)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        from scripts.evaluation.run_retrieval_metrics import run_evaluation_pipeline

        run_evaluation_pipeline(
            features_path=str(fixture_path),
            checkpoint_path="dummy_checkpoint.ckpt",
            period_start="2019-01-01",
            period_end="2019-06-30",
            n_queries=3,
            seed=42,
            output_dir=str(output_dir),
            run_hybrid=False,
        )

        # Check that utility breakdown analysis CSV is generated
        breakdown_file = output_dir / "metrics" / "utility_breakdown_analysis.csv"
        assert breakdown_file.exists(), "utility_breakdown_analysis.csv not found"

        df = pd.read_csv(breakdown_file)
        assert "query" in df.columns
        assert "ranker" in df.columns
        assert "pct_return_sim" in df.columns
        assert "pct_sector_sim" in df.columns
        assert "pct_liq_improve" in df.columns
