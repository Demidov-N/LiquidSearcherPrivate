"""Tests for ground truth reference computations.

Tests for src/evaluation/ground_truth.py
"""

import numpy as np
import pandas as pd
import pytest


class TestReturnSimilarity:
    """Tests for 120-day return similarity computation."""

    def test_compute_return_similarity(self):
        """Test 120-day Pearson return similarity computation."""
        np.random.seed(42)
        returns_df = pd.DataFrame(
            {
                "date": pd.date_range("2019-01-01", periods=120, freq="D"),
                "AAPL": np.random.randn(120) * 0.02,
                "MSFT": np.random.randn(120) * 0.02,
            }
        ).set_index("date")

        from src.evaluation.ground_truth import compute_return_similarity_120d

        result = compute_return_similarity_120d(returns_df, "AAPL", min_overlap=80)

        assert isinstance(result, pd.Series)
        assert "MSFT" in result.index
        assert 0.0 <= result.loc["MSFT"] <= 1.0  # Normalized to [0,1]


class TestSectorSimilarity:
    """Tests for sector similarity computation."""

    def test_compute_sector_similarity(self):
        """Test sector similarity computation."""
        from src.evaluation.ground_truth import compute_sector_similarity

        # Query: ggroup=10, gsector=1
        candidates = pd.DataFrame(
            {
                "symbol": ["SAME_GGROUP", "SAME_SECTOR", "DIFF_SECTOR"],
                "gsector": [1, 1, 2],
                "ggroup": [10, 11, 20],
            }
        )

        result = compute_sector_similarity(
            query_gsector=1,
            query_ggroup=10,
            candidates_df=candidates,
        )

        assert result["SAME_GGROUP"] == 1.0
        assert result["SAME_SECTOR"] == 0.5
        assert result["DIFF_SECTOR"] == 0.0


class TestSizeSimilarity:
    """Tests for size similarity computation."""

    def test_compute_size_similarity(self):
        """Test size similarity using market_cap percentiles."""
        from src.evaluation.ground_truth import compute_size_similarity

        snapshot_df = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT", "TSLA"],
                "market_cap": [1e12, 9e11, 1e10],  # AAPL largest, TSLA smallest
            }
        )

        result = compute_size_similarity(
            query_symbol="AAPL",
            snapshot_df=snapshot_df,
        )

        # AAPL vs itself should be 1.0 (perfect match)
        # AAPL vs MSFT should be high but not 1.0
        # AAPL vs TSLA should be low
        assert result["AAPL"] == 1.0
        assert 0.0 <= result["MSFT"] <= 1.0
        assert 0.0 <= result["TSLA"] <= 1.0
        assert result["TSLA"] < result["MSFT"]


class TestSimilarityScore:
    """Tests for composite SimilarityScore computation."""

    def test_compute_similarity_score(self):
        """Test composite SimilarityScore with balanced weights."""
        from src.evaluation.ground_truth import compute_similarity_score

        np.random.seed(42)
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
            }
        )

        result = compute_similarity_score(
            query_symbol="AAPL",
            returns_df=returns_df,
            snapshot_df=snapshot_df,
            min_overlap=80,
        )

        assert isinstance(result, pd.Series)
        assert "MSFT" in result.index
        assert 0.0 <= result["MSFT"] <= 1.0


class TestBinaryRelevance:
    """Tests for binary relevance computation."""

    def test_build_binary_relevance(self):
        """Test binary relevance: top quartile of SimilarityScore."""
        from src.evaluation.ground_truth import build_binary_relevance

        similarity_scores = pd.Series(
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            index=["A", "B", "C", "D", "E", "F", "G", "H"],
        )

        binary = build_binary_relevance(similarity_scores, top_percentile=0.25)

        # Top 25% (2 items) should be 1, rest 0
        assert binary["A"] == 1
        assert binary["B"] == 1
        assert binary["H"] == 0


class TestGradedRelevance:
    """Tests for graded relevance computation."""

    def test_build_graded_relevance(self):
        """Test graded relevance with 3/2/1/0 bands."""
        from src.evaluation.ground_truth import build_graded_relevance

        similarity_scores = pd.Series(
            [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5],
            index=["A", "B", "C", "D", "E", "F", "G", "H"],
        )

        graded = build_graded_relevance(similarity_scores)

        # Top 10% (0.8 items -> 1 item) = grade 3
        assert graded["A"] == 3

        # Next 15% = grade 2
        assert graded["B"] == 2

        # Same quartile but farther = grade 1
        # Bottom = grade 0


class TestLiquidityUplift:
    """Tests for LiquidityUplift computation."""

    def test_compute_liquidity_uplift(self):
        """Test LiquidityUplift = LiquidityScore_candidate - LiquidityScore_query."""
        from src.evaluation.ground_truth import compute_liquidity_uplift

        liquidity_scores = pd.Series([0.8, 0.6, 0.4, 0.2], index=["A", "B", "C", "D"])

        uplift = compute_liquidity_uplift(query_symbol="A", liquidity_scores=liquidity_scores)

        assert uplift["B"] == 0.6 - 0.8  # -0.2
        assert uplift["C"] == 0.4 - 0.8  # -0.4
        assert uplift["D"] == 0.2 - 0.8  # -0.6
        assert "A" not in uplift.index  # Exclude self


class TestUtilityScore:
    """Tests for UtilityScore computation."""

    def test_compute_utility_score(self):
        """Test UtilityScore = SimilarityScore * max(0, LiquidityUplift)."""
        from src.evaluation.ground_truth import compute_utility_score

        similarity = pd.Series([0.9, 0.8, 0.7], index=["A", "B", "C"])
        uplift = pd.Series([0.2, -0.1, 0.3], index=["A", "B", "C"])

        utility = compute_utility_score(similarity, uplift)

        # A: 0.9 * 0.2 = 0.18
        assert utility["A"] == 0.9 * 0.2
        # B: 0.8 * 0 = 0 (negative uplift clamped)
        assert utility["B"] == 0.0
        # C: 0.7 * 0.3 = 0.21
        assert utility["C"] == 0.7 * 0.3
