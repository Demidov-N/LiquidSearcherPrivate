"""Tests for liquidity label utilities."""

import numpy as np
import pandas as pd

from src.evaluation.utils.liquidity_labels import (
    aggregate_period_liquidity,
    aggregate_trailing_20d_liquidity,
    assign_liquidity_quartiles,
    compute_daily_liquidity_proxies,
    compute_graded_relevance,
)


class TestComputeDailyLiquidityProxies:
    """Test daily liquidity proxy computation."""

    def test_adds_expected_columns(self):
        """Test that spread_pct, amihud, and turnover columns are added."""
        df = pd.DataFrame(
            {
                "symbol": ["AAA", "AAA"],
                "date": pd.to_datetime(["2019-01-02", "2019-01-03"]),
                "askhi": [10.2, 10.1],
                "bidlo": [9.8, 9.9],
                "prc": [10.0, 10.0],
                "vol": [1000, 1200],
                "ret": [0.01, -0.02],
                "shrout": [10000, 10000],
            }
        )

        result = compute_daily_liquidity_proxies(df)

        assert {"spread_pct", "amihud", "turnover"}.issubset(result.columns)
        assert result["spread_pct"].notna().all()
        assert result["amihud"].notna().all()
        assert result["turnover"].notna().all()

    def test_spread_pct_formula(self):
        """Test spread_pct is computed correctly."""
        df = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "date": pd.to_datetime(["2019-01-02"]),
                "askhi": [10.0],
                "bidlo": [9.0],
                "prc": [10.0],
                "vol": [1000],
                "ret": [0.01],
                "shrout": [10000],
            }
        )

        result = compute_daily_liquidity_proxies(df)
        expected_spread = (10.0 - 9.0) / ((10.0 + 9.0) / 2)  # 0.105263...
        assert abs(result["spread_pct"].item() - expected_spread) < 1e-6

    def test_handles_zero_denominators(self):
        """Test that zero denominators produce NaN, not crash."""
        df = pd.DataFrame(
            {
                "symbol": ["AAA", "BBB"],
                "date": pd.to_datetime(["2019-01-02", "2019-01-02"]),
                "askhi": [0.0, 10.0],
                "bidlo": [0.0, 10.0],  # zero denominator for spread on AAA
                "prc": [10.0, 10.0],
                "vol": [1000, 1000],
                "ret": [0.01, 0.01],
                "shrout": [10000, 0],  # zero denominator for turnover
            }
        )

        result = compute_daily_liquidity_proxies(df)

        # Should not crash; both invalid paths should be represented.
        assert np.isnan(result.loc[result["symbol"] == "AAA", "spread_pct"]).item()
        assert np.isnan(result.loc[result["symbol"] == "BBB", "turnover"]).item()

    def test_handles_negative_denominators(self):
        """Test that negative/zero denominators produce NaN for spread."""
        df = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "date": pd.to_datetime(["2019-01-02"]),
                "askhi": [5.0],
                "bidlo": [10.0],  # askhi < bidlo -> negative numerator
                "prc": [10.0],
                "vol": [1000],
                "ret": [0.01],
                "shrout": [10000],
            }
        )

        result = compute_daily_liquidity_proxies(df)

        # Denominator = (5 + 10) / 2 = 7.5 > 0, so spread_pct should be computed
        assert result["spread_pct"].notna().iloc[0]

    def test_handles_negative_price_volume(self):
        """Test that negative price * volume produces NaN for amihud."""
        df = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "date": pd.to_datetime(["2019-01-02"]),
                "askhi": [10.0],
                "bidlo": [9.0],
                "prc": [-10.0],  # negative price
                "vol": [1000],
                "ret": [0.01],
                "shrout": [10000],
            }
        )

        result = compute_daily_liquidity_proxies(df)

        # abs(prc) * vol > 0, so amihud should be computed
        assert result["amihud"].notna().iloc[0]

    def test_handles_negative_shrout(self):
        """Test that negative shrout produces NaN for turnover."""
        df = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "date": pd.to_datetime(["2019-01-02"]),
                "askhi": [10.0],
                "bidlo": [9.0],
                "prc": [10.0],
                "vol": [1000],
                "ret": [0.01],
                "shrout": [-1000],  # negative shrout
            }
        )

        result = compute_daily_liquidity_proxies(df)

        # shrout < 0 should produce NaN
        assert np.isnan(result["turnover"].iloc[0])

    def test_turnover_formula(self):
        """Test turnover is computed correctly."""
        df = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "date": pd.to_datetime(["2019-01-02"]),
                "askhi": [10.0],
                "bidlo": [9.0],
                "prc": [10.0],
                "vol": [1000],
                "ret": [0.01],
                "shrout": [10000],
            }
        )

        result = compute_daily_liquidity_proxies(df)
        expected_turnover = 1000 / 10000  # 0.1
        assert abs(result["turnover"].item() - expected_turnover) < 1e-6

    def test_amihud_formula(self):
        """Test amihud formula abs(ret)/(abs(prc)*vol)."""
        df = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "date": pd.to_datetime(["2019-01-02"]),
                "askhi": [10.0],
                "bidlo": [9.0],
                "prc": [10.0],
                "vol": [1000],
                "ret": [-0.02],
                "shrout": [10000],
            }
        )

        result = compute_daily_liquidity_proxies(df)
        expected_amihud = abs(-0.02) / (abs(10.0) * 1000)
        assert abs(result["amihud"].item() - expected_amihud) < 1e-12


class TestAggregatePeriodLiquidity:
    """Test period-level liquidity aggregation."""

    def test_builds_liquidity_score(self):
        """Test LiquidityScore is created and ordered correctly."""
        df = pd.DataFrame(
            {
                "symbol": ["AAA", "AAA", "BBB", "BBB"],
                "date": pd.to_datetime(["2019-01-02", "2019-01-03", "2019-01-02", "2019-01-03"]),
                "spread_pct": [0.01, 0.02, 0.20, 0.30],
                "amihud": [0.001, 0.002, 0.04, 0.05],
                "turnover": [0.30, 0.25, 0.02, 0.01],
            }
        )

        result = aggregate_period_liquidity(df)

        assert "LiquidityScore" in result.columns
        # AAA is more liquid (lower spread, lower amihud, higher turnover)
        assert (
            result.loc[result["symbol"] == "AAA", "LiquidityScore"].item()
            > result.loc[result["symbol"] == "BBB", "LiquidityScore"].item()
        )

    def test_uses_median_aggregation(self):
        """Test that median aggregation is used for period-level values."""
        df = pd.DataFrame(
            {
                "symbol": ["AAA"] * 3,
                "date": pd.to_datetime(["2019-01-02", "2019-01-03", "2019-01-04"]),
                "spread_pct": [0.01, 0.03, 0.02],  # median should be 0.02
                "amihud": [0.001, 0.003, 0.002],  # median should be 0.002
                "turnover": [0.10, 0.30, 0.20],  # median should be 0.20
            }
        )

        result = aggregate_period_liquidity(df)

        # Check that we get one row per symbol
        assert len(result) == 1
        # The spread_pct_median should be 0.02 (the median)
        assert result["spread_pct_median"].item() == 0.02

    def test_invalid_rows_excluded_from_aggregation(self):
        """Test that rows with NaN in any proxy are excluded before aggregation."""
        # 5 rows: 3 valid, 2 invalid (one with NaN spread_pct, one with NaN amihud)
        # Median of [0.01, 0.02, 0.03] = 0.02 (not affected by invalid rows)
        df = pd.DataFrame(
            {
                "symbol": ["AAA"] * 5,
                "date": pd.to_datetime(
                    ["2019-01-02", "2019-01-03", "2019-01-04", "2019-01-05", "2019-01-06"]
                ),
                "spread_pct": [0.01, 0.02, 0.03, np.nan, 0.05],
                "amihud": [0.001, 0.002, 0.003, 0.004, np.nan],
                "turnover": [0.10, 0.20, 0.30, 0.40, 0.50],
            }
        )

        result = aggregate_period_liquidity(df)

        # Should still return 1 row for AAA
        assert len(result) == 1
        # Median should be computed over 3 valid rows only
        assert result["spread_pct_median"].item() == 0.02
        assert result["amihud_median"].item() == 0.002
        assert result["turnover_median"].item() == 0.20

    def test_invalid_rows_affect_median_values(self):
        """Test that including vs excluding invalid rows gives different median values."""
        # Without filtering: all 4 rows would be used, median might be NaN or different
        # With filtering: only rows 1, 2, 3 are valid, median is based on those
        df_with_invalid = pd.DataFrame(
            {
                "symbol": ["AAA", "AAA", "AAA", "AAA"],
                "date": pd.to_datetime(["2019-01-02", "2019-01-03", "2019-01-04", "2019-01-05"]),
                "spread_pct": [0.01, 0.02, 0.03, 0.04],
                "amihud": [0.001, 0.002, 0.003, np.nan],  # Last row invalid
                "turnover": [0.10, 0.20, 0.30, 0.40],
            }
        )

        result = aggregate_period_liquidity(df_with_invalid)

        # Median should be computed from valid rows only (first 3)
        # spread_pct median of [0.01, 0.02, 0.03] = 0.02
        # amihud median of [0.001, 0.002, 0.003] = 0.002
        assert result["spread_pct_median"].item() == 0.02
        assert result["amihud_median"].item() == 0.002

    def test_liquidity_score_uses_exact_weights(self):
        """LiquidityScore must use 0.4/0.4/0.2 weighting exactly."""
        df = pd.DataFrame(
            {
                "symbol": ["AAA", "AAA", "BBB", "BBB"],
                "date": pd.to_datetime(["2019-01-02", "2019-01-03", "2019-01-02", "2019-01-03"]),
                "spread_pct": [0.01, 0.01, 0.05, 0.05],
                "amihud": [0.001, 0.001, 0.01, 0.01],
                "turnover": [0.10, 0.10, 0.20, 0.20],
            }
        )

        result = aggregate_period_liquidity(df)

        for _, row in result.iterrows():
            expected = (
                0.4 * (1 - row["spread_rank"])
                + 0.4 * (1 - row["amihud_rank"])
                + 0.2 * row["turnover_rank"]
            )
            assert abs(row["LiquidityScore"] - expected) < 1e-12

    def test_all_invalid_rows_returns_empty_result(self):
        """All-invalid input should return an empty aggregated frame, not crash."""
        df = pd.DataFrame(
            {
                "symbol": ["AAA", "BBB"],
                "date": pd.to_datetime(["2019-01-02", "2019-01-02"]),
                "spread_pct": [np.nan, np.nan],
                "amihud": [0.001, np.nan],
                "turnover": [np.nan, 0.2],
            }
        )

        result = aggregate_period_liquidity(df)

        assert result.empty
        assert {"symbol", "spread_rank", "amihud_rank", "turnover_rank", "LiquidityScore"}.issubset(
            result.columns
        )


class TestAssignLiquidityQuartiles:
    """Test liquidity quartile assignment."""

    def test_assigns_quartiles_0_to_3(self):
        """Test quartiles are assigned as 0, 1, 2, 3."""
        scores = pd.Series([0.1, 0.3, 0.5, 0.7], index=["A", "B", "C", "D"])

        result = assign_liquidity_quartiles(scores)

        assert set(result.unique()).issubset({0, 1, 2, 3})
        assert len(result.unique()) == 4

    def test_deterministic_for_same_input(self):
        """Test that quartile assignment is deterministic."""
        scores = pd.Series([0.1, 0.3, 0.5, 0.7], index=["A", "B", "C", "D"])

        result1 = assign_liquidity_quartiles(scores)
        result2 = assign_liquidity_quartiles(scores)

        assert result1.equals(result2)

    def test_handles_ties(self):
        """Test that tied scores get the same quartile."""
        scores = pd.Series([0.5, 0.5, 0.5, 0.5], index=["A", "B", "C", "D"])

        result = assign_liquidity_quartiles(scores)

        # All same score should get same quartile
        assert result.nunique() == 1


class TestComputeGradedRelevance:
    """Test graded relevance computation."""

    def test_respects_distance_bands(self):
        """Test that closer candidates get higher relevance."""
        query_score = 0.80
        candidate_scores = pd.Series([0.79, 0.70, 0.55, 0.10], index=["A", "B", "C", "D"])
        quartiles = pd.Series([3, 3, 2, 0], index=["A", "B", "C", "D"])

        graded = compute_graded_relevance(
            query_score, candidate_scores, quartiles, query_quartile=3
        )

        assert graded["A"] >= graded["B"]
        assert graded["D"] == 0

    def test_fallback_to_same_quartile(self):
        """Test that same-quartile candidates get at least relevance 1."""
        query_score = 0.50
        # Same quartile but far in score
        candidate_scores = pd.Series([0.20, 0.10], index=["A", "B"])
        quartiles = pd.Series([2, 2], index=["A", "B"])

        graded = compute_graded_relevance(
            query_score, candidate_scores, quartiles, query_quartile=2
        )

        # Both in same quartile should get at least 1
        assert (graded >= 1).all()

    def test_graded_values_are_valid(self):
        """Test that graded relevance values are in {0, 1, 2, 3}."""
        query_score = 0.5
        candidate_scores = pd.Series([0.5, 0.4, 0.3, 0.2, 0.1], index=list("ABCDE"))
        quartiles = pd.Series([2, 2, 1, 0, 0], index=list("ABCDE"))

        graded = compute_graded_relevance(
            query_score, candidate_scores, quartiles, query_quartile=2
        )

        assert set(graded.unique()).issubset({0, 1, 2, 3})

    def test_small_n_ensures_at_least_one_relevance_3(self):
        """Test that with small n, at least one candidate gets relevance 3."""
        # n=3: ceil(0.10 * 3) = 1, so exactly 1 candidate gets relevance 3
        query_score = 0.50
        candidate_scores = pd.Series([0.49, 0.48, 0.10], index=["A", "B", "C"])
        quartiles = pd.Series([2, 2, 0], index=["A", "B", "C"])

        graded = compute_graded_relevance(
            query_score, candidate_scores, quartiles, query_quartile=2
        )

        # At least one candidate should have relevance 3
        assert (graded == 3).any()
        # Exactly 1 should have relevance 3 (ceil(0.10 * 3) = 1)
        assert (graded == 3).sum() == 1

    def test_exact_graded_bands_for_n10(self):
        """Test exact band counts for n=10."""
        # n=10: ceil(0.10 * 10) = 1, ceil(0.25 * 10) = 3
        # So: 1 gets 3, 2 get 2 (ranks 2-3), rest depend on quartile
        query_score = 0.50
        candidate_scores = pd.Series(
            [0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.40],
            index=list("ABCDEFGHIJ"),
        )
        # Quartiles: A=0, B=1, C=2, D=3, E=0, F=1, G=2, H=3, I=0, J=1
        # D and H are same quartile (3) as query, so they get relevance 1
        quartiles = pd.Series([0, 1, 2, 3, 0, 1, 2, 3, 0, 1], index=list("ABCDEFGHIJ"))

        graded = compute_graded_relevance(
            query_score, candidate_scores, quartiles, query_quartile=3
        )

        # Exactly 1 should have relevance 3 (A, closest)
        assert (graded == 3).sum() == 1
        # Exactly 2 should have relevance 2 (B, C)
        assert (graded == 2).sum() == 2
        # D and H are same quartile as query, so they get 1
        assert (graded == 1).sum() == 2
        # Others should be 0 (not same quartile as query)
        assert (graded == 0).sum() == 5

    def test_same_quartile_overrides_distance(self):
        """Test that same quartile gets relevance 1 even if far in distance."""
        query_score = 0.50
        # C is in same quartile but far in distance
        candidate_scores = pd.Series([0.49, 0.48, 0.10], index=["A", "B", "C"])
        quartiles = pd.Series([3, 3, 2], index=["A", "B", "C"])

        graded = compute_graded_relevance(
            query_score, candidate_scores, quartiles, query_quartile=2
        )

        # C is same quartile but far, should get 1 not 0
        assert graded["C"] == 1

    def test_n1_single_candidate_gets_relevance_3(self):
        """Test that n=1 gives the single candidate relevance 3."""
        query_score = 0.50
        candidate_scores = pd.Series([0.49], index=["A"])
        quartiles = pd.Series([2], index=["A"])

        graded = compute_graded_relevance(
            query_score, candidate_scores, quartiles, query_quartile=2
        )

        # Single candidate should get relevance 3
        assert graded["A"] == 3

    def test_tie_at_threshold_gets_deterministic_relevance(self):
        """Ties at top-10 threshold should share relevance 3."""
        # n=4: ceil(10%)=1 and threshold distance is 0.01.
        # A/B/C are tied at 0.01 and should all receive relevance 3 under
        # distance-threshold banding. D is farther and different quartile.
        query_score = 0.50
        candidate_scores = pd.Series([0.49, 0.49, 0.49, 0.48], index=["A", "B", "C", "D"])
        quartiles = pd.Series([3, 3, 3, 0], index=["A", "B", "C", "D"])

        graded = compute_graded_relevance(
            query_score, candidate_scores, quartiles, query_quartile=3
        )

        # All tied-at-threshold names get relevance 3.
        assert graded["A"] == 3
        assert graded["B"] == 3
        assert graded["C"] == 3
        # D is farther and different quartile, gets 0
        assert graded["D"] == 0

    def test_tie_at_25pct_threshold_relevance_1_fallback(self):
        """Test that same-quartile candidates at 25% threshold get relevance 1."""
        # n=6: ceil(0.10 * 6) = 1, ceil(0.25 * 6) = 2
        # A has distance 0.01 (top 10%), B-E have distance 0.02 (next 15%)
        # F has distance 0.03 but same quartile - should get 1
        query_score = 0.50
        candidate_scores = pd.Series(
            [0.49, 0.48, 0.48, 0.48, 0.48, 0.47], index=["A", "B", "C", "D", "E", "F"]
        )
        quartiles = pd.Series([3, 3, 3, 3, 3, 3], index=["A", "B", "C", "D", "E", "F"])

        graded = compute_graded_relevance(
            query_score, candidate_scores, quartiles, query_quartile=3
        )

        # A gets 3. B-E tie at the 25% threshold distance and all get 2.
        # F is same quartile but farther, so it gets fallback relevance 1.
        assert graded["A"] == 3
        assert graded["B"] == 2
        assert graded["C"] == 2
        assert graded["D"] == 2
        assert graded["E"] == 2
        assert graded["F"] == 1

    def test_tie_labels_stable_under_input_reordering(self):
        """Tie outcomes should be independent of candidate input order."""
        query_score = 0.50
        candidate_scores = pd.Series([0.49, 0.49, 0.49, 0.48], index=["A", "B", "C", "D"])
        quartiles = pd.Series([3, 3, 3, 0], index=["A", "B", "C", "D"])

        graded_original = compute_graded_relevance(
            query_score,
            candidate_scores,
            quartiles,
            query_quartile=3,
        )

        reordered = ["C", "A", "D", "B"]
        graded_reordered = compute_graded_relevance(
            query_score,
            candidate_scores.loc[reordered],
            quartiles.loc[reordered],
            query_quartile=3,
        )

        assert graded_original.sort_index().equals(graded_reordered.sort_index())


class TestAggregateTrailing20dLiquidity:
    """Test trailing 20-day liquidity aggregation."""

    def test_returns_only_symbols_with_20d_data(self):
        """Test that only symbols with at least 20 days of data are included."""
        # Create 25 days of data for 2 symbols
        dates = pd.date_range("2019-01-01", periods=25, freq="D")
        df = pd.DataFrame(
            {
                "symbol": ["AAA"] * 25 + ["BBB"] * 10,
                "date": list(dates) + list(dates[:10]),
                "spread_pct": [0.01] * 25 + [0.02] * 10,
                "amihud": [0.001] * 25 + [0.002] * 10,
                "turnover": [0.1] * 25 + [0.05] * 10,
            }
        )

        snapshot_date = pd.Timestamp("2019-01-25")
        result = aggregate_trailing_20d_liquidity(df, snapshot_date)

        # Only AAA has 20+ days of data
        assert len(result) == 1
        assert result["symbol"].item() == "AAA"

    def test_uses_median_over_trailing_20_days(self):
        """Test that median aggregation is used for trailing 20 days."""
        dates = pd.date_range("2019-01-01", periods=25, freq="D")
        spread_values = [0.01, 0.02, 0.03] * 8 + [0.01]  # 25 values, median of first 20 is 0.02
        df = pd.DataFrame(
            {
                "symbol": ["AAA"] * 25,
                "date": list(dates),
                "spread_pct": spread_values,
                "amihud": [0.001] * 25,
                "turnover": [0.1] * 25,
            }
        )

        snapshot_date = pd.Timestamp("2019-01-25")
        result = aggregate_trailing_20d_liquidity(df, snapshot_date)

        # Should get median of last 20 days
        assert result["spread_pct_median"].item() == 0.02

    def test_exact_20_trading_days_selection(self):
        """Test that exactly 20 trading days are used, not calendar approximation."""
        # Create data spanning more than 20 calendar days but with gaps
        # Jan 1-7 2019 is 7 days (no weekends), Jan 8-14 is another 7 days
        # We'll use 21 dates but skip some in between to show calendar isn't used
        dates = pd.to_datetime(
            [
                "2019-01-02",
                "2019-01-03",
                "2019-01-04",  # Wed-Fri (3)
                "2019-01-07",
                "2019-01-08",
                "2019-01-09",
                "2019-01-10",
                "2019-01-11",  # Mon-Fri (5) = 8
                "2019-01-14",
                "2019-01-15",
                "2019-01-16",
                "2019-01-17",
                "2019-01-18",  # Mon-Fri (5) = 13
                "2019-01-22",
                "2019-01-23",
                "2019-01-24",
                "2019-01-25",
                "2019-01-28",  # Mon-Fri (5) = 18
                "2019-01-29",
                "2019-01-30",
                "2019-01-31",  # Tue-Thu (3) = 21
            ]
        )
        # Last 20 dates would be from index 1 onwards
        spread_values = list(range(21))  # 0-20
        df = pd.DataFrame(
            {
                "symbol": ["AAA"] * 21,
                "date": list(dates),
                "spread_pct": spread_values,
                "amihud": [0.001] * 21,
                "turnover": [0.1] * 21,
            }
        )

        snapshot_date = pd.Timestamp("2019-01-31")
        result = aggregate_trailing_20d_liquidity(df, snapshot_date)

        # Should use exact last 20 dates (indices 1-20), median of 1..20 is 10.5
        # With 20 values (1-20), median is average of 10 and 11 = 10.5
        assert result["spread_pct_median"].item() == 10.5

    def test_trading_days_not_calendar_days(self):
        """Test that 20 trading days ignores calendar -30d approximation."""
        # With calendar -30d, Jan 31 - 30 days = Jan 1
        # But with trading days, Jan 31 minus 20 trading days is around Jan 3-4
        # We need 20 non-weekend trading days ending Jan 31
        dates = pd.to_datetime(
            [
                "2019-01-02",
                "2019-01-03",
                "2019-01-04",  # 3 trading days
                "2019-01-07",
                "2019-01-08",
                "2019-01-09",
                "2019-01-10",
                "2019-01-11",  # 8
                "2019-01-14",
                "2019-01-15",
                "2019-01-16",
                "2019-01-17",
                "2019-01-18",  # 13
                "2019-01-22",
                "2019-01-23",
                "2019-01-24",
                "2019-01-25",
                "2019-01-28",  # 18
                "2019-01-29",
                "2019-01-30",
                "2019-01-31",  # 21
            ]
        )
        df = pd.DataFrame(
            {
                "symbol": ["AAA"] * 21,
                "date": list(dates),
                "spread_pct": list(range(21)),
                "amihud": [0.001] * 21,
                "turnover": [0.1] * 21,
            }
        )

        snapshot_date = pd.Timestamp("2019-01-31")
        result = aggregate_trailing_20d_liquidity(df, snapshot_date)

        # If it used -30 calendar days, it would include Jan 2 (index 0)
        # With exact 20 trading days, index 0 is excluded
        # Median of indices 1-20 (values 1-20) = 10.5
        assert result["spread_pct_median"].item() == 10.5

    def test_symbol_level_exact_20d_aggregation(self):
        """Test that symbol-level medians are computed over exact 20 dates."""
        dates = pd.date_range("2019-01-01", periods=25, freq="D")
        df = pd.DataFrame(
            {
                "symbol": ["AAA"] * 25 + ["BBB"] * 25,
                "date": list(dates) + list(dates),
                "spread_pct": list(range(25)) + list(range(25, 50)),
                "amihud": [0.001] * 50,
                "turnover": [0.1] * 50,
            }
        )

        snapshot_date = pd.Timestamp("2019-01-25")
        result = aggregate_trailing_20d_liquidity(df, snapshot_date)

        # Both AAA and BBB have exactly 25 days, so both included
        assert len(result) == 2

        # AAA: last 20 values are 5-24, median is (14+15)/2 = 14.5
        # BBB: last 20 values are 30-49, median is (39+40)/2 = 39.5
        aaa_median = result.loc[result["symbol"] == "AAA", "spread_pct_median"].item()
        bbb_median = result.loc[result["symbol"] == "BBB", "spread_pct_median"].item()
        assert aaa_median == 14.5
        assert bbb_median == 39.5

    def test_trailing20d_excludes_invalid_rows_before_aggregation(self):
        """Invalid rows should be dropped before trailing-20d medians."""
        dates = pd.date_range("2019-01-01", periods=21, freq="D")
        df = pd.DataFrame(
            {
                "symbol": ["AAA"] * 21,
                "date": list(dates),
                "spread_pct": [0.01] * 20 + [np.nan],  # latest row invalid
                "amihud": [0.001] * 21,
                "turnover": [0.1] * 21,
            }
        )

        snapshot_date = pd.Timestamp("2019-01-21")
        result = aggregate_trailing_20d_liquidity(df, snapshot_date)

        # After dropping invalid latest row, exactly 20 valid distinct dates remain.
        assert len(result) == 1
        assert result["symbol"].item() == "AAA"

    def test_liquidity_score20d_uses_exact_weights(self):
        """LiquidityScore20d must use 0.4/0.4/0.2 weighting exactly."""
        dates = pd.date_range("2019-01-01", periods=21, freq="D")
        df = pd.DataFrame(
            {
                "symbol": ["AAA"] * 21 + ["BBB"] * 21,
                "date": list(dates) + list(dates),
                "spread_pct": [0.01] * 21 + [0.03] * 21,
                "amihud": [0.001] * 21 + [0.01] * 21,
                "turnover": [0.2] * 21 + [0.1] * 21,
            }
        )

        snapshot_date = pd.Timestamp("2019-01-21")
        result = aggregate_trailing_20d_liquidity(df, snapshot_date)

        for _, row in result.iterrows():
            expected = (
                0.4 * (1 - row["spread_rank"])
                + 0.4 * (1 - row["amihud_rank"])
                + 0.2 * row["turnover_rank"]
            )
            assert abs(row["LiquidityScore20d"] - expected) < 1e-12

    def test_duplicate_rows_same_date_do_not_count_toward_eligibility(self):
        """Test that duplicate rows on same date count as 1 trading day, not 2."""
        # Create 20 rows but only 19 distinct dates due to duplicates on Jan 2
        # 19 unique dates = should NOT be eligible (needs 20 distinct)
        dates = pd.to_datetime(
            ["2019-01-02"] * 2  # 2 duplicates on this date
            + list(pd.date_range("2019-01-03", periods=18, freq="D"))  # 18 more = 19 total
        )
        df = pd.DataFrame(
            {
                "symbol": ["AAA"] * 20,
                "date": list(dates),
                "spread_pct": list(range(20)),
                "amihud": [0.001] * 20,
                "turnover": [0.1] * 20,
            }
        )

        snapshot_date = pd.Timestamp("2019-01-31")
        result = aggregate_trailing_20d_liquidity(df, snapshot_date)

        # Should return empty since we only have 19 distinct dates (not 20)
        assert len(result) == 0

    def test_distinct_trading_dates_not_row_count(self):
        """Test that 20 distinct dates is required, not 20 rows."""
        # 20 rows but some are duplicates on same date
        dates = pd.to_datetime(
            ["2019-01-02"] * 2
            + ["2019-01-03"] * 2
            + list(pd.date_range("2019-01-04", periods=16, freq="D"))
        )
        df = pd.DataFrame(
            {
                "symbol": ["AAA"] * 20,
                "date": list(dates),
                "spread_pct": list(range(20)),
                "amihud": [0.001] * 20,
                "turnover": [0.1] * 20,
            }
        )

        snapshot_date = pd.Timestamp("2019-01-31")
        result = aggregate_trailing_20d_liquidity(df, snapshot_date)

        # Should return empty - only 18 distinct dates (not 20)
        assert len(result) == 0

    def test_valid_20_distinct_dates_is_eligible(self):
        """Test that 20 distinct dates makes symbol eligible despite duplicates."""
        # 22 rows but 20 distinct dates - should be eligible
        dates = pd.to_datetime(
            ["2019-01-02"] * 2
            + ["2019-01-03"] * 2
            + list(pd.date_range("2019-01-04", periods=18, freq="D"))
        )
        df = pd.DataFrame(
            {
                "symbol": ["AAA"] * 22,
                "date": list(dates),
                "spread_pct": list(range(22)),
                "amihud": [0.001] * 22,
                "turnover": [0.1] * 22,
            }
        )

        snapshot_date = pd.Timestamp("2019-01-31")
        result = aggregate_trailing_20d_liquidity(df, snapshot_date)

        # Should be eligible - 20 distinct dates
        assert len(result) == 1
        assert result["symbol"].item() == "AAA"

    def test_trailing_window_is_symbol_specific_not_global_dates(self):
        """A symbol with its own latest 20 dates should remain eligible.

        This guards against global-date windowing where missing one market-wide date
        would incorrectly drop an otherwise valid symbol.
        """
        # AAA has 20 distinct dates up to Jan 31 but skips Jan 30.
        aaa_dates = pd.to_datetime(
            [
                "2019-01-01",
                "2019-01-02",
                "2019-01-03",
                "2019-01-04",
                "2019-01-07",
                "2019-01-08",
                "2019-01-09",
                "2019-01-10",
                "2019-01-11",
                "2019-01-14",
                "2019-01-15",
                "2019-01-16",
                "2019-01-17",
                "2019-01-18",
                "2019-01-22",
                "2019-01-23",
                "2019-01-24",
                "2019-01-25",
                "2019-01-29",
                "2019-01-31",
            ]
        )

        # BBB provides the missing Jan 30 and extra market-wide dates.
        bbb_dates = pd.to_datetime(pd.date_range("2019-01-01", periods=25, freq="D"))

        df = pd.DataFrame(
            {
                "symbol": ["AAA"] * len(aaa_dates) + ["BBB"] * len(bbb_dates),
                "date": list(aaa_dates) + list(bbb_dates),
                "spread_pct": [0.02] * len(aaa_dates) + [0.03] * len(bbb_dates),
                "amihud": [0.002] * len(aaa_dates) + [0.003] * len(bbb_dates),
                "turnover": [0.2] * len(aaa_dates) + [0.1] * len(bbb_dates),
            }
        )

        snapshot_date = pd.Timestamp("2019-01-31")
        result = aggregate_trailing_20d_liquidity(df, snapshot_date)

        # AAA should still be eligible with symbol-specific trailing 20 dates.
        assert set(result["symbol"].tolist()) == {"AAA", "BBB"}

    def test_conflicting_duplicate_same_day_rows_are_order_independent(self):
        """Same-day duplicate conflicts should yield deterministic results."""
        base_dates = pd.date_range("2019-01-01", periods=20, freq="D")

        df1 = pd.DataFrame(
            {
                "symbol": ["AAA"] * 22,
                "date": list(base_dates) + [pd.Timestamp("2019-01-10"), pd.Timestamp("2019-01-10")],
                "spread_pct": [0.02] * 20 + [0.90, 0.10],
                "amihud": [0.002] * 20 + [0.90, 0.10],
                "turnover": [0.2] * 20 + [0.90, 0.10],
            }
        )

        # Reverse input order to check order independence.
        df2 = df1.iloc[::-1].reset_index(drop=True)

        snapshot = pd.Timestamp("2019-01-20")
        r1 = aggregate_trailing_20d_liquidity(df1, snapshot)
        r2 = aggregate_trailing_20d_liquidity(df2, snapshot)

        assert r1[
            ["spread_pct_median", "amihud_median", "turnover_median", "LiquidityScore20d"]
        ].equals(r2[["spread_pct_median", "amihud_median", "turnover_median", "LiquidityScore20d"]])
