# Liquidity Retrieval Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Block 4 retrieval evaluation so LiquidSearcher can compare embedding retrieval, return-correlation, liquidity-distance, and hybrid reranking on `Recall@10`, `Spearman`, and `nDCG@10` for 2019 liquidity-peer retrieval.

**Architecture:** Add one focused utility for liquidity labels, one focused metrics module for ranker construction and scoring, one plotting module for output figures, and one CLI script that orchestrates deterministic query sampling, evaluation, and artifact writing. Keep the primary benchmark path separate from the hybrid reranker extension so representation-quality claims remain clean.

**Tech Stack:** Python, pandas, numpy, torch, scikit-learn, matplotlib/seaborn, pytest, mypy, ruff

---

## File Map

### New Files

- `src/evaluation/utils/liquidity_labels.py`
  - Compute `spread_pct`, `amihud`, `turnover`
  - Aggregate daily values to stock-level liquidity features for `2019-01-01` to `2019-12-31`
  - Build `LiquidityScore`, quartiles, graded relevance helpers, and point-in-time `LiquidityScore20d`

- `src/evaluation/metrics/retrieval.py`
  - Deterministic query sampling
  - Candidate-universe intersection logic
  - Ranker construction for embedding, correlation, liquidity-distance, hybrid extension, and correlation_rerank
  - `Recall@10`, `Spearman`, `nDCG@10`, optional `Precision@10`

- `src/evaluation/visualizations/retrieval_plots.py`
  - Overall metric bar charts
  - Sector and liquidity-quartile grouped plots

- `scripts/evaluation/run_retrieval_metrics.py`
  - End-to-end CLI for Block 4
  - Writes metrics CSVs to `results/metrics/`
  - Writes retrieval artifacts to `results/retrieval/`

- `tests/test_evaluation_liquidity_labels.py`
  - Unit tests for liquidity proxy computation, aggregation, quartiles, and graded relevance

- `tests/test_evaluation_retrieval_metrics.py`
  - Unit tests for query sampling, candidate eligibility, metric computation, and rankers on small deterministic fixtures

### Modified Files

- `src/evaluation/utils/__init__.py`
  - Export liquidity label helpers

- `src/evaluation/metrics/__init__.py`
  - Export retrieval metric helpers

- `src/evaluation/visualizations/__init__.py`
  - Export retrieval plot helpers

- `src/evaluation/__init__.py`
  - Export top-level retrieval interfaces if that matches current pattern

---

### Task 1: Add Liquidity Label Utilities

**Files:**
- Create: `src/evaluation/utils/liquidity_labels.py`
- Modify: `src/evaluation/utils/__init__.py`
- Test: `tests/test_evaluation_liquidity_labels.py`

- [ ] **Step 1: Write the failing liquidity utility tests**

```python
import pandas as pd

from src.evaluation.utils.liquidity_labels import (
    compute_daily_liquidity_proxies,
    aggregate_period_liquidity,
    assign_liquidity_quartiles,
    compute_graded_relevance,
)


def test_compute_daily_liquidity_proxies_adds_expected_columns():
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


def test_aggregate_period_liquidity_builds_liquidity_score():
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
    assert result.loc[result["symbol"] == "AAA", "LiquidityScore"].item() > result.loc[
        result["symbol"] == "BBB", "LiquidityScore"
    ].item()


def test_compute_graded_relevance_respects_distance_bands():
    query_score = 0.80
    candidate_scores = pd.Series([0.79, 0.70, 0.55, 0.10], index=["A", "B", "C", "D"])
    quartiles = pd.Series([3, 3, 2, 0], index=["A", "B", "C", "D"])

    graded = compute_graded_relevance(query_score, candidate_scores, quartiles, query_quartile=3)

    assert graded["A"] >= graded["B"]
    assert graded["D"] == 0
```

- [ ] **Step 2: Run the new liquidity tests and confirm they fail**

Run: `python -m pytest tests/test_evaluation_liquidity_labels.py -v`

Expected: `ImportError` or missing-function failures for `liquidity_labels.py`

- [ ] **Step 3: Implement minimal liquidity proxy helpers**

Add `src/evaluation/utils/liquidity_labels.py` with functions shaped like:

```python
def compute_daily_liquidity_proxies(df: pd.DataFrame) -> pd.DataFrame: ...

def aggregate_period_liquidity(df: pd.DataFrame) -> pd.DataFrame: ...

def assign_liquidity_quartiles(scores: pd.Series) -> pd.Series: ...

def compute_graded_relevance(
    query_score: float,
    candidate_scores: pd.Series,
    candidate_quartiles: pd.Series,
    query_quartile: int,
) -> pd.Series: ...

def aggregate_trailing_20d_liquidity(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame: ...
```

Implementation notes:
- filter impossible denominators before division
- use medians for period aggregation
- percentile-rank cross-sectionally before building `LiquidityScore`
- keep functions pure and dataframe-in/dataframe-out

- [ ] **Step 4: Export the new helpers**

Update `src/evaluation/utils/__init__.py` to export the public liquidity label functions.

- [ ] **Step 5: Run the liquidity tests again**

Run: `python -m pytest tests/test_evaluation_liquidity_labels.py -v`

Expected: PASS

- [ ] **Step 6: Typecheck and lint the liquidity module**

Run: `python -m mypy src/evaluation/utils/liquidity_labels.py`

Run: `python -m ruff check src/evaluation/utils/liquidity_labels.py tests/test_evaluation_liquidity_labels.py`

Expected: no errors

---

### Task 2: Add Retrieval Metrics Core

**Files:**
- Create: `src/evaluation/metrics/retrieval.py`
- Modify: `src/evaluation/metrics/__init__.py`
- Test: `tests/test_evaluation_retrieval_metrics.py`

- [ ] **Step 1: Write failing tests for candidate eligibility and metric math**

```python
import numpy as np
import pandas as pd

from src.evaluation.metrics.retrieval import (
    filter_valid_candidates,
    recall_at_k,
    ndcg_at_k,
    spearman_against_reference,
)


def test_filter_valid_candidates_keeps_query_specific_intersection():
    df = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "has_embedding": [True, True, False],
            "has_liquidity_label": [True, False, True],
            "correlation_overlap": [60, 60, 60],
        }
    )

    result = filter_valid_candidates(df)

    assert result["symbol"].tolist() == ["A"]


def test_recall_at_k_counts_binary_hits():
    relevance = pd.Series([1, 0, 1, 0], index=["A", "B", "C", "D"])
    ranking = ["A", "B", "D", "C"]

    assert recall_at_k(ranking, relevance, k=2) == 0.5


def test_ndcg_at_k_rewards_better_ordering():
    graded = pd.Series([3, 2, 0], index=["A", "B", "C"])
    assert ndcg_at_k(["A", "B", "C"], graded, k=3) > ndcg_at_k(["B", "A", "C"], graded, k=3)
```

- [ ] **Step 2: Run retrieval tests and confirm they fail**

Run: `python -m pytest tests/test_evaluation_retrieval_metrics.py -v`

Expected: import failures because `retrieval.py` does not exist yet

- [ ] **Step 3: Implement core metric helpers**

Add `src/evaluation/metrics/retrieval.py` with focused functions:

```python
def filter_valid_candidates(candidates: pd.DataFrame) -> pd.DataFrame: ...

def recall_at_k(ranking: list[str], binary_relevance: pd.Series, k: int = 10) -> float: ...

def ndcg_at_k(ranking: list[str], graded_relevance: pd.Series, k: int = 10) -> float: ...

def spearman_against_reference(
    ranking_scores: pd.Series,
    reference_scores: pd.Series,
) -> float: ...
```

Implementation notes:
- use the query-specific intersection rule from the spec
- if no relevant items exist for a query, return `np.nan` and drop it from means later
- use `sklearn.metrics.ndcg_score` only if it keeps implementation simple; otherwise compute directly

- [ ] **Step 4: Add deterministic query sampling tests**

Extend `tests/test_evaluation_retrieval_metrics.py` with:

```python
from src.evaluation.metrics.retrieval import sample_query_set


def test_sample_query_set_is_deterministic_for_seed_42():
    df = pd.DataFrame(
        {
            "symbol": [f"S{i}" for i in range(30)],
            "gsector": [10] * 10 + [20] * 10 + [30] * 10,
            "market_cap_tier": [0, 1, 2] * 10,
            "liquidity_quartile": [0, 1, 2, 3, 0] * 6,
        }
    )

    first = sample_query_set(df, n_queries=12, seed=42)
    second = sample_query_set(df, n_queries=12, seed=42)

    assert first["symbol"].tolist() == second["symbol"].tolist()
```

- [ ] **Step 5: Implement query sampling and ranker-score helpers**

Add more functions:

```python
def sample_query_set(snapshot_df: pd.DataFrame, n_queries: int = 50, seed: int = 42) -> pd.DataFrame: ...

def build_correlation_scores(
    returns_wide: pd.DataFrame,
    query_symbol: str,
    snapshot_date: pd.Timestamp,
    lookback: int = 60,
    min_overlap: int = 40,
) -> pd.Series: ...

def normalize_scores(scores: pd.Series) -> pd.Series: ...

def build_hybrid_scores(
    embedding_scores: pd.Series,
    liquidity20d_scores: pd.Series,
    alpha: float = 0.7,
) -> pd.Series: ...
```

Sampling implementation requirements from the spec:
- keep only eligible query stocks with valid embedding inputs, valid liquidity labels, and at least 60 trailing returns
- compute `market_cap_tier` from cross-sectional terciles
- form strata as `(gsector, market_cap_tier)`
- use proportional allocation with minimum 1 query for any stratum with at least 10 eligible stocks
- use seed `42` for deterministic sampling, fill, and trim behavior

- [ ] **Step 6: Add an end-to-end small ranking test**

Add one deterministic fixture where:
- one candidate should rank first by liquidity-distance
- correlation ranking differs
- hybrid reranking only reorders inside a top-50 shortlist mock

Assert metric outputs are numerically stable.

- [ ] **Step 7: Add missing-hybrid-liquidity edge-case test**

Add a test that verifies:
- a candidate in the embedding top-50 shortlist with missing trailing-20-day liquidity inputs is **not dropped**
- its hybrid liquidity similarity is set to the lowest possible value
- the shortlist membership remains unchanged

- [ ] **Step 8: Re-run retrieval tests**

Run: `python -m pytest tests/test_evaluation_retrieval_metrics.py -v`

Expected: PASS

- [ ] **Step 9: Export retrieval helpers**

Update `src/evaluation/metrics/__init__.py` with retrieval exports.

- [ ] **Step 10: Typecheck and lint retrieval core**

Run: `python -m mypy src/evaluation/metrics/retrieval.py`

Run: `python -m ruff check src/evaluation/metrics/retrieval.py tests/test_evaluation_retrieval_metrics.py`

Expected: no errors

---

### Task 3: Add Embedding Snapshot Builder For Evaluation

**Files:**
- Modify: `src/evaluation/metrics/retrieval.py`
- Test: `tests/test_evaluation_retrieval_metrics.py`

- [ ] **Step 1: Write a failing test for snapshot assembly**

```python
from src.evaluation.metrics.retrieval import build_snapshot_frame


def test_build_snapshot_frame_keeps_last_2019_row_per_symbol():
    ...
```

Test requirements:
- one symbol with two 2019 rows returns the last date
- non-2019 rows are excluded
- required columns survive into the snapshot

- [ ] **Step 2: Run the snapshot test and confirm failure**

Run: `python -m pytest tests/test_evaluation_retrieval_metrics.py -k snapshot -v`

Expected: missing `build_snapshot_frame`

- [ ] **Step 3: Implement snapshot and embedding helpers**

Add to `src/evaluation/metrics/retrieval.py`:

```python
def build_snapshot_frame(period_df: pd.DataFrame) -> pd.DataFrame: ...

def prepare_model_inputs(row: pd.Series) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

def compute_embedding_scores(
    snapshot_df: pd.DataFrame,
    checkpoint_path: str | Path,
) -> pd.DataFrame: ...
```

Implementation notes:
- reuse the same feature-preparation logic style already present in `src/evaluation/stock_similarity.py`
- keep the model loading isolated in one helper
- store one joint embedding vector per symbol in the snapshot frame

- [ ] **Step 4: Run targeted tests for snapshot logic**

Run: `python -m pytest tests/test_evaluation_retrieval_metrics.py -k snapshot -v`

Expected: PASS

- [ ] **Step 5: Run mypy and ruff on the updated retrieval module**

Run: `python -m mypy src/evaluation/metrics/retrieval.py`

Run: `python -m ruff check src/evaluation/metrics/retrieval.py`

Expected: no errors

---

### Task 4: Add Retrieval Plotting Utilities

**Files:**
- Create: `src/evaluation/visualizations/retrieval_plots.py`
- Modify: `src/evaluation/visualizations/__init__.py`
- Test: `tests/test_evaluation_retrieval_metrics.py`

- [ ] **Step 1: Write a failing smoke test for figure generation**

```python
from pathlib import Path
import pandas as pd

from src.evaluation.visualizations.retrieval_plots import plot_overall_metrics


def test_plot_overall_metrics_writes_png(tmp_path: Path):
    metrics = pd.DataFrame(
        {
            "ranker": ["embedding", "correlation"],
            "Recall@10": [0.6, 0.4],
            "Spearman": [0.3, 0.1],
            "nDCG@10": [0.7, 0.5],
        }
    )

    output_path = tmp_path / "overall.png"
    plot_overall_metrics(metrics, output_path)

    assert output_path.exists()
```

- [ ] **Step 2: Run the plotting smoke test and confirm failure**

Run: `python -m pytest tests/test_evaluation_retrieval_metrics.py -k plot -v`

Expected: missing plotting module or function

- [ ] **Step 3: Implement minimal plotting helpers**

Add `src/evaluation/visualizations/retrieval_plots.py` with:

```python
def plot_overall_metrics(metrics_df: pd.DataFrame, output_path: Path) -> None: ...

def plot_grouped_metrics(metrics_df: pd.DataFrame, group_col: str, output_path: Path) -> None: ...
```

Implementation notes:
- keep styling consistent with existing SHAP/UMAP plotting choices
- accept precomputed dataframes; do not mix plotting with metric computation

- [ ] **Step 4: Export the plotting helpers**

Update `src/evaluation/visualizations/__init__.py`.

- [ ] **Step 5: Re-run the plotting smoke test**

Run: `python -m pytest tests/test_evaluation_retrieval_metrics.py -k plot -v`

Expected: PASS

- [ ] **Step 6: Lint the plotting module**

Run: `python -m ruff check src/evaluation/visualizations/retrieval_plots.py`

Expected: no errors

---

### Task 5: Add Block 4 CLI Script

**Files:**
- Create: `scripts/evaluation/run_retrieval_metrics.py`
- Modify: `src/evaluation/__init__.py`
- Test: `tests/test_evaluation_retrieval_metrics.py`

- [ ] **Step 1: Write a failing CLI smoke test**

Use `pytest` plus `subprocess` or direct function call to assert that a small synthetic fixture run writes:
- `results/metrics/recall_spearman.csv`
- `results/metrics/retrieval_metrics_overall.csv`
- `results/metrics/retrieval_metrics_by_sector.csv`
- `results/metrics/retrieval_metrics_by_market_cap.csv`
- `results/metrics/retrieval_metrics_by_liquidity_quartile.csv`
- `results/retrieval/query_manifest.csv`
- `results/retrieval/per_query/*.parquet`

Example skeleton:

```python
def test_run_retrieval_metrics_writes_expected_outputs(tmp_path: Path):
    ...
```

The CLI test should also assert that at least one per-query parquet contains the required columns:
- `query_ticker`
- `query_date`
- `candidate_ticker`
- `embedding_score`
- `correlation_score`
- `liquidity_distance_score`
- `binary_relevance`
- `graded_relevance`
- `embedding_rank`
- `correlation_rank`
- `liquidity_distance_rank`

If `--run-hybrid` is enabled, also require:
- `hybrid_score`
- `hybrid_rank`

- [ ] **Step 2: Run the CLI smoke test and confirm failure**

Run: `python -m pytest tests/test_evaluation_retrieval_metrics.py -k cli -v`

Expected: module or file not found

- [ ] **Step 3: Implement the CLI entrypoint**

Add `scripts/evaluation/run_retrieval_metrics.py` with arguments:

```python
--checkpoint checkpoints/last.ckpt
--features data/processed/all_features.parquet
--period-start 2019-01-01
--period-end 2019-12-31
--n-queries 50
--seed 42
--output-dir results
--run-hybrid
```

Main flow:
- load period data using `FeatureLoader`
- compute period-level liquidity labels
- build last-date snapshot per symbol
- compute embeddings for snapshot
- sample deterministic query set
- evaluate primary rankers on shared candidate intersections
- optionally evaluate hybrid extension on embedding shortlist
- write summary CSVs to `results/metrics/`
- write query manifest, per-query parquet files, and figures to `results/retrieval/`

Reporting requirement:
- primary benchmark table contains only `embedding`, `correlation`, `liquidity_distance`
- hybrid metrics are written to a separate extension table/file, for example `results/metrics/retrieval_metrics_hybrid.csv`

- [ ] **Step 4: Wire top-level exports only if needed**

If useful and consistent with current patterns, add public imports in `src/evaluation/__init__.py`. Keep this minimal.

- [ ] **Step 5: Re-run the CLI smoke test**

Run: `python -m pytest tests/test_evaluation_retrieval_metrics.py -k cli -v`

Expected: PASS

- [ ] **Step 6: Syntax, lint, and typecheck the CLI**

Run: `python -m py_compile scripts/evaluation/run_retrieval_metrics.py`

Run: `python -m ruff check scripts/evaluation/run_retrieval_metrics.py`

Expected: no errors

---

### Task 6: Run Real Verification On A Small Slice

**Files:**
- Modify: `scripts/evaluation/run_retrieval_metrics.py` only if verification exposes a real bug
- Test: `tests/test_evaluation_liquidity_labels.py`, `tests/test_evaluation_retrieval_metrics.py`

- [ ] **Step 1: Run focused unit tests**

Run: `python -m pytest tests/test_evaluation_liquidity_labels.py tests/test_evaluation_retrieval_metrics.py -v`

Expected: PASS

- [ ] **Step 2: Run a small real-data smoke test**

Run:

```bash
uv run python -m scripts.evaluation.run_retrieval_metrics \
    --checkpoint checkpoints/last.ckpt \
    --features data/processed/all_features.parquet \
    --period-start 2019-01-01 \
    --period-end 2019-12-31 \
    --n-queries 5 \
    --seed 42 \
    --output-dir results_smoke \
    --run-hybrid
```

Expected:
- summary CSVs written
- `results_smoke/metrics/recall_spearman.csv` written
- `results_smoke/metrics/retrieval_metrics_overall.csv` written
- `results_smoke/metrics/retrieval_metrics_by_sector.csv` written
- `results_smoke/metrics/retrieval_metrics_by_market_cap.csv` written
- `results_smoke/metrics/retrieval_metrics_by_liquidity_quartile.csv` written
- `results_smoke/metrics/retrieval_metrics_hybrid.csv` written when `--run-hybrid` is enabled
- `results_smoke/retrieval/per_query/*.parquet` written
- `results_smoke/retrieval/query_manifest.csv` written
- `results_smoke/retrieval/figures/*.png` written
- no crashes on candidate-intersection logic

- [ ] **Step 3: Inspect smoke-test outputs**

Check:
- `results_smoke/retrieval/query_manifest.csv`
- `results_smoke/retrieval/per_query/`
- `results_smoke/metrics/recall_spearman.csv`

Expected:
- ranker rows include `embedding`, `correlation`, `liquidity_distance`
- hybrid metrics live in a separate extension file, not mixed into the primary benchmark rows
- metric values are finite where applicable
- per-query parquet files contain the required score/rank/relevance columns

- [ ] **Step 4: Run one file-scoped lint/typecheck pass for touched files**

Run:

```bash
python -m mypy src/evaluation/utils/liquidity_labels.py src/evaluation/metrics/retrieval.py
python -m ruff check src/evaluation/utils/liquidity_labels.py src/evaluation/metrics/retrieval.py src/evaluation/visualizations/retrieval_plots.py scripts/evaluation/run_retrieval_metrics.py
```

Expected: no errors

---

### Task 7: Run Full Block 4 Evaluation

**Files:**
- No new code unless smoke test exposed bugs

- [ ] **Step 1: Run full evaluation on 50 queries**

Run:

```bash
uv run python -m scripts.evaluation.run_retrieval_metrics \
    --checkpoint checkpoints/last.ckpt \
    --features data/processed/all_features.parquet \
    --period-start 2019-01-01 \
    --period-end 2019-12-31 \
    --n-queries 50 \
    --seed 42 \
    --output-dir results \
    --run-hybrid
```

- [ ] **Step 2: Inspect final artifacts**

Confirm presence of:
- `results/retrieval/query_manifest.csv`
- `results/retrieval/per_query/*.parquet`
- `results/retrieval/figures/*.png`
- `results/metrics/recall_spearman.csv`
- `results/metrics/retrieval_metrics_overall.csv`
- `results/metrics/retrieval_metrics_by_sector.csv`
- `results/metrics/retrieval_metrics_by_market_cap.csv`
- `results/metrics/retrieval_metrics_by_liquidity_quartile.csv`
- `results/metrics/retrieval_metrics_hybrid.csv` when `--run-hybrid` is enabled

- [ ] **Step 3: Summarize final outcomes for Block 4**

Capture:
- which primary ranker wins on `Recall@10`
- which primary ranker wins on `Spearman`
- which primary ranker wins on `nDCG@10`
- whether hybrid improves practical retrieval

---

## Notes For The Implementer

- Reuse existing patterns from `src/evaluation/stock_similarity.py` for model loading and input preparation instead of inventing a new model-loading style.
- Do not add new data preprocessing pipelines for this block; all required fields already exist in `data/processed/all_features.parquet`.
- Keep the hybrid reranker logically separated from the primary benchmark path in both code and outputs.
- Do not overbuild a class hierarchy. Small pure functions are sufficient here.
- If a helper grows beyond ~100-150 lines, split by responsibility rather than adding more flags.

## Definition Of Done

Block 4 is complete when:
- liquidity labels are computed reproducibly for 2019
- deterministic query sampling works with seed `42`
- candidate-universe intersection is enforced for primary rankers
- `Recall@10`, `Spearman`, and `nDCG@10` are computed and written
- primary outputs and per-query artifacts are generated
- hybrid reranker runs only as a separate extension path
- targeted tests pass
- small real-data smoke test passes
