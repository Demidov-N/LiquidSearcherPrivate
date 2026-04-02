# Separate Temporal/Tabular Embeddings + Multi-Characteristic Evaluation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate temporal and tabular embeddings separately as individual rankers, and score them against multiple ground truth characteristics (liquidity uplift, return similarity, fundamentals, etc.) to identify what the model is actually good for.

**Architecture:** 
1. Create separate rankers: `temporal_only` (128-dim), `tabular_only` (128-dim), `joint` (256-dim existing)
2. Remove pearson correlation baseline
3. Evaluate against multiple ground truth signals:
   - LiquidityUplift (positive = more liquid than query)
   - ReturnSimilarity (120-day Pearson correlation)
   - SectorSimilarity (GICS match)
   - FundamentalsSimilarity (market cap, other fundamentals)
   - Individual characteristics: liquidity, spread, turnover, volatility, beta
4. Report Recall@10 and nDCG@10 separately for each
5. Use top quartile (75th percentile) as similarity threshold
6. Clearly report threshold limits in outputs

**Tech Stack:** PyTorch, pandas, existing evaluation pipeline

---

## Key Design Decisions

### Rankers (4 total)
| Ranker | Description |
|--------|-------------|
| `temporal_only` | 128-dim temporal encoder (price/volume patterns) |
| `tabular_only` | 128-dim tabular encoder (fundamentals + GICS) |
| `joint` | 256-dim concatenated (existing) |
| `tabular_rerank` | tabular top-50 reranked by liquidity |

**Note:** NO pearson correlation, NO spearman correlation, NO embedding_rerank (keep it simple)

### Ground Truth References (6 total)
| Reference | What it measures | Threshold |
|-----------|------------------|-----------|
| `LiquidityUplift` | candidate_liquidity > query_liquidity (positive uplift) | uplift > 0 |
| `ReturnSimilarity` | 120-day Pearson return correlation | top quartile (≥75th percentile) |
| `SectorSimilarity` | GICS sector/group match | same ggroup=1.0, same sector=0.5 |
| `FundamentalsSim` | Market cap + other fundamentals similarity | top quartile |
| `LiquidityChar` | LiquidityScore similarity | top quartile |
| `TurnoverChar` | Turnover rate similarity | top quartile |

### Scoring
- **Recall@10**: Binary - is the stock in the ground truth top quartile?
- **nDCG@10**: Graded - how well ranked are the top quartile stocks?
- Report SEPARATELY - no averaging

---

## File Structure

**Files to Modify:**
- `src/models/dual_encoder.py` - Add inference mode for separate embeddings
- `scripts/evaluation/run_retrieval_metrics.py` - Major refactor for new rankers and references
- `src/evaluation/ground_truth.py` - Add multi-reference scoring functions

**No new files needed**

---

## Task 1: Add Inference Methods for Separate Embeddings

**Files:**
- Modify: `src/models/dual_encoder.py`

- [ ] **Step 1: Add get_temporal_embedding method**

```python
def get_temporal_embedding(self, price_data):
    """Get only temporal embedding for inference.
    
    Args:
        price_data: (batch, 60, 13) OHLCV features
        
    Returns:
        (batch, 128) temporal embeddings
    """
    self.eval()
    with torch.no_grad():
        return self.temporal_encoder(price_data)
```

- [ ] **Step 2: Add get_tabular_embedding method**

```python
def get_tabular_embedding(self, fundamentals, categorical):
    """Get only tabular embedding for inference.
    
    Args:
        fundamentals: (batch, 15) continuous features
        categorical: (batch, 2) [gsector, ggroup] indices
        
    Returns:
        (batch, 128) tabular embeddings
    """
    self.eval()
    with torch.no_grad():
        return self.tabular_encoder(fundamentals, categorical)
```

- [ ] **Step 3: Verify syntax**

```bash
python -m py_compile src/models/dual_encoder.py
```

- [ ] **Step 4: Commit**

```bash
git add src/models/dual_encoder.py
git commit -m "feat: add separate temporal and tabular embedding inference methods"
```

---

## Task 2: Create Ground Truth Reference Functions

**Files:**
- Modify: `src/evaluation/ground_truth.py`

- [ ] **Step 1: Add compute_return_similarity_120d (if not exists)**

Should already exist from previous work. Verify it's there and working.

- [ ] **Step 2: Add compute_fundamentals_similarity**

```python
def compute_fundamentals_similarity(
    query_symbol: str,
    snapshot_df: pd.DataFrame,
    fund_columns: list[str] = None,
) -> pd.Series:
    """Compute fundamentals similarity (market cap, etc.).
    
    Uses percentile rank similarity on fundamental characteristics.
    
    Args:
        query_symbol: Query symbol
        snapshot_df: DataFrame with fundamental columns
        fund_columns: List of columns to use. If None, uses ['market_cap']
        
    Returns:
        Series indexed by symbol with fundamentals similarity [0,1]
    """
    if fund_columns is None:
        fund_columns = ['market_cap']  # Default to market cap
    
    # Filter to available columns
    available_cols = [c for c in fund_columns if c in snapshot_df.columns]
    if not available_cols:
        return pd.Series(dtype=float)
    
    # Get query row
    query_row = snapshot_df[snapshot_df['symbol'] == query_symbol]
    if len(query_row) == 0:
        return pd.Series(dtype=float)
    
    # Compute composite across available fundamentals
    candidates = snapshot_df[snapshot_df['symbol'] != query_symbol].copy()
    
    similarities = pd.Series(0.0, index=candidates['symbol'])
    
    for col in available_cols:
        # Drop NaN for this column
        valid = candidates.dropna(subset=[col])
        if len(valid) == 0:
            continue
        
        # Compute percentile ranks
        ranks = valid[col].rank(pct=True)
        query_rank = query_row[col].iloc[0]
        
        # Similarity = 1 - abs(rank difference)
        col_sim = 1.0 - (ranks - query_rank).abs()
        col_sim.index = valid['symbol']
        
        # Add to composite (average)
        similarities = similarities.add(col_sim, fill_value=0)
    
    # Average across fundamentals
    similarities = similarities / len(available_cols)
    
    return similarities
```

- [ ] **Step 3: Add compute_characteristic_references (multi-reference)**

```python
def compute_characteristic_references(
    query_symbol: str,
    snapshot_df: pd.DataFrame,
    returns_df: pd.DataFrame = None,
) -> dict[str, pd.Series]:
    """Compute all ground truth reference scores.
    
    Returns dict with keys:
    - 'LiquidityUplift': candidate_liq - query_liq (positive = better)
    - 'ReturnSimilarity': 120-day Pearson correlation [0,1] normalized
    - 'SectorSimilarity': 1.0 same ggroup, 0.5 same sector, 0.0 different
    - 'FundamentalsSim': market cap and fundamentals similarity [0,1]
    - 'LiquidityChar': LiquidityScore percentile similarity [0,1]
    - 'TurnoverChar': Turnover percentile similarity [0,1]
    
    Args:
        query_symbol: Query symbol
        snapshot_df: DataFrame with all characteristics
        returns_df: Optional returns DataFrame for return similarity
        
    Returns:
        Dictionary mapping reference name to similarity/uplift Series
    """
    results = {}
    
    # 1. LiquidityUplift (raw difference, positive = better)
    if 'LiquidityScore' in snapshot_df.columns:
        query_liq = snapshot_df[snapshot_df['symbol'] == query_symbol]['LiquidityScore'].iloc[0]
        candidates = snapshot_df[snapshot_df['symbol'] != query_symbol]
        results['LiquidityUplift'] = candidates.set_index('symbol')['LiquidityScore'] - query_liq
    
    # 2. ReturnSimilarity (if returns_df provided)
    if returns_df is not None and query_symbol in returns_df.columns:
        results['ReturnSimilarity'] = compute_return_similarity_120d(
            returns_df, query_symbol, min_overlap=80
        )
    
    # 3. SectorSimilarity
    query_row = snapshot_df[snapshot_df['symbol'] == query_symbol]
    if len(query_row) > 0 and 'gsector' in query_row.columns:
        query_gsector = query_row['gsector'].iloc[0]
        query_ggroup = query_row['ggroup'].iloc[0]
        results['SectorSimilarity'] = compute_sector_similarity(
            query_gsector, query_ggroup, snapshot_df
        )
    
    # 4. FundamentalsSim
    results['FundamentalsSim'] = compute_fundamentals_similarity(
        query_symbol, snapshot_df, fund_columns=['market_cap']
    )
    
    # 5. LiquidityChar (LiquidityScore percentile similarity)
    if 'LiquidityScore' in snapshot_df.columns:
        liq_ranks = snapshot_df['LiquidityScore'].rank(pct=True)
        query_liq_rank = liq_ranks[snapshot_df['symbol'] == query_symbol].iloc[0]
        liq_sim = 1.0 - (liq_ranks - query_liq_rank).abs()
        liq_sim.index = snapshot_df['symbol']
        results['LiquamentalsChar'] = liq_sim.drop(query_symbol, errors='ignore')
    
    # 6. TurnoverChar (if turnover available)
    if 'turnover' in snapshot_df.columns or 'turnover_rank' in snapshot_df.columns:
        col = 'turnover_rank' if 'turnover_rank' in snapshot_df.columns else 'turnover'
        turn_ranks = snapshot_df[col].rank(pct=True)
        query_turn_rank = turn_ranks[snapshot_df['symbol'] == query_symbol].iloc[0]
        turn_sim = 1.0 - (turn_ranks - query_turn_rank).abs()
        turn_sim.index = snapshot_df['symbol']
        results['TurnoverChar'] = turn_sim.drop(query_symbol, errors='ignore')
    
    return results
```

- [ ] **Step 4: Add build_binary_relevance_with_threshold** 

```python
def build_binary_relevance_with_threshold(
    scores: pd.Series,
    threshold_type: str = 'top_quartile',
    custom_threshold: float = None,
) -> tuple[pd.Series, float]:
    """Build binary relevance with explicit threshold reporting.
    
    Args:
        scores: Series of scores indexed by symbol
        threshold_type: 'top_quartile', 'positive', or 'custom'
        custom_threshold: If threshold_type='custom', use this value
        
    Returns:
        Tuple of (binary_relevance, threshold_value_used)
    """
    if len(scores) == 0:
        return pd.Series(dtype=int), np.nan
    
    if threshold_type == 'top_quartile':
        threshold = scores.quantile(0.75)
        binary = (scores >= threshold).astype(int)
    elif threshold_type == 'positive':
        threshold = 0.0
        binary = (scores > threshold).astype(int)
    elif threshold_type == 'custom':
        threshold = custom_threshold if custom_threshold is not None else scores.median()
        binary = (scores >= threshold).astype(int)
    else:
        raise ValueError(f"Unknown threshold_type: {threshold_type}")
    
    return binary, threshold
```

- [ ] **Step 5: Verify syntax**

```bash
python -m py_compile src/evaluation/ground_truth.py
```

- [ ] **Step 6: Commit**

```bash
git add src/evaluation/ground_truth.py
git commit -m "feat: add multi-reference ground truth functions with threshold reporting"
```

---

## Task 3: Refactor Evaluation Script - New Rankers Only

**Files:**
- Modify: `scripts/evaluation/run_retrieval_metrics.py`

- [ ] **Step 1: Define new ranker list**

Replace the existing ranker definitions with:

```python
# NEW: 4 rankers - temporal, tabular, joint, tabular_rerank
RANKERS = {
    "temporal_only": "Temporal-only (price/volume)",
    "tabular_only": "Tabular-only (fundamentals/GICS)",
    "joint": "Joint (concatenated)",
    "tabular_rerank": "Tabular + Liquidity Rerank",
}

# REMOVE: pearson_corr, spearman_corr, embedding, embedding_rerank, pearson_corr_rerank, spearman_corr_rerank
```

- [ ] **Step 2: Create compute_temporal_only_scores function**

```python
def compute_temporal_only_scores(
    snapshot_df: pd.DataFrame,
    period_df: pd.DataFrame,
    query_symbols: list[str],
    checkpoint_path: str,
    device: str = "cpu",
) -> dict[str, pd.Series]:
    """Compute temporal-only embedding similarities.
    
    Returns dict mapping query -> Series of temporal similarities.
    """
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hyperparams = checkpoint.get("hyper_parameters", {})
    
    model = DualEncoder(
        temporal_input_dim=hyperparams.get("temporal_input_dim", 13),
        tabular_continuous_dim=hyperparams.get("tabular_continuous_dim", 15),
        embedding_dim=hyperparams.get("embedding_dim", 128),
    )
    state_dict = checkpoint.get("state_dict", checkpoint)
    if any(k.startswith("model.") for k in state_dict):
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    
    # Compute temporal embeddings for all symbols
    temporal_embs = {}
    
    for symbol in snapshot_df["symbol"]:
        symbol_data = period_df[period_df["symbol"] == symbol].sort_values("date").tail(60)
        if len(symbol_data) == 0:
            continue
        
        # Build temporal features only
        temporal = torch.zeros(1, 60, 13)
        temporal_cols = [...]  # Fill in from data_module
        for i, col in enumerate(temporal_cols):
            if col in symbol_data.columns:
                val = symbol_data[col].iloc[-1] if not symbol_data[col].isna().all() else 0
                temporal[0, -1, i] = float(val) if not pd.isna(val) else 0.0
        
        with torch.no_grad():
            temporal_emb = model.get_temporal_embedding(temporal.to(device))
            temporal_embs[symbol] = temporal_emb.cpu()
    
    # Compute similarities for each query
    scores_dict = {}
    symbol_list = list(temporal_embs.keys())
    
    for query in query_symbols:
        if query not in temporal_embs:
            continue
        
        query_emb = temporal_embs[query]
        candidates = [s for s in symbol_list if s != query]
        
        sims = []
        for candidate in candidates:
            cand_emb = temporal_embs[candidate]
            sim = F.cosine_similarity(query_emb, cand_emb, dim=-1).item()
            sims.append(sim)
        
        scores_dict[query] = pd.Series(sims, index=candidates)
    
    return scores_dict
```

- [ ] **Step 3: Create compute_tabular_only_scores function**

Similar structure but using `get_tabular_embedding`.

- [ ] **Step 4: Modify main evaluation loop to use only 4 rankers**

Replace the existing 6-ranker loop with 4-ranker loop:

```python
# NEW: 4 rankers only
ranker_scores = {}

# Ranker 1: Temporal-only
ranker_scores["temporal_only"] = compute_temporal_only_scores(...)

# Ranker 2: Tabular-only  
ranker_scores["tabular_only"] = compute_tabular_only_scores(...)

# Ranker 3: Joint (existing method)
ranker_scores["joint"] = compute_joint_embedding_scores(...)

# Ranker 4: Tabular rerank (tabular top-50 + liquidity rerank)
ranker_scores["tabular_rerank"] = compute_tabular_rerank_scores(...)
```

- [ ] **Step 5: Compute multi-reference ground truths**

```python
# For each query, compute all 6 references
references = compute_characteristic_references(
    query_symbol=query,
    snapshot_df=snapshot_df,
    returns_df=returns_120d,
)

# Build relevance for each reference with threshold reporting
relevance = {}
thresholds = {}

for ref_name, ref_scores in references.items():
    if ref_name == 'LiquidityUplift':
        # Binary: positive = relevant
        rel, thresh = build_binary_relevance_with_threshold(
            ref_scores, threshold_type='positive'
        )
    else:
        # Top quartile = relevant
        rel, thresh = build_binary_relevance_with_threshold(
            ref_scores, threshold_type='top_quartile'
        )
    
    relevance[ref_name] = rel
    thresholds[ref_name] = thresh
```

- [ ] **Step 6: Compute and store metrics separately**

```python
# For each ranker and each reference, compute Recall@10 and nDCG@10 separately
for ranker_name in RANKERS.keys():
    for ref_name in references.keys():
        # Get binary and graded relevance
        binary_rel = relevance[ref_name]
        graded_rel = build_graded_relevance(references[ref_name])  # Your existing function
        
        # Compute metrics
        recall = recall_at_k(ranking_list, binary_rel, k=10)
        ndcg = ndcg_at_k(ranking_list, graded_rel, k=10)
        
        # Store separately (NOT averaged)
        query_result[f"{ranker_name}_{ref_name}_Recall@10"] = recall
        query_result[f"{ranker_name}_{ref_name}_nDCG@10"] = ndcg

# Also store thresholds used
query_result["thresholds"] = thresholds  # Dict of threshold values
```

- [ ] **Step 7: Update CSV outputs with threshold reporting**

```python
# Create summary CSV with thresholds explicitly stated
summary_data = []

for ranker_name in RANKERS.keys():
    for ref_name in references.keys():
        row = {
            "ranker": ranker_name,
            "reference": ref_name,
            "threshold_type": "positive" if ref_name == "LiquidityUplift" else "top_quartile(75%)",
            "threshold_value": thresholds[ref_name],
            "Recall@10_mean": results_df[f"{ranker_name}_{ref_name}_Recall@10"].mean(),
            "nDCG@10_mean": results_df[f"{ranker_name}_{ref_name}_nDCG@10"].mean(),
        }
        summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(output_dir / "metrics" / "evaluation_summary_with_thresholds.csv", index=False)
```

- [ ] **Step 8: Verify syntax**

```bash
python -m py_compile scripts/evaluation/run_retrieval_metrics.py
```

- [ ] **Step 9: Commit**

```bash
git add scripts/evaluation/run_retrieval_metrics.py
git commit -m "feat: refactor evaluation with temporal/tabular/joint rankers and multi-reference scoring"
```

---

## Task 4: Generate Summary Report

- [ ] **Step 1: Run evaluation**

```bash
rm -rf results/retrieval_separate
uv run python -m scripts.evaluation.run_retrieval_metrics \
    --features data/processed/all_features.parquet \
    --checkpoint checkpoints/last.ckpt \
    --output-dir results/retrieval_separate \
    2>&1 | tail -50
```

- [ ] **Step 2: Verify outputs**

```python
import pandas as pd

# Check summary
summary = pd.read_csv("results/retrieval_separate/metrics/evaluation_summary_with_thresholds.csv")
print("Rankers:", summary['ranker'].unique())
print("References:", summary['reference'].unique())
print("\nSample of summary:")
print(summary.head(10))

# Check thresholds are reported
print("\nThresholds used:")
for ref in summary['reference'].unique():
    thresh = summary[summary['reference']==ref]['threshold_value'].iloc[0]
    print(f"  {ref}: {thresh:.4f}")
```

- [ ] **Step 3: Generate interpretation report**

Create a simple text report showing what each ranker is good at:

```
=== EVALUATION RESULTS ===

Thresholds Used:
- LiquidityUplift: > 0 (positive uplift required)
- ReturnSimilarity: >= 0.75 (top quartile correlation)
- SectorSimilarity: >= 0.50 (same sector or better)
- FundamentalsSim: >= 0.75 (top quartile market cap similarity)
- LiquidityChar: >= 0.75 (top quartile liquidity score)
- TurnoverChar: >= 0.75 (top quartile turnover)

Ranker Performance (Recall@10):

Temporal-only:
  - LiquidityUplift:  X.XX%  [finding liquid substitutes?]
  - ReturnSimilarity: X.XX%  [finding correlated returns?]
  - SectorSimilarity: X.XX%  [finding same sector?]
  - FundamentalsSim:  X.XX%  [finding similar size?]

Tabular-only:
  - LiquidityUplift:  X.XX%
  - ReturnSimilarity: X.XX%
  - SectorSimilarity: X.XX%
  - FundamentalsSim:  X.XX%

Joint:
  - ...

Tabular + Rerank:
  - ...

=== INTERPRETATION ===
- Temporal-only best at: [characteristic with highest score]
- Tabular-only best at: [characteristic with highest score]
- Joint best at: [characteristic with highest score]
- Rerank improves: [which characteristics improved]
```

- [ ] **Step 4: Final commit**

```bash
git commit --allow-empty -m "chore: complete separate temporal/tabular evaluation"
```

---

## Summary

| Task | What Changed | Output |
|------|--------------|--------|
| 1 | Added `get_temporal_embedding` and `get_tabular_embedding` | Can inference separate components |
| 2 | Added multi-reference ground truth functions | 6 different scoring references |
| 3 | Refactored to 4 rankers (temporal, tabular, joint, tabular_rerank) | No pearson correlation, clean comparison |
| 4 | Generated evaluation with threshold reporting | Clear understanding of what works |

## Key Outputs

1. **4 rankers only**: temporal_only, tabular_only, joint, tabular_rerank
2. **6 references**: LiquidityUplift, ReturnSimilarity, SectorSimilarity, FundamentalsSim, LiquidityChar, TurnoverChar
3. **Thresholds explicitly reported**: top quartile (75%) or positive uplift
4. **Separate scores**: Recall@10 and nDCG@10 reported separately for each combination

Co-Authored-By: Claude Code <noreply@anthropic.com>
