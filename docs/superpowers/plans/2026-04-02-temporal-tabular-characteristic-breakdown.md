# Add Temporal/Tabular Similarity Breakdown to Evaluation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Break down embedding similarity into temporal and tabular components, plus add underlying stock characteristic similarities (spread, turnover, volatility, etc.).

**Architecture:** Extend the evaluation pipeline to compute and store: (1) temporal embedding similarity, (2) tabular embedding similarity, (3) joint embedding similarity (existing), (4) individual stock characteristic similarities. Update notebook with new visualizations.

**Tech Stack:** PyTorch, pandas, Jupyter notebook

---

## Background

Currently the evaluation only computes **joint embedding similarity** (256-dim = 128 temporal + 128 tabular concatenated). 

The user wants to see:
1. **Temporal similarity** - how similar are the price/volume patterns?
2. **Tabular similarity** - how similar are the fundamentals/GICS features?
3. **Individual characteristic similarities** - spread similarity, turnover similarity, volatility similarity, etc.

This will help understand what drives the model's retrieval decisions.

---

## File Structure

**Files to Modify:**
- `scripts/evaluation/run_retrieval_metrics.py` - Add temporal/tabular/characteristic similarity computation
- `src/evaluation/ground_truth.py` - Add characteristic similarity computation functions
- `notebooks/02_retrieval_evaluation.ipynb` - Add new visualizations

---

## Task 1: Add Method to Compute Separate Temporal and Tabular Embeddings

**Files:**
- Modify: `src/models/dual_encoder.py`

- [ ] **Step 1: Add get_separate_embeddings method**

Add after `get_joint_embedding` method (around line 117):

```python
def get_separate_embeddings(self, price_data, fundamentals, categorical):
    """Get separate temporal and tabular embeddings for analysis.
    
    Args:
        price_data: (batch, 60, 13) - OHLCV features
        fundamentals: (batch, 15) - continuous features
        categorical: (batch, 2) - [gsector, ggroup] indices
    
    Returns:
        dict with keys:
            'temporal': (batch, 128) temporal embeddings
            'tabular': (batch, 128) tabular embeddings
            'joint': (batch, 256) concatenated embeddings
    """
    temporal_emb = self.temporal_encoder(price_data)
    tabular_emb = self.tabular_encoder(fundamentals, categorical)
    joint_emb = torch.cat([temporal_emb, tabular_emb], dim=-1)
    
    return {
        'temporal': temporal_emb,
        'tabular': tabular_emb,
        'joint': joint_emb
    }
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile src/models/dual_encoder.py`

- [ ] **Step 3: Commit**

```bash
git add src/models/dual_encoder.py
git commit -m "feat: add get_separate_embeddings method for temporal/tabular breakdown"
```

---

## Task 2: Add Characteristic Similarity Computation

**Files:**
- Modify: `src/evaluation/ground_truth.py`

- [ ] **Step 1: Add compute_characteristic_similarity function**

Add after `compute_utility_score` function:

```python
def compute_characteristic_similarity(
    query_symbol: str,
    snapshot_df: pd.DataFrame,
    characteristics: list[str] = None,
) -> dict[str, pd.Series]:
    """Compute similarity based on individual stock characteristics.
    
    For each characteristic (e.g., spread, turnover, volatility),
    compute how similar each candidate is to the query stock.
    
    Similarity = 1 - normalized_distance where distance is absolute difference
    in percentile ranks.
    
    Args:
        query_symbol: Query symbol
        snapshot_df: DataFrame with symbol and characteristic columns
        characteristics: List of column names to compute similarity for.
                        If None, uses all numeric columns ending with 
                        '_rank' or standard liquidity proxies.
        
    Returns:
        Dictionary mapping characteristic name to similarity Series
    """
    if query_symbol not in snapshot_df["symbol"].values:
        return {}
    
    # Auto-detect characteristics if not specified
    if characteristics is None:
        # Look for percentile rank columns or standard liquidity proxies
        potential_cols = [
            "spread_rank", "amihud_rank", "turnover_rank",  # From liquidity_labels
            "LiquidityScore",  # Composite
            "spread_pct", "amihud", "turnover",  # Raw values
            "volatility", "beta", "market_cap",  # Risk/size characteristics
        ]
        characteristics = [c for c in potential_cols if c in snapshot_df.columns]
    
    results = {}
    
    for char in characteristics:
        if char not in snapshot_df.columns:
            continue
            
        # Get valid rows (non-NaN for this characteristic)
        valid_df = snapshot_df.dropna(subset=[char])
        
        if len(valid_df) == 0 or query_symbol not in valid_df["symbol"].values:
            continue
        
        # Compute percentile ranks
        ranks = valid_df[char].rank(pct=True)
        
        # Get query rank
        query_rank = ranks[valid_df["symbol"] == query_symbol].iloc[0]
        
        # Compute similarity: 1 - absolute difference in percentile ranks
        similarities = 1.0 - (ranks - query_rank).abs()
        similarities.index = valid_df["symbol"]
        
        # Exclude query itself
        similarities = similarities.drop(query_symbol, errors='ignore')
        
        results[char] = similarities
    
    return results
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile src/evaluation/ground_truth.py`

- [ ] **Step 3: Commit**

```bash
git add src/evaluation/ground_truth.py
git commit -m "feat: add compute_characteristic_similarity for individual stock features"
```

---

## Task 3: Extend Evaluation Script to Compute Temporal/Tabular/Characteristic Similarities

**Files:**
- Modify: `scripts/evaluation/run_retrieval_metrics.py`

- [ ] **Step 1: Modify compute_embedding_scores_from_embeddings to return separate similarities**

Find function `compute_embedding_scores_from_embeddings` and modify to optionally return temporal/tabular breakdown.

**Current signature:**
```python
def compute_embedding_scores_from_embeddings(
    embeddings: dict[str, torch.Tensor],
    query_symbols: list[str],
    snapshot_df: pd.DataFrame,
) -> dict[str, pd.Series]:
```

**New signature and implementation:**
```python
def compute_embedding_scores_from_embeddings(
    embeddings: dict[str, torch.Tensor],
    query_symbols: list[str],
    snapshot_df: pd.DataFrame,
    separate_components: bool = False,
) -> dict[str, pd.Series] | tuple[dict[str, pd.Series], dict[str, pd.Series], dict[str, pd.Series]]:
    """Compute embedding scores with optional temporal/tabular breakdown.
    
    Args:
        embeddings: Dict mapping symbol -> embedding tensor
        query_symbols: List of query symbols to compute scores for
        snapshot_df: DataFrame with snapshot data
        separate_components: If True, also return temporal and tabular similarities separately
        
    Returns:
        If separate_components=False: scores_dict (joint similarity only)
        If separate_components=True: (joint_scores, temporal_scores, tabular_scores)
    """
    # ... existing code for setup ...
    
    if separate_components:
        # Check if embeddings have the right shape (should be 256 for joint, or dict with 'temporal'/'tabular')
        first_emb = list(embeddings.values())[0]
        if isinstance(first_emb, dict):
            # Embeddings are already separated
            temporal_embeddings = {k: v['temporal'] for k, v in embeddings.items()}
            tabular_embeddings = {k: v['tabular'] for k, v in embeddings.items()}
            joint_embeddings = {k: v['joint'] for k, v in embeddings.items()}
        else:
            # Need to use model to get separate embeddings - this requires model access
            # For now, skip separate computation if not pre-computed
            temporal_embeddings = {}
            tabular_embeddings = {}
            joint_embeddings = embeddings
    
    # ... rest of existing code ...
```

Actually, it's better to modify the batch embedding computation to store all three. Let me create a cleaner approach:

**Better approach: Modify compute_embeddings_batch to return all three:**

Add parameter to `compute_embeddings_batch`:
```python
def compute_embeddings_batch(
    snapshot_df: pd.DataFrame,
    period_df: pd.DataFrame,
    model: DualEncoder,
    device: str = "cpu",
    window_size: int = 60,
    return_separate: bool = False,  # NEW PARAMETER
) -> dict[str, torch.Tensor] | dict[str, dict[str, torch.Tensor]]:
    """Compute embeddings for all symbols.
    
    Args:
        ...
        return_separate: If True, return dict of dicts with 'temporal', 'tabular', 'joint' keys
        
    Returns:
        If return_separate=False: {symbol: joint_embedding}
        If return_separate=True: {symbol: {'temporal': t_emb, 'tabular': tab_emb, 'joint': j_emb}}
    """
```

- [ ] **Step 2: Add new function compute_separate_embedding_similarities**

Add new function after `compute_embedding_scores`:

```python
def compute_separate_embedding_similarities(
    snapshot_df: pd.DataFrame,
    period_df: pd.DataFrame,
    query_symbols: list[str],
    checkpoint_path: str,
    device: str = "cpu",
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], dict[str, pd.Series]]:
    """Compute joint, temporal, and tabular embedding similarities separately.
    
    Returns:
        Tuple of (joint_scores_dict, temporal_scores_dict, tabular_scores_dict)
        Each dict maps query_symbol -> Series of similarities indexed by candidate_symbol
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
    
    # Compute embeddings for all symbols with separate components
    embeddings_dict = {}
    
    for symbol in snapshot_df["symbol"]:
        # Get symbol data
        symbol_data = period_df[period_df["symbol"] == symbol].sort_values("date").tail(window_size)
        if len(symbol_data) == 0:
            continue
        
        # Build features
        temporal, tabular, categorical = _build_symbol_features(
            symbol_data, symbol, snapshot_df
        )
        
        # Get separate embeddings
        with torch.no_grad():
            separate_embs = model.get_separate_embeddings(
                temporal.to(device),
                tabular.to(device),
                categorical.to(device)
            )
        
        embeddings_dict[symbol] = {
            'temporal': separate_embs['temporal'].cpu(),
            'tabular': separate_embs['tabular'].cpu(),
            'joint': separate_embs['joint'].cpu()
        }
    
    # Compute similarities for each query
    joint_scores_dict = {}
    temporal_scores_dict = {}
    tabular_scores_dict = {}
    
    symbol_list = list(embeddings_dict.keys())
    
    for query in query_symbols:
        if query not in embeddings_dict:
            continue
        
        query_embs = embeddings_dict[query]
        candidates = [s for s in symbol_list if s != query]
        
        # Compute each type of similarity
        joint_sims = []
        temporal_sims = []
        tabular_sims = []
        
        for candidate in candidates:
            cand_embs = embeddings_dict[candidate]
            
            joint_sim = F.cosine_similarity(
                query_embs['joint'], cand_embs['joint'], dim=-1
            ).item()
            temporal_sim = F.cosine_similarity(
                query_embs['temporal'], cand_embs['temporal'], dim=-1
            ).item()
            tabular_sim = F.cosine_similarity(
                query_embs['tabular'], cand_embs['tabular'], dim=-1
            ).item()
            
            joint_sims.append(joint_sim)
            temporal_sims.append(temporal_sim)
            tabular_sims.append(tabular_sim)
        
        joint_scores_dict[query] = pd.Series(joint_sims, index=candidates)
        temporal_scores_dict[query] = pd.Series(temporal_sims, index=candidates)
        tabular_scores_dict[query] = pd.Series(tabular_sims, index=candidates)
    
    return joint_scores_dict, temporal_scores_dict, tabular_scores_dict
```

- [ ] **Step 3: Update per-query loop to store all similarity types**

In the main evaluation loop, after computing similarities, store them:

```python
# NEW: Compute temporal and tabular similarities separately
(
    joint_scores_dict,
    temporal_scores_dict, 
    tabular_scores_dict
) = compute_separate_embedding_similarities(
    snapshot_df=snapshot_df,
    period_df=period_df,
    query_symbols=[query],  # Just this query
    checkpoint_path=checkpoint_path,
    device=device,
)

# NEW: Compute characteristic similarities
from src.evaluation.ground_truth import compute_characteristic_similarity
char_sims = compute_characteristic_similarity(query, snapshot_df)

# In per-query data storage, add:
per_query_data = {
    # ... existing columns ...
    # Embedding similarities broken down
    "emb_joint_score": joint_scores_dict.get(query, pd.Series()).reindex(candidates).fillna(0.5).values,
    "emb_temporal_score": temporal_scores_dict.get(query, pd.Series()).reindex(candidates).fillna(0.5).values,
    "emb_tabular_score": tabular_scores_dict.get(query, pd.Series()).reindex(candidates).fillna(0.5).values,
    # Characteristic similarities
    "char_liquidity_sim": char_sims.get("LiquidityScore", pd.Series(0.5, index=candidates)).values,
    "char_spread_sim": char_sims.get("spread_rank", pd.Series(0.5, index=candidates)).values,
    "char_turnover_sim": char_sims.get("turnover_rank", pd.Series(0.5, index=candidates)).values,
    # ... rest of existing columns ...
}
```

- [ ] **Step 4: Verify syntax**

Run: `python -m py_compile scripts/evaluation/run_retrieval_metrics.py`

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluation/run_retrieval_metrics.py
git commit -m "feat: compute and store temporal/tabular/characteristic similarities"
```

---

## Task 4: Add Breakdown Analysis for New Similarity Types

**Files:**
- Modify: `scripts/evaluation/run_retrieval_metrics.py`

- [ ] **Step 1: Extend utility breakdown to include temporal/tabular/characteristic breakdowns**

In the breakdown analysis section, add columns for:
- pct_temporal_sim (high temporal similarity)
- pct_tabular_sim (high tabular similarity)  
- pct_liquidity_char_sim (high liquidity characteristic similarity)

```python
# In breakdown computation, add:
high_temporal_sim = (top10["emb_temporal_score"] > 0.7).sum()  # threshold for "high"
high_tabular_sim = (top10["emb_tabular_score"] > 0.7).sum()
high_liq_char_sim = (top10["char_liquidity_sim"] > 0.7).sum()

breakdown_results.append({
    # ... existing fields ...
    "high_temporal_sim": high_temporal_sim,
    "high_tabular_sim": high_tabular_sim,
    "high_liq_char_sim": high_liq_char_sim,
    "pct_temporal_sim": high_temporal_sim / len(top10) * 100,
    "pct_tabular_sim": high_tabular_sim / len(top10) * 100,
    "pct_liq_char_sim": high_liq_char_sim / len(top10) * 100,
})
```

- [ ] **Step 2: Commit**

```bash
git add scripts/evaluation/run_retrieval_metrics.py
git commit -m "feat: extend breakdown analysis with temporal/tabular/characteristic components"
```

---

## Task 5: Regenerate Jupyter Notebook with New Visualizations

**Files:**
- Modify: `notebooks/02_retrieval_evaluation.ipynb`

- [ ] **Step 1: Add new section for Temporal/Tabular Breakdown**

Insert as new Section 3 (shifting current sections):

```markdown
---
## 3. Temporal vs Tabular Similarity Breakdown

The dual-encoder model produces three types of embeddings:
- **Temporal** (128-dim): Price/volume patterns over 60 days
- **Tabular** (128-dim): Fundamentals + GICS sector/group
- **Joint** (256-dim): Concatenation of both

This section analyzes what drives retrieval: behavioral similarity (temporal) vs fundamental similarity (tabular).
```

- [ ] **Step 2: Add visualization code**

```python
# Load per-query data with separate similarities
sample_query = "AAPL"  # or first available
query_file = METRICS_DIR / f"../retrieval/per_query/{sample_query}.parquet"

if query_file.exists():
    query_df = pd.read_parquet(query_file)
    
    # Check if new columns exist
    has_separate = all(c in query_df.columns for c in [
        "emb_temporal_score", "emb_tabular_score", "emb_joint_score"
    ])
    
    if has_separate:
        # Get top 10 by joint embedding
        top10 = query_df.nlargest(10, "emb_joint_score")
        
        # Create scatter plot: temporal vs tabular
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(
            top10["emb_temporal_score"],
            top10["emb_tabular_score"],
            c=top10["emb_joint_score"],
            s=200,
            cmap="RdYlGn",
            alpha=0.7,
            edgecolors="black",
            linewidth=1
        )
        
        # Add labels for each point
        for idx, row in top10.iterrows():
            ax.annotate(
                row["candidate_symbol"],
                (row["emb_temporal_score"], row["emb_tabular_score"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8
            )
        
        ax.set_xlabel("Temporal Similarity (Price/Volume Patterns)", fontweight="bold")
        ax.set_ylabel("Tabular Similarity (Fundamentals/Sector)", fontweight="bold")
        ax.set_title(f"Query {sample_query}: Top-10 Breakdown by Similarity Type\n(Color = Joint Similarity)", 
                     fontsize=12, fontweight="bold")
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Joint Similarity", fontweight="bold")
        
        # Add diagonal line (temporal = tabular)
        min_val = min(top10["emb_temporal_score"].min(), top10["emb_tabular_score"].min())
        max_val = max(top10["emb_temporal_score"].max(), top10["emb_tabular_score"].max())
        ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.3, label="Temporal = Tabular")
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(METRICS_DIR / f"../figures/temporal_tabular_breakdown_{sample_query}.png", 
                    dpi=150, bbox_inches="tight")
        plt.show()
        
        # Show summary table
        summary = top10[["candidate_symbol", "emb_temporal_score", "emb_tabular_score", "emb_joint_score"]].copy()
        summary.columns = ["Symbol", "Temporal Sim", "Tabular Sim", "Joint Sim"]
        print(f"\n### Top-10 for Query {sample_query} - Similarity Breakdown\n")
        print(summary.round(3).to_string(index=False))
    else:
        print("Separate temporal/tabular scores not found. Re-run evaluation with updated code.")
else:
    print(f"Query file not found: {query_file}")
```

- [ ] **Step 3: Add Characteristic Similarity Section**

```markdown
---
## 4. Individual Characteristic Similarities

Beyond embeddings, we can compute similarity on individual stock characteristics:
- **LiquidityScore similarity**: Overall liquidity rank similarity
- **Spread similarity**: Bid-ask spread pattern similarity  
- **Turnover similarity**: Trading volume turnover similarity
- **Volatility similarity**: Price volatility similarity
```

- [ ] **Step 4: Add characteristic similarity visualization**

```python
# Visualize characteristic similarities
if has_separate and "char_liquidity_sim" in query_df.columns:
    char_cols = [c for c in query_df.columns if c.startswith("char_")]
    
    if char_cols:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, char_col in enumerate(char_cols[:4]):  # Max 4 characteristics
            ax = axes[idx]
            
            # Plot distribution of this characteristic similarity
            ax.hist(query_df[char_col], bins=30, alpha=0.7, color="steelblue", edgecolor="black")
            
            # Highlight top-10
            top10_vals = query_df.nlargest(10, "emb_joint_score")[char_col]
            ax.axvline(top10_vals.mean(), color="red", linestyle="--", linewidth=2, 
                      label=f"Top-10 Average: {top10_vals.mean():.3f}")
            
            char_name = char_col.replace("char_", "").replace("_", " ").title()
            ax.set_xlabel(f"{char_name} Similarity", fontweight="bold")
            ax.set_ylabel("Count", fontweight="bold")
            ax.set_title(f"{char_name} Similarity Distribution", fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(char_cols), 4):
            axes[idx].axis("off")
        
        plt.suptitle(f"Query {sample_query}: Characteristic Similarity Distributions\n(Red Line = Top-10 Average)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(METRICS_DIR / f"../figures/characteristic_similarities_{sample_query}.png",
                    dpi=150, bbox_inches="tight")
        plt.show()
        
        # Summary table of top-10 by characteristic
        print(f"\n### Top-10 Characteristic Similarities for Query {sample_query}\n")
        char_summary = top10[["candidate_symbol"] + char_cols].copy()
        char_summary.columns = ["Symbol"] + [c.replace("char_", "").replace("_", " ").title() for c in char_cols]
        print(char_summary.round(3).to_string(index=False))
```

- [ ] **Step 5: Clear outputs and commit**

```bash
jupyter nbconvert --clear-output --inplace notebooks/02_retrieval_evaluation.ipynb
git add notebooks/02_retrieval_evaluation.ipynb
git commit -m "feat: add temporal/tabular/characteristic similarity visualizations"
```

---

## Task 6: Regenerate Full Evaluation with New Features

- [ ] **Step 1: Run full evaluation**

```bash
rm -rf results/retrieval_v3
uv run python -m scripts.evaluation.run_retrieval_metrics \
    --features data/processed/all_features.parquet \
    --checkpoint checkpoints/last.ckpt \
    --output-dir results/retrieval_v3 \
    2>&1 | tail -100
```

- [ ] **Step 2: Verify new columns in parquets**

```python
import pandas as pd
df = pd.read_parquet("results/retrieval_v3/retrieval/per_query/GT.parquet")
print("New columns:", [c for c in df.columns if "emb_" in c or "char_" in c])
```

- [ ] **Step 3: Verify breakdown CSV has new columns**

```python
import pandas as pd
df = pd.read_csv("results/retrieval_v3/metrics/utility_breakdown_analysis.csv")
print("New breakdown columns:", [c for c in df.columns if "temporal" in c or "tabular" in c or "char_" in c])
```

- [ ] **Step 4: Final commit**

```bash
git commit --allow-empty -m "chore: regenerate evaluation with temporal/tabular/characteristic breakdowns"
```

---

## Summary of Changes

| Task | What Changed | New Data Produced |
|------|--------------|-------------------|
| 1 | Added `get_separate_embeddings` to model | Can get temporal/tabular/joint separately |
| 2 | Added `compute_characteristic_similarity` | Similarity on individual stock features |
| 3 | Extended evaluation to compute all similarities | emb_temporal_score, emb_tabular_score, char_*_sim |
| 4 | Extended breakdown analysis | pct_temporal_sim, pct_tabular_sim, etc. |
| 5 | Added notebook visualizations | Scatter plots, distribution plots, tables |
| 6 | Regenerated evaluation | Full results with all new columns |

## Expected Outcomes

After implementation:
1. **Per-query parquets** contain:
   - `emb_joint_score` (existing)
   - `emb_temporal_score` (new)
   - `emb_tabular_score` (new)
   - `char_liquidity_sim`, `char_spread_sim`, etc. (new)

2. **Notebook shows:**
   - Scatter plot: temporal vs tabular similarity for top-10
   - Distribution plots: individual characteristic similarities
   - Tables showing breakdown by similarity type

3. **Breakdown CSVs** include:
   - Percentage of top-10 driven by temporal similarity
   - Percentage driven by tabular similarity
   - Percentage driven by each characteristic

Co-Authored-By: Claude Code <noreply@anthropic.com>
