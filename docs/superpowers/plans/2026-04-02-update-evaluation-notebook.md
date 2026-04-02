# Update Evaluation Notebook for New Schema and Component Analysis

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update `notebooks/02_retrieval_evaluation.ipynb` to use the new 6-ranker schema and add component breakdown analysis (return/sector/size similarity).

**Architecture:** The notebook currently uses the old 5-ranker schema (embedding, correlation, liquidity_distance, correlation_rerank, hybrid). It needs to be updated to the new 6-ranker schema (embedding, pearson_corr, spearman_corr, embedding_rerank, pearson_corr_rerank, spearman_corr_rerank) and include visualizations for the three reference signals (Similarity, LiquidityUplift, Utility) plus component similarity breakdown.

**Tech Stack:** Jupyter notebook, pandas, matplotlib, seaborn

---

## Background

The current notebook (`notebooks/02_retrieval_evaluation.ipynb`) is outdated:
- Uses old ranker names: `correlation`, `liquidity_distance`, `correlation_rerank`, `hybrid`
- New evaluation produces: `pearson_corr`, `spearman_corr`, `embedding_rerank`, `pearson_corr_rerank`, `spearman_corr_rerank`
- Missing: Component breakdown visualization (return/sector/size similarity from `utility_breakdown_averaged.csv`)

## Changes Required

### Task 1: Update RANKER_LABELS and File Path

**File:** `notebooks/02_retrieval_evaluation.ipynb`
**Cell:** Code cell near line 72-83

- [ ] **Step 1: Read the notebook cell containing RANKER_LABELS**

Find the cell with:
```python
RANKER_LABELS = {
    "embedding"           : "Embedding",
    "pearson_corr"        : "Pearson Corr.",
    ...
}
```

- [ ] **Step 2: Verify current RANKER_LABELS matches new schema**

Should already be:
```python
RANKER_LABELS = {
    "embedding"           : "Embedding",
    "pearson_corr"        : "Pearson Corr.",
    "spearman_corr"       : "Spearman Corr.",
    "embedding_rerank"    : "Emb. + Rerank",
    "pearson_corr_rerank" : "Pearson + Rerank",
    "spearman_corr_rerank": "Spearman + Rerank",
}
```

- [ ] **Step 3: Verify METRICS_DIR points to retrieval_v2**

Should be: `METRICS_DIR = PROJECT_ROOT / "results" / "retrieval_v2" / "metrics"`

- [ ] **Step 4: Clear notebook outputs**

Run: `jupyter nbconvert --clear-output --inplace notebooks/02_retrieval_evaluation.ipynb`

- [ ] **Step 5: Commit if any changes made**

```bash
git add notebooks/02_retrieval_evaluation.ipynb
git commit -m "chore: verify notebook uses new ranker schema and retrieval_v2 path"
```

---

### Task 2: Add Per-Reference Signal Comparison Section

**File:** `notebooks/02_retrieval_evaluation.ipynb`
**Location:** After "1. Load Results" section

Add new markdown and code cells to compare metrics across the three reference signals:

- [ ] **Step 1: Add markdown header**

Insert after line ~104:
```markdown
---
## 2. Per-Reference Signal Comparison

Compare how each ranker performs against different ground truth references.
```

- [ ] **Step 2: Add code cell to load per-reference CSVs**

```python
# Load per-reference metrics
similarity_df = pd.read_csv(METRICS_DIR / "retrieval_similarity.csv")
liquidity_df = pd.read_csv(METRICS_DIR / "retrieval_liquidity_uplift.csv")
utility_df = pd.read_csv(METRICS_DIR / "retrieval_utility.csv")

print("Similarity reference metrics:")
print(similarity_df.to_string(index=False))
print("\nLiquidityUplift reference metrics:")
print(liquidity_df.to_string(index=False))
print("\nUtility reference metrics:")
print(utility_df.to_string(index=False))
```

- [ ] **Step 3: Add visualization code cell**

```python
# Create side-by-side comparison heatmap
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

refs = {
    "Similarity": similarity_df,
    "LiquidityUplift": liquidity_df,
    "Utility": utility_df,
}

for ax, (ref_name, df) in zip(axes, refs.items()):
    # Prepare data for heatmap
    heatmap_data = df.set_index("metric_name")[list(RANKER_LABELS.keys())].rename(columns=RANKER_LABELS)
    
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Score"},
        ax=ax,
    )
    ax.set_title(f"{ref_name} Reference", fontweight="bold")
    ax.set_xlabel("Ranker")
    ax.set_ylabel("Metric")

plt.suptitle("Retrieval Metrics by Reference Signal", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(METRICS_DIR / "../figures/metrics_by_reference_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
```

- [ ] **Step 4: Clear outputs and commit**

```bash
jupyter nbconvert --clear-output --inplace notebooks/02_retrieval_evaluation.ipynb
git add notebooks/02_retrieval_evaluation.ipynb
git commit -m "feat: add per-reference signal comparison section"
```

---

### Task 3: Add Component Breakdown Analysis Section

**File:** `notebooks/02_retrieval_evaluation.ipynb`
**Location:** After Task 2's new section

- [ ] **Step 1: Add markdown header**

```markdown
---
## 3. Component Breakdown Analysis

Analyze what drives each ranker's top-10 results:
- **Return similarity**: Behavioral similarity (60-day return correlation)
- **Sector similarity**: Industry/sector match
- **Liquidity improvement**: Positive LiquidityUplift (more liquid than query)
```

- [ ] **Step 2: Add code cell to load breakdown data**

```python
# Load utility breakdown analysis
breakdown_path = METRICS_DIR / "utility_breakdown_averaged.csv"

if breakdown_path.exists():
    breakdown_df = pd.read_csv(breakdown_path)
    print("Utility Breakdown (averaged across all queries):")
    print(breakdown_df.to_string())
else:
    print(f"Warning: {breakdown_path} not found. Run evaluation first.")
    breakdown_df = None
```

- [ ] **Step 3: Add stacked bar chart visualization**

```python
if breakdown_df is not None:
    # Prepare data
    breakdown_df = breakdown_df.reset_index()
    breakdown_df = breakdown_df.rename(columns={
        "pct_return_sim": "Return Similarity (%)",
        "pct_sector_sim": "Sector Similarity (%)",
        "pct_liq_improve": "Liquidity Improvement (%)",
    })
    
    # Rename rankers for display
    breakdown_df["ranker"] = breakdown_df["ranker"].map(RANKER_LABELS)
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(breakdown_df))
    width = 0.6
    
    p1 = ax.bar(x, breakdown_df["Return Similarity (%)"], width, label="Return Similarity", color="#2E86AB")
    p2 = ax.bar(x, breakdown_df["Sector Similarity (%)"], width, 
                bottom=breakdown_df["Return Similarity (%)"], 
                label="Sector Similarity", color="#A23B72")
    p3 = ax.bar(x, breakdown_df["Liquidity Improvement (%)"], width,
                bottom=breakdown_df["Return Similarity (%)"] + breakdown_df["Sector Similarity (%)"],
                label="Liquidity Improvement", color="#06A77D")
    
    ax.set_xlabel("Ranker", fontweight="bold")
    ax.set_ylabel("Percentage of Top-10 Results (%)", fontweight="bold")
    ax.set_title("What Drives Each Ranker's Top-10 Results?\n(Component Breakdown)", 
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(breakdown_df["ranker"], rotation=45, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 105)
    
    # Add value labels
    for i, row in breakdown_df.iterrows():
        total = row["Return Similarity (%)"] + row["Sector Similarity (%)"] + row["Liquidity Improvement (%)"]
        ax.annotate(f"{total:.1f}%", (i, total + 2), ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(METRICS_DIR / "../figures/component_breakdown_stacked.png", dpi=150, bbox_inches="tight")
    plt.show()
```

- [ ] **Step 4: Add component-by-ranker table**

```python
if breakdown_df is not None:
    # Display formatted table
    display_df = breakdown_df.copy()
    display_df.columns = ["Ranker", "Return Sim (%)", "Sector Sim (%)", "Liq. Improve (%)", "Total (%)"]
    display_df["Total (%)"] = (display_df["Return Sim (%)"] + 
                               display_df["Sector Sim (%)"] + 
                               display_df["Liq. Improve (%)"]).round(1)
    
    print("\n### Component Breakdown by Ranker\n")
    print(display_df.to_string(index=False))
```

- [ ] **Step 5: Clear outputs and commit**

```bash
jupyter nbconvert --clear-output --inplace notebooks/02_retrieval_evaluation.ipynb
git add notebooks/02_retrieval_evaluation.ipynb
git commit -m "feat: add component breakdown analysis section"
```

---

### Task 4: Add Detailed Per-Query Component Analysis

**File:** `notebooks/02_retrieval_evaluation.ipynb`
**Location:** After Task 3

- [ ] **Step 1: Add markdown header**

```markdown
---
## 4. Per-Query Component Drill-Down

Examine component breakdown for individual queries.
```

- [ ] **Step 2: Add code cell for query selector**

```python
# Load detailed breakdown
breakdown_detail_path = METRICS_DIR / "utility_breakdown_analysis.csv"

if breakdown_detail_path.exists():
    breakdown_detail = pd.read_csv(breakdown_detail_path)
    
    # Show available queries
    queries = breakdown_detail["query"].unique()
    print(f"Available queries ({len(queries)} total):")
    print(", ".join(queries[:20]), "..." if len(queries) > 20 else "")
    
    # Pick a sample query for analysis
    sample_query = queries[0]
    print(f"\nSample query: {sample_query}")
else:
    print(f"Warning: {breakdown_detail_path} not found")
    breakdown_detail = None
```

- [ ] **Step 3: Add query-specific breakdown visualization**

```python
if breakdown_detail is not None:
    # Filter for sample query
    query_data = breakdown_detail[breakdown_detail["query"] == sample_query]
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(query_data))
    width = 0.25
    
    ax.bar(x - width, query_data["pct_return_sim"], width, label="Return Similarity", color="#2E86AB")
    ax.bar(x, query_data["pct_sector_sim"], width, label="Sector Similarity", color="#A23B72")
    ax.bar(x + width, query_data["pct_liq_improve"], width, label="Liquidity Improvement", color="#06A77D")
    
    ax.set_xlabel("Ranker", fontweight="bold")
    ax.set_ylabel("Percentage (%)", fontweight="bold")
    ax.set_title(f"Component Breakdown for Query: {sample_query}", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(query_data["ranker"].map(RANKER_LABELS), rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(METRICS_DIR / f"../figures/component_breakdown_{sample_query}.png", dpi=150, bbox_inches="tight")
    plt.show()
```

- [ ] **Step 4: Clear outputs and commit**

```bash
jupyter nbconvert --clear-output --inplace notebooks/02_retrieval_evaluation.ipynb
git add notebooks/02_retrieval_evaluation.ipynb
git commit -m "feat: add per-query component drill-down section"
```

---

### Task 5: Final Verification and Documentation Update

- [ ] **Step 1: Verify notebook runs without errors**

```bash
cd /home/redbear/Projects/LiquidSearcher
uv run jupyter nbconvert --to notebook --execute notebooks/02_retrieval_evaluation.ipynb --output /tmp/test.ipynb 2>&1 | tail -50
```

- [ ] **Step 2: Clear all outputs before final commit**

```bash
jupyter nbconvert --clear-output --inplace notebooks/02_retrieval_evaluation.ipynb
```

- [ ] **Step 3: Update notebook markdown documentation**

Update the opening markdown cell (lines ~8-30) to document the new sections:

```markdown
# Block 4: Retrieval Evaluation — Recall@K, nDCG@10, Spearman ρ

This notebook covers the quantitative retrieval evaluation block (v2 output — 6 rankers × 3 reference signals).

## Sections
1. **Load Results** — Overall metrics summary
2. **Per-Reference Signal Comparison** — Compare rankers across Similarity, LiquidityUplift, and Utility references
3. **Component Breakdown Analysis** — What drives top-10 results (return/sector/liquidity composition)
4. **Per-Query Drill-Down** — Individual query analysis

| Metric | What it measures |
|--------|------------------|
| **Recall@10** | Fraction of ground-truth peers found in top-10 |
| **nDCG@10** | Ranked quality — rewards placing most-relevant peers at top |
| **Spearman ρ** | Rank correlation between predicted ranking and reference ordering |

... rest of existing content ...
```

- [ ] **Step 4: Final commit**

```bash
git add notebooks/02_retrieval_evaluation.ipynb
git commit -m "docs: update notebook documentation for new sections"
```

---

## Summary of Changes

| Task | What Changed | Purpose |
|------|--------------|---------|
| 1 | Verified RANKER_LABELS and path | Ensure notebook uses correct schema |
| 2 | Added per-reference comparison | Compare rankers across 3 ground truth signals |
| 3 | Added component breakdown | Visualize what drives top-10 (return/sector/liquidity) |
| 4 | Added per-query drill-down | Detailed single-query analysis |
| 5 | Documentation update | Explain all sections |

## Expected Outcomes

After updates, the notebook will show:
1. **Heatmaps** comparing all 6 rankers across 3 reference signals
2. **Stacked bar charts** showing component composition (% return sim, % sector sim, % liquidity improve)
3. **Per-query breakdowns** for detailed analysis
4. All visualizations saved to `results/retrieval_v2/figures/`

Co-Authored-By: Claude Code <noreply@anthropic.com>
