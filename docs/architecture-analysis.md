# Dual-Encoder Architecture Analysis for Stock Substitute Recommendation

## Executive Summary

**What we're building:** A dual-encoder model that learns to embed stocks into a shared 256-dimensional space where similar stocks (good substitutes) are close together.

**Why dual-encoder:** We have TWO different types of data about each stock:
1. **Temporal data (time-series):** OHLCV price patterns, trading behavior (technical features)
2. **Tabular data (fundamentals):** Risk metrics, valuation ratios, momentum scores, sector codes

These are fundamentally different modalities—like images and text in CLIP. A dual-encoder lets each modality have its own specialized encoder while learning to map both into the SAME embedding space.

**Critical insight on training (from research):**
- **Initial training:** Static 2010-2022 split for backtesting/validation
- **Production deployment:** MUST use rolling windows due to market concept drift
- Financial markets are non-stationary—static models degrade over time
- Rolling window (last 5-10 years, refreshed quarterly) is standard best practice

---

## Why This Architecture? (The CLIP Connection)

OpenAI's CLIP revolutionized multimodal learning by showing that:

> **"You can align completely different data types (images + text) into a shared embedding space using contrastive learning"**

**How CLIP works:**
- Image goes through Vision Transformer → 512-dim embedding
- Text goes through Language Transformer → 512-dim embedding  
- Contrastive loss pulls matching pairs together, pushes non-matching apart
- Result: You can search images with text and vice versa

**Our application (CRITICAL CLARIFICATION - Issue #5):**

**TRAINING (CLIP-style dot product):**
- Stock A's price behavior (temporal) → 128-dim embedding
- Stock A's fundamentals (tabular) → 128-dim embedding
- **Dot product (cosine similarity)** between the two 128-dim vectors
- Contrastive loss operates on this similarity score
- **NO concatenation during training**

**INFERENCE (concatenation for similarity search):**
- Concatenate [temporal || fundamental] → 256-dim joint embedding
- Use for nearest-neighbor search across stock universe
- Can also use dot product directly (faster, single modality queries)

**Key insight:** The 256-dim vector is an **inference artifact**—the loss operates on the two 128-dim vectors separately.

---

## Architecture Components

### 1. Temporal Encoder (Price Behavior → 128-dim)

**Input:** Technical/OHLCV features (13 features from technical group)
- z_close_5d, z_close_10d, z_close_20d (price momentum)
- z_high, z_low (intraday patterns)
- z_volume_5d, z_volume_10d, z_volume_20d (volume anomalies)
- ma_ratio_5, ma_ratio_10, ma_ratio_15, ma_ratio_20, ma_ratio_25 (trend strength)

**Architecture Choice: BiMT-TCN (TCN + Lightweight Transformer)**

**Updated from Review (Issue #1):** 2025 research shows Transformer outperforms TCN on stock sequences even for short windows. Best practice is hybrid:

| Architecture | Pros | Cons | Verdict |
|-------------|------|------|--------------|
| **LSTM** | Simple, proven for finance | Vanishing gradients, slow | Outdated |
| **TCN-only** | Efficient dilated convs, stable | Misses global dependencies | **Insufficient (was wrong)** |
| **Transformer-only** | State-of-art attention | Memory heavy for long seqs | Overkill alone |
| **BiMT-TCN (ours)** | **TCN local + Transformer global** | Slightly more complex | **✓ Best of both** |

**Implementation:**
```python
TemporalEncoder(
    # TCN for local multi-scale patterns (5/10/20-day windows)
    tcn_input_dim=13,
    tcn_hidden_dim=64,
    tcn_layers=3,
    tcn_kernel_size=3,
    tcn_dilations=[1, 2, 4, 8],
    
    # Lightweight Transformer for global dependencies
    transformer_heads=4,
    transformer_layers=2,
    transformer_dropout=0.1,
    
    # Output
    output_dim=128
)
```

**Why this works:**
- TCN extracts local patterns efficiently (dilated convs for long memory)
- Transformer (2 layers, 4 heads) captures cross-timestep relationships
- Together: local pattern detection + global context understanding

---

### 1b. Sector Classification: Why GICS (Not SIC/NAICS)

**Research Finding:** GICS consistently outperforms SIC, NAICS, and Fama-French 48-industry classification for financial analysis. [arxiv 2305.01028](https://arxiv.org/pdf/2305.01028.pdf)

**Why GICS wins:**
- **Market-oriented:** Groups companies by how investors/markets treat them
- **SIC/NAICS:** Production-oriented (steel and aluminum are neighbors despite different stock behavior)
- **Direct evidence:** GICS explains cross-sectional variation in financial characteristics, valuation multiples, and stock returns better than all alternatives
- **For substitutes:** We want stocks investors treat as peers, not stocks sharing a factory floor

**Compustat Fields (from WRDS):**

| Field | Description | Granularity | Classes | Embedding Dim |
|-------|-------------|-------------|---------|---------------|
| `gsector` | GICS Sector | 11 classes | 10, 15, 20...60 (Info Tech=45) | **8-dim** |
| `ggroup` | GICS Industry Group | 25 classes | 1010, 1510, 2010... | **16-dim** |
| `gind` | GICS Industry | 74 classes | *Too sparse, not used* | — |
| `gsubind` | GICS Sub-Industry | 163 classes | *Way too sparse for Russell 2000* | — |

**Why drop gind/gsubind:** At 74 and 163 classes respectively, you get extremely thin categories for small caps (many with <5 stocks). Embedding layer can't learn from 3 examples.

**Encoding: Learned Embeddings (Not One-Hot/Integer)**

Rule of thumb: `dim ≈ ceil(n_classes^0.25 × 4)`
- 11 sectors → ~7-8 dim, we use **8**
- 25 industry groups → ~10-12 dim, we use **16**

**Why learned embeddings matter:**
- **One-hot:** Treats all sectors as equidistant — wrong (Energy≠Materials distance = Energy≠Utilities)
- **Integer:** Implies false ordinal (sector 4 ≠ 2×sector 2)
- **Learned:** Model discovers Energy and Materials cluster together (similar commodity shocks), Utilities and Staples cluster as defensives

**Pre-trained consideration:** User mentions we may use pre-trained sector embeddings (not training from scratch) — research if small pre-trained GICS embeddings exist.

---

### 2. Tabular Encoder (Fundamentals → 128-dim)

**Input:** Combined features from market_risk + momentum + valuation + sector
- Market risk: 2 features (market_beta_60d, downside_beta_60d) - *Note: FF5 factor loadings excluded from methodology*
- Volatility: 4 features (realized vol_20d/60d, idiosyncratic_vol, vol_of_vol)
- Momentum: 5 features (mom_1m, mom_3m, mom_6m, mom_12_1m, macd)
- Valuation: 4 features (log_mktcap, pe_ratio, pb_ratio, roe)
- Sector: 2 categorical **(gsector=11 classes + ggroup=25 classes)**

**Total:** ~15 continuous + 2 categorical features (simplified from FF5-inclusive version)

**Architecture Choice: TabMixer (2025)**

**Updated from Review (Issue #2):** TabMixer specifically designed for tabular data with:
- MLP-Mixer style channel-wise and instance-wise mixing
- <0.01% FLOPs of FT-Transformer
- Handles missing values natively (critical for Compustat quarterly data)
- Transfer learning support

**Previous approach (ResNet-MLP + attention) was ad-hoc. TabMixer is theoretically grounded.**

**Implementation:**
```python
TabularEncoder(
    # Feature dimensions
    continuous_dim=15,          # Market risk (2) + momentum (5) + valuation (4) + volatility (4)
    categorical_dims=[11, 25],  # gsector (11), ggroup (25)
    
    # GICS learned embeddings (from Compustat fields)
    embedding_dims=[8, 16],     # gsector→8-dim, ggroup→16-dim (rule of thumb: ceil(n^0.25×4))
    
    # TabMixer architecture
    mixer_layers=4,             # MLP-Mixer depth
    hidden_dim=128,             # Mixing dimension
    expansion_factor=4,         # Channel expansion in mixer blocks
    
    # Output
    output_dim=128,             # Final tabular embedding
    dropout=0.1,
    
    # Missing value handling (critical for Compustat quarterly data)
    handle_missing='learned_mask'  # Native to TabMixer
)

# Forward pass:
# sector_emb = nn.Embedding(11, 8)(gsector_id)      # (batch, 8)
# group_emb = nn.Embedding(25, 16)(ggroup_id)        # (batch, 16)  
# sector_features = concat([sector_emb, group_emb])  # (batch, 24)
# → Concatenate with continuous features → TabMixer → 128-dim output
```

**Why TabMixer wins:**
- Purpose-built for tabular data (unlike ad-hoc attention on MLP)
- Handles missing fundamentals without imputation hacks
- Directly comparable to FT-Transformer in ablations
- Fast training (3-4x faster than Transformer)

---

## Contrastive Learning Design

### Critical Issue #3: Positive Pair Definition

**Original (flawed):** "Same stock at different times"

**Problem:** A growth stock that became value, or pre/post-FDA approval biotech—these should NOT be pulled together in embedding space. Characteristics change over time.

**From ACM 2024 Contrastive CL paper:** They use **statistical hypothesis test on return distributions**—two stocks are positive pair only if return distributions are statistically similar.

**Our Three Options:**

```python
# Option A (best - statistical test):
# Stocks i and j are positive pair at time t if their 60-day return 
# distributions cannot be rejected as same (two-sample KS test, p > 0.05)

# Option B (practical - regime-stable):
# Same stock, but only adjacent windows where rolling beta/vol 
# haven't shifted > 1 standard deviation

# Option C (MVP - simple but documented limitation):
# Same stock different times (fastest to implement)
```

**Decision:**
- **MVP Phase:** Start with Option C, document as known limitation
- **Production:** Implement Option A for principled positive pairs
- This is where we can differentiate from ACM 2024 paper

---

### Critical Issue #4: Hard Negative Mining (Do NOT Defer)

**Original (wrong):** "Start with in-batch negatives, add hard negatives later"

**Problem:** Random in-batch negatives are **too easy**—most random pairs genuinely differ (energy vs healthcare, micro-cap vs large-cap). Model learns to separate obvious things, then stops learning. Result: embeddings separate sectors but **fail at within-sector substitutes**—our core use case.

**From OpenReview 2024 (financial embeddings):** Hard negatives are **critical** for financial embedding quality.

**GICS-Structured Hard Negative Sampling (implement from day 1):**

**Sector/Industry hierarchy for negative sampling:**

```python
def sample_hard_negatives(target_stock, batch, n_hard=8):
    """
    Three-tier negative sampling using GICS hierarchy:
    
    - Easy negatives: Different gsector (11 classes) - EXCLUDE, too easy
    - Hard negatives: Same gsector, different ggroup (25 classes)
    - Hardest negatives: Same ggroup (industry group), different beta/vol quintile
    
    Example: Target = AAPL (sector=45 Tech, group=4510 Software)
    - Easy: JPM (sector=40 Financials) ← Skip
    - Hard: MSFT (sector=45 Tech, group=4520 Hardware) ← Different group
    - Hardest: GOOGL (group=4510 Software, but different beta) ← Same group, different risk
    """
    # Medium-hard: Same GICS sector (11 classes), different industry group (25 classes)
    same_sector_diff_group = [
        s for s in batch 
        if s.gsector == target_stock.gsector 
        and s.ggroup != target_stock.ggroup
    ]
    
    # Hardest: Same GICS industry group (ggroup), different factor profile
    same_group = [s for s in batch if s.ggroup == target_stock.ggroup]
    
    # Within same group, find stocks with different beta (market risk)
    diff_beta_same_group = [
        s for s in same_group 
        if abs(s.beta - target_stock.beta) > 0.3
    ][:n_hard//2]
    
    # Also find stocks with different momentum but similar beta
    diff_mom_same_beta = [
        s for s in same_group
        if abs(s.beta - target_stock.beta) < 0.1
        and abs(s.mom_12_1m - target_stock.mom_12_1m) > 0.5
    ][:n_hard//2]
    
    return diff_beta_same_group + diff_mom_same_beta
```

**Why GICS structure matters:**
- Random cross-sector negatives (Energy vs Healthcare) are trivial — model learns nothing
- Same `ggroup` (industry group) negatives force model to learn subtle distinctions
- This is exactly the within-sector discrimination PMs need for substitutes
- Using `ggroup` (25 classes) not `gsector` (11) — more granular, better for small-caps

**Sector embedding double duty:**
- **Feature role:** Passed through nn.Embedding(11, 8) + nn.Embedding(25, 16)
- **Sampling role:** Structures contrastive training (same ggroup = hardest negatives)
- Makes sector usage far more principled than just appending a category code

---

## Loss Functions

### 1. InfoNCE (Baseline) with Dot Product

**Corrected formulation (Issue #5 clarification):**
```
L = -log[ exp(sim(t_i, f_i)/τ) / Σ_j exp(sim(t_i, f_j)/τ) ]
```

Where:
- t_i = temporal embedding of stock i (128-dim)
- f_i = fundamental embedding of stock i (128-dim)
- sim(t_i, f_i) = dot product (cosine similarity) between two 128-dim vectors
- **NO concatenation in loss computation**
- τ = temperature (0.07 default)

**Implementation:** `info-nce-pytorch` package

---

### 2. RankSCL (Ordinal Similarity)

**What it does:** Captures degrees of similarity, not just binary

**Example:**
```
Target: AAPL
Candidates: MSFT (very similar), GOOGL (similar), XOM (different)

InfoNCE: Pulls MSFT/GOOGL closer, pushes XOM away
         (ignores MSFT vs GOOGL distinction)
         
RankSCL: MSFT > GOOGL > XOM (preserves rank order)
```

**When to use:** If we have ranking data about substitute quality

---

### 3. Sigmoid-Softmax (Third Ablation)

**From ACM 2024:** Shows sigmoid-softmax works best for financial embeddings

**Plan:** Test all three in ablations
- InfoNCE (baseline)
- RankSCL (ordinal)
- Sigmoid-Softmax (financial-specific)

---

## Training Strategy: Rolling Windows vs Static Training

### The Problem with Static Training Periods

**Research confirms:** Financial markets suffer from **concept drift** - relationships between features change over time due to:
- Regime shifts (bull/bear markets, high/low volatility)
- Structural changes (new regulations, market structure evolution)
- Macro shocks (COVID, financial crises, geopolitical events)

> *"Static models are particularly at risk of drift... regularly updating a model can arm it against the threat of decreased model performance over time"* - Element61 on Concept Drift

> *"Simple models with daily rolling windows achieve lower prediction error than complex ML models fitted with static windows"* - EmergentMind on Rolling Window Strategy

**Key insight:** A model trained on 2010-2022 data will degrade when deployed in 2024 because market dynamics shift.

### Two-Phase Training Strategy

#### Phase 1: Initial Training (For Backtesting/Validation)
**Purpose:** Build and validate the architecture before deployment

**Static split approach (what MVP.md specifies):**
- Train: 2010–2022 (12 years of historical data)
- Validation: 2023 (tune hyperparameters)
- Test: 2024 (out-of-sample stress testing)

This gives us a baseline model that we can validate on known historical events (COVID crash, meme stocks, etc.).

#### Phase 2: Production Deployment (Rolling/Online Learning)
**Purpose:** Keep model current as markets evolve

**Options (from research):**

**Option A: Periodic Full Retraining (Simplest)**
- Retrain full model every month or quarter
- Use rolling window: always last 5-10 years of data
- Most common in production financial ML

**Option B: Triggered Retraining (Efficient)**
- Monitor embedding quality or feature drift metrics
- Retrain only when drift detected
- Can reduce retraining frequency by 10-100x while maintaining performance

**Option C: Online/Incremental Learning (Advanced)**
- Update model weights daily with new data
- No full retraining needed
- Harder to implement but most responsive

**Recommended hybrid approach:**
```
Production Mode:
├── Base model: Pre-trained on 2015-2024 (rolling 10 years)
├── Daily: Incremental updates or inference only
├── Monthly: Evaluate if full retrain needed
├── Quarterly: Scheduled full retrain with last 10 years
└── Trigger: Emergency retrain if regime shift detected
```

### Training Configuration (Initial Phase)

**Data split (time-based, critical for finance):**
- Train: 2010–2022 (12 years) - for architecture validation
- Validation: 2023 (1 year)
- Test: 2024 (out-of-sample, stress events)

**Production will use rolling windows instead**

**Training details:**
```python
optimizer: AdamW
learning_rate: 1e-4 (with cosine decay)
batch_size: 256-512 (as large as fits in memory)
temperature: 0.07 (InfoNCE default)
embedding_dim: 256 (128 + 128)
max_epochs: 100 (with early stopping on validation)
```

### Why This Matters for Our Use Case

**Stock substitution is particularly sensitive to regime changes:**
- In 2021 (meme stock era): correlations broke down, fundamentals mattered less
- In 2022 (rate hikes): growth/value dynamics shifted dramatically
- In 2024 (AI boom): tech sector relationships changed

**A model trained on 2010-2022 would recommend growth stocks as substitutes in 2022-2024, potentially missing the regime shift to value.**

**Solution:** Rolling window training ensures the model always learns from recent market dynamics.

---

## Validation: How Do We Know It's Working?

**Embedding space quality metrics:**

1. **Sector clustering:** Stocks in same sector should cluster together
   - Metric: Silhouette score for sector labels

2. **Beta preservation:** High-beta stocks should be near other high-beta stocks
   - Metric: Correlation between beta difference and embedding distance

3. **Substitute tracking error:** If we use embedding neighbors as substitutes, do they track well?
   - Metric: Out-of-sample tracking error on validation set

4. **Rank quality:** For RankSCL, measure if ranking is preserved
   - Metric: Kendall's tau correlation between true and predicted rankings

5. **Beta preservation (refined):**
   - **What:** After recommending substitute, measure how well beta is preserved
   - **Metric:** `|beta_substitute − beta_original|` (absolute difference)
   - **Why:** PMs ask "will my market exposure change?" - beta is key risk metric
   - Note: Only using market beta (FF5 factors excluded from methodology)

---

## Summary of Architecture Choices (REVISED)

| Component | Previous Proposal | **Revised (Post-Review)** | Reason |
|-----------|-------------------|---------------------------|--------|
| **Temporal Encoder** | TCN only | **TCN + lightweight Transformer (BiMT-TCN)** | 2025 research: Transformer outperforms TCN on stock sequences |
| **Tabular Encoder** | ResNet MLP + ad-hoc attention | **TabMixer** | Purpose-built for tabular, handles missing data, citable |
| **Sector Classification** | GICS (unspecified) | **GICS `gsector` + `ggroup` from Compustat** | Market-oriented (not production), empirically best for finance |
| **Sector Encoding** | Learned embeddings (unspecified) | **nn.Embedding: gsector→8-dim, ggroup→16-dim** | Discovers sector relationships automatically |
| **Positive pairs** | Same stock different time | **Statistical test (KS test) OR regime-stable windows** | Avoids noisy positives across regime changes |
| **Negative sampling** | In-batch (defer hard negatives) | **GICS-structured: same `ggroup`, different beta/vol** | Random negatives too easy; within-group hardest |
| **Loss** | InfoNCE → RankSCL | **InfoNCE + RankSCL + Sigmoid-Softmax (ablations)** | ACM 2024 shows sigmoid-softmax best for financial embeddings |
| **Joint embedding** | Concatenate 128+128 | **Separate during training (dot product loss), concat at inference** | Matches CLIP correctly; clarifies architecture |
| **Training** | 2010-2022 static | **Static for validation → Rolling window for production** | Concept drift in financial markets |
| **Validation** | Existing metrics | **Beta preservation metric** | Measures if market exposure preserved in substitutes |
| **Pre-trained** | Train from scratch | **Consider pre-trained sector embeddings** | User suggestion: small pre-trained GICS embeddings |

**Philosophy:** Start simple, validate embedding quality, upgrade components if needed. The dual-encoder architecture is the key innovation—specific encoder choices evolve based on research.

---

## Open Questions / Decisions

1. **Training Strategy (CRITICAL - based on research):**
   - **Current MVP plan:** Static 2010-2022 training (for initial validation only)
   - **Production need:** Rolling window (last 5-10 years, refreshed monthly/quarterly)
   - **Research shows:** Financial models degrade due to concept drift; rolling windows outperform static
   - **Decision:** 
     - Phase 1: Use static split for backtesting (proves architecture works)
     - Phase 2: Implement rolling window retraining for production
     - Monitor: Track embedding quality to detect when retrain needed

2. **Positive pair definition (Issue #3):**
   - **Option A (best):** Statistical test—stocks i and j are positive pair if 60-day return distributions cannot be rejected as same (KS test, p > 0.05)
   - **Option B (practical):** Same stock, but only adjacent windows where rolling beta/vol haven't shifted > 1 std
   - **Option C (MVP):** Same stock different times (fastest, but known limitation)
   - **Decision:** Start with C for MVP, implement A for production differentiation

3. **Temporal encoder depth:**
   - TCN: 3 layers (local patterns)
   - Transformer: 2 layers, 4 heads (global dependencies)
   - Can ablate if training speed issues

4. **TabMixer vs alternatives:**
   - TabMixer is theoretically grounded, handles missing values
   - Can compare against ResNet-MLP baseline in ablations

5. **Hard negative sampling (Issue #4 - IMPLEMENT NOW):**
   - **Sector-aware from day 1:** Same sector, different beta/vol quintiles
   - **Not deferred:** Critical for within-sector discrimination
   - **Implementation:** Sample 50% hard negatives, 50% random (for diversity)

6. **Loss function ablations:**
   - Test all three: InfoNCE, RankSCL, Sigmoid-Softmax
   - Let validation metrics decide winner

---

## References

### Core Architecture Papers
- **CLIP:** "Learning Transferable Visual Models From Natural Language Supervision" (OpenAI, 2021)
- **TCN:** "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" (Bai et al., 2018)

### Updated Temporal Encoder (2025 Research)
- **BiMT-TCN:** "Stock Price Prediction Based on Sliding Window and Multi-Task Learning" (arXiv 2504.16361, 2025) - Shows Transformer outperforms TCN on stock sequences: https://arxiv.org/html/2504.16361v1
- **TCN-Transformer Hybrid:** "BiMT-TCN: Bidirectional Multi-Task Temporal Convolutional Network" (ScienceDirect 2025) - TCN extracts local features, Transformer captures global: https://www.sciencedirect.com/science/article/pii/S0950705125013048

### Updated Tabular Encoder (2025 Research)
- **TabMixer:** "TabMixer: A Simple Yet Strong Architecture for Tabular Data" (PMC 2025) - Purpose-built for tabular, handles missing values: https://pmc.ncbi.nlm.nih.gov/articles/PMC12053537/
- **FT-Transformer:** "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., NeurIPS 2021)

### Contrastive Learning & Positive Pairs
- **ACM Contrastive CL (2024):** "Contrastive Learning for Financial Time Series" - Uses statistical hypothesis test on return distributions for positive pairs: https://arxiv.org/pdf/2407.18645.pdf
- **RankSCL:** "Rank Supervised Contrastive Learning for Time Series Classification" (Ren et al., ICDM 2024)
- **InfoNCE:** "Representation Learning with Contrastive Predictive Coding" (Oord et al., 2018)
- **Financial Text Embeddings (2024):** OpenReview paper on hard negative mining for financial embeddings: https://arxiv.org/html/2408.15710v1

### Training & Concept Drift
- **Concept Drift:** "Concept Drift: What Is It and How To Address It" (Element61) - Static models at risk of drift: https://www.element61.be/en/resource/concept-drift-what-it-and-how-address-it
- **Rolling Windows:** "Rolling Window Strategy" (EmergentMind) - Rolling windows beat static for finance: https://www.emergentmind.com/topics/rolling-window-strategy
- **Entropy-Triggered Retraining:** "Entropy-Triggered Retraining in Deployed ML Systems" (arXiv 2026) - Selective retraining based on drift detection: https://arxiv.org/html/2601.00554

### Model Retraining Best Practices
- **KDnuggets:** "The Ultimate Guide to Model Retraining" - Financial models need regular retraining: https://www.kdnuggets.com/2019/12/ultimate-guide-model-retraining.html
- **Comet:** "Importance of Machine Learning Model Retraining in Production" - Periodic retraining for financial data: https://www.comet.com/site/blog/importance-of-machine-learning-model-retraining-in-production/
- **ML in Production:** "The Ultimate Guide to Model Retraining" - Rolling windows for time series: https://mlinproduction.com/model-retraining/
- **PremAI:** "Continual Learning: How Modern Models Stay Smarter Over Time" - Hybrid approach: continual + periodic quarterly retraining: https://blog.premai.io/continual-learning-how-ai-models-stay-smarter-over-time/

### Financial ML & Market Dynamics
- **Subex:** "Machine Learning in Financial Markets" - Need frequent retraining due to regime shifts: https://www.subex.com/blog/machine-learning-in-financial-markets-applications-effectiveness-and-limitations/
- **ScienceDirect:** "Deep Learning for Financial Forecasting" - Need frequent retraining in non-stationary markets: https://www.sciencedirect.com/science/article/pii/S1059056025008822
- **Resonanz Capital:** "Benefits and Pitfalls of ML in Trading" - Dynamic model updating required: https://resonanzcapital.com/insights/benefits-pitfalls-and-mitigation-strategies-of-applying-ml-to-financial-modelling
- **Frontiers in AI:** "AI in Financial Market Prediction" - Ensemble models that adjust weightings when conditions shift: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1696423/full
