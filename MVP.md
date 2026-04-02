# PROJECT BRIEF: Liquidity Risk Management — Stock Substitute Recommendation System

> **⚠️ NOTE: This is a PROTOTYPE.** The goal is to get something functional first. Clean code, perfect documentation, and edge cases can be handled later. Speed > perfection for now.
>
> **BASED ON:** This specification is derived from the liquidity project MVP. It defines the structure and workflow patterns for this new implementation.

_Version: 1.0 | Updated: March 2026 | Status: Pre-implementation_

---

## METADATA

- project_type: Quantitative Finance / Deep Learning
- institutional_buyer: Asset managers (open-ended funds, long-only equity)
- problem_class: Liquidity risk, portfolio management, stock substitution
- novelty_claim: First dual-encoder contrastive model combining temporal price behavior
  and fundamental characteristics with liquidity-aware post-filtering, designed
  specifically for PM liquidity-shock-driven substitution
- backtest_period: 2010–2024
- universe: Russell 2000 + S&P 400 (~2,400 US small/mid-cap stocks)
- data_platform: WRDS (Wharton Research Data Services)

---

## PROBLEM STATEMENT

### Core Problem

Small-cap and international equity positions are structurally illiquid.
When a liquidity shock occurs, portfolio managers must either:
  (A) Force the illiquid trade — absorbing full market impact + spread cost
  (B) Freeze the position — violating redemption obligations

Neither is acceptable. There is no tool that instantly identifies a
liquid, fundamentally equivalent substitute.

### Quantitative Evidence

- BIS WP 1229 (2024): US small-cap avg bid-ask spread = 148 bps vs 17 bps large-cap
- BIS WP 1229 (2024): Spread fragility adds $400M–$1B annually in trading costs
- FCA UK (2021): COVID Mar 2020 — spreads spiked 5x; market depth fell 75%;
  impairment persisted through Feb 2021 (>11 months)
- BIS Quarterly Review Sep 2024: Small caps disproportionately hit in Aug 2024
  carry unwind

### Shock Trigger Conditions

- Bid-ask spread exceeds 2% of stock price, OR
- 5-day average volume craters >50% vs. 20-day moving average, OR
- Amihud ILLIQ innovation (DArLiQ residual) z-score > 2.0

---

## SYSTEM ARCHITECTURE

### Design Principle

TWO-STAGE ARCHITECTURE:
  Stage A — Embedding: Learn stock similarity on risk/return characteristics
  Stage B — Filtering: Apply hard liquidity constraints to ranked candidates

CRITICAL RULE: Liquidity is a CONSTRAINT, not a similarity dimension.
Do NOT embed liquidity features. This would cause the model to recommend
similarly illiquid stocks.

### Pipeline Summary

Shock detected on stock X
↓
[STAGE 2] Dual-encoder embedding → 256-dim vector for all stocks
↓
[STAGE 3] Cosine similarity → Top 20-50 nearest neighbors to X
↓
[STAGE 3] RankSCL re-ranking → ordered by ordinal embedding distance
↓
[STAGE 4] Liquidity hard filters → remove failing candidates
↓
Output: 5-10 liquid, fundamentally equivalent substitutes surfaced to PM

text

---

## STAGE 0: DATA COLLECTION

### Primary Platform

WRDS (Wharton Research Data Services) — accessed via Brown University

### Datasets

| dataset         | wrds_library   | key_variables                              | used_for                                      |
|-----------------|----------------|--------------------------------------------|-----------------------------------------------|
| CRSP Daily      | crsp           | PRC, RET, VOL, SHROUT, BIDLO, ASKHI        | OHLCV, beta, returns, liquidity metrics       |
| Compustat Annual| comp           | AT, SEQ, NI, CSHO, PRCC_F, EPSPX           | P/E, P/B, ROE, market cap, earnings quality   |
| Compustat Qtrly | comp           | Same, quarterly frequency                 | More timely fundamental updates               |
| TAQ             | taq            | BID, ASK, PRICE, SIZE (intraday)           | True microstructure spreads (expensive)       |
| IBES Summary    | ibes           | EPS estimates, actuals, surprise           | Earnings quality, analyst coverage proxy      |
| OptionMetrics   | optionm        | Implied vol surface                        | Forward-looking vol (optional enrichment)     |
| Ken French      | public (free)  | FF5 daily factor returns                   | SMB, HML, MOM, RMW, CMA loadings per stock   |

### Notes on TAQ

TAQ intraday data is expensive. Fallback strategy:
  Use ML-based liquidity estimation (ScienceDirect 2025) — boosting trees
  trained to estimate effective bid-ask spread from daily OHLCV only.
  Validated to closely match true microstructure measures.
  Features: daily return, dollar volume, high-low range, turnover,
  rolling vol, market cap, price level, zero-return ratio.

### Data Splits

- train:      2010–2022
- validation: 2023
- test:       2024 (contains Aug 2024 carry unwind — out-of-sample stress event)

---

## STAGE 1: FEATURE ENGINEERING

### Normalization Methods Reference

| situation                                      | method                          | formula                                    |
|------------------------------------------------|---------------------------------|--------------------------------------------|
| Continuous factor, roughly normal              | Cross-sectional z-score         | z = (x - μ_t) / σ_t across all stocks     |
| Continuous factor, right-skewed                | log then cross-sectional z-score| z = (log(x) - μ_t) / σ_t                  |
| Fat-tailed / extreme outliers (momentum, P/E)  | Cross-sectional rank            | rank(x_t) / N → scaled to [0, 1]          |
| Within-stock temporal behavior (OHLCV)         | Rolling time-series z-score     | z = (x_t - μ_rolling) / σ_rolling, 252-day|
| Categorical (sector, industry)                 | Learned trainable embedding     | nn.Embedding(n_classes, dim)               |
| Pre-processing guard (always first)            | Winsorize                       | clip to [1%, 99%] or [2%, 98%]             |

Cross-sectional = computed across all stocks at time t (daily recompute)
Time-series = computed over stock's own 252-day rolling history

### Group 1: Systematic Risk Exposure

purpose: Preserve portfolio's market sensitivity and factor loadings

| factor              | computation                                  | raw_range   | transformation                        |
|---------------------|----------------------------------------------|-------------|---------------------------------------|
| market_beta_60d     | OLS regression R_stock vs R_market, 60 days  | [-0.5, 3.0] | Winsorize [1%,99%] → CS z-score       |
| downside_beta       | Beta on negative market return days only     | [-0.5, 3.0] | Winsorize [1%,99%] → CS z-score       |
| smb_loading         | FF5 regression coefficient                   | [-2, 2]     | CS z-score                            |
| hml_loading         | FF5 regression coefficient                   | [-2, 2]     | CS z-score                            |
| mom_loading         | FF5 regression coefficient                   | [-2, 2]     | CS z-score                            |
| rmw_loading         | FF5 regression coefficient                   | [-2, 2]     | CS z-score                            |
| cma_loading         | FF5 regression coefficient                   | [-2, 2]     | CS z-score                            |

research: Beta-sorted portfolios (Cemmap 2024), Multi-factor neutral (arXiv Dec 2024),
          Systematic mispricing (Management Science 2025)

### Group 2: Volatility Profile

purpose: Ensure substitute has similar variance contribution

| factor              | computation                                  | raw_range      | transformation             |
|---------------------|----------------------------------------------|----------------|----------------------------|
| realized_vol_20d    | std(daily_returns) × sqrt(252), 20-day       | [0.005, 0.15]  | log(vol) → CS z-score      |
| realized_vol_60d    | Same, 60-day window                          | [0.005, 0.15]  | log(vol) → CS z-score      |
| idiosyncratic_vol   | Residual vol after beta regression           | [0.003, 0.12]  | log(vol) → CS z-score      |
| vol_of_vol          | std of rolling 20-day vol estimates          | Right-skewed   | Winsorize → CS z-score     |

note: Log first because vol is multiplicative/lognormal.
research: Low-vol + momentum (T&F 2024), Idiosyncratic vol cross-section (MS 2025)

### Group 3: Return Momentum & Trend

purpose: Prevent substituting high-momentum stocks with mean-reverting ones

| factor              | computation                                  | raw_range       | transformation             |
|---------------------|----------------------------------------------|-----------------|----------------------------|
| mom_1m              | (P_t / P_{t-21}) - 1                         | [-30%, +50%]    | CS rank → [0, 1]           |
| mom_3m              | (P_t / P_{t-63}) - 1                         | [-40%, +80%]    | CS rank → [0, 1]           |
| mom_6m              | (P_t / P_{t-126}) - 1                        | [-50%, +150%]   | CS rank → [0, 1]           |
| mom_12_1m           | Jegadeesh-Titman: 12M return skip last 1M    | [-60%, +200%]   | CS rank → [0, 1]           |
| macd                | EMA(12) - EMA(26)                            | [-5, +5]        | CS z-score                 |

note: Rank normalization because return distributions are fat-tailed.
      A meme stock at 3000% would dominate z-score space.
research: Factor performance 2024 (Verdad/Carson Group), RIC-NN (AAAI 2020)

### Group 4: Valuation & Fundamentals

purpose: Prevent replacing value stocks with growth stocks (different macro sensitivity)

| factor              | computation                                  | raw_range       | transformation                        |
|---------------------|----------------------------------------------|-----------------|---------------------------------------|
| log_mktcap          | ln(price × shares_outstanding)               | [$50M, $3T]     | log(mktcap) → CS z-score              |
| pe_ratio            | Price / EPS_trailing                         | [5, 200]+       | Winsorize [2%,98%] → CS rank → [0,1]  |
| pb_ratio            | Price / Book_value                           | [0.5, 50]+      | log(P/B) → CS z-score                 |
| roe                 | Net_income / Equity                          | [-50%, +80%]    | Winsorize [2%,98%] → CS z-score       |
| earnings_quality    | Accruals ratio or cash/earnings              | [-1, 1]         | CS z-score                            |

note: P/E uses rank because it can be negative, undefined (zero earnings),
      or extremely high. Rank handles all three cases without imputation.
research: Portfolio optimization with sectors (MDPI 2024), AQR strategic alloc.

### Group 5: Multi-Scale OHLCV Price Behavior

purpose: Capture how the stock moves behaviorally across multiple timeframes
source: SimStock (Hwang, Zohren, Lee — Oxford/UNIST, arXiv July 2024)

| factor              | formula                           | windows               | transformation                    |
|---------------------|-----------------------------------|-----------------------|-----------------------------------|
| z_close             | (Close_t / Close_{t-1}) - 1       | 5, 10, 20-day MA      | Rolling z-score (252-day) per stock|
| z_high              | (High / Close) - 1                | Rolling               | Rolling z-score (252-day)         |
| z_low               | (Low / Close) - 1                 | Rolling               | Rolling z-score (252-day)         |
| z_volume            | (Volume_t / Volume_{t-1}) - 1     | 5, 10, 20-day MA      | Rolling z-score (252-day)         |
| ma_ratio_n          | (Price / MA_n) - 1                | 5, 10, 15, 20, 25-day | Rolling z-score                   |

note: Time-series normalization (not cross-sectional) because you want to detect
      how this stock deviates from its OWN norm — regime detection at stock level.

### Group 6: Sector / Industry (Categorical)

purpose: Structural anchor — substitutes must be economically related

| factor              | classes    | transformation                                    |
|---------------------|------------|---------------------------------------------------|
| gics_sector         | 11         | nn.Embedding(11, 8)  — trainable                  |
| gics_industry_group | 25         | nn.Embedding(25, 16) — trainable                  |

note: Do NOT one-hot encode. Learned embeddings discover sector relationships
      (e.g., Energy and Materials will naturally cluster close together).

### Ablation Plan

Experiment by including/excluding each group independently.
Primary evaluation metric: downstream tracking error of recommended substitutes.
Expected result: G1 (beta) and G5 (OHLCV) will be highest importance;
G4 (fundamentals) will help for value/growth differentiation.

---

## STAGE 2: DUAL-ENCODER CONTRASTIVE EMBEDDING MODEL

### Architecture

OHLCV features (G5)
↓
[Temporal Encoder] ← TCN or Transformer
↓
128-dim temporal embed ─────────┐
↓
[CLIP-style Contrastive Loss]
InfoNCE or RankSCL
↓
Joint 256-dim Stock Embedding
↑
128-dim fundamental embed ─────────┘
↑
[Tabular Encoder] ← FT-Transformer or MLP-Attention
↑
Factor features (G1-G4 + G6)

text

### Contrastive Loss Design

- Positive pairs: same stock's temporal and fundamental embeddings
- Negative pairs: different stocks' embeddings
- InfoNCE: standard baseline
- RankSCL (novel): ordinal supervision — encodes not just "similar/different"
  but HOW similar, producing continuous gradient in embedding space

### Training Config

- optimizer: AdamW
- batch_size: 256–512
- temperature: 0.07 (standard InfoNCE default)
- embedding_dim: 256 (128 per encoder)
- train_period: 2010–2022
- val_period: 2023
- test_period: 2024

### Novel Contribution vs. Existing Work

| paper                              | features_used              | gap_filled_by_this_system              |
|------------------------------------|----------------------------|----------------------------------------|
| Contrastive CL (ACM 2024)          | Returns only               | No fundamentals; no liquidity aware    |
| SimStock (Oxford 2024)             | OHLCV + sector             | No fundamentals; no liquidity filter   |
| Asset Embeddings (Chicago 2025)    | Portfolio holdings (13-F)  | Requires 13-F data; no explicit factors|
| Cross-Sectional Retrieval (2025)   | Future-aligned returns     | Returns-only; no liquidity constraint  |
| THIS SYSTEM                        | OHLCV + factors + FF5      | Dual-modal + hard liquidity constraint |

---

## STAGE 3: SUBSTITUTE RANKING

### Inference Pipeline

1. Shock detected on target stock X
2. Compute embedding of X using trained dual-encoder
3. Cosine similarity search across full universe → top 20–50 candidates
4. Re-rank using RankSCL ordinal distance
   (embedding Euclidean distance = direct substitute rank score)
5. Pass ranked candidates to liquidity filter

### Ablation Baselines

- Pearson correlation-based similarity (classical benchmark)
- TS2Vec embeddings + XGBoost LTR (ML benchmark)
- SimStock OHLCV-only model (partial model benchmark)
- RankNet (existing DL-to-rank benchmark)

---

## STAGE 4: LIQUIDITY HARD FILTERS

### Filter Logic

Applied AFTER ranking. Any candidate failing ANY gate is removed.
This is not a soft penalty — it is a binary exclude/include decision.

### Gates

| gate                    | metric                                    | threshold                                 | source                           |
|-------------------------|-------------------------------------------|-------------------------------------------|----------------------------------|
| amihud_illiq            | |R_daily| / DollarVolume                    | Bottom 30% of sector ILLIQ                | BIS WP 1229 (2024)               |
| illiq_innovation        | DArLiQ autoregressive residual            | z-score < 2.0 in last 10 days             | DArLiQ model (2023)              |
| dollar_volume           | Price × Volume, 20-day avg                | ≥ 2× target stock's dollar volume        | Market impact constraint         |
| est_bid_ask_spread      | ML-estimated effective spread from OHLCV  | < 50 bps                                  | ScienceDirect ML est. (2025)     |
| spread_volatility       | Rolling 5-day std of estimated spread     | < 20 bps                                  | L-VaR framework (SSRN 2021)      |
| zero_return_ratio       | Fraction of zero-return days in 60-day    | < 5%                                      | Structural illiquidity proxy     |
| turnover_ratio          | Volume / shares_outstanding               | > target stock's turnover                 | Float-normalized activity        |

### DArLiQ Decomposition Note

DArLiQ decomposes the Amihud ILLIQ series into:

- Long-run trend component (structural decline over months)
- Short-run autoregressive component (the acute shock)
Use ONLY the residual (innovation) as trigger signal.
Do not alert on structural trend — only on anomalous short-run deviations.

### Output

5–10 liquid substitute stocks, ranked by embedding similarity,
all passing liquidity gates. Surfaced to PM interface.

---

## STAGE 5: BACKTESTING & VALIDATION

### Stress Events for Backtesting

| event                    | date_range           | why_useful                                              |
|--------------------------|----------------------|---------------------------------------------------------|
| COVID Crash              | Feb 20 – Apr 3 2020  | Spreads 5×; depth -75%; cross-asset liquidity impairment|
| COVID Recovery Period    | Apr 2020 – Feb 2021  | Tests persistence: impairment lasted 11 months          |
| Meme Stock Volatility    | Jan 22 – Feb 5 2021  | Small-cap volume crater post-frenzy; exact trigger case |
| 2022 Rate Shock          | Jan – Oct 2022       | Macro regime shift; growth/value rotation               |
| August 2024 Carry Unwind | Aug 5–9, 2024        | Small caps hit hardest; most recent out-of-sample       |

### Primary Validation Metrics

| metric                       | computation                                                  | target           |
|------------------------------|--------------------------------------------------------------|------------------|
| tracking_error               | std(R_substitute - R_original) annualized                    | < 2% annualized  |
| transaction_cost_savings     | Amihud market impact (forced) - spread cost (substitute)     | > 30 bps         |
| beta_preservation            | |beta_substitute - beta_original|                            | < 0.10           |
| vol_preservation             | |vol_substitute - vol_original| annualized                   | < 1%             |
| factor_exposure_preservation | max |loading_substitute - loading_original| across FF5 factors| < 0.20           |

### Baseline Comparison

BASELINE A (null): Force the illiquid trade — accept full Amihud market impact
BASELINE B (classical): Pearson correlation nearest neighbor substitute
BASELINE C (partial ML): TS2Vec + XGBoost LTR substitute
PROPOSED: Dual-encoder + RankSCL + liquidity filter

Win conditions for proposed vs. baseline A:
  tracking_error AND transaction_cost_savings AND beta_preservation all met.

---

## KEY RESEARCH CITATIONS

| citation_id | authors_year                        | key_finding                                                          |
|-------------|-------------------------------------|----------------------------------------------------------------------|
| BIS_1229    | Aliyev et al., BIS WP 1229, Nov 2024| Small-cap spread 148bps vs 17bps; fragility adds $400M-1B/yr        |
| FCA_2021    | Mittendorf et al., FCA UK, May 2021 | COVID: spreads 5×, depth -75%, impairment 11+ months                |
| BIS_Q324    | BIS Quarterly Review, Sep 2024      | Aug 2024 carry unwind: small caps disproportionately hit             |
| SIMSTOCK    | Hwang et al., arXiv Jul 2024        | Multi-scale OHLCV features for temporal stock representation         |
| ACM_CL      | ACM DL 2024                         | Contrastive learning of asset embeddings from returns only           |
| ASSET_EMBED | Gabaix et al., Chicago Booth 2025   | Portfolio holdings as implicit stock embeddings via Word2Vec/BERT    |
| DARLIQ      | DArLiQ, 2023                        | Dynamic autoregressive liquidity decomposition (trend + shock)       |
| ML_LIQ      | ScienceDirect 2025                  | ML estimation of bid-ask spread from daily OHLCV without tick data  |
| LVAR        | SSRN 2021                           | Liquidity-adjusted VaR; spread volatility as key stress risk driver  |
| RANKSCL     | arXiv 2024                          | Rank-supervised contrastive learning for ordinal similarity          |
| STATCL      | arXiv Oct 2024                      | Non-stationarity-aware contrastive learning for regime shifts        |
| CROSSSEC    | arXiv Feb 2025                      | Cross-sectional asset retrieval via soft contrastive learning        |

---

## OPEN EXPERIMENTS / DECISIONS

| decision                          | options                                  | how_to_decide                                  |
|-----------------------------------|------------------------------------------|------------------------------------------------|
| Temporal encoder architecture     | TCN vs. Transformer                      | Ablation on val tracking error                 |
| Contrastive loss                  | InfoNCE vs. RankSCL vs. SoftCLT          | Ablation on val tracking error + rank quality  |
| Which factor groups to include    | G1–G6 individually and combined          | Systematic ablation; start with all, remove one|
| Liquidity estimation method       | TAQ true spreads vs. ML-estimated        | Cost/access constraint; ML est. is fallback    |
| Positive pair definition          | Same stock both encoders vs. temporally  | Compare: CLIP-style vs. time-windowed          |
|                                   | adjacent windows of same stock           |                                                |
| Shock threshold sensitivity       | 2% spread / 50% volume drop              | Sensitivity analysis on recall/precision       |

---

## IMPLEMENTATION CHECKLIST

### Data Collection

- [ ] Pull CRSP daily (2010–2024) for Russell 2000 + S&P 400 universe
- [ ] Pull Compustat annual + quarterly fundamentals, merge on PERMNO/GVKEY
- [ ] Download Ken French FF5 daily factors (free: mba.tuck.dartmouth.edu/pages/faculty/ken.french)
- [ ] Download IBES summary for analyst coverage + earnings surprise
- [ ] Decision: TAQ or ML-estimated spreads (recommend ML-estimated first)

### Feature Engineering

- [ ] Compute rolling 60-day betas vs. S&P 500 (or Russell 2000 index)
- [ ] Compute FF5 loadings via rolling 252-day regressions per stock
- [ ] Compute all 4 momentum windows
- [ ] Compute realized vol (20d, 60d) and idiosyncratic vol
- [ ] Compute log market cap, P/B, ROE, P/E from Compustat merge
- [ ] Compute OHLCV features (zClose, zHigh, zLow, zVolume, MA ratios)
- [ ] Map GICS codes to integer IDs
- [ ] Apply all normalizations (winsorize → transform → normalize)

### Liquidity Metrics

- [ ] Compute Amihud ILLIQ per stock per day from CRSP
- [ ] Fit DArLiQ model per stock; extract ILLIQ innovations
- [ ] Train ML bid-ask spread estimator on OHLCV features
- [ ] Compute zero-return ratio, turnover ratio, dollar volume

### Model Training

- [ ] Build temporal encoder (TCN or Transformer)
- [ ] Build tabular encoder (FT-Transformer)
- [ ] Implement InfoNCE loss (baseline)
- [ ] Implement RankSCL loss (proposed)
- [ ] Train dual-encoder, monitor embedding space quality
- [ ] Validate: sector clustering, beta ordering in embedding space

### Backtesting

- [ ] Identify and label stress event dates
- [ ] Simulate shock detection → substitute retrieval → filter
- [ ] Compute tracking error, transaction cost savings, factor preservation
- [ ] Compare vs. all baselines
- [ ] Statistical significance tests on metric differences

---
