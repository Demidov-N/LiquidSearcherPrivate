"""Stock similarity analysis: compare embedding similarity vs feature correlation."""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Literal
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from src.models.dual_encoder import DualEncoder
from src.training.data_module import TEMPORAL_FEATURE_NAMES, TABULAR_CONTINUOUS_NAMES

SECTOR_MAP = {
    '10': 0, '15': 1, '20': 2, '25': 3, '30': 4,
    '35': 5, '40': 6, '45': 7, '50': 8, '55': 9, '60': 10
}

FEATURE_GROUPS = {
    'momentum': ['mom_1m', 'mom_3m', 'mom_6m', 'mom_12m', 'mom_12_1m'],
    'volatility': ['realized_vol_20d', 'realized_vol_60d', 'beta', 'idiosyncratic_vol'],
    'fundamentals': ['roe', 'roa', 'debt_to_equity', 'price_to_book', 'price_to_earnings'],
    'size': ['market_cap'],
    'price_action': ['z_close', 'z_volume', 'ma_ratio_5d', 'ma_ratio_20d'],
}


def _prepare_features(row: pd.Series):
    """Prepare model inputs from a dataframe row."""
    temporal = torch.zeros(1, 60, 13)
    for i, col in enumerate(TEMPORAL_FEATURE_NAMES):
        if col in row:
            val = row[col]
            if not pd.isna(val):
                temporal[0, -1, i] = float(val)
    
    tabular = torch.zeros(1, 15)
    for i, col in enumerate(TABULAR_CONTINUOUS_NAMES):
        if col in row:
            val = row[col]
            if not pd.isna(val):
                tabular[0, i] = float(val)
    
    categorical = torch.zeros(1, 2, dtype=torch.long)
    if 'gsector' in row:
        val = row['gsector']
        if not pd.isna(val):
            sector_code = str(int(float(val)))
            if sector_code in SECTOR_MAP:
                categorical[0, 0] = SECTOR_MAP[sector_code]
    
    return temporal, tabular, categorical


def _compute_feature_correlation(
    query_row: pd.Series,
    candidates_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    """Compute correlation-based similarity using raw features."""
    query_features = []
    for col in feature_cols:
        val = query_row.get(col)
        query_features.append(0.0 if pd.isna(val) else float(val))
    query_features = np.array(query_features)
    
    candidate_features = []
    for col in feature_cols:
        candidate_features.append(candidates_df[col].fillna(0).values)
    candidate_features = np.column_stack(candidate_features)
    
    query_norm = query_features / (np.linalg.norm(query_features) + 1e-8)
    candidate_norms = candidate_features / (np.linalg.norm(candidate_features, axis=1, keepdims=True) + 1e-8)
    
    similarities = candidate_norms @ query_norm
    
    return pd.Series(similarities, index=candidates_df.index)


def analyze_stock_similarity(
    ticker: str,
    feature_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    period_start: str = "2019-01-01",
    period_end: str = "2020-01-31",
    aggregation: Literal["end_period", "mean"] = "end_period",
) -> dict:
    """Analyze similarity of one stock to all others."""
    print(f"\n{'='*70}")
    print(f"STOCK SIMILARITY ANALYSIS: {ticker}")
    print(f"{'='*70}\n")
    
    print(f"Loading data from {feature_path}...")
    df = pd.read_parquet(feature_path)
    df['date'] = pd.to_datetime(df['date'])
    
    mask = (df['date'] >= period_start) & (df['date'] <= period_end)
    period_df = df[mask].copy()
    
    if aggregation == "end_period":
        stocks_df = period_df.sort_values('date').groupby('symbol').last().reset_index()
    else:
        numeric_cols = period_df.select_dtypes(include=['float64', 'int64']).columns
        stocks_df = period_df.groupby('symbol')[numeric_cols].mean().reset_index()
        stocks_df['date'] = pd.Timestamp(period_end)
        stocks_df['gsector'] = period_df.groupby('symbol')['gsector'].first().values
        stocks_df['ggroup'] = period_df.groupby('symbol')['ggroup'].first().values
    
    print(f"Period: {period_start} to {period_end}")
    print(f"Stocks: {len(stocks_df)}\n")
    
    query_idx = stocks_df[stocks_df['symbol'] == ticker].index
    if len(query_idx) == 0:
        raise ValueError(f"Ticker {ticker} not found in period")
    query_idx = query_idx[0]
    query_row = stocks_df.iloc[query_idx]
    
    print(f"Query: {ticker}")
    print(f"  Sector: {query_row.get('gsector', 'N/A')}")
    print(f"  Market Cap (z): {query_row.get('market_cap', 'N/A'):.3f}")
    print(f"  Beta: {query_row.get('beta', 'N/A'):.3f}")
    print()
    
    print("Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    hyperparams = checkpoint.get('hyper_parameters', {})
    
    model = DualEncoder(
        temporal_input_dim=hyperparams.get('temporal_input_dim', 13),
        tabular_continuous_dim=hyperparams.get('tabular_continuous_dim', 15),
        embedding_dim=hyperparams.get('embedding_dim', 128),
    )
    
    state_dict = checkpoint.get('state_dict', checkpoint)
    if any(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    device = torch.device('cpu')
    model.to(device)
    
    print("Computing embeddings...")
    embeddings = []
    with torch.no_grad():
        for idx, row in stocks_df.iterrows():
            temporal, tabular, categorical = _prepare_features(row)
            temporal = temporal.to(device)
            tabular = tabular.to(device)
            categorical = categorical.to(device)
            
            emb = model.get_joint_embedding(temporal, tabular, categorical)
            embeddings.append(emb.cpu().numpy())
            
            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1}/{len(stocks_df)}")
    
    embeddings = np.vstack(embeddings)
    embeddings_tensor = torch.from_numpy(embeddings)
    
    print("Computing similarities...\n")
    query_emb = embeddings_tensor[query_idx:query_idx+1]
    
    embedding_sim = F.cosine_similarity(query_emb, embeddings_tensor).numpy().flatten()
    
    feature_sims = {}
    for group_name, feature_cols in FEATURE_GROUPS.items():
        available_cols = [c for c in feature_cols if c in stocks_df.columns]
        if available_cols:
            feature_sims[group_name] = _compute_feature_correlation(
                query_row, stocks_df, available_cols
            )
    
    results = pd.DataFrame({
        'ticker': stocks_df['symbol'].values,
        'sector': stocks_df['gsector'].values,
        'ggroup': stocks_df['ggroup'].values,
        'market_cap': stocks_df['market_cap'].values,
        'beta': stocks_df['beta'].values,
        'embedding_similarity': embedding_sim,
    })
    
    for group_name, sims in feature_sims.items():
        results[f'feature_{group_name}'] = sims.values
    
    results = results[results['ticker'] != ticker].copy()
    results = results.sort_values('embedding_similarity', ascending=False).reset_index(drop=True)
    results['rank'] = results.index + 1
    results['same_sector'] = results['sector'] == query_row['gsector']
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"similarity_{ticker}_{period_end}.csv"
    results.to_csv(output_path, index=False)
    print(f"Saved: {output_path}\n")
    
    print(f"{'='*70}")
    print(f"TOP 20 MOST SIMILAR TO {ticker}")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'Ticker':<8} {'Sector':<7} {'Emb Sim':<10} {'Momentum':<10} {'Fundamentals':<12} {'Same Sector'}")
    print("-" * 70)
    
    for i, row in results.head(20).iterrows():
        momentum = f"{row.get('feature_momentum', np.nan):.3f}" if 'feature_momentum' in results.columns else "N/A"
        fundamentals = f"{row.get('feature_fundamentals', np.nan):.3f}" if 'feature_fundamentals' in results.columns else "N/A"
        same = "✓" if row['same_sector'] else ""
        print(f"{row['rank']:<5} {row['ticker']:<8} {str(row['sector']):<7} {row['embedding_similarity']:.4f}     {momentum:<10} {fundamentals:<12} {same}")
    
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    
    top50 = results.head(50)
    sector_counts = top50['sector'].value_counts().sort_index()
    query_sector = query_row['gsector']
    
    print(f"\nSector distribution in top 50:")
    for sector, count in sector_counts.items():
        pct = count / 50 * 100
        marker = " ← Query sector" if sector == query_sector else ""
        print(f"  Sector {sector}: {count} ({pct:.0f}%){marker}")
    
    baseline_sector_pct = (stocks_df['gsector'] == query_sector).mean() * 100
    top50_sector_pct = (top50['sector'] == query_sector).mean() * 100
    lift = top50_sector_pct / baseline_sector_pct if baseline_sector_pct > 0 else 0
    
    print(f"\nSame-sector concentration:")
    print(f"  Baseline (all stocks): {baseline_sector_pct:.1f}%")
    print(f"  Top 50 similar: {top50_sector_pct:.1f}%")
    print(f"  Lift: {lift:.2f}x")
    
    print(f"\nCorrelation between embedding similarity and features:")
    for group_name in feature_sims.keys():
        col = f'feature_{group_name}'
        if col in results.columns:
            corr = results['embedding_similarity'].corr(results[col])
            print(f"  {group_name}: {corr:.3f}")
    
    print(f"\n{'='*70}\n")
    
    return {
        'output_path': str(output_path),
        'query_ticker': ticker,
        'query_sector': query_row.get('gsector'),
        'n_stocks': len(results),
        'top50_sector_pct': top50_sector_pct,
        'baseline_sector_pct': baseline_sector_pct,
        'sector_lift': lift,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze stock similarity")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker (e.g., AAPL)")
    parser.add_argument("--feature-path", type=str, default="data/processed/all_features.parquet")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/last.ckpt")
    parser.add_argument("--output-dir", type=str, default="results/similarity_analysis")
    parser.add_argument("--period-start", type=str, default="2019-01-01")
    parser.add_argument("--period-end", type=str, default="2020-01-31")
    
    args = parser.parse_args()
    
    analyze_stock_similarity(
        ticker=args.ticker,
        feature_path=args.feature_path,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        period_start=args.period_start,
        period_end=args.period_end,
    )
