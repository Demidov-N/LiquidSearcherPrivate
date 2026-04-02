"""Analyze data coverage after preprocessing."""

import pandas as pd
import sys

def analyze_coverage(parquet_path):
    """Analyze what percentage of stocks have fundamentals/GICS."""
    print(f"\n{'='*60}")
    print("DATA COVERAGE ANALYSIS")
    print(f"{'='*60}\n")
    
    df = pd.read_parquet(parquet_path)
    
    # Basic stats
    total_symbols = df['symbol'].nunique()
    total_rows = len(df)
    
    print(f"Total symbols: {total_symbols}")
    print(f"Total rows: {total_rows:,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print()
    
    # Check which columns exist
    print("Available columns:")
    for col in sorted(df.columns):
        print(f"  - {col}")
    print()
    
    # Fundamental columns to check
    fundamental_cols = ['beta', 'idiosyncratic_vol', 'roe', 'roa', 'debt_to_equity', 
                       'price_to_book', 'price_to_earnings', 'market_cap']
    
    gics_cols = ['gsector', 'ggroup']
    
    # Check coverage per symbol
    print("COVERAGE BY SYMBOL:")
    print(f"{'='*60}")
    
    symbol_stats = []
    for symbol in sorted(df['symbol'].unique()):
        symbol_df = df[df['symbol'] == symbol]
        
        stats = {
            'symbol': symbol,
            'rows': len(symbol_df),
            'has_beta': symbol_df['beta'].notna().any() if 'beta' in df.columns else False,
            'has_fundamentals': False,
            'has_gics': False,
        }
        
        # Check if any fundamental columns exist and have data
        for col in fundamental_cols:
            if col in df.columns and symbol_df[col].notna().any():
                stats['has_fundamentals'] = True
                break
        
        # Check GICS
        for col in gics_cols:
            if col in df.columns and symbol_df[col].notna().any():
                stats['has_gics'] = True
                break
        
        symbol_stats.append(stats)
    
    stats_df = pd.DataFrame(symbol_stats)
    
    # Calculate percentages
    with_beta = stats_df['has_beta'].sum()
    with_fundamentals = stats_df['has_fundamentals'].sum()
    with_gics = stats_df['has_gics'].sum()
    
    print(f"\nSUMMARY:")
    print(f"  Symbols with beta: {with_beta}/{total_symbols} ({with_beta/total_symbols*100:.1f}%)")
    print(f"  Symbols with fundamentals: {with_fundamentals}/{total_symbols} ({with_fundamentals/total_symbols*100:.1f}%)")
    print(f"  Symbols with GICS: {with_gics}/{total_symbols} ({with_gics/total_symbols*100:.1f}%)")
    print()
    
    # Show examples of missing data
    print(f"\nSYMBOLS WITHOUT FUNDAMENTALS ({total_symbols - with_fundamentals} symbols):")
    print(f"{'='*60}")
    missing_fund = stats_df[~stats_df['has_fundamentals']]['symbol'].tolist()
    print(f"  {', '.join(missing_fund[:20])}")
    if len(missing_fund) > 20:
        print(f"  ... and {len(missing_fund) - 20} more")
    print()
    
    print(f"\nSYMBOLS WITHOUT GICS ({total_symbols - with_gics} symbols):")
    print(f"{'='*60}")
    missing_gics = stats_df[~stats_df['has_gics']]['symbol'].tolist()
    print(f"  {', '.join(missing_gics[:20])}")
    if len(missing_gics) > 20:
        print(f"  ... and {len(missing_gics) - 20} more")
    print()
    
    # Row-level coverage (what % of rows have data)
    print(f"\nROW-LEVEL COVERAGE:")
    print(f"{'='*60}")
    for col in fundamental_cols + gics_cols:
        if col in df.columns:
            coverage = df[col].notna().mean() * 100
            print(f"  {col}: {coverage:.1f}% of rows have data")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}\n")
    
    return stats_df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Check for existing processed files
        import os
        from pathlib import Path
        
        data_dir = Path("data/processed")
        if data_dir.exists():
            parquet_files = list(data_dir.glob("*.parquet"))
            if parquet_files:
                print("Found processed files:")
                for i, f in enumerate(parquet_files, 1):
                    print(f"  {i}. {f}")
                print("\nRun: python analyze_coverage.py data/processed/FILENAME.parquet")
            else:
                print("No parquet files found in data/processed/")
                print("Process data first with:")
                print("  python -m scripts.preprocess_features ...")
        else:
            print("data/processed/ directory not found")
    else:
        analyze_coverage(sys.argv[1])
