"""Quick sanity check script for data pipeline - Core functionality only."""

import sys
sys.path.insert(0, '/home/redbear/projects/liquidity-new/.worktrees/data-collection')

import os
os.environ['WRDS_USERNAME'] = 'niknatan'
os.environ['WRDS_PASSWORD'] = 'Nikita201001!'

from src.data.wrds_loader import WRDSDataLoader
from src.features.processor import FeatureProcessor
from src.data.universe import SymbolUniverse
import pandas as pd
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Small test subset
TEST_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
START_DATE = '2023-01-01'
END_DATE = '2024-12-31'

def test_price_loading():
    """Test loading price data from WRDS."""
    logger.info("="*60)
    logger.info("TEST 1: Price Data Loading from WRDS")
    logger.info("="*60)
    logger.info(f"Symbols: {TEST_SYMBOLS}")
    logger.info(f"Date range: {START_DATE} to {END_DATE}")
    
    try:
        start_time = time.time()
        
        with WRDSDataLoader() as loader:
            logger.info("\nConnecting to WRDS and fetching prices...")
            prices = loader.fetch_prices_batch(TEST_SYMBOLS, START_DATE, END_DATE)
            
        elapsed = time.time() - start_time
        
        logger.info(f"\n✓ SUCCESS! Loaded {len(prices)} price rows in {elapsed:.1f}s")
        logger.info(f"  Columns: {list(prices.columns)}")
        logger.info(f"  Date range: {prices['date'].min()} to {prices['date'].max()}")
        logger.info(f"  Unique PERMNOs: {prices['permno'].nunique()}")
        logger.info(f"  Avg rows per symbol: {len(prices) / prices['permno'].nunique():.0f}")
        
        return True, prices
        
    except Exception as e:
        logger.error(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_feature_computation(prices_df):
    """Test feature computation with real data."""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Feature Computation with Polars")
    logger.info("="*60)
    
    try:
        start_time = time.time()
        
        logger.info(f"\nProcessing {len(prices_df)} rows...")
        processor = FeatureProcessor()
        
        # Add symbol column if missing (WRDS data has permno but not symbol)
        if 'symbol' not in prices_df.columns:
            prices_df = prices_df.copy()
            # Map permnos to symbols (simplified mapping)
            permno_to_symbol = {
                10107: 'AAPL',
                14593: 'MSFT', 
                84788: 'GOOGL',
                90319: 'AMZN',
                93436: 'TSLA'
            }
            prices_df['symbol'] = prices_df['permno'].map(permno_to_symbol)
        
        # Compute features (without betas/fundamentals)
        features = processor.process_batch(
            prices_df=prices_df,
            betas_df=None,
            fundamentals_df=None,
            gics_df=None,
        )
        
        elapsed = time.time() - start_time
        
        logger.info(f"\n✓ SUCCESS! Computed features in {elapsed:.1f}s")
        logger.info(f"  Output rows: {len(features)}")
        logger.info(f"  Features created: {len(features.columns)}")
        logger.info(f"  Sample features: {list(features.columns)[:10]}")
        
        # Save sample output
        output_path = Path('data/processed/sanity_check_output.parquet')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.head(1000).to_parquet(output_path, index=False)
        logger.info(f"  ✓ Sample saved to: {output_path}")
        
        return True, features
        
    except Exception as e:
        logger.error(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_batch_processing():
    """Test batch processing with SymbolUniverse."""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Batch Processing with tqdm Progress")
    logger.info("="*60)
    
    try:
        universe = SymbolUniverse(TEST_SYMBOLS, batch_size=2)
        logger.info(f"\nCreated universe with {len(universe)} symbols")
        logger.info(f"Batch size: 2 symbols per batch")
        
        batch_count = 0
        total_symbols = 0
        
        for batch in universe.batches(desc="Processing batches"):
            batch_count += 1
            total_symbols += len(batch)
            logger.info(f"  Batch {batch_count}: {batch}")
        
        logger.info(f"\n✓ SUCCESS! Processed {batch_count} batches with {total_symbols} total symbols")
        return True
        
    except Exception as e:
        logger.error(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all sanity checks."""
    logger.info("\n" + "="*60)
    logger.info("DATA PIPELINE SANITY CHECK")
    logger.info("Testing core functionality with 5 stocks, 2 years")
    logger.info("="*60)
    
    results = []
    prices_df = None
    features_df = None
    
    # Test 1: Price loading
    success, prices_df = test_price_loading()
    results.append(("Price Data Loading", success))
    
    # Test 2: Feature computation (only if price loading succeeded)
    if success and prices_df is not None:
        success, features_df = test_feature_computation(prices_df)
        results.append(("Feature Computation", success))
    else:
        results.append(("Feature Computation", False))
    
    # Test 3: Batch processing
    success = test_batch_processing()
    results.append(("Batch Processing", success))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SANITY CHECK SUMMARY")
    logger.info("="*60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{name:<30} {status}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    logger.info("\n" + "="*60)
    if passed_count == total_count:
        logger.info(f"✓ ALL TESTS PASSED ({passed_count}/{total_count})")
        logger.info("Core data pipeline is working correctly!")
        logger.info("Ready for full-scale processing")
        logger.info("="*60)
        return 0
    else:
        logger.info(f"✗ SOME TESTS FAILED ({passed_count}/{total_count} passed)")
        logger.info("Review errors above")
        logger.info("="*60)
        return 1

if __name__ == '__main__':
    sys.exit(main())
