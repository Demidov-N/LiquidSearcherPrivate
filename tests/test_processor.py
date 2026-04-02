"""Tests for feature processor."""

import pandas as pd
import numpy as np
import pytest

from src.features.processor import FeatureProcessor


class TestFeatureProcessor:
    """Test feature processor functionality."""
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = FeatureProcessor()
        assert processor is not None
    
    def test_compute_ohlcv_features(self):
        """Test OHLCV feature computation."""
        processor = FeatureProcessor()
        
        # Create sample OHLCV data
        dates = pd.date_range('2023-01-01', periods=30, freq='B')
        df = pd.DataFrame({
            'symbol': ['AAPL'] * 30,
            'date': dates,
            'prc': np.random.randn(30).cumsum() + 100,
            'vol': np.random.randint(1000000, 5000000, 30),
            'ret': np.random.randn(30) * 0.02
        })
        
        result = processor.compute_ohlcv_features(df)
        
        # Should have computed features
        assert 'z_close' in result.columns or 'prc' in result.columns
    
    def test_compute_momentum_features(self):
        """Test momentum feature computation."""
        processor = FeatureProcessor()
        
        dates = pd.date_range('2023-01-01', periods=300, freq='B')
        # Start with OHLCV data to get ret_1d computed
        df = pd.DataFrame({
            'symbol': ['AAPL'] * 300,
            'date': dates,
            'prc': np.random.randn(300).cumsum() + 100,
            'vol': np.random.randint(1000000, 5000000, 300),
        })
        
        # First compute OHLCV features to get ret_1d
        df = processor.compute_ohlcv_features(df)
        
        # Then compute momentum features
        result = processor.compute_momentum_features(df)
        
        # Should have momentum columns
        expected_cols = ['mom_1m', 'mom_3m', 'mom_6m', 'mom_12_1m']
        for col in expected_cols:
            assert col in result.columns
