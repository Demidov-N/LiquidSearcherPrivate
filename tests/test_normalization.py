"""Tests for feature normalization."""

import pandas as pd
import numpy as np
import pytest

from src.features.normalization import (
    winsorize,
    cross_sectional_zscore,
    rank_normalize,
    two_pass_normalization
)


class TestNormalization:
    """Test normalization functions."""
    
    def test_winsorize(self):
        """Test winsorization."""
        data = pd.Series([1, 2, 3, 4, 5, 100, -100])
        result = winsorize(data, lower=0.01, upper=0.99)
        
        # Extreme values should be clipped
        assert result.max() < 100
        assert result.min() > -100
    
    def test_cross_sectional_zscore(self):
        """Test cross-sectional z-score normalization."""
        df = pd.DataFrame({
            'date': ['2023-01-01'] * 5,
            'value': [1, 2, 3, 4, 5]
        })
        
        result = cross_sectional_zscore(df, 'value', 'date')
        
        # Should have mean ~0 and std ~1 per date
        assert abs(result.mean()) < 0.01
        assert abs(result.std() - 1.0) < 0.01
    
    def test_rank_normalize(self):
        """Test rank normalization."""
        data = pd.Series([1, 5, 3, 2, 4])
        result = rank_normalize(data)
        
        # Should be in [0, 1]
        assert result.min() >= 0
        assert result.max() <= 1
        assert len(result) == len(data)
    
    def test_two_pass_normalization(self):
        """Test two-pass normalization pipeline."""
        df = pd.DataFrame({
            'symbol': ['A', 'A', 'B', 'B'],
            'date': ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02'],
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [10.0, 20.0, 30.0, 40.0]
        })
        
        result = two_pass_normalization(
            df,
            feature_cols=['feature1', 'feature2'],
            date_col='date'
        )
        
        assert 'feature1_zscore' in result.columns
        assert 'feature2_zscore' in result.columns
