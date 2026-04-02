"""Tests for WRDS data loader."""

import pytest
from unittest.mock import Mock, patch
import pandas as pd

from src.data.wrds_loader import WRDSDataLoader


class TestWRDSDataLoader:
    """Test WRDS data loader functionality."""
    
    @patch('src.data.wrds_loader.wrds.Connection')
    def test_initialization(self, mock_conn):
        """Test loader initialization."""
        loader = WRDSDataLoader()
        assert loader is not None
    
    @patch('src.data.wrds_loader.wrds.Connection')
    def test_fetch_prices_batch(self, mock_conn_class):
        """Test fetching price data for a symbol batch."""
        # Mock the connection and raw_sql method
        mock_conn = Mock()
        mock_conn_class.return_value = mock_conn
        
        # Mock return data
        mock_data = pd.DataFrame({
            'permno': [1, 1, 2, 2],
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02']),
            'prc': [100.0, 101.0, 50.0, 51.0],
            'vol': [1000000, 1100000, 500000, 550000],
            'ret': [0.0, 0.01, 0.0, 0.02]
        })
        mock_conn.raw_sql.return_value = mock_data
        
        loader = WRDSDataLoader()
        result = loader.fetch_prices_batch(['AAPL', 'MSFT'], '2023-01-01', '2023-01-31')
        
        assert isinstance(result, pd.DataFrame)
        assert 'permno' in result.columns
        assert 'date' in result.columns
