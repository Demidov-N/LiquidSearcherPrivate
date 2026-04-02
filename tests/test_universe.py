"""Tests for symbol universe management."""

import pytest
from src.data.universe import SymbolUniverse


class TestSymbolUniverse:
    """Test symbol universe functionality."""
    
    def test_create_batches(self):
        """Test batching of symbol list."""
        symbols = [f"SYM{i:04d}" for i in range(2500)]
        universe = SymbolUniverse(symbols, batch_size=1000)
        
        batches = list(universe.batches())
        assert len(batches) == 3
        assert len(batches[0]) == 1000
        assert len(batches[1]) == 1000
        assert len(batches[2]) == 500
    
    def test_batch_iteration_with_tqdm(self):
        """Test batch iteration includes progress tracking."""
        symbols = [f"SYM{i:04d}" for i in range(100)]
        universe = SymbolUniverse(symbols, batch_size=25)
        
        batches = list(universe.batches())
        assert len(batches) == 4
    
    def test_get_all_symbols(self):
        """Test retrieving all symbols."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        universe = SymbolUniverse(symbols)
        
        assert universe.get_all_symbols() == sorted(symbols)
