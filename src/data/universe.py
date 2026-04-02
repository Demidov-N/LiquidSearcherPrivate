"""Symbol universe management with batch processing."""

from typing import Iterator, List

from tqdm import tqdm


class SymbolUniverse:
    """Manage a universe of stock symbols with batch processing.
    
    Provides efficient iteration through large symbol lists with
    progress tracking via tqdm.
    
    Attributes:
        symbols: List of all symbols in the universe
        batch_size: Number of symbols per batch
        total_symbols: Total count of symbols
    """
    
    def __init__(self, symbols: List[str], batch_size: int = 750):
        """Initialize symbol universe.
        
        Args:
            symbols: List of stock symbols
            batch_size: Number of symbols per batch (default 750 for 30GB RAM)
        """
        self.symbols = sorted(set(symbols))  # Remove duplicates, sort
        self.batch_size = batch_size
        self.total_symbols = len(self.symbols)
    
    def batches(self, desc: str = "Processing symbols") -> Iterator[List[str]]:
        """Iterate through symbols in batches with progress bar.
        
        Args:
            desc: Description for progress bar
            
        Yields:
            List of symbols for each batch
        """
        num_batches = (self.total_symbols + self.batch_size - 1) // self.batch_size
        
        with tqdm(total=num_batches, desc=desc, unit="batch") as pbar:
            for i in range(0, self.total_symbols, self.batch_size):
                batch = self.symbols[i:i + self.batch_size]
                yield batch
                pbar.update(1)
    
    def get_all_symbols(self) -> List[str]:
        """Get all symbols in the universe."""
        return self.symbols
    
    def __len__(self) -> int:
        """Return total number of symbols."""
        return self.total_symbols
