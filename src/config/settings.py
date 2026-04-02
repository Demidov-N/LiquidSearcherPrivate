"""Configuration settings for data collection and feature processing."""

from pathlib import Path
from typing import Optional


class Settings:
    """Application settings with sensible defaults."""
    
    # Data directories
    data_dir: Path = Path("data")
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    cache_dir: Path = data_dir / "cache"
    
    # Batch processing
    batch_size: int = 750  # Symbols per batch (500-1000 range)
    
    # WRDS settings
    wrds_username: Optional[str] = None
    wrds_password: Optional[str] = None
    
    # Feature computation
    use_precomputed_betas: bool = True  # Use WRDS Beta Suite
    beta_lookback: int = 60  # Days for beta calculation
    vol_lookback: int = 20  # Days for volatility
    
    # Date ranges
    start_date: str = "2010-01-01"
    end_date: str = "2024-12-31"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()
