"""Tabular encoder using TabMixer with GICS configuration."""

from src.models.base import BaseEncoder
from src.models.mixer import TabMixer


class TabularEncoder(BaseEncoder):
    """Tabular encoder for fundamentals + GICS embeddings.
    
    Default configuration:
    - Continuous: 15 features (risk 4 + momentum 5 + valuation 4 + misc)
    - Categorical: gsector (11→8-dim), ggroup (25→16-dim)
    - Total input: 15 + 8 + 16 = 39 dimensions
    - Hidden: 128 dimensions  
    - Output: 128 dimensions (L2 normalized)
    
    Args:
        continuous_dim: Number of continuous features (default 15)
        categorical_dims: List of cardinalities (default [11, 25])
        embedding_dims: Embedding dimensions (default [8, 16])
        hidden_dim: Hidden dimension (default 128)
        num_blocks: Number of mixer blocks (default 4)
        output_dim: Output dimension (default 128)
        dropout: Dropout rate (default 0.1)
    
    Example:
        encoder = TabularEncoder()
        
        # Continuous features
        fundamentals = torch.randn(32, 15)
        
        # Categorical: gsector (0-10), ggroup (0-24)
        sectors = torch.randint(0, 11, (32,))
        groups = torch.randint(0, 25, (32,))
        categorical = torch.stack([sectors, groups], dim=1)
        
        embeddings = encoder(fundamentals, categorical)  # (32, 128)
    """
    
    def __init__(
        self,
        continuous_dim=15,
        categorical_dims=None,
        embedding_dims=None,
        hidden_dim=128,
        num_blocks=4,
        output_dim=128,
        dropout=0.1
    ):
        if categorical_dims is None:
            categorical_dims = [11, 25]  # gsector, ggroup
        if embedding_dims is None:
            embedding_dims = [8, 16]
        
        total_emb_dim = sum(embedding_dims)
        total_input_dim = continuous_dim + total_emb_dim
        
        super().__init__(input_dim=total_input_dim, output_dim=output_dim)
        
        self.continuous_dim = continuous_dim
        self.categorical_dims = categorical_dims
        self.embedding_dims = embedding_dims
        
        # Build TabMixer
        self.mixer = TabMixer(
            continuous_dim=continuous_dim,
            categorical_dims=categorical_dims,
            embedding_dims=embedding_dims,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            output_dim=output_dim,
            dropout=dropout,
            handle_missing=True
        )
    
    def forward(self, x_continuous, x_categorical):
        """
        Args:
            x_continuous: (batch, 15) - continuous features
            x_categorical: (batch, 2) - [gsector, ggroup] indices
        Returns:
            (batch, 128) - L2 normalized tabular embedding
        """
        return self.mixer(x_continuous, x_categorical)
