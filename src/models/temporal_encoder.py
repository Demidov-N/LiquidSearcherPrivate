"""BiMT-TCN temporal encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseEncoder
from src.models.tcn import TemporalConvNet
from src.models.positional_encoding import PositionalEncoding


class TemporalEncoder(BaseEncoder):
    """BiMT-TCN: TCN + Transformer for temporal patterns.
    
    Processes 60-day OHLCV windows to produce 128-dim temporal embeddings.
    
    Architecture:
    Input (60, 13)
        ↓
    TCN (dilations 1,2,4,8) → (60, 64)
        ↓
    Positional Encoding → (60, 64)
        ↓
    Transformer (2 layers, 4 heads) → (60, 64)
        ↓
    Global Average Pooling → (64,)
        ↓
    Projection (64→128) + L2 Norm → (128,)
    
    Args:
        input_dim: Input features (default 13 for OHLCV)
        hidden_dim: Hidden dimension (default 64)
        transformer_layers: Transformer depth (default 2)
        transformer_heads: Attention heads (default 4)
        output_dim: Output dimension (default 128)
        dropout: Dropout rate (default 0.1)
    """
    
    def __init__(
        self,
        input_dim=13,
        hidden_dim=64,
        transformer_layers=2,
        transformer_heads=4,
        output_dim=128,
        dropout=0.1
    ):
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        
        self.hidden_dim = hidden_dim
        
        # TCN for local patterns
        self.tcn = TemporalConvNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dilations=[1, 2, 4, 8],
            dropout=dropout
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=hidden_dim,
            max_len=500,
            dropout=dropout
        )
        
        # Transformer for global dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )
        
        # Projection and normalization
        self.projection = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        """Forward: prices → temporal embedding.
        
        Args:
            x: (batch, 60, 13)
        Returns:
            (batch, 128) - L2 normalized
        """
        # TCN
        x = self.tcn(x)                    # (batch, 60, 64)
        
        # Positional encoding
        x = self.pos_encoder(x)          # (batch, 60, 64)
        
        # Transformer
        x = self.transformer(x)          # (batch, 60, 64)
        
        # Global average pooling
        x = x.mean(dim=1)                # (batch, 64)
        
        # Projection
        x = self.projection(x)           # (batch, 128)
        x = self.norm(x)                 # (batch, 128)
        
        # L2 normalize (critical for dot product)
        x = F.normalize(x, p=2, dim=1)
        
        return x
