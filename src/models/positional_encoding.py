"""Sinusoidal positional encoding for transformers."""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017).
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        d_model: Model dimension (must match input)
        max_len: Maximum sequence length to precompute
        dropout: Dropout rate
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Precompute positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        
        # Position indices: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Divisor term: 1 / 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not parameter, persisted in state_dict)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input.
        
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
