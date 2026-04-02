"""MLP-Mixer blocks for tabular data (TabMixer)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixerBlock(nn.Module):
    """MLP-Mixer block for tabular data.
    
    For tabular data with shape (batch, dim), we use feature-wise mixing
    across the feature dimension (like channel-mixing in images).
    
    Architecture:
    Input: (batch, dim)
        ↓
    LayerNorm
        ↓
    Expansion MLP: dim → dim*expansion → dim
        ↓
    Add residual
    """
    
    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        x = x + residual
        return x


class TabMixer(nn.Module):
    """TabMixer: MLP-Mixer encoder for tabular data.
    
    Handles continuous features + categorical embeddings with missing value support.
    Uses multiple mixer blocks with residual connections for feature interaction learning.
    
    Args:
        continuous_dim: Number of continuous features
        categorical_dims: List of categorical cardinalities (e.g., [11, 25])
        embedding_dims: Embedding dimension for each categorical (e.g., [8, 16])
        hidden_dim: Hidden dimension (default 128)
        num_blocks: Number of mixer blocks (default 4)
        output_dim: Output dimension (default 128)
        dropout: Dropout rate (default 0.1)
        handle_missing: Whether to handle NaN values (default True)
    """
    
    def __init__(
        self,
        continuous_dim,
        categorical_dims=None,
        embedding_dims=None,
        hidden_dim=128,
        num_blocks=4,
        output_dim=128,
        dropout=0.1,
        handle_missing=True
    ):
        super().__init__()
        
        categorical_dims = categorical_dims or []
        embedding_dims = embedding_dims or []
        
        # Validate
        if len(categorical_dims) != len(embedding_dims):
            raise ValueError("categorical_dims and embedding_dims must match")
        
        # Categorical embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat, emb_dim)
            for num_cat, emb_dim in zip(categorical_dims, embedding_dims)
        ])
        
        # Input projection
        total_emb_dim = sum(embedding_dims)
        self.input_proj = nn.Linear(continuous_dim + total_emb_dim, hidden_dim)
        
        # Mixer blocks
        self.blocks = nn.ModuleList([
            MixerBlock(dim=hidden_dim, expansion_factor=4, dropout=dropout)
            for _ in range(num_blocks)
        ])
        
        # Output projection
        if hidden_dim != output_dim:
            self.output_proj = nn.Linear(hidden_dim, output_dim)
        else:
            self.output_proj = None
        
        self.norm = nn.LayerNorm(output_dim)
        self.handle_missing = handle_missing
    
    def forward(self, x_continuous, x_categorical=None):
        """
        Args:
            x_continuous: (batch, continuous_dim)
            x_categorical: (batch, num_categorical) - optional
        Returns:
            (batch, output_dim) - L2 normalized
        """
        # Handle NaN values in continuous features
        if self.handle_missing:
            x_continuous = torch.nan_to_num(x_continuous, nan=0.0)
        
        # Embed categorical features
        if len(self.embeddings) > 0:
            if x_categorical is None:
                raise ValueError("x_categorical required when categorical embeddings exist")
            
            embeddings = []
            for i, emb_layer in enumerate(self.embeddings):
                idx = x_categorical[:, i].long()
                emb = emb_layer(idx)
                embeddings.append(emb)
            
            x_cat = torch.cat(embeddings, dim=1)
            x = torch.cat([x_continuous, x_cat], dim=1)
        else:
            x = x_continuous
        
        # Project to hidden dimension
        x = self.input_proj(x)
        
        # Apply mixer blocks
        for block in self.blocks:
            x = block(x)
        
        # Project to output dimension
        if self.output_proj is not None:
            x = self.output_proj(x)
        
        # Final normalization and L2 normalization
        x = self.norm(x)
        x = F.normalize(x, p=2, dim=1)
        
        return x
