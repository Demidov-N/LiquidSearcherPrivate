"""Temporal Convolutional Network with causal convolutions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Causal 1D convolution (no look-ahead).
    
    Output at time t only depends on inputs [0, t], not [t+1, ...].
    """
    
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=0, dilation=dilation)
    
    def forward(self, x):
        # Pad left side only (causal)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TemporalConvNet(nn.Module):
    """TCN with multi-scale dilated convolutions.
    
    Uses exponentially increasing dilations to capture patterns at multiple 
    timescales without increasing kernel size.
    
    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension for layers
        dilations: List of dilations (default [1, 2, 4, 8])
        kernel_size: Kernel size (default 3)
        dropout: Dropout rate (default 0.1)
    """
    
    def __init__(self, input_dim, hidden_dim=64, dilations=None, kernel_size=3, dropout=0.1):
        super().__init__()
        
        if dilations is None:
            dilations = [1, 2, 4, 8]
        
        layers = []
        for i, d in enumerate(dilations):
            in_ch = input_dim if i == 0 else hidden_dim
            layers.extend([
                CausalConv1d(in_ch, hidden_dim, kernel_size, d),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (batch, seq, features) -> (batch, features, seq)
        x = x.transpose(1, 2)
        x = self.network(x)
        # Back to (batch, seq, features)
        x = x.transpose(1, 2)
        return x
