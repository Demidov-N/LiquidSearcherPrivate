"""Tests for TCN module."""

import torch
from src.models.tcn import TemporalConvNet


def test_tcn_output_shape():
    """Test TCN produces correct output shape."""
    tcn = TemporalConvNet(input_dim=13, hidden_dim=64, dilations=[1, 2, 4, 8])
    x = torch.randn(32, 60, 13)  # (batch, seq, features)
    out = tcn(x)
    assert out.shape == (32, 60, 64)


def test_tcn_causal():
    """Test no look-ahead (causal convolution)."""
    tcn = TemporalConvNet(input_dim=1, hidden_dim=16, dilations=[1])
    tcn.eval()
    
    # Input with impulse at position 5
    x = torch.zeros(1, 10, 1)
    x[0, 5, 0] = 1.0
    
    out = tcn(x)
    # Output at position 4 should be zero (no look-ahead)
    assert out[0, 4, 0].item() == 0.0
