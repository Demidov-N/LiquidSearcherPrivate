"""Tests for BiMT-TCN temporal encoder."""

import torch
from src.models.temporal_encoder import TemporalEncoder


def test_output_shape():
    """Test output is correct shape and normalized."""
    encoder = TemporalEncoder(input_dim=13, output_dim=128)
    x = torch.randn(32, 60, 13)
    out = encoder(x)
    assert out.shape == (32, 128)


def test_l2_normalized():
    """Test that output is L2 normalized."""
    encoder = TemporalEncoder(input_dim=13, output_dim=128, dropout=0.0)
    encoder.eval()
    x = torch.randn(16, 60, 13)
    out = encoder(x)
    
    norms = torch.norm(out, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_no_nan():
    """Test no NaN in output."""
    encoder = TemporalEncoder(input_dim=13, output_dim=128)
    x = torch.randn(8, 60, 13)
    out = encoder(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_different_sequence_lengths():
    """Test with different sequence lengths."""
    encoder = TemporalEncoder(input_dim=13, output_dim=128)
    
    for seq_len in [30, 60, 100]:
        x = torch.randn(4, seq_len, 13)
        out = encoder(x)
        assert out.shape == (4, 128)


def test_inherits_base():
    """Test that it inherits from BaseEncoder."""
    from src.models.base import BaseEncoder
    encoder = TemporalEncoder()
    assert isinstance(encoder, BaseEncoder)
    assert encoder.output_dim == 128
    assert encoder.input_dim == 13
