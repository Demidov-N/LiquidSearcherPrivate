"""Tests for tabular encoder."""

import torch
from src.models.tabular_encoder import TabularEncoder


def test_basic_forward():
    """Test basic forward pass."""
    encoder = TabularEncoder()
    
    x_cont = torch.randn(32, 15)
    x_cat = torch.randint(0, 5, (32, 2))
    
    out = encoder(x_continuous=x_cont, x_categorical=x_cat)
    assert out.shape == (32, 128)


def test_gics_embeddings():
    """Test GICS sector/group embeddings with actual codes."""
    encoder = TabularEncoder()
    
    # Test actual GICS sector/group indices (not codes)
    # gsector has 11 classes (0-10), ggroup has 25 classes (0-24)
    gsector = torch.tensor([0, 2, 5, 8, 10])  # Various sectors
    ggroup = torch.tensor([0, 5, 10, 15, 24])  # Various groups
    x_cat = torch.stack([gsector, ggroup], dim=1)
    
    x_cont = torch.randn(5, 15)
    out = encoder(x_continuous=x_cont, x_categorical=x_cat)
    
    assert out.shape == (5, 128)
    assert not torch.isnan(out).any()


def test_inherits_base():
    """Test inheritance from BaseEncoder."""
    from src.models.base import BaseEncoder
    encoder = TabularEncoder()
    assert isinstance(encoder, BaseEncoder)
    assert encoder.input_dim == 39  # 15 + 8 + 16
    assert encoder.output_dim == 128


def test_different_batch_sizes():
    """Test with various batch sizes."""
    encoder = TabularEncoder()
    
    for batch_size in [1, 8, 32, 256]:
        x_cont = torch.randn(batch_size, 15)
        x_cat = torch.randint(0, 11, (batch_size, 2))
        out = encoder(x_continuous=x_cont, x_categorical=x_cat)
        assert out.shape == (batch_size, 128)


def test_output_normalized():
    """Test L2 normalization."""
    encoder = TabularEncoder(dropout=0.0)
    encoder.eval()
    
    x_cont = torch.randn(16, 15)
    x_cat = torch.randint(0, 11, (16, 2))
    out = encoder(x_continuous=x_cont, x_categorical=x_cat)
    
    norms = torch.norm(out, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
