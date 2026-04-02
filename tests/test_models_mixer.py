"""Tests for TabMixer."""

import torch
from src.models.mixer import MixerBlock, TabMixer


def test_mixer_block_shape():
    """Test mixer block preserves shape."""
    block = MixerBlock(dim=128)
    x = torch.randn(32, 128)
    out = block(x)
    assert out.shape == x.shape


def test_mixer_block_transforms():
    """Test that output differs from input (transformation happens)."""
    block = MixerBlock(dim=128, expansion_factor=4, dropout=0.0)
    block.eval()
    x = torch.randn(16, 128)
    out = block(x)
    assert not torch.allclose(out, x, atol=1e-4)


def test_tabmixer_continuous_only():
    """Test with continuous features only."""
    mixer = TabMixer(continuous_dim=15, hidden_dim=128, num_blocks=4, output_dim=128)
    x = torch.randn(32, 15)
    out = mixer(x)
    assert out.shape == (32, 128)


def test_tabmixer_with_categorical():
    """Test with categorical embeddings."""
    mixer = TabMixer(
        continuous_dim=15,
        categorical_dims=[11, 25],
        embedding_dims=[8, 16],
        hidden_dim=128,
        num_blocks=4,
        output_dim=128
    )
    
    x_cont = torch.randn(32, 15)
    x_cat = torch.randint(0, 5, (32, 2))
    out = mixer(x_cont, x_cat)
    assert out.shape == (32, 128)


def test_l2_normalized():
    """Test L2 normalization."""
    mixer = TabMixer(continuous_dim=10, hidden_dim=64, num_blocks=2, output_dim=64, dropout=0.0)
    mixer.eval()
    x = torch.randn(16, 10)
    out = mixer(x)
    
    norms = torch.norm(out, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_missing_values():
    """Test handling of NaN values."""
    mixer = TabMixer(continuous_dim=15, hidden_dim=64, num_blocks=2, output_dim=64)
    
    x = torch.randn(8, 15)
    x[0, 3] = float('nan')  # Add missing value
    
    out = mixer(x)
    assert not torch.isnan(out).any()
