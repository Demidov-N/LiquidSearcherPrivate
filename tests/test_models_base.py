"""Tests for base encoder class."""

import pytest
import torch
from src.models.base import BaseEncoder


def test_base_encoder_is_abstract():
    """Test that BaseEncoder cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseEncoder(input_dim=10, output_dim=128)


def test_valid_subclass_works():
    """Test that valid subclass with forward method works."""

    class ValidEncoder(BaseEncoder):
        def __init__(self):
            super().__init__(input_dim=10, output_dim=128)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.randn(x.shape[0], self.output_dim)

    encoder = ValidEncoder()
    x = torch.randn(32, 10)
    output = encoder(x)
    assert output.shape == (32, 128)
