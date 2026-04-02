"""Tests for DualEncoder LightningModule."""

import pytest
import torch
import pytorch_lightning as pl
from src.training.module import DualEncoderModule


def test_module_initialization():
    """Test module can be initialized."""
    module = DualEncoderModule(
        temporal_input_dim=13,
        tabular_continuous_dim=15,
        tabular_categorical_dims=[11, 25],
        embedding_dim=128,
        temperature=0.07,
        lr=1e-4,
    )
    assert module is not None
    assert module.model is not None


def test_training_step():
    """Test training step produces loss."""
    module = DualEncoderModule(
        temporal_input_dim=13,
        tabular_continuous_dim=15,
        embedding_dim=128,
    )

    batch = {
        "temporal": torch.randn(16, 60, 13),
        "tabular_cont": torch.randn(16, 15),
        "tabular_cat": torch.randint(0, 5, (16, 2)),
    }

    loss = module.training_step(batch, batch_idx=0)
    assert loss is not None
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_validation_step():
    """Test validation step logs metrics."""
    module = DualEncoderModule(
        temporal_input_dim=13,
        tabular_continuous_dim=15,
        embedding_dim=128,
    )

    batch = {
        "temporal": torch.randn(16, 60, 13),
        "tabular_cont": torch.randn(16, 15),
        "tabular_cat": torch.randint(0, 5, (16, 2)),
    }

    loss = module.validation_step(batch, batch_idx=0)
    assert loss is not None
