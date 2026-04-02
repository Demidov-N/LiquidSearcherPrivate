"""End-to-end tests for training pipeline."""

import pytest
import torch
import pytorch_lightning as pl

from src.training.data_module import StockDataModule
from src.training.module import DualEncoderModule


def test_data_module_setup():
    """Test DataModule can be set up with mock data."""
    dm = StockDataModule(
        feature_dir="data/processed/features",
        train_start="2010-01-01",
        train_end="2018-12-31",
        val_start="2020-01-01",
        val_end="2020-12-31",
        symbols=["AAPL", "MSFT"],
        batch_size=4,
        num_workers=0,
    )
    dm.setup("fit")
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert len(dm.train_dataset) > 0
    assert len(dm.val_dataset) > 0


def test_training_step_with_real_data():
    """Test training step with actual data from dataloader."""
    dm = StockDataModule(
        feature_dir="data/processed/features",
        train_start="2010-01-01",
        train_end="2018-12-31",
        val_start="2020-01-01",
        val_end="2020-12-31",
        symbols=["AAPL", "MSFT"],
        batch_size=4,
        num_workers=0,
    )
    dm.setup("fit")

    model = DualEncoderModule(
        temporal_input_dim=13,
        tabular_continuous_dim=15,
        embedding_dim=128,
    )

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    loss = model.training_step(batch, batch_idx=0)
    assert loss is not None
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_one_epoch_training():
    """Test full epoch training with Lightning Trainer."""
    pl.seed_everything(42)

    dm = StockDataModule(
        feature_dir="data/processed/features",
        train_start="2010-01-01",
        train_end="2018-12-31",
        val_start="2020-01-01",
        val_end="2020-12-31",
        symbols=["AAPL", "MSFT"],
        batch_size=4,
        num_workers=0,
    )

    model = DualEncoderModule(
        temporal_input_dim=13,
        tabular_continuous_dim=15,
        embedding_dim=128,
        lr=1e-4,
    )

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        enable_checkpointing=False,
        logger=False,
    )

    trainer.fit(model, dm)

    assert trainer.current_epoch >= 0
    assert trainer.global_step > 0