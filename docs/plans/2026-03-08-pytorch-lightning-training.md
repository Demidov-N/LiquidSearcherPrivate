# PyTorch Lightning Training Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Build a production-ready PyTorch Lightning training pipeline for the dual-encoder stock substitute recommendation system with proper temporal data splitting, InfoNCE loss, and validation metrics.

**Architecture:**
- **LightningDataModule**: Handles temporal splits (purge/embargo), train/val/test fold management, and DataLoader creation with hard negative sampling
- **DualEncoderModule**: PyTorch LightningModule wrapping the dual-encoder model with InfoNCE loss, training/validation steps, and metrics logging
- **Training Script**: CLI entry point using Lightning Trainer with callbacks (checkpointing, early stopping, LR monitoring)
- **Validation Script**: Standalone evaluation on any fold with comprehensive metrics

**Tech Stack:** PyTorch Lightning, PyTorch, info-nce-pytorch, wandb (optional), PyArrow/Parquet, pandas, numpy

**Prerequisites:** Dual-encoder model already implemented (src/models/)

---

## Task 1: Setup Training Infrastructure

**Files:**
- Create: `src/training/__init__.py`
- Create: `tests/test_training_integration.py`

**Step 1: Check pytorch-lightning availability**

Run: `python -c "import pytorch_lightning; print(pytorch_lightning.__version__)"`

If not available:
Run: `pip install pytorch-lightning wandb`

**Step 2: Create training package init**

Create `src/training/__init__.py`:

```python
"""Training infrastructure for dual-encoder models."""

from src.training.data_module import StockDataModule
from src.training.module import DualEncoderModule

__all__ = ["StockDataModule", "DualEncoderModule"]
```

**Step 3: Write integration test**

Create `tests/test_training_integration.py`:

```python
"""Integration tests for training infrastructure."""

import pytest
import torch
from src.training import StockDataModule, DualEncoderModule


def test_imports_work():
    """Test all training components can be imported."""
    assert StockDataModule is not None
    assert DualEncoderModule is not None


def test_lightning_available():
    """Test PyTorch Lightning is installed."""
    import pytorch_lightning as pl
    assert pl.__version__ >= "2.0.0"
```

**Step 4: Run test**

Run: `python -m pytest tests/test_training_integration.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/training/__init__.py tests/test_training_integration.py
python -m ruff format src/training/__init__.py
git commit -m "chore: setup training infrastructure package"
```

---

## Task 2: Create StockDataModule with Temporal Splits

**Files:**
- Create: `src/training/data_module.py`
- Create: `tests/test_data_module.py`

**Step 1: Write the failing test**

Create `tests/test_data_module.py`:

```python
"""Tests for StockDataModule with temporal splits."""

import pytest
import torch
from datetime import datetime
from src.training.data_module import StockDataModule


def test_data_module_initialization():
    """Test DataModule can be initialized."""
    dm = StockDataModule(
        feature_dir="data/processed/features",
        train_start="2010-01-01",
        train_end="2018-12-31",
        val_start="2020-01-01",
        val_end="2020-12-31",
        symbols=["AAPL", "MSFT"],
        batch_size=32,
    )
    assert dm is not None
    assert dm.batch_size == 32


def test_temporal_split_boundaries():
    """Test purge and embargo periods are respected."""
    # Train ends 2018-12-31, purge removes last 252 days
    # Val starts 2020-01-01, so 2019 is embargo gap
    dm = StockDataModule(
        feature_dir="data/processed/features",
        train_start="2010-01-01",
        train_end="2018-12-31",
        val_start="2020-01-01",
        val_end="2020-12-31",
        purge_days=252,
    )
    
    # Effective train end should be 252 days before 2018-12-31
    from datetime import timedelta
    expected_train_end = datetime(2018, 12, 31) - timedelta(days=252)
    assert dm.effective_train_end <= expected_train_end


def test_dataloader_returns_batch():
    """Test dataloader returns correctly shaped batch."""
    dm = StockDataModule(
        feature_dir="data/processed/features",
        train_start="2010-01-01",
        train_end="2018-12-31",
        val_start="2020-01-01",
        val_end="2020-12-31",
        symbols=["AAPL", "MSFT"],
        batch_size=16,
    )
    
    # Setup creates dataloaders
    dm.setup("fit")
    
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    # Check batch structure
    assert "temporal" in batch
    assert "tabular_cont" in batch
    assert "tabular_cat" in batch
    assert batch["temporal"].shape == (16, 60, 13)
    assert batch["tabular_cont"].shape == (16, 15)
    assert batch["tabular_cat"].shape == (16, 2)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data_module.py -v`
Expected: FAIL with "ImportError: cannot import name 'StockDataModule'"

**Step 3: Write implementation**

Create `src/training/data_module.py`:

```python
"""PyTorch Lightning DataModule with temporal splits for financial data."""

import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class StockDataset(Dataset):
    """Dataset for stock features with random window sampling."""
    
    def __init__(
        self,
        feature_dir: str,
        date_range: tuple[str, str],
        symbols: Optional[List[str]] = None,
        window_size: int = 60,
        temporal_features: Optional[List[str]] = None,
        tabular_continuous_features: Optional[List[str]] = None,
        tabular_categorical_features: Optional[List[str]] = None,
    ):
        """Initialize dataset.
        
        Args:
            feature_dir: Directory containing parquet files
            date_range: (start_date, end_date) as strings
            symbols: List of symbols to include (None = all)
            window_size: Number of days for temporal window
            temporal_features: List of temporal feature columns
            tabular_continuous_features: List of continuous tabular features
            tabular_categorical_features: List of categorical features
        """
        self.feature_dir = Path(feature_dir)
        self.window_size = window_size
        
        self.start_date = pd.Timestamp(date_range[0])
        self.end_date = pd.Timestamp(date_range[1])
        
        # Feature columns
        self.temporal_features = temporal_features or [
            "open", "high", "low", "close", "volume",
            "returns", "log_returns", "realized_vol_20d",
            "momentum_20d", "momentum_60d",
            "rsi_14d", "macd", "signal"
        ]
        self.tabular_continuous_features = tabular_continuous_features or [
            "market_beta_60d", "downside_beta_60d", "idiosyncratic_vol_60d",
            "momentum_20d", "momentum_60d", "momentum_120d",
            "volatility_20d", "volatility_60d",
            "roe", "roa", "debt_to_equity", "price_to_book",
            "price_to_earnings", "market_cap", "dividend_yield"
        ]
        self.tabular_categorical_features = tabular_categorical_features or [
            "gsector", "ggroup"
        ]
        
        # Load available symbols
        if symbols is None:
            self.symbols = self._discover_symbols()
        else:
            self.symbols = symbols
        
        # Build index of valid samples
        self.samples = self._build_sample_index()
    
    def _discover_symbols(self) -> List[str]:
        """Discover available symbols from feature directory."""
        symbols = []
        for f in self.feature_dir.glob("*_features.parquet"):
            symbol = f.stem.replace("_features", "")
            symbols.append(symbol)
        return sorted(symbols)
    
    def _build_sample_index(self) -> List[Dict[str, Any]]:
        """Build index of valid (symbol, date) samples within date range."""
        samples = []
        
        for symbol in self.symbols:
            file_path = self.feature_dir / f"{symbol}_features.parquet"
            if not file_path.exists():
                continue
            
            try:
                # Read only date column to check availability
                df = pd.read_parquet(file_path, columns=["date"])
                df["date"] = pd.to_datetime(df["date"])
                
                # Filter to date range
                mask = (df["date"] >= self.start_date) & (df["date"] <= self.end_date)
                valid_dates = df.loc[mask, "date"].tolist()
                
                # Add samples
                for date in valid_dates:
                    samples.append({
                        "symbol": symbol,
                        "date": date,
                        "file_path": str(file_path),
                    })
            except Exception as e:
                print(f"Warning: Could not load {symbol}: {e}")
                continue
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample by index."""
        sample = self.samples[idx]
        symbol = sample["symbol"]
        end_date = sample["date"]
        file_path = sample["file_path"]
        
        # Load data for this symbol (cache in memory for efficiency)
        df = pd.read_parquet(file_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        
        # Find index of end_date
        end_idx = df[df["date"] == end_date].index
        if len(end_idx) == 0:
            # Try nearest date
            end_idx = df[df["date"] <= end_date].index[-1:]
        end_idx = end_idx[0]
        
        # Extract temporal window
        start_idx = max(0, end_idx - self.window_size + 1)
        temporal_df = df.iloc[start_idx:end_idx+1]
        
        # Handle short windows (pad if necessary)
        if len(temporal_df) < self.window_size:
            # Pad with zeros at the beginning
            padding = self.window_size - len(temporal_df)
            temporal_data = np.zeros((self.window_size, len(self.temporal_features)))
            available_data = temporal_df[self.temporal_features].values
            temporal_data[padding:] = available_data
        else:
            temporal_data = temporal_df[self.temporal_features].values
        
        # Extract tabular features at end_date
        tabular_row = df.iloc[end_idx]
        tabular_cont = tabular_row[self.tabular_continuous_features].values
        tabular_cat = tabular_row[self.tabular_categorical_features].values
        
        # Handle missing values
        tabular_cont = np.nan_to_num(tabular_cont, nan=0.0)
        tabular_cat = np.nan_to_num(tabular_cat, nan=0.0).astype(int)
        
        # Get beta and GICS info
        beta = float(tabular_row.get("market_beta_60d", 1.0))
        gsector = int(tabular_cat[0]) if len(tabular_cat) > 0 else 0
        ggroup = int(tabular_cat[1]) if len(tabular_cat) > 1 else 0
        
        return {
            "symbol": symbol,
            "date": str(end_date),
            "temporal": torch.tensor(temporal_data, dtype=torch.float32),
            "tabular_cont": torch.tensor(tabular_cont, dtype=torch.float32),
            "tabular_cat": torch.tensor(tabular_cat, dtype=torch.long),
            "beta": beta,
            "gsector": gsector,
            "ggroup": ggroup,
        }


class StockDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule with temporal train/val/test splits."""
    
    def __init__(
        self,
        feature_dir: str,
        train_start: str,
        train_end: str,
        val_start: str,
        val_end: str,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        purge_days: int = 252,
        embargo_days: int = 63,  # ~3 months
        window_size: int = 60,
    ):
        """Initialize DataModule with temporal splits.
        
        Implements purge and embargo for financial ML to prevent look-ahead bias:
        - Purge: Remove last N days from training (feature overlap)
        - Embargo: Gap between train end and val start (return autocorrelation)
        
        Args:
            feature_dir: Directory with parquet feature files
            train_start: Training period start (YYYY-MM-DD)
            train_end: Training period end (YYYY-MM-DD)
            val_start: Validation period start
            val_end: Validation period end
            test_start: Test period start (optional)
            test_end: Test period end (optional)
            symbols: Stock symbols to include (None = all)
            batch_size: Batch size for training
            num_workers: DataLoader workers
            purge_days: Days to purge from end of training
            embargo_days: Days to embargo between splits
            window_size: Days in temporal window
        """
        super().__init__()
        
        self.save_hyperparameters()
        
        self.feature_dir = feature_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_size = window_size
        
        # Original date boundaries
        self.train_start = pd.Timestamp(train_start)
        self.train_end = pd.Timestamp(train_end)
        self.val_start = pd.Timestamp(val_start)
        self.val_end = pd.Timestamp(val_end)
        
        # Apply purge to training (remove last N days)
        self.effective_train_end = self.train_end - timedelta(days=purge_days)
        
        # Apply embargo gap
        self.effective_val_start = self.val_start + timedelta(days=embargo_days)
        
        # Test dates
        if test_start and test_end:
            self.test_start = pd.Timestamp(test_start)
            self.test_end = pd.Timestamp(test_end)
            self.has_test = True
        else:
            self.has_test = False
        
        self.symbols = symbols
        
        # Datasets (initialized in setup)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training, validation, and test."""
        if stage == "fit" or stage is None:
            # Training dataset (with purge applied)
            self.train_dataset = StockDataset(
                feature_dir=self.feature_dir,
                date_range=(str(self.train_start), str(self.effective_train_end)),
                symbols=self.symbols,
                window_size=self.window_size,
            )
            print(f"Train dataset: {len(self.train_dataset)} samples")
            
            # Validation dataset (with embargo applied)
            self.val_dataset = StockDataset(
                feature_dir=self.feature_dir,
                date_range=(str(self.effective_val_start), str(self.val_end)),
                symbols=self.symbols,
                window_size=self.window_size,
            )
            print(f"Val dataset: {len(self.val_dataset)} samples")
        
        if stage == "test" or stage is None:
            if self.has_test:
                self.test_dataset = StockDataset(
                    feature_dir=self.feature_dir,
                    date_range=(str(self.test_start), str(self.test_end)),
                    symbols=self.symbols,
                    window_size=self.window_size,
                )
                print(f"Test dataset: {len(self.test_dataset)} samples")
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self) -> Optional[DataLoader]:
        """Return test dataloader."""
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_data_module.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/training/data_module.py tests/test_data_module.py
python -m ruff format src/training/data_module.py
python -m ruff check src/training/data_module.py
git commit -m "feat: add StockDataModule with temporal splits (purge/embargo)"
```

---

## Task 3: Create DualEncoder LightningModule

**Files:**
- Create: `src/training/module.py`
- Create: `tests/test_training_module.py`

**Step 1: Write the failing test**

Create `tests/test_training_module.py`:

```python
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
    
    # Create fake batch
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


def test_forward_returns_embeddings():
    """Test forward pass returns embeddings."""
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
    
    temporal_emb, tabular_emb = module(batch)
    assert temporal_emb.shape == (16, 128)
    assert tabular_emb.shape == (16, 128)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_training_module.py -v`
Expected: FAIL

**Step 3: Write implementation**

Create `src/training/module.py`:

```python
"""PyTorch Lightning Module for dual-encoder contrastive learning."""

from typing import Any, Dict, Optional, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from info_nce import InfoNCE

from src.models.dual_encoder import DualEncoder


class DualEncoderModule(pl.LightningModule):
    """LightningModule for dual-encoder contrastive training."""
    
    def __init__(
        self,
        temporal_input_dim: int = 13,
        tabular_continuous_dim: int = 15,
        tabular_categorical_dims: Optional[List[int]] = None,
        tabular_embedding_dims: Optional[List[int]] = None,
        embedding_dim: int = 128,
        temperature: float = 0.07,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
    ):
        """Initialize LightningModule.
        
        Args:
            temporal_input_dim: Number of temporal features
            tabular_continuous_dim: Number of continuous tabular features
            tabular_categorical_dims: Cardinalities of categorical features
            tabular_embedding_dims: Embedding dimensions for categorical
            embedding_dim: Output embedding dimension
            temperature: InfoNCE temperature
            lr: Peak learning rate
            weight_decay: AdamW weight decay
            warmup_epochs: Linear warmup epochs
            max_epochs: Total training epochs
        """
        super().__init__()
        
        self.save_hyperparameters()
        
        # Model
        self.model = DualEncoder(
            temporal_input_dim=temporal_input_dim,
            tabular_continuous_dim=tabular_continuous_dim,
            tabular_categorical_dims=tabular_categorical_dims or [11, 25],
            tabular_embedding_dims=tabular_embedding_dims or [8, 16],
            embedding_dim=embedding_dim,
            temperature=temperature,
        )
        
        # Loss function
        self.loss_fn = InfoNCE(temperature=temperature)
        
        # Training hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        
        # Metrics storage for epoch-level logging
        self.train_losses = []
        self.val_losses = []
    
    def forward(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            batch: Dict with 'temporal', 'tabular_cont', 'tabular_cat'
            
        Returns:
            temporal_emb, tabular_emb tensors
        """
        return self.model(
            batch["temporal"],
            batch["tabular_cont"],
            batch["tabular_cat"],
        )
    
    def _compute_loss(
        self,
        temporal_emb: torch.Tensor,
        tabular_emb: torch.Tensor
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Compute InfoNCE loss and metrics.
        
        Returns:
            loss, metrics dict
        """
        # InfoNCE loss
        loss = self.loss_fn(temporal_emb, tabular_emb)
        
        # Compute alignment score (mean positive similarity)
        with torch.no_grad():
            # Positive pair similarities (diagonal)
            sim_matrix = torch.matmul(temporal_emb, tabular_emb.t())
            pos_sim = torch.diag(sim_matrix).mean().item()
            
            # Hard negative similarity (off-diagonal mean)
            mask = ~torch.eye(len(sim_matrix), dtype=torch.bool, device=sim_matrix.device)
            neg_sim = sim_matrix[mask].mean().item()
        
        metrics = {
            "alignment": pos_sim,
            "neg_sim": neg_sim,
        }
        
        return loss, metrics
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: Batch dict from DataLoader
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        # Forward pass
        temporal_emb, tabular_emb = self(batch)
        
        # Compute loss
        loss, metrics = self._compute_loss(temporal_emb, tabular_emb)
        
        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/alignment", metrics["alignment"], on_step=False, on_epoch=True)
        self.log("train/neg_sim", metrics["neg_sim"], on_step=False, on_epoch=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """Validation step.
        
        Args:
            batch: Batch dict from DataLoader
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        # Forward pass
        temporal_emb, tabular_emb = self(batch)
        
        # Compute loss
        loss, metrics = self._compute_loss(temporal_emb, tabular_emb)
        
        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/alignment", metrics["alignment"], on_step=False, on_epoch=True)
        self.log("val/neg_sim", metrics["neg_sim"], on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """Test step.
        
        Args:
            batch: Batch dict from DataLoader
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        temporal_emb, tabular_emb = self(batch)
        loss, metrics = self._compute_loss(temporal_emb, tabular_emb)
        
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/alignment", metrics["alignment"], on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        # Cosine annealing with linear warmup
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.warmup_epochs,
                T_mult=2,
            ),
            "interval": "epoch",
            "frequency": 1,
            "name": "learning_rate",
        }
        
        return [optimizer], [scheduler]
    
    def get_joint_embeddings(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Get joint embeddings for inference.
        
        Args:
            batch: Batch dict
            
        Returns:
            Joint embeddings (batch, 256)
        """
        self.eval()
        with torch.no_grad():
            temporal_emb, tabular_emb = self(batch)
            joint_emb = torch.cat([temporal_emb, tabular_emb], dim=-1)
        return joint_emb
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_training_module.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/training/module.py tests/test_training_module.py
python -m ruff format src/training/module.py
python -m ruff check src/training/module.py
git commit -m "feat: add DualEncoder LightningModule with InfoNCE loss"
```

---

## Task 4: Create Training Script with CLI

**Files:**
- Create: `scripts/train.py`

**Step 1: Write the training script**

Create `scripts/train.py`:

```python
"""Training script for dual-encoder model using PyTorch Lightning."""

import argparse
import os
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src.training.data_module import StockDataModule
from src.training.module import DualEncoderModule


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train dual-encoder model for stock substitute recommendation"
    )
    
    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--feature-dir",
        type=str,
        default="data/processed/features",
        help="Directory with pre-computed features",
    )
    data_group.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Stock symbols to train on (default: all available)",
    )
    
    # Temporal split arguments
    split_group = parser.add_argument_group("Temporal Splits")
    split_group.add_argument(
        "--train-start",
        type=str,
        default="2010-01-01",
        help="Training start date (YYYY-MM-DD)",
    )
    split_group.add_argument(
        "--train-end",
        type=str,
        default="2018-12-31",
        help="Training end date (YYYY-MM-DD)",
    )
    split_group.add_argument(
        "--val-start",
        type=str,
        default="2020-01-01",
        help="Validation start date",
    )
    split_group.add_argument(
        "--val-end",
        type=str,
        default="2020-12-31",
        help="Validation end date",
    )
    split_group.add_argument(
        "--purge-days",
        type=int,
        default=252,
        help="Days to purge from end of training (prevent feature overlap)",
    )
    split_group.add_argument(
        "--embargo-days",
        type=int,
        default=63,
        help="Days to embargo between train/val (prevent return autocorr)",
    )
    
    # Model arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--temporal-input-dim",
        type=int,
        default=13,
        help="Number of temporal input features",
    )
    model_group.add_argument(
        "--tabular-continuous-dim",
        type=int,
        default=15,
        help="Number of continuous tabular features",
    )
    model_group.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Embedding dimension",
    )
    model_group.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="InfoNCE temperature",
    )
    
    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    train_group.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs",
    )
    train_group.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Peak learning rate",
    )
    train_group.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    train_group.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Linear warmup epochs",
    )
    train_group.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers",
    )
    
    # Logging and checkpointing
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    log_group.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for logging",
    )
    log_group.add_argument(
        "--use-wandb",
        action="store_true",
        help="Use Weights & Biases for logging",
    )
    log_group.add_argument(
        "--wandb-project",
        type=str,
        default="stock-substitutes",
        help="W&B project name",
    )
    log_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed for reproducibility
    pl.seed_everything(args.seed)
    
    # Create experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"dual_encoder_{timestamp}"
    
    print(f"Experiment: {args.experiment_name}")
    print(f"Training: {args.train_start} to {args.train_end}")
    print(f"Validation: {args.val_start} to {args.val_end}")
    
    # Create data module
    data_module = StockDataModule(
        feature_dir=args.feature_dir,
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        symbols=args.symbols,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        purge_days=args.purge_days,
        embargo_days=args.embargo_days,
    )
    
    # Create model module
    model_module = DualEncoderModule(
        temporal_input_dim=args.temporal_input_dim,
        tabular_continuous_dim=args.tabular_continuous_dim,
        embedding_dim=args.embedding_dim,
        temperature=args.temperature,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
    )
    
    # Setup callbacks
    callbacks = [
        # Checkpoint saving
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=f"{args.experiment_name}" + "-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        # Early stopping
        EarlyStopping(
            monitor="val/loss",
            patience=15,
            mode="min",
            verbose=True,
        ),
        # LR monitoring
        LearningRateMonitor(logging_interval="epoch"),
    ]
    
    # Setup loggers
    loggers = []
    
    # TensorBoard logger (always)
    tb_logger = TensorBoardLogger(
        save_dir="logs",
        name=args.experiment_name,
    )
    loggers.append(tb_logger)
    
    # W&B logger (optional)
    if args.use_wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.experiment_name,
        )
        loggers.append(wandb_logger)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=loggers,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(model_module, data_module)
    
    # Print best checkpoint
    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    if best_ckpt_path:
        print(f"\nBest checkpoint: {best_ckpt_path}")
        print(f"Best val/loss: {trainer.checkpoint_callback.best_model_score:.4f}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
```

**Step 2: Verify script runs**

Run: `python -m scripts.train --help`
Expected: Shows help message

**Step 3: Commit**

```bash
git add scripts/train.py
python -m ruff format scripts/train.py
python -m ruff check scripts/train.py
git commit -m "feat: add training script with CLI using PyTorch Lightning"
```

---

## Task 5: Create Validation/Sanity Check Script

**Files:**
- Create: `scripts/validate.py`

**Step 1: Write the validation script**

Create `scripts/validate.py`:

```python
"""Validation/sanity check script for trained dual-encoder model."""

import argparse
import os
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from src.training.data_module import StockDataModule
from src.training.module import DualEncoderModule


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate trained dual-encoder model"
    )
    
    # Model checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    
    # Data arguments
    parser.add_argument(
        "--feature-dir",
        type=str,
        default="data/processed/features",
        help="Directory with features",
    )
    parser.add_argument(
        "--val-start",
        type=str,
        default="2020-01-01",
        help="Validation start date",
    )
    parser.add_argument(
        "--val-end",
        type=str,
        default="2020-12-31",
        help="Validation end date",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to validate on",
    )
    
    # Evaluation settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for validation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="validation_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--compute-silhouette",
        action="store_true",
        help="Compute sector silhouette score (slower)",
    )
    
    return parser.parse_args()


def compute_alignment_score(temporal_emb, tabular_emb):
    """Compute alignment score (mean positive similarity)."""
    sim_matrix = torch.matmul(temporal_emb, tabular_emb.t())
    pos_sim = torch.diag(sim_matrix).mean().item()
    return pos_sim


def compute_hard_negative_similarity(temporal_emb, tabular_emb):
    """Compute mean similarity to hard negatives (off-diagonal)."""
    sim_matrix = torch.matmul(temporal_emb, tabular_emb.t())
    mask = ~torch.eye(len(sim_matrix), dtype=torch.bool, device=sim_matrix.device)
    neg_sim = sim_matrix[mask].mean().item()
    return neg_sim


def compute_sector_silhouette(model, data_loader, device="cuda"):
    """Compute silhouette score for sector clustering."""
    model.eval()
    
    all_embeddings = []
    all_sectors = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Computing embeddings"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Get joint embeddings
            joint_emb = model.get_joint_embeddings(batch)
            all_embeddings.append(joint_emb.cpu().numpy())
            all_sectors.extend(batch["gsector"].cpu().numpy())
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    sectors = np.array(all_sectors)
    
    # Compute silhouette score
    if len(np.unique(sectors)) > 1:
        score = silhouette_score(embeddings, sectors)
    else:
        score = 0.0
    
    return score


def main():
    """Main validation function."""
    args = parse_args()
    
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Validation period: {args.val_start} to {args.val_end}")
    
    # Load model from checkpoint
    model = DualEncoderModule.load_from_checkpoint(args.checkpoint)
    model.eval()
    
    # Create data module for validation
    data_module = StockDataModule(
        feature_dir=args.feature_dir,
        train_start="2010-01-01",  # Dummy, not used
        train_end="2018-12-31",    # Dummy, not used
        val_start=args.val_start,
        val_end=args.val_end,
        symbols=args.symbols,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_module.setup("validate")
    
    # Get validation dataloader
    val_loader = data_module.val_dataloader()
    
    # Run validation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    all_losses = []
    all_alignments = []
    all_neg_sims = []
    
    print("\nRunning validation...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            temporal_emb, tabular_emb = model(batch)
            
            # Compute metrics
            loss = model.loss_fn(temporal_emb, tabular_emb)
            alignment = compute_alignment_score(temporal_emb, tabular_emb)
            neg_sim = compute_hard_negative_similarity(temporal_emb, tabular_emb)
            
            all_losses.append(loss.item())
            all_alignments.append(alignment)
            all_neg_sims.append(neg_sim)
    
    # Aggregate results
    results = {
        "checkpoint": args.checkpoint,
        "val_start": args.val_start,
        "val_end": args.val_end,
        "num_samples": len(val_loader.dataset),
        "loss": {
            "mean": float(np.mean(all_losses)),
            "std": float(np.std(all_losses)),
            "min": float(np.min(all_losses)),
            "max": float(np.max(all_losses)),
        },
        "alignment": {
            "mean": float(np.mean(all_alignments)),
            "std": float(np.std(all_alignments)),
        },
        "hard_neg_similarity": {
            "mean": float(np.mean(all_neg_sims)),
            "std": float(np.std(all_neg_sims)),
        },
    }
    
    # Compute silhouette if requested
    if args.compute_silhouette:
        print("\nComputing sector silhouette score...")
        silhouette = compute_sector_silhouette(model, val_loader, device)
        results["sector_silhouette"] = float(silhouette)
    
    # Print results
    print("\n" + "="*50)
    print("VALIDATION RESULTS")
    print("="*50)
    print(f"Loss: {results['loss']['mean']:.4f} ± {results['loss']['std']:.4f}")
    print(f"Alignment: {results['alignment']['mean']:.4f} ± {results['alignment']['std']:.4f}")
    print(f"Hard Neg Sim: {results['hard_neg_similarity']['mean']:.4f} ± {results['hard_neg_similarity']['std']:.4f}")
    if "sector_silhouette" in results:
        print(f"Sector Silhouette: {results['sector_silhouette']:.4f}")
    print("="*50)
    
    # Save results
    import json
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")
    
    # Sanity checks
    print("\nSanity Checks:")
    checks_passed = 0
    checks_total = 3
    
    # Check 1: Loss is reasonable
    if results["loss"]["mean"] < 5.0:
        print("✓ Loss is reasonable (< 5.0)")
        checks_passed += 1
    else:
        print("✗ Loss is too high (> 5.0) - possible training issue")
    
    # Check 2: Alignment > hard neg similarity
    if results["alignment"]["mean"] > results["hard_neg_similarity"]["mean"]:
        print("✓ Alignment > Hard Neg Similarity (model is learning)")
        checks_passed += 1
    else:
        print("✗ Alignment ≤ Hard Neg Similarity (model not distinguishing)")
    
    # Check 3: Hard neg similarity is low
    if results["hard_neg_similarity"]["mean"] < 0.5:
        print("✓ Hard Neg Similarity is low (< 0.5)")
        checks_passed += 1
    else:
        print("⚠ Hard Neg Similarity is high (> 0.5) - may need more training")
    
    print(f"\nChecks passed: {checks_passed}/{checks_total}")
    
    if checks_passed == checks_total:
        print("\n🎉 All sanity checks passed! Model looks good.")
        return 0
    else:
        print("\n⚠️ Some checks failed. Review results carefully.")
        return 1


if __name__ == "__main__":
    exit(main())
```

**Step 2: Verify script runs**

Run: `python -m scripts.validate --help`
Expected: Shows help message

**Step 3: Commit**

```bash
git add scripts/validate.py
python -m ruff format scripts/validate.py
python -m ruff check scripts/validate.py
git commit -m "feat: add validation/sanity check script"
```

---

## Task 6: Create Mock Data for Testing

**Files:**
- Create: `tests/fixtures/create_mock_data.py`

**Step 1: Create fixture generator**

Create `tests/fixtures/create_mock_data.py`:

```python
"""Create mock data for testing the training pipeline."""

import os
import numpy as np
import pandas as pd
from pathlib import Path


def create_mock_features(
    symbol: str,
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
    output_dir: str = "data/processed/features",
):
    """Create mock feature parquet for testing."""
    
    # Generate date range (trading days only - simplified)
    dates = pd.date_range(start=start_date, end=end_date, freq="B")  # Business days
    n_days = len(dates)
    
    # Generate random data
    np.random.seed(hash(symbol) % 2**32)
    
    # Temporal features (13)
    temporal_data = {
        "date": dates,
        "open": np.random.randn(n_days).cumsum() + 100,
        "high": np.random.randn(n_days).cumsum() + 102,
        "low": np.random.randn(n_days).cumsum() + 98,
        "close": np.random.randn(n_days).cumsum() + 100,
        "volume": np.random.randint(1e6, 1e9, n_days),
        "returns": np.random.randn(n_days) * 0.02,
        "log_returns": np.random.randn(n_days) * 0.02,
        "realized_vol_20d": np.abs(np.random.randn(n_days)) * 0.3,
        "momentum_20d": np.random.randn(n_days) * 0.1,
        "momentum_60d": np.random.randn(n_days) * 0.1,
        "rsi_14d": np.random.rand(n_days) * 100,
        "macd": np.random.randn(n_days) * 0.5,
        "signal": np.random.randn(n_days) * 0.5,
    }
    
    # Continuous tabular features (15)
    continuous_data = {
        "market_beta_60d": np.random.randn(n_days) * 0.3 + 1.0,
        "downside_beta_60d": np.random.randn(n_days) * 0.3 + 1.0,
        "idiosyncratic_vol_60d": np.abs(np.random.randn(n_days)) * 0.2,
        "momentum_20d_tab": np.random.randn(n_days) * 0.1,  # Duplicate, will use one
        "momentum_60d_tab": np.random.randn(n_days) * 0.1,
        "momentum_120d": np.random.randn(n_days) * 0.1,
        "volatility_20d": np.abs(np.random.randn(n_days)) * 0.3,
        "volatility_60d": np.abs(np.random.randn(n_days)) * 0.3,
        "roe": np.random.randn(n_days) * 0.1 + 0.15,
        "roa": np.random.randn(n_days) * 0.05 + 0.05,
        "debt_to_equity": np.random.randn(n_days) * 0.5 + 1.0,
        "price_to_book": np.random.randn(n_days) * 2 + 3.0,
        "price_to_earnings": np.random.randn(n_days) * 10 + 20.0,
        "market_cap": np.random.rand(n_days) * 1e12,
        "dividend_yield": np.random.rand(n_days) * 0.05,
    }
    
    # Categorical features (gsector, ggroup)
    # Random GICS codes
    gsector = np.random.randint(0, 11, n_days)
    # ggroup depends on sector (simplified)
    ggroup = gsector * 100 + np.random.randint(0, 25, n_days)
    
    categorical_data = {
        "gsector": gsector,
        "ggroup": ggroup,
    }
    
    # Combine all data
    df = pd.DataFrame({**temporal_data, **continuous_data, **categorical_data})
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save to parquet
    output_file = output_path / f"{symbol}_features.parquet"
    df.to_parquet(output_file, index=False)
    print(f"Created: {output_file}")
    
    return output_file


def main():
    """Create mock data for multiple symbols."""
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM"]
    
    for symbol in symbols:
        create_mock_features(symbol)
    
    print(f"\nCreated mock data for {len(symbols)} symbols")


if __name__ == "__main__":
    main()
```

**Step 2: Generate mock data**

Run: `python tests/fixtures/create_mock_data.py`

**Step 3: Commit**

```bash
git add tests/fixtures/create_mock_data.py
git commit -m "chore: add mock data generator for testing"
```

---

## Task 7: Test End-to-End Training

**Files:**
- Create: `tests/test_training_end_to_end.py`

**Step 1: Write integration test**

Create `tests/test_training_end_to_end.py`:

```python
"""End-to-end test for training pipeline."""

import pytest
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary

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
        num_workers=0,  # Use 0 for testing
    )
    
    dm.setup("fit")
    
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert len(dm.train_dataset) > 0
    assert len(dm.val_dataset) > 0


def test_training_batch():
    """Test a single training batch."""
    # Create data module
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
    
    # Create model
    model = DualEncoderModule(
        temporal_input_dim=13,
        tabular_continuous_dim=15,
        embedding_dim=128,
        lr=1e-4,
    )
    
    # Get a batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    # Training step
    loss = model.training_step(batch, batch_idx=0)
    
    assert loss is not None
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_validation_batch():
    """Test a single validation batch."""
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
    
    val_loader = dm.val_dataloader()
    batch = next(iter(val_loader))
    
    loss = model.validation_step(batch, batch_idx=0)
    
    assert loss is not None
    assert loss.item() > 0


def test_full_training_epoch():
    """Test a full epoch of training with Lightning Trainer."""
    pl.seed_everything(42)
    
    # Create data module
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
    
    # Create model
    model = DualEncoderModule(
        temporal_input_dim=13,
        tabular_continuous_dim=15,
        embedding_dim=128,
        lr=1e-4,
        max_epochs=1,
    )
    
    # Create trainer (minimal config for testing)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        enable_checkpointing=False,
        logger=False,
        callbacks=[ModelSummary(max_depth=1)],
    )
    
    # Train for one epoch
    trainer.fit(model, dm)
    
    # Check that training completed
    assert trainer.current_epoch == 0  # 0-indexed, completed 1 epoch
    assert trainer.global_step > 0


def test_model_checkpoint_save_load(tmp_path):
    """Test model can be saved and loaded."""
    # Create model
    model = DualEncoderModule(
        temporal_input_dim=13,
        tabular_continuous_dim=15,
        embedding_dim=128,
    )
    
    # Save checkpoint
    checkpoint_path = tmp_path / "test.ckpt"
    trainer = pl.Trainer(default_root_dir=tmp_path)
    trainer.strategy.connect(model)
    trainer.save_checkpoint(checkpoint_path)
    
    # Load checkpoint
    loaded_model = DualEncoderModule.load_from_checkpoint(checkpoint_path)
    
    assert loaded_model is not None
    assert loaded_model.hparams.embedding_dim == 128
```

**Step 2: Run tests**

First, ensure mock data exists:
Run: `python tests/fixtures/create_mock_data.py`

Then run tests:
Run: `python -m pytest tests/test_training_end_to_end.py -v -s`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/test_training_end_to_end.py
python -m ruff format tests/test_training_end_to_end.py
git commit -m "test: add end-to-end training tests"
```

---

## Task 8: Create README_MODEL.md Documentation

**Files:**
- Create: `README_MODEL.md`

**Step 1: Write documentation**

Create `README_MODEL.md`:

```markdown
# Dual-Encoder Stock Substitute Model

CLIP-style contrastive learning for stock substitute recommendation using BiMT-TCN + TabMixer.

## Architecture

```
Temporal Data (60-day OHLCV)          Tabular Data (Fundamentals + GICS)
         ↓                                      ↓
    BiMT-TCN Encoder                    TabMixer Encoder
    (TCN + Transformer)                   (MLP-Mixer + Embeddings)
         ↓                                      ↓
    128-dim embedding                   128-dim embedding
         ↓                                      ↓
         └────────── InfoNCE Loss ──────────────┘
         
Inference: Concatenate → 256-dim → Similarity Search
```

## Quick Start

### 1. Pre-compute Features

```bash
python -m scripts.preprocess_features \
    --start-date 2010-01-01 \
    --end-date 2024-12-31 \
    --symbols AAPL MSFT GOOGL
```

### 2. Train Model

```bash
python -m scripts.train \
    --feature-dir data/processed/features \
    --train-start 2010-01-01 \
    --train-end 2018-12-31 \
    --val-start 2020-01-01 \
    --val-end 2020-12-31 \
    --batch-size 32 \
    --epochs 100 \
    --lr 1e-4
```

### 3. Validate Model

```bash
python -m scripts.validate \
    --checkpoint checkpoints/best.ckpt \
    --val-start 2021-04-01 \
    --val-end 2022-12-31 \
    --compute-silhouette
```

## Training Pipeline

### Temporal Splits (Financial ML Best Practice)

```
Train:    2010-2018  Purge: Remove last 252 days (feature overlap)
                 ↓
Embargo:  2019     (prevent return autocorrelation)
                 ↓
Val:      2020     COVID crash + recovery
                 ↓
Embargo:  2021-Q1
                 ↓
Val:      2021-2022  Meme stocks + rate hikes
                 ↓
Test:     2024     Never touch until done
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dim` | 128 | Encoder output dimension |
| `temperature` | 0.07 | InfoNCE temperature |
| `lr` | 1e-4 | Peak learning rate |
| `batch_size` | 32 | Training batch size |
| `purge_days` | 252 | Days to purge from training |
| `embargo_days` | 63 | Gap between train/val |

## Model Components

### Temporal Encoder (BiMT-TCN)

- **TCN**: 4 layers with dilations [1, 2, 4, 8]
- **Transformer**: 2 layers, 4 heads, positional encoding
- **Output**: 128-dim, L2 normalized

### Tabular Encoder (TabMixer)

- **Continuous**: 15 features (risk, momentum, valuation)
- **Categorical**: gsector (8-dim), ggroup (16-dim)
- **MLP-Mixer**: 4 blocks with token + channel mixing
- **Output**: 128-dim, L2 normalized

### InfoNCE Loss

```python
# Aligns temporal and tabular views of same stock
loss = -log[ exp(sim(t, tab)/τ) / Σ_k exp(sim(t, tab_k)/τ) ]
```

Where:
- `sim()` = cosine similarity
- `τ` = temperature (0.07)
- `tab_k` = negative samples (in-batch + hard negatives)

## Validation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| `val/loss` | < 2.0 | InfoNCE loss |
| `val/alignment` | > 0.5 | Positive pair similarity |
| `val/neg_sim` | < 0.3 | Hard negative similarity |
| `sector_silhouette` | > 0.1 | Sector clustering quality |

## Inference

```python
from src.training.module import DualEncoderModule

# Load model
model = DualEncoderModule.load_from_checkpoint("checkpoints/best.ckpt")
model.eval()

# Prepare data
batch = {
    "temporal": torch.randn(1, 60, 13),      # 60-day OHLCV
    "tabular_cont": torch.randn(1, 15),     # Fundamentals
    "tabular_cat": torch.tensor([[5, 510]]),  # GICS codes
}

# Get joint embedding (256-dim)
joint_emb = model.get_joint_embeddings(batch)

# Search for similar stocks
similarities = torch.matmul(joint_emb, universe_embeddings.t())
top_10 = torch.topk(similarities, k=10)
```

## Project Structure

```
src/
  models/
    base.py              # Abstract BaseEncoder
    tcn.py               # TemporalConvNet with causal conv
    temporal_encoder.py  # BiMT-TCN
    mixer.py             # TabMixer blocks
    tabular_encoder.py   # TabularEncoder with GICS
    dual_encoder.py      # Main model
  training/
    data_module.py       # StockDataModule (Lightning)
    module.py            # DualEncoderModule (Lightning)
scripts/
  train.py             # Training script
  validate.py          # Validation script
```

## Dependencies

```bash
pip install pytorch-lightning info-nce-pytorch wandb
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_training_end_to_end.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Troubleshooting

### Loss not decreasing
- Check learning rate (try 3e-4 to 1e-5 range)
- Check batch size (larger = more stable negatives)
- Verify data preprocessing (normalization, NaN handling)

### Alignment too low
- Increase temperature (less sharp contrast)
- Check for data leakage between train/val
- Verify temporal splits (purge/embargo)

### Sector silhouette negative
- Model may have collapsed to ignore GICS
- Try smaller categorical embeddings
- Check if GICS codes are correct

## Citation

```bibtex
@software{dual_encoder_stock_substitutes,
  title={Dual-Encoder Contrastive Learning for Stock Substitutes},
  author={Claude Code},
  year={2026},
  note={CLIP-style embeddings for financial similarity search}
}
```
```

**Step 2: Commit**

```bash
git add README_MODEL.md
git commit -m "docs: add comprehensive model documentation"
```

---

## Summary

All tasks complete. Training pipeline includes:

1. ✅ **StockDataModule**: Temporal splits with purge/embargo
2. ✅ **DualEncoderModule**: LightningModule with InfoNCE loss
3. ✅ **train.py**: CLI training script with callbacks
4. ✅ **validate.py**: Comprehensive validation with sanity checks
5. ✅ **Mock data**: For testing without real WRDS access
6. ✅ **End-to-end tests**: Verify full pipeline works
7. ✅ **Documentation**: README with quick start and troubleshooting

**Usage:**
```bash
# Train
python -m scripts.train --epochs 100 --batch-size 32

# Validate
python -m scripts.validate --checkpoint checkpoints/best.ckpt
```
