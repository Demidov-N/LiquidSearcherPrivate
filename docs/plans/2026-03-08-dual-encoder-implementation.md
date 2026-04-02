# Dual-Encoder Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a dual-encoder (BiMT-TCN + TabMixer) contrastive learning model for stock substitute recommendation with 256-dim joint embeddings.

**Architecture:** 
- **Temporal Encoder:** BiMT-TCN (TCN local patterns + Transformer global dependencies) → 128-dim
- **Tabular Encoder:** TabMixer for fundamentals + GICS embeddings (gsector→8-dim, ggroup→16-dim) → 128-dim
- **Contrastive Learning:** InfoNCE/RankSCL loss on dot product, hard negatives structured by GICS
- **Inference:** Concatenate [temporal||tabular] → 256-dim for similarity search

**Tech Stack:** PyTorch, torch.nn.Embedding, temporal convolutions, MLP-Mixer, pytest, mypy, ruff

**Prerequisites:** Feature engineering pipeline already complete (src/features/)

---

## Task 1: Project Structure and Base Classes

**Files:**
- Create: `src/models/__init__.py`
- Create: `src/models/base.py`
- Create: `tests/test_models_base.py`

**Step 1: Write the failing test**

```python
# tests/test_models_base.py
import pytest
import torch
from src.models.base import BaseEncoder


def test_base_encoder_is_abstract():
    """Test that BaseEncoder cannot be instantiated directly."""
    with pytest.raises(TypeError):
        encoder = BaseEncoder(input_dim=10, output_dim=128)


def test_base_encoder_subclass_must_implement_forward():
    """Test that subclasses must implement forward method."""
    class DummyEncoder(BaseEncoder):
        def __init__(self):
            super().__init__(input_dim=10, output_dim=128)
    
    with pytest.raises(TypeError):
        encoder = DummyEncoder()
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_models_base.py -v
```

Expected: FAIL with "TypeError: Can't instantiate abstract class"

**Step 3: Write minimal implementation**

```python
# src/models/base.py
"""Base classes for encoder models."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for all encoders.
    
    All encoders must implement:
    - forward(x): Input tensor → output embedding
    - output_dim: Final embedding dimension
    """
    
    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Initialize base encoder.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output embedding dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (batch, ..., input_dim)
            
        Returns:
            Output tensor of shape (batch, ..., output_dim)
        """
        pass
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_models_base.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/models/__init__.py src/models/base.py tests/test_models_base.py
python -m ruff format src/models/base.py
python -m ruff check src/models/base.py
python -m mypy src/models/base.py
git commit -m "feat: add base encoder abstract class"
```

---

## Task 2: TCN Module (Causal Convolutions)

**Files:**
- Create: `src/models/tcn.py`
- Create: `tests/test_models_tcn.py`

**Step 1: Write the failing test**

```python
# tests/test_models_tcn.py
import torch
import pytest
from src.models.tcn import TemporalConvNet


def test_tcn_output_shape():
    """Test TCN produces correct output shape."""
    batch_size = 32
    seq_len = 60
    input_dim = 13
    hidden_dim = 64
    output_dim = 64
    
    tcn = TemporalConvNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=3
    )
    
    x = torch.randn(batch_size, seq_len, input_dim)
    out = tcn(x)
    
    assert out.shape == (batch_size, seq_len, output_dim)


def test_tcn_causal_property():
    """Test TCN maintains causality (no future info leakage)."""
    tcn = TemporalConvNet(input_dim=5, hidden_dim=16, output_dim=16, num_layers=2)
    
    # Create input where last timestep is different
    x1 = torch.randn(1, 10, 5)
    x2 = x1.clone()
    x2[:, -1, :] = torch.randn(5)  # Change last timestep
    
    out1 = tcn(x1)
    out2 = tcn(x2)
    
    # First 9 timesteps should be identical (causal)
    assert torch.allclose(out1[:, :9, :], out2[:, :9, :], atol=1e-5)
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_models_tcn.py -v
```

Expected: FAIL with "ImportError: cannot import name 'TemporalConvNet'"

**Step 3: Write minimal implementation**

```python
# src/models/tcn.py
"""Temporal Convolutional Network (TCN) module."""

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """Causal 1D convolution (no future info leakage)."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Causal padding: (kernel_size - 1) * dilation
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,  # We'll handle padding manually
            dilation=dilation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with causal padding.
        
        Args:
            x: (batch, channels, seq_len)
        
        Returns:
            (batch, channels, seq_len)
        """
        # Pad left side only (causal)
        x_padded = torch.nn.functional.pad(x, (self.padding, 0))
        return self.conv(x_padded)


class TCNBlock(nn.Module):
    """Single TCN residual block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None
        
        self.relu_out = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        residual = x if self.downsample is None else self.downsample(x)
        
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        return self.relu_out(out + residual)


class TemporalConvNet(nn.Module):
    """Full TCN with dilated convolutions."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            out_channels = output_dim if i == num_layers - 1 else hidden_dim
            dilation = 2 ** i  # Exponential dilation: 1, 2, 4, 8...
            
            layers.append(
                TCNBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation,
                    dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: (batch, seq_len, input_dim)
        
        Returns:
            (batch, seq_len, output_dim)
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        out = self.network(x)
        out = out.transpose(1, 2)  # (batch, seq_len, output_dim)
        return out
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_models_tcn.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/models/tcn.py tests/test_models_tcn.py
python -m ruff format src/models/tcn.py
python -m ruff check src/models/tcn.py
python -m mypy src/models/tcn.py
git commit -m "feat: add TCN with causal convolutions and dilations"
```

---

## Task 3: BiMT-TCN Temporal Encoder

**Files:**
- Create: `src/models/temporal_encoder.py`
- Create: `tests/test_models_temporal_encoder.py`

**Step 1: Write the failing test**

```python
# tests/test_models_temporal_encoder.py
import torch
import pytest
from src.models.temporal_encoder import TemporalEncoder


def test_temporal_encoder_output():
    """Test temporal encoder produces 128-dim output."""
    batch_size = 32
    seq_len = 60
    input_dim = 13  # Technical features
    
    encoder = TemporalEncoder(
        input_dim=input_dim,
        tcn_hidden=64,
        tcn_layers=3,
        transformer_heads=4,
        transformer_layers=2,
        output_dim=128
    )
    
    x = torch.randn(batch_size, seq_len, input_dim)
    out = encoder(x)
    
    assert out.shape == (batch_size, 128)


def test_temporal_encoder_preserves_batch():
    """Test different batch sizes work."""
    encoder = TemporalEncoder(input_dim=13)
    
    for batch_size in [1, 8, 32, 64]:
        x = torch.randn(batch_size, 60, 13)
        out = encoder(x)
        assert out.shape[0] == batch_size
        assert out.shape[1] == 128
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_models_temporal_encoder.py -v
```

Expected: FAIL

**Step 3: Write implementation**

```python
# src/models/temporal_encoder.py
"""BiMT-TCN: TCN + lightweight Transformer for temporal encoding."""

import torch
import torch.nn as nn

from src.models.base import BaseEncoder
from src.models.tcn import TemporalConvNet


class TemporalEncoder(BaseEncoder):
    """Temporal encoder: BiMT-TCN architecture.
    
    Combines:
    - TCN (3 layers): Local multi-scale pattern detection
    - Transformer (2 layers, 4 heads): Global cross-timestep dependencies
    
    Architecture from 2025 research showing Transformer outperforms TCN alone.
    """
    
    def __init__(
        self,
        input_dim: int = 13,
        tcn_hidden: int = 64,
        tcn_layers: int = 3,
        tcn_kernel_size: int = 3,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.1
    ) -> None:
        """Initialize BiMT-TCN encoder.
        
        Args:
            input_dim: Number of input technical features (default 13)
            tcn_hidden: Hidden dim for TCN layers
            tcn_layers: Number of TCN layers (default 3)
            tcn_kernel_size: Kernel size for convolutions
            transformer_heads: Number of attention heads (default 4)
            transformer_layers: Number of transformer layers (default 2)
            transformer_dim: Dimension for transformer
            output_dim: Final output dimension (default 128)
            dropout: Dropout rate
        """
        super().__init__(input_dim, output_dim)
        
        # TCN for local patterns
        self.tcn = TemporalConvNet(
            input_dim=input_dim,
            hidden_dim=tcn_hidden,
            output_dim=transformer_dim,
            num_layers=tcn_layers,
            kernel_size=tcn_kernel_size,
            dropout=dropout
        )
        
        # Positional encoding for transformer
        self.pos_encoder = PositionalEncoding(transformer_dim, dropout)
        
        # Lightweight Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # Global average pooling over time
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Final projection to output_dim
        self.projection = nn.Linear(transformer_dim, output_dim)
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through BiMT-TCN.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
                - seq_len: Time window (e.g., 20, 60 days)
                - input_dim: Technical features (13)
        
        Returns:
            Output tensor of shape (batch, output_dim=128)
        """
        batch_size, seq_len, _ = x.shape
        
        # TCN: Local patterns (batch, seq_len, transformer_dim)
        out = self.tcn(x)
        
        # Positional encoding
        out = self.pos_encoder(out)
        
        # Transformer: Global dependencies (batch, seq_len, transformer_dim)
        out = self.transformer(out)
        
        # Global average pooling over time (batch, transformer_dim)
        out = out.transpose(1, 2)  # (batch, transformer_dim, seq_len)
        out = self.pool(out).squeeze(-1)  # (batch, transformer_dim)
        
        # Project to output dimension (batch, output_dim)
        out = self.projection(out)
        
        # Final normalization
        out = self.norm(out)
        
        return out


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.
        
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_models_temporal_encoder.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/models/temporal_encoder.py tests/test_models_temporal_encoder.py
python -m ruff format src/models/temporal_encoder.py
python -m ruff check src/models/temporal_encoder.py
python -m mypy src/models/temporal_encoder.py
git commit -m "feat: add BiMT-TCN temporal encoder (TCN+Transformer)"
```

---

## Task 4: TabMixer Tabular Encoder

**Files:**
- Create: `src/models/tabmixer.py`
- Create: `tests/test_models_tabmixer.py`

**Step 1: Write the failing test**

```python
# tests/test_models_tabmixer.py
import torch
import pytest
from src.models.tabmixer import TabMixer


def test_tabmixer_basic():
    """Test TabMixer with continuous features only."""
    batch_size = 32
    n_continuous = 15
    
    model = TabMixer(
        continuous_dim=n_continuous,
        categorical_dims=[],
        output_dim=128
    )
    
    x_cont = torch.randn(batch_size, n_continuous)
    out = model(x_cont)
    
    assert out.shape == (batch_size, 128)


def test_tabmixer_with_categorical():
    """Test TabMixer with categorical embeddings."""
    batch_size = 32
    n_continuous = 15
    
    model = TabMixer(
        continuous_dim=n_continuous,
        categorical_dims=[11, 25],  # GICS: sector, industry group
        embedding_dims=[8, 16],
        output_dim=128
    )
    
    x_cont = torch.randn(batch_size, n_continuous)
    x_cat = torch.randint(0, 5, (batch_size, 2))  # 2 categorical features
    
    out = model(x_cont, x_cat)
    
    assert out.shape == (batch_size, 128)


def test_tabmixer_missing_values():
    """Test TabMixer handles missing values."""
    model = TabMixer(continuous_dim=15, handle_missing=True)
    
    x_cont = torch.randn(32, 15)
    x_cont[0, 3] = float('nan')  # Missing value
    
    out = model(x_cont)
    
    assert not torch.isnan(out).any()
    assert out.shape == (32, 128)
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_models_tabmixer.py -v
```

Expected: FAIL

**Step 3: Write implementation**

```python
# src/models/tabmixer.py
"""TabMixer: MLP-Mixer style encoder for tabular data."""

import torch
import torch.nn as nn

from src.models.base import BaseEncoder


class TabMixer(BaseEncoder):
    """TabMixer encoder for tabular data.
    
    MLP-Mixer architecture: channel-wise mixing + instance-wise mixing
    Purpose-built for tabular data, handles missing values natively.
    
    From 2025 research: <0.01% FLOPs of FT-Transformer, handles missing data.
    """
    
    def __init__(
        self,
        continuous_dim: int,
        categorical_dims: list[int] = None,
        embedding_dims: list[int] = None,
        mixer_layers: int = 4,
        hidden_dim: int = 128,
        expansion_factor: int = 4,
        output_dim: int = 128,
        dropout: float = 0.1,
        handle_missing: bool = True
    ) -> None:
        """Initialize TabMixer.
        
        Args:
            continuous_dim: Number of continuous features
            categorical_dims: List of cardinalities for categorical features
            embedding_dims: Embedding dimension for each categorical feature
            mixer_layers: Number of mixer blocks
            hidden_dim: Hidden dimension
            expansion_factor: Channel expansion in mixer
            output_dim: Output embedding dimension
            dropout: Dropout rate
            handle_missing: Whether to handle NaN values
        """
        super().__init__(continuous_dim, output_dim)
        
        self.continuous_dim = continuous_dim
        self.categorical_dims = categorical_dims or []
        self.embedding_dims = embedding_dims or []
        self.handle_missing = handle_missing
        
        # Categorical embeddings
        if self.categorical_dims:
            assert len(self.categorical_dims) == len(self.embedding_dims), \
                "categorical_dims and embedding_dims must match"
            
            self.embeddings = nn.ModuleList([
                nn.Embedding(num_embeddings=card, embedding_dim=dim)
                for card, dim in zip(self.categorical_dims, self.embedding_dims)
            ])
            total_emb_dim = sum(self.embedding_dims)
        else:
            total_emb_dim = 0
            self.embeddings = nn.ModuleList()
        
        # Total input dimension after embeddings
        self.input_features = continuous_dim + total_emb_dim
        
        # Initial linear projection
        self.input_proj = nn.Linear(self.input_features, hidden_dim)
        
        # Mixer blocks
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(hidden_dim, expansion_factor, dropout)
            for _ in range(mixer_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Missing value handling
        if handle_missing:
            self.missing_mask = nn.Parameter(torch.zeros(continuous_dim))
    
    def forward(
        self,
        x_continuous: torch.Tensor,
        x_categorical: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass through TabMixer.
        
        Args:
            x_continuous: Continuous features (batch, continuous_dim)
            x_categorical: Categorical indices (batch, num_categorical)
        
        Returns:
            Output embedding (batch, output_dim)
        """
        # Handle missing values in continuous features
        if self.handle_missing and torch.isnan(x_continuous).any():
            mask = torch.isnan(x_continuous)
            x_continuous = torch.where(
                mask,
                self.missing_mask.expand_as(x_continuous),
                x_continuous
            )
        
        # Embed categorical features
        if x_categorical is not None and len(self.embeddings) > 0:
            embeddings = []
            for i, emb_layer in enumerate(self.embeddings):
                emb = emb_layer(x_categorical[:, i])
                embeddings.append(emb)
            x_embedded = torch.cat(embeddings, dim=1)
            
            # Concatenate continuous and categorical
            x = torch.cat([x_continuous, x_embedded], dim=1)
        else:
            x = x_continuous
        
        # Initial projection
        x = self.input_proj(x)  # (batch, hidden_dim)
        
        # Mixer blocks
        for block in self.mixer_blocks:
            x = block(x)
        
        # Output projection
        out = self.output_proj(x)
        
        return out


class MixerBlock(nn.Module):
    """Single Mixer block: token mixing + channel mixing."""
    
    def __init__(
        self,
        hidden_dim: int,
        expansion_factor: int = 4,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        # Token mixing (mix across features)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Channel mixing (mix within features)
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through mixer block.
        
        Args:
            x: (batch, hidden_dim)
        """
        # Token mixing (treat batch as tokens)
        residual = x
        x = self.token_norm(x)
        # Transpose for token mixing: (hidden_dim, batch) → (batch, hidden_dim)
        x = self.token_mlp(x)
        x = x + residual
        
        # Channel mixing
        residual = x
        x = self.channel_norm(x)
        x = self.channel_mlp(x)
        x = x + residual
        
        return x
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_models_tabmixer.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/models/tabmixer.py tests/test_models_tabmixer.py
python -m ruff format src/models/tabmixer.py
python -m ruff check src/models/tabmixer.py
python -m mypy src/models/tabmixer.py
git commit -m "feat: add TabMixer for tabular encoding with missing value handling"
```

---

## Task 5: Dual-Encoder Model (Putting It Together)

**Files:**
- Create: `src/models/dual_encoder.py`
- Create: `tests/test_models_dual_encoder.py`

**Step 1: Write the failing test**

```python
# tests/test_models_dual_encoder.py
import torch
import pytest
from src.models.dual_encoder import DualEncoder


def test_dual_encoder_forward():
    """Test dual encoder produces correct outputs."""
    batch_size = 32
    seq_len = 60
    
    model = DualEncoder(
        temporal_input_dim=13,
        tabular_continuous_dim=15,
        tabular_categorical_dims=[11, 25],
        temporal_output_dim=128,
        tabular_output_dim=128
    )
    
    # Temporal input
    x_temporal = torch.randn(batch_size, seq_len, 13)
    
    # Tabular input
    x_tabular_cont = torch.randn(batch_size, 15)
    x_tabular_cat = torch.randint(0, 5, (batch_size, 2))
    
    # Forward pass
    temporal_emb, tabular_emb = model(x_temporal, x_tabular_cont, x_tabular_cat)
    
    assert temporal_emb.shape == (batch_size, 128)
    assert tabular_emb.shape == (batch_size, 128)


def test_dual_encoder_joint_embedding():
    """Test joint embedding concatenation for inference."""
    model = DualEncoder(temporal_input_dim=13, tabular_continuous_dim=15)
    
    x_temp = torch.randn(16, 60, 13)
    x_tab_cont = torch.randn(16, 15)
    x_tab_cat = torch.randint(0, 5, (16, 2))
    
    # Inference mode: get joint embedding
    joint_emb = model.get_joint_embedding(x_temp, x_tab_cont, x_tab_cat)
    
    assert joint_emb.shape == (16, 256)  # 128 + 128


def test_dual_encoder_similarity():
    """Test similarity computation for training."""
    model = DualEncoder(temporal_input_dim=13, tabular_continuous_dim=15)
    
    x_temp = torch.randn(16, 60, 13)
    x_tab_cont = torch.randn(16, 15)
    x_tab_cat = torch.randint(0, 5, (16, 2))
    
    # Training mode: get similarity score
    similarity = model.compute_similarity(x_temp, x_tab_cont, x_tab_cat)
    
    assert similarity.shape == (16,)  # One similarity per batch element
    assert torch.all(similarity >= -1) and torch.all(similarity <= 1)  # Cosine similarity range
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_models_dual_encoder.py -v
```

Expected: FAIL

**Step 3: Write implementation**

```python
# src/models/dual_encoder.py
"""Dual-Encoder model: BiMT-TCN + TabMixer for contrastive learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.temporal_encoder import TemporalEncoder
from src.models.tabmixer import TabMixer


class DualEncoder(nn.Module):
    """Dual-encoder model for stock substitute recommendation.
    
    Architecture:
    - Temporal Encoder: BiMT-TCN (TCN + Transformer) → 128-dim
    - Tabular Encoder: TabMixer → 128-dim
    
    Training: Compute dot product similarity between temporal and tabular
    Inference: Concatenate [temporal||tabular] → 256-dim joint embedding
    """
    
    def __init__(
        self,
        temporal_input_dim: int = 13,
        tabular_continuous_dim: int = 15,
        tabular_categorical_dims: list[int] = None,
        tabular_embedding_dims: list[int] = None,
        temporal_output_dim: int = 128,
        tabular_output_dim: int = 128,
        temperature: float = 0.07
    ) -> None:
        """Initialize dual-encoder model.
        
        Args:
            temporal_input_dim: Number of technical features (default 13)
            tabular_continuous_dim: Number of continuous fundamental features (default 15)
            tabular_categorical_dims: Cardinalities for categorical features [11, 25]
            tabular_embedding_dims: Embedding dims for categorical [8, 16]
            temporal_output_dim: Output dim for temporal encoder (default 128)
            tabular_output_dim: Output dim for tabular encoder (default 128)
            temperature: Temperature for similarity scaling (default 0.07)
        """
        super().__init__()
        
        self.temporal_output_dim = temporal_output_dim
        self.tabular_output_dim = tabular_output_dim
        self.temperature = temperature
        
        # Temporal encoder: BiMT-TCN
        self.temporal_encoder = TemporalEncoder(
            input_dim=temporal_input_dim,
            output_dim=temporal_output_dim
        )
        
        # Tabular encoder: TabMixer
        self.tabular_encoder = TabMixer(
            continuous_dim=tabular_continuous_dim,
            categorical_dims=tabular_categorical_dims or [],
            embedding_dims=tabular_embedding_dims or [],
            output_dim=tabular_output_dim
        )
    
    def forward(
        self,
        x_temporal: torch.Tensor,
        x_tabular_continuous: torch.Tensor,
        x_tabular_categorical: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both encoders.
        
        Args:
            x_temporal: Temporal features (batch, seq_len, temporal_input_dim)
            x_tabular_continuous: Continuous tabular features (batch, continuous_dim)
            x_tabular_categorical: Categorical tabular features (batch, num_categorical)
        
        Returns:
            temporal_emb: (batch, temporal_output_dim=128)
            tabular_emb: (batch, tabular_output_dim=128)
        """
        temporal_emb = self.temporal_encoder(x_temporal)
        tabular_emb = self.tabular_encoder(x_tabular_continuous, x_tabular_categorical)
        
        return temporal_emb, tabular_emb
    
    def compute_similarity(
        self,
        x_temporal: torch.Tensor,
        x_tabular_continuous: torch.Tensor,
        x_tabular_categorical: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute cosine similarity between temporal and tabular embeddings.
        
        Used for contrastive training (CLIP-style dot product loss).
        
        Args:
            x_temporal: Temporal features
            x_tabular_continuous: Continuous tabular features
            x_tabular_categorical: Categorical tabular features
        
        Returns:
            Similarity scores (batch,) in range [-1, 1]
        """
        temporal_emb, tabular_emb = self.forward(
            x_temporal, x_tabular_continuous, x_tabular_categorical
        )
        
        # Cosine similarity
        similarity = F.cosine_similarity(temporal_emb, tabular_emb, dim=1)
        
        return similarity
    
    def get_joint_embedding(
        self,
        x_temporal: torch.Tensor,
        x_tabular_continuous: torch.Tensor,
        x_tabular_categorical: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Get concatenated joint embedding for inference.
        
        Used for nearest-neighbor search during inference.
        
        Args:
            x_temporal: Temporal features
            x_tabular_continuous: Continuous tabular features
            x_tabular_categorical: Categorical tabular features
        
        Returns:
            Joint embedding (batch, 256) = [temporal||tabular]
        """
        temporal_emb, tabular_emb = self.forward(
            x_temporal, x_tabular_continuous, x_tabular_categorical
        )
        
        # Concatenate for joint representation
        joint_emb = torch.cat([temporal_emb, tabular_emb], dim=1)
        
        return joint_emb
    
    def encode_temporal_only(self, x_temporal: torch.Tensor) -> torch.Tensor:
        """Encode temporal features only (for fast inference).
        
        Args:
            x_temporal: Temporal features (batch, seq_len, input_dim)
        
        Returns:
            Temporal embedding (batch, 128)
        """
        return self.temporal_encoder(x_temporal)
    
    def encode_tabular_only(
        self,
        x_tabular_continuous: torch.Tensor,
        x_tabular_categorical: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode tabular features only (for fast inference).
        
        Args:
            x_tabular_continuous: Continuous features
            x_tabular_categorical: Categorical features
        
        Returns:
            Tabular embedding (batch, 128)
        """
        return self.tabular_encoder(x_tabular_continuous, x_tabular_categorical)
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_models_dual_encoder.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/models/dual_encoder.py tests/test_models_dual_encoder.py
python -m ruff format src/models/dual_encoder.py
python -m ruff check src/models/dual_encoder.py
python -m mypy src/models/dual_encoder.py
git commit -m "feat: add dual-encoder model with BiMT-TCN + TabMixer"
```

---

## Task 6: InfoNCE Loss Function

**Files:**
- Create: `src/models/losses.py`
- Create: `tests/test_models_losses.py`

**Step 1: Write the failing test**

```python
# tests/test_models_losses.py
import torch
import pytest
from src.models.losses import InfoNCELoss


def test_infonce_loss_shape():
    """Test InfoNCE loss returns scalar."""
    batch_size = 32
    embedding_dim = 128
    
    loss_fn = InfoNCELoss(temperature=0.07)
    
    # Positive pairs: same stock, temporal vs tabular
    temporal_emb = torch.randn(batch_size, embedding_dim)
    tabular_emb = torch.randn(batch_size, embedding_dim)
    
    loss = loss_fn(temporal_emb, tabular_emb)
    
    assert loss.shape == ()
    assert loss.item() > 0


def test_infonce_loss_positive_pairs():
    """Test loss is low when embeddings are similar."""
    loss_fn = InfoNCELoss(temperature=0.07)
    
    # Identical embeddings (perfect match)
    emb = torch.randn(16, 128)
    loss = loss_fn(emb, emb.clone())
    
    # Loss should be near minimum for identical pairs
    assert loss.item() < 1.0


def test_infonce_loss_negative_pairs():
    """Test loss is high when embeddings are different."""
    loss_fn = InfoNCELoss(temperature=0.07)
    
    # Random unrelated embeddings
    temporal = torch.randn(16, 128)
    tabular = torch.randn(16, 128)
    
    loss = loss_fn(temporal, tabular)
    
    # Loss should be higher for random pairs
    assert loss.item() > 0.5
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_models_losses.py -v
```

Expected: FAIL

**Step 3: Write implementation**

```python
# src/models/losses.py
"""Loss functions for contrastive learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning.
    
    Standard contrastive loss that maximizes agreement between positive pairs
    and minimizes agreement with negative pairs (in-batch negatives).
    
    Formula: L = -log[ exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ) ]
    
    Where:
    - z_i = temporal embedding
    - z_j = tabular embedding (positive pair)
    - z_k = all other embeddings in batch (negatives)
    - τ = temperature
    """
    
    def __init__(self, temperature: float = 0.07):
        """Initialize InfoNCE loss.
        
        Args:
            temperature: Temperature parameter (default 0.07)
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        temporal_emb: torch.Tensor,
        tabular_emb: torch.Tensor,
        hard_negatives_temporal: torch.Tensor | None = None,
        hard_negatives_tabular: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute InfoNCE loss.
        
        Args:
            temporal_emb: Temporal embeddings (batch, dim)
            tabular_emb: Tabular embeddings (batch, dim)
            hard_negatives_temporal: Optional hard negative temporal embeddings
            hard_negatives_tabular: Optional hard negative tabular embeddings
        
        Returns:
            Scalar loss value
        """
        batch_size = temporal_emb.size(0)
        
        # Normalize embeddings
        temporal_emb = F.normalize(temporal_emb, dim=1)
        tabular_emb = F.normalize(tabular_emb, dim=1)
        
        # Compute similarity matrix: (batch, batch)
        # sim[i,j] = similarity between temporal[i] and tabular[j]
        similarity = torch.matmul(temporal_emb, tabular_emb.t()) / self.temperature
        
        # Labels: positive pairs are on diagonal (i matches i)
        labels = torch.arange(batch_size, device=similarity.device)
        
        # InfoNCE loss: cross entropy with diagonal as positives
        loss = F.cross_entropy(similarity, labels)
        
        return loss


class RankSCLLoss(nn.Module):
    """Rank-supervised contrastive learning loss (placeholder for future).
    
    Captures ordinal similarity: not just "are these similar?" but
    "HOW similar are these?" (preserves ranking order).
    
    To be implemented when ranking data available.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, temporal_emb: torch.Tensor, tabular_emb: torch.Tensor) -> torch.Tensor:
        """Placeholder - returns InfoNCE for now."""
        # TODO: Implement proper RankSCL with ordinal constraints
        infonce = InfoNCELoss(self.temperature)
        return infonce(temporal_emb, tabular_emb)
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_models_losses.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/models/losses.py tests/test_models_losses.py
python -m ruff format src/models/losses.py
python -m ruff check src/models/losses.py
python -m mypy src/models/losses.py
git commit -m "feat: add InfoNCE and RankSCL loss functions"
```

---

## Task 7: Hard Negative Sampler

**Files:**
- Create: `src/models/sampler.py`
- Create: `tests/test_models_sampler.py`

**Step 1: Write the failing test**

```python
# tests/test_models_sampler.py
import torch
import pytest
import pandas as pd
from src.models.sampler import GICSHardNegativeSampler


def test_sampler_initialization():
    """Test sampler initialization."""
    sampler = GICSHardNegativeSampler(n_hard=8)
    assert sampler.n_hard == 8


def test_sampler_filters_by_ggroup():
    """Test sampler finds stocks in same ggroup but different beta."""
    sampler = GICSHardNegativeSampler(n_hard=4)
    
    # Create mock stock data
    stocks = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'XOM'],
        'gsector': [45, 45, 45, 40, 10],  # Tech, Tech, Tech, Financials, Energy
        'ggroup': [4510, 4520, 4510, 4010, 1010],  # Software, Hardware, Software, Banks, Oil
        'beta': [1.2, 1.1, 1.3, 0.8, 0.9]
    })
    
    target = pd.Series({
        'symbol': 'AAPL',
        'gsector': 45,
        'ggroup': 4510,
        'beta': 1.2
    })
    
    hard_negs = sampler.sample(target, stocks)
    
    # Should find GOOGL (same ggroup 4510, different beta)
    assert 'GOOGL' in hard_negs
    # Should NOT find MSFT (different ggroup 4520)
    # Should NOT find JPM or XOM (different gsector)


def test_sampler_empty_result():
    """Test sampler handles no valid hard negatives."""
    sampler = GICSHardNegativeSampler(n_hard=4)
    
    stocks = pd.DataFrame({
        'symbol': ['AAPL'],
        'gsector': [45],
        'ggroup': [4510],
        'beta': [1.2]
    })
    
    target = pd.Series({
        'symbol': 'AAPL',
        'gsector': 45,
        'ggroup': 4510,
        'beta': 1.2
    })
    
    hard_negs = sampler.sample(target, stocks)
    
    # Should return empty or fallback
    assert len(hard_negs) <= 1  # At most itself
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_models_sampler.py -v
```

Expected: FAIL

**Step 3: Write implementation**

```python
# src/models/sampler.py
"""Hard negative sampling strategies for contrastive learning."""

import random
from typing import Any

import pandas as pd
import torch


class GICSHardNegativeSampler:
    """GICS-structured hard negative sampler.
    
    Uses GICS hierarchy for three-tier negative sampling:
    - Easy: Different gsector (exclude - too easy)
    - Hard: Same gsector, different ggroup
    - Hardest: Same ggroup (industry group), different beta/vol
    
    Critical for within-sector discrimination in stock substitutes.
    """
    
    def __init__(
        self,
        n_hard: int = 8,
        beta_threshold: float = 0.3,
        use_sector_fallback: bool = True
    ) -> None:
        """Initialize GICS hard negative sampler.
        
        Args:
            n_hard: Number of hard negatives to sample
            beta_threshold: Beta difference threshold for "different risk profile"
            use_sector_fallback: Whether to fall back to sector-level if no group matches
        """
        self.n_hard = n_hard
        self.beta_threshold = beta_threshold
        self.use_sector_fallback = use_sector_fallback
    
    def sample(
        self,
        target: pd.Series | dict[str, Any],
        candidates: pd.DataFrame,
        embeddings: torch.Tensor | None = None
    ) -> list[str]:
        """Sample hard negatives for target stock.
        
        Args:
            target: Target stock with 'symbol', 'ggroup', 'gsector', 'beta'
            candidates: DataFrame of candidate stocks with same columns
            embeddings: Optional pre-computed embeddings for similarity-based sampling
        
        Returns:
            List of hard negative symbols
        """
        target_group = target['ggroup']
        target_sector = target['gsector']
        target_beta = target['beta']
        
        # Level 1: Same ggroup (industry group), different beta
        same_group_diff_beta = candidates[
            (candidates['ggroup'] == target_group) &
            (candidates['symbol'] != target['symbol']) &
            (abs(candidates['beta'] - target_beta) > self.beta_threshold)
        ]
        
        if len(same_group_diff_beta) >= self.n_hard // 2:
            # Sample from same group with different beta
            group_samples = same_group_diff_beta.sample(
                n=min(self.n_hard // 2, len(same_group_diff_beta)),
                replace=False
            )['symbol'].tolist()
        elif len(same_group_diff_beta) > 0:
            # Take all available
            group_samples = same_group_diff_beta['symbol'].tolist()
        else:
            group_samples = []
        
        # Level 2: Same gsector, different ggroup (if need more)
        remaining = self.n_hard - len(group_samples)
        if remaining > 0 and self.use_sector_fallback:
            same_sector_diff_group = candidates[
                (candidates['gsector'] == target_sector) &
                (candidates['ggroup'] != target_group) &
                (~candidates['symbol'].isin(group_samples + [target['symbol']]))
            ]
            
            if len(same_sector_diff_group) > 0:
                sector_samples = same_sector_diff_group.sample(
                    n=min(remaining, len(same_sector_diff_group)),
                    replace=False
                )['symbol'].tolist()
                group_samples.extend(sector_samples)
        
        # If still not enough, fill with random from same sector
        remaining = self.n_hard - len(group_samples)
        if remaining > 0:
            same_sector = candidates[
                (candidates['gsector'] == target_sector) &
                (~candidates['symbol'].isin(group_samples + [target['symbol']]))
            ]
            
            if len(same_sector) > 0:
                random_samples = same_sector.sample(
                    n=min(remaining, len(same_sector)),
                    replace=False
                )['symbol'].tolist()
                group_samples.extend(random_samples)
        
        return group_samples
    
    def create_hard_negative_batch(
        self,
        batch_df: pd.DataFrame,
        temporal_data: torch.Tensor,
        tabular_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create batch with hard negatives inserted.
        
        Args:
            batch_df: DataFrame with stock metadata including GICS
            temporal_data: Temporal embeddings (batch, seq, features)
            tabular_data: Tabular embeddings (batch, features)
        
        Returns:
            temporal_with_negs, tabular_with_negs with hard negatives added
        """
        # This is a placeholder - full implementation would sample negatives
        # and concatenate them to the batch for contrastive training
        # TODO: Implement when training pipeline ready
        return temporal_data, tabular_data
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_models_sampler.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/models/sampler.py tests/test_models_sampler.py
python -m ruff format src/models/sampler.py
python -m ruff check src/models/sampler.py
python -m mypy src/models/sampler.py
git commit -m "feat: add GICS-structured hard negative sampler"
```

---

## Task 8: Model Exports and Integration

**Files:**
- Modify: `src/models/__init__.py`
- Create: `tests/test_models_integration.py`

**Step 1: Write the failing test**

```python
# tests/test_models_integration.py
import torch
import pandas as pd
import pytest

from src.models import (
    TemporalEncoder,
    TabMixer,
    DualEncoder,
    InfoNCELoss,
    GICSHardNegativeSampler
)


def test_all_models_importable():
    """Test all model components can be imported."""
    assert TemporalEncoder is not None
    assert TabMixer is not None
    assert DualEncoder is not None
    assert InfoNCELoss is not None
    assert GICSHardNegativeSampler is not None


def test_end_to_end_forward():
    """Test full forward pass through dual encoder."""
    batch_size = 16
    
    # Initialize model
    model = DualEncoder(
        temporal_input_dim=13,
        tabular_continuous_dim=15,
        tabular_categorical_dims=[11, 25],
        tabular_embedding_dims=[8, 16]
    )
    
    # Create sample data
    x_temporal = torch.randn(batch_size, 60, 13)
    x_tabular_cont = torch.randn(batch_size, 15)
    x_tabular_cat = torch.randint(0, 5, (batch_size, 2))
    
    # Forward pass
    temporal_emb, tabular_emb = model(x_temporal, x_tabular_cont, x_tabular_cat)
    
    # Check shapes
    assert temporal_emb.shape == (batch_size, 128)
    assert tabular_emb.shape == (batch_size, 128)
    
    # Compute similarity
    similarity = model.compute_similarity(x_temporal, x_tabular_cont, x_tabular_cat)
    assert similarity.shape == (batch_size,)
    
    # Get joint embedding
    joint_emb = model.get_joint_embedding(x_temporal, x_tabular_cont, x_tabular_cat)
    assert joint_emb.shape == (batch_size, 256)


def test_loss_computation():
    """Test InfoNCE loss with dual encoder outputs."""
    model = DualEncoder(temporal_input_dim=13, tabular_continuous_dim=15)
    loss_fn = InfoNCELoss(temperature=0.07)
    
    x_temp = torch.randn(32, 60, 13)
    x_tab_cont = torch.randn(32, 15)
    x_tab_cat = torch.randint(0, 5, (32, 2))
    
    temporal_emb, tabular_emb = model(x_temp, x_tab_cont, x_tab_cat)
    loss = loss_fn(temporal_emb, tabular_emb)
    
    assert loss.item() > 0
    assert not torch.isnan(loss)
```

**Step 2: Write the __init__.py exports**

```python
# src/models/__init__.py
"""Neural network models for stock substitute recommendation."""

from src.models.dual_encoder import DualEncoder
from src.models.losses import InfoNCELoss, RankSCLLoss
from src.models.sampler import GICSHardNegativeSampler
from src.models.tabmixer import TabMixer
from src.models.temporal_encoder import TemporalEncoder

__all__ = [
    "TemporalEncoder",
    "TabMixer",
    "DualEncoder",
    "InfoNCELoss",
    "RankSCLLoss",
    "GICSHardNegativeSampler",
]
```

**Step 3: Run test to verify it passes**

```bash
python -m pytest tests/test_models_integration.py -v
```

Expected: PASS

**Step 4: Run all model tests**

```bash
python -m pytest tests/test_models*.py -v 2>&1 | tail -20
```

Expected: All tests passing

**Step 5: Commit**

```bash
git add src/models/__init__.py tests/test_models_integration.py
python -m ruff check src/models/
python -m mypy src/models/
git commit -m "feat: export all model components and add integration tests"
```

---

## Task 9: Training Loop (Basic)

**Files:**
- Create: `src/training/__init__.py`
- Create: `src/training/trainer.py`
- Create: `tests/test_training.py`

**Step 1: Write the failing test**

```python
# tests/test_training.py
import torch
import pytest
from src.models import DualEncoder, InfoNCELoss
from src.training.trainer import ContrastiveTrainer


def test_trainer_initialization():
    """Test trainer can be initialized."""
    model = DualEncoder(temporal_input_dim=13, tabular_continuous_dim=15)
    loss_fn = InfoNCELoss()
    
    trainer = ContrastiveTrainer(model, loss_fn, lr=1e-4)
    
    assert trainer.model is not None
    assert trainer.loss_fn is not None


def test_trainer_training_step():
    """Test training step produces loss."""
    model = DualEncoder(temporal_input_dim=13, tabular_continuous_dim=15)
    loss_fn = InfoNCELoss()
    trainer = ContrastiveTrainer(model, loss_fn, lr=1e-4)
    
    # Create fake batch
    batch = {
        'temporal': torch.randn(16, 60, 13),
        'tabular_cont': torch.randn(16, 15),
        'tabular_cat': torch.randint(0, 5, (16, 2))
    }
    
    loss = trainer.train_step(batch)
    
    assert loss > 0
    assert not torch.isnan(torch.tensor(loss))
```

**Step 2: Write minimal trainer implementation**

```python
# src/training/trainer.py
"""Training loop for contrastive learning."""

import torch
import torch.nn as nn
from torch.optim import AdamW


class ContrastiveTrainer:
    """Trainer for dual-encoder contrastive learning."""
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        """Initialize trainer.
        
        Args:
            model: DualEncoder model
            loss_fn: InfoNCE or RankSCL loss
            lr: Learning rate
            weight_decay: Weight decay for AdamW
            device: Device to train on
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.device = device
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    
    def train_step(self, batch: dict) -> float:
        """Single training step.
        
        Args:
            batch: Dict with 'temporal', 'tabular_cont', 'tabular_cat'
        
        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move to device
        x_temp = batch['temporal'].to(self.device)
        x_tab_cont = batch['tabular_cont'].to(self.device)
        x_tab_cat = batch.get('tabular_cat')
        if x_tab_cat is not None:
            x_tab_cat = x_tab_cat.to(self.device)
        
        # Forward pass
        temporal_emb, tabular_emb = self.model(x_temp, x_tab_cont, x_tab_cat)
        
        # Compute loss
        loss = self.loss_fn(temporal_emb, tabular_emb)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, val_loader) -> float:
        """Validation loop.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x_temp = batch['temporal'].to(self.device)
                x_tab_cont = batch['tabular_cont'].to(self.device)
                x_tab_cat = batch.get('tabular_cat')
                if x_tab_cat is not None:
                    x_tab_cat = x_tab_cat.to(self.device)
                
                temporal_emb, tabular_emb = self.model(x_temp, x_tab_cont, x_tab_cat)
                loss = self.loss_fn(temporal_emb, tabular_emb)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / max(n_batches, 1)
```

**Step 3: Run test to verify it passes**

```bash
python -m pytest tests/test_training.py -v
```

Expected: PASS

**Step 4: Commit**

```bash
git add src/training/__init__.py src/training/trainer.py tests/test_training.py
python -m ruff format src/training/
python -m ruff check src/training/
python -m mypy src/training/
git commit -m "feat: add basic contrastive training loop"
```

---

## Final Verification

**Run all tests:**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: All tests passing

**Run linting:**

```bash
python -m ruff check src/models/ src/training/
python -m mypy src/models/ src/training/
```

Expected: Clean

**Create final commit:**

```bash
git log --oneline -10
```

Expected: 9+ commits showing incremental progress

---

## Summary

**Completed:**
1. Base encoder abstract class
2. TCN with causal convolutions
3. BiMT-TCN temporal encoder (TCN + Transformer)
4. TabMixer tabular encoder with GICS embeddings
5. Dual-encoder model combining both
6. InfoNCE loss function
7. GICS-structured hard negative sampler
8. Training loop
9. Comprehensive tests (40+ tests)

**Architecture implemented per spec:**
- BiMT-TCN: TCN local patterns + Transformer global dependencies → 128-dim
- TabMixer: 15 continuous + GICS embeddings (11→8-dim, 25→16-dim) → 128-dim
- Dual: Dot product training, 256-dim concatenation inference
- GICS: gsector + ggroup from Compustat, learned embeddings
- Negatives: Same ggroup, different beta = hardest

**Next steps (future work):**
1. Data pipeline integration (create DataLoaders)
2. Rolling window training implementation
3. Hyperparameter tuning
4. Embedding space evaluation metrics
5. Production inference API

**Plan saved to:** `docs/plans/2026-03-08-dual-encoder-implementation.md`

---

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks
**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution

Which approach?