"""Base encoder class."""

from abc import ABC, abstractmethod

import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for all encoders.

    All encoders must implement:
    - forward(x): Input tensor → output embedding

    Attributes:
        input_dim: Input feature dimension
        output_dim: Output embedding dimension
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
    def forward(self, x):
        """Forward pass. Must be implemented by subclass."""
        pass
