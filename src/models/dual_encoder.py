"""Dual-encoder: BiMT-TCN + TabMixer with CLIP-style contrastive learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from info_nce import InfoNCE

from src.models.temporal_encoder import TemporalEncoder
from src.models.tabular_encoder import TabularEncoder


class DualEncoder(nn.Module):
    """Dual-encoder for stock substitute recommendation.

    Architecture:
    - Temporal Encoder (BiMT-TCN): 60-day OHLCV → 128-dim
    - Tabular Encoder (TabMixer): fundamentals + GICS → 128-dim
    - Training: InfoNCE loss aligns temporal and tabular of same stock
    - Inference: Concatenate → 256-dim for similarity search

    Args:
        temporal_input_dim: Price features (default 13)
        tabular_continuous_dim: Continuous fundamentals (default 15)
        tabular_categorical_dims: Categorical cardinalities (default [11, 25])
        tabular_embedding_dims: Embedding dims (default [8, 16])
        embedding_dim: Encoder output dim (default 128)
        temperature: InfoNCE temperature (default 0.07)

    Example:
        model = DualEncoder()

        # Training
        price_data = torch.randn(32, 60, 13)
        fundamentals = torch.randn(32, 15)
        categorical = torch.randint(0, 11, (32, 2))

        loss, temp_emb, tab_emb = model(
            price_data, fundamentals, categorical, mode='train'
        )

        # Inference (similarity search)
        model.eval()
        joint_emb = model.get_joint_embedding(
            price_data, fundamentals, categorical
        )  # (32, 256)

        similarities = F.cosine_similarity(joint_emb[0:1], joint_emb)
        top_10 = torch.topk(similarities, k=10)
    """

    def __init__(
        self,
        temporal_input_dim=13,
        tabular_continuous_dim=15,
        tabular_categorical_dims=None,
        tabular_embedding_dims=None,
        embedding_dim=128,
        temperature=0.07,
    ):
        super().__init__()

        if tabular_categorical_dims is None:
            tabular_categorical_dims = [11, 25]
        if tabular_embedding_dims is None:
            tabular_embedding_dims = [8, 16]

        self.embedding_dim = embedding_dim

        # Temporal encoder: BiMT-TCN
        self.temporal_encoder = TemporalEncoder(
            input_dim=temporal_input_dim, output_dim=embedding_dim
        )

        # Tabular encoder: TabMixer
        self.tabular_encoder = TabularEncoder(
            continuous_dim=tabular_continuous_dim,
            categorical_dims=tabular_categorical_dims,
            embedding_dims=tabular_embedding_dims,
            output_dim=embedding_dim,
        )

        # InfoNCE loss for contrastive learning
        self.infonce_loss = InfoNCE(temperature=temperature)

    def forward(self, price_data, fundamentals, categorical, mode="train"):
        """
        Args:
            price_data: (batch, 60, 13) - OHLCV features
            fundamentals: (batch, 15) - continuous features
            categorical: (batch, 2) - [gsector, ggroup] indices
            mode: 'train' or 'inference'

        Returns:
            train: (loss, temporal_emb, tabular_emb)
            inference: joint_emb (batch, 256)
        """
        temporal_emb = self.temporal_encoder(price_data)
        tabular_emb = self.tabular_encoder(fundamentals, categorical)

        if mode == "train":
            loss = self.infonce_loss(temporal_emb, tabular_emb)
            return loss, temporal_emb, tabular_emb

        elif mode == "inference":
            joint_emb = torch.cat([temporal_emb, tabular_emb], dim=-1)
            return joint_emb

        else:
            raise ValueError(f"Invalid mode: {mode}")

    def get_joint_embedding(self, price_data, fundamentals, categorical):
        """Convenience method for inference."""
        return self.forward(price_data, fundamentals, categorical, mode="inference")

    def get_temporal_embedding(self, price_data):
        """Get only temporal embedding for inference.

        Args:
            price_data: (batch, 60, 13) OHLCV features

        Returns:
            (batch, 128) temporal embeddings
        """
        self.eval()
        with torch.no_grad():
            return self.temporal_encoder(price_data)

    def get_tabular_embedding(self, fundamentals, categorical):
        """Get only tabular embedding for inference.

        Args:
            fundamentals: (batch, 15) continuous features
            categorical: (batch, 2) [gsector, ggroup] indices

        Returns:
            (batch, 128) tabular embeddings
        """
        self.eval()
        with torch.no_grad():
            return self.tabular_encoder(fundamentals, categorical)

    def compute_similarity(self, query_emb, candidate_emb):
        """Compute cosine similarity."""
        return F.cosine_similarity(query_emb, candidate_emb, dim=-1)
