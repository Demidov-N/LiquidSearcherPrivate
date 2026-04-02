"""Cross-Attention Dual-Encoder: Learned fusion between temporal and tabular modalities."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from info_nce import InfoNCE

from src.models.dual_encoder import DualEncoder


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion layer for combining temporal and tabular embeddings.

    Each modality attends to the other, allowing information exchange between
    price/volume patterns and fundamental features.

    Architecture:
    - Multi-head cross-attention (temporal → tabular and tabular → temporal)
    - Feed-forward network after attention
    - Residual connections
    """

    def __init__(self, embedding_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Cross-attention: Temporal attends to Tabular
        self.temporal_cross_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention: Tabular attends to Temporal
        self.tabular_cross_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward networks
        self.temporal_ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

        self.tabular_ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

        # Layer normalization
        self.temporal_norm1 = nn.LayerNorm(embedding_dim)
        self.temporal_norm2 = nn.LayerNorm(embedding_dim)
        self.tabular_norm1 = nn.LayerNorm(embedding_dim)
        self.tabular_norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, temporal_emb, tabular_emb):
        """
        Args:
            temporal_emb: (batch, 128) - temporal features
            tabular_emb: (batch, 128) - tabular features

        Returns:
            temporal_fused: (batch, 128) - temporal with tabular context
            tabular_fused: (batch, 128) - tabular with temporal context
        """
        # Add sequence dimension for attention (treat as single-token sequences)
        temporal_seq = temporal_emb.unsqueeze(1)  # (batch, 1, 128)
        tabular_seq = tabular_emb.unsqueeze(1)  # (batch, 1, 128)

        # Temporal attends to Tabular
        temporal_attended, _ = self.temporal_cross_attn(temporal_seq, tabular_seq, tabular_seq)
        temporal_attended = temporal_attended.squeeze(1)  # (batch, 128)
        temporal_fused = self.temporal_norm1(temporal_emb + temporal_attended)

        # Feed-forward on temporal
        temporal_ffn_out = self.temporal_ffn(temporal_fused)
        temporal_fused = self.temporal_norm2(temporal_fused + temporal_ffn_out)

        # Tabular attends to Temporal
        tabular_attended, _ = self.tabular_cross_attn(tabular_seq, temporal_seq, temporal_seq)
        tabular_attended = tabular_attended.squeeze(1)  # (batch, 128)
        tabular_fused = self.tabular_norm1(tabular_emb + tabular_attended)

        # Feed-forward on tabular
        tabular_ffn_out = self.tabular_ffn(tabular_fused)
        tabular_fused = self.tabular_norm2(tabular_fused + tabular_ffn_out)

        return temporal_fused, tabular_fused


class GatedFusion(nn.Module):
    """Gated fusion mechanism to control information flow between modalities.

    Learns a dynamic gate to balance temporal vs tabular information.
    """

    def __init__(self, embedding_dim=128):
        super().__init__()
        self.gate_layer = nn.Sequential(nn.Linear(embedding_dim * 2, embedding_dim), nn.Sigmoid())

    def forward(self, temporal_emb, tabular_emb):
        """
        Args:
            temporal_emb: (batch, 128)
            tabular_emb: (batch, 128)

        Returns:
            fused_emb: (batch, 128) - gated combination
        """
        concat = torch.cat([temporal_emb, tabular_emb], dim=-1)
        gate = self.gate_layer(concat)  # (batch, 128)

        # Weighted combination
        fused = gate * temporal_emb + (1 - gate) * tabular_emb
        return fused


class CrossAttentionDualEncoder(nn.Module):
    """Dual-encoder with learned cross-modal fusion.

    Architecture:
    1. Temporal Encoder (BiMT-TCN): 60-day OHLCV → 128-dim
    2. Tabular Encoder (TabMixer): fundamentals + GICS → 128-dim
    3. Cross-Attention Fusion: exchange information between modalities
    4. Gated Fusion: dynamically combine attended embeddings
    5. Training: InfoNCE loss + optional auxiliary tasks

    Improvements over simple concatenation:
    - Cross-attention allows temporal features to query tabular context
    - Gating learns dynamic importance weighting
    - Fused embedding captures cross-modal patterns

    Args:
        temporal_input_dim: Price features (default 13)
        tabular_continuous_dim: Continuous fundamentals (default 15)
        tabular_categorical_dims: Categorical cardinalities (default [11, 25])
        tabular_embedding_dims: Embedding dims (default [8, 16])
        embedding_dim: Encoder output dim (default 128)
        temperature: InfoNCE temperature (default 0.07)
        num_heads: Cross-attention heads (default 4)
        dropout: Dropout rate (default 0.1)
        use_gating: Whether to use gated fusion (default True)

    Example:
        model = CrossAttentionDualEncoder()

        # Training
        price_data = torch.randn(32, 60, 13)
        fundamentals = torch.randn(32, 15)
        categorical = torch.randint(0, 11, (32, 2))

        loss, outputs = model(price_data, fundamentals, categorical, mode='train')

        # Inference
        model.eval()
        joint_emb = model.get_joint_embedding(price_data, fundamentals, categorical)
        # joint_emb: (32, 256) - fused representation
    """

    def __init__(
        self,
        temporal_input_dim=13,
        tabular_continuous_dim=15,
        tabular_categorical_dims=None,
        tabular_embedding_dims=None,
        embedding_dim=128,
        temperature=0.07,
        num_heads=4,
        dropout=0.1,
        use_gating=True,
    ):
        super().__init__()

        if tabular_categorical_dims is None:
            tabular_categorical_dims = [11, 25]
        if tabular_embedding_dims is None:
            tabular_embedding_dims = [8, 16]

        self.embedding_dim = embedding_dim
        self.use_gating = use_gating

        # Base encoders
        self.temporal_encoder = DualEncoder(
            temporal_input_dim=temporal_input_dim,
            tabular_continuous_dim=tabular_continuous_dim,
            tabular_categorical_dims=tabular_categorical_dims,
            tabular_embedding_dims=tabular_embedding_dims,
            embedding_dim=embedding_dim,
            temperature=temperature,
        ).temporal_encoder

        self.tabular_encoder = DualEncoder(
            temporal_input_dim=temporal_input_dim,
            tabular_continuous_dim=tabular_continuous_dim,
            tabular_categorical_dims=tabular_categorical_dims,
            tabular_embedding_dims=tabular_embedding_dims,
            embedding_dim=embedding_dim,
            temperature=temperature,
        ).tabular_encoder

        # Cross-attention fusion
        self.cross_attention = CrossAttentionFusion(
            embedding_dim=embedding_dim, num_heads=num_heads, dropout=dropout
        )

        # Optional gating
        if use_gating:
            self.gated_fusion = GatedFusion(embedding_dim=embedding_dim)

        # Projection head for contrastive learning
        self.temporal_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.tabular_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # InfoNCE loss
        self.infonce_loss = InfoNCE(temperature=temperature)

        # Optional auxiliary heads (can be trained with auxiliary losses)
        self.sector_classifier = nn.Linear(embedding_dim * 2, 11)  # 11 GICS sectors
        self.liquidity_predictor = nn.Linear(embedding_dim * 2, 1)  # Liquidity score

    def forward(self, price_data, fundamentals, categorical, mode="train"):
        """
        Args:
            price_data: (batch, 60, 13) - OHLCV features
            fundamentals: (batch, 15) - continuous features
            categorical: (batch, 2) - [gsector, ggroup] indices
            mode: 'train', 'inference', or 'auxiliary'

        Returns:
            train: (loss, dict with embeddings)
            inference: joint_emb (batch, 256)
            auxiliary: (loss, dict with all predictions)
        """
        # Encode modalities
        temporal_emb = self.temporal_encoder(price_data)  # (batch, 128)
        tabular_emb = self.tabular_encoder(fundamentals, categorical)  # (batch, 128)

        # Cross-attention fusion
        temporal_fused, tabular_fused = self.cross_attention(temporal_emb, tabular_emb)

        # Gated or simple fusion
        if self.use_gating:
            fused_emb = self.gated_fusion(temporal_fused, tabular_fused)
        else:
            fused_emb = temporal_fused + tabular_fused  # Residual-style

        # Joint embedding for similarity search
        joint_emb = torch.cat([temporal_fused, tabular_fused], dim=-1)  # (batch, 256)

        if mode == "train":
            # Project for contrastive learning
            temp_proj = self.temporal_projection(temporal_fused)
            tab_proj = self.tabular_projection(tabular_fused)

            # InfoNCE loss between fused modalities
            loss = self.infonce_loss(temp_proj, tab_proj)

            outputs = {
                "temporal_emb": temporal_emb,
                "tabular_emb": tabular_emb,
                "temporal_fused": temporal_fused,
                "tabular_fused": tabular_fused,
                "fused_emb": fused_emb,
                "joint_emb": joint_emb,
            }
            return loss, outputs

        elif mode == "inference":
            return joint_emb

        elif mode == "auxiliary":
            # Auxiliary task predictions
            sector_logits = self.sector_classifier(joint_emb)
            liquidity_pred = self.liquidity_predictor(joint_emb)

            outputs = {
                "temporal_emb": temporal_emb,
                "tabular_emb": tabular_emb,
                "joint_emb": joint_emb,
                "sector_logits": sector_logits,
                "liquidity_pred": liquidity_pred,
            }
            return outputs

        else:
            raise ValueError(f"Invalid mode: {mode}")

    def get_joint_embedding(self, price_data, fundamentals, categorical):
        """Get joint embedding for inference (similarity search)."""
        self.eval()
        with torch.no_grad():
            return self.forward(price_data, fundamentals, categorical, mode="inference")

    def get_temporal_embedding(self, price_data):
        """Get temporal embedding only."""
        self.eval()
        with torch.no_grad():
            return self.temporal_encoder(price_data)

    def get_tabular_embedding(self, fundamentals, categorical):
        """Get tabular embedding only."""
        self.eval()
        with torch.no_grad():
            return self.tabular_encoder(fundamentals, categorical)

    def get_fused_embeddings(self, price_data, fundamentals, categorical):
        """Get all embeddings including fused versions.

        Returns:
            dict with temporal, tabular, fused, and joint embeddings
        """
        self.eval()
        with torch.no_grad():
            temporal_emb = self.temporal_encoder(price_data)
            tabular_emb = self.tabular_encoder(fundamentals, categorical)
            temporal_fused, tabular_fused = self.cross_attention(temporal_emb, tabular_emb)

            if self.use_gating:
                fused_emb = self.gated_fusion(temporal_fused, tabular_fused)
            else:
                fused_emb = temporal_fused + tabular_fused

            joint_emb = torch.cat([temporal_fused, tabular_fused], dim=-1)

            return {
                "temporal": temporal_emb,
                "tabular": tabular_emb,
                "temporal_fused": temporal_fused,
                "tabular_fused": tabular_fused,
                "fused": fused_emb,
                "joint": joint_emb,
            }

    def compute_similarity(self, query_emb, candidate_emb, use_fused=False):
        """Compute cosine similarity.

        Args:
            query_emb: (batch, dim) query embeddings
            candidate_emb: (batch, dim) candidate embeddings
            use_fused: if True, use 128-dim fused embeddings; else use 256-dim joint
        """
        if use_fused and query_emb.shape[-1] == 128:
            return F.cosine_similarity(query_emb, candidate_emb, dim=-1)
        else:
            return F.cosine_similarity(query_emb, candidate_emb, dim=-1)
