"""PyTorch Lightning Module for cross-attention dual-encoder with auxiliary losses."""

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from src.models.cross_attention_dual_encoder import CrossAttentionDualEncoder


class CrossAttentionDualEncoderModule(pl.LightningModule):
    """PyTorch Lightning module for training CrossAttentionDualEncoder.

    Training objectives:
    1. Primary: InfoNCE contrastive loss (temporal ↔ tabular alignment)
    2. Auxiliary (optional):
       - Sector classification loss (predict GICS sector)
       - Liquidity prediction loss (predict liquidity score)

    Args:
        temporal_input_dim: Price features (default 13)
        tabular_continuous_dim: Continuous fundamentals (default 15)
        embedding_dim: Encoder output dim (default 128)
        temperature: InfoNCE temperature (default 0.1)
        num_heads: Cross-attention heads (default 4)
        dropout: Dropout rate (default 0.1)
        use_gating: Whether to use gated fusion (default True)
        use_auxiliary_losses: Whether to train with auxiliary tasks (default False)
        aux_loss_weight: Weight for auxiliary losses (default 0.1)
        lr: Learning rate (default 1e-4)
        weight_decay: Weight decay (default 0.01)
        warmup_epochs: Warmup epochs (default 5)
        max_epochs: Total epochs (default 100)
    """

    def __init__(
        self,
        temporal_input_dim: int = 13,
        tabular_continuous_dim: int = 15,
        tabular_categorical_dims: Optional[List[int]] = None,
        tabular_embedding_dims: Optional[List[int]] = None,
        embedding_dim: int = 128,
        temperature: float = 0.1,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_gating: bool = True,
        use_auxiliary_losses: bool = False,
        aux_loss_weight: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = CrossAttentionDualEncoder(
            temporal_input_dim=temporal_input_dim,
            tabular_continuous_dim=tabular_continuous_dim,
            tabular_categorical_dims=tabular_categorical_dims or [11, 25],
            tabular_embedding_dims=tabular_embedding_dims or [8, 16],
            embedding_dim=embedding_dim,
            temperature=temperature,
            num_heads=num_heads,
            dropout=dropout,
            use_gating=use_gating,
        )

        self.use_auxiliary = use_auxiliary_losses
        self.aux_weight = aux_loss_weight

    def forward(self, batch: Dict[str, torch.Tensor], mode: str = "train"):
        """Forward pass through the model.

        Args:
            batch: Dictionary with temporal, tabular_cont, tabular_cat, symbol
            mode: 'train', 'inference', or 'auxiliary'

        Returns:
            loss and metrics (train), or embeddings (inference)
        """
        price_data = batch["temporal"]
        fundamentals = batch["tabular_cont"]
        categorical = batch["tabular_cat"]

        if mode == "train":
            loss, outputs = self.model(price_data, fundamentals, categorical, mode="train")
            return loss, outputs
        elif mode == "inference":
            return self.model(price_data, fundamentals, categorical, mode="inference")
        elif mode == "auxiliary":
            return self.model(price_data, fundamentals, categorical, mode="auxiliary")
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def encode(self, batch: Dict[str, torch.Tensor]):
        """Return L2-normalised embeddings (temporal, tabular, fused, joint)."""
        # Get fused embeddings
        embeddings = self.model.get_fused_embeddings(
            batch["temporal"], batch["tabular_cont"], batch["tabular_cat"]
        )

        # Normalize for contrastive learning
        temporal_emb = F.normalize(embeddings["temporal"], dim=-1)
        tabular_emb = F.normalize(embeddings["tabular"], dim=-1)
        temporal_fused = F.normalize(embeddings["temporal_fused"], dim=-1)
        tabular_fused = F.normalize(embeddings["tabular_fused"], dim=-1)
        fused_emb = F.normalize(embeddings["fused"], dim=-1)
        joint_emb = F.normalize(embeddings["joint"], dim=-1)

        return {
            "temporal": temporal_emb,
            "tabular": tabular_emb,
            "temporal_fused": temporal_fused,
            "tabular_fused": tabular_fused,
            "fused": fused_emb,
            "joint": joint_emb,
        }

    def _info_nce(
        self,
        temporal_emb: torch.Tensor,
        tabular_emb: torch.Tensor,
        symbols: List[str],
    ) -> tuple[torch.Tensor, dict]:
        """Compute InfoNCE loss with false-negative masking."""
        B = len(symbols)
        tau = self.hparams.temperature

        # Similarity matrix (B, B)
        sim = torch.matmul(temporal_emb, tabular_emb.t()) / tau

        # False-negative mask — same symbol different augmentations
        sym_tensor = torch.tensor([hash(s) for s in symbols], device=sim.device)
        same_sym = sym_tensor[:, None] == sym_tensor[None, :]
        eye = torch.eye(B, dtype=torch.bool, device=sim.device)
        false_neg = same_sym & ~eye

        sim_masked = sim.clone()
        sim_masked[false_neg] = -1e9

        labels = torch.arange(B, device=sim.device)

        loss = (F.cross_entropy(sim_masked, labels) + F.cross_entropy(sim_masked.t(), labels)) / 2

        with torch.no_grad():
            pos_sim = sim.diagonal().mean().item()
            neg_sim = sim[~eye].mean().item()
            fn_frac = false_neg.float().sum().item() / max(1, B * (B - 1))

        return loss, {"alignment": pos_sim, "neg_sim": neg_sim, "fn_frac": fn_frac}

    def _auxiliary_losses(
        self, batch: Dict[str, Any], outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute auxiliary task losses.

        Optional tasks:
        - Sector classification (GICS sector prediction)
        - Liquidity prediction (liquidity score regression)
        """
        losses = {}

        # Get auxiliary predictions
        aux_outputs = self.model(
            batch["temporal"], batch["tabular_cont"], batch["tabular_cat"], mode="auxiliary"
        )

        # Sector classification loss
        if "sector" in batch and "sector_logits" in aux_outputs:
            sector_loss = F.cross_entropy(aux_outputs["sector_logits"], batch["sector"])
            losses["sector"] = sector_loss

        # Liquidity prediction loss (MSE)
        if "liquidity_score" in batch and "liquidity_pred" in aux_outputs:
            liq_loss = F.mse_loss(aux_outputs["liquidity_pred"].squeeze(), batch["liquidity_score"])
            losses["liquidity"] = liq_loss

        return losses

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """Training step with InfoNCE + optional auxiliary losses."""
        # Primary InfoNCE loss
        loss, outputs = self.forward(batch, mode="train")

        # Use fused embeddings for contrastive loss
        temporal_emb = outputs["temporal_fused"]
        tabular_emb = outputs["tabular_fused"]

        loss_info_nce, metrics = self._info_nce(temporal_emb, tabular_emb, batch["symbol"])

        total_loss = loss_info_nce

        # Auxiliary losses (if enabled and data available)
        if self.use_auxiliary:
            aux_losses = self._auxiliary_losses(batch, outputs)
            for name, aux_loss in aux_losses.items():
                total_loss = total_loss + self.aux_weight * aux_loss
                self.log(f"train/aux_{name}_loss", aux_loss, on_step=False, on_epoch=True)

        # Logging
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/info_nce", loss_info_nce, on_step=False, on_epoch=True)
        self.log("train/alignment", metrics["alignment"], on_step=False, on_epoch=True)
        self.log("train/neg_sim", metrics["neg_sim"], on_step=False, on_epoch=True)
        self.log("train/fn_frac", metrics["fn_frac"], on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """Validation step."""
        # Primary InfoNCE loss
        loss, outputs = self.forward(batch, mode="train")

        # Use fused embeddings
        temporal_emb = outputs["temporal_fused"]
        tabular_emb = outputs["tabular_fused"]

        loss_info_nce, metrics = self._info_nce(temporal_emb, tabular_emb, batch["symbol"])

        total_loss = loss_info_nce

        # Auxiliary losses (validation)
        if self.use_auxiliary:
            aux_losses = self._auxiliary_losses(batch, outputs)
            for name, aux_loss in aux_losses.items():
                total_loss = total_loss + self.aux_weight * aux_loss
                self.log(f"val/aux_{name}_loss", aux_loss, on_step=False, on_epoch=True)

        # Logging
        self.log("val/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/info_nce", loss_info_nce, on_step=False, on_epoch=True)
        self.log("val/alignment", metrics["alignment"], on_step=False, on_epoch=True)
        self.log("val/neg_sim", metrics["neg_sim"], on_step=False, on_epoch=True)

        return total_loss

    def configure_optimizers(self):
        """Configure optimizer with cosine annealing + warmup."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        warmup = self.hparams.warmup_epochs
        total = self.hparams.max_epochs

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return epoch / max(1, warmup)
            progress = (epoch - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def get_joint_embeddings(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Get joint embeddings for downstream tasks."""
        self.eval()
        with torch.no_grad():
            return self.forward(batch, mode="inference")

    def get_fused_embeddings(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Get all embeddings including fused versions."""
        self.eval()
        with torch.no_grad():
            embeddings = self.model.get_fused_embeddings(
                batch["temporal"], batch["tabular_cont"], batch["tabular_cat"]
            )
            # Normalize all
            return {k: F.normalize(v, dim=-1) for k, v in embeddings.items()}
