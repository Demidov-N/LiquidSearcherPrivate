"""PyTorch Lightning Module for dual-encoder contrastive learning."""

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from src.models.dual_encoder import DualEncoder


class DualEncoderModule(pl.LightningModule):

    def __init__(
        self,
        temporal_input_dim: int = 13,
        tabular_continuous_dim: int = 15,
        tabular_categorical_dims: Optional[List[int]] = None,
        tabular_embedding_dims: Optional[List[int]] = None,
        embedding_dim: int = 128,
        temperature: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = DualEncoder(
            temporal_input_dim=temporal_input_dim,
            tabular_continuous_dim=tabular_continuous_dim,
            tabular_categorical_dims=tabular_categorical_dims or [11, 25],
            tabular_embedding_dims=tabular_embedding_dims or [8, 16],
            embedding_dim=embedding_dim,
            temperature=temperature,
        )

    def encode(self, batch: Dict[str, torch.Tensor]):
        """Return L2-normalised (temporal_emb, tabular_emb)."""
        temporal_emb = self.model.temporal_encoder(batch["temporal"])
        tabular_emb  = self.model.tabular_encoder(
            batch["tabular_cont"], batch["tabular_cat"]
        )
        temporal_emb = F.normalize(temporal_emb, dim=-1)
        tabular_emb  = F.normalize(tabular_emb,  dim=-1)
        return temporal_emb, tabular_emb

    def _info_nce(
        self,
        temporal_emb: torch.Tensor,
        tabular_emb:  torch.Tensor,
        symbols:      List[str],
    ) -> tuple[torch.Tensor, dict]:
        B   = len(symbols)
        tau = self.hparams.temperature

        sim = torch.matmul(temporal_emb, tabular_emb.t()) / tau  # (B, B)

        # False-negative mask — stays on GPU, no numpy round-trip
        sym_tensor = torch.tensor(
            [hash(s) for s in symbols], device=sim.device
        )
        same_sym  = sym_tensor[:, None] == sym_tensor[None, :]   # (B, B)
        eye       = torch.eye(B, dtype=torch.bool, device=sim.device)
        false_neg = same_sym & ~eye

        sim_masked = sim.clone()
        sim_masked[false_neg] = -1e9

        labels = torch.arange(B, device=sim.device)

        loss = (
            F.cross_entropy(sim_masked,    labels) +
            F.cross_entropy(sim_masked.t(), labels)
        ) / 2

        with torch.no_grad():
            pos_sim = sim.diagonal().mean().item()
            neg_sim = sim[~eye].mean().item()
            fn_frac = false_neg.float().sum().item() / max(1, B * (B - 1))

        return loss, {"alignment": pos_sim, "neg_sim": neg_sim, "fn_frac": fn_frac}

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        temporal_emb, tabular_emb = self.encode(batch)
        loss, metrics = self._info_nce(temporal_emb, tabular_emb, batch["symbol"])

        self.log("train/loss",      loss,                  on_step=True,  on_epoch=True, prog_bar=True)
        self.log("train/alignment", metrics["alignment"],  on_step=False, on_epoch=True)
        self.log("train/neg_sim",   metrics["neg_sim"],    on_step=False, on_epoch=True)
        self.log("train/fn_frac",   metrics["fn_frac"],    on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        temporal_emb, tabular_emb = self.encode(batch)
        loss, metrics = self._info_nce(temporal_emb, tabular_emb, batch["symbol"])

        self.log("val/loss",      loss,                  on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/alignment", metrics["alignment"],  on_step=False, on_epoch=True)
        self.log("val/neg_sim",   metrics["neg_sim"],    on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        warmup = self.hparams.warmup_epochs
        total  = self.hparams.max_epochs

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return epoch / max(1, warmup)
            progress = (epoch - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def get_joint_embeddings(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Concatenated (temporal ‖ tabular) embeddings for downstream tasks."""
        self.eval()
        with torch.no_grad():
            temporal_emb, tabular_emb = self.encode(batch)
        return torch.cat([temporal_emb, tabular_emb], dim=-1)