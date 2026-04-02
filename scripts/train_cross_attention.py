"""Training script for Cross-Attention Dual-Encoder model.

This script trains the CrossAttentionDualEncoder with learned cross-modal fusion
(cross-attention + gating), saving checkpoints to checkpoints_cross/ directory.

Example:
    # Basic training
    uv run python -m scripts.train_cross_attention \
        --epochs 100 \
        --batch-size 32 \
        --lr 1e-4

    # With auxiliary losses (sector + liquidity prediction)
    uv run python -m scripts.train_cross_attention \
        --use-auxiliary-losses \
        --aux-loss-weight 0.1 \
        --epochs 100

    # Different architecture parameters
    uv run python -m scripts.train_cross_attention \
        --num-heads 8 \
        --dropout 0.2 \
        --no-gating  # Disable gated fusion

    # Resume from checkpoint
    uv run python -m scripts.train_cross_attention \
        --resume-from-last
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import CSVLogger

from src.training.data_module import StockDataModule
from src.training.cross_attention_module import CrossAttentionDualEncoderModule


def parse_args():
    parser = argparse.ArgumentParser(description="Train cross-attention dual-encoder model")

    # Data parameters
    parser.add_argument("--feature-dir", type=str, default="data/processed/features")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--train-start", type=str, default="2010-01-01")
    parser.add_argument("--train-end", type=str, default="2018-12-31")
    parser.add_argument("--val-start", type=str, default="2020-01-01")
    parser.add_argument("--val-end", type=str, default="2020-12-31")

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=None,
        help="Subsample training set each epoch (None = full dataset)",
    )

    # Model architecture parameters
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num-heads", type=int, default=4, help="Cross-attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--no-gating", action="store_true", help="Disable gated fusion")

    # Auxiliary losses
    parser.add_argument(
        "--use-auxiliary-losses",
        action="store_true",
        help="Train with auxiliary tasks (sector + liquidity)",
    )
    parser.add_argument(
        "--aux-loss-weight", type=float, default=0.1, help="Weight for auxiliary losses"
    )

    # Checkpoint parameters
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints_cross",
        help="Directory to save checkpoints (default: checkpoints_cross)",
    )

    # Resume functionality
    parser.add_argument(
        "--resume-from", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--resume-from-last",
        action="store_true",
        help="Resume from the last.ckpt checkpoint in checkpoint-dir",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"cross_attention_{timestamp}"

    # Setup logging
    logs_dir = Path("logs") / experiment_name
    logs_dir.mkdir(parents=True, exist_ok=True)

    # File logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / "training.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Training: {args.train_start} to {args.train_end}")
    logger.info(f"Validation: {args.val_start} to {args.val_end}")
    logger.info(f"Checkpoint dir: {args.checkpoint_dir}")
    logger.info(
        f"Architecture: {args.num_heads} heads, dropout={args.dropout}, gating={not args.no_gating}"
    )
    logger.info(f"Auxiliary losses: {args.use_auxiliary_losses}")
    logger.info(f"Logging to: {logs_dir / 'training.log'}")

    print(f"Experiment: {experiment_name}")
    print(f"Training: {args.train_start} to {args.train_end}")
    print(f"Validation: {args.val_start} to {args.val_end}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(
        f"Architecture: {args.num_heads} heads, dropout={args.dropout}, gating={not args.no_gating}"
    )
    print(f"Auxiliary losses: {args.use_auxiliary_losses}")
    print(f"Logging to: {logs_dir / 'training.log'}")

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
        samples_per_epoch=args.samples_per_epoch,
    )

    # Create model module with cross-attention
    model_module = CrossAttentionDualEncoderModule(
        temporal_input_dim=13,
        tabular_continuous_dim=15,
        embedding_dim=args.embedding_dim,
        temperature=args.temperature,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_gating=not args.no_gating,
        use_auxiliary_losses=args.use_auxiliary_losses,
        aux_loss_weight=args.aux_loss_weight,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
    )

    # Setup checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{experiment_name}" + "-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,  # Always save the last checkpoint as last.ckpt
        ),
        EarlyStopping(monitor="val/loss", patience=15, mode="min"),
        TQDMProgressBar(refresh_rate=10),
    ]

    # CSV Logger for metrics
    csv_logger = CSVLogger(logs_dir, name="", version=0)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=csv_logger,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        enable_progress_bar=True,
    )

    # Handle resume functionality
    ckpt_path = None
    if args.resume_from:
        ckpt_path = args.resume_from
        if not Path(ckpt_path).exists():
            logger.error(f"Checkpoint file not found: {ckpt_path}")
            return
        logger.info(f"Resuming from specified checkpoint: {ckpt_path}")
        print(f"Resuming from specified checkpoint: {ckpt_path}")
    elif args.resume_from_last:
        last_ckpt = checkpoint_dir / "last.ckpt"
        if last_ckpt.exists():
            ckpt_path = str(last_ckpt)
            logger.info(f"Resuming from last checkpoint: {ckpt_path}")
            print(f"Resuming from last checkpoint: {ckpt_path}")
        else:
            logger.warning(f"--resume-from-last specified but {last_ckpt} not found")
            logger.warning("Starting fresh training...")
            print(f"Warning: --resume-from-last specified but {last_ckpt} not found")
            print("Starting fresh training...")

    print("\nStarting cross-attention model training...")
    print("=" * 70)
    trainer.fit(model_module, data_module, ckpt_path=ckpt_path)

    print("\nTraining complete!")
    print(f"Best checkpoints saved to: {checkpoint_dir}")
    print(f"Final model (last.ckpt): {checkpoint_dir / 'last.ckpt'}")


if __name__ == "__main__":
    main()
