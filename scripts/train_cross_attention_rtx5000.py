"""Optimized training script for RTX A5000 (24GB VRAM) + 100GB RAM.

This configuration maximizes GPU utilization and training robustness.

Example:
    # Full training with all optimizations
    uv run python -m scripts.train_cross_attention_rtx5000 --epochs 200

    # Quick test run
    uv run python -m scripts.train_cross_attention_rtx5000 --epochs 10 --batch-size 64
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
    LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src.training.data_module import StockDataModule
from src.training.cross_attention_module import CrossAttentionDualEncoderModule


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train cross-attention model optimized for RTX A5000"
    )

    # Data parameters
    parser.add_argument("--feature-dir", type=str, default="data/processed/features")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--train-start", type=str, default="2010-01-01")
    parser.add_argument("--train-end", type=str, default="2018-12-31")
    parser.add_argument("--val-start", type=str, default="2020-01-01")
    parser.add_argument("--val-end", type=str, default="2020-12-31")

    # ===============================
    # RTX A5000 OPTIMIZED PARAMETERS
    # ===============================

    # Batch size: 24GB VRAM allows 128-256 batch size easily
    # With cross-attention (more memory), use 128 for safety
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size (128-256 for RTX A5000)"
    )

    # Workers: 100GB RAM allows many workers
    # Set to 8-16 for fast data loading
    parser.add_argument(
        "--num-workers", type=int, default=12, help="Data loading workers (8-16 for 100GB RAM)"
    )

    # Enable pin_memory for faster CPU→GPU transfer
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        default=True,
        help="Pin memory for faster data transfer",
    )

    # ===============================
    # TRAINING ROBUSTNESS
    # ===============================

    parser.add_argument("--epochs", type=int, default=200, help="More epochs for convergence")
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Higher LR with larger batches (LR ∝ sqrt(batch_size))",
    )
    parser.add_argument("--weight-decay", type=float, default=0.001, help="L2 regularization")
    parser.add_argument("--warmup-epochs", type=int, default=10, help="Longer warmup for stability")
    parser.add_argument("--seed", type=int, default=42)

    # Gradient clipping (essential for attention models)
    parser.add_argument(
        "--gradient-clip-val", type=float, default=1.0, help="Gradient clipping value"
    )

    # Mixed precision (AMP) - 2x speedup, less memory
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        choices=["32", "16-mixed", "bf16-mixed"],
        help="Mixed precision training (16-mixed recommended)",
    )

    # Subsampling for faster epochs during early training
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=None,
        help="Subsample training set (None = full dataset, good for final training)",
    )

    # ===============================
    # MODEL ARCHITECTURE (SCALED UP)
    # ===============================

    # Larger model with 24GB VRAM
    parser.add_argument(
        "--embedding-dim", type=int, default=256, help="Larger embeddings (128→256 with more VRAM)"
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num-heads", type=int, default=8, help="More attention heads (4→8)")
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="Higher dropout for regularization"
    )
    parser.add_argument(
        "--no-gating",
        action="store_true",
        help="Disable gated fusion (keep enabled for better performance)",
    )

    # Attention dropout (additional regularization)
    parser.add_argument(
        "--attention-dropout", type=float, default=0.1, help="Dropout in attention layers"
    )

    # ===============================
    # AUXILIARY TASKS (RECOMMENDED)
    # ===============================

    parser.add_argument(
        "--use-auxiliary-losses",
        action="store_true",
        default=True,
        help="Train with auxiliary tasks (forces cross-modal learning)",
    )
    parser.add_argument(
        "--aux-loss-weight",
        type=float,
        default=0.2,
        help="Weight for auxiliary losses (higher = stronger auxiliary signal)",
    )

    # ===============================
    # CHECKPOINTING & MONITORING
    # ===============================

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints_cross",
        help="Directory to save checkpoints",
    )

    # Save checkpoints more frequently
    parser.add_argument(
        "--save-top-k", type=int, default=5, help="Save top K checkpoints (not just 3)"
    )
    parser.add_argument(
        "--save-every-n-epochs",
        type=int,
        default=5,
        help="Save checkpoint every N epochs regardless of metric",
    )

    # Early stopping patience
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=30,
        help="Patience for early stopping (longer = more training)",
    )

    # ===============================
    # LOGGING
    # ===============================

    parser.add_argument("--log-every-n-steps", type=int, default=25, help="Log frequency")
    parser.add_argument(
        "--enable-tensorboard", action="store_true", default=True, help="Enable TensorBoard logging"
    )

    # Resume functionality
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--resume-from-last",
        action="store_true",
        help="Resume from the last.ckpt checkpoint",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"cross_attention_rtx5000_{timestamp}"

    # Setup logging
    logs_dir = Path("logs") / experiment_name
    logs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / "training.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    # Log GPU info
    import torch

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        print(f"✓ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.warning("No GPU available!")
        print("⚠ No GPU available!")

    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Batch size: {args.batch_size} (optimized for 24GB VRAM)")
    logger.info(f"Workers: {args.num_workers} (optimized for 100GB RAM)")
    logger.info(f"Precision: {args.precision} (mixed precision for speed)")
    logger.info(f"Embedding dim: {args.embedding_dim} (scaled up)")
    logger.info(f"Attention heads: {args.num_heads}")

    print(f"\nExperiment: {experiment_name}")
    print(f"Training: {args.train_start} to {args.train_end}")
    print(f"Batch size: {args.batch_size} | Workers: {args.num_workers}")
    print(f"Precision: {args.precision} | Embedding: {args.embedding_dim}")
    print(f"Logging to: {logs_dir / 'training.log'}")

    # Create data module with optimized settings
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
        pin_memory=args.pin_memory,  # Faster data transfer to GPU
    )

    # Create model with scaled-up architecture
    model_module = CrossAttentionDualEncoderModule(
        temporal_input_dim=13,
        tabular_continuous_dim=15,
        embedding_dim=args.embedding_dim,  # 256 instead of 128
        temperature=args.temperature,
        num_heads=args.num_heads,  # 8 instead of 4
        dropout=args.dropout,  # 0.2
        use_gating=not args.no_gating,
        use_auxiliary_losses=args.use_auxiliary_losses,
        aux_loss_weight=args.aux_loss_weight,
        lr=args.lr,  # 5e-4 instead of 1e-4
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,  # 10 instead of 5
        max_epochs=args.epochs,
    )

    # Setup checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Advanced callbacks
    callbacks = [
        # Save best checkpoints
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{experiment_name}" + "-best-{{epoch:02d}}-{{val/loss:.4f}}",
            monitor="val/loss",
            mode="min",
            save_top_k=args.save_top_k,
            save_last=True,
            verbose=True,
        ),
        # Save periodic checkpoints (for long training)
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{experiment_name}" + "-periodic-{{epoch:02d}}",
            every_n_epochs=args.save_every_n_epochs,
            save_top_k=-1,  # Keep all periodic checkpoints
        ),
        # Early stopping with patience
        EarlyStopping(
            monitor="val/loss",
            patience=args.early_stopping_patience,
            mode="min",
            verbose=True,
        ),
        # Learning rate monitoring
        LearningRateMonitor(logging_interval="step"),
        # Progress bar
        TQDMProgressBar(refresh_rate=10),
    ]

    # Loggers
    loggers = [CSVLogger(logs_dir, name="", version=0)]
    if args.enable_tensorboard:
        loggers.append(TensorBoardLogger(logs_dir, name="tensorboard"))

    # Trainer with all optimizations
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=loggers,
        accelerator="gpu",  # Force GPU
        devices=1,  # Single RTX A5000
        precision=args.precision,  # 16-mixed for speed
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=args.log_every_n_steps,
        enable_progress_bar=True,
        # Performance optimizations
        enable_model_summary=True,
        enable_checkpointing=True,
        benchmark=True,  # cudnn.benchmark for consistent input sizes
        deterministic=False,  # Faster but non-deterministic
    )

    # Handle resume
    ckpt_path = None
    if args.resume_from:
        ckpt_path = args.resume_from
        if not Path(ckpt_path).exists():
            logger.error(f"Checkpoint not found: {ckpt_path}")
            return
        logger.info(f"Resuming from: {ckpt_path}")
        print(f"Resuming from: {ckpt_path}")
    elif args.resume_from_last:
        last_ckpt = checkpoint_dir / "last.ckpt"
        if last_ckpt.exists():
            ckpt_path = str(last_ckpt)
            logger.info(f"Resuming from: {ckpt_path}")
            print(f"Resuming from: {ckpt_path}")
        else:
            logger.warning(f"No last.ckpt found, starting fresh")
            print("Starting fresh training...")

    print("\n" + "=" * 70)
    print("STARTING TRAINING WITH RTX A5000 OPTIMIZATIONS")
    print("=" * 70)
    print(f"• Batch size: {args.batch_size} (maximize GPU utilization)")
    print(f"• Precision: {args.precision} (2x speedup)")
    print(f"• Workers: {args.num_workers} (fast data loading)")
    print(f"• Embedding dim: {args.embedding_dim} (larger model)")
    print(f"• Attention heads: {args.num_heads}")
    print(f"• Gradient clipping: {args.gradient_clip_val}")
    print(f"• Auxiliary losses: {args.use_auxiliary_losses}")
    print("=" * 70 + "\n")

    trainer.fit(model_module, data_module, ckpt_path=ckpt_path)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best checkpoints: {checkpoint_dir}")
    print(f"Final model: {checkpoint_dir / 'last.ckpt'}")
    print(f"Logs: {logs_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
