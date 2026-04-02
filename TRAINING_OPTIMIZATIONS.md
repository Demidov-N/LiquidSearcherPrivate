# RTX A5000 Training Optimizations

## Your Hardware
- **GPU**: NVIDIA RTX A5000 (24GB VRAM)
- **RAM**: 100GB
- **CUDA**: 12.9

## Optimizations Applied

### 1. Batch Size Scaling (32 → 128)
**Why**: 24GB VRAM can easily handle 128+ batch size
- Larger batches = more stable gradients
- Better GPU utilization (higher % of time computing, not transferring)
- Learning rate scales with √batch_size: 1e-4 → 5e-4

### 2. Mixed Precision (FP16)
**Why**: 2x speedup, 50% less memory
- `precision="16-mixed"` in PyTorch Lightning
- RTX A5000 has Tensor Cores that accelerate FP16
- Minimal accuracy loss, massive speed gain

### 3. More Data Workers (4 → 12)
**Why**: 100GB RAM allows aggressive prefetching
- CPU loads next batch while GPU trains current
- `pin_memory=True` for faster CPU→GPU transfer
- Prevents GPU starvation (waiting for data)

### 4. Larger Model Architecture
| Component | Default | Optimized | Why |
|-----------|---------|-----------|-----|
| Embedding dim | 128 | 256 | More capacity with 24GB |
| Attention heads | 4 | 8 | Better multi-view learning |
| Dropout | 0.1 | 0.2 | Regularization for larger model |

### 5. Training Robustness
- **Gradient clipping**: Prevents exploding gradients in attention
- **Longer warmup**: 5 → 10 epochs (stable early training)
- **Higher weight decay**: 0.01 → 0.001 (L2 regularization)
- **Auxiliary losses**: 0.1 → 0.2 (stronger cross-modal signal)

### 6. Checkpointing Strategy
- Save top-5 (not just 3) checkpoints
- Save every 5 epochs regardless of metric
- Early stopping patience: 15 → 30 (train longer)

## Quick Start

```bash
# Full training (recommended)
uv run python -m scripts.train_cross_attention_rtx5000 \
    --epochs 200 \
    --batch-size 128 \
    --use-auxiliary-losses

# Quick test (10 epochs, smaller batch)
uv run python -m scripts.train_cross_attention_rtx5000 \
    --epochs 10 \
    --batch-size 64

# Maximum scale (push the GPU)
uv run python -m scripts.train_cross_attention_rtx5000 \
    --epochs 200 \
    --batch-size 256 \
    --embedding-dim 512 \
    --num-heads 16
```

## Monitoring GPU Usage

```bash
# Watch GPU in real-time (run in separate terminal)
watch -n 1 nvidia-smi

# Or use nvitop (better visualization)
nvitop

# Check training logs
tail -f logs/cross_attention_rtx5000_*/training.log
```

## Expected Performance

| Metric | Default | RTX A5000 Optimized | Improvement |
|--------|---------|---------------------|-------------|
| Time/epoch | ~5 min | ~2 min | **2.5x faster** |
| GPU utilization | ~40% | ~95% | **+55%** |
| Batch size | 32 | 128 | **4x larger** |
| Model capacity | 128-dim | 256-dim | **2x larger** |
| Training time (100 epochs) | ~8 hours | ~3 hours | **2.7x faster** |

## Troubleshooting

**If you get OOM (Out of Memory):**
```bash
# Reduce batch size
--batch-size 64

# Or reduce model size
--embedding-dim 128 --num-heads 4
```

**If GPU utilization is low (<80%):**
```bash
# Increase workers
--num-workers 16

# Or increase batch size
--batch-size 256
```

**If training is unstable:**
```bash
# Lower learning rate
--lr 3e-4

# Increase gradient clipping
--gradient-clip-val 0.5

# Reduce auxiliary loss weight
--aux-loss-weight 0.1
```

## After Training

Evaluate the model:
```bash
uv run python -m scripts.evaluation.run_retrieval_metrics \
    --features data/processed/all_features.parquet \
    --checkpoint checkpoints_cross/last.ckpt \
    --output-dir results/retrieval_cross_rtx5000
```
