"""Memory detection and batch size optimization utilities."""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def get_available_memory_mb() -> Optional[int]:
    """Get available system memory in MB.
    
    Returns:
        Available memory in MB, or None if cannot detect
    """
    try:
        # Try /proc/meminfo on Linux
        if os.path.exists('/proc/meminfo'):
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        # Format: "MemAvailable:   12345678 kB"
                        parts = line.split()
                        if len(parts) >= 2:
                            kb = int(parts[1])
                            return kb // 1024  # Convert to MB
        
        # Fallback: try psutil if available
        try:
            import psutil
            mem = psutil.virtual_memory()
            return mem.available // (1024 * 1024)  # Convert to MB
        except ImportError:
            pass
        
        # Another fallback: check cgroup limits (Docker/containers)
        if os.path.exists('/sys/fs/cgroup/memory/memory.limit_in_bytes'):
            with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                limit_bytes = int(f.read().strip())
                # If limit is very high (e.g., max int), treat as unlimited
                if limit_bytes < 2**60:  # Sanity check
                    return limit_bytes // (1024 * 1024)
        
        return None
        
    except Exception as e:
        logger.warning(f"Could not detect available memory: {e}")
        return None


def get_recommended_batch_size(
    available_mb: Optional[int] = None,
    safety_factor: float = 0.6
) -> int:
    """Calculate recommended batch size based on available memory.
    
    Conservative estimates (memory per symbol):
    - Price data: ~50KB per symbol per day (compressed)
    - Feature computation: ~3x expansion
    - Safety buffer for Polars operations
    
    Args:
        available_mb: Available memory in MB (auto-detect if None)
        safety_factor: Fraction of memory to use (default 60% for safety)
        
    Returns:
        Recommended number of symbols per batch
    """
    if available_mb is None:
        available_mb = get_available_memory_mb()
    
    if available_mb is None:
        logger.warning("Could not detect memory, using conservative default")
        return 200  # Very conservative default
    
    # Calculate usable memory
    usable_mb = int(available_mb * safety_factor)
    
    # Memory requirements (empirical estimates):
    # - Base overhead: 500 MB
    # - Per symbol: ~2 MB for 2 years of daily data + features
    base_overhead_mb = 500
    mb_per_symbol = 2.0
    
    if usable_mb <= base_overhead_mb:
        logger.warning(f"Very limited memory ({available_mb} MB), using minimum batch size")
        return 50
    
    available_for_data = usable_mb - base_overhead_mb
    recommended = int(available_for_data / mb_per_symbol)
    
    # Round to nice numbers and clamp to reasonable range
    if recommended < 50:
        batch_size = 50
    elif recommended < 100:
        batch_size = 100
    elif recommended < 200:
        batch_size = 200
    elif recommended < 400:
        batch_size = 400
    elif recommended < 750:
        batch_size = 750
    elif recommended < 1000:
        batch_size = 1000
    elif recommended < 1500:
        batch_size = 1500
    elif recommended < 2000:
        batch_size = 2000
    else:
        batch_size = 2400  # Process all symbols at once
    
    logger.info(f"Memory detection: {available_mb} MB available")
    logger.info(f"Recommended batch size: {batch_size} symbols")
    logger.info(f"Estimated memory usage: ~{int(batch_size * mb_per_symbol + base_overhead_mb)} MB")
    
    return batch_size


def print_memory_status():
    """Print current memory status for user information."""
    mem_mb = get_available_memory_mb()
    
    if mem_mb is None:
        print("Memory status: Could not detect (will use conservative defaults)")
    else:
        gb = mem_mb / 1024
        print(f"Memory status: {mem_mb:,} MB ({gb:.1f} GB) available")
        
        batch_size = get_recommended_batch_size(mem_mb)
        print(f"Auto-selected batch size: {batch_size} symbols")
        
        if mem_mb < 4000:
            print("⚠️  Warning: Low memory detected. Processing may be slow.")
            print("   Consider: closing other applications, or using a machine with more RAM")
        elif mem_mb > 30000:
            print("✓ Good memory available. Processing will be efficient.")
        
    print()


if __name__ == '__main__':
    # Test the module
    print("Testing memory detection...")
    print_memory_status()
