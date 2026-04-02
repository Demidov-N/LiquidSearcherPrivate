"""Integration tests for training infrastructure."""


def test_lightning_available():
    """Test PyTorch Lightning is installed."""
    import pytorch_lightning as pl

    version = getattr(pl, "__version__", "0.0.0")
    assert version >= "2.0.0"


def test_imports_work():
    """Test training package can be imported."""
    from src import training

    assert training is not None
