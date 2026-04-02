"""Tests for dual encoder system."""

import torch
from src.models.dual_encoder import DualEncoder


def test_training_mode():
    """Test training mode with InfoNCE loss."""
    model = DualEncoder()
    
    price_data = torch.randn(32, 60, 13)
    fundamentals = torch.randn(32, 15)
    categorical = torch.randint(0, 5, (32, 2))
    
    loss, temp_emb, tab_emb = model(
        price_data, fundamentals, categorical, mode='train'
    )
    
    assert loss.shape == ()
    assert loss.item() > 0
    assert temp_emb.shape == (32, 128)
    assert tab_emb.shape == (32, 128)


def test_inference_mode():
    """Test inference mode with joint embedding."""
    model = DualEncoder()
    model.eval()
    
    price_data = torch.randn(16, 60, 13)
    fundamentals = torch.randn(16, 15)
    categorical = torch.randint(0, 5, (16, 2))
    
    with torch.no_grad():
        joint_emb = model(
            price_data, fundamentals, categorical, mode='inference'
        )
    
    assert joint_emb.shape == (16, 256)  # 128 + 128


def test_embeddings_normalized():
    """Test L2 normalization."""
    model = DualEncoder()
    
    price_data = torch.randn(8, 60, 13)
    fundamentals = torch.randn(8, 15)
    categorical = torch.randint(0, 5, (8, 2))
    
    _, temp_emb, tab_emb = model(
        price_data, fundamentals, categorical, mode='train'
    )
    
    temp_norms = torch.norm(temp_emb, dim=1)
    tab_norms = torch.norm(tab_emb, dim=1)
    
    assert torch.allclose(temp_norms, torch.ones_like(temp_norms), atol=1e-5)
    assert torch.allclose(tab_norms, torch.ones_like(tab_norms), atol=1e-5)


def test_similarity_search():
    """Test similarity computation."""
    model = DualEncoder()
    model.eval()
    
    with torch.no_grad():
        query_emb = model.get_joint_embedding(
            torch.randn(1, 60, 13),
            torch.randn(1, 15),
            torch.randint(0, 5, (1, 2))
        )
        
        candidate_emb = model.get_joint_embedding(
            torch.randn(100, 60, 13),
            torch.randn(100, 15),
            torch.randint(0, 5, (100, 2))
        )
    
    similarities = torch.nn.functional.cosine_similarity(
        query_emb, candidate_emb
    )
    
    assert similarities.shape == (100,)
    assert similarities.min() >= -1.0
    assert similarities.max() <= 1.0


def test_save_load():
    """Test model save and load."""
    import tempfile
    import os
    
    model = DualEncoder()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        torch.save(model.state_dict(), f.name)
        temp_path = f.name
    
    try:
        new_model = DualEncoder()
        new_model.load_state_dict(torch.load(temp_path))
        
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)
    finally:
        os.unlink(temp_path)
