import torch
import pytest
from models.ban_student import BANLayer

def test_ban_layer_shapes():
    ban = BANLayer(dim=128)
    batch_size = 2
    seq_len = 3
    
    # Valid inputs
    h_c = torch.randn(batch_size, seq_len, 128)
    h_p = torch.randn(batch_size, seq_len, 128)
    output = ban(h_c, h_p)
    assert output.shape == (batch_size, 128)

    # Test shape validation
    with pytest.raises(AssertionError, match="Expected 3D tensor"):
        ban(torch.randn(batch_size, 128), h_p)
    
    with pytest.raises(AssertionError, match="Expected feature dim"):
        ban(torch.randn(batch_size, seq_len, 64), h_p)
        
    with pytest.raises(AssertionError, match="Batch sizes must match"):
        ban(torch.randn(3, seq_len, 128), h_p) 