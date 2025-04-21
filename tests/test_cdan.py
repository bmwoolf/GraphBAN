import torch
import pytest
from models.cdan_module import CDAN

def test_cdan():
    cdan = CDAN(feature_dim=128)
    batch_size = 2
    
    # Valid input
    features = torch.randn(batch_size, 128)
    output = cdan(features)
    assert output.shape == (batch_size, 2)
    
    # Test shape validation
    with pytest.raises(AssertionError, match="Expected 2D tensor"):
        cdan(torch.randn(batch_size, 1, 128))
        
    with pytest.raises(AssertionError, match="Expected feature dimension"):
        cdan(torch.randn(batch_size, 64)) 