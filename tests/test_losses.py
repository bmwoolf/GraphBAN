import torch
import pytest
from models.losses import DistillationLoss

def test_distillation_loss():
    loss_fn = DistillationLoss(alpha=0.5, feature_dim=128)
    batch_size = 2
    
    # Valid inputs
    student = torch.randn(batch_size, 128)
    teacher = torch.randn(batch_size, 128)
    loss = loss_fn(student, teacher)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar loss
    
    # Test dimension validation
    with pytest.raises(AssertionError, match="Too many dimensions"):
        loss_fn(torch.randn(2, 2, 2, 128), teacher)
        
    # Test feature dimension validation
    with pytest.raises(AssertionError, match="Expected student feature dim"):
        loss_fn(torch.randn(batch_size, 64), teacher) 