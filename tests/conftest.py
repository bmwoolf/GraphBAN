import pytest
import torch

@pytest.fixture
def device():
    return torch.device('cpu')

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def feature_dim():
    return 128 