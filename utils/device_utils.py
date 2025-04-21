import torch

def get_device():
    """Get the best available device for PyTorch computations."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def move_to_device(model, device=None):
    """Move model to specified device or best available device."""
    if device is None:
        device = get_device()
    return model.to(device) 