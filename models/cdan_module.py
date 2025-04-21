import torch
import torch.nn as nn

class CDAN(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, features):
        """Domain adversarial network.
        
        Args:
            features: Input features (B, feature_dim)
            
        Returns:
            torch.Tensor: Domain predictions (B, 2)
        """
        # Shape validation
        assert features.dim() == 2, f"Expected 2D tensor, got shape {features.shape}"
        assert features.shape[1] == self.feature_dim, \
            f"Expected feature dimension {self.feature_dim}, got {features.shape[1]}"

        return self.discriminator(features)