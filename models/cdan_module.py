import torch
import torch.nn as nn

class CDAN(nn.Module):
    def __init__(self, feature_dim, num_domains=2):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_domains)
        )

    def forward(self, features):
        return self.discriminator(features)