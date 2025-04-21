import torch.nn as nn
from models.ban_student import BANLayer

class StudentModel(nn.Module):
    def __init__(self, dim=128):
        """Student model with bilinear attention.
        
        Args:
            dim: Feature dimension
        """
        super().__init__()
        self.ban = BANLayer(dim)
        # Remove the extra linear layer that's squashing to size 1
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim),  # Keep the dimension
            nn.ReLU()
        )

    def forward(self, h_c, h_p):
        """
        Args:
            h_c: Compound features (B, 1, dim)
            h_p: Protein features (B, 1, dim)
        Returns:
            torch.Tensor: Features of shape (B, dim)
        """
        joint = self.ban(h_c, h_p)  # (B, dim)
        return self.classifier(joint)  # (B, dim)
