import torch.nn as nn
from models.ban_student import BANLayer

class StudentModel(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.ban = BANLayer(dim)
        self.classifier = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def forward(self, h_c, h_p):
        joint = self.ban(h_c, h_p)
        return self.classifier(joint)
