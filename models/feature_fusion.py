import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.fc1 = nn.Linear(384, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, f_struct, f_llm):
        f_llm = self.fc1(f_llm)
        f_llm = torch.relu(self.fc2(f_llm))
        f_llm = torch.relu(self.fc3(f_llm))
        fused = (f_struct + f_llm) * f_llm.T + f_llm  # matmul + addition (Eq. 2)
        return self.dropout(fused)
