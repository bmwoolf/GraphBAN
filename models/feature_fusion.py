import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, f_struct, f_llm):
        # Ensure 2D tensors
        if f_struct.dim() == 1:
            f_struct = f_struct.unsqueeze(0)
        if f_llm.dim() == 1:
            f_llm = f_llm.unsqueeze(0)
            
        # Match batch sizes if needed
        if f_struct.shape[0] != f_llm.shape[0]:
            f_struct = f_struct.expand(f_llm.shape[0], -1)
            
        # Combine features
        f_combined = torch.cat([f_struct, f_llm], dim=1)
        x = torch.relu(self.fc1(f_combined))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        fused = x * f_llm + f_struct
        return self.dropout(fused)
